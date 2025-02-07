import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import ParallelStrategy
from pytorch_lightning.callbacks import Callback

from torch.optim.lr_scheduler import MultiStepLR
import torch.distributed as dist

import numpy as np

import logging

import cv2

from pathlib import Path


from EventEgoPoseEstimation.model import EgoHPE

from EventEgoPoseEstimation.dataset.dataset_utils import collate_variable_size, create_image

from EventEgoPoseEstimation.dataset import EgoEvent, AugmentedEgoEvent, TemoralWrapper, CombinedEgoEvent

from EventEgoPoseEstimation.utils.utils import AverageMeter, save_checkpoint, create_logger

from EventEgoPoseEstimation.utils.skeleton import Skeleton

from configs.settings import config as cfg

from EventEgoPoseEstimation.core.function import compute_fn, _print_name_value

from EventEgoPoseEstimation.core.evaluate import accuracy

from EventEgoPoseEstimation.core.loss import SegmentationLoss, HeatMapJointsMSELoss, J3dMSELoss

from EventEgoPoseEstimation.utils.vis import plot_heatmaps, save_debug_images, save_debug_3d_joints, save_debug_segmenation, save_debug_eros, generate_skeleton_image, dump_sketelon_image

from EventEgoPoseEstimation.core.inference import get_j2d_from_hms

import torchvision

logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.EXP_NAME, 'train')

torch.set_float32_matmul_precision('medium')

class EventEgoPoseEstimation(LightningModule):

    def __init__(
        self,
        model_cfg: dict,
        dataset_type: str,
        training_type: str,
        temporal_steps: int,
        batch_size: int,
        workers: int,
        lr: float,
        lr_decay_epochs: tuple,
        dataset_kwargs: dict = {}
    ):
        
        super().__init__()

        assert dataset_type in ["real", "synthetic"]

        self.dataset_type = dataset_type
        self.dataset_kwargs = dataset_kwargs

        self.training_type = training_type

        self.model = EgoHPE(cfg, **model_cfg)

        self.input_channel = model_cfg['input_channel']
        self.image_size = model_cfg['image_size']
        self.temporal_bins = int(self.input_channel / 2)
        self.model_batch_size = int(model_cfg['batch_size'])

        self.batch_size = batch_size
        self.workers = workers

        self.lr = lr
        self.lr_decay_epochs = lr_decay_epochs

        self.temporal_steps = temporal_steps

        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.criterions = {
            'j3d': J3dMSELoss(
                use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
            ).cuda(),
            'delta_j3d': J3dMSELoss(
                use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
            ).cuda(),
            'seg': SegmentationLoss().cuda()
        }

        self.all_gt_j3ds = []
        self.all_preds_j3d = []
        self.all_vis_j3d = []
        self.all_frame_indices = []

        self.batch_time = AverageMeter()
        self.acc_j3d = AverageMeter()

        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()

        self.losses = AverageMeter()

        self.j3d_delta_losses = AverageMeter()
        self.j3d_losses = AverageMeter()    
        self.seg_losses = AverageMeter()

        self.acc = AverageMeter()
        self.acc_j3d_val = AverageMeter()

        self.global_steps = 0

        self.all_gt_j3ds = []
        self.all_preds_j3d = []
        self.all_vis_j3d = []
        self.all_frame_indices = []
        self.train_dataloader_len = 0
        self.val_dataloader_len = 0\

        # self.initial_pose = torch.from_numpy(np.load('./initial_pose.npy')).unsqueeze(0).expand(self.model_batch_size, -1, -1)

        self.automatic_optimization = False

       
        
        # self.scaler = self.trainer.precision_plugin.scaler  # Lightning provides this if AMP is enabled

        # self.example_input_array = torch.Tensor(1, 1, 32768, 4)


    def forward(self, x):
        return self.model(x)
    
    def setup(self, stage: str):
        if isinstance(self.trainer.strategy, ParallelStrategy):
            num_processes = max(1, self.trainer.strategy.num_processes)
            self.batch_size = int(self.batch_size / num_processes)
            self.workers = int(self.workers / num_processes)

        if cfg.DATASET.TYPE == 'Combined':
            TrainDataset = CombinedEgoEvent
        else:
            TrainDataset = EgoEvent
    
        if stage == "fit":
            if cfg.DATASET.BG_AUG:
                pretrain_dataset = AugmentedEgoEvent(cfg, EgoEvent(cfg, temporal_bins=self.temporal_bins, split='train'))
            else:
                pretrain_dataset = TrainDataset(cfg, temporal_bins=self.temporal_bins, split='train')

            if cfg.DATASET.BG_AUG:
                finetune_dataset = AugmentedEgoEvent(cfg, EgoEvent(cfg, temporal_bins=self.temporal_bins, split='train', finetune=True))
            else:
                finetune_dataset = EgoEvent(cfg, temporal_bins=self.temporal_bins, split='train', finetune=True)

            cfg.DATASET.TYPE = 'Real'    
            self.eval_dataset = EgoEvent(cfg, temporal_bins=self.temporal_bins, split='test')


            if self.training_type == 'pretrain':
                pretraining = True
                logger.info("Training type: Pretrain")
                self.train_dataset = TemoralWrapper(pretrain_dataset, cfg.DATASET.TEMPORAL_STEPS, augment=True)
            elif self.training_type == 'finetune':
                pretraining = False
                logger.info("Training type: Finetune")
                self.train_dataset = TemoralWrapper(finetune_dataset, cfg.DATASET.TEMPORAL_STEPS, augment=False)
            else:
                assert False, f"Invalid training type: {self.training_type}"

        # TODO: Fix this
        if stage == "test" or stage == "predict":
            if self.training_type == 'pretrain':
                cfg.DATASET.SYN_ROOT = cfg.DATASET.SYN_TEST_ROOT 
                cfg.DATASET.TYPE = 'Synthetic'
                cfg.DATASET.BG_AUG = False
                cfg.DATASET.TRAIN_TEST_SPLIT = 0
            elif self.training_type == 'finetune':
                cfg.DATASET.TYPE = 'Real'
                cfg.DATASET.BG_AUG = False
            else:
                assert False, f"Invalid training type: {self.training_type}"
            # cfg.DATASET.TYPE = 'Real'
            # cfg.DATASET.BG_AUG = False
            self.test_dataset = EgoEvent(cfg, temporal_bins=self.temporal_bins, split='test')
            # self.eval_dataset = TemoralWrapper(self.test_dataset_e, cfg.DATASET.TEMPORAL_STEPS, augment=False)

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_variable_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True
        )
        self.train_dataloader_len = len(dataloader)
        return dataloader

    def val_dataloader(self):
        dataloader =  torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_variable_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True
        )
        self.val_dataloader_len = len(dataloader)
        return dataloader

    def test_dataloader(self):
        dataloader =  torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_variable_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True
        )
        self.val_dataloader_len = len(dataloader)
        return dataloader

    def training_step(self, batch, batch_idx):
        end = time.time()

        failure_flag = torch.tensor([0.0], device=self.device)
        loss = None
        # scaler = self.trainer.precision_plugin.scaler  # Lightning provides this if AMP is enabled
        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True) # set_to_none=True is more efficient and saves memory

        # try:
        with torch.amp.autocast('cuda', enabled=True):
            inp, outputs, gt_hms, gt_j3d, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index, status = compute_fn(self.model, batch, self.temporal_steps, device=self.device)

            valid_seg = valid_seg.expand(-1, self.temporal_bins)
            valid_seg = valid_seg.view(self.model_batch_size, self.temporal_bins, 1, 1, 1)

            meta = {'j3d': gt_j3d, 'j2d': gt_j2d, 'vis_j2d': vis_j2d, 'vis_j3d': vis_j3d}
            
            # pred_hms = outputs['hms']
            pred_seg = outputs['seg']
            pred_eros = outputs['eros']
            
            gt_j3d = gt_j3d  * 1000 # scale to mm
            pred_j3d = outputs['j3d'] * 1000 # scale to mm
            pred_j3d_deltas = outputs['delta_j3d'] * 1000 # scale to mm
            # initial_pose = self.initial_pose
            # initial_j3d = initial_pose.cuda() * 1000

            # all_gt_j3d = torch.cat([initial_j3d.unsqueeze(1), gt_j3d], dim=1)
            all_gt_j3d = gt_j3d

            gt_j3d_deltas = all_gt_j3d[:, 1:, :, :] - all_gt_j3d[:, :-1, :, :]

        # dump_sketelon_image(test_gt_j3d, f"./visualizations/{10}_gt_j3d.png")

        # with torch.amp.autocast('cuda', enabled=True):
            loss_j3d_delta = self.criterions['delta_j3d'](pred_j3d_deltas, gt_j3d_deltas, vis_j3d * 1e-2)  # scale to 1e-2
            loss_seg = self.criterions['seg'](pred_seg, gt_seg, valid_seg)
            loss_j3d = self.criterions['j3d'](pred_j3d, gt_j3d, vis_j3d * 1e-2)  # scale to 1e-2
                
            loss = loss_j3d_delta + loss_j3d + loss_seg


        # pred_j2d = get_j2d_from_hms(cfg, pred_hms)

        pred_seg_detached = torch.sigmoid(pred_seg.detach().clone())


        avg_acc, cnt = accuracy(gt_j3d, pred_j3d, valid_j3d)
        
        self.j3d_delta_losses.update(loss_j3d_delta.detach(), inp.size(0))
        self.seg_losses.update(loss_seg.detach(), inp.size(0))
        self.j3d_losses.update(loss_j3d.detach(), inp.size(0))
        self.losses.update(loss.detach(), inp.size(0))
        self.acc.update(avg_acc, cnt)

        self.batch_time.update(time.time() - end)
        end = time.time()

        representation = outputs['representation']
        representation_image = create_image(representation)
        pred_eros_image = create_image(pred_eros.detach())

            # TODO: save checkpoint - fix this later
            # if batch_idx % 5000 == 0:
            #     filename = f'{i}_it_checkpoint.pth'
            #     save_checkpoint(self.current_epoch + 1, {
            #     'epoch': self.current_epoch + 1,
            #     'model': cfg.MODEL.NAME,
            #     'state_dict': model.state_dict(),
            #     'best_state_dict': model.module.state_dict(),
            #     'perf': 1e6,
            #     'optimizer': optimizer.state_dict(),
            # }, False, output_dir, tb_log_dir, filename=filename)
            # try:
        if batch_idx % cfg.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                'Speed {speed:.1f} samples/s\t' \
                'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                'Delta_J3D_Loss {hms_loss.val:.5f} ({hms_loss.avg:.5f})\t' \
                'J3D_Loss {j3d_loss.val:.5f} ({j3d_loss.avg:.5f})\t' \
                'SEG_Loss {seg_loss.val:.5f} ({seg_loss.avg:.5f})\t' \
                'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                'MPJPE {acc.val:.3f} ({acc.avg:.3f})'.format(
                    self.current_epoch, batch_idx, self.train_dataloader_len, batch_time=self.batch_time,
                    speed=inp.size(0)/self.batch_time.val,
                    data_time=self.data_time, 
                    loss=self.losses, 
                    j3d_loss=self.j3d_losses, 
                    hms_loss=self.j3d_delta_losses,
                    seg_loss=self.seg_losses,
                    acc=self.acc
                    )
            logger.info(msg)

            memory_stats = torch.cuda.memory_stats("cuda:0")
        # logger.info("Batch shape: {}".format(inp.shape))
            # logger.info(f"Current allocated memory: {memory_stats['allocated_bytes.all.current'] / (1024 ** 2):.2f} MB")
            # logger.info(f"Peak allocated memory: {memory_stats['allocated_bytes.all.peak'] / (1024 ** 2):.2f} MB")
        # logger.info(f"Current reserved memory: {memory_stats['reserved_bytes.all.current'] / (1024 ** 2):.2f} MB")
            logger.info(f"Peak reserved memory: {memory_stats['reserved_bytes.all.peak'] / (1024 ** 2):.2f} MB")
        if batch_idx % cfg.PRINT_FREQ == 0:
            self.log('train_loss', self.losses.avg)
            self.log('train_acc', self.acc.avg)

            self.global_steps = self.global_steps + 1

            if int(self.batch_size) < 4:
                n_images = int(self.batch_size)
            else:
                n_images = 4

            if batch_idx % (cfg.PRINT_FREQ * 4) == 0:
                try:
                    inp = inp.detach().cpu().numpy()
                    gt_j3d = gt_j3d
                    pred_j3d = pred_j3d
                    gt_seg = gt_seg
                    pred_seg_detached = pred_seg_detached
                    pred_eros_image = pred_eros_image
                    representation_image = representation_image

                    # save_debug_images(self, cfg, representation_image, meta, gt_hms, pred_hms, 'train', self.global_steps, n_images=n_images)
                    save_debug_3d_joints(self, cfg, inp, meta, gt_j3d, pred_j3d, 'train', global_step=self.global_steps)
                    save_debug_segmenation(self, cfg, inp, meta, gt_seg, pred_seg_detached, 'train', global_step=self.global_steps)
                    save_debug_eros(self, cfg, representation_image, meta, pred_eros_image, 'train', global_step=self.global_steps)
                except Exception as e:
                    logger.error("Error in saving debug data : {}".format(e))

        # except Exception as e:
        #     logger.info("Error in batch : {}".format(e))
        #     failure_flag = torch.tensor([1.0], device=self.device)
        
        # Synchronize flags across all processes
        dist.all_reduce(failure_flag, op=dist.ReduceOp.SUM)

        # Manual optimization logic
        if failure_flag.item() > 0:
            loss = torch.tensor(0.010, requires_grad=True, device=self.device)
            self.manual_backward(loss)
            # Optionally: log or handle this case appropriately
        else:
            self.manual_backward(loss)
            optimizer.step()

            # Optionally, step the learning rate scheduler:
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()
        
        return loss

    def eval_step(self, batch, batch_idx, prefix, vis=False):
        inp_W, inp_H = self.image_size
        # inp_W, inp_H = cfg.MODEL.IMAGE_SIZE  
        hm_W, hm_H = cfg.MODEL.HEATMAP_SIZE

        buffer = torch.zeros(self.model_batch_size, self.input_channel, inp_H, inp_W, device=self.device)

        key = torch.ones(self.model_batch_size, 1, hm_H, hm_W, device=self.device)

        end = time.time()
        
        if vis == False:
            with torch.amp.autocast('cuda', enabled=True):

                try:
                    inp, outputs, gt_hms, gt_j3d, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index, status = compute_fn(self.model, batch, self.temporal_steps, buffer, key, batch_first=True, device=self.device)
                except RuntimeError as e:
                    logger.info("Error in batch : {}".format(e))
                    return

                
                meta = {'j3d': gt_j3d, 'j2d': gt_j2d, 'vis_j2d': vis_j2d, 'vis_j3d': vis_j3d}

                pred_j3d = outputs['j3d'] * 1000 # scale to mm        
                gt_j3d = gt_j3d * 1000 # scale to mm

                # pred_j3d = pred_j3d[:, -1, :, :]
                # gt_j3d = gt_j3d[:, -1, :, :]

                
                avg_acc, cnt = accuracy(gt_j3d, pred_j3d, valid_j3d)
                self.acc_j3d_val.update(avg_acc, cnt)
                
                # measure elapsed time
                self.batch_time.update(time.time() - end)
                end = time.time()

                # representation = outputs['representation']
                # representation_image = create_image(representation)

                pred_j3d = pred_j3d.detach()

                self.all_preds_j3d.append(pred_j3d.detach())
                self.all_gt_j3ds.append(gt_j3d.detach())
                self.all_vis_j3d.append(valid_j3d.detach())
                self.all_frame_indices.append(frame_index.detach())

            if batch_idx % cfg.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    'MPJPE {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                        batch_idx, self.val_dataloader_len, batch_time=self.batch_time,
                        acc=self.acc_j3d_val)
                logger.info(msg)

        elif prefix == "test" and vis == True:
            global_steps=batch_idx
            tb_log_dir = self.logger.log_dir
            test_and_generate_vis(cfg, self.model, self.test_dataset, tb_log_dir, global_steps)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test", vis=False)

    def configure_optimizers(self):
        optimizer = None
        if cfg.TRAIN.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=cfg.TRAIN.WD,
                nesterov=cfg.TRAIN.NESTEROV
            )
        elif cfg.TRAIN.OPTIMIZER == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr
            )

        # TODO: check the LR decay epochs (currently not used)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, self.lr_decay_epochs, cfg.TRAIN.LR_FACTOR
        )

        return [optimizer], [lr_scheduler]

def test_and_generate_vis(cfg, model, test_dataset, tb_log_dir, global_steps):
    fps = 30
    seq_time_in_sec = 30

    seq_len = seq_time_in_sec * fps
    data_len = len(test_dataset)

    start = np.random.randint(0, data_len - cfg.DATASET.TEMPORAL_STEPS)
    stop = min(start + seq_len, data_len)

    tb_log_dir = Path(tb_log_dir)
    video_path = str(tb_log_dir / f'{global_steps}.mp4')

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (400 * 2, 300 * 1))

    model.eval()


    for i in range(start, stop):
        gt_j3d = []
        inps = []
        gt_hms = []
        
        data, meta = test_dataset[i]

        inp = data['x']

        gt_j3d_ = data['j3d']

        inps.append(inp[None, None, ...])
        gt_j3d.append(gt_j3d_[None, ...])
        gt_hms.append(data['hms'][None, ...])

        inps = torch.cat(inps, dim=0).cuda()
        
        with torch.no_grad():
            outputs = model(inps)

        pred_j3ds = outputs['j3d'].detach()
        # preds_hms = outputs['hms'].detach()
        # pred_j2ds = get_j2d_from_hms(cfg, preds_hms)
        
        gt_j3ds = torch.cat(gt_j3d, dim=0).detach()
        # gt_hms = torch.cat(gt_hms, dim=0).detach()
        # gt_hm_j2ds = get_j2d_from_hms(cfg, gt_hms)

        representation = outputs['representation']
        representation_image = create_image(representation)

        T, B, N, C = inps.shape

        for i in range(T):
            gt_j3d = gt_j3ds[i]
            gt_hm = gt_hms[i]
            # gt_hm_j2d = gt_hm_j2ds[i]
            
            pred_j3d = pred_j3ds[i]
            # pred_j2d = pred_j2ds[i]
            # pred_hm = preds_hms[i]

            inp = representation_image
            inp = inp[i]
            grid = torchvision.utils.make_grid(inp)
            inp = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).detach()
           

            # pred_hm_image = plot_heatmaps(inp, pred_hm)    
            # gt_hm_image = plot_heatmaps(inp, gt_hm)

            inp = inp.astype(np.uint8)

            # inp_w_gt_hm_j2d = Skeleton.draw_2d_skeleton(inp, gt_hm_j2d, lines=True)
            # inp = Skeleton.draw_2d_skeleton(inp, pred_j2d, lines=True)
                    
            color = generate_skeleton_image(gt_j3d, pred_j3d)
            color = color[..., ::-1]
            
            color = cv2.resize(color, (400, 300))
            inp = cv2.resize(inp, (400, 300))
            pred_hm_image = cv2.resize(pred_hm_image, (400, 300))
            gt_hm_image = cv2.resize(gt_hm_image, (400, 300))
            inp_w_gt_hm_j2d = cv2.resize(inp_w_gt_hm_j2d, (400, 300))           
            
            hstack1 = np.concatenate([inp, color], axis=1)
            hstack2 = np.concatenate([gt_hm_image, pred_hm_image], axis=1)
            hstack3 = np.concatenate([inp_w_gt_hm_j2d, np.zeros_like(inp_w_gt_hm_j2d)], axis=1)
            
            vstack = np.concatenate([hstack1, hstack2, hstack3], axis=0)
            video.write(vstack)

    video.release()



class EvaluateCallback(Callback):
    def __init__(self):
        pass
    
    def on_test_end(self, trainer, pl_module):
        """
        This hook is called after all test batches are completed.
        Access model's variables here.
        """
        # Ensure you're working with the correct model type
        model = pl_module  # 'pl_module' is the model instance

        all_preds_j3d = np.concatenate(model.all_preds_j3d, axis=0)
        all_gt_j3ds = np.concatenate(model.all_gt_j3ds, axis=0)
        all_vis_j3d = np.concatenate(model.all_vis_j3d, axis=0)
        all_frame_indices = np.concatenate(model.all_frame_indices, axis=0)

        np.save("all_preds_j3d.npy", all_preds_j3d)
        np.save("all_gt_j3ds.npy", all_gt_j3ds)
        np.save("all_vis_j3d.npy", all_vis_j3d)
        np.save("all_frame_indices.npy", all_frame_indices)

        # all_frame_indices = np.array(model.all_frame_indices)

        # name_values, perf_indicator = val_dataset.evaluate_dataset(config, frame_indices=all_frame_indices, all_gt_j3ds=all_gt_j3ds, all_preds_j3d=all_preds_j3d, all_vis_j3d=all_vis_j3d)
        if model.dataset_type == "real":
            name_values, _ = model.eval_dataset.evaluate_dataset(cfg, frame_indices=all_frame_indices, all_gt_j3ds=all_gt_j3ds, all_preds_j3d=all_preds_j3d, all_vis_j3d=all_vis_j3d)

            model_name = cfg.MODEL.NAME
            print("Error Per Actions : ")
            if isinstance(name_values, list):
                for name_value in name_values:
                    _print_name_value(name_value, model_name)
            else:
                _print_name_value(name_values, model_name)

        
        name_values, _ = model.eval_dataset.evaluate_joints(cfg, all_gt_j3ds=all_gt_j3ds, all_preds_j3d=all_preds_j3d, all_vis_j3d=all_vis_j3d)
        model_name = cfg.MODEL.NAME
        print("Error Per Joints : ")
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)
