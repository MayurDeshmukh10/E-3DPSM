import os
import time
from typing import Optional

from EventEgoPoseEstimation import model
import torch
from torch.utils.data import Dataset
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback

import numpy as np

import cv2

from pathlib import Path

import matplotlib.pyplot as plt


from EventEgoPoseEstimation.model import EgoHPE

from EventEgoPoseEstimation.dataset.dataset_utils import camera_to_j2d_batch

from EventEgoPoseEstimation.dataset import EgoEvent, EgoEventv2, AugmentedEgoEvent, TemoralWrapper, CombinedEgoEvent, EgoEventSequence

from EventEgoPoseEstimation.utils.utils import AverageMeter, create_logger

from configs.settings import config as cfg

from EventEgoPoseEstimation.core.function import _print_name_value, compute_fn_v3, compute_fn_v4

from EventEgoPoseEstimation.core.evaluate import accuracy, compute_motion_jitter

from EventEgoPoseEstimation.core.loss import SegmentationLoss, BoneLengthLoss, JointMSELoss, HeatMapJointsMSELoss, BoneOrientationLoss, BoneLoss

from EventEgoPoseEstimation.utils.vis import generate_skeleton, save_pose_images, save_debug_images, save_debug_3d_joints, save_debug_segmenation, save_debug_eros, generate_skeleton_image, dump_sketelon_image, drift_plot

from EventEgoPoseEstimation.core.inference import get_j2d_from_hms
from EventEgoPoseEstimation.dataset.dataset_utils import event_augmentation, save_augmented_data


import ocam

logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.EXP_NAME, 'train')

torch.set_float32_matmul_precision('medium')

import os


SAVE_VISUALIZATION = False
VISUALIZATION_PATH = '/scratch/inf0/user/mdeshmuk/visualization/EE3D-R/ours'


class EventEgoPoseEstimation(LightningModule):

    def __init__(
        self,
        model_cfg: dict,
        dataset_type: str,
        training_type: str,
        temporal_steps: int,
        sample_step: int,
        use_bg_augmentation: bool,
        batch_size: int,
        workers: int,
        tbptt_steps: int,
        fixed_sequence_length: int,
        lr: float,
        lr_decay_epochs: tuple,
        loss_weights: dict,
        dataset: dict
    ):
        
        super().__init__()

        assert dataset_type in ["real", "synthetic"]

        self.dataset_type = dataset_type

        self.syn_preprocessed_input_path = dataset['syn_preprocessed_input_path']
        self.real_preprocessed_input_path = dataset['real_preprocessed_input_path']
        self.syn_dataset_root_path = dataset['syn_dataset_root_path']
        self.syn_test_dataset_root_path = dataset['syn_test_dataset_root_path']
        self.real_dataset_root_path = dataset['real_dataset_root_path']
        self.bg_dataset_root_path = dataset['bg_dataset_root_path']
        self.bg_preprocessed_input_path = dataset['bg_preprocessed_input_path']
        self.wild_preprocessed_input_path = dataset['wild_preprocessed_input_path']
        self.wild_dataset_root_path = dataset['wild_dataset_root_path']


        self.training_type = training_type

        # self.model = torch.compile(EgoHPE(cfg, **model_cfg), mode='reduce-overhead')
        self.model = EgoHPE(cfg, **model_cfg)

        self.input_channel = model_cfg['input_channel']
        self.image_size = model_cfg['image_size']
        self.temporal_bins = int(self.input_channel / 2)
        self.model_batch_size = int(model_cfg['batch_size'])
        self.use_bg_augmentation = use_bg_augmentation
        self.batch_size = batch_size
        self.workers = workers

        self.sample_step = sample_step
        self.lr = lr
        self.lr_decay_epochs = lr_decay_epochs

        self.temporal_steps = temporal_steps

        self.fixed_sequence_length = fixed_sequence_length
        self.truncated_bptt_steps = tbptt_steps

        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # Losses
        self.criterions = {
            'j3d': JointMSELoss(use_target_weight=True).cuda(),
            'delta_j3d': JointMSELoss(use_target_weight=True).cuda(),
            'j2d': JointMSELoss(use_target_weight=True).cuda(),
            'bone_length': BoneLengthLoss(use_target_weight=True).cuda(),
            'bone_orientations': BoneOrientationLoss(use_target_weight=False).cuda(),
            'bone_loss': BoneLoss().cuda()
        }

        # performance metrics
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()

        # Training metrics
        self.j3d_delta_losses = AverageMeter()
        self.j3d_losses = AverageMeter()    
        self.seg_losses = AverageMeter()
        self.j2d_losses = AverageMeter()
        self.heatmap_losses = AverageMeter()
        self.bone_length_losses = AverageMeter()
        self.bone_angle_losses = AverageMeter()
        self.losses = AverageMeter()
        self.acc = AverageMeter()

        # Validation metrics
        self.acc_j3d_val = AverageMeter()
        self.jitter_j3d_val = AverageMeter()
        self.j3d_loss_val = AverageMeter()
        self.acc_occl_val = AverageMeter()
        self.all_gt_j3ds = []
        self.all_preds_j3d = []
        self.all_vis_j3d = []
        self.all_frame_indices = []
        self.all_occluded_joints = []
        self.total_count = 0
        self.per_joint_count = torch.tensor([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0.,  0.,  0., 0., 0.,  0.], device='cuda:0')

        self.global_steps = 0
        self.global_val_steps = 0

        self.train_dataloader_len = 0
        self.val_dataloader_len = 0

        # loss weights
        self.wgt_bone_length = loss_weights['bone_length']
        self.wgt_bone_angle = loss_weights['bone_angle']
        self.wgt_j3d = loss_weights['j3d']
        self.wgt_j2d = loss_weights['j2d']
        self.wgt_seg = loss_weights['seg']
        self.wgt_j3d_delta = loss_weights['delta_j3d']

        self.s5_states = None

        self.ocam_model = ocam.to_ocam_model('/CT/EventEgo3Dv2/work/egoposeformer/pose_estimation/models/utils/intrinsics.json')

        self.automatic_optimization = True

        self.count = 0

        self.s5_state = {
            0: None,
            3: None
        }

        print("Training Sequence Length :", self.temporal_steps)

    def forward(self, x):
        return self.model(x)
    
    def setup(self, stage: str):
    
        if stage == "fit":

            if self.training_type == 'train':
                logger.info("Training type: Finetune")

                finetune_dataset = AugmentedEgoEvent(
                                    cfg,
                                    EgoEvent(cfg, self.real_preprocessed_input_path, self.real_dataset_root_path, temporal_bins=self.temporal_bins, split='train', finetune=True),
                                    bg_data_root=self.bg_dataset_root_path, 
                                    bg_preprocessed_root=self.bg_preprocessed_input_path,
                                    split='finetune',
                                    temporal_bins=self.temporal_bins)
                
                # cfg.DATASET.TYPE = 'Real'
                eval_dataset = AugmentedEgoEvent(
                                    cfg,
                                    EgoEvent(cfg, self.real_preprocessed_input_path, self.real_dataset_root_path, temporal_bins=self.temporal_bins, split='test'),
                                    bg_data_root=self.bg_dataset_root_path, 
                                    bg_preprocessed_root=self.bg_preprocessed_input_path,
                                    split='finetune',
                                    temporal_bins=self.temporal_bins)

                self.eval_dataset = TemoralWrapper(eval_dataset, self.temporal_steps, split='test', sample_step=self.sample_step)
                self.train_dataset = TemoralWrapper(finetune_dataset, self.temporal_steps, split='train', sample_step=self.sample_step)

            
            elif self.training_type == 'finetuning':

                finetune_dataset = AugmentedEgoEvent(
                                    cfg,
                                    EgoEvent(cfg, self.wild_preprocessed_input_path, self.wild_dataset_root_path, temporal_bins=self.temporal_bins, split='train', finetune=True),
                                    bg_data_root=self.bg_dataset_root_path, 
                                    bg_preprocessed_root=self.bg_preprocessed_input_path,
                                    split='finetune',
                                    temporal_bins=self.temporal_bins)
                
                # cfg.DATASET.TYPE = 'Real'
                eval_dataset = AugmentedEgoEvent(
                                    cfg,
                                    EgoEvent(cfg, self.wild_preprocessed_input_path, self.wild_dataset_root_path, temporal_bins=self.temporal_bins, split='test'),
                                    bg_data_root=self.bg_dataset_root_path, 
                                    bg_preprocessed_root=self.bg_preprocessed_input_path,
                                    split='finetune',
                                    temporal_bins=self.temporal_bins)

                self.eval_dataset = TemoralWrapper(eval_dataset, self.temporal_steps, split='test', sample_step=self.sample_step)
                self.train_dataset = TemoralWrapper(finetune_dataset, self.temporal_steps, split='train', sample_step=self.sample_step)
            else:
                assert False, f"Invalid training type: {self.training_type}"

        if stage == "test" or stage == "predict":
            if self.training_type == 'train':
                test_dataset = self.real_dataset_root_path
                test_preprocessed_input = self.real_preprocessed_input_path
            elif self.training_type == 'finetuning':
                test_dataset = self.wild_dataset_root_path
                test_preprocessed_input = self.wild_preprocessed_input_path
            else:
                assert False, f"Invalid training type: {self.training_type}"
            
            self.test_dataset = EgoEventSequence(cfg, 
                                        test_preprocessed_input,
                                        test_dataset,
                                        self.bg_preprocessed_input_path,
                                        self.bg_dataset_root_path,
                                        training_type=self.training_type,
                                        use_bg_augmentation=False,
                                        fixed_sequence_length=self.fixed_sequence_length,
                                        temporal_bins=self.temporal_bins, 
                                        split='test')


    def train_dataloader(self):
        print("Total workers : ", self.workers)
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # collate_fn=collate_variable_size,
            shuffle=True, # TODO: change this
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True
        )
        self.train_dataloader_len = len(dataloader)
        return dataloader

    def val_dataloader(self):
        dataloader =  torch.utils.data.DataLoader(
            self.eval_dataset,
            # batch_size=1,
            batch_size=self.batch_size,
            # collate_fn=collate_variable_size,
            # collate_fn=lambda x: x[0],
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True
        )
        self.val_dataloader_len = len(dataloader)
        return dataloader

    def test_dataloader(self):
        dataloader =  torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            # batch_size=self.batch_size,
            collate_fn=lambda x: x[0],
            num_workers=self.workers,
            pin_memory=True, # TODO: check this
            drop_last=True
        )
        self.val_dataloader_len = len(dataloader)
        return dataloader


    def on_load_checkpoint(self, checkpoint: dict):
        checkpoint['optimizer_states'][0]['param_groups'][0]['lr'] =  self.lr

    def training_step(self, batch, batch_idx):
        end = time.time()
        loss = torch.tensor(0.1, requires_grad=True, device=self.device)

        self.model.kalman_filter.reset()

        prev_s5_states = {
            0: None,
            1: None,
            2: None,
            3: None
        }

        inps, outputs, meta_data = compute_fn_v4(self.model, batch, prev_s5_states, augmentation=True)

        gt_poses = meta_data['gt_j3d']
        gt_j2d = meta_data['gt_j2d']
        vis_j2d = meta_data['vis_j2d']
        vis_j3d = meta_data['vis_j3d']
        valid_j3d = meta_data['valid_j3d']
        vis_ja = meta_data['vis_ja']

        pred_poses_2d = camera_to_j2d_batch(outputs['abs_poses'].view(-1, 16, 3), self.image_size, self.ocam_model)
        gt_poses_2d = camera_to_j2d_batch(gt_poses.view(-1, 16, 3), self.image_size, self.ocam_model)
        pred_poses_2d = pred_poses_2d.reshape(outputs['abs_poses'].shape[0], outputs['abs_poses'].shape[1], 16, 2)
        gt_poses_2d = gt_poses_2d.reshape(gt_poses.shape[0], gt_poses.shape[1], 16, 2)

        gt_poses = gt_poses  * 1000 # scale to mm
        pred_poses = outputs['abs_poses'] * 1000 # scale to mm
        pred_delta_poses = outputs['delta_poses'] * 1000 # scale to mm
        valid_seg = valid_seg.view(self.temporal_steps, self.batch_size, 1, 1, 1)
        gt_delta_poses = gt_poses[1:, :, :, :] - gt_poses[:-1, :, :, :]
        
        s5_states = outputs['s5_states']
        for stage, s5_state in s5_states.items():
            s5_state.detach()

        loss_j3d_delta = self.criterions['delta_j3d'](pred_delta_poses, gt_delta_poses, vis_j3d[1:, :] * self.wgt_j3d_delta)
        loss_j3d = self.criterions['j3d'](pred_poses, gt_poses, vis_j3d * self.wgt_j3d)
        loss_j2d = self.criterions['j2d'](pred_poses_2d, gt_poses_2d, vis_j2d * self.wgt_j2d)
        loss_bone_length = self.criterions['bone_length'](pred_poses, gt_poses, vis_j3d * self.wgt_bone_length)
        loss_angle = self.criterions['bone_loss'](pred_poses, gt_poses, vis_ja * self.wgt_bone_angle, vis_ja * self.wgt_bone_length)

        loss_seg = 0

        loss = loss_j3d + loss_j2d + loss_bone_length + loss_j3d_delta + loss_angle

        self._update_metrics(loss, loss_angle, loss_j3d_delta, loss_seg, loss_j3d, loss_j2d, 
                        loss_bone_length, gt_poses, pred_poses, valid_j3d, inps)

        end = time.time()
        
        if batch_idx % cfg.PRINT_FREQ == 0:
            self._log_metrics()
            self._log_training_progress(batch_idx, end)
            self._log_memory_stats()
        
        return loss
    
    def evaluate(self, model, batch, s5_state, batch_idx):
        model.eval()
        
        with torch.no_grad():
            inps, outputs, meta_data = compute_fn_v4(model, batch, s5_state, augmentation=False)
        
        gt_abs_poses = meta_data['gt_j3d'] * 1000.0
        valid_j3d = meta_data['valid_j3d']
        valid_joints = meta_data['valid_joints']
        frame_index = meta_data['frame_index']
        vis_j3d = meta_data['vis_j3d']

        pred_abs_poses = outputs['abs_poses'] * 1000 # scale to mm
        
        # gt_abs_poses = gt_abs_poses_og * 1000 # scale to mm
        # valid_j3d = valid_j3d

        s5_state = outputs['s5_states']

        val_loss_j3d = self.criterions['j3d'](pred_abs_poses.unsqueeze(0), gt_abs_poses.unsqueeze(0), vis_j3d.unsqueeze(0) * self.wgt_j3d)
        self.j3d_loss_val.update(val_loss_j3d, inps.size(0))

        avg_acc, cnt = accuracy(gt_abs_poses, pred_abs_poses, valid_j3d)
        avg_acc_occlusion, cnt_occl = accuracy(gt_abs_poses, pred_abs_poses, (1 - valid_joints).unsqueeze(-1))

        self.per_joint_count += (1 - valid_joints).squeeze(1).sum(dim=0)
        self.total_count += valid_joints.shape[0] * valid_joints.shape[1]

        avg_jitter = compute_motion_jitter(pred_abs_poses, gt_abs_poses, valid_j3d)
        self.jitter_j3d_val.update(avg_jitter, cnt)
        self.acc_j3d_val.update(avg_acc, cnt)
        self.acc_occl_val.update(avg_acc_occlusion, cnt_occl)

        self.all_preds_j3d.append(pred_abs_poses.detach().cpu())
        self.all_gt_j3ds.append(gt_abs_poses.detach().cpu())
        self.all_vis_j3d.append(valid_j3d.detach().cpu())
        self.all_occluded_joints.append((1 - valid_joints).cpu())
        self.all_frame_indices.append(frame_index.detach().cpu())

        return s5_state
    
    
    def test_step_sequence(self, batch, batch_idx, prefix):
        self.model.eval()

        for idx in range(0, self.fixed_sequence_length, self.truncated_bptt_steps):
            start = idx
            end = start + self.truncated_bptt_steps
            data_batch = batch[start:end]
            batch_d = torch.utils.data._utils.collate.default_collate([data_batch])

            self.s5_state = self.evaluate(self.model, batch_d, self.s5_state, self.count)
            self.count = self.count + 1

        self.log('val_loss', self.j3d_loss_val.avg, sync_dist=True, batch_size=self.batch_size)
        self.log('val_acc', self.acc_j3d_val.avg, sync_dist=True, batch_size=self.batch_size)
        self.log('val_jitter', self.jitter_j3d_val.avg, sync_dist=True, batch_size=self.batch_size)
        self.log('val_acc_occl', self.acc_occl_val.avg, sync_dist=True, batch_size=self.batch_size)

            
        msg = 'Test: [{0}/{1}]\t' \
            'MPJPE {acc.val:.4f} ({acc.avg:.4f})\t' \
            'Jitter {jitter.val:.4f} ({jitter.avg:.4f})\t' \
            'Val loss  {val_loss.val:.4f} ({val_loss.avg:.4f})\t' \
            'Occlusion MPJPE  {acc_occl.val:.4f} ({acc_occl.avg:.4f})\t'.format(
                batch_idx, getattr(self.trainer, f"num_{prefix}_batches"),
                acc=self.acc_j3d_val, val_loss=self.j3d_loss_val, jitter=self.jitter_j3d_val, acc_occl=self.acc_occl_val)
        logger.info(msg)

    
    def eval_step(self, batch, batch_idx):
        self.model.eval()

        self.model.kalman_filter.reset()

        prev_s5_states = {
            0: None,
            1: None,
            2: None,
            3: None
        }

        # for idx in range(0, self.fixed_sequence_length, self.truncated_bptt_steps):
        start_time = time.time()

        # start = idx
        start = 0

        # end = start + self.truncated_bptt_steps
        end = start + len(batch)

        data_batch = batch[start:end]
        batch_d = batch


        inps, outputs, meta_data = compute_fn_v4(self.model, batch_d, prev_s5_states, augmentation=False)

        gt_abs_poses = meta_data['gt_j3d']
        vis_j3d = meta_data['vis_j3d']
        valid_j3d = meta_data['valid_j3d']

        pred_abs_poses_t = outputs['abs_poses'] * 1000 # scale to mm
        gt_j3d_t = gt_abs_poses * 1000 # scale to mm

        pred_abs_poses = pred_abs_poses_t
        gt_abs_poses = gt_j3d_t


        val_loss_j3d = self.criterions['j3d'](pred_abs_poses.unsqueeze(0), gt_abs_poses.unsqueeze(0), vis_j3d.unsqueeze(0) * self.wgt_j3d)
        self.j3d_loss_val.update(val_loss_j3d, inps.size(0))
        
        avg_acc, cnt = accuracy(gt_abs_poses, pred_abs_poses, valid_j3d)
        self.acc_j3d_val.update(avg_acc, cnt)

        avg_jitter = compute_motion_jitter(pred_abs_poses, gt_abs_poses, valid_j3d)
        self.jitter_j3d_val.update(avg_jitter, cnt)

        # measure elapsed time
        self.batch_time.update(time.time() - start_time)
        end = time.time()

        self.log('val_loss', self.j3d_loss_val.avg, sync_dist=True, batch_size=self.batch_size)
        self.log('val_acc', self.acc_j3d_val.avg, sync_dist=True, batch_size=self.batch_size)
        self.log('val_jitter', self.jitter_j3d_val.avg, sync_dist=True, batch_size=self.batch_size)


        self.global_val_steps = self.global_val_steps + 1
        
        msg = 'Test: [{0}/{1}]\t' \
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
            'MPJPE {acc.val:.4f} ({acc.avg:.4f})\t' \
            'Jitter {jitter.val:.4f} ({jitter.avg:.4f})\t' \
            'Val loss  {val_loss.val:.4f} ({val_loss.avg:.4f})\t'.format(
                batch_idx, self.trainer.num_val_batches, batch_time=self.batch_time,
                acc=self.acc_j3d_val, val_loss=self.j3d_loss_val, jitter=self.jitter_j3d_val)
        logger.info(msg)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.test_step_sequence(batch, batch_idx, "test")

    def visualize(self, lnes: np.ndarray):
        # if torch.Tensor, convert
        if isinstance(lnes, torch.Tensor):
            # assume shape (C, H, W)
            lnes = lnes.permute(1, 2, 0).detach().cpu().numpy()
        lnes = (lnes.copy() * 255).astype(np.uint8)
                    
        h, w = lnes.shape[:2]
        b = lnes[..., :1]      # channel 0 → blue
        r = lnes[..., 1:]      # channel 1 → red
        g = np.zeros((h, w, 1), dtype=np.uint8)  # green = 0

        rgb = np.concatenate([r, g, b], axis=2)
        return rgb

    def _log_metrics(self):
        self.log('train_total_loss', self.losses.avg)
        self.log('train_MPJPE', self.acc.avg)
        self.log('train_j3d_loss', self.j3d_losses.avg)
        self.log('train_j3d_delta_loss', self.j3d_delta_losses.avg)
        self.log('train_j2d_loss', self.j2d_losses.avg)
        # self.log('train_heatmap_loss', self.heatmap_losses.avg)
        self.log('train_bone_length_loss', self.bone_length_losses.avg)
        self.log('train_seg_loss', self.seg_losses.avg)
        self.log('train_bone_angle_loss', self.bone_angle_losses.avg)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr)

        self.global_steps = self.global_steps + 1

    def _log_memory_stats(self):
        memory_stats = torch.cuda.memory_stats("cuda:0")
        logger.info(f"Current allocated memory: {memory_stats['allocated_bytes.all.current'] / (1024 ** 2):.2f} MB")
        logger.info(f"Peak reserved memory: {memory_stats['reserved_bytes.all.peak'] / (1024 ** 2):.2f} MB")
    
    def _update_metrics(self, loss, loss_angle, loss_j3d_delta, loss_seg, loss_j3d, loss_j2d, 
                   loss_bone_length, gt_poses, pred_poses, valid_j3d, inps):
        avg_acc, cnt = accuracy(gt_poses, pred_poses, valid_j3d)

        self.j3d_delta_losses.update(loss_j3d_delta, inps.size(0))
        self.seg_losses.update(loss_seg, inps.size(0))
        self.j3d_losses.update(loss_j3d, inps.size(0))
        self.j2d_losses.update(loss_j2d, inps.size(0))
        self.bone_length_losses.update(loss_bone_length, inps.size(0))
        self.bone_angle_losses.update(loss_angle, inps.size(0))
        self.losses.update(loss, inps.size(0))
        self.acc.update(avg_acc, cnt)

    def _log_training_progress(self, batch_idx, end):
        self.batch_time.update(time.time() - end)
        
        msg = 'Epoch: [{0}][{1}/{2}]\t' \
            'Delta_J3D_Loss {j3d_delta_loss.val:.5f} ({j3d_delta_loss.avg:.5f})\t' \
            'J3D_Loss {j3d_loss.val:.5f} ({j3d_loss.avg:.5f})\t' \
            'SEG_Loss {seg_loss.val:.5f} ({seg_loss.avg:.5f})\t' \
            'J2D_Loss {j2d_loss.val:.5f} ({j2d_loss.avg:.5f})\t' \
            'Bone_Length_Loss {bone_length_loss.val:.5f} ({bone_length_loss.avg:.5f})\t' \
            'Bone_Angle_Loss {bone_angle_loss.val:.5f} ({bone_angle_loss.avg:.5f})\t' \
            'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
            'MPJPE {acc.val:.3f} ({acc.avg:.3f})'.format(
                self.current_epoch, batch_idx, self.trainer.num_training_batches,
                loss=self.losses, 
                j3d_loss=self.j3d_losses, 
                j3d_delta_loss=self.j3d_delta_losses,
                seg_loss=self.seg_losses,
                j2d_loss=self.j2d_losses,
                bone_length_loss=self.bone_length_losses,
                bone_angle_loss=self.bone_angle_losses,
                acc=self.acc
            )
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        print("LR :", current_lr)
        logger.info(msg)

    def on_validation_epoch_start(self):
        self.all_gt_j3ds = []
        self.all_preds_j3d = []
        self.all_vis_j3d = []
        self.all_occluded_joints = []
        self.all_frame_indices = []
        self.acc_j3d_val.reset()
        self.acc_occl_val.reset()
        self.jitter_j3d_val.reset()
        self.j3d_loss_val.reset()

    def on_train_epoch_start(self):
        self.batch_time.reset()
        self.data_time.reset()

    def on_fit_start(self):
        self.j3d_delta_losses.reset()
        self.seg_losses.reset()
        self.j3d_losses.reset()
        self.j2d_losses.reset()
        self.heatmap_losses.reset()
        self.bone_length_losses.reset()
        self.losses.reset()
        self.acc.reset()

    def configure_optimizers(self):
        optimizer = None

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

        # TODO: check the LR decay epochs (currently not used)
        lr_decay_epochs = list(self.lr_decay_epochs)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, lr_decay_epochs, cfg.TRAIN.LR_FACTOR
        )

        return [optimizer], [lr_scheduler]

class EvaluateCallback(Callback):
    def __init__(self):
        pass
    
    def on_test_end(self, trainer, pl_module):
        """
        This hook is called after all test batches are completed.
        Access model's variables here.
        """
        model = pl_module  # 'pl_module' is the model instance

        if model.test_dataset is not None:
            dataset = model.test_dataset
        else:
            dataset = model.eval_dataset

        all_preds_j3d = np.concatenate(model.all_preds_j3d, axis=0)
        all_gt_j3ds = np.concatenate(model.all_gt_j3ds, axis=0)
        all_vis_j3d = np.concatenate(model.all_vis_j3d, axis=0)
        all_frame_indices = np.concatenate(model.all_frame_indices, axis=0)
        all_occluded_joints = np.concatenate(model.all_occluded_joints, axis=0)

        np.save("/CT/EventEgo3Dv2/work/code_variations/dp_att_lkf_lnes_update_deform_att/Final_results/ee3dr_all_preds_j3d.npy", all_preds_j3d)
        np.save("/CT/EventEgo3Dv2/work/code_variations/dp_att_lkf_lnes_update_deform_att/Final_results/ee3dr_all_gt_j3ds.npy", all_gt_j3ds)
        np.save("/CT/EventEgo3Dv2/work/code_variations/dp_att_lkf_lnes_update_deform_att/Final_results/ee3dr_all_vis_j3d.npy", all_vis_j3d)
        np.save("/CT/EventEgo3Dv2/work/code_variations/dp_att_lkf_lnes_update_deform_att/Final_results/ee3dr_all_frame_indices.npy", all_frame_indices)

        # all_frame_indices = np.array(model.all_frame_indices)

        if model.dataset_type == "real":
            name_values, _ = dataset.evaluate_dataset(cfg, frame_indices=all_frame_indices, all_gt_j3ds=all_gt_j3ds, all_preds_j3d=all_preds_j3d, all_vis_j3d=all_vis_j3d)

            model_name = cfg.MODEL.NAME
            print("Error Per Actions : ")
            if isinstance(name_values, list):
                for name_value in name_values:
                    _print_name_value(name_value, model_name)
            else:
                _print_name_value(name_values, model_name)

        name_values, name_values_occl, _ = dataset.evaluate_joints(cfg, all_gt_j3ds=all_gt_j3ds, all_preds_j3d=all_preds_j3d, all_vis_j3d=all_vis_j3d, all_valid_joints=all_occluded_joints)
        model_name = cfg.MODEL.NAME
        print("Error Per Joints : ")
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        print("Occlusion Error Per Joints : ")
        if isinstance(name_values_occl, list):
            for name_value in name_values_occl:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values_occl, model_name)

        
        print("Avg Jitter e_smooth : ", model.jitter_j3d_val.avg)

        print("Per joint count : ", model.per_joint_count)
        print("Total count : ", model.total_count)

        sequence_range = {
            'walk': [0, 3500],
            'crouch': [3500, 6500],
            'pushup': [6500, 9000],
            'boxing': [9000, 12350],
            'kick': [12350, 15200],
            'dance': [15200, 17800],
            'inter. with env': [17800, 20700],
            'crawl': [20700, 23800],
            'sports': [23800, 33000],
            'jump': [33000, 200000], # max frame index
        }


        if SAVE_VISUALIZATION:
            for seq_name, seq_range in sequence_range.items():
                start_index, end_index = seq_range

                # import pdb; pdb.set_trace()
                current_seq_indices = all_frame_indices >= start_index
                current_seq_indices = np.logical_and(current_seq_indices, all_frame_indices < end_index)

                gt_j3ds = all_gt_j3ds[current_seq_indices] / 1000.0 # scale to mm
                preds_j3d = all_preds_j3d[current_seq_indices] / 1000.0 # scale to mm
                vis_j3d = all_vis_j3d[current_seq_indices]
                # event_repr = all_event_repr[:, np.newaxis, :][current_seq_indices]
                # event_repr = all_event_repr[current_seq_indices]
                frame_indices = all_frame_indices[current_seq_indices]

                os.makedirs(f"{VISUALIZATION_PATH}/{seq_name}", exist_ok=True)


                for i in range(gt_j3ds.shape[0]):
                    color = dump_sketelon_image(gt_j3ds[i], preds_j3d[i], "./visualizations/new_dataloader")
                    # import pdb; pdb.set_trace()

                    # event_image = visualize(event_repr[i].transpose(1, 2, 0))
                    # event_image = visualize(event_repr[i])

                    # concatenated_image = create_concatenated_image(color, event_image)
                    # output_path = f"{VISUALIZATION_PATH}/{seq_name}/{i}.png"
                    # cv2.imwrite(output_path, concatenated_image)

                    # concatenated_image = create_concatenated_image(color, event_image)
                    output_path = f"{VISUALIZATION_PATH}/{seq_name}/{i}.png"
                    cv2.imwrite(output_path, color)
