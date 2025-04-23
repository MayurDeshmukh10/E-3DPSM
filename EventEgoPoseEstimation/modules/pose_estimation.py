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

import matplotlib.pyplot as plt


from EventEgoPoseEstimation.model import EgoHPE

from EventEgoPoseEstimation.dataset.dataset_utils import collate_variable_size, create_image, camera_to_j2d_batch

from EventEgoPoseEstimation.dataset import EgoEvent, EgoEventv2, AugmentedEgoEvent, TemoralWrapper, CombinedEgoEvent, EgoEventSequence

from EventEgoPoseEstimation.utils.utils import AverageMeter, save_checkpoint, create_logger

from EventEgoPoseEstimation.utils.skeleton import Skeleton

from configs.settings import config as cfg

from EventEgoPoseEstimation.core.function import compute_fn, _print_name_value, compute_fn_v2, compute_fn_v3, compute_fn_v4, compute_fn_new

from EventEgoPoseEstimation.core.evaluate import accuracy, accuracy_with_vis, accuracy_test, create_concatenated_image, compute_motion_jitter

# from EventEgoPoseEstimation.core.kalman_filter import apply_kalman_filtering

from EventEgoPoseEstimation.core.loss import SegmentationLoss, BoneLengthLoss, JointMSELoss, HeatMapJointsMSELoss, BoneOrientationLoss, BoneLoss

from EventEgoPoseEstimation.utils.vis import save_pose_images, save_debug_images, save_debug_3d_joints, save_debug_segmenation, save_debug_eros, generate_skeleton_image, dump_sketelon_image, drift_plot

from EventEgoPoseEstimation.core.inference import get_j2d_from_hms
from EventEgoPoseEstimation.dataset.dataset_utils import event_augmentation, save_augmented_data


import ocam

import torchvision

logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.EXP_NAME, 'train')

torch.set_float32_matmul_precision('medium')

import itertools
import os
import psutil

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

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
            'heatmap': HeatMapJointsMSELoss(use_target_weight=True).cuda(),
            'j2d': JointMSELoss(use_target_weight=True).cuda(),
            'seg': SegmentationLoss().cuda(),
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
        self.all_gt_j3ds = []
        self.all_preds_j3d = []
        self.all_vis_j3d = []
        self.all_frame_indices = []

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
        self.wgt_heatmap = loss_weights['heatmap']
        self.wgt_j3d_delta = loss_weights['delta_j3d']

        self.s5_states = None

        self.ocam_model = ocam.to_ocam_model('/CT/EventEgo3Dv2/work/egoposeformer/pose_estimation/models/utils/intrinsics.json')


        # self.initial_pose = torch.from_numpy(np.load('./initial_pose.npy')).unsqueeze(0).expand(self.model_batch_size, -1, -1)

        self.automatic_optimization = True

        # self.scaler = self.trainer.precision_plugin.scaler  # Lightning provides this if AMP is enabled

        # self.example_input_array = torch.Tensor(1, 1, 32768, 4)

        self.count = 0

        self.s5_state = None

    def forward(self, x):
        return self.model(x)
    
    def setup(self, stage: str):
    
        if stage == "fit":

            if self.training_type == 'pretrain':
                logger.info("Training type: Pretrain")
                cfg.DATASET.TYPE = 'Synthetic'
                if self.use_bg_augmentation:
                    pretrain_dataset = AugmentedEgoEvent(cfg, 
                                        EgoEvent(cfg, self.syn_preprocessed_input_path, self.syn_dataset_root_path, temporal_bins=self.temporal_bins, split='train'), 
                                        bg_data_root=self.bg_dataset_root_path, 
                                        bg_preprocessed_root=self.bg_preprocessed_input_path,
                                        split='train',
                                        temporal_bins=self.temporal_bins)
                else:
                    pretrain_dataset = EgoEvent(cfg, self.syn_preprocessed_input_path, self.syn_dataset_root_path, temporal_bins=self.temporal_bins, split='train')

                # cfg.DATASET.TYPE = 'Synthetic'
                # cfg.DATASET.SYN_ROOT = cfg.DATASET.SYN_TEST_ROOT 
                # TODO: change test to val again
                eval_dataset = AugmentedEgoEvent(
                                    cfg, 
                                    EgoEvent(cfg, self.syn_preprocessed_input_path, self.syn_test_dataset_root_path, temporal_bins=self.temporal_bins, split='test'), 
                                    bg_data_root=self.bg_dataset_root_path, 
                                    bg_preprocessed_root=self.bg_preprocessed_input_path,
                                    split='test', 
                                    temporal_bins=self.temporal_bins)
                
                self.eval_dataset = TemoralWrapper(eval_dataset, self.temporal_steps, split='test', sample_step=self.sample_step)
                self.train_dataset = TemoralWrapper(pretrain_dataset, self.temporal_steps, split='train', sample_step=self.sample_step)

            elif self.training_type == 'finetune':
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

            
            elif self.training_type == 'EE3D-W-finetuning':

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

                # finetune_dataset = EgoEventv2(cfg, self.real_dataset_root_path, self.real_preprocessed_input_path, temporal_bins=self.temporal_bins, split='train', finetune=True)
                # eval_dataset = EgoEventv2(cfg, self.real_dataset_root_path, self.real_preprocessed_input_path, temporal_bins=self.temporal_bins, split='test', finetune=True)

                # finetune_dataset = EgoEventv2(cfg, temporal_bins=self.temporal_bins, split='train', finetune=True)
                # eval_dataset = EgoEventv2(cfg, temporal_bins=self.temporal_bins, split='test', finetune=True)

                # self.eval_dataset = TemoralWrapper(eval_dataset, self.temporal_steps, split='test', sample_step=self.sample_step)
                # self.train_dataset = TemoralWrapper(finetune_dataset, self.temporal_steps, split='train', sample_step=self.sample_step)

                # self.eval_dataset = TemoralWrapper(EgoEvent(cfg, self.real_preprocessed_input_path, self.real_dataset_root_path, temporal_bins=self.temporal_bins, split='val'), self.temporal_steps, split='val', sample_step=self.sample_step)
                # self.train_dataset = TemoralWrapper(finetune_dataset, self.temporal_steps, split='train', sample_step=self.sample_step)
            else:
                assert False, f"Invalid training type: {self.training_type}"

        # TODO: Fix this
        # if stage == "test" or stage == "predict":
        #     if self.training_type == 'pretrain':

        #         # cfg.DATASET.SYN_ROOT = cfg.DATASET.SYN_TEST_ROOT 
        #         # cfg.DATASET.TYPE = 'Synthetic'
        #         # cfg.DATASET.BG_AUG = False
        #         # cfg.DATASET.TRAIN_TEST_SPLIT = 0
        #         test_dataset = self.syn_test_dataset_root_path
        #         test_preprocessed_input = self.syn_preprocessed_input_path

        #     elif self.training_type == 'finetune':
        #         # cfg.DATASET.TYPE = 'Real'
        #         # cfg.DATASET.BG_AUG = False
        #         test_dataset = self.real_dataset_root_path
        #         test_preprocessed_input = self.real_preprocessed_input_path
        #     else:
        #         assert False, f"Invalid training type: {self.training_type}"
 

        #     test_dataset = EgoEvent(cfg, test_preprocessed_input, test_dataset, temporal_bins=self.temporal_bins, split='test')
        #     self.test_dataset = TemoralWrapper(test_dataset, self.temporal_steps, split='test', sample_step=self.sample_step)

        if stage == "test" or stage == "predict":
            if self.training_type == 'pretrain':

                # cfg.DATASET.SYN_ROOT = cfg.DATASET.SYN_TEST_ROOT 
                # cfg.DATASET.TYPE = 'Synthetic'
                # cfg.DATASET.BG_AUG = False
                # cfg.DATASET.TRAIN_TEST_SPLIT = 0
                test_dataset = self.syn_test_dataset_root_path
                test_preprocessed_input = self.syn_preprocessed_input_path

            elif self.training_type == 'finetune':
                # cfg.DATASET.TYPE = 'Real'
                # cfg.DATASET.BG_AUG = False
                test_dataset = self.real_dataset_root_path
                test_preprocessed_input = self.real_preprocessed_input_path
            else:
                assert False, f"Invalid training type: {self.training_type}"


            # eval_dataset = AugmentedEgoEvent(
                                    # cfg,
                                    # EgoEvent(cfg, self.real_preprocessed_input_path, self.real_dataset_root_path, temporal_bins=self.temporal_bins, split='test'),
                                    # bg_data_root=self.bg_dataset_root_path, 
                                    # bg_preprocessed_root=self.bg_preprocessed_input_path,
                                    # split='finetune',
                                    # temporal_bins=self.temporal_bins)

            # self.test_dataset = TemoralWrapper(eval_dataset, self.temporal_steps, split='test', sample_step=self.sample_step)
            
            
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
            batch_size=1,
            # batch_size=self.batch_size,
            # collate_fn=collate_variable_size,
            collate_fn=lambda x: x[0],
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


    # def on_load_checkpoint(self, checkpoint: dict):
    #     checkpoint['optimizer_states'][0]['param_groups'][0]['lr'] =  self.lr

    def training_step(self, batch, batch_idx):
        end = time.time()
        loss = torch.tensor(0.1, requires_grad=True, device=self.device)

        self.model.kalman_filter.reset()

        inps, outputs, gt_hms, gt_poses, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index, pose_filename, vis_ja = compute_fn_v3(self.model, batch)
        # inps, outputs, gt_hms, gt_poses, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index, vis_ja = compute_fn_new(self.model, batch)


        meta = {'j3d': gt_poses, 'j2d': gt_j2d, 'vis_j2d': vis_j2d, 'vis_j3d': vis_j3d}

        
        pred_poses_2d = camera_to_j2d_batch(outputs['abs_poses'].view(-1, 16, 3), self.image_size, self.ocam_model)

        # import pdb; pdb.set_trace()

        # T, B, _, _ = outputs['abs_poses'].shape
        # gt_poses = gt_poses.reshape(T, B, 16, 3)
        # gt_seg = gt_seg.reshape(T, B, 1, 192, 256)
        # vis_j3d = vis_j3d.reshape(T, B, 16, 1)
        # vis_j2d = vis_j2d.reshape(T, B, 16, 1)
        # vis_ja = vis_ja.reshape(T, B, 16, 1)
        # valid_j3d = valid_j3d.reshape(T, B, 16, 1)


        gt_poses_2d = camera_to_j2d_batch(gt_poses.view(-1, 16, 3), self.image_size, self.ocam_model)

        # gt_poses_2d = camera_to_j2d_batch(gt_poses.view(-1, 16, 3), self.image_size, self.ocam_model)

        pred_poses_2d = pred_poses_2d.reshape(outputs['abs_poses'].shape[0], outputs['abs_poses'].shape[1], 16, 2)
        gt_poses_2d = gt_poses_2d.reshape(gt_poses.shape[0], gt_poses.shape[1], 16, 2)


        gt_poses_temp = gt_poses
        gt_poses = gt_poses  * 1000 # scale to mm
        pred_poses = outputs['abs_poses'] * 1000 # scale to mm
        pred_delta_poses = outputs['delta_poses'] * 1000 # scale to mm
        pred_seg = outputs['seg']
        valid_seg = valid_seg.view(self.temporal_steps, self.batch_size, 1, 1, 1)
        gt_delta_poses = gt_poses[1:, :, :, :] - gt_poses[:-1, :, :, :]
        # pred_heatmaps = outputs['heatmaps']
        
        s5_states = outputs['s5_states']
        s5_states.detach()


        # for i in range(gt_poses.shape[0]):
            # dump_sketelon_image(gt_poses_temp[i][0], gt_poses_temp[i][0], f"./visualizations/sanity/{i}_gt_j3d.png")
            # save_pose_images(pred_poses_2d.detach().cpu(), gt_poses_2d.detach().cpu(), '/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/sanity', mask_images=gt_seg.detach().cpu())

        loss_j3d_delta = self.criterions['delta_j3d'](pred_delta_poses, gt_delta_poses, vis_j3d[1:, :] * self.wgt_j3d_delta)
        loss_seg = self.criterions['seg'](pred_seg, gt_seg, valid_seg * self.wgt_seg)
        loss_j3d = self.criterions['j3d'](pred_poses, gt_poses, vis_j3d * self.wgt_j3d)
        loss_j2d = self.criterions['j2d'](pred_poses_2d, gt_poses_2d, vis_j2d * self.wgt_j2d)
        # loss_heatmaps = self.criterions['heatmap'](pred_heatmaps, gt_hms, vis_j2d * self.wgt_heatmap)
        loss_bone_length = self.criterions['bone_length'](pred_poses, gt_poses, vis_j3d * self.wgt_bone_length)
        loss_angle = self.criterions['bone_loss'](pred_poses, gt_poses, vis_ja * self.wgt_bone_angle, vis_ja * self.wgt_bone_length)

        loss = loss_j3d + loss_j2d + loss_seg + loss_bone_length + loss_j3d_delta + loss_angle

        self._update_metrics(loss, loss_angle, loss_j3d_delta, loss_seg, loss_j3d, loss_j2d, 
                        loss_bone_length, gt_poses, pred_poses, valid_j3d, inps)

        end = time.time()

        # self._log_metrics()
        # self._log_training_progress(batch_idx, end)
        # # print("Input shape ", inp.shape)
        # self._log_memory_stats()
        
        if batch_idx % cfg.PRINT_FREQ == 0:
            self._log_metrics()
            self._log_training_progress(batch_idx, end)
            self._log_memory_stats()

            # if int(self.batch_size) < 4:
            #     n_images = int(self.batch_size)
            # else:
            #     n_images = 4

            # if batch_idx % (cfg.PRINT_FREQ * 4) == 0:
            #     try:
            #         inp = inp.detach().cpu().numpy()
            #         gt_poses = gt_poses.detach()
            #         pred_poses = pred_poses.detach()
            #         gt_seg = gt_seg
            #         # pred_seg_detached = pred_seg_detached
            #         # pred_eros_image = pred_eros_image
            #         # representation_image = representation_image

            #         # save_debug_images(self, cfg, representation_image, meta, gt_hms, pred_hms, 'train', self.global_steps, n_images=n_images)
            #         save_debug_3d_joints(self, cfg, inp, meta, gt_poses, pred_poses, 'train', global_step=self.global_steps)
            #         # save_debug_segmenation(self, cfg, inp, meta, gt_seg, pred_seg_detached, 'train', global_step=self.global_steps)
            #         # save_debug_eros(self, cfg, representation_image, meta, pred_eros_image, 'train', global_step=self.global_steps)
            #     except Exception as e:
            #         logger.error("Error in saving debug data : {}".format(e))

        
        return loss

    
    def evaluate(self, model, batch, s5_state, batch_idx):
        model.eval()
        with torch.no_grad():
            inps, outputs, gt_hms, gt_abs_poses_og, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index, pose_filename = compute_fn_v4(model, batch, s5_state)

        pred_abs_poses = outputs['abs_poses'] * 1000 # scale to mm  (previous pose + current delta + kalman filtering)
        gt_abs_poses = gt_abs_poses_og * 1000 # scale to mm
        valid_j3d = valid_j3d

        s5_state = outputs['s5_states']

        val_loss_j3d = self.criterions['j3d'](pred_abs_poses.unsqueeze(0), gt_abs_poses.unsqueeze(0), vis_j3d.unsqueeze(0) * self.wgt_j3d)
        self.j3d_loss_val.update(val_loss_j3d, inps.size(0))

        # avg_acc, cnt, self.count = accuracy_with_vis(gt_abs_poses, pred_abs_poses, valid_j3d, batch_idx, outputs['abs_poses'].detach(), gt_abs_poses_og.detach(), inps, None, self.count)
        avg_acc, cnt = accuracy(gt_abs_poses, pred_abs_poses, valid_j3d)
        avg_jitter = compute_motion_jitter(pred_abs_poses, gt_abs_poses, valid_j3d)
        self.jitter_j3d_val.update(avg_jitter, cnt)
        self.acc_j3d_val.update(avg_acc, cnt)

        self.all_preds_j3d.append(pred_abs_poses.detach().cpu())
        self.all_gt_j3ds.append(gt_abs_poses.detach().cpu())
        self.all_vis_j3d.append(valid_j3d.detach().cpu())
        self.all_frame_indices.append(frame_index.detach().cpu())

        return s5_state
    
    
    def eval_step(self, batch, batch_idx, prefix, vis=False):
        self.model.eval()

        # self.model.kalman_filter.reset()

        # s5_state = None

        # import pdb; pdb.set_trace()


        if self.training_type in ['finetune', 'pretrain']:
            for idx in range(0, self.fixed_sequence_length, self.truncated_bptt_steps):
                self.model.kalman_filter.reset()
                s5_state = None
                start = idx
                end = start + self.truncated_bptt_steps
                data_batch = batch[start:end]
                batch_d = torch.utils.data._utils.collate.default_collate([data_batch])

                self.s5_state = self.evaluate(self.model, batch_d, self.s5_state, batch_idx)

        
        elif self.training_type == 'pretrain':
            start = 0
            end = start + len(batch)
            data_batch = batch[start:end]
            batch_d = torch.utils.data._utils.collate.default_collate([data_batch])

            self.s5_state = self.evaluate(self.model, batch_d, self.s5_state, batch_idx)

        self.log('val_loss', self.j3d_loss_val.avg, sync_dist=True, batch_size=self.batch_size)
        self.log('val_acc', self.acc_j3d_val.avg, sync_dist=True, batch_size=self.batch_size)
        self.log('val_jitter', self.jitter_j3d_val.avg, sync_dist=True, batch_size=self.batch_size)
            
        msg = 'Test: [{0}/{1}]\t' \
            'MPJPE {acc.val:.4f} ({acc.avg:.4f})\t' \
            'Jitter {jitter.val:.4f} ({jitter.avg:.4f})\t' \
            'Val loss  {val_loss.val:.4f} ({val_loss.avg:.4f})\t'.format(
                batch_idx, getattr(self.trainer, f"num_{prefix}_batches"),
                acc=self.acc_j3d_val, val_loss=self.j3d_loss_val, jitter=self.jitter_j3d_val)
        logger.info(msg)

    
    # def eval_step(self, batch, batch_idx, prefix, vis=False):
    #     self.model.eval()

    #     self.model.kalman_filter.reset()

    #     if vis == False:

    #         # for idx in range(0, self.fixed_sequence_length, self.truncated_bptt_steps):
    #         start_time = time.time()

    #         # start = idx
    #         start = 0

    #         # end = start + self.truncated_bptt_steps
    #         end = start + len(batch)

    #         data_batch = batch[start:end]


    #         batch_d = torch.utils.data._utils.collate.default_collate([data_batch])


    #         inps, outputs, gt_hms, gt_abs_poses_og, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index, pose_filename = compute_fn_v4(self.model, batch_d)
    #         # inps, outputs, gt_hms, gt_abs_poses_og, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index, vis_ja= compute_fn_new(self.model, batch)
            

    #         # T, B = self.temporal_steps, self.batch_size
            
    #         # gt_abs_poses_og = gt_abs_poses_og.reshape(T, B, 16, 3)
    #         # gt_seg = gt_seg.reshape(T, B, 1, 192, 256)
    #         # vis_j3d = vis_j3d.reshape(T, B, 16, 1)
    #         # vis_j2d = vis_j2d.reshape(T, B, 16, 1)
    #         # vis_ja = vis_ja.reshape(T, B, 16, 1)
    #         # valid_j3d = valid_j3d.reshape(T, B, 16, 1)


    #         pred_abs_poses_t = outputs['abs_poses'] * 1000 # scale to mm  (previous pose + current delta + kalman filtering)
    #         # pred_abs_poses_t = outputs['poses_old'] * 1000 # previous pose + current delta + no kalman filtering
    #         # pred_abs_poses_t = outputs['all_abs_poses'] * 1000 # only current predicted abs pose
    #         gt_j3d_t = gt_abs_poses_og * 1000 # scale to mm

    #         pred_delta_poses = outputs['delta_poses'] * 1000 

    #         # # Apply Kalman filtering to reduce drift
    #         # confidence_estimator = None  # Use default confidence values
    #         # # Alternatively, create a more sophisticated confidence estimator
    #         # # confidence_estimator = lambda abs_pose, delta: (0.7, 0.3)  # Example fixed confidence

    #         # filtered_poses = apply_kalman_filtering(
    #         #     pred_abs_poses_t, 
    #         #     pred_delta_poses, 
    #         #     confidence_estimator
    #         # )

    #         # ----- Apply Kalman filtering to correct drift -----
    #         # We apply the filter per sequence (per batch element). Adjust dt if needed.
    #         # dt = 1/1000.  # Change this if your sequence has a different time step.
    #         # # Q_scale = 3.0
    #         # # R_scale = 1.0

    #         # Q_scale = 5.0
    #         # R_scale = 0.1

    #         # T, B, J, _ = pred_abs_poses_t.shape
    #         # fused_abs_poses = torch.zeros_like(pred_abs_poses_t)
    #         # # Loop over batch elements and apply the filter on each sequence
    #         # for b in range(B):
    #         #     # Convert the T x J x 3 tensor to numpy array.
    #         #     abs_seq = pred_abs_poses_t[:, b, :, :].detach().cpu().numpy()
    #         #     # Apply Kalman filter to the sequence.
    #         #     filtered_seq = apply_kf_to_sequence(abs_seq, dt=dt, Q_scale=Q_scale, R_scale=R_scale)
    #         #     # Convert back to tensor and store.
    #         #     fused_abs_poses[:, b, :, :] = torch.from_numpy(filtered_seq).to(pred_abs_poses_t.device)

    #         # Q_scale = 100.0
    #         # R_scale = 10.0

    #         # T, B, J, _ = pred_abs_poses_t.shape
    #         # fused_abs_poses1 = torch.zeros_like(pred_abs_poses_t)
    #         # # Loop over batch elements and apply the filter on each sequence
    #         # for b in range(B):
    #         #     # Convert the T x J x 3 tensor to numpy array.
    #         #     abs_seq = pred_abs_poses_t[:, b, :, :].detach().cpu().numpy()
    #         #     # Apply Kalman filter to the sequence.
    #         #     filtered_seq = apply_kf_to_sequence(abs_seq, dt=dt, Q_scale=Q_scale, R_scale=R_scale)
    #         #     # Convert back to tensor and store.
    #         #     fused_abs_poses1[:, b, :, :] = torch.from_numpy(filtered_seq).to(pred_abs_poses_t.device)

    #         # # pred_abs_poses = pred_abs_poses[-1, :, :, :] # use abs poses of last temporal step
    #         # # gt_abs_poses = gt_abs_poses[-1, :, :, :]

    #         # pred_abs_poses = pred_abs_poses_t[-1, :, :, :] # use abs poses of last temporal step
    #         # pred_abs_poses = fused_abs_poses[-1, :, :, :] # use abs poses of last temporal step

    #         # gt_abs_poses = gt_j3d_t[-1, :, :, :]
    #         valid_j3d_t = valid_j3d

    #         pred_abs_poses = pred_abs_poses_t
    #         gt_abs_poses = gt_j3d_t


    #         # valid_j3d = valid_j3d[-1, :, :, :]
    #         # vis_j3d = vis_j3d[-1, :, :, :]


    #         gt_j3d = gt_j3d_t.squeeze(1)
    #         # valid_j3d = valid_j3d.squeeze(1)


    #         val_loss_j3d = self.criterions['j3d'](pred_abs_poses.unsqueeze(0), gt_abs_poses.unsqueeze(0), vis_j3d.unsqueeze(0) * self.wgt_j3d)
    #         self.j3d_loss_val.update(val_loss_j3d, inps.size(0))

    #         # process = psutil.Process(os.getpid())
    #         # # Convert bytes to MB
    #         # memory_usage_mb = process.memory_info().rss / (1024 ** 2)

    #         # print(f"Current process memory usage: {memory_usage_mb:.2f} MB")

    #         # pred_abs_poses_no_kf = outputs['poses_old'] * 1000 # previous pose + current delta + no kalman filtering
    #         # pred_abs_poses_only = outputs['all_abs_poses'] * 1000 # only current predicted abs pose

    #         # a_abs_only = []
    #         # a_w_kf = []
    #         # a_wo_kf = []
    #         # import pdb; pdb.set_trace()

    #         # for i in range(len(batch)):
    #         #     # acc_no_pf, _ = accuracy(gt_abs_poses[i], pred_abs_poses_no_kf[i], valid_j3d[i])
    #         #     # acc_only_abs, _ = accuracy(gt_abs_poses[i], pred_abs_poses_only[i], valid_j3d[i])

    #         #     # avg_acc, cnt = accuracy(gt_abs_poses[i], pred_abs_poses[i], valid_j3d[i])
    #         #     # a_abs_only.append(acc_only_abs)
    #         #     # a_w_kf.append(avg_acc)
    #         #     # a_wo_kf.append(acc_no_pf)

    #         #     color = dump_sketelon_image(gt_abs_poses_og[i][0].detach(), outputs['abs_poses'][i][0].detach(), f"./visualizations/new_dataloader")
    #         #     voxel_image = save_augmented_data(inps[i], 'test')
    #         #     concatenated_image = create_concatenated_image(color, voxel_image)
    #         #     output_path = f"./visualizations/output_for_vis_lnes/{i}.png"
    #         #     cv2.imwrite(output_path, concatenated_image)




    #         # avg_acc = torch.Tensor(a_w_kf)
    #         # acc_only_abs = torch.Tensor(a_abs_only)
    #         # acc_no_pf = torch.Tensor(a_wo_kf)

    #         # all_pred_abs = outputs['all_abs_poses'] * 1000
    #         # poses_old = outputs['poses_old'] * 1000
    #         # # re_poses = outputs['re_poses'] * 1000
    #         # abs_avg_acc_all_steps, cnt_0 = accuracy_test(gt_j3d_t[:, 0, ...], all_pred_abs[:, 0, ...], valid_j3d_t[:, 0, ...])
    #         # delta_avg_acc_all_steps, cnt_0 = accuracy_test(gt_j3d_t[:, 0, ...], pred_abs_poses_t[:, 0, ...], valid_j3d_t[:, 0, ...])
    #         # # reposes_avg_acc_all_steps, cnt_0 = accuracy_test(gt_j3d_t[:, 0, ...], re_poses[:, 0, ...], valid_j3d_t[:, 0, ...])

    #         # pose_old_avg_acc_all_steps, cnt_0 = accuracy_test(gt_j3d_t[:, 0, ...], poses_old[:, 0, ...], valid_j3d_t[:, 0, ...])


    #         # filtered_avg_acc_all_steps, cnt_0 = accuracy_test(gt_j3d_t[:, 0, ...], fused_abs_poses[:, 0, ...], valid_j3d_t[:, 0, ...])
    #         # filtered_avg_acc_all_steps_q10, cnt_0 = accuracy_test(gt_j3d_t[:, 0, ...], fused_abs_poses1[:, 0, ...], valid_j3d_t[:, 0, ...])

    #         # import pdb; pdb.set_trace()

    #         # seq_end = 20

    #         # acc_no_pf, _ = accuracy(gt_abs_poses, pred_abs_poses_no_kf, valid_j3d)
    #         # acc_only_abs, _ = accuracy(gt_abs_poses, pred_abs_poses_only, valid_j3d)

    #         # avg_acc, cnt = accuracy(gt_abs_poses, pred_abs_poses, valid_j3d)


    #         # drift_plot(avg_acc[:seq_end], acc_only_abs[:seq_end], acc_no_pf[:seq_end])

    #         # import pdb; pdb.set_trace()


    #         # print("Input shape : ", inp.shape)
    #         # avg_acc, cnt = accuracy_with_vis(gt_abs_poses, pred_abs_poses, valid_j3d, batch_idx, outputs['abs_poses'].detach(), gt_abs_poses_og.detach(), outputs['voxel_representations'], pose_filename, frame_index)

    #         # for i in range(gt_abs_poses_og.size(0)):
    #         #     dump_sketelon_image(gt_abs_poses_og[i][0].detach(), outputs['abs_poses'][i][0].detach(), f"./visualizations/new_dataloader/{batch_idx}_bin-{i}_mpjpe_{avg_acc.item()}.png")
            
    #         avg_acc, cnt = accuracy(gt_abs_poses, pred_abs_poses, valid_j3d)
    #         self.acc_j3d_val.update(avg_acc, cnt)

    #         avg_jitter = compute_motion_jitter(pred_abs_poses, gt_abs_poses, valid_j3d)
    #         self.jitter_j3d_val.update(avg_jitter, cnt)


            
    #         # measure elapsed time
    #         self.batch_time.update(time.time() - start_time)
    #         end = time.time()

    #         if prefix == 'test':
    #             self.all_preds_j3d.append(pred_abs_poses.detach().cpu())
    #             self.all_gt_j3ds.append(gt_abs_poses.detach().cpu())
    #             self.all_vis_j3d.append(valid_j3d.detach().cpu())
    #             self.all_frame_indices.append(frame_index.detach().cpu())


    #         self.log('val_loss', self.j3d_loss_val.avg, sync_dist=True, batch_size=self.batch_size)
    #         self.log('val_acc', self.acc_j3d_val.avg, sync_dist=True, batch_size=self.batch_size)
    #         self.log('val_jitter', self.jitter_j3d_val.avg, sync_dist=True, batch_size=self.batch_size)

    #         # if batch_idx % cfg.PRINT_FREQ == 0:

    #         self.global_val_steps = self.global_val_steps + 1
            
    #         msg = 'Test: [{0}/{1}]\t' \
    #             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
    #             'MPJPE {acc.val:.4f} ({acc.avg:.4f})\t' \
    #             'Jitter {jitter.val:.4f} ({jitter.avg:.4f})\t' \
    #             'Val loss  {val_loss.val:.4f} ({val_loss.avg:.4f})\t'.format(
    #                 batch_idx, self.trainer.num_val_batches, batch_time=self.batch_time,
    #                 acc=self.acc_j3d_val, val_loss=self.j3d_loss_val, jitter=self.jitter_j3d_val)
    #         logger.info(msg)

    #     elif prefix == "test" and vis == True:
    #         global_steps=batch_idx
    #         tb_log_dir = self.logger.log_dir
    #         test_and_generate_vis(cfg, self.model, self.test_dataset, tb_log_dir, global_steps)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test", vis=False)
    

    # def on_train_batch_start(self, batch, batch_idx):
    #     if self.s5_states is not None:
    #         self.s5_states.detach()
    #         self.s5_states = None


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
        # logger.info("Batch shape: {}".format(inp.shape))
        logger.info(f"Current allocated memory: {memory_stats['allocated_bytes.all.current'] / (1024 ** 2):.2f} MB")
        # logger.info(f"Peak allocated memory: {memory_stats['allocated_bytes.all.peak'] / (1024 ** 2):.2f} MB")
        # logger.info(f"Current reserved memory: {memory_stats['reserved_bytes.all.current'] / (1024 ** 2):.2f} MB")
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
        self.all_frame_indices = []
        self.acc_j3d_val.reset()
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


        if self.training_type == 'finetune':
            with torch.no_grad():
                default_process_var = 1e-3
                default_measurement_var = 1e-2
                self.model.kalman_filter.log_process_var.data.fill_(np.log(default_process_var))
                self.model.kalman_filter.log_measurement_var.data.fill_(np.log(default_measurement_var))

        # if self.lr:
        #     # Overwrite learning rate after running LearningRateFinder
        #     for optimizer in self.trainer.optimizers:
        #         for param_group in optimizer.param_groups:
        #             param_group["lr"] = self.lr

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

def test_and_generate_vis(cfg, model, test_dataset, tb_log_dir, global_steps):
    fps = 30
    seq_time_in_sec = 60

    seq_len = seq_time_in_sec * fps
    data_len = len(test_dataset)

    # start = np.random.randint(0, data_len - cfg.DATASET.TEMPORAL_STEPS)
    start = np.random.randint(0, data_len - seq_len)
    stop = min(start + seq_len, data_len)

    tb_log_dir = Path(tb_log_dir)
    video_path = str(tb_log_dir / f'{global_steps}.mp4')

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (400 * 1, 300 * 1))

    model.eval()

    print(f"Path : {video_path}")
    print(f"Generating video from {start} to {stop}")
    for i in range(start, stop):
        gt_j3d = []
        inps = []
        gt_hms = []
        
        data, meta = test_dataset[i]

        inp = data['x']

        gt_j3d_ = data['j3d']

        inps.append(inp[None, None, ...])
        gt_j3d.append(gt_j3d_[None, ...])
        # gt_hms.append(data['hms'][None, ...])

        inps = torch.cat(inps, dim=0).cuda()
        
        with torch.no_grad():
            outputs = model(inps)

        pred_j3ds = outputs['j3d'].detach()
        # preds_hms = outputs['hms'].detach()
        # pred_j2ds = get_j2d_from_hms(cfg, preds_hms)
        
        gt_j3ds = torch.cat(gt_j3d, dim=0).detach()
        # gt_hms = torch.cat(gt_hms, dim=0).detach()
        # gt_hm_j2ds = get_j2d_from_hms(cfg, gt_hms)

        # representation = outputs['representation']
        # representation_image = create_image(representation)

        T, B, N, C = inps.shape

        for i in range(T):
            gt_j3d = gt_j3ds[i]
            # gt_hm = gt_hms[i]
            # gt_hm_j2d = gt_hm_j2ds[i]
            
            pred_j3d = pred_j3ds[i]

            # if gt_j3d.dim() == 0 or pred_j3d.dim() == 0:
            #     continue


            # TODO: Include all bins in the visualization
            # for j in range(len(gt_j3d)):

            # try:
            # gt_j3d = gt_j3d[j]
            # pred_j3d = pred_j3d[j]
            # except:
            #     import pdb; pdb.set_trace()


            # import pdb; pdb.set_trace()
            gt_j3d = gt_j3d[-1, :, :]
            pred_j3d = pred_j3d[-1, :, :]

            # pred_j2d = pred_j2ds[i]
            # pred_hm = preds_hms[i]

            # inp = representation_image
            # inp = inp[i]
            # grid = torchvision.utils.make_grid(inp)
            # inp = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).detach()
        

            # pred_hm_image = plot_heatmaps(inp, pred_hm)    
            # gt_hm_image = plot_heatmaps(inp, gt_hm)

            # inp = inp.astype(np.uint8)

            # inp_w_gt_hm_j2d = Skeleton.draw_2d_skeleton(inp, gt_hm_j2d, lines=True)
            # inp = Skeleton.draw_2d_skeleton(inp, pred_j2d, lines=True)
                    
            color = generate_skeleton_image(gt_j3d, pred_j3d)
            color = color[..., ::-1]
            
            color = cv2.resize(color, (400, 300))
            # inp = cv2.resize(inp, (400, 300))
            # pred_hm_image = cv2.resize(pred_hm_image, (400, 300))
            # gt_hm_image = cv2.resize(gt_hm_image, (400, 300))
            # inp_w_gt_hm_j2d = cv2.resize(inp_w_gt_hm_j2d, (400, 300))           
            
            # hstack1 = np.concatenate([inp, color], axis=1)
            # hstack2 = np.concatenate([gt_hm_image, pred_hm_image], axis=1)
            # hstack3 = np.concatenate([inp_w_gt_hm_j2d, np.zeros_like(inp_w_gt_hm_j2d)], axis=1)
            
            # vstack = np.concatenate([hstack1, hstack2, hstack3], axis=0)
            video.write(color)
            # video.write(vstack)

    video.release()

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

        # np.save("all_preds_j3d.npy", all_preds_j3d)
        # np.save("all_gt_j3ds.npy", all_gt_j3ds)
        # np.save("all_vis_j3d.npy", all_vis_j3d)
        # np.save("all_frame_indices.npy", all_frame_indices)

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

        name_values, _ = dataset.evaluate_joints(cfg, all_gt_j3ds=all_gt_j3ds, all_preds_j3d=all_preds_j3d, all_vis_j3d=all_vis_j3d)
        model_name = cfg.MODEL.NAME
        print("Error Per Joints : ")
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        
        print("Avg Jitter e_smooth : ", model.jitter_j3d_val.avg)
