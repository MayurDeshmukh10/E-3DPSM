import torch
from torch import nn
from torch.nn import functional as F
from .base import Event3DPoseNet
from os.path import join, dirname, isfile
import numpy as np
from EventEgoPoseEstimation.utils.vis import visualize_temporal_bins
from EventEgoPoseEstimation.dataset.dataset_utils import event_augmentation, save_augmented_data, event_augmentation_v2

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from .learnable_kf import LearnableKalmanFilter


class EgoHPE(nn.Module):
    def __init__(self,
            config,
            num_joints,
            eros,
            input_channel,
            posenet_input_channel,
            image_size,
            batch_size,
            ):
        super(EgoHPE, self).__init__()

        self.n_joints = num_joints

        self.num_bins = int(input_channel / 2)

        # self.blaze_pose = BlazePose(config)
        self.event_3d_posenet = Event3DPoseNet(
            input_channels=posenet_input_channel,
            num_bins=self.num_bins,
            num_joints=num_joints
        )

        self.enable_eros = eros
        self.inp_chn = input_channel
        self.width, self.height = image_size
        self.hm_width, self.hm_height = config.MODEL.HEATMAP_SIZE

        self.batch_size = batch_size

        self.kalman_filter = LearnableKalmanFilter(
            state_dim=16*3,
            batch_size=self.batch_size,
            measurement_var=1e-2,
            process_var=1e-3
        )

        # self.EROS = EROS(inp_chn=self.num_bins, height=self.height, width=self.width, batch_size=self.batch_size)


    def forward(self, x, s5_state=None, augmentation_data={}, device='cuda'):

        # s5_state = None
        re_s5_state = None
        pose = None
        old_pose = None
        re_pose = None

        T, B, C, H, W = x.shape
      
        delta_poses = []
        poses = []
        seg_outs = []
        heatmaps = []
        s5_states = []
        voxel_outs = []
        event_voxels = []
        all_abs_poses = []
        re_poses = []
        poses_old = []

        old_min, old_max, new_min, new_max = x.min(), x.max(), 0, 1
        out = (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

        out = event_augmentation_v2(self, out, augmentation_data)

        outs = self.event_3d_posenet(out, s5_state, pose, old_pose, self.kalman_filter, first_temporal_step=True)

        s5_states = outs['s5_state']
        poses = outs['pose']
        delta_poses = outs['delta_pose']
        seg_outs = outs['seg']
        all_abs_poses = outs['abs_pose']
        poses_old = outs['current_pose_old']


        outputs = {}
        outputs['abs_poses'] = poses  # previous pose + current delta + kalman filtering
        outputs['seg'] = seg_outs
        outputs['delta_poses'] = delta_poses
        outputs['all_abs_poses'] = all_abs_poses # only current predicted abs pose
        outputs['poses_old'] = poses_old # previous pose + current delta + no kalman filtering
        outputs['s5_states'] = s5_states



        # for i in range(T):

        #     out = x[i]
        #     old_min, old_max, new_min, new_max = out.min(), out.max(), 0, 1
        #     out = (out - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

        #     # out = self.EROS(x[i], device)

        #     # import pdb; pdb.set_trace()
        #     # if not all(is_empty(inner) for sublist in augmentation_data['bg_data'] for inner in sublist):
        #         # out = event_augmentation(self, out, i, augmentation_data)

            
        #     # out = event_voxels[i]
        #     # out_copy = event_voxels_copy[i]
            

        #     # for ii, o in enumerate(out):
        #     # #     visualize_temporal_bins(o.detach(), f'/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/opt_augmented/{ii}.png')
        #     # #     visualize_temporal_bins(out_copy[ii].detach(), f'/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/opt_augmented/og_{ii}.png')
        #     #     save_augmented_data(x[ii][0].detach(), f'/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/test_kalman/aug/sum_{ii}.png')
        #     #     save_augmented_data(old_x[ii][0].detach(), f'/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/test_kalman/og/sum_og_{ii}.png')


        #     outs, kf = self.event_3d_posenet(out, s5_state, pose, old_pose, self.kalman_filter, first_temporal_step=(i == 0))
        #     # outs = self.event_3d_posenet(out, s5_state, pose, first_temporal_step=True)

        #     # re_outs, kf = self.event_3d_posenet(out, re_s5_state, re_pose, kf, first_temporal_step=first_step)

        #     # re_s5_state = re_outs['s5_state'].detach()
        #     # re_pose = re_outs['pose'] 

        #     # re_poses.append(re_pose)


        #     # outs = self.event_3d_posenet(event_voxels[i], s5_state, pose, first_temporal_step=(i == 0))

        #     # import pdb; pdb.set_trace()

        #     old_pose = outs['current_pose_old']

        #     s5_state = outs['s5_state'].detach()
        #     pose = outs['pose']
        #     delta_pose = outs['delta_pose']
        #     seg_out = outs['seg']
        #     seg_feature = outs['seg_feature']
        #     abs_pose = outs['abs_pose']

        #     poses_old.append(old_pose)
        #     all_abs_poses.append(abs_pose)
        #     # heatmap = outs['heatmaps']
        #     if delta_pose is not None:
        #         delta_poses.append(delta_pose)
        #     poses.append(pose)
        #     # s5_states.append(s5_state)
        #     seg_outs.append(seg_out)
        #     # heatmaps.append(heatmap)
        #     voxel_outs.append(out)

        # abs_poses = torch.stack(poses)
        # if len(delta_poses) > 0:
        #     delta_poses = torch.stack(delta_poses)
        # seg_outs = torch.stack(seg_outs)
        # # heatmaps = torch.stack(heatmaps)
        # voxel_representations = torch.stack(voxel_outs)

        # all_abs_poses = torch.stack(all_abs_poses)
        # poses_old = torch.stack(poses_old)
        # s5_states = torch.stack(s5_states)

        # re_poses = torch.stack(re_poses)
        
        # outputs = {}
        # outputs['abs_poses'] = abs_poses  # previous pose + current delta + kalman filtering
        # outputs['seg'] = seg_outs
        # outputs['delta_poses'] = delta_poses
        # # outputs['heatmaps'] = heatmaps
        # outputs['voxel_representations'] = voxel_representations

        # outputs['all_abs_poses'] = all_abs_poses # only current predicted abs pose
        # outputs['poses_old'] = poses_old # previous pose + current delta + no kalman filtering
        # outputs['re_poses'] = re_poses
        # outputs['s5_states'] = s5_states
 
        return outputs

def is_empty(item):
    if isinstance(item, torch.Tensor):
        return item.numel() == 0  # True if the tensor has zero elements
    else:
        return len(item) == 0     # Works for lists and other sequences