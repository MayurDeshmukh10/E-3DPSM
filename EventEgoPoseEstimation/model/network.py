from torch import nn

from EventEgoPoseEstimation.dataset.dataset_utils import event_augmentation_v2

from .base import Event3DPoseNet
from .learnable_kf import LearnableKalmanFilter


class EgoHPE(nn.Module):
    def __init__(
        self,
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
        self.event_3d_posenet = Event3DPoseNet(
            input_channels=posenet_input_channel,
            num_bins=self.num_bins,
            num_joints=num_joints,
        )

        self.enable_eros = eros
        self.inp_chn = input_channel
        self.width, self.height = image_size
        self.hm_width, self.hm_height = config.MODEL.HEATMAP_SIZE
        self.batch_size = batch_size

        self.kalman_filter = LearnableKalmanFilter(
            state_dim=16 * 3,
            batch_size=self.batch_size,
            measurement_var=1e-2,
            process_var=1e-3,
        )

    def forward(self, x, s5_state=None, augmentation_data=None, device='cuda'):
        del device

        if augmentation_data is None:
            augmentation_data = {}

        pose = None
        old_pose = None

        old_min, old_max, new_min, new_max = x.min(), x.max(), 0, 1
        out = (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
        out = event_augmentation_v2(self, out, augmentation_data)

        outs = self.event_3d_posenet(
            out,
            s5_state,
            pose,
            old_pose,
            self.kalman_filter,
            first_temporal_step=True,
        )

        return {
            'abs_poses': outs['pose'],
            'delta_poses': outs['delta_pose'],
            'all_abs_poses': outs['abs_pose'],
            'poses_old': outs['current_pose_old'],
            's5_states': outs['s5_state'],
        }
