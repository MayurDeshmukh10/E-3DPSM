import torch
import torch.nn as nn

from .blazebase import BlazeBlock, DecoderConv, Head
from .s5.s5_model import S5Block
import torch.nn.functional as F
from einops import rearrange
from .encoder import EncoderBlock, ResidualBlock, SpatialTransformer


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from filterpy.kalman import KalmanFilter

def apply_kf_to_sequence(abs_poses, delta_poses, dt=1/1000., Q_scale=3.0, R_scale=1.0):
    state_vector_dim = 16 * 3
    measurement_vector_dim = 16 * 3 
    dim = 16 * 3

    kf = KalmanFilter(dim_x=state_vector_dim, dim_z=measurement_vector_dim)

    kf.x = abs_poses[0].reshape(-1)
    
    # kf.P = np.eye(dim) * 1.0 # State covariance matrix: initial uncertainty in the state
    kf.F = np.eye(dim) # State transition matrix: identity since state update is additive
    kf.B = np.eye(dim) # Control transition matrix: identity to map the delta (u) directly to state
    kf.H = np.eye(dim) # Measurement function: identity since we directly measure joint positions
    
    
    kf.Q = np.eye(dim) * process_var # Process noise covariance matrix: uncertainty in the prediction (delta)
    kf.R = np.eye(dim) * measurement_var # Measurement noise covariance matrix: uncertainty in the absolute pose measurement

    u = delta_change.reshape(-1, 1)
    z = abs_pose_measurement.reshape(-1, 1)

    kf.predict(u=u)


class JointRegressor(nn.Module):
    def __init__(self, in_channels, num_joints):
        super(JointRegressor, self).__init__()
        self.num_joints = num_joints

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.LayerNorm(in_channels // 4),
            nn.GELU(),
            nn.Linear(in_channels // 4, in_channels // 8),
            nn.LayerNorm(in_channels // 8),
            nn.GELU(),
            nn.Linear(in_channels // 8, num_joints * 3)
        )

    def forward(self, features):
        x = self.mlp(features)  # [B, num_joints * 3]
        x = x.view(x.size(0), self.num_joints, 3)  # Reshape to [B, num_joints, 3]
        return x

class PoseEmbedding(nn.Module):
    def __init__(self, num_joints, embed_dim=32):
        super(PoseEmbedding, self).__init__()
        self.num_joints = num_joints
        self.fc = nn.Sequential(
            nn.Linear(num_joints * 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, embed_dim)
        )
        
    def forward(self, pose):
        # pose: [B, num_joints, 3]
        B = pose.size(0)
        x = pose.view(B, -1)  # Flatten to shape [B, num_joints * 3]
        embedding = self.fc(x)  # [B, embed_dim]
        return embedding

class Event3DPoseNet(nn.Module):
    def __init__(self, 
        num_bins,
        input_channels,
        num_joints
    ):
        super(Event3DPoseNet, self).__init__()
        
        self.num_bins = num_bins
        self.inp_chn = input_channels
        self.n_joints = num_joints
        self.pose_embed_dim = 64
        self._define_layers()

    def _define_layers(self):

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inp_chn, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.encoder_backbone = nn.ModuleList([
            nn.Sequential(
                EncoderBlock(16, 32),
                ResidualBlock(32, 32),
                SpatialTransformer(32, 4, 8)
            ),
            nn.Sequential(
                EncoderBlock(32, 64),
                ResidualBlock(64, 64),
                SpatialTransformer(64, 4, 16)
            ),
            nn.Sequential(
                EncoderBlock(64, 128),
                ResidualBlock(128, 128),
                SpatialTransformer(128, 4, 32)
            ),
            nn.Sequential(
                EncoderBlock(128, 192),
                ResidualBlock(192, 192),
                SpatialTransformer(192, 4, 48)
            ),
        ])

        self.s5_block = S5Block(dim=192, state_dim=192, bidir=False, bandlimit=0.5)

        self.segmentation_decoder = nn.ModuleList([
            DecoderConv(192, 192, 2),
            DecoderConv(2 * 192, 128, 2, sampler='up'),
            DecoderConv(2 * 128, 64, 2, sampler='up'),
            DecoderConv(2 * 64, 32, 2, sampler='up'),
        ])
        self.segmentation_head = Head(2 * 32, 1)
        self.seg_proj_conv = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=1)

        self.pose_embedding = PoseEmbedding(self.n_joints, embed_dim=self.pose_embed_dim)

        self.initial_pose_head = JointRegressor(9216, num_joints=self.n_joints)
        self.delta_head = JointRegressor(9216 + self.pose_embed_dim, num_joints=self.n_joints)


    def forward(self, x, states, previous_pose, previous_pose_old, kalman_filter, first_temporal_step=False):
        
        H, W = x.shape[-2:]
        B = x.shape[0]

        feature_maps = {}
        delta_pose = None

        x_bin = x

        x = self.conv1(x_bin)

        feature_maps = []

        for encoder_block in self.encoder_backbone:
            x = encoder_block(x)
            feature_maps.append(x)

        x = feature_maps[-1]
        feature_maps = feature_maps[::-1]
        h_new, w_new = x.shape[-2:]

        if states is None:
            states = self.s5_block.s5.initial_state(
                batch_size=B * h_new * w_new
            ).to(x.device)
        else:
            states = rearrange(states, "B C H W -> (B H W) C").contiguous()

        x = rearrange(x, "B C H W -> (B H W) 1 C").contiguous()

        x, states = self.s5_block(x, states)

        states = states.detach()

        x = rearrange(
            x, "(B H W) 1 C -> B C H W", B=B, H=int(h_new), W=int(w_new)
        ).contiguous()

        last_layer_features = x

        states = rearrange(states, "(B H W) C -> B C H W", H=h_new, W=w_new).contiguous()


        for index, seg in enumerate(self.segmentation_decoder):
            x_seg = seg(x)
            x = torch.cat([x_seg, feature_maps[index]], dim=1)

        seg_f = self.segmentation_head(x) 
        
        # project the segmentation features to the same dimension as the states
        seg_features = self.seg_proj_conv(F.adaptive_avg_pool2d(x, output_size=(6, 8)))

        states = states + seg_features # element-wise addition of the segmentation features

        seg_f_detached = torch.sigmoid(seg_f.detach().clone())

        seg_out = F.interpolate(seg_f, size=(H, W), mode='bilinear', align_corners=True)

        x = last_layer_features

        x = x.view(B, -1)

        if first_temporal_step:
            current_pose = self.initial_pose_head(x)
            kalman_filter.x = current_pose.reshape(-1, 1).cpu()
            abs_pose = current_pose
            current_pose_old = current_pose
        else:
            abs_pose = self.initial_pose_head(x)
            previous_pose_emb = self.pose_embedding(previous_pose)
            x = torch.cat([x, previous_pose_emb], dim=1) # [batch_size, 9216 + 64]
            delta_pose = self.delta_head(x)
            # import pdb; pdb.set_trace()

            kalman_filter.predict(u=delta_pose.reshape(-1, 1).cpu())
            kalman_filter.update(abs_pose.reshape(-1, 1).cpu())
            # import pdb; pdb.set_trace()
            current_pose = torch.from_numpy(kalman_filter.x.reshape(16, 3)).unsqueeze(0).cuda().float()
            current_pose_old = previous_pose_old + delta_pose

        final_state = states

        return {
            'delta_pose': delta_pose,
            'seg': seg_out,
            'seg_feature': seg_f_detached,
            's5_state': final_state,
            'pose': current_pose,
            'abs_pose': abs_pose,
            'current_pose_old': current_pose_old
            # 'heatmaps': hms_out
        }, kalman_filter
