import torch
import torch.nn as nn

from .blazebase import BlazeBlock, DecoderConv, Head
from .s5.s5_model import S5Block
import torch.nn.functional as F
from einops import rearrange
from .encoder import EncoderBlock, ResidualBlock, SpatialTransformer
from .decoder import JointQueryDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

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
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, embed_dim)
        )
        
    def forward(self, pose):
        # pose: [B, num_joints, 3]
        B, J, _ = pose.shape
        x = pose.view(B * J, 3)  # Flatten to shape [B, num_joints * 3]
        embedding = self.fc(x)  # [B, embed_dim]
        embedding = embedding.view(B, J, -1)
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
                SpatialTransformer(192, 6, 32)
            ),
        ])

        # self.s5_block = S5Block(dim=192, state_dim=96, bidir=True, bandlimit=None)

        self.s5_blocks = nn.ModuleList([
            S5Block(dim=32, state_dim=16, bidir=True, bandlimit=0.5),
            # S5Block(dim=64, state_dim=32, bidir=True, bandlimit=0.5),
            # S5Block(dim=128, state_dim=64, bidir=True, bandlimit=0.5),
            S5Block(dim=192, state_dim=96, bidir=True, bandlimit=0.5)
        ])

        # self.decoder = nn.ModuleList([
        #     DecoderConv(192, 192, 2),
        #     DecoderConv(2 * 192, 128, 2, sampler='up'),
        #     DecoderConv(2 * 128, 64, 2, sampler='up'),
        #     DecoderConv(2 * 64, 32, 2, sampler='up'),
        # ])

        self.query_decoder = JointQueryDecoder(192, self.n_joints)


        # self.segmentation_head = Head(2 * 32, 1)
        # self.seg_proj_conv = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=1)

        self.pose_embedding = PoseEmbedding(self.n_joints, embed_dim=self.pose_embed_dim)

        # self.initial_pose_head = JointRegressor(9216, num_joints=self.n_joints)
        # self.delta_head = JointRegressor(9216 + self.pose_embed_dim, num_joints=self.n_joints)
        self.initial_pose_head = nn.Linear(192, 3)
        self.delta_head = nn.Linear(256, 3)

    def forward(self, x, prev_5_states, previous_pose, previous_pose_old, kalman_filter, first_temporal_step=False):
        
        H, W = x.shape[-2:]
        B = x.shape[1]
        T = x.shape[0]

        # Pre-allocate feature maps list with known length and process S5 states more efficiently
        feature_maps = [None] * len(self.encoder_backbone)
        s5_states = {}
        delta_pose = None

        # Single rearrangement at the beginning
        x = rearrange(x, "L B C H W -> (L B) C H W").contiguous()
        x = self.conv1(x)

        # Process each encoder stage
        for stage, encoder_block in enumerate(self.encoder_backbone):
            # Apply encoder block
            x = encoder_block(x)
            new_h, new_w = x.shape[-2:]
            
            # Apply S5 blocks only for stages 0 and 3 as in the original code
            if stage in [0, 3]:

                s5_block = self.s5_blocks[stage % 2]
                # Prepare S5 state - avoiding conditionals within the loop if possible
                s5_state = None
                if prev_5_states[stage] is not None:
                    s5_state = rearrange(prev_5_states[stage], "B C H W -> (B H W) C").contiguous()
                else:
                    s5_state = s5_block.s5.initial_state(batch_size=B * new_h * new_w).to(x.device)
                
                # Reshape for S5 processing - do this in one step with precise dimensions
                x_s5 = x.view(T, B, x.size(1), new_h, new_w)
                x_s5 = x_s5.permute(1, 3, 4, 0, 2).reshape(B * new_h * new_w, T, -1)
                
                # Process with S5 block
                x_s5, new_s5_state = s5_block(x_s5, s5_state)
                
                # Reshape back efficiently
                x = x_s5.view(B, new_h, new_w, T, -1).permute(3, 0, 4, 1, 2).reshape(T * B, -1, new_h, new_w)
                
                # Store new S5 state
                s5_states[stage] = new_s5_state.view(B, new_h, new_w, -1).permute(0, 3, 1, 2)
            
            # Store feature maps directly in their final positions (no need to reverse later)
            # import pdb; pdb.set_trace()
            feature_maps[len(feature_maps) - 1 - stage] = x
        
        # Last feature map is now the first element
        x = feature_maps[0]
        
        x = x.view(T, B, 192, 6, 8)

        delta_poses = list()
        current_poses = list()
        current_poses_old = list()
        abs_poses = list()


        for i in range(T):
            curr_x = x[i].flatten(2).permute(2, 0, 1)
            decoded_x = self.query_decoder(curr_x)
            if i == 0:
                current_pose = self.initial_pose_head(decoded_x)
                if not kalman_filter.initialized:
                    kalman_filter.initialize(current_pose.reshape(B, -1))
                previous_pose = current_pose
                abs_pose = current_pose
                current_pose_old = current_pose
                previous_pose_old = current_pose_old
            else:
                abs_pose = self.initial_pose_head(decoded_x)
                previous_pose_emb = self.pose_embedding(previous_pose)
                x_feat = torch.cat([decoded_x, previous_pose_emb], dim=2) # [batch_size, 16, 256]
                delta_pose = self.delta_head(x_feat)

                kalman_filter.predict(u=delta_pose.reshape(B, -1))
                kalman_filter.update(abs_pose.reshape(B, -1))

                current_pose = kalman_filter.x.reshape(B, 16, 3)
                
                previous_pose = current_pose
                current_pose_old = previous_pose_old + delta_pose
                delta_poses.append(delta_pose)

            current_poses.append(current_pose)
            current_poses_old.append(current_pose_old)
            abs_poses.append(abs_pose)


        delta_poses = torch.stack(delta_poses, dim=0)
        current_poses = torch.stack(current_poses, dim=0)
        current_poses_old = torch.stack(current_poses_old, dim=0)
        abs_poses = torch.stack(abs_poses, dim=0)

        return {
            'delta_pose': delta_poses,
            's5_state': s5_states,
            'pose': current_poses,
            'abs_pose': abs_poses,
            'current_pose_old': current_poses_old
        }
