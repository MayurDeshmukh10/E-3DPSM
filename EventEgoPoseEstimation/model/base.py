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


    def forward(self, x, prev_5_states, previous_pose, previous_pose_old, kalman_filter, first_temporal_step=False):
        
        H, W = x.shape[-2:]
        B = x.shape[1]
        T = x.shape[0]

        feature_maps = list()
        s5_states = dict()
        delta_pose = None

        # times = []
        # frq = []

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        # start_time = time.perf_counter()

        x = rearrange(x, "L B C H W -> (L B) C H W").contiguous()  # where B' = (L B) is the new batch size

        x = self.conv1(x)

        for stage, encoder_block in enumerate(self.encoder_backbone):
            x = encoder_block(x)
            feature_maps.append(x)

        x = feature_maps[-1]
        feature_maps = feature_maps[::-1]
        h_new, w_new = x.shape[-2:]

        if prev_5_states is None:
            states = self.s5_block.s5.initial_state(
                batch_size=B * h_new * w_new
            ).to(x.device)
        else:
            states = rearrange(prev_5_states, "B C H W -> (B H W) C").contiguous()

        # x = rearrange(x, "B C H W -> (B H W) 1 C").contiguous()
        x = rearrange(x, "(L B) C H W -> (B H W) L C", L=T).contiguous()
        
        x, states = self.s5_block(x, states)

        x = rearrange(
                x, "(B H W) L C -> (L B) C H W", B=B, H=int(h_new), W=int(w_new)
        ).contiguous()


        last_layer_features = x

        states = rearrange(states, "(B H W) C -> B C H W", H=h_new, W=w_new).contiguous()


        for index, seg in enumerate(self.segmentation_decoder):
            x_seg = seg(x)
            x = torch.cat([x_seg, feature_maps[index]], dim=1)

        seg_f = self.segmentation_head(x) 
        
        # project the segmentation features to the same dimension as the states
        seg_features = self.seg_proj_conv(F.adaptive_avg_pool2d(x, output_size=(6, 8)))

        # states = states + seg_features # element-wise addition of the segmentation features

        seg_f_detached = torch.sigmoid(seg_f.detach().clone())

        seg_out = F.interpolate(seg_f, size=(H, W), mode='bilinear', align_corners=True)

        x = last_layer_features

        x = x + seg_features # element-wise addition of the segmentation features

        x = x.view(T, B, -1)

        delta_poses = list()
        current_poses = list()
        current_poses_old = list()
        abs_poses = list()


        for i in range(T):
            if i == 0:
                current_pose = self.initial_pose_head(x[i])
                if not kalman_filter.initialized:
                    kalman_filter.initialize(current_pose.reshape(B, -1))
                previous_pose = current_pose
                abs_pose = current_pose
                current_pose_old = current_pose
                previous_pose_old = current_pose_old
            else:
                abs_pose = self.initial_pose_head(x[i])
                previous_pose_emb = self.pose_embedding(previous_pose)
                x_feat = torch.cat([x[i], previous_pose_emb], dim=1) # [batch_size, 9216 + 64]
                delta_pose = self.delta_head(x_feat)

                kalman_filter.predict(u=delta_pose.reshape(B, -1))
                kalman_filter.update(abs_pose.reshape(B, -1))

                current_pose = kalman_filter.x.reshape(B, 16, 3)
                
                previous_pose = current_pose
                current_pose_old = previous_pose_old + delta_pose
                delta_poses.append(delta_pose)

            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            # end_time = time.perf_counter()

            # elapsed_time = end_time - start_time



            # frequency = 1 / elapsed_time if elapsed_time > 0 else float('inf')

            # times.append(elapsed_time)
            # frq.append(frequency)

            current_poses.append(current_pose)
            current_poses_old.append(current_pose_old)
            abs_poses.append(abs_pose)


        seg_out = seg_out.view(T, B, 1, H, W)

        delta_poses = torch.stack(delta_poses, dim=0)
        current_poses = torch.stack(current_poses, dim=0)
        current_poses_old = torch.stack(current_poses_old, dim=0)
        abs_poses = torch.stack(abs_poses, dim=0)

        # mean_time = sum(times) / len(times)
        # mean_freq = sum(frq) / len(frq)

        # print(f"Total inference time for one iteration: {mean_time * 1000:.3f} ms")
        # print(f"Update frequency: {mean_freq:.3f} Hz")

        states = states.detach()

        return {
            'delta_pose': delta_poses,
            'seg': seg_out,
            'seg_feature': seg_f_detached,
            's5_state': states,
            'pose': current_poses,
            'abs_pose': abs_poses,
            'current_pose_old': current_poses_old
        }
