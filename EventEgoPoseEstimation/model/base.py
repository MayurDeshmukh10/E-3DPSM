import torch
import torch.nn as nn

from .blazebase import BlazeBlock, DecoderConv, Head
from .s5.s5_model import S5Block
import torch.nn.functional as F
from einops import rearrange


# class DeltaHead(nn.Module):
#     def __init__(self, in_channels, num_joints, pose_embed_dim=64):
#         """
#         Args:
#             in_channels (int): Number of input channels from the feature map.
#             num_joints (int): Number of joints to predict.
#         """
#         super(DeltaHead, self).__init__()
#         self.num_joints = num_joints
        
#         # A simple head using convolutional layers followed by adaptive pooling
#         self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
#         self.bn1   = nn.BatchNorm2d(64)
#         self.relu  = nn.ReLU(inplace=True)
        
#         # Optionally, add another conv layer to refine the features
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn2   = nn.BatchNorm2d(64)
        
#         # Global pooling to collapse spatial dimensions to 1x1
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

#         self.pose_embedding = PoseEmbedding(num_joints, embed_dim=pose_embed_dim)
        
#         # A fully connected layer to map to the delta predictions per joint (3 coordinates per joint)
#         self.fc = nn.Linear(128, num_joints * 3)

#     def forward(self, features, current_pose):
#         # TODO: 
#         # Pass through convolutional layers
#         x = self.conv1(features)    # [B, 64, H, W]
#         x = self.bn1(x)
#         x = self.relu(x)
        
#         x = self.conv2(x)           # [B, 64, H, W]
#         x = self.bn2(x)
#         x = self.relu(x)
        
#         # TODO: no pooling. just flatten it
#         # Global average pooling
#         # x = self.global_pool(x)     # [B, 64, 1, 1]
#         # x = x.view(x.size(0), -1)     # [B, 64]
#         x = features.view(x.shape[0], -1)  # [B, C*H*W]

#         pose_emb = self.pose_embedding(current_pose)  # [B, pose_embed_dim]

#         combined = torch.cat([x, pose_emb], dim=1)  # [B, 64 + pose_embed_dim]

#         # Fully connected layer to produce deltas
#         x = self.fc(combined)              # [B, num_joints * 3]
        
#         # Reshape to [B, num_joints, 3]
#         delta = x.view(x.size(0), self.num_joints, 3)
#         return delta


import torch
import torch.nn as nn
import torch.nn.functional as F

class UpBlock(nn.Module):
    """
    Upsampling block that first upsamples the input,
    then concatenates the corresponding skip connection,
    and applies two convolutional layers.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()
        # Upsample the low-resolution feature map using transposed convolution
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Convolutional block after concatenating with the skip connection
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        # Ensure the spatial dimensions match before concatenation
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class Pose3DDecoder(nn.Module):
    """
    A U-Net–style decoder for 3D human pose estimation.
    
    Expects:
      - x: the bottleneck feature (e.g., [batch, 192, 6, 8])
      - feature_maps: list of skip connection features from the encoder,
          ordered as [third_encoder, second_encoder, first_encoder], with shapes:
              [batch, 128, 12, 16],
              [batch, 64, 24, 32],
              [batch, 32, 48, 64]
              
    Produces:
      - joints: 3D joint coordinates in shape [batch, num_joints, 3]
    """
    def __init__(self, num_joints=16):
        super(Pose3DDecoder, self).__init__()
        # Define the upsampling blocks
        self.up_block3 = UpBlock(in_channels=192, skip_channels=128, out_channels=128)
        self.up_block2 = UpBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.up_block1 = UpBlock(in_channels=64, skip_channels=32, out_channels=32)
        
        # Instead of global pooling, we flatten the spatial dimensions and use an MLP.
        # The final feature map is [batch, 32, 48, 64] so the flattened size is:
        flattened_size = 32 * 48 * 64  # 98304 features
        
        # MLP layers to reduce dimensionality to (num_joints * 3)
        self.mlp = nn.Sequential(
            nn.Linear(flattened_size, 8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_joints * 3)
        )
        
    def forward(self, x, feature_maps):
        # feature_maps is a list: [skip3, skip2, skip1]
        x = self.up_block3(x, feature_maps[0])  # [batch, 128, 12, 16]
        x = self.up_block2(x, feature_maps[1])  # [batch, 64, 24, 32]
        x = self.up_block1(x, feature_maps[2])  # [batch, 32, 48, 64]
        
        # Flatten the spatial dimensions without pooling so spatial details are preserved.
        x = x.view(x.size(0), -1)
        joints = self.mlp(x)
        joints = joints.view(x.size(0), -1, 3)  # reshape to [batch, num_joints, 3]
        return joints



class InitialPoseHead(nn.Module):
    def __init__(self, in_channels, num_joints):
        super(InitialPoseHead, self).__init__()
        self.num_joints = num_joints
        

        self.fc1 = nn.Linear(in_channels, 1024)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(512, num_joints)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = x.view(x.size(0), self.num_joints, 3)
        return x


class DeltaHead(nn.Module):
    def __init__(self, in_channels, num_joints, pose_embed_dim=64, hidden_dim=512):
        super(DeltaHead, self).__init__()
        self.num_joints = num_joints

        self.spatial_reducer = nn.Sequential(
            
            nn.Conv2d(in_channels, 128, 3, stride=2, padding=1), # [B,64,48,64] → [B,128,24,32]
            nn.GroupNorm(8, 128),
            nn.GELU(),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # [B,128,24,32] → [B,256,12,16]
            nn.GroupNorm(16, 256),
            nn.GELU(),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1), # [B,256,12,16] → [B,512,6,8]
            nn.GroupNorm(32, 512),
            nn.GELU(),
        )

        self.pose_embedding = PoseEmbedding(num_joints, embed_dim=pose_embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*1 + pose_embed_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            
            nn.Linear(512, num_joints*3)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, features, current_pose):

        x = self.spatial_reducer(features)

        pose_emb = self.pose_embedding(current_pose)  # [B, 64]

        # x = torch.global_avg_pool2d(x, keepdim=True)  # [B, 512, 1, 1]

        x = self.global_pool(x)  # [B, 512, 1, 1]

        x = x.view(x.size(0), -1) # [B, 512*6*8]

        combined = torch.cat([x, pose_emb], dim=1)  # [B, 512*6*8 + 64]

        # Fully connected layer to produce deltas
        x = self.mlp(combined) # [B, num_joints * 3]
        
        # Reshape to [B, num_joints, 3]
        delta = x.view(x.size(0), self.num_joints, 3)

        return delta

class InitialPosePredictor(nn.Module):
    def __init__(self, in_channels, num_joints, pose_embed_dim=64):
        super(InitialPosePredictor, self).__init__()
        self.num_joints = num_joints
        
        # Convolutional feature extraction.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling to reduce spatial dimensions.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers mapping pooled features to absolute joint coordinates.
        self.fc = nn.Sequential(
            nn.Linear(32, pose_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pose_embed_dim, num_joints * 3)
        )


    def forward(self, features):

        x = self.conv(features)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # [B, 32]
        x = self.fc(x)             # [B, num_joints * 3]
        initial_pose = x.view(x.size(0), self.num_joints, 3)

        return initial_pose
    

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

class JointRegressor(nn.Module):
    def __init__(self, in_channels, n_joints):
        super(JointRegressor, self).__init__()

        self.j3d_regressor = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(True),

            nn.Linear(512, n_joints * 3),
        )
        self.n_joints = n_joints
        
    def forward(self, x):
        x = self.j3d_regressor(x)
        x = x.view(-1, self.n_joints, 3)
        
        return x

class Event3DPoseNet(nn.Module):
    def __init__(self, 
        num_bins,
        input_channels,
        num_joints
    ):
        super(Event3DPoseNet, self).__init__()
        
        self.num_bins = num_bins
        # self.conv_initial = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        # self.s5_blocks = nn.ModuleList([s5_block(32) for _ in range(num_bins)])
        # self.res_blocks = nn.ModuleList([ResBlock(32) for _ in range(4)])
        # self.upsample_blocks = nn.ModuleList([UpSample(32 // (2**i)) for i in range(4)])
        self.inp_chn = input_channels
        self.n_joints = num_joints
        self.pose_embed_dim = 64
        self._define_layers()

    
    def _define_layers(self):

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inp_chn, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.backbone1 = nn.ModuleList([
            BlazeBlock(16, 32, 3),
            BlazeBlock(32, 64, 4),
            BlazeBlock(64, 128, 5),
            BlazeBlock(128, 192, 6),
        ])

        # self.s5_blocks = nn.ModuleList([
        #     # S5Block(dim=32, state_dim=32, bidir=False, bandlimit=0.5),
        #     # S5Block(dim=64, state_dim=64, bidir=False, bandlimit=0.5),
        #     # S5Block(dim=128, state_dim=128, bidir=False, bandlimit=0.5),
        #     S5Block(dim=192, state_dim=192, bidir=False, bandlimit=0.5),
        # ])

        # self.initial_pose_predictor = InitialPosePredictor(192, self.n_joints, self.pose_embed_dim)

        self.s5_block = S5Block(dim=192, state_dim=192, bidir=False, bandlimit=0.5)

        # self.heatmap_decoder = nn.ModuleList([
        #     DecoderConv(192, 192, 2),
        #     DecoderConv(2 * 192, 128, 2, sampler='up'),
        #     DecoderConv(2 * 128, 64, 2, sampler='up'),
        #     DecoderConv(2 * 64, 32, 2, sampler='up'),
        # ])

        self.segmentation_decoder = nn.ModuleList([
            DecoderConv(192, 192, 2),
            DecoderConv(2 * 192, 128, 2, sampler='up'),
            DecoderConv(2 * 128, 64, 2, sampler='up'),
            DecoderConv(2 * 64, 32, 2, sampler='up'),
        ])

        # self.initial_pose_decoder = nn.ModuleList([
        #     DecoderConv(192, 192, 2),
        #     DecoderConv(2 * 192, 128, 2, sampler='up'),
        #     DecoderConv(2 * 128, 64, 2, sampler='up'),
        #     DecoderConv(2 * 64, 32, 2, sampler='up'),
        # ])

        self.delta_decoder = nn.ModuleList([
            DecoderConv(192, 192, 2),
            DecoderConv(2 * 192, 128, 2, sampler='up'),
            DecoderConv(2 * 128, 64, 2, sampler='up'),
            DecoderConv(2 * 64, 32, 2, sampler='up'),
        ])

        # self.initial_pose_head = Head(2 * 32, self.n_joints, activation='relu')

        self.segmentation_head = Head(2 * 32, 1)

        self.delta_head = DeltaHead(2 * 32, num_joints=self.n_joints, pose_embed_dim=self.pose_embed_dim)

        self.intital_pose_decoder = Pose3DDecoder(num_joints=self.n_joints)

        # self.heatmap_head = Head(2 * 32, self.n_joints, activation='relu')

        # self.joint_regressor_hms = JointRegressor(64, self.n_joints)

        # self.initial_pose_head = InitialPoseHead(9216, self.n_joints)

        # self.delta_head = DeltaHead(in_channels=192, num_joints=self.n_joints, pose_embed_dim=self.pose_embed_dim)

        self.seg_proj_conv = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=1)


    def forward(self, x, states, previous_pose, first_temporal_step=False):
        
        H, W = x.shape[-2:]
        B = x.shape[0]

        feature_maps = {}
        delta_pose = None

        x_bin = x

        x = self.conv1(x_bin)

        feature_maps = []

        for blaze_block in self.backbone1:
            x = blaze_block(x)
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
        # seg_f_detached_list.append(seg_f_detached)

        seg_out = F.interpolate(seg_f, size=(H, W), mode='bilinear', align_corners=True)

        x = last_layer_features
        hms_outs = []

        # for index, pose_decoder in enumerate(self.initial_pose_decoder):
        #     x = pose_decoder(x)
        #     x = torch.cat([x, feature_maps[index]], dim=1)

            # if index >= 1:
            #     inter_hms_out = F.interpolate(x[:, :self.n_joints], size=(H // 4, W // 4), mode='bilinear', align_corners=True)
            #     hms_outs.append(inter_hms_out)

        # import pdb; pdb.set_trace()
        # hms_out = self.heatmap_head(x)
    
        # for i in range(len(hms_outs)):
        #     hms_out = hms_out + hms_outs[i] 
        # hms_out = hms_out / (len(hms_outs) + 1)

        if first_temporal_step:
            # current_pose = self.initial_pose_head(x.view(x.shape[0], -1))
            pose_docoder_feature_maps = [feature_maps[1], feature_maps[2], feature_maps[3]]
            current_pose = self.intital_pose_decoder(last_layer_features, pose_docoder_feature_maps)
            # current_pose = self.joint_regressor_hms(hms_out)
        else:
            x = last_layer_features
            for index, delta_decoder in enumerate(self.delta_decoder):
                x = delta_decoder(x)
                x = torch.cat([x, feature_maps[index]], dim=1)
            
            # import pdb; pdb.set_trace()
            delta_pose = self.delta_head(x, previous_pose)
            # delta_pose = self.delta_head(last_layer_features, previous_pose)
            current_pose = previous_pose + delta_pose

        final_state = states

        return {
            'delta_pose': delta_pose,
            'seg': seg_out,
            'seg_feature': seg_f_detached,
            's5_state': final_state,
            'pose': current_pose,
            # 'heatmaps': hms_out
        }
