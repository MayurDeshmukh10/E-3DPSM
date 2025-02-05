import torch
import torch.nn as nn

from .blazebase import BlazeBlock, DecoderConv, Head
from .s5.s5_model import S5Block
import torch.nn.functional as F
from einops import rearrange


class DeltaHead(nn.Module):
    def __init__(self, in_channels, num_joints):
        """
        Args:
            in_channels (int): Number of input channels from the feature map.
            num_joints (int): Number of joints to predict.
        """
        super(DeltaHead, self).__init__()
        self.num_joints = num_joints
        
        # A simple head using convolutional layers followed by adaptive pooling
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        
        # Optionally, add another conv layer to refine the features
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        
        # Global pooling to collapse spatial dimensions to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # A fully connected layer to map to the delta predictions per joint (3 coordinates per joint)
        self.fc = nn.Linear(64, num_joints * 3)

    def forward(self, features):
        """
        Args:
            features (Tensor): Input feature map of shape [B, in_channels, H, W]
        Returns:
            delta (Tensor): Predicted deltas of shape [B, num_joints, 3]
        """
        # Pass through convolutional layers
        x = self.conv1(features)    # [B, 64, H, W]
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)           # [B, 64, H, W]
        x = self.bn2(x)
        x = self.relu(x)
        
        # Global average pooling
        x = self.global_pool(x)     # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)     # [B, 64]
        
        # Fully connected layer to produce deltas
        x = self.fc(x)              # [B, num_joints * 3]
        
        # Reshape to [B, num_joints, 3]
        delta = x.view(x.size(0), self.num_joints, 3)
        return delta

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

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)

class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

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

        self.s5_block = S5Block(dim=192, state_dim=192, bidir=False, bandlimit=0.5)

        self.heatmap_decoder = nn.ModuleList([
            DecoderConv(192, 192, 2),
            DecoderConv(2 * 192, 128, 2, sampler='up'),
            DecoderConv(2 * 128, 64, 2, sampler='up'),
            DecoderConv(2 * 64, 32, 2, sampler='up'),
        ])

        self.segmentation_decoder = nn.ModuleList([
            DecoderConv(192, 192, 2),
            DecoderConv(2 * 192, 128, 2, sampler='up'),
            DecoderConv(2 * 128, 64, 2, sampler='up'),
            DecoderConv(2 * 64, 32, 2, sampler='up'),
        ])

        self.heatmap_head = Head(2 * 32, self.n_joints, activation='relu')
        self.segmentation_head = Head(2 * 32, 1)
                
        self.delta_head = DeltaHead(in_channels=192, num_joints=self.n_joints)

        self.joint_regressor_hms = JointRegressor(16, self.n_joints)

        # self.upsample_blocks = nn.ModuleList([
        #     UpSample(64),
        #     UpSample(128),
        #     UpSample(192)
        # ])

    def forward(self, x, states, initial_pose):
        H, W = x.shape[-2:]
        B = x.shape[0]
        sequence_length = 1

        # Input shape: [B, C, H, W] where C = 2 * num_bins
        outputs = []

        pose = initial_pose.cuda()

        feature_maps = {}
        states_store = {}


        abs_poses = []
        delta_poses = []
        seg_f_detached_list = []
        seg_out_list = []

        # if states is None:
        #     initial_states = []
        #     for s5_block in self.s5_blocks:
        #         # s5_block.s5.initial_state
        #         # states = s5_block.s5.initial_state(
        #         #     batch_size=B * 48 * 64
        #         # ).to(x.device)

        #         initial_states.append(-10)
        #     # import pdb; pdb.set_trace()
        #     initial_states[-1] = 10 # some random value
        #     states = initial_states
        
        input = x

        for i in range(self.num_bins):
            
            # x_bin = input[:, i*2:(i+1)*2, :, :]
            x_bin = input[:, [i, i + self.num_bins], :, :]

            x = self.conv1(x_bin)

            feature_maps = []

            internal_states = []
            for blaze_block in self.backbone1:
                x = blaze_block(x)
                feature_maps.append(x)
                # feature_maps[i].append(x)
                # feature_maps.append(x)
                # internal_states.append(current_block_state)

            
            x = feature_maps[-1]
            feature_maps = feature_maps[::-1]
            h_new, w_new = x.shape[-2:]
            if states is None:
                states = self.s5_block.s5.initial_state(
                    batch_size=B * h_new * w_new
                ).to(x.device)
            else:
                states = rearrange(states, "B C H W -> (B H W) C")

            x = rearrange(x, "B C H W -> (B H W) 1 C")


            x, states = self.s5_block(x, states)

            states = states.detach()

            x = rearrange(
                x, "(B H W) 1 C -> B C H W", B=B, H=int(h_new), W=int(w_new)
            )
            last_layer_features = x

            states = rearrange(states, "(B H W) C -> B C H W", H=h_new, W=w_new)

            for i, seg in enumerate(self.segmentation_decoder):
                # import pdb; pdb.set_trace()

                x_seg = seg(x)
                x = torch.cat([x_seg, feature_maps[i]], dim=1)

            seg_f = self.segmentation_head(x) 
        
            seg_f_detached = torch.sigmoid(seg_f.detach().clone())
            seg_f_detached_list.append(seg_f_detached)

            seg_out = F.interpolate(seg_f, size=(H, W), mode='bilinear', align_corners=True)
            seg_out_list.append(seg_out)

            delta_pose = self.delta_head(last_layer_features)

            pose = pose + delta_pose

            abs_poses.append(pose)
            delta_poses.append(delta_pose)

        
        final_state = states
        
        abs_poses = torch.stack(abs_poses, dim=1)
        delta_poses = torch.stack(delta_poses, dim=1)
        seg_out = torch.stack(seg_out_list, dim=1)

        return {
            'j3d': abs_poses,
            'delta_j3d': delta_poses,
            'seg': seg_out,
            'seg_feature': seg_f_detached,
            'prev_states': final_state
            # 'states_store': states_store
        }

        

# Example usage
if __name__ == "__main__":
    # Replace this with your S5 implementation
    class S5Mock(nn.Module):
        def __init__(self, channels):
            super(S5Mock, self).__init__()
            self.fc = nn.Conv2d(channels, channels, kernel_size=1)

        def forward(self, x):
            return self.fc(x)

    input_channels = 2
    num_bins = 9
    model = Event3DPoseNet(input_channels, num_bins, S5Mock)

    # Input voxel grid: Batch size 4, Height 128, Width 128
    voxel_grid = torch.randn(4, 2 * num_bins, 128, 128)
    output = model(voxel_grid)
    print("Output shape:", output.shape)
