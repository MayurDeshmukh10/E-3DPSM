import torch
from torch import nn
from torch.nn import functional as F
from .base import Event3DPoseNet
from os.path import join, dirname, isfile
import numpy as np
from EventEgoPoseEstimation.utils.vis import visualize_temporal_bins
from EventEgoPoseEstimation.dataset.dataset_utils import event_augmentation, save_augmented_data, event_augmentation_v2


# class ValueLayer(nn.Module):
#     def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
#         assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
#         assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

#         nn.Module.__init__(self)
#         self.mlp = nn.ModuleList()
#         self.activation = activation

#         # create mlp
#         in_channels = 1
#         for out_channels in mlp_layers[1:]:
#             self.mlp.append(nn.Linear(in_channels, out_channels))
#             in_channels = out_channels

#         # init with trilinear kernel
#         path = join(dirname(__file__), "quantization_layer_init", f"{num_channels}_trilinear_init.pth")
#         if isfile(path):
#             state_dict = torch.load(path)
#             self.load_state_dict(state_dict)
#         else:
#             self.init_kernel(num_channels)

#     def forward(self, x):
#         # create sample of batchsize 1 and input channels 1
#         x = x[None,...,None]

#         # apply mlp convolution
#         for i in range(len(self.mlp[:-1])):
#             x = self.activation(self.mlp[i](x))

#         x = self.mlp[-1](x)
#         x = x.squeeze()

#         return x

#     def init_kernel(self, num_channels):
#         ts = torch.zeros((1, 2000))
#         optim = torch.optim.Adam(self.parameters(), lr=1e-2)

#         torch.manual_seed(1)

#         for _ in range(1000):  # converges in a reasonable time
#             optim.zero_grad()

#             ts.uniform_(-1, 1)

#             # gt
#             gt_values = self.trilinear_kernel(ts, num_channels)

#             # pred
#             values = self.forward(ts)

#             # optimize
#             loss = (values - gt_values).pow(2).sum()

#             loss.backward()
#             optim.step()
        
#         save_path = join(dirname(__file__), "quantization_layer_init", f"{num_channels}_trilinear_init.pth")
#         torch.save(self.state_dict(), save_path)


#     def trilinear_kernel(self, ts, num_channels):
#         gt_values = torch.zeros_like(ts)

#         gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
#         gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

#         gt_values[ts < -1.0 / (num_channels-1)] = 0
#         gt_values[ts > 1.0 / (num_channels-1)] = 0

#         return gt_values


# class QuantizationLayer(nn.Module):
#     def __init__(self, dim,
#                  batch_size,
#                  mlp_layers=[1, 100, 100, 1],
#                  activation=nn.LeakyReLU(negative_slope=0.1)):
#         nn.Module.__init__(self)
#         self.value_layer = ValueLayer(mlp_layers,
#                                       activation=activation,
#                                       num_channels=dim[0])
#         self.dim = dim
#         self.batch_size = batch_size

#     def forward(self, events_list, device):

#         B = len(events_list)

#         num_voxels = int(2 * np.prod(self.dim) * B)
#         # vox = events_list[0][0].new_full([num_voxels,], fill_value=0)
#         vox = torch.zeros(num_voxels, device=device)

#         C, H, W = self.dim

#         # x_values = [events[:, 0] for events in events_list]
#         # y_values = [events[:, 1] for events in events_list]
#         # t_values = [events[:, 2] for events in events_list]
#         # p_values = [events[:, 3] for events in events_list]
#         # b_values = [i * torch.ones((len(events)), device=device) for i, events in enumerate(events_list)]
        

#         # x = torch.cat(x_values, dim=0)
#         # y = torch.cat(y_values, dim=0)
#         # t = torch.cat(t_values, dim=0)
#         # p = torch.cat(p_values, dim=0)
#         # b = torch.cat(b_values, dim=0)

#         # p = (p+1)/2  # maps polarity to 0, 1

#         # x_idx = x
#         # y_idx = W * y
#         # channel_offset = W * H * C * p
#         # batch_offset = W * H * C * 2 * b

#         # # Summing to get final index
#         # idx_before_bins = x_idx + y_idx + channel_offset + batch_offset

#         x = torch.cat([events[:, 0] for events in events_list], dim=0)
#         y = torch.cat([events[:, 1] for events in events_list], dim=0)
#         t = torch.cat([events[:, 2] for events in events_list], dim=0)
#         p = torch.cat([(events[:, 3] + 1) / 2 for events in events_list], dim=0)  # Map polarity to 0, 1
#         b = torch.cat([i * torch.ones(len(events), device=device) for i, events in enumerate(events_list)], dim=0)

#         # Precompute base indices components
#         idx_before_bins = (x + W * y + (W * H * C) * p + (W * H * C * 2) * b)
#         wh = W * H  # Precompute to avoid repeated calculation

#         # Loop through bins to compute values and accumulate in voxel grid
#         for i_bin in range(C):
#             values = t * self.value_layer.forward(t - i_bin / (C - 1))
#             # values = t * self.value_layer.trilinear_kernel((t - i_bin / (C - 1)), 5)

#             # Calculate final index for this bin
#             idx = idx_before_bins + W * H * i_bin

#             # Ensure idx is within bounds
#             if (idx < 0).any() or (idx >= num_voxels).any():
#                 print(f"Out-of-bounds indices detected: min={idx.min()}, max={idx.max()}, expected range=[0, {num_voxels-1}]")
#                 idx = idx.clamp(0, num_voxels - 1)  # Clamp values within bounds

#             # Accumulate values in the voxel grid
#             # vox.put_(idx.long(), values, accumulate=True)
#             vox.scatter_add_(0, idx.long(), values)

#         vox = vox.view(-1, 2, C, H, W)
#         vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

#         return vox

# class EROS(nn.Module):
#     def  __init__(self, inp_chn, width, height, batch_size) -> None:
#         super(EROS, self).__init__()

#         self.quantization_layer = QuantizationLayer(
#             dim=(inp_chn, height, width),
#             mlp_layers=[1, 30, 30, 1],  # test and change if necessary
#             activation=nn.LeakyReLU(negative_slope=0.1),
#             batch_size=batch_size
#         )
#         self.width = width
#         self.height = height
        
#     def forward(self, events_list, device):
#         # events_list = []
#         # for i, event_batch in enumerate(events):
#         #     valid_mask = ~(event_batch == -10).all(dim=1) # remove invalid events
#         #     event_batch = event_batch[valid_mask]
#         #     events_list.append(event_batch)
        
#         quantized_events = self.quantization_layer(events_list, device)

#         # visualize_temporal_bins(quantized_events[0], '/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/test/28')

#         out = quantized_events
#         old_min, old_max, new_min, new_max = out.min(), out.max(), 0, 1
#         out = (out - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

#         return out


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

        # self.EROS = EROS(inp_chn=self.num_bins, height=self.height, width=self.width, batch_size=self.batch_size)


    def forward(self, x, augmentation_data, device='cuda'):

        s5_state = None
        pose = None

        T, B, C, H, W = x.shape
      
        delta_poses = []
        poses = []
        seg_outs = []
        heatmaps = []
        s5_states = []
        voxel_outs = []
        event_voxels = []

        # for i in range(T):

        #     # discretize raw events into voxel grid (20 x 192 x 256)
        #     out = self.EROS(x[i], device)
        #     event_voxels.append(out)

        
        # event_voxels = torch.stack(event_voxels)

        # event_voxels_copy = event_voxels.clone()

        # event_voxels = event_augmentation_v2(self, event_voxels, augmentation_data)

        # iterate over temporal steps
        # import pdb; pdb.set_trace()
        for i in range(T):

            out = x[i]
            old_min, old_max, new_min, new_max = out.min(), out.max(), 0, 1
            out = (out - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

            # out = self.EROS(x[i], device)

            # import pdb; pdb.set_trace()
            # if not all(is_empty(inner) for sublist in augmentation_data['bg_data'] for inner in sublist):
                # out = event_augmentation(self, out, i, augmentation_data)

            
            # out = event_voxels[i]
            # out_copy = event_voxels_copy[i]
            

            # for ii, o in enumerate(out):
            #     visualize_temporal_bins(o.detach(), f'/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/opt_augmented/{ii}.png')
            #     visualize_temporal_bins(out_copy[ii].detach(), f'/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/opt_augmented/og_{ii}.png')
            #     save_augmented_data(o.detach(), f'/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/opt_augmented/sum_{ii}.png')
            #     save_augmented_data(out_copy[ii].detach(), f'/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/opt_augmented/sum_og_{ii}.png')

            outs = self.event_3d_posenet(out, s5_state, pose, first_temporal_step=(i == 0))

            # outs = self.event_3d_posenet(event_voxels[i], s5_state, pose, first_temporal_step=(i == 0))

            # import pdb; pdb.set_trace()

            s5_state = outs['s5_state'].detach()
            pose = outs['pose']
            delta_pose = outs['delta_pose']
            seg_out = outs['seg']
            seg_feature = outs['seg_feature']
            heatmap = outs['heatmaps']
            if delta_pose is not None:
                delta_poses.append(delta_pose)
            poses.append(pose)
            # s5_states.append(s5_state)
            seg_outs.append(seg_out)
            heatmaps.append(heatmap)
            voxel_outs.append(out)

        abs_poses = torch.stack(poses)
        if len(delta_poses) > 0:
            delta_poses = torch.stack(delta_poses)
        seg_outs = torch.stack(seg_outs)
        heatmaps = torch.stack(heatmaps)
        voxel_representations = torch.stack(voxel_outs)
        # s5_states = torch.stack(s5_states)
        
        outputs = {}
        outputs['abs_poses'] = abs_poses  
        outputs['seg'] = seg_outs
        outputs['delta_poses'] = delta_poses
        outputs['heatmaps'] = heatmaps
        outputs['voxel_representations'] = voxel_representations
        # outputs['s5_states'] = s5_states
 
        return outputs

def is_empty(item):
    if isinstance(item, torch.Tensor):
        return item.numel() == 0  # True if the tensor has zero elements
    else:
        return len(item) == 0     # Works for lists and other sequences