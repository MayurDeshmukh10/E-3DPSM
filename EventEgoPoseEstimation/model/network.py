import torch
from torch import nn
from torch.nn import functional as F
from .blazepose import BlazePose
from .base import Event3DPoseNet
from os.path import join, dirname, isfile
import numpy as np
from EventEgoPoseEstimation.utils.vis import visualize_temporal_bins


class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", f"{num_channels}_trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in range(1000):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()
        
        save_path = join(dirname(__file__), "quantization_layer_init", f"{num_channels}_trilinear_init.pth")
        torch.save(self.state_dict(), save_path)


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values


class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 batch_size,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0])
        self.dim = dim
        self.batch_size = batch_size

    def forward(self, events_list, device):

        B = len(events_list)

        num_voxels = int(2 * np.prod(self.dim) * B)
        vox = events_list[0][0].new_full([num_voxels,], fill_value=0)

        C, H, W = self.dim

        x_values = [events[:, 0] for events in events_list]
        y_values = [events[:, 1] for events in events_list]
        t_values = [events[:, 2] for events in events_list]
        p_values = [events[:, 3] for events in events_list]
        b_values = [i * torch.ones((len(events)), device=device) for i, events in enumerate(events_list)]
        

        x = torch.cat(x_values, dim=0)
        y = torch.cat(y_values, dim=0)
        t = torch.cat(t_values, dim=0)
        p = torch.cat(p_values, dim=0)
        b = torch.cat(b_values, dim=0)

        p = (p+1)/2  # maps polarity to 0, 1

        x_idx = x
        y_idx = W * y
        
        # x_idx = W * x             # Maps height (x) to row offsets
        # y_idx = y                 # Maps width (y)
        channel_offset = W * H * C * p
        batch_offset = W * H * C * 2 * b

        # Summing to get final index
        idx_before_bins = x_idx + y_idx + channel_offset + batch_offset

        # Loop through bins to compute values and accumulate in voxel grid
        for i_bin in range(C):
            values = t * self.value_layer.forward(t - i_bin / (C - 1))
            # values = t * self.value_layer.trilinear_kernel((t - i_bin / (C - 1)), 5)

            # Calculate final index for this bin
            idx = idx_before_bins + W * H * i_bin

            # Ensure idx is within bounds
            if (idx < 0).any() or (idx >= num_voxels).any():
                print(f"Out-of-bounds indices detected: min={idx.min()}, max={idx.max()}, expected range=[0, {num_voxels-1}]")
                idx = idx.clamp(0, num_voxels - 1)  # Clamp values within bounds

            # Accumulate values in the voxel grid
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

        return vox

class ConfidenceNetwork(nn.Module):
    def __init__(self):
        super(ConfidenceNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=3//2),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=3//2),
            nn.PReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=3//2),
            nn.PReLU(),
            nn.Conv2d(64, 1, 1, stride=1, padding=0, bias=False), 
        )

    def forward(self, key):
        confidence = self.network(key)
       
        return torch.sigmoid(key.detach() * confidence)


def crop_and_resize_to_resolution(x, output_resolution=(224, 224)):
        B, C, W, H = x.shape
        if H > W:
            h = H // 2
            x = x[:, :, h - W // 2:h + W // 2, :]
        else:
            h = W // 2
            x = x[:, :, :, h - H // 2:h + H // 2]

        x = F.interpolate(x, size=output_resolution)

        return x

class EROS(nn.Module):
    def  __init__(self, inp_chn, width, height, batch_size) -> None:
        super(EROS, self).__init__()

        self.quantization_layer = QuantizationLayer(
            dim=(inp_chn, height, width),
            mlp_layers=[1, 30, 30, 1],  # test and change if necessary
            activation=nn.LeakyReLU(negative_slope=0.1),
            batch_size=batch_size
        )
        
        self.confidence_network = ConfidenceNetwork()
        self.width = width
        self.height = height
        
    def forward(self, buffer, events, key, device):
        events_list = []
        for i, event_batch in enumerate(events):
            valid_mask = ~(event_batch == -10).all(dim=1) # remove invalid events
            event_batch = event_batch[valid_mask]
            events_list.append(event_batch)
        
        quantized_events = self.quantization_layer(events_list, device)

        # visualize_temporal_bins(quantized_events[0], '/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/test/28')

        # quantized_events = crop_and_resize_to_resolution(quantized_events, (self.height, self.width))
        _, _, height, width = quantized_events.shape
            
        # confidence = self.confidence_network(key) 

        # confidence = F.interpolate(confidence, size=(height, width), mode='bilinear', align_corners=False)
        # confidence = F.interpolate(confidence, size=(inp.shape[2], inp.shape[3]), mode='bilinear', align_corners=False)

        # try:
        # TODO: Visualize this after applying the confidence
        # TODO: Does it make sense to apply confidence of (1, 192, 256) to all bins (18, 192, 256)
        # out = buffer * confidence + quantized_events
        out = quantized_events


        old_min, old_max, new_min, new_max = out.min(), out.max(), 0, 1
        out = (out - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

        # return out, confidence, buffer, quantized_events
        return out, buffer, quantized_events.clone().detach()


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
        # self.hm_width, self.hm_height = config.heatmap_size
        self.hm_width, self.hm_height = config.MODEL.HEATMAP_SIZE

        self.batch_size = batch_size

        self.EROS = EROS(inp_chn=self.num_bins, height=self.height, width=self.width, batch_size=self.batch_size)
        
    def forward(self, x, prev_buffer=None, prev_key=None, batch_first=False, device='cuda'):

        buffer = prev_buffer

        prev_states = None

        T, B, N, C = x.shape
        
        if buffer is None:
            # buffer = torch.zeros_like(x[0, :, :, :, :])
            buffer = torch.zeros(self.batch_size, self.inp_chn, self.height, self.width, device=device)
        
        key = prev_key

        if key is None:
            key = torch.ones(self.batch_size, 1, self.hm_height, self.hm_width, device=device)

                    
        eross = []
        delta_j3ds = []
        j3ds = []
        seg_outs = []

        # confidences = []
        buffers = []
        # states_store = []

        for i in range(T):
            if self.enable_eros:
                out, buffer, representation = self.EROS(buffer, x[i], key, device)
            else:
                out = x[i]

            buffers.append(buffer)
            # confidences.append(confidence)
     
            outs = self.event_3d_posenet(out, prev_states)
            
            prev_states = outs['prev_states']

            # states_store.append(outs['states_store'])

            # buffer = out

            # x_hms = outs['hms']
            j3d = outs['j3d']

            delta_j3d = outs['delta_j3d']
                                    
            seg_out = outs['seg']
            seg_feature = outs['seg_feature']
            
            key = seg_feature
            
            eross.append(buffer)
            delta_j3ds.append(delta_j3d)
            j3ds.append(j3d)
            seg_outs.append(seg_out)

        if prev_buffer is not None:
            prev_buffer.copy_(buffer)
            
        if prev_key is not None:
            prev_key.copy_(key)

        eross = torch.cat(eross, dim=0)
        j3ds = torch.cat(j3ds, dim=0)
        seg_outs = torch.cat(seg_outs, dim=0)
         
        # confidences = torch.cat(confidences, dim=0)
        buffers = torch.cat(buffers, dim=0)     
        delta_j3ds = torch.cat(delta_j3ds, dim=0)  
        
        outputs = {}
        outputs['j3d'] = j3ds  
        
        outputs['eros'] = eross
        outputs['seg'] = seg_outs
 
        # outputs['confidence'] = confidences
        outputs['buffer'] = buffers
        outputs['representation'] = representation
        outputs['prev_states'] = prev_states
        outputs['delta_j3d'] = delta_j3ds
        # outputs['states_store'] = states_store
 
        return outputs

