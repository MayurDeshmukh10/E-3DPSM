import cv2
import numpy as np
import torch

from typing import Any


def add_salt_and_pepper(image, low_th, high_th):
    saltpepper_noise = np.zeros_like(image)
    cv2.randu(saltpepper_noise, 0, 255)

    image[saltpepper_noise < low_th] = 0
    image[saltpepper_noise > high_th] = 255


class Rectangle:
    def __init__(self, x_tl, y_tl, width, height):
        self.x_tl = x_tl
        self.y_tl = y_tl
        self.width = width
        self.height = height

    def intersect(self, rect):
        x_tl = max(self.x_tl, rect.x_tl)
        y_tl = max(self.y_tl, rect.y_tl)
        x_br = min(self.x_tl + self.width, rect.x_tl + rect.width)
        y_br = min(self.y_tl + self.height, rect.y_tl + rect.height)
        if x_tl < x_br and y_tl < y_br:
            return Rectangle(x_tl, y_tl, x_br - x_tl, y_br - y_tl)
        return None

    __and__ = intersect

    def equal(self, rect):
        x_tl_diff = self.x_tl - rect.x_tl
        y_tl_diff = self.y_tl - rect.y_tl
        x_br_diff = self.width - rect.width
        y_br_diff = self.height - rect.height
        diff = x_tl_diff + y_tl_diff + x_br_diff + y_br_diff
        if diff == 0:
            return True
        return False

    __eq__ = equal


class EROS:

    def __init__(self, kernel_size, height, width, decay_base=0.3):
        self.kernel_size = kernel_size
        self.frame_height = height
        self.frame_width = width
        self.decay_base = decay_base
        self._image = np.zeros((height, width), dtype=np.uint8)

        if self.kernel_size % 2 != 0:
            self.kernel_size += 1

    def get_frame(self):
        return self._image

    def update(self, vx, vy):
        odecay = self.decay_base ** (1.0 / self.kernel_size)
        half_kernel = int(self.kernel_size / 2)
        roi_full = Rectangle(0, 0, self.frame_width, self.frame_height)
        roi_raw = Rectangle(0, 0, self.kernel_size, self.kernel_size)

        roi_raw.x_tl = vx - half_kernel
        roi_raw.y_tl = vy - half_kernel
        roi_valid = roi_raw & roi_full

        if roi_valid is None:
            return True
        roi = [roi_valid.y_tl, roi_valid.y_tl + roi_valid.height, roi_valid.x_tl, roi_valid.x_tl + roi_valid.width]
        update_mask = np.ones((roi[1] - roi[0], roi[3] - roi[2]), dtype=np.float) * odecay
        self._image[roi[0]:roi[1], roi[2]:roi[3]] = np.multiply(self._image[roi[0]:roi[1], roi[2]:roi[3]],
                                                                update_mask).astype(np.uint8)
        self._image[vy, vx] = 255

        return roi_raw != roi_valid
    def reset_frame(self):
        pass


class ResizeTransform:
    def __init__(self, cfg, height, width):
        self.source_height = height
        self.source_width = width

        self.taget_width = cfg.MODEL.IMAGE_SIZE[0]
        self.taget_height = cfg.MODEL.IMAGE_SIZE[1]

        self.sx = (self.taget_width / self.source_width)
        self.sy = (self.taget_height / self.source_height)

    def __call__(self, x, y) -> Any:
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        x = x * self.sx
        y = y * self.sy

        return x, y

    @property
    def height(self):
        return self.taget_height
    
    @property
    def width(self):
        return self.taget_width


class LNES:
    def __init__(self, cfg, height, width):
        self.resize_transform = ResizeTransform(cfg, height, width)
        
        lnes_config = cfg.DATASET.LNES
        self.windows_time_ms = lnes_config.WINDOWS_TIME_MS

    def __call__(self, data_batch) -> Any:
        windows_time_ms = self.windows_time_ms

        if data_batch.shape[-1] == 6:
            xs, ys, ts, ps, fs, segmentation = data_batch.T
        elif data_batch.shape[-1] == 5:
            xs, ys, ts, ps, fs = data_batch.T
        elif data_batch.shape[-1] == 4:
            xs, ys, ts, ps = data_batch.T
        else:
            raise ValueError('Invalid data_batch shape')

        ts = ts.astype(np.float32)
        
        ts = (ts[-1] - ts) * 1e-3 # microseconds to milliseconds 
        
        selected_indices = ts < windows_time_ms

        xs = xs[selected_indices]
        ys = ys[selected_indices]
        ts = ts[selected_indices]
        ps = ps[selected_indices].astype(np.int32)

        if data_batch.shape[-1] == 6:
            segmentation = segmentation[selected_indices]

        xs, ys = self.resize_transform(xs, ys)
        width, height = self.resize_transform.width, self.resize_transform.height

        xs = xs.astype(np.int32)
        ys = ys.astype(np.int32)

        lnes = np.zeros((height, width, 2))
        lnes[ys, xs, ps] = 1.0 - (ts / windows_time_ms)

        data = {
            'input': lnes,
            'coord_x': xs,
            'coord_y': ys,
        }

        if data_batch.shape[-1] >= 5:
            fs = fs[selected_indices]
            data['frame_index'] = fs[-1]            
        
        if data_batch.shape[-1] == 6:
            data['segmentation_indices'] = segmentation.astype(np.uint8)

        return data

    def visualize(cls, lnes: np.ndarray):
        if isinstance(lnes, torch.Tensor):
            lnes = lnes.permute(1, 2, 0)
            lnes = lnes.detach()   

        lnes = lnes.copy() * 255
        lnes = lnes.astype(np.uint8)
                
        h, w = lnes.shape[:2]
    
        b = lnes[..., :1]
        r = lnes[..., 1:]
        g = np.zeros((h, w, 1), dtype=np.uint8)

        lnes = np.concatenate([r, g, b], axis=2).astype(np.uint8)

        return lnes


class EventFrame:
    def __init__(self, cfg, height, width):
        self.height = height
        self.width = width

        self.resize_transform = ResizeTransform(cfg, height, width)

    def __call__(self, data_batch) -> Any:
        if data_batch.shape[-1] == 6:
            xs, ys, ts, ps, fs, segmentation = data_batch.T
        elif data_batch.shape[-1] == 5:
            xs, ys, ts, ps, fs = data_batch.T
        elif data_batch.shape[-1] == 4:
            xs, ys, ts, ps = data_batch.T
        else:
            raise ValueError('Invalid data_batch shape')

        ts = ts.astype(np.float32)
        ts = (ts[-1] - ts) * 1e-3 # microseconds to milliseconds 

        xs, ys = self.resize_transform(xs, ys)
        width, height = self.resize_transform.width, self.resize_transform.height

        xs = xs.astype(np.int32)
        ys = ys.astype(np.int32)
        ps = ps.astype(np.int32)

        ef = np.zeros((height, width, 2))
        ef[ys, xs, ps] = 1.0

        data = {
            'input': ef,
            'coord_x': xs,
            'coord_y': ys,
        }

        if data_batch.shape[-1] >= 5:
            data['frame_index'] = fs[-1]            
        
        if data_batch.shape[-1] == 6:
            data['segmentation_indices'] = segmentation.astype(np.uint8)

        return data

    def visualize(cls, ef: np.ndarray):
        if isinstance(ef, torch.Tensor):
            ef = ef.permute(1, 2, 0)
            ef = ef.detach()   

        ef = ef.copy() * 255
        ef = ef.astype(np.uint8)
                
        h, w = ef.shape[:2]
    
        b = ef[..., :1]
        r = ef[..., 1:]
        g = np.zeros((h, w, 1), dtype=np.uint8)

        ef = np.concatenate([r, g, b], axis=2).astype(np.uint8)

        return ef
    
def compute_temporal_weights(ts, num_bins):

    ts = torch.from_numpy(ts).float()
    # Ensure ts is in [0, 1]
    ts = torch.clamp(ts, 0.0, 1.0)
    scaled_ts = ts * (num_bins - 1)
    lower_bin = torch.floor(scaled_ts).long()
    upper_bin = lower_bin + 1
    fractional = scaled_ts - lower_bin

    # Clamp bins to valid range
    lower_bin = torch.clamp(lower_bin, 0, num_bins - 1)
    upper_bin = torch.clamp(upper_bin, 0, num_bins - 1)

    weights = torch.zeros((ts.shape[0], num_bins), device=ts.device)
    rows = torch.arange(ts.size(0), device=ts.device)
    
    
    # Assign weights to lower and upper bins
    weights[rows, lower_bin] = 1 - fractional
    weights[rows, upper_bin] = fractional
    
    return weights

def interpolate_missing_bins(bin_frame_indices, missing_value=-1):
    """
    Replace missing values in bin_frame_indices (indicated by missing_value) using linear interpolation.
    
    Args:
        bin_frame_indices (torch.Tensor): 1D tensor of bin frame indices.
        missing_value (int, optional): The placeholder value indicating missing data. Default is -1.
    
    Returns:
        torch.Tensor: Tensor with missing values interpolated.
    """
    bin_frame_indices = bin_frame_indices.clone()  # Avoid in-place modifications
    num_bins = bin_frame_indices.size(0)
    
    # Find indices where the value is not missing
    valid_mask = bin_frame_indices != missing_value
    valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
    
    # If no valid indices exist, you might want to set a default (or raise an error)
    if valid_indices.numel() == 0:
        raise ValueError("No valid bin indices available for interpolation.")

    # Process each bin index
    for i in range(num_bins):
        if bin_frame_indices[i] == missing_value:
            # Find the closest valid indices before and after
            prev_valid = valid_indices[valid_indices < i]
            next_valid = valid_indices[valid_indices > i]

            if prev_valid.numel() > 0 and next_valid.numel() > 0:
                # Both previous and next valid bins exist: use linear interpolation.
                left_idx = prev_valid[-1].item()
                right_idx = next_valid[0].item()
                left_val = bin_frame_indices[left_idx].float()
                right_val = bin_frame_indices[right_idx].float()

                # Calculate a weight for the interpolation based on positions
                alpha = (i - left_idx) / (right_idx - left_idx)
                interpolated_value = (1 - alpha) * left_val + alpha * right_val
                bin_frame_indices[i] = int(round(interpolated_value.item()))
            elif prev_valid.numel() > 0:
                # Only previous valid exists (e.g., missing at the end): use the previous value.
                left_idx = prev_valid[-1].item()
                bin_frame_indices[i] = bin_frame_indices[left_idx]
            elif next_valid.numel() > 0:
                # Only next valid exists (e.g., missing at the beginning): use the next value.
                right_idx = next_valid[0].item()
                bin_frame_indices[i] = bin_frame_indices[right_idx]
                
    return bin_frame_indices

class RawEvent:
    def __init__(self, cfg, height, width, temporal_bins):
        self.height = height
        self.width = width

        self.resize_transform = ResizeTransform(cfg, height, width)

        self.temporal_bins = temporal_bins

    def __call__(self, data_batch) -> Any:
        if data_batch.shape[-1] == 6:
            xs, ys, ts, ps, fs, segmentation = data_batch.T
        elif data_batch.shape[-1] == 5:
            xs, ys, ts, ps, fs = data_batch.T
        elif data_batch.shape[-1] == 4:
            xs, ys, ts, ps = data_batch.T
        else:
            raise ValueError('Invalid data_batch shape')

        ts_min = ts.min()
        ts_max = ts.max()
        ts_range = ts_max - ts_min

        if ts_range == 0:
            ts = ts - ts_min
        else:
            ts = (ts - ts.min()) / (ts.max() - ts.min()) # normalize timestamps
        
        ts = ts.astype(np.float32)
        
        # xs, ys = self.resize_transform(xs, ys)
        # xs, ys = self.resize_transform(xs, ys)
        # width, height = self.resize_transform.width, self.resize_transform.height

        

        # print(ts)
        # ts = (ts[-1] - ts) * 1e-3 # microseconds to milliseconds
        # ts = ts * 1e-3 # microseconds to milliseconds 

        # print("frame index: ", fs[-1])

        xs, ys = self.resize_transform(xs, ys)
        width, height = self.resize_transform.width, self.resize_transform.height

        xs = xs.astype(np.int32)
        ys = ys.astype(np.int32)
        ps = ps.astype(np.int32)

        fs = torch.from_numpy(fs)

        weights = compute_temporal_weights(ts, self.temporal_bins)
        weighted_fs = weights * fs.unsqueeze(1)  # [num_events, num_bins]
        sum_weighted_fs = torch.sum(weighted_fs, dim=0)  # [num_bins]
        sum_weights = torch.sum(weights, dim=0)  # [num_bins]

        # Avoid division by zero
        average_fs = sum_weighted_fs / (sum_weights + 1e-6)
        bin_frame_indices = torch.round(average_fs).to(torch.long)

        # Handle bins with no events (optional: interpolate or use default)
        # TODO: Add interpolation for empty bins
        bin_frame_indices[sum_weights == 0] = -1  # Placeholder for no data
        if -1 in bin_frame_indices:
            bin_frame_indices = interpolate_missing_bins(bin_frame_indices)
            # print("Interpolated missing bins", bin_frame_indices)

        # print("frame index: ", fs[-1])
        # print("bin_frame_indices: ", bin_frame_indices)

        events_tensor = np.stack([xs, ys, ts, ps], axis=1)

        events_tensor = torch.tensor(events_tensor, dtype=torch.float32)
        

        data = {
            'input': events_tensor,
            'coord_x': xs,
            'coord_y': ys,
        }

        if data_batch.shape[-1] >= 5:
            # data['frame_index'] = fs[-1]
            # data['frame_index'] = torch.tensor([fs[-1]])
            data['frame_index'] = bin_frame_indices        
        
        if data_batch.shape[-1] == 6:
            data['segmentation_indices'] = segmentation.astype(np.uint8)

        return data

    def visualize(cls, events_tensor, height, width):
        if isinstance(events_tensor, torch.Tensor):
            # ef = ef.permute(1, 2, 0)
            events_tensor = events_tensor.detach()

        ef = np.zeros((height, width, 3), dtype=np.uint8)

        # import pdb; pdb.set_trace()
        xs, ys, _, ps, _ = events_tensor.T
        xs = xs.astype(np.int32)
        ys = ys.astype(np.int32)

        # Mark positive polarity events in red and negative in blue
        ef[ys, xs, 0] = (ps == 1) * 255  # Red channel for positive events
        ef[ys, xs, 2] = (ps == 0) * 255  # Blue channel for negative events

        return ef
