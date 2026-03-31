import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

def compute_fn_v3(model, batch, s5_state, prev_buffer=None, prev_key=None, batch_first=False):
    inps = []

    frame_index = []
    gt_hms = []
    gt_j2d = []
    gt_j3d = []
    gt_seg = []
    vis_j2d = []
    vis_j3d = []
    valid_j3d = []
    valid_seg = []
    filename = []
    use_bg = []
    bg_data = []
    vis_ja = []
    for (data, meta) in batch:
        inp = data['x']
        inps.append(inp[None, ...])    

        gt_hms_ = data['hms']
        gt_j3d_ = data['j3d'] 
        gt_seg_ = data['segmentation_mask']

        gt_j2d_ = meta['j2d']

        vis_j2d_ = meta['vis_j2d']
        vis_j3d_ = meta['vis_j3d']
        valid_j3d_ = meta['valid_j3d']
        valid_seg_ = meta['valid_seg']
        frame_index_ = meta['frame_index']
        filename_ = meta['pose_filename']
        use_bg_ = meta['use_bg']
        bg_data_ = meta['bg_data']
        vis_ja_ = meta['vis_ja']
        

        gt_hms.append(gt_hms_)
        gt_j3d.append(gt_j3d_)
        gt_seg.append(gt_seg_)

        gt_j2d.append(gt_j2d_)
        vis_j2d.append(vis_j2d_)
        vis_j3d.append(vis_j3d_)
        valid_j3d.append(valid_j3d_)
        valid_seg.append(valid_seg_)
        filename.append(filename_)

        use_bg.append(use_bg_)
        bg_data.append(bg_data_)
        vis_ja.append(vis_ja_)


        frame_index.append(frame_index_)

    del batch


    inps = torch.cat(inps, dim=0).permute(0, 1, 4, 2, 3)

    gt_hms = torch.stack(gt_hms)
    gt_j3d = torch.stack(gt_j3d)
    gt_seg = torch.stack(gt_seg)

    gt_j2d = torch.stack(gt_j2d)
    vis_j2d = torch.stack(vis_j2d)
    vis_j3d = torch.stack(vis_j3d)
    valid_j3d = torch.stack(valid_j3d)
    valid_seg = torch.stack(valid_seg)
    frame_index = torch.stack(frame_index)

    use_bg = torch.stack(use_bg)
    bg_data = torch.stack(bg_data).cuda().permute(0, 1, 4, 2, 3)
    vis_ja = torch.stack(vis_ja).cuda()


    augmentation_data = {
        'bg_mask': gt_seg,
        'use_bg': use_bg,
        'bg_data': bg_data
    }

    # inps = inps.permute(0, 1, 4, 2, 3)
    outputs = model(inps, s5_state, augmentation_data=augmentation_data)
    
    T, B, C, H, W = inps.shape
    return inps.view(T * B, C, H, W), outputs, gt_hms, gt_j3d, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index, filename, vis_ja

def compute_fn_v4(model, batch, s5_state, augmentation=False):

    inps = []

    frame_index = []
    gt_j2d = []
    gt_j3d = []
    gt_seg = []
    vis_j2d = []
    vis_j3d = []
    valid_j3d = []
    filename = []
    use_bg = []
    bg_data = []
    vis_ja = []
    valid_joints = []
    for (data, meta) in batch:
        inp = data['x']
        inps.append(inp[None, ...])    

        gt_j3d_ = data['j3d'] 
        gt_seg_ = data['segmentation_mask']

        gt_j2d_ = meta['j2d']

        vis_j2d_ = meta['vis_j2d']
        vis_j3d_ = meta['vis_j3d']
        valid_j3d_ = meta['valid_j3d']
        frame_index_ = meta['frame_index']
        filename_ = meta['pose_filename']
        valid_joints_ = meta['valid_joints']
        vis_ja_ = meta['vis_ja']

        gt_j3d.append(gt_j3d_)
        gt_seg.append(gt_seg_)
        gt_j2d.append(gt_j2d_)
        vis_j2d.append(vis_j2d_)
        vis_j3d.append(vis_j3d_)
        valid_j3d.append(valid_j3d_)
        filename.append(filename_)
        vis_ja.append(vis_ja_)
        valid_joints.append(valid_joints_)

        if augmentation:
            use_bg_ = meta['use_bg']
            bg_data_ = meta['bg_data']
            use_bg.append(use_bg_)
            bg_data.append(bg_data_)

        frame_index.append(frame_index_)

    del batch

    inps = torch.cat(inps, dim=0).cuda().permute(0, 1, 4, 2, 3)

    gt_j3d = torch.stack(gt_j3d).cuda()
    gt_seg = torch.stack(gt_seg).cuda()

    gt_j2d = torch.stack(gt_j2d)
    vis_j2d = torch.stack(vis_j2d).cuda()
    vis_j3d = torch.stack(vis_j3d).cuda()
    valid_j3d = torch.stack(valid_j3d).cuda()
    frame_index = torch.stack(frame_index)
    valid_joints = torch.stack(valid_joints).cuda()
    vis_ja = torch.stack(vis_ja).cuda()


    if augmentation:
        use_bg = torch.stack(use_bg)
        bg_data = torch.stack(bg_data).cuda().permute(0, 1, 4, 2, 3)
        augmentation_data = {
            'bg_mask': gt_seg,
            'use_bg': use_bg,
            'bg_data': bg_data
        }
    else: 
        augmentation_data = {}

    outputs = model(inps, s5_state, augmentation_data=augmentation_data)

    meta_data = {
        'gt_j3d': gt_j3d,
        'gt_j2d': gt_j2d,
        'vis_j2d': vis_j2d,
        'vis_j3d': vis_j3d,
        'valid_j3d': valid_j3d,
        'frame_index': frame_index,
        'filename': filename,
        'valid_joints': valid_joints,
        'vis_ja': vis_ja
    }
    
    T, B, C, H, W = inps.shape
    return inps.view(T * B, C, H, W), outputs, meta_data

def percentile(t, q):
    B, C, H, W = t.shape
    k = 1 + round(.01 * float(q) * (C * H * W - 1))
    result = t.view(B, -1).kthvalue(k).values
    return result[:,None,None,None]

def create_image(representation):
    B, C, H, W = representation.shape
    # import pdb; pdb.set_trace()

    representation = representation[:, 8*2:(8+1)*2, :, :]
    if isinstance(representation, torch.Tensor):
        representation = representation.permute(0, 3, 2, 1)
        representation = representation.detach()

    representation = representation * 255
    representation = representation.astype(np.uint8)

    B, h, w, _ = representation.shape
    
    r = representation[..., :1]
    b = representation[..., 1:]
    g = np.zeros((B, h, w, 1), dtype=np.uint8)

    # import pdb; pdb.set_trace()

    representation = np.concatenate([r, g, b], axis=-1).astype(np.uint8)

    representation = torch.from_numpy(representation).permute(0, 3, 2, 1).to(torch.float32).cuda()
    # representation = representation.view(B, 3, C // 3, H, W).sum(2)

    # # Perform robust min-max normalization
    # robust_max_vals = percentile(representation, 95)
    # robust_min_vals = percentile(representation, 5)

    # representation = (representation - robust_min_vals) / (robust_max_vals - robust_min_vals)
    # representation = torch.clamp(representation, 0, 1)

    # # Apply gamma correction
    # gamma = 0.5
    # representation = representation ** gamma

    return representation

# def create_image(representation):
#     B, C, H, W = representation.shape
#     representation = representation.view(B, 3, C // 3, H, W).sum(2)

#     # do robust min max norm
#     representation = representation.detach().cpu()
#     robust_max_vals = percentile(representation, 99)
#     robust_min_vals = percentile(representation, 1)

#     representation = (representation - robust_min_vals)/(robust_max_vals - robust_min_vals)
#     # representation = torch.clamp(255*representation, 0, 255).byte()
#     # representation = torchvision.utils.make_grid(representation)

#     return representation

def recursive_detach(inp):
    if isinstance(inp, torch.Tensor):
        return inp.detach()
    if isinstance(inp, list):
        return [recursive_detach(x) for x in inp]
    if isinstance(inp, tuple):
        return tuple([recursive_detach(x) for x in inp])
    if isinstance(inp, dict):
        return {k: recursive_detach(v) for k, v in inp.items()}
    raise NotImplementedError
    
def resize_transform(cfs, x, y, height, width):
    x = x.detach()
    y = y.detach()
    target_width = cfs.MODEL.IMAGE_SIZE[0]
    target_height = cfs.MODEL.IMAGE_SIZE[1]
    sx = (target_width / width)
    sy = (target_height / height)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x = x * sx
    y = y * sy
    return x, y


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
