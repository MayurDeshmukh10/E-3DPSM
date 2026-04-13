import logging

import torch

logger = logging.getLogger(__name__)


def prepare_batch_and_forward(model, batch, s5_state, augmentation=False):
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

    for data, meta in batch:
        inps.append(data['x'][None, ...])

        gt_j3d.append(data['j3d'])
        gt_seg.append(data['segmentation_mask'])
        gt_j2d.append(meta['j2d'])
        vis_j2d.append(meta['vis_j2d'])
        vis_j3d.append(meta['vis_j3d'])
        valid_j3d.append(meta['valid_j3d'])
        filename.append(meta['pose_filename'])
        vis_ja.append(meta['vis_ja'])
        valid_joints.append(meta['valid_joints'])
        frame_index.append(meta['frame_index'])

        if augmentation:
            use_bg.append(meta['use_bg'])
            bg_data.append(meta['bg_data'])

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
        augmentation_data = {
            'bg_mask': gt_seg,
            'use_bg': torch.stack(use_bg),
            'bg_data': torch.stack(bg_data).cuda().permute(0, 1, 4, 2, 3),
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
        'vis_ja': vis_ja,
    }

    t_steps, batch_size, channels, height, width = inps.shape
    return inps.view(t_steps * batch_size, channels, height, width), outputs, meta_data


def log_metric_table(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)

    logger.info('| Arch ' + ' '.join([f'| {name}' for name in names]) + ' |')
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'

    logger.info(
        '| ' + full_arch_name + ' ' + ' '.join([f'| {value:.3f}' for value in values]) + ' |'
    )
