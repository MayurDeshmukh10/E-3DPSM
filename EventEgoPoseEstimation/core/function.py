import os
import time
import logging
from typing import Any

import numpy as np
import torch
import cv2

from pathlib import Path


from EventEgoPoseEstimation.core.inference import get_j2d_from_hms
from EventEgoPoseEstimation.core.evaluate import accuracy, root_accuracy
from EventEgoPoseEstimation.utils.vis import save_debug_images, save_debug_3d_joints, save_debug_segmenation, save_debug_eros, plot_heatmaps, generate_skeleton_image
from EventEgoPoseEstimation.utils.skeleton import Skeleton
from torch.nn.utils.rnn import pad_sequence
import torchvision
from torch.cuda.amp import autocast, GradScaler

from torch.profiler import profile, ProfilerActivity

from EventEgoPoseEstimation.utils.utils import save_checkpoint

logger = logging.getLogger(__name__)

# scaler = GradScaler()

def compute_fn(model, batch, temporal_steps, device='cuda'):
    inps = []
    new_inps = []
    frame_index = []
    gt_hms = []
    gt_j2d = []
    gt_j3d = []
    gt_seg = []
    vis_j2d = []
    vis_j3d = []
    valid_j3d = []
    valid_seg = []
    bg_data = []
    bg_mask = []
    use_bg = []
    filename = []

    augmentation_data = {}

    data_batch = batch[0]
    meta_batch = batch[1]

    if len(data_batch[0]['x']) < temporal_steps:
        temporal_steps = 1

    for i in range(temporal_steps):
        inps_t = []
        frame_index_t = []
        gt_hms_t = []
        gt_j2d_t = []
        gt_j3d_t = []
        gt_seg_t = []
        vis_j2d_t = []
        vis_j3d_t = []
        valid_j3d_t = []
        valid_seg_t = []
        bg_data_t = []
        bg_mask_t = []
        filename_t = []
        use_bg_t = []
        skip_augmentation_data = False
        for data, meta in zip(data_batch, meta_batch):
            inp = data['x'][i]
            inps_t.append(inp)

            gt_hms_ = data['hms'][i]
            gt_j3d_ = data['j3d'][i]
            gt_seg_ = data['segmentation_mask'][i]

            gt_j2d_ = meta['j2d'][i]

            vis_j2d_ = meta['vis_j2d'][i]
            vis_j3d_ = meta['vis_j3d'][i]
            valid_j3d_ = meta['valid_j3d'][i]
            valid_seg_ = meta['valid_seg'][i]
            frame_index_ = meta['frame_index'][i]
            pose_filename_ = meta['pose_filename'][i]
             
            try:
                # if meta['use_bg'][i] == True:
                #     bg_data_ = meta['bg_data'][i]
                # else:
                #     bg_data_ = []
                bg_data_ = meta['bg_data'][i]
                use_bg_ = meta['use_bg'][i]
            except KeyError: # case where dataloader for test and val
                skip_augmentation_data = True
                bg_data_ = []

            if not skip_augmentation_data:
                bg_data_t.append(bg_data_)
                use_bg_t.append(use_bg_)

            gt_hms_t.append(gt_hms_)
            gt_j3d_t.append(gt_j3d_)
            gt_seg_t.append(gt_seg_)

            gt_j2d_t.append(gt_j2d_)
            vis_j2d_t.append(vis_j2d_)
            vis_j3d_t.append(vis_j3d_)
            valid_j3d_t.append(valid_j3d_)
            valid_seg_t.append(valid_seg_)
            
            filename_t.append(pose_filename_)
            frame_index.append(frame_index_)
        
        # bg_data.append(bg_data_t)
        if not skip_augmentation_data:
            bg_data.append(torch.stack(bg_data_t))
            use_bg.append(torch.stack(use_bg_t))

        gt_j3d.append(torch.stack(gt_j3d_t))
        gt_hms.append(torch.stack(gt_hms_t))
        gt_seg.append(torch.stack(gt_seg_t))
        gt_j2d.append(torch.stack(gt_j2d_t))
        vis_j2d.append(torch.stack(vis_j2d_t))
        vis_j3d.append(torch.stack(vis_j3d_t))
        valid_j3d.append(torch.stack(valid_j3d_t))
        valid_seg.append(torch.stack(valid_seg_t))
        filename.append(filename_t)
        # filename.append

        
        max_rows = max([inp.shape[0] for inp in inps_t])
        padding_value = torch.tensor([-10, -10, -10, -10], dtype=torch.float32, device=device)
        padded_inps_t = [
            torch.cat([inp, padding_value.repeat(max_rows - inp.shape[0], 1)], dim=0)
            for inp in inps_t
        ]
        inps.append(torch.stack(padded_inps_t))
        
    max_rows = max([inp.shape[1] for inp in inps])
    padding_value = torch.tensor([-10, -10, -10, -10], dtype=torch.float32, device=device)
    temp = []
    for ip in inps:
        aa = [
                torch.cat([i, padding_value.repeat(max_rows - i.shape[0], 1)], dim=0)
                for i in ip
            ]
        temp.append(torch.stack(aa))


    inps = torch.stack(temp)
    gt_hms = torch.stack(gt_hms)
    gt_j3d = torch.stack(gt_j3d)
    gt_seg = torch.stack(gt_seg)
    gt_j2d = torch.stack(gt_j2d)
    vis_j2d = torch.stack(vis_j2d)
    vis_j3d = torch.stack(vis_j3d)
    valid_j3d = torch.stack(valid_j3d)
    # valid_seg = torch.cat([v.unsqueeze(0) for v in valid_seg])
    valid_seg = torch.stack(valid_seg)
    frame_index = torch.cat([v.unsqueeze(0) for v in frame_index], dim=0)
    # filename = torch.stack(filename)

    if not skip_augmentation_data:
        bg_data = torch.stack(bg_data)
        use_bg = torch.stack(use_bg)

        augmentation_data = { 
            'bg_mask': gt_seg,
            'bg_data': bg_data,
            'use_bg': use_bg
        }
    
    _, _, N, C = inps.shape


    

    del batch
    
    outputs = model(inps, augmentation_data)
    # except torch.OutOfMemoryError:
    #     return inps, _, gt_hms, gt_j3d, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index, False

    T, B, N, C = inps.shape
    return inps.view(T * B, N, C), outputs, gt_hms, gt_j3d, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index, filename

def compute_fn_v2(model, batch, temporal_steps, device='cuda'):
    data_batch, meta_batch = batch
    
    # Early validation and adjustment of temporal_steps
    if len(data_batch[0]['x']) < temporal_steps:
        temporal_steps = len(data_batch[0]['x'])
    
    # Pre-allocate lists with known sizes where possible
    batch_size = len(data_batch)
    inps = []
    gt_hms = []
    gt_j3d = []
    gt_seg = []
    gt_j2d = []
    vis_j2d = []
    vis_j3d = []
    valid_j3d = []
    valid_seg = []
    frame_index = []
    filename = []
    
    # Check if augmentation data is available only once
    skip_augmentation_data = False
    try:
        _ = meta_batch[0]['bg_data'][0]
        _ = meta_batch[0]['use_bg'][0]
    except KeyError:
        skip_augmentation_data = True
    
    # Only create these if needed
    bg_data = [] if not skip_augmentation_data else None
    use_bg = [] if not skip_augmentation_data else None
    
    # padding_value = torch.tensor([-10, -10, -10, -10], dtype=torch.float32, device=device)
    
    # Process temporal steps more efficiently
    for i in range(temporal_steps):
        # Pre-allocate lists with known size
        inps_t = []
        gt_hms_t = []
        gt_j3d_t = []
        gt_seg_t = []
        gt_j2d_t = []
        vis_j2d_t = []
        vis_j3d_t = []
        valid_j3d_t = []
        valid_seg_t = []
        filename_t = []
        frame_indices_t = []
        
        if not skip_augmentation_data:
            bg_data_t = []
            use_bg_t = []
        
        # Find max_rows for this temporal step during data collection
        max_rows_t = 0
        
        # Process each item in the batch
        for data_idx, (data, meta) in enumerate(zip(data_batch, meta_batch)):
            # Extract data directly (avoid multiple dictionary lookups)
            inp = data['x'][i]
            # max_rows_t = max(max_rows_t, inp.shape[0])
            inps_t.append(inp)
            
            # Collect ground truth data
            gt_hms_t.append(data['hms'][i])
            gt_j3d_t.append(data['j3d'][i])
            gt_seg_t.append(data['segmentation_mask'][i])
            
            # Collect meta data
            gt_j2d_t.append(meta['j2d'][i])
            vis_j2d_t.append(meta['vis_j2d'][i])
            vis_j3d_t.append(meta['vis_j3d'][i])
            valid_j3d_t.append(meta['valid_j3d'][i])
            valid_seg_t.append(meta['valid_seg'][i])
            frame_indices_t.append(meta['frame_index'][i])
            filename_t.append(meta['pose_filename'][i])
            
            # Collect augmentation data if available
            if not skip_augmentation_data:
                bg_data_t.append(meta['bg_data'][i])
                use_bg_t.append(meta['use_bg'][i])
        
        # # Stack and pad inputs for this temporal step in one operation
        # padded_inps_t = torch.stack([
        #     torch.cat([inp, padding_value.repeat(max_rows_t - inp.shape[0], 1)], dim=0)
        #     for inp in inps_t
        # ])
        
        inps.append(inps_t)
        frame_index.extend(frame_indices_t)
        
        # Stack all collected tensors for this temporal step
        gt_hms.append(torch.stack(gt_hms_t))
        gt_j3d.append(torch.stack(gt_j3d_t))
        gt_seg.append(torch.stack(gt_seg_t))
        gt_j2d.append(torch.stack(gt_j2d_t))
        vis_j2d.append(torch.stack(vis_j2d_t))
        vis_j3d.append(torch.stack(vis_j3d_t))
        valid_j3d.append(torch.stack(valid_j3d_t))
        valid_seg.append(torch.stack(valid_seg_t))
        filename.append(filename_t)
        
        # Stack augmentation data if available
        if not skip_augmentation_data:
            bg_data.append(torch.stack(bg_data_t))
            use_bg.append(torch.stack(use_bg_t))
    
    # Find maximum rows across all temporal steps
    # max_rows = max(inp.shape[1] for inp in inps)
    
    # # Process and stack all inputs with uniform padding
    # inps = torch.stack([
    #     torch.stack([
    #         torch.cat([row, padding_value.repeat(max_rows - row.shape[0], 1)], dim=0)
    #         for row in temporal_step
    #     ])
    #     for temporal_step in inps
    # ])
    
    # Stack all temporal data
    gt_hms = torch.stack(gt_hms)
    gt_j3d = torch.stack(gt_j3d)
    gt_seg = torch.stack(gt_seg)
    gt_j2d = torch.stack(gt_j2d)
    vis_j2d = torch.stack(vis_j2d)
    vis_j3d = torch.stack(vis_j3d)
    valid_j3d = torch.stack(valid_j3d)
    valid_seg = torch.stack(valid_seg)
    
    # Process frame indices as a single operation
    frame_index = torch.stack([idx.unsqueeze(0) for idx in frame_index])
    
    # Create augmentation data dictionary only if needed
    augmentation_data = {}
    if not skip_augmentation_data:
        bg_data = torch.stack(bg_data)
        use_bg = torch.stack(use_bg)
        
        augmentation_data = {
            'bg_mask': gt_seg,
            'bg_data': bg_data,
            'use_bg': use_bg
        }
    
    # Free memory before running the model
    del batch
    # torch.cuda.empty_cache()  # Explicitly clear GPU cache
    
    # Run model
    outputs = model(inps, augmentation_data)
    
    # Reshape inputs for return
    # T, B, N, C = inps.shape
    return (len(inps) * len(inps[0])), outputs, gt_hms, gt_j3d, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index, filename

def compute_fn_v3(model, batch, prev_buffer=None, prev_key=None, batch_first=False):
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

    inps = torch.cat(inps, dim=0)

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
    bg_data = torch.stack(bg_data).cuda()
    vis_ja = torch.stack(vis_ja).cuda()


    augmentation_data = {
        'bg_mask': gt_seg,
        'use_bg': use_bg,
        'bg_data': bg_data
    }

    outputs = model(inps, augmentation_data=augmentation_data)
    
    T, B, C, H, W = inps.shape
    return inps.view(T * B, C, H, W), outputs, gt_hms, gt_j3d, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index, filename, vis_ja

def compute_fn_v4(model, batch, prev_buffer=None, prev_key=None, batch_first=False):

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

    inps = torch.cat(inps, dim=0).cuda()

    gt_hms = torch.stack(gt_hms)
    gt_j3d = torch.stack(gt_j3d).cuda()
    gt_seg = torch.stack(gt_seg).cuda()

    gt_j2d = torch.stack(gt_j2d)
    vis_j2d = torch.stack(vis_j2d).cuda()
    vis_j3d = torch.stack(vis_j3d).cuda()
    valid_j3d = torch.stack(valid_j3d).cuda()
    valid_seg = torch.stack(valid_seg).cuda()
    frame_index = torch.stack(frame_index)

    use_bg = torch.stack(use_bg)
    bg_data = torch.stack(bg_data).cuda()
    vis_ja = torch.stack(vis_ja).cuda()


    augmentation_data = {
        'bg_mask': gt_seg,
        'use_bg': use_bg,
        'bg_data': bg_data
    }

    # import pdb; pdb.set_trace()

    outputs = model(inps, augmentation_data=augmentation_data)
    
    T, B, C, H, W = inps.shape
    return inps.view(T * B, C, H, W), outputs, gt_hms, gt_j3d, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index, filename, vis_ja

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

def train(config, train_loader, model, criterions, optimizer, epoch, output_dir, tb_log_dir, writer_dict, pretraining=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    hms_losses = AverageMeter()
    j3d_losses = AverageMeter()    
    seg_losses = AverageMeter()

    acc = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()

    temporal_steps = config.DATASET.TEMPORAL_STEPS

    batch_skipped = 0
    valid_counter = 0

    for i, batch in enumerate(train_loader):

    
        if valid_counter > config.TRAIN_ITERATIONS_PER_EPOCH: break

        data_time.update(time.time() - end)

        # try:
        data = batch[0]
        meta = batch[1]
        # except Exception as e:
        #     print("Error in batch")
        #     print(e)
        #     continue
            

        # logger.info("Batch shape: {}".format(inp.shape))
        # memory_stats = torch.cuda.memory_stats("cuda:0")
        # logger.info("Before model")
        # logger.info(f"Current allocated memory: {memory_stats['allocated_bytes.all.current'] / (1024 ** 2):.2f} MB")
        # logger.info(f"Peak allocated memory: {memory_stats['allocated_bytes.all.peak'] / (1024 ** 2):.2f} MB")
        # logger.info(f"Current reserved memory: {memory_stats['reserved_bytes.all.current'] / (1024 ** 2):.2f} MB")
        # logger.info(f"Peak reserved memory: {memory_stats['reserved_bytes.all.peak'] / (1024 ** 2):.2f} MB")
        # try:
        inp, outputs, gt_hms, gt_j3d, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index, pose_filename = compute_fn(model, batch, temporal_steps)
        if status is False: # if out of memory
            batch_skipped += 1
            print("Input shape: ", inp.shape)
            print("FAILED COUNTER: ", batch_skipped)
            continue
        # except torch.OutOfMemoryError:
        #     batch_skipped += 1
        #     print("FAILED COUNTER: ", batch_skipped)
        #     continue

        valid_counter += 1

        meta = {'j3d': gt_j3d, 'j2d': gt_j2d, 'vis_j2d': vis_j2d, 'vis_j3d': vis_j3d}

        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        
        # pred_hms = outputs['hms']
        pred_seg = outputs['seg']
        pred_eros = outputs['eros']
        
        gt_j3d = gt_j3d  * 1000 # scale to mm
        pred_j3d = outputs['j3d'] * 1000 # scale to mm

        # with autocast():
        with torch.amp.autocast('cuda', enabled=True):
            # loss_hms = criterions['hms'](pred_hms, gt_hms, vis_j2d * 10)  # scale to 10
            loss_seg = criterions['seg'](pred_seg, gt_seg, valid_seg)
            loss_j3d = criterions['j3d'](pred_j3d, gt_j3d, vis_j3d * 1e-2)  # scale to 1e-2
            loss_delta_j3d = criterions['delta_j3d'](pred_j3d, gt_j3d, vis_j3d * 1e-2)  # scale to 1e-2
                
            # loss = loss_hms + loss_j3d + loss_seg
        
  
        pred_j2d = get_j2d_from_hms(config, pred_hms)

        pred_seg_detached = torch.sigmoid(pred_seg.detach().clone())

        # compute gradient and do update step
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        

        # B = int((1+inp[-1,-1]).item())
        hms_losses.update(loss_hms.item(), inp.size(0))
        seg_losses.update(loss_seg.item(), inp.size(0))
        j3d_losses.update(loss_j3d.item(), inp.size(0))

        losses.update(loss.item(), inp.size(0))
        
        avg_acc, cnt = accuracy(gt_j3d, pred_j3d, valid_j3d)
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        representation = outputs['representation']
        representation_image = create_image(representation)
        pred_eros_image = create_image(pred_eros.detach())


        outputs['prev_states'].detach()
        # states_store = outputs['states_store']

        # recursive_detach(states_store)

        # recursive_detach(states_store)

        torch.cuda.empty_cache()

        # print("After model")
        # memory_stats = torch.cuda.memory_stats("cuda:0")
        # logger.info("Batch shape: {}".format(inp.shape))
        # logger.info(f"Peak reserved memory: {memory_stats['reserved_bytes.all.peak'] / (1024 ** 2):.2f} MB")
        # # import pdb; pdb.set_trace()
        # logger.info(f"Current allocated memory: {memory_stats['allocated_bytes.all.current'] / (1024 ** 2):.2f} MB")
        # logger.info(f"Peak allocated memory: {memory_stats['allocated_bytes.all.peak'] / (1024 ** 2):.2f} MB")
        # logger.info(f"Current reserved memory: {memory_stats['reserved_bytes.all.current'] / (1024 ** 2):.2f} MB")
        
        # import pdb; pdb.set_trace()

        if i % 5000 == 0:
            filename = f'{i}_it_checkpoint.pth'
            save_checkpoint(epoch + 1, {
            'epoch': epoch + 1,
            'model': config.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': 1e6,
            'optimizer': optimizer.state_dict(),
        }, False, output_dir, tb_log_dir, filename=filename)


        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                'Speed {speed:.1f} samples/s\t' \
                'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                'HM_Loss {hms_loss.val:.5f} ({hms_loss.avg:.5f})\t' \
                'J3D_Loss {j3d_loss.val:.5f} ({j3d_loss.avg:.5f})\t' \
                'SEG_Loss {seg_loss.val:.5f} ({seg_loss.avg:.5f})\t' \
                'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                'MPJPE {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    speed=inp.size(0)/batch_time.val,
                    data_time=data_time, 
                    loss=losses, 
                    j3d_loss=j3d_losses, 
                    hms_loss=hms_losses,
                    seg_loss=seg_losses,
                    acc=acc
                    )
            logger.info(msg)

            memory_stats = torch.cuda.memory_stats("cuda:0")
            logger.info("Batch shape: {}".format(inp.shape))
            # logger.info(f"Current allocated memory: {memory_stats['allocated_bytes.all.current'] / (1024 ** 2):.2f} MB")
            # logger.info(f"Peak allocated memory: {memory_stats['allocated_bytes.all.peak'] / (1024 ** 2):.2f} MB")
            # logger.info(f"Current reserved memory: {memory_stats['reserved_bytes.all.current'] / (1024 ** 2):.2f} MB")
            logger.info(f"Peak reserved memory: {memory_stats['reserved_bytes.all.peak'] / (1024 ** 2):.2f} MB")

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            # writer.add_scalar('train_hms_loss', hms_losses.val, global_steps)
            # writer.add_scalar('train_j3d_loss', j3d_losses.val, global_steps)
            # writer.add_scalar('train_seg_loss', seg_losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            if int(config.BATCH_SIZE) < 4:
                n_images = int(config.BATCH_SIZE)
            else:
                n_images = 4

            if i % (config.PRINT_FREQ * 4) == 0:
                # try:
                save_debug_images(config, representation_image, meta, gt_hms, pred_j2d, pred_hms, 'train', writer, global_steps, n_images=n_images)
                save_debug_3d_joints(config, inp, meta, gt_j3d, pred_j3d, 'train', writer, global_steps)
                save_debug_segmenation(config, inp, meta, gt_seg, pred_seg_detached, 'train', writer, global_steps)
                save_debug_eros(config, representation_image, meta, pred_eros_image, 'train', writer, global_steps)
                # except Exception as e:
                    # print("Error in saving debug data")
                    # print(e)

        


def validate(config, val_loader, val_dataset, model, criterions, output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
        
    acc_j3d = AverageMeter()

    buffer = None
    key = None

    # switch to evaluate mode
    model.eval()

    inp_W, inp_H = config.MODEL.IMAGE_SIZE  
    hm_W, hm_H = config.MODEL.HEATMAP_SIZE

    # buffer = torch.zeros(1, 4, inp_H, inp_W).cuda() # change
    buffer = torch.zeros(config.BATCH_SIZE, config.MODEL.INPUT_CHANNEL, inp_H, inp_W).cuda() # change

    key = torch.ones(config.BATCH_SIZE, 1, hm_H, hm_W).cuda()

    temporal_steps = config.DATASET.TEMPORAL_STEPS

    all_frame_indices = []
    all_gt_j3ds = []
    all_preds_j3d = []
    all_vis_j3d = []
    batch_skipped = 0
    valid_counter = 0
    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            # if i > config.TEST_ITERATIONS_PER_EPOCH: break
            # inp, outputs, gt_hms, gt_j3d, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index = compute_fn(model, batch, temporal_steps, buffer, key, batch_first=True)
            # try:
            inp, outputs, gt_hms, gt_j3d, gt_seg, gt_j2d, vis_j2d, vis_j3d, valid_j3d, valid_seg, frame_index, status = compute_fn(model, batch, temporal_steps, buffer, key, batch_first=True)
            
            if status is False: # if out of memory
                batch_skipped += 1
                print("Input shape: ", inp.shape)
                print("FAILED COUNTER: ", batch_skipped)
                continue
            
            valid_counter += 1
            
            meta = {'j3d': gt_j3d, 'j2d': gt_j2d, 'vis_j2d': vis_j2d, 'vis_j3d': vis_j3d}

            pred_hms = outputs['hms']            
            pred_j3d = outputs['j3d'] * 1000 # scale to mm        
            gt_j3d = gt_j3d * 1000 # scale to mm
            
            avg_acc, cnt = accuracy(gt_j3d, pred_j3d, valid_j3d)
            acc_j3d.update(avg_acc, cnt)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            representation = outputs['representation']
            representation_image = create_image(representation)

            pred_j3d = pred_j3d.detach().detach()
            preds_j2d = get_j2d_from_hms(config, pred_hms)
    
            all_preds_j3d.append(pred_j3d)
            all_gt_j3ds.append(gt_j3d.detach())
            all_vis_j3d.append(valid_j3d.detach())
            all_frame_indices.append(frame_index.detach())


            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    'MPJPE {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                        i, len(val_loader), batch_time=batch_time,
                        acc=acc_j3d)
                logger.info(msg)

        all_preds_j3d = np.concatenate(all_preds_j3d, axis=0)
        all_gt_j3ds = np.concatenate(all_gt_j3ds, axis=0)
        all_vis_j3d = np.concatenate(all_vis_j3d, axis=0)
        all_frame_indices = np.concatenate(all_frame_indices, axis=0)

        name_values, perf_indicator = val_dataset.evaluate_dataset(config, frame_indices=all_frame_indices, all_gt_j3ds=all_gt_j3ds, all_preds_j3d=all_preds_j3d, all_vis_j3d=all_vis_j3d)
                
        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)
        
        name_values, perf_indicator = val_dataset.evaluate_joints(config, all_gt_j3ds=all_gt_j3ds, all_preds_j3d=all_preds_j3d, all_vis_j3d=all_vis_j3d)
        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)
            
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar(
            'valid_acc_j3d',
            acc_j3d.avg,
            global_steps
        )

        for key, value in name_values.items():
            writer.add_scalar(
                f'valid_{key}',
                value,
                global_steps
            )
        writer_dict['valid_global_steps'] = global_steps + 1

        writer = writer_dict['writer']

        try:
            save_debug_images(config, representation_image, meta, gt_hms, preds_j2d, pred_hms, 'test', writer, global_steps)
            save_debug_3d_joints(config, inp, meta, gt_j3d, pred_j3d, 'test', writer, global_steps)
        except Exception as e:
            print("Error in saving debug images or 3d joints")
            print(e)


    return perf_indicator
    
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

def test(cfg, valid_loader, valid_dataset, model, tb_log_dir, writer_dict, seq_time_in_sec=10):
    fps = 30
    
    seq_len = seq_time_in_sec * fps
    data_len = len(valid_dataset)
    
    global_steps = writer_dict['valid_global_steps']
    np.random.seed(int(global_steps))
    start = np.random.randint(0, data_len - cfg.DATASET.TEMPORAL_STEPS)
    stop = min(start + seq_len, data_len)

    tb_log_dir = Path(tb_log_dir)
    video_path = str(tb_log_dir / f'{global_steps}.mp4')

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (400 * 2, 300 * 3))
    model.eval()
        
    for i in range(start, stop):
        gt_j3d = []
        inps = []
        gt_hms = []
        
        data, meta = valid_dataset[i]

        inp = data['x']
        # max_len, feature_dim = inp.shape
        # indices = torch.arange(0, 1, dtype=torch.float32).view(-1, 1, 1).expand(-1, max_len, 1)
        # inp = torch.cat([inp, indices.squeeze(0)], dim=1).cuda()

        gt_j3d_ = data['j3d']
        # gt_hms = data['hms']

        inps.append(inp[None, None, ...])
        gt_j3d.append(gt_j3d_[None, ...])
        gt_hms.append(data['hms'][None, ...])

        inps = torch.cat(inps, dim=0).cuda()
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            outputs = model(inps)

        pred_j3ds = outputs['j3d'].detach()
        preds_hms = outputs['hms'].detach()
        pred_j2ds = get_j2d_from_hms(cfg, preds_hms)
        
        gt_j3ds = torch.cat(gt_j3d, dim=0).detach()
        gt_hms = torch.cat(gt_hms, dim=0).detach()
        gt_hm_j2ds = get_j2d_from_hms(cfg, gt_hms)

        representation = outputs['representation']
        representation_image = create_image(representation)

        T, B, N, C = inps.shape

        for i in range(T):
            gt_j3d = gt_j3ds[i]
            gt_hm = gt_hms[i]
            gt_hm_j2d = gt_hm_j2ds[i]
            
            pred_j3d = pred_j3ds[i]
            pred_j2d = pred_j2ds[i]
            pred_hm = preds_hms[i]

            inp = representation_image
            inp = inp[i]
            grid = torchvision.utils.make_grid(inp)
            inp = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).detach()
           

            pred_hm_image = plot_heatmaps(inp, pred_hm)    
            gt_hm_image = plot_heatmaps(inp, gt_hm)

            inp = inp.astype(np.uint8)

            inp_w_gt_hm_j2d = Skeleton.draw_2d_skeleton(inp, gt_hm_j2d, lines=True)
            inp = Skeleton.draw_2d_skeleton(inp, pred_j2d, lines=True)
                    
            color = generate_skeleton_image(gt_j3d, pred_j3d)
            color = color[..., ::-1]
            
            color = cv2.resize(color, (400, 300))
            inp = cv2.resize(inp, (400, 300))
            pred_hm_image = cv2.resize(pred_hm_image, (400, 300))
            gt_hm_image = cv2.resize(gt_hm_image, (400, 300))
            inp_w_gt_hm_j2d = cv2.resize(inp_w_gt_hm_j2d, (400, 300))           
            
            hstack1 = np.concatenate([inp, color], axis=1)
            hstack2 = np.concatenate([gt_hm_image, pred_hm_image], axis=1)
            hstack3 = np.concatenate([inp_w_gt_hm_j2d, np.zeros_like(inp_w_gt_hm_j2d)], axis=1)
            
            vstack = np.concatenate([hstack1, hstack2, hstack3], axis=0)
            video.write(vstack)

    video.release()


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
