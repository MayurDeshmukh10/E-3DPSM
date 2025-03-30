import numpy as np
import torch
from EventEgoPoseEstimation.utils.vis import save_pose_images, save_debug_images, save_debug_3d_joints, save_debug_segmenation, save_debug_eros, generate_skeleton_image, dump_sketelon_image
from EventEgoPoseEstimation.dataset.dataset_utils import event_augmentation, save_augmented_data


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(gt3ds, preds, valid_j3d):
    if isinstance(gt3ds, torch.Tensor):
        gt3ds = gt3ds.detach()
    if isinstance(preds, torch.Tensor):
        preds = preds.detach()
        
    if isinstance(valid_j3d, torch.Tensor):
        valid_j3d = valid_j3d.detach()

    gt3ds1 = gt3ds * valid_j3d
    preds1 = preds * valid_j3d

    # cnt = np.sum(valid_j3d)
    cnt = torch.sum(valid_j3d)
    if cnt == 0:
        return 0, 0
    
    # joint_error = np.sqrt(np.sum((gt3ds1 - preds1)**2, axis=-1))

    joint_error = torch.sqrt(torch.sum((gt3ds1 - preds1) ** 2, dim=-1))
    # joint_error = np.sum(joint_error) / cnt
    joint_error = torch.sum(joint_error) / cnt
    
    return joint_error, cnt


def accuracy_test(gt3ds, preds, valid_j3d):
    if isinstance(gt3ds, torch.Tensor):
        gt3ds = gt3ds.detach()
    if isinstance(preds, torch.Tensor):
        preds = preds.detach()
        
    if isinstance(valid_j3d, torch.Tensor):
        valid_j3d = valid_j3d.detach()

    gt3ds1 = gt3ds * valid_j3d
    preds1 = preds * valid_j3d

    # cnt = np.sum(valid_j3d)
    # cnt = torch.sum(valid_j3d)
    # if cnt == 0:
    #     return 0, 0
    
    # joint_error = np.sqrt(np.sum((gt3ds1 - preds1)**2, axis=-1))

    joint_error = torch.sqrt(torch.sum((gt3ds1 - preds1) ** 2, dim=-1))

    # joint_error = np.sum(joint_error) / cnt
    # joint_error = torch.sum(joint_error) / cnt
    joint_error = torch.mean(joint_error, axis=1)
    
    return joint_error, 10

import cv2
import numpy as np

def create_concatenated_image(color, voxel_image):
    # # Resize color image to match the height of voxel_image
    # target_height = voxel_image.shape[0]  # 192
    # scale_factor = target_height / color.shape[0]
    # target_width = int(color.shape[1] * scale_factor)

    # resized_color = cv2.resize(color, (target_width, target_height))

    # # Convert images to the same data type if needed
    # if resized_color.dtype != voxel_image.dtype:
    #     if np.issubdtype(voxel_image.dtype, np.floating):
    #         resized_color = resized_color.astype(voxel_image.dtype)
    #     else:
    #         voxel_image = voxel_image.astype(resized_color.dtype)

    # # Concatenate the images horizontally
    # concatenated_image = np.hstack((resized_color, voxel_image))

    target_height = color.shape[0]  # 1080
    scale_factor = target_height / voxel_image.shape[0]
    target_width = int(voxel_image.shape[1] * scale_factor)

    resized_voxel_image = cv2.resize(voxel_image, (target_width, target_height))

    # Convert images to the same data type if needed
    if color.dtype != resized_voxel_image.dtype:
        if np.issubdtype(color.dtype, np.floating):
            resized_voxel_image = resized_voxel_image.astype(color.dtype)
        else:
            color = color.astype(resized_voxel_image.dtype)

    # Concatenate the images horizontally
    concatenated_image = np.hstack((color, resized_voxel_image))

    return concatenated_image

def accuracy_with_vis(gt3ds, preds, valid_j3d, batch_idx, pred_abs_poses_t, gt_j3d_t, input, pose_filename, frame_indexes):
    if isinstance(gt3ds, torch.Tensor):
        gt3ds = gt3ds.detach()
    if isinstance(preds, torch.Tensor):
        preds = preds.detach()
        
    if isinstance(valid_j3d, torch.Tensor):
        valid_j3d = valid_j3d.detach()

    gt3ds1 = gt3ds * valid_j3d
    preds1 = preds * valid_j3d

    # cnt = np.sum(valid_j3d)
    cnt = torch.sum(valid_j3d)
    if cnt == 0:
        return 0, 0
    
    # joint_error = np.sqrt(np.sum((gt3ds1 - preds1)**2, axis=-1))
    joint_error = torch.sqrt(torch.sum((gt3ds1 - preds1) ** 2, dim=-1))

    for i in range(joint_error.shape[0]):
        avg_error = joint_error[i].sum() / 16
        if avg_error > 500:
            for j in range(pred_abs_poses_t.size(0)):
                color = dump_sketelon_image(gt_j3d_t[j][i].detach(), pred_abs_poses_t[j][i].detach(), f"./visualizations/new_dataloader")
                voxel = input[j][i]
                voxel_image = save_augmented_data(voxel, 'test')
                concatenated_image = create_concatenated_image(color, voxel_image)
                filename = pose_filename[j][i]
                frame_index = frame_indexes[j][i]
                # output_path = './visualizations/new_dataloader_val_debug/train/{avg_err:.3f}_bat_{batch_idx}_{file}_ts_{ts}.png'.format(avg_err=avg_error.item(), batch_idx=batch_idx, file=filename, ts=j)
                output_path = './visualizations/new_dataloader_val_debug/test/bat_{batch_idx}_fi_{frame_index}_{file}_ts_{ts}_{avg_err:.3f}.png'.format(avg_err=avg_error.item(), batch_idx=batch_idx, file=filename, ts=j, frame_index=frame_index)
                # output_path = f"./visualizations/new_dataloader_val_debug/test/{avg_error.item()}_{batch_idx}_temporal_step_{j}.png"
                cv2.imwrite(output_path, concatenated_image)
                break


    # joint_error = np.sum(joint_error) / cnt
    joint_error = torch.sum(joint_error) / cnt
    
    return joint_error, cnt


def root_accuracy(gt3ds, preds, valid_j3d):
    if isinstance(gt3ds, torch.Tensor):
        gt3ds = gt3ds.detach().detach()
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().detach()
        
    if isinstance(valid_j3d, torch.Tensor):
        valid_j3d = valid_j3d.detach().detach()

    if len(valid_j3d.shape) == 1:
        valid_j3d = valid_j3d[..., None, None]

    gt3ds = gt3ds * valid_j3d
    preds = preds * valid_j3d
    
    cnt = np.sum(valid_j3d)
    if cnt == 0:
        return 0, 0

    joint_error = np.sqrt(np.sum((gt3ds - preds)**2, axis=-1))
    joint_error = np.sum(joint_error) / cnt

    return joint_error, cnt