import cv2
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
from EventEgoPoseEstimation.dataset import transforms

from EventEgoPoseEstimation.dataset.metrics import compute_3d_errors_batch


logger = logging.getLogger(__name__)


class Joints3DDataset(Dataset):
    def __init__(self, cfg, root, is_train, temporal_bins):
        super().__init__()
    
        self.is_train = is_train
        self.root = root

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA

        self.temporal_bins = temporal_bins

        self.cfg = cfg
        self.db = []

    def __len__(self,):
        raise NotImplementedError

    def transform(self, data, anno, kwargs):     
        j2ds = []
        j3ds = []
        vis_j2ds = []
        vis_j3ds = []
        valid_j3ds = []
        frame_indexes = []
        rgb_frame_indexes = []
        scales_x = []
        scales_y = []
        valid_segs = []
        ego_to_global_spaces = []
        segmentation_masks = []
        targets = []

        for i in range(self.temporal_bins):
            rgb_frame_index = anno['rgb_frame_index'][i]
        
        
            # coord_x = data['coord_x']
            # coord_y = data['coord_y']
            # segmentation_indices = data['segmentation_indices']
            
            segmentation_mask = anno['segmentation_mask'][i]

            ego_to_global_space = anno['ego_to_global_space'][i]
            j3d = anno['j3d'][i]
            j2d = anno['j2d'][i]
            vis_j2d = anno['vis_j2d'][i]
            vis_j3d = anno['vis_j3d'][i]
            valid_seg = float(anno['valid_seg'][i])
        
            img_h, img_w = segmentation_mask.shape[:2]

            target_width = self.image_size[0]
            target_height = self.image_size[1]
            
            sx = target_width / img_w
            sy = target_height / img_h

            segmentation_mask = cv2.resize(segmentation_mask, (int(target_width), int(target_height)), interpolation=cv2.INTER_AREA)
            segmentation_mask = torch.from_numpy(segmentation_mask).float().unsqueeze(0)

            # inp_tensor = torch.clamp_(inp_tensor, 0, 1)
            segmentation_mask = torch.clamp_(segmentation_mask, 0, 1)
        
            j2d[:, 0] *= sx
            j2d[:, 1] *= sy

        



        # if "augment" in kwargs and kwargs['augment'] is True:                        
        #     add_mask = (np.random.rand(target_height, target_width, 2) > 0.9995).astype(np.float32)
        #     add_mask = cv2.dilate(add_mask, np.ones((2, 2)))
        #     inp = inp + add_mask
            
        #     if "A" in kwargs:
        #         A = kwargs['A']            
        #         inp = transforms.crop(inp, A, [target_width, target_height])
        #         segmentation_mask = transforms.crop(segmentation_mask, A, [target_width, target_height])
        #         j2d = transforms.affine_transform_pts(j2d, A)
        
        #     if np.random.rand() > 0.5:
        #         inp = transforms.random_dropout(inp, np.random.rand() * 0.1)
            
        #     if "flip_axis" in kwargs and kwargs['flip_axis'] is True:
        #         inp = transforms.flip_axis(inp, -1)
            
        #     if "flip_lr" in kwargs and kwargs['flip_lr'] is True:
        #         inp = transforms.flip_lr(inp)
        #         segmentation_mask = transforms.flip_lr(segmentation_mask)
        #         j2d = transforms.flip_lr_joints(inp, j2d)
                
        # inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).float()
        # segmentation_mask = torch.from_numpy(segmentation_mask).float().unsqueeze(0)

        # # inp_tensor = torch.clamp_(inp_tensor, 0, 1)
        # segmentation_mask = torch.clamp_(segmentation_mask, 0, 1)

        
        # inp_h, inp_w = inp.shape[:2]

        # invalid_j2d = (j2d[:, 0] < 0) + (j2d[:, 1] < 0) + (j2d[:, 0] >= inp_w) + (j2d[:, 1] >= inp_h)
        # valid_j2d = 1 - invalid_j2d[:, None]
            valid_j2d = np.ones_like(j2d)

            # During validation, the network should learn a prior for the occluded joints.
            if self.is_train is False:
                if vis_j3d.mean() > 0:
                    vis_j3d = np.ones_like(vis_j3d)
                else:
                    vis_j3d = np.zeros_like(vis_j3d)

                if vis_j2d.mean() > 0:
                    vis_j2d = np.ones_like(vis_j2d)
                else:
                    vis_j2d = np.zeros_like(vis_j2d)
                
            vis_j2d = vis_j2d * valid_j2d
            target, vis_j2d = self.generate_target(j2d, vis_j2d)


            if self.is_train: # During training, we apply weights to the joints.
                heatmap_sequence = {"Head": 1, # 0
                                    "Neck": 1, # 1
                                    "Right_shoulder": 1, # 2 
                                    "Right_elbow": 1.5, # 3
                                    "Right_wrist": 1.5, # 4
                                    "Left_shoulder": 1, # 5
                                    "Left_elbow": 1.5, # 6
                                    "Left_wrist": 1.5, # 7
                                    "Right_hip": 1, # 8
                                    "Right_knee": 2, # 9
                                    "Right_ankle": 2, # 10
                                    "Right_foot": 2, # 11
                                    "Left_hip": 1, # 12 
                                    "Left_knee": 2, # 13
                                    "Left_ankle": 2, #14
                                    "Left_foot": 2} # 15

                weight = [w for _, w in heatmap_sequence.items()]
                weight = np.array(weight)[:, None]
                vis_j2d = vis_j2d * weight
                vis_j3d = vis_j3d * weight
        
            target = torch.from_numpy(target)
            j3d = torch.from_numpy(j3d)
            j2d = torch.from_numpy(j2d)
                    
            vis_j2d = torch.from_numpy(vis_j2d)
            vis_j3d = torch.from_numpy(vis_j3d)

            j2ds.append(j2d)
            j3ds.append(j3d)
            vis_j2ds.append(vis_j2d[:, :1])
            vis_j3ds.append(vis_j2d[:, :1])
            valid_j3ds.append(vis_j3d[:, :1])
            rgb_frame_indexes.append(rgb_frame_index)
            scales_x.append(sx)
            scales_y.append(sy)
            valid_segs.append(valid_seg)
            ego_to_global_spaces.append(ego_to_global_space)
            segmentation_masks.append(segmentation_mask)
            targets.append(target)


        inp = data['input']
        frame_indexes = data['frame_index']

        # meta = {
        #     'j2d': j2d,
        #     'j3d': j3d,
        #     'vis_j2d': vis_j2d[:, :1],
        #     'vis_j3d': vis_j2d[:, :1],
        #     'valid_j3d': vis_j3d[:, :1],
        #     'frame_index': frame_index,
        #     'rgb_frame_index': rgb_frame_index,
        #     'scale_x': sx,
        #     'scale_y': sy,
        #     'valid_seg': valid_seg,
        #     'ego_to_global_space': ego_to_global_space,
        # }
        j2ds = torch.stack(j2ds, dim=0)
        j3ds = torch.stack(j3ds, dim=0)
        vis_j2ds = torch.stack(vis_j2ds, dim=0)
        vis_j3ds = torch.stack(vis_j3ds, dim=0)
        valid_j3ds = torch.stack(valid_j3ds, dim=0)
        segmentation_masks = torch.stack(segmentation_masks, dim=0)
        targets = torch.stack(targets, dim=0)
        

        meta = {
            'j2d': j2ds,
            'j3d': j3ds,
            'vis_j2d': vis_j2ds,
            'vis_j3d': vis_j3ds,
            'valid_j3d': valid_j3ds,
            'frame_index': frame_indexes,
            'rgb_frame_index': rgb_frame_indexes,
            'scale_x': scales_x,
            'scale_y': scales_y,
            'valid_seg': valid_segs,
            'ego_to_global_space': ego_to_global_spaces,
        }

        return {'x': inp, 'hms': targets, 'weight': vis_j2ds, 'j3d': j3ds, 'j2d': j2ds, 'segmentation_mask': segmentation_masks}, meta
        # return {'x': inp, 'hms': target, 'weight': vis_j2d, 'j3d': j3d, 'j2d': j2d, 'segmentation_mask': segmentation_mask}, meta

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            feat_stride = self.image_size / self.heatmap_size

            
            for joint_id in range(self.num_joints):
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    @classmethod
    def evaluate_dataset(cls, cfg, frame_indices, all_gt_j3ds, all_preds_j3d, all_vis_j3d):
        sequence_range = {
            'walk': [0, 3500],
            'crouch': [3500, 6500],
            'pushup': [6500, 9000],
            'boxing': [9000, 12350],
            'kick': [12350, 15200],
            'dance': [15200, 17800],
            'inter. with env': [17800, 20700],
            'crawl': [20700, 23800],
            'sports': [23800, 33000],
            'jump': [33000, 200000], # max frame index
        }
                               
        MPJPE = []
        PAMPJPE = []
        for seq_name, seq_range in sequence_range.items():
            start_index, end_index = seq_range
            
            current_seq_indices = frame_indices >= start_index             
            current_seq_indices = np.logical_and(current_seq_indices, frame_indices < end_index)
            
            try:
                gt_j3ds = all_gt_j3ds[current_seq_indices]
                preds_j3d = all_preds_j3d[current_seq_indices]
                vis_j3d = all_vis_j3d[current_seq_indices]
            except:
                print(f"Warning: Mismatch in array dimensions. frame_indices has {len(frame_indices)}, but all_preds_j3d has {len(all_preds_j3d)}.")
                min_length = min(len(all_preds_j3d), len(frame_indices))
                # frame_indices = frame_indices[:min_length]
                preds_j3d = all_preds_j3d[:min_length]
                gt_j3ds = all_gt_j3ds[:min_length]
                vis_j3d = all_vis_j3d[:min_length]
            
            errors, errors_pa = compute_3d_errors_batch(gt_j3ds, preds_j3d, vis_j3d)

            seq_mpjpe = np.mean(errors) 
            seq_pampjpe = np.mean(errors_pa)
        
            MPJPE.append(seq_mpjpe)
            PAMPJPE.append(seq_pampjpe)
        
        name_values = []           
        for i, seq_name in enumerate(sequence_range.keys()):
            name_values.append((f'{seq_name}_MPJPE', MPJPE[i]))
        name_values.append(('MPJPE', np.mean(MPJPE)))

        for i, seq_name in enumerate(sequence_range.keys()):
            name_values.append((f'{seq_name}_PAMPJPE', PAMPJPE[i]))
        name_values.append(('PAMPJPE', np.mean(PAMPJPE)))

        name_values = OrderedDict(name_values)
                
        return name_values, MPJPE
    
    @classmethod
    def evaluate_joints(cls, cfg, all_gt_j3ds, all_preds_j3d, all_vis_j3d):
        min_length = min(len(all_preds_j3d), len(all_gt_j3ds), len(all_vis_j3d))
        all_gt_j3ds = all_gt_j3ds[:min_length]
        all_preds_j3d = all_preds_j3d[:min_length]
        all_vis_j3d = all_vis_j3d[:min_length]
        errors, errors_pa = compute_3d_errors_batch(all_gt_j3ds, all_preds_j3d, all_vis_j3d)
        
        MPJPE = np.mean(errors)
        PAMPJPE = np.mean(errors_pa)

        # TODO: Remove try-except block if this works.
        try:
            MPJPE_std = np.std(errors)
            PAMPJPE_std = np.std(errors_pa)
        except Exception as e:
            print(f"Error: {e}")
            MPJPE_std = 0
            PAMPJPE_std = 0

        name_values = []

        heatmap_sequence = ["Head", # 0
                            "Neck", # 1
                            "Right_shoulder", # 2 
                            "Right_elbow", # 3
                            "Right_wrist", # 4
                            "Left_shoulder", # 5
                            "Left_elbow", # 6
                            "Left_wrist", # 7
                            "Right_hip", # 8
                            "Right_knee", # 9
                            "Right_ankle", # 10
                            "Right_foot", # 11
                            "Left_hip", # 12 
                            "Left_knee", # 13
                            "Left_ankle", #14
                            "Left_foot"] # 15

        for i, joint_name in enumerate(heatmap_sequence):
            name_values.append((f'{joint_name}_MPJPE', errors[i]))
        name_values.append(('MPJPE', MPJPE))
        name_values.append(('MPJPE_std', MPJPE_std))

        for i, joint_name in enumerate(heatmap_sequence):
            name_values.append((f'{joint_name}_PAMPJPE', errors_pa[i]))
        name_values.append(('PAMPJPE', PAMPJPE))
        name_values.append(('PAMPJPE_std', PAMPJPE_std))

        name_values = OrderedDict(name_values)

        return name_values, MPJPE

    def generate_location_maps(self, j2d, j3d, vis_j3d):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target = np.zeros((self.num_joints,
                           self.heatmap_size[1],
                           self.heatmap_size[0], 3), 
                           dtype=np.float32)

        tmp_size = self.sigma * 3
        feat_stride = self.image_size / self.heatmap_size

        
        for joint_id in range(self.num_joints):
            mu_x = int(j2d[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(j2d[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                vis_j3d[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = vis_j3d[joint_id, 0]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]][..., None]
                
                target[joint_id, :, :, 0] *= j3d[joint_id, 0]
                target[joint_id, :, :, 1] *= j3d[joint_id, 1]
                target[joint_id, :, :, 2] *= j3d[joint_id, 2]

        target = torch.from_numpy(target).permute(0, 3, 1, 2) 
       
        return target, vis_j3d