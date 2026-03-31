import os
import numpy as np
import hashlib
import h5py
import math
import traceback
import functools

from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
import torch
import cv2
from EventEgoPoseEstimation.dataset import transforms
import torch.nn.functional as F
import ocam
import kornia
                                                     
h5py_File = functools.partial(h5py.File, libver='latest', swmr=True)


def generate_path_split(data_path: Path, cfg):
    if os.path.exists(data_path / 'train.txt') and os.path.exists(data_path / 'val.txt') and os.path.exists(data_path / 'test.txt'):
        return
    
    folders = []
    for folder in natsorted(os.listdir(data_path)):
        if cfg.DATASET.TYPE == 'Real' and not os.path.isfile(data_path / folder / 'synced_local_pose_gt.pickle'):
            continue
                
        if os.path.isfile(data_path / folder / 'event_meta.json'):
            folders.append(folder)
    
    total_dataset_len = len(folders)

    train_len = math.ceil(total_dataset_len * cfg.DATASET.TRAIN_TEST_SPLIT)
    test_len = total_dataset_len - train_len

    val_len = test_len // 2
    
    train_folders = natsorted(folders[:train_len])
    rest_folders = natsorted(folders[train_len:])
    
    val_folders = rest_folders[:val_len]
    test_folders = rest_folders[val_len:]
    
    print('Train folders => ', ', '.join(train_folders))
    print('Val folders => ', ', '.join(val_folders))
    print('Test folders => ', ', '.join(test_folders))

    train_txt = data_path / 'train.txt'
    with open(train_txt, 'w') as f:
        for folder in train_folders:
            f.write(folder + '\n')

    test_txt = data_path / 'val.txt'
    with open(test_txt, 'w') as f:
        for folder in val_folders:
            f.write(folder + '\n')
        
    test_txt = data_path / 'test.txt'
    with open(test_txt, 'w') as f:
        for folder in test_folders:
            f.write(folder + '\n')


def generate_indices(dataset_root, datasets, shuffle=False, make_equal_len=False):
    if not len(datasets):
        print('No datasets found')
        return []
    
    max_dataset_len = 0
    dataset_string = f'shuffle_{shuffle}_equal_len_{make_equal_len}'
    for dataset in datasets:
        dataset_string += str(dataset.data_path) + str(len(dataset)) 
        max_dataset_len = max(max_dataset_len, len(dataset))
    
    if make_equal_len:
        lengths = max_dataset_len * len(datasets)
    else:
        lengths = sum([len(dataset) for dataset in datasets])
            
    dataset_hash = hashlib.md5(dataset_string.encode('utf-8')).hexdigest()
    indices_path = dataset_root / 'indices' 
    if indices_path.exists() is False:
        indices_path.mkdir()

    indices_path = indices_path / f'{dataset_hash}.h5'    
    
    if os.path.exists(indices_path):
        try:
            h5_file = h5py_File(indices_path, 'r')
        except:
            print(traceback.format_exc())
            print('Corrupted indices file, generating new indices')
            os.remove(indices_path)
            
            return generate_indices(dataset_root, datasets, shuffle, make_equal_len)
        
        if 'indices' in h5_file and len(h5_file['indices']):
            indices = h5_file['indices']

            if len(indices) == lengths:
                print('loading indices from file')

                if len(indices) < 128 ** 5: # small dataset, store in memory.
                    indices = np.array(indices)

                return indices

        h5_file.close()


    h5_file = h5py.File(indices_path, 'w')

    rng = np.random.default_rng()

    create_dataset = True
    for idx, dataset in enumerate(tqdm(datasets, desc='Generating indices')):
        dataset_len = len(dataset)

        container_space = np.iinfo(np.uint32).max - 1 - dataset_len
        if container_space < 0:
            assert f'Not enough space in datatype for dataset: {dataset.data_path}, of length {dataset_len}'

        if make_equal_len:
            sample_indices = [np.arange(0, dataset_len, dtype=np.uint32) for _ in range(0, max_dataset_len, dataset_len)]            
            sample_idx = np.concatenate(sample_indices, axis=0)[:max_dataset_len]
            dataset_idx = np.ones(max_dataset_len, dtype=np.uint16) * idx
        else:
            sample_idx = np.arange(0, dataset_len, dtype=np.uint32)
            dataset_idx = np.ones(dataset_len, dtype=np.uint16) * idx
                
         # assert sample_idx is less than np.iinfo(np.uint32).max
        if shuffle:
            rng.shuffle(sample_idx)
            
        dataset_indices = np.concatenate([dataset_idx[..., np.newaxis],  
                                          sample_idx[..., np.newaxis]], axis=-1)
        
        if create_dataset:
            h5_file.create_dataset('indices', data=dataset_indices, 
                                   chunks=True, 
                                   maxshape=(None, dataset_indices.shape[1]))
            create_dataset = False
        else:
            indices = h5_file['indices']

            # Determine the shape of the existing dataset
            existing_shape = indices.shape
            # Determine the shape of the new data
            new_shape = dataset_indices.shape
            # Adjust the shape of the existing dataset to accommodate the new data
            indices.resize(existing_shape[0] + new_shape[0], axis=0)
            # Append the new data to the existing dataset
            indices[existing_shape[0]:] = dataset_indices
    
    h5_file.close()

    return generate_indices(dataset_root, datasets, shuffle, make_equal_len)

def save_augmented_data(data, path):
    # Assuming data is a torch tensor of shape (20, 192, 256)
    # Collapse the temporal bins for each polarity:
    data = data * 255  # Scale to [0, 255]
    positive_channel = data[:10].sum(dim=0)  # Shape: (192, 256)
    negative_channel = data[10:].sum(dim=0)  # Shape: (192, 256)
    
    # (Optional) Clamp or normalize if necessary:
    # For example, if you want values in the range [0, 255]:
    positive_channel = positive_channel.clamp(0, 255)
    negative_channel = negative_channel.clamp(0, 255)
    
    # Convert to numpy arrays (if data is on GPU, .cpu() is needed):
    positive_channel = positive_channel.cpu().numpy().astype(np.uint8)
    negative_channel = negative_channel.cpu().numpy().astype(np.uint8)
    
    # Get spatial dimensions (192, 256)
    # h, w = data.shape[:2]
    h, w = positive_channel.shape
    
    # Add channel dimension so each becomes (h, w, 1)
    positive_channel = positive_channel[..., None]
    negative_channel = negative_channel[..., None]
    
    # Create a blank channel (you can use it for visualization or as a separator)
    blank_channel = np.zeros((h, w, 1), dtype=np.uint8)
    
    # Concatenate to form a three-channel image: 
    # [positive, blank, negative]
    image = np.concatenate([positive_channel, blank_channel, negative_channel], axis=2)

    # return image
    
    # Save the image using OpenCV
    cv2.imwrite(path, image)

def save_mask_image(mask, path):
    """
    Saves a mask tensor as an image.
    
    Parameters:
        mask (torch.Tensor or np.ndarray): Mask with shape (1, H, W) or (H, W).
            Expected to contain binary values (0 or 1) or booleans.
        path (str): Path where the image will be saved.
    """
    # If mask is a torch tensor, move to CPU and convert to numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # If mask has an extra channel dimension (1, H, W), squeeze it
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    
    # Convert mask to uint8. If it's binary, multiply by 255.
    if mask.dtype != np.uint8:
        # Check if the mask is binary or boolean and convert accordingly:
        if mask.dtype == np.bool_ or np.array_equal(np.unique(mask), [0, 1]):
            mask = (mask.astype(np.uint8)) * 255
        else:
            mask = mask.astype(np.uint8)
    
    # Save the mask image using OpenCV
    cv2.imwrite(path, mask)


def random_dropout(x, p):
    mask = (torch.rand_like(x) > p).float()
    return x * mask

def flip_axis(x, axis):
    import pdb; pdb.set_trace()

    x = x.permute(axis, *[i for i in range(x.dim()) if i != axis])  # Swap `axis` with the first dim
    x = torch.flip(x, [0])  # Flip along the first dim
    x = x.permute(*[i for i in range(1, x.dim()) if i != axis], 0)  # Swap back to original order
    
    return x.clone()  # Ensures a new tensor is returned

def event_augmentation(self, event_voxel, temporal_step, augmentation_data):
    bg_data_list = augmentation_data['bg_data'][temporal_step]
    bg_mask_list = augmentation_data['bg_mask'][temporal_step]

    event_voxels = []
    target_width = self.width
    target_height = self.height

    for index, (bg_data, bg_mask_tensor) in enumerate(zip(bg_data_list, bg_mask_list)):
        if type(bg_data) == list:
            voxel = event_voxel[index]
            event_voxels.append(voxel)
            continue

        bg_mask = bg_mask_tensor.cpu().numpy()
        bg_mask = cv2.dilate(bg_mask, np.ones((2, 2)), iterations=1)
        bg_mask = ~bg_mask.astype(bool)

        voxel = event_voxel[index]
        # save_mask_image(bg_mask, f'/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/opt_augmented/mask_{index}.png')
        mask = bg_mask.squeeze(0)
        voxel[:10, mask] += bg_data[:, :, 0][mask] # for all bins positive events
        voxel[10:, mask] += bg_data[:, :, 1][mask] # for all bins negative events
        # voxel = voxel.clamp(0, 1) # causing problems in backpropagation
        event_voxels.append(voxel)

    if len(event_voxels) != 0:
        out = torch.stack(event_voxels, dim=0)
        add_mask = (torch.rand(target_height, target_width, 20) > 0.9995).float()
        add_mask = add_mask.unsqueeze(0).permute(0, 3, 1, 2)  # [1, 20, 192, 256]
        add_mask = add_mask.expand(out.shape[0], -1, -1, -1).cuda()  # [2, 20, 192, 256]
        out = out + add_mask

        if np.random.rand() > 0.5:
            out = random_dropout(out, np.random.rand() * 0.1)

        # if np.random.random() < 0.5:
        #     out = flip_axis(out, -1)

        event_voxel = out

    return event_voxel.cuda()

def event_augmentation_v2(self, event_voxel, augmentation_data):

    if not augmentation_data:
        return event_voxel

    if torch.all(augmentation_data['use_bg'] == False):
        return event_voxel

    bg_mask = augmentation_data['bg_mask']

    T, B, C, W, H = bg_mask.shape

    bg_data = augmentation_data['bg_data']
    # bg_mask = bg_mask.view(-1, 1, W, H)
    # use_bg = augmentation_data['use_bg'].view(-1)
    use_bg = augmentation_data['use_bg']

    # event_voxel = event_voxel.view(-1, event_voxel.shape[2], W, H)

    temporal_bins = event_voxel.shape[2]

    kernel = torch.ones((2,2), device=event_voxel.device)
    filtered_masks = bg_mask[use_bg]

    dilated_masks = kornia.morphology.dilation(filtered_masks, kernel).expand(-1, temporal_bins, -1, -1)
    
    

    # for i in range(0, dilated_masks.shape[0]):
    #     save_mask_image(dilated_masks[i][0], f'/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/opt_augmented/mask_{i}.png')
    # save_mask_image(dilated_masks[0][0], f'/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/opt_augmented/mask_1.png')
    # save_mask_image(dilated_masks[10][0], f'/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/opt_augmented/mask_2.png')

    event_voxel[use_bg] = event_voxel[use_bg] * dilated_masks + bg_data[use_bg] * (1 - dilated_masks)

    # Random noise augmentation
    target_height, target_width = event_voxel.shape[-2:]
    # add_mask = (torch.rand(event_voxel.shape[0], event_voxel.shape[1], temporal_bins, target_height, target_width, device=event_voxel.device) > 0.9995).float()

    
    # Noise addition
    # event_voxel[use_bg, :, :, :] = event_voxel[use_bg, :, :, :] + add_mask[use_bg, :, :, :]
    # event_voxel[use_bg] = event_voxel[use_bg] + add_mask[use_bg]
    
    # Random dropout
    # if np.random.rand() > 0.5:
    #     dropout_rate = np.random.rand() * 0.1
    #     event_voxel = random_dropout(event_voxel, dropout_rate)
    
    # event_voxel = event_voxel.view(T, B, temporal_bins, W, H)

    return event_voxel

def is_empty(item):
    if isinstance(item, torch.Tensor):
        return item.numel() == 0  # True if the tensor has zero elements
    else:
        return len(item) == 0     # Works for lists and other sequences

def camera_to_j2d_batch(gt_j3d, image_size, ocam_model):

    h, w = ocam_model['height'], ocam_model['width']
    gt_j3d = gt_j3d.clone()
    gt_j3d[:, :, 2] *= -1

    point_2Ds = ocam.world2cam_torch_batch(gt_j3d[:, :, None, :], ocam_model)[:, :, 0, :]
    point_2Ds[:, :, 1] = h - point_2Ds[:, :, 1]
    
    width, height = image_size
    sx = width / w
    sy = height / h

    point_2Ds[:, :, 0] *= sx
    point_2Ds[:, :, 1] *= sy
    
    return point_2Ds.unsqueeze(1)

    # # width, height = config.MODEL.IMAGE_SIZE   
    # # width, height = (256, 256)
    # # sx = width / w
    # # sy = height / h

    # # point_2Ds[:, :, 0] *= sx
    # # point_2Ds[:, :, 1] *= sy

    # point_2Ds[:, :, 0] = point_2Ds[:, :, 0] / w
    # point_2Ds[:, :, 1] = point_2Ds[:, :, 1] / h

    # in_fov = (
    #     (point_2Ds[..., 0] > 0)
    #     & (point_2Ds[..., 1] > 0)
    #     & (point_2Ds[..., 0] < 1)
    #     & (point_2Ds[..., 1] < 1)
    # )

    # point_2Ds = point_2Ds.clamp(min=0.0, max=1.0)
    
    # return point_2Ds.unsqueeze(1), in_fov.unsqueeze(1)