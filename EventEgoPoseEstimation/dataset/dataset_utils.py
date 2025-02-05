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

def collate_variable_size(batch):
    
    final_data_list = []
    final_meta_list = []
    # Iterate over the batch of tuples

    # print("batch len : ", len(batch))
    for i, items in enumerate(batch):
        data_list = []
        meta_list = []
        events = []
        if type(items) is tuple:
            items = [items] 
        # print("len ", type(items))
        for item in items:
            # print('item', item[0].shape)
            data_list.append(item[0])
            meta_list.append(item[1])
    
        # print('data_list', len(data_list))
        for _, data in enumerate(data_list):
            # ev = np.concatenate([data['x'], i*np.ones((len(data['x']),1), dtype=np.float32)],1)
            # ev = np.hstack([data['x'], i * np.ones((len(data['x']), 1), dtype=np.float32)])
            events.append(data['x'])

        # x_padded = pad_sequence([data['x'] for data in data_list], batch_first=True)
        
        # batch_size, max_len, feature_dim = x_padded.shape

        # Create a column of indices for each sequence in the batch
        # indices = torch.arange(1, batch_size+1, dtype=torch.float32).view(-1, 1, 1).expand(-1, max_len, 1)
        # indices = torch.arange(0, batch_size, dtype=torch.float32).view(-1, 1, 1).expand(-1, max_len, 1)

        # x_padded = torch.cat([x_padded, indices], dim=2)  # Concatenate along the feature dimension
        # x_padded = x_padded.reshape(-1, 5)
        # print(x_padded)

        # events = torch.from_numpy(np.concatenate(events,0))
        # events = torch.from_numpy(np.vstack(events))
        # events = torch.from_numpy(events)
        # for item in data_list:
        #     for x in item['hms']:
        #         print('item', x.shape)
        
        collated_data = {
            'x': events,
            'hms': torch.stack([item['hms'] for item in data_list]),
            'weight': torch.stack([item['weight'] for item in data_list]),
            'j3d': torch.stack([item['j3d'] for item in data_list]),
            'j2d': torch.stack([item['j2d'] for item in data_list]),
            'segmentation_mask': torch.stack([item['segmentation_mask'] for item in data_list])
        }

        # Collate the meta part
        collated_meta = default_collate(meta_list)
        final_data_list.append(collated_data)
        final_meta_list.append(collated_meta)

    # ego_to_global_space_np = np.array([item['ego_to_global_space'] for item in meta_list])
    # ego_to_global_space = torch.tensor(ego_to_global_space_np)
    # collated_meta = {
    #     'j2d': torch.stack([item['j2d'] for item in meta_list]),
    #     'j3d': torch.stack([item['j3d'] for item in meta_list]),
    #     'vis_j2d': torch.stack([item['vis_j2d'] for item in meta_list]),
    #     'vis_j3d': torch.stack([item['vis_j3d'] for item in meta_list]),
    #     'valid_j3d': torch.stack([item['valid_j3d'] for item in meta_list]),
    #     'frame_index': torch.tensor([item['frame_index'] for item in meta_list]),
    #     'rgb_frame_index': torch.tensor([item['rgb_frame_index'] for item in meta_list]),
    #     'scale_x': torch.tensor([item['scale_x'] for item in meta_list]),
    #     'scale_y': torch.tensor([item['scale_y'] for item in meta_list]),
    #     'valid_seg': torch.tensor([item['valid_seg'] for item in meta_list]),
    #     'ego_to_global_space': ego_to_global_space
    # }
    # return collated_data, collated_meta
    # print('final_data_list', len(final_data_list))
    # print('final_meta_list', len(final_meta_list))
    return final_data_list, final_meta_list

def create_image(representation):
   
    representation = representation.to(torch.uint8)

    B, h, w, _ = representation.shape

    r = representation[..., :1]  # Extract red channel
    b = representation[..., 1:]  # Extract blue channel
    g = torch.zeros((B, h, w, 1), dtype=torch.uint8, device=representation.device)

    representation = torch.cat([r, g, b], dim=-1).to(torch.uint8)

    representation = representation.permute(0, 3, 2, 1).to(torch.float32).cuda()
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
