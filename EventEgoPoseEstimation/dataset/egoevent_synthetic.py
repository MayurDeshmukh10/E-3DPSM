import sys; sys.path.append('../')
import json
import numpy as np
import json
import random
import logging
import cv2
import os
import glob


from EventEgoPoseEstimation.dataset.dataset_utils import h5py_File

from pathlib import Path
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class SyntheticEventStream(Dataset):
    def __init__(self, data_path, cfg, split, is_train, augmentation):
        super().__init__()

        self.data_path = Path(data_path)
        self.cfg = cfg
        self.split = split

        # split = 'train'

        with open(self.data_path / 'event_meta.json', 'r') as f:
            meta = json.load(f)
            
        self.height = meta['height']
        self.width = meta['width']
        
        self.stream_path = self.data_path / 'events.h5'
        self.fin = None 
        self.is_augmentation = augmentation

        self.batch_size = cfg.DATASET.EVENT_BATCH_SIZE
        self.max_frame_time = cfg.DATASET.SYNTHETIC.MAX_FRAME_TIME_IN_MS
        self.is_train = is_train

        if not self.is_augmentation:
            self.frame_offsets = np.load(f"/scratch/inf0/user/mdeshmuk/EE3D-S-frame-offsets/{split}/{self.data_path.name}.npy")
            valid_mask = self.frame_offsets > 0
            self.valid_indices = np.where(valid_mask)[0] # valid frame indices where there are events
            self.cumulative_offsets = np.cumsum(self.frame_offsets)
            self.cumulative_offsets = np.insert(self.cumulative_offsets, 0, 0)
            self.num_valid_frames = len(self.valid_indices)

        self.filename = self.data_path.name

        # self.chunk_size = 100000
        # self.current_chunk = None
        # self.current_chunk_start = 0
        # self.current_chunk_end = 0

    # def _read_chunk(self, start_idx):
    #     end_idx = start_idx + self.chunk_size
    #     self.current_chunk = self.fin[start_idx:end_idx]
    #     self.current_chunk_start = start_idx
    #     self.current_chunk_end = end_idx

    def init_stream(self):
        self.fin = h5py_File(self.stream_path, 'r')['event']
        
    def __len__(self):
        if self.is_augmentation:
        # if self.is_augmentation or self.split in ['val', 'test', 'train']:

            with h5py_File(self.stream_path, 'r') as f:
                return f['event'].shape[0] // self.batch_size

        return self.num_valid_frames #// 20 # Take every 10th frame
    
    # def get_event_batch(self, frame_idx):
    #     if self.is_augmentation or self.split in ['val', 'test']:
    #     # if self.is_augmentation:

    #         num_events = self.batch_size
    #         idx = frame_idx * num_events

    #         if self.is_train:
    #             max_frame_time = random.randint(2, self.max_frame_time)
    #         else:
    #             max_frame_time = self.max_frame_time

    #         frame_time = 0
    #         data_batches = []           
    #         while frame_time < max_frame_time: 

    #             if (self.current_chunk is None
    #             or idx < self.current_chunk_start
    #             or (idx + num_events) > self.current_chunk_end):
    #                 self._read_chunk(idx)

    #             data_batch = self.fin[idx: idx + num_events]

    #             data_batch = self.current_chunk[idx - self.current_chunk_start: idx - self.current_chunk_start + num_events]

    #             ts = data_batch[:, 2] 
                
    #             if not len(ts): 
    #                 break

    #             ts = (ts[-1] - ts[0]) * 1e-3 # microseconds to milliseconds 


    #             data_batches.append(data_batch)
                
    #             frame_time += ts
    #             idx += num_events

    #         if len(data_batches) == 0:
    #             raise StopIteration
            
    #         data_batches_np = np.concatenate(data_batches, axis=0)
        
    #         del data_batches

    #         return data_batches_np
        

    #     start = self.cumulative_offsets[frame_idx]
    #     end = self.cumulative_offsets[frame_idx + 1]
    #     # try:
    #     #     end = self.cumulative_offsets[frame_idx + 2] # combine 2 frames
    #     # except:
    #     #     end = self.cumulative_offsets[frame_idx + 1] # last frame

    #     data_batch = self.fin[start:end]

    #     # try:
    #     #     if frame_idx != data_batch[-1, 4]:
    #     #         import pdb; pdb.set_trace()
                
    #     #     # print("file name: ", self.filename)
    #     #     # print("frame ts max : ", data_batch[:, 2].max())
    #     #     # print("frame ts min : ", data_batch[:, 2].min())
    #     #     print("Uni : ", np.unique(data_batch[:, 2]))
    #     # except:
    #     #     print("Problem")

    #     out = np.array(data_batch)
    #     return out
    

    def get_event_batch(self, frame_idx):
        # if self.is_augmentation or self.split in ['val', 'train', 'test']: # remove train currently here for testing
        if self.is_augmentation:

            num_events = self.batch_size
            idx = frame_idx * num_events

            if self.is_train:
                max_frame_time = random.randint(10, self.max_frame_time)
            else:
                max_frame_time = self.max_frame_time

            frame_time = 0
            data_batches = []           
            while frame_time < max_frame_time: 
                data_batch = self.fin[idx: idx + num_events]
                ts = data_batch[:, 2] 
                
                if not len(ts): 
                    break

                ts = (ts[-1] - ts[0]) * 1e-3 # microseconds to milliseconds 

                data_batches.append(data_batch)
                
                frame_time += ts
                idx += num_events

            if len(data_batches) == 0:
                # print("batch zero")
                # print("Frame IDx", frame_idx)
                # print("Idx", idx)
                # print("")
                # mayur
                raise StopIteration
            
            data_batches_np = np.concatenate(data_batches, axis=0)
        
            del data_batches

            return data_batches_np, self.filename

        
        start = self.cumulative_offsets[frame_idx]

        # try:
        #     end = self.cumulative_offsets[frame_idx + 10]
        # except:
        end = self.cumulative_offsets[frame_idx + 1]
        data_batch = self.fin[start:end]

        # try:
        #     if frame_idx != data_batch[-1, 4]:
        #         import pdb; pdb.set_trace()
                
        #     # print("file name: ", self.filename)
        #     # print("frame ts max : ", data_batch[:, 2].max())
        #     # print("frame ts min : ", data_batch[:, 2].min())
        #     print("Uni : ", np.unique(data_batch[:, 2]))
        # except:
        #     print("Problem")

        out = np.array(data_batch)
        return out, self.filename

    
    def __getitem__(self, idx):
        if self.fin is None: self.init_stream() # Done to ensure multiprocessing works

        if isinstance(idx, tuple):
            idx, kwargs = idx
        else:
            kwargs = {}

        actual_frame_idx = self.valid_indices[idx]
        # actual_frame_idx = idx # for time window appoarch
        data_batch = self.get_event_batch(actual_frame_idx)
        return data_batch

    def get_annoation(self, index):
        metadata_path = self.data_path / str(int(index)) / 'metadata.json'
        segmentation_path = self.data_path / str(int(index)) / 'Segmentation' / 'Image0003.jpg'
        valid_joint_path = self.data_path / str(int(index)) / 'valid_joints.json'

        segmentation_mask = cv2.cvtColor(cv2.imread(str(segmentation_path)), cv2.COLOR_BGR2GRAY)   
        human_indices = (segmentation_mask < 127)

        segmentation_mask[human_indices] = 1
        segmentation_mask[~human_indices] = 0
    
        if not os.path.exists(metadata_path):
            return {
                'rgb_frame_index': -1,
                'ego_to_global_space': None,
                'valid_seg': False,
                'ego_j3d': None, 
                'ego_j2d': None, 
                'segmentation_mask': segmentation_mask,
                'valid_joints': None
            }

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        ego_j3d = np.array(metadata['human_body']['camera']['joints_3d'])
        ego_j2d = np.array(metadata['human_body']['camera']['joints_2d'])

        smpl_to_ego_joint_indices = list(self.cfg.SMPL_TO_JOINTS16.values())
                
        ego_j3d = ego_j3d[smpl_to_ego_joint_indices]
        ego_j2d = ego_j2d[smpl_to_ego_joint_indices]

        if valid_joint_path.exists():
            with open(valid_joint_path, 'r') as f:
                valid_joints = json.load(f)
                valid_joints = list(valid_joints.values()) 
                valid_joints = np.array(valid_joints, dtype=np.float32)
        else:
            valid_joints = np.ones(16, dtype=np.float32)
    
        return {
            'rgb_frame_index': index,
            'valid_seg': True,
            'ego_j3d': ego_j3d, 
            'ego_j2d': ego_j2d, 
            'segmentation_mask': segmentation_mask,
            'ego_to_global_space': None,
            'valid_joints': valid_joints,
        }
