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
    def __init__(self, data_path, cfg, is_train):
        super().__init__()

        self.data_path = Path(data_path)
        self.cfg = cfg

        with open(self.data_path / 'event_meta.json', 'r') as f:
            meta = json.load(f)
            
        self.height = meta['height']
        self.width = meta['width']
        
        self.stream_path = self.data_path / 'events.h5'
        self.fin = None 

        self.batch_size = cfg.DATASET.EVENT_BATCH_SIZE
        self.max_frame_time = cfg.DATASET.SYNTHETIC.MAX_FRAME_TIME_IN_MS
        self.is_train = is_train

        self.frame_offsets = np.load(f"/scratch/inf0/user/mdeshmuk/EE3D-S-frame-offsets/{self.data_path.name}.npy")
        self.cumulative_offsets = np.insert(np.cumsum(self.frame_offsets), 0, 0)
        self.num_frames = len(self.frame_offsets)

        # self.index = 1

    def init_stream(self):
        self.fin = h5py_File(self.stream_path, 'r')['event']
        
    def __len__(self):
        return self.num_frames // 10 # Take every 10th frame
        # return len(self.frame_offsets)    
        # with h5py_File(self.stream_path, 'r') as f:
        #     return f['event'].shape[0] // self.batch_size
        #     # return f['event'].shape[0]


    def get_event_batch(self, frame_idx):
        # print("__getitem__ Index: ", frame_idx)

        # TODO: problem if frame_idx is 0 FIX THIS
        start = self.cumulative_offsets[frame_idx]
        end = self.cumulative_offsets[frame_idx + 1]
        data_batch = self.fin[start:end]

        # print("start", start)
        # try:
        #     print("Frame index", data_batch[-1, 4])
        # except:
        #     print("Problem")

        return np.array(data_batch)

    
    def __getitem__(self, idx):
        if self.fin is None: self.init_stream() # Done to ensure multiprocessing works

        data_batch = self.get_event_batch(idx)
        return data_batch

    def get_annoation(self, indexes):
        rgb_indexes = []
        ego_j3ds = []
        ego_j2ds = []
        segmentation_masks = []
        valid_joints_list = []

        for index in indexes:
            try:
                metadata_path = self.data_path / str(int(index)) / 'metadata.json'
                segmentation_path = self.data_path / str(int(index)) / 'Segmentation' / 'Image0003.jpg'
                valid_joint_path = self.data_path / str(int(index)) / 'valid_joints.json'

                segmentation_mask = cv2.cvtColor(cv2.imread(str(segmentation_path)), cv2.COLOR_BGR2GRAY)   
                human_indices = (segmentation_mask < 127)

                segmentation_mask[human_indices] = 1
                segmentation_mask[~human_indices] = 0

                segmentation_masks.append(segmentation_mask)
            
                if not os.path.exists(metadata_path):
                    rgb_indexes.append(-1)
                    ego_j3ds.append(None)
                    ego_j2ds.append(None)
                    segmentation_masks.append(segmentation_mask)
                    valid_joints.append(None)

                    # return {
                    #     'rgb_frame_index': -1,
                    #     'ego_to_global_space': None,
                    #     'valid_seg': False,
                    #     'ego_j3d': None, 
                    #     'ego_j2d': None, 
                    #     'segmentation_mask': segmentation_mask,
                    #     'valid_joints': None
                    # }

                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                ego_j3d = np.array(metadata['human_body']['camera']['joints_3d'])
                ego_j2d = np.array(metadata['human_body']['camera']['joints_2d'])

                smpl_to_ego_joint_indices = list(self.cfg.SMPL_TO_JOINTS16.values())
                        
                ego_j3d = ego_j3d[smpl_to_ego_joint_indices]
                ego_j3ds.append(ego_j3d)

                ego_j2d = ego_j2d[smpl_to_ego_joint_indices]
                ego_j2ds.append(ego_j2d)


                if valid_joint_path.exists():
                    with open(valid_joint_path, 'r') as f:
                        valid_joints = json.load(f)
                        valid_joints = list(valid_joints.values()) 
                        valid_joints = np.array(valid_joints, dtype=np.float32)
                else:
                    valid_joints = np.ones(16, dtype=np.float32)

                valid_joints_list.append(valid_joints)
                rgb_indexes.append(index)

            except Exception as e:
                logger.error(f'Error in getting annotation for index {index}. Error: {e}')
                rgb_indexes.append(-1)
                ego_j3ds.append(None)
                ego_j2ds.append(None)
                segmentation_masks.append(np.zeros((self.height, self.width), dtype=np.uint8))
                valid_joints_list.append(None)
                continue
    
        return {
            'rgb_frame_index': rgb_indexes,
            'valid_seg': True,
            'ego_j3d': ego_j3ds, 
            'ego_j2d': ego_j2ds, 
            'segmentation_mask': segmentation_masks,
            'ego_to_global_space': None,
            'valid_joints': valid_joints_list,
        }
