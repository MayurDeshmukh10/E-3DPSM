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
    def __init__(self, processed_input_path, data_path, cfg, split, is_train, augmentation):
        super().__init__()

        self.data_path = Path(data_path)
        self.cfg = cfg
        self.split = split

        self.processed_input_path = Path(processed_input_path)

        # split = 'test'

        with open(self.processed_input_path / 'meta.json', 'r') as f:
            meta = json.load(f)
            
        self.height = 480
        self.width = 640
        
        self.total_frames = meta['total_frames']
        
        self.stream_path = self.processed_input_path / 'lnes.hdf5'

        self.fin = None 
        self.is_augmentation = augmentation

        self.is_train = is_train

        self.filename = self.data_path.name


    def init_stream(self):
        self.fin = h5py_File(self.stream_path, 'r')
        
    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, idx):
        if self.fin is None: self.init_stream() # Done to ensure multiprocessing works

        if isinstance(idx, tuple):
            idx, kwargs = idx
        else:
            kwargs = {}

        data_batch = self.fin[str(idx)]['input']
        frame_id = self.fin[str(idx)]['frame_index'][()]

        data_batch = np.array(data_batch).astype(np.float32)
        # data_batch = np.transpose(data_batch, (2, 0, 1))
        return data_batch, frame_id, self.filename

    def get_annoation(self, index):
        metadata_path = self.data_path / str(int(index)) / 'metadata.json'
        segmentation_path = self.data_path / str(int(index)) / 'Segmentation' / 'Image0003.jpg'
        valid_joint_path = self.data_path / str(int(index)) / 'valid_joints.json'

        if segmentation_path.exists():
            segmentation_mask = cv2.cvtColor(cv2.imread(str(segmentation_path)), cv2.COLOR_BGR2GRAY)
            valid_seg = True
        else:
            segmentation_mask = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            valid_seg = False
            

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
            'valid_seg': valid_seg,
            'ego_j3d': ego_j3d, 
            'ego_j2d': ego_j2d, 
            'segmentation_mask': segmentation_mask,
            'ego_to_global_space': None,
            'valid_joints': valid_joints,
        }
