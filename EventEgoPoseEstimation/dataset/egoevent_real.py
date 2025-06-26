import sys; sys.path.append('../')
import cv2
import random
import json
import logging
import pickle
import numpy as np

from EventEgoPoseEstimation.dataset.dataset_utils import h5py_File
from EventEgoPoseEstimation.dataset.transforms import rotate_points

from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class RealEventStream(Dataset):
    def __init__(self, processed_input_path, data_path, cfg, split, is_train, augmentation):
        super().__init__()

        self.data_path = Path(data_path)

        self.processed_input_path = Path(processed_input_path)

        with open(self.processed_input_path / 'meta.json', 'r') as f:
            meta = json.load(f)
            
        self.height = 480
        self.width = 640

        self.total_frames = meta['total_frames']
        
        self.stream_path = self.processed_input_path / 'lnes.hdf5'

        self.fin = None 

        self.local_pose_gt_path = self.data_path / 'synced_local_pose_gt.pickle'
        
        marker_to_fisheye_matrix = pickle.load(open(self.data_path / 'marker_to_fisheye_matrix.pickle', 'rb'))['marker_to_fisheye_matrix']
        self.ego_to_board_space = np.linalg.inv(marker_to_fisheye_matrix)
        
        self.global_pose_gt_path = self.data_path / 'synced_pose_gt.pickle'

        if not self.local_pose_gt_path.exists():
            self.pose_list = None
            
        else:  
            with open(self.local_pose_gt_path, 'rb') as f:
                self.pose_list = pickle.load(f)

        if not self.global_pose_gt_path.exists():
            self.frame_start_index = 0
        else:
            with open(self.global_pose_gt_path, 'rb') as f:
                self.frame_start_index = pickle.load(f)['frame_start_index']    


        self.batch_size = cfg.DATASET.EVENT_BATCH_SIZE
        self.max_frame_time = cfg.DATASET.REAL.MAX_FRAME_TIME_IN_MS
        self.is_train = is_train

        self.is_augmentation = augmentation

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
        index = int(index)
        try:
            anno = self.pose_list[index] # TODO: Fix this -1 one frame offset
        except IndexError:
            return {
            'rgb_frame_index': -1,
            'ego_to_global_space': None,
            'valid_seg': False,
            'ego_j3d': None, 
            'ego_j2d': None, 
            'valid_joints': np.zeros(16, dtype=np.float32),
            'segmentation_mask': np.zeros((self.height, self.width), dtype=np.uint8)
        }
        
        ego_j3d = anno['ego_j3d']
        ego_j2d = anno['ego_j2d']
        global_to_board_space = anno.get('global_to_board_space', None)

        segmentation_path = self.data_path / 'Blender_Segmentation' /str(int(index)) / 'Segmentation' / 'Image0003.jpg'
        
        valid_joint_path = self.data_path / 'valid_joints' /f'{int(index)}.json'
        
        if valid_joint_path.exists():
            with open(valid_joint_path, 'r') as f:
                valid_joints = json.load(f)
                valid_joints = list(valid_joints.values()) 
                valid_joints = np.array(valid_joints, dtype=np.float32)
        else:
            valid_joints = np.zeros(16, dtype=np.float32)
        
        if segmentation_path.exists():
            segmentation_mask = cv2.imread(str(segmentation_path))
            valid_seg = True        
        else:
            segmentation_mask = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            valid_seg = False

        segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)
        
        segmentation_mask[segmentation_mask > 127] = 1
        
        if not np.any(segmentation_mask):
            valid_seg = False

        if ego_j3d is not None:
            ego_j3d = rotate_points(ego_j3d, 'x', 180.0)
        
        if global_to_board_space is not None:
            board_to_global_space = np.linalg.inv(global_to_board_space)
            ego_to_global_space = board_to_global_space @ self.ego_to_board_space
        else:
            ego_to_global_space = None
        
        return {
            'rgb_frame_index': self.frame_start_index + index,
            'ego_to_global_space': ego_to_global_space,
            'valid_seg': valid_seg,
            'ego_j3d': ego_j3d, 
            'ego_j2d': ego_j2d, 
            'valid_joints': valid_joints,
            'segmentation_mask': segmentation_mask
        }
