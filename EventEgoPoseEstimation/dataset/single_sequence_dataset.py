import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from EventEgoPoseEstimation.dataset.Joints3DDataset import Joints3DDataset
from EventEgoPoseEstimation.dataset.representation import EROS, LNES, EventFrame, RawEvent
from EventEgoPoseEstimation.dataset.egoevent_synthetic import SyntheticEventStream
from EventEgoPoseEstimation.dataset.egoevent_real import RealEventStream
from EventEgoPoseEstimation.dataset.dataset_utils import generate_path_split, generate_indices

def get_representation(cfg, width, height, temporal_bins, augmentation):
    representation = cfg.DATASET.REPRESENTATION
    
    if representation == 'EROS':
        eros_config = cfg.DATASET.EROS    
        repr = EROS(eros_config.kernel_size, height, width, eros_config.DECAY_BASE)
    elif representation == 'LNES':
        repr = LNES(cfg, height, width)
    elif representation == 'EventFrame':
        repr = EventFrame(cfg, height, width)
    elif representation == 'RawEvent':
        repr = RawEvent(cfg, height, width, temporal_bins, augmentation)
    else:
        raise NotImplementedError

    return repr


class SingleSequenceDataset(Joints3DDataset):    
    def prepare_anno(self, item):
        if item['ego_to_global_space'] is None:
            ego_to_global_space = None
        else:
            ego_to_global_space = item['ego_to_global_space']
        j3d = item['ego_j3d']
        j2d = item['ego_j2d']
        segmentation_mask = item['segmentation_mask']
        if type(item['valid_seg']) == bool:
            valid_seg = item['valid_seg']
        else:
            valid_seg = item['valid_seg']
        rgb_frame_index = item['rgb_frame_index']

        if j3d is None or j2d is None:
            vis_j2d = np.zeros((self.num_joints, 2))
            vis_j3d = np.zeros((self.num_joints, 3))
            j3d = np.ones((self.num_joints, 3)) * -1
            j2d = np.ones((self.num_joints, 2)) * -1
        else:
            vis_j2d = np.ones_like(j2d)
            vis_j3d = np.ones_like(j3d)

        if ego_to_global_space is None:
            ego_to_global_space = np.eye(4)

        return {
            'valid_seg': valid_seg,
            'j2d': j2d.astype(np.float32),
            'j3d': j3d.astype(np.float32),	
            'vis_j2d': vis_j2d.astype(np.float32),
            'vis_j3d': vis_j3d.astype(np.float32),
            'segmentation_mask': segmentation_mask,
            'ego_to_global_space': ego_to_global_space,
            'rgb_frame_index': int(rgb_frame_index)
        }
    
    def __init__(self, cfg, preprocessed_item_path, dataset_item_path, is_train, split, temporal_bins, augmentation=False):
        super().__init__(cfg, dataset_item_path, is_train, temporal_bins)

        if cfg.DATASET.TYPE == 'Synthetic':
            dataset = SyntheticEventStream(preprocessed_item_path, dataset_item_path, cfg, split, is_train, augmentation)
        else:
            dataset = RealEventStream(preprocessed_item_path, dataset_item_path, cfg, split, is_train, augmentation)
        
        self.data_path = dataset_item_path
        self.split = split
        self.dataset = dataset  # This dataset now represents one pose sequence (all frames)
        self.num_joints = cfg.NUM_JOINTS
        self.temporal_bins = temporal_bins
        print("Loading data for split : ", self.split)
        print(f'{self.data_path} => load {len(self.dataset)} frames for this pose sequence')

        self.is_train = is_train

    def isvalid(self):
        return len(self.dataset) > 0

    def __len__(self):
        # Instead of returning the number of frames, we now treat the entire pose sequence as one sample.
        return 1

    def __getitem__(self, idx):
        # We expect idx to always be 0 since __len__ returns 1
        if idx != 0:
            raise IndexError("SingleSequenceDataset only contains one sample (the full pose sequence).")
            
        kwargs = {}
        # Lists to collect data for the whole sequence.
        sequence_data = []
        frame_ids = []
        annotations = []
        pose_filenames = []
        
        # Iterate through all frames in the underlying event stream.
        for i in range(len(self.dataset)):
            # Each __getitem__ call on self.dataset returns a single frame (or event)
            data_batch, frame_id, pose_filename = self.dataset[i, kwargs]
            
            if len(data_batch) == 0:
                continue  # or raise an error if appropriate
            
            anno = self.dataset.get_annoation(frame_id)
            anno = self.prepare_anno(anno)
            anno['pose_filename'] = pose_filename
            
            # Apply any transformation defined in your Joints3DDataset / SingleSequenceDataset.
            transformed = self.transform(data_batch, frame_id, anno, kwargs)
            sequence_data.append(transformed)
            # frame_ids.append(frame_id)
            # annotations.append(anno)
            # pose_filenames.append(pose_filename)
        
        # Optionally, if your transformed data are tensors and share the same shape, you can stack them:
        # if len(sequence_data) > 0 and isinstance(sequence_data[0], torch.Tensor):
        #     sequence_data = torch.stack(sequence_data, dim=0)
        
        # Return a dictionary containing the entire sequence.
        return sequence_data
        # return {
        #     "data": sequence_data,         # List or tensor of transformed frames/events.
        #     "frame_ids": frame_ids,          # List of frame IDs.
        #     "annotations": annotations,      # List of annotation dictionaries.
        #     "pose_filenames": pose_filenames # List of pose filenames.
        # }
