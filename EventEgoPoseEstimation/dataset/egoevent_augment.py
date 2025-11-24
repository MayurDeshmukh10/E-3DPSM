import torch
import random
import cv2
import copy
import numpy as np
import os
import tempfile
from rich import print
from pathlib import Path
from torch.utils.data import Dataset


from EventEgoPoseEstimation.dataset.Joints3DDataset import Joints3DDataset
from EventEgoPoseEstimation.dataset.representation import EROS, LNES, EventFrame
from EventEgoPoseEstimation.dataset.egoevent_synthetic import SyntheticEventStream
from EventEgoPoseEstimation.dataset.egoevent_real import RealEventStream
from EventEgoPoseEstimation.dataset.dataset_utils import generate_path_split, generate_indices
from EventEgoPoseEstimation.dataset.egoevent import SingleSequenceDataset


class AugmentedEgoEvent(Dataset): 
    def __init__(self, cfg, target_dataset, bg_data_root, bg_preprocessed_root, split, temporal_bins):
        super().__init__()
        cfg = copy.deepcopy(cfg)


        self.bg_dataset_path = Path(bg_data_root)
        self.bg_preprocessed_item_path = Path(bg_preprocessed_root)
        
        self.target_dataset = target_dataset

        self.temporal_bins = temporal_bins

        self.split = split

        is_train = target_dataset.is_train
        
        # datasets = list()
        # for item in os.listdir(dataset_root):
        #     data_path = dataset_root / item
            
        #     if os.path.isdir(data_path):
        #         dataset = SingleSequenceDataset(cfg, data_path, is_train, split, temporal_bins, augmentation=True)
        #         if dataset.isvalid():
        #             self.visualize = dataset.visualize
        #             datasets.append(dataset)

        self.bg_datasets = list()
        for item in os.listdir(self.bg_dataset_path):
            data_path = self.bg_dataset_path / item
            preprocessed_input_path = self.bg_preprocessed_item_path / item
            
            if os.path.isdir(preprocessed_input_path):
                # dataset = SingleSequenceDataset(cfg, preprocessed_input_path, data_path, is_train, split, temporal_bins, training_type='Real', augmentation=True)
                dataset = RealEventStream(preprocessed_input_path, data_path, cfg, split, is_train, augmentation=True)
                self.bg_datasets.append(dataset)

        self.lengths = [len(dataset) for dataset in self.bg_datasets]
        self.total_length = sum(self.lengths)
        self.bg_datasets_count = len(self.bg_datasets)

        self.is_train = is_train
        
        # index_path = Path(tempfile.mkdtemp('egoevent_combined'))        
        # self.indices = generate_indices(index_path, self.datasets)
        # self.index_len = len(self.indices)

        print(f'BG Dataset root: {self.bg_dataset_path}, is_train: {self.is_train}')
        print('BG Datasets: ')
        for dataset in self.bg_datasets:
            print(dataset.data_path)
        print('Total number of BG events: ', self.total_length)

    def __len__(self):
        return len(self.target_dataset)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            _, kwargs = idx
        else:
            kwargs = {}
            
        data, meta = self.target_dataset[idx]

        bg_dataset_idx = np.random.randint(0, self.bg_datasets_count)
        bg_sample_idx = np.random.randint(0, self.lengths[bg_dataset_idx])
        
        meta['use_bg'] = False

        bg_data, frame_id, filename = self.bg_datasets[bg_dataset_idx][bg_sample_idx, {}]

        meta['bg_data'] = bg_data

        if self.split == 'train' and random.random() < 0.5:
            if torch.sum(meta['vis_j3d']) != 0:
                meta['use_bg'] = True
        
        return data, meta

    @classmethod
    def evaluate_joints(self, *args, **kwargs):
        return Joints3DDataset.evaluate_joints(*args, **kwargs)

    @classmethod    
    def evaluate_dataset(cls, *args, **kwargs):
        return Joints3DDataset.evaluate_dataset(*args, **kwargs)