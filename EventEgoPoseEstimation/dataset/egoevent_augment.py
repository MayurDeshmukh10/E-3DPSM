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
    def __init__(self, cfg, target_dataset, split, temporal_bins):
        super().__init__()
        cfg = copy.deepcopy(cfg)
        
        cfg.DATASET.TYPE = 'Real'

        self.target_dataset = target_dataset
        is_train = target_dataset.is_train
        dataset_root = Path(cfg.DATASET.BACKGROUND_DATASET_ROOT)
        self.temporal_bins = temporal_bins

        self.split = split
        
        datasets = list()
        for item in os.listdir(dataset_root):
            data_path = dataset_root / item
            
            if os.path.isdir(data_path):
                dataset = SingleSequenceDataset(cfg, data_path, is_train, split, temporal_bins, augmentation=True)
                if dataset.isvalid():
                    self.visualize = dataset.visualize
                    datasets.append(dataset)

        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.lengths)

        self.data_path = dataset_root
        self.dataset_root = dataset_root
        self.is_train = is_train
        
        index_path = Path(tempfile.mkdtemp('egoevent_combined'))        
        self.indices = generate_indices(index_path, self.datasets)
        self.index_len = len(self.indices)

        print("Index len: ", self.index_len)

        print(f'BG Dataset root: {dataset_root}, is_train: {is_train}')
        print('BG Datasets: ')
        for dataset in datasets:
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
        
        meta['use_bg'] = False

        aidx = np.random.randint(0, self.index_len - 1)
        dataset_idx, sample_idx = self.indices[aidx]
        bg_data, bg_meta = self.datasets[dataset_idx][sample_idx, kwargs]
        meta['bg_data'] = bg_data['x']

        if torch.sum(meta['vis_j3d']) != 0:
            if self.split == 'train' and random.random() < 0.5:
                meta['use_bg'] = True
        
        return data, meta

    @classmethod
    def evaluate_joints(self, *args, **kwargs):
        return Joints3DDataset.evaluate_joints(*args, **kwargs)

    @classmethod    
    def evaluate_dataset(cls, *args, **kwargs):
        return Joints3DDataset.evaluate_dataset(*args, **kwargs)