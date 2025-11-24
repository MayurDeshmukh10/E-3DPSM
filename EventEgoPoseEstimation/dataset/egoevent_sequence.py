import copy
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch
import os
import random
from tqdm import tqdm


from EventEgoPoseEstimation.dataset.Joints3DDataset import Joints3DDataset
from EventEgoPoseEstimation.dataset.representation import EROS, LNES, EventFrame, RawEvent
from EventEgoPoseEstimation.dataset.egoevent_synthetic_sequence import SyntheticEventSequenceStream
from EventEgoPoseEstimation.dataset.egoevent_real_sequence import RealEventSequenceStream
from EventEgoPoseEstimation.dataset.egoevent_real import RealEventStream
from EventEgoPoseEstimation.dataset.dataset_utils import generate_path_split, generate_indices

class LazySequence:
    def __init__(self, underlying_dataset, bg_datasets, transform_fn, prepare_anno_fn, split, kwargs=None):
        """
        underlying_dataset: the dataset instance (e.g., SyntheticEventStream or RealEventStream)
          that returns a single frame/event given an index.
        transform_fn: a callable to transform the raw data.
        prepare_anno_fn: a callable to prepare annotations.
        kwargs: additional arguments (if any) passed to underlying indexing.
        """
        self.underlying = underlying_dataset
        self.transform = transform_fn
        self.prepare_anno = prepare_anno_fn
        self.kwargs = kwargs if kwargs is not None else {}
        self._length = len(underlying_dataset)
        # self.bg_datasets = bg_datasets
        # self.bg_datasets_count = len(self.bg_datasets)
        # self.lengths = [len(dataset) for dataset in self.bg_datasets]
        self.split = split

    def __getitem__(self, index):

        # bg_dataset_idx = np.random.randint(0, self.bg_datasets_count)
        # bg_sample_idx = np.random.randint(0, self.lengths[bg_dataset_idx])

        # Support both integer and slice indexing.
        if isinstance(index, int):
            # Load one frame on demand.
            data_batch, frame_id, pose_filename = self.underlying[index, self.kwargs]
            if len(data_batch) == 0:
                # Handle empty frame case (e.g. skip or raise an error)
                return None
            anno = self.underlying.get_annoation(frame_id)
            anno = self.prepare_anno(anno)
            anno['pose_filename'] = pose_filename
            data, meta = self.transform(data_batch, frame_id, anno, self.kwargs)

            meta['use_bg'] = False

            # bg_data, frame_id, filename = self.bg_datasets[bg_dataset_idx][bg_sample_idx, {}]
            # meta['bg_data'] = bg_data

            if self.split == 'train' and random.random() < 0.5:
                if torch.sum(meta['vis_j3d']) != 0:
                    meta['use_bg'] = True

            return data, meta

        elif isinstance(index, slice):
            # Return a list by iterating over the requested slice.
            indices = range(*index.indices(self._length))
            return [self.__getitem__(i) for i in indices]
        else:
            raise TypeError("Invalid index type for LazySequence.")

    def __len__(self):
        return self._length


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

        valid_joints = item['valid_joints']

        if j3d is None or j2d is None:
            vis_j2d = np.zeros((self.num_joints, 2))
            vis_j3d = np.zeros((self.num_joints, 3))
            vis_ja = np.zeros((self.num_joints, 1))
            j3d = np.ones((self.num_joints, 3)) * -1
            j2d = np.ones((self.num_joints, 2)) * -1
        else:
            vis_j2d = np.ones_like(j2d)
            vis_j3d = np.ones_like(j3d)
            vis_ja = np.ones((self.num_joints, 1))

        if ego_to_global_space is None:
            ego_to_global_space = np.eye(4)

        return {
            'valid_seg': valid_seg,
            'j2d': j2d.astype(np.float32),
            'j3d': j3d.astype(np.float32),	
            'vis_j2d': vis_j2d.astype(np.float32),
            'vis_j3d': vis_j3d.astype(np.float32),
            'vis_ja': vis_ja.astype(np.float32),
            'segmentation_mask': segmentation_mask,
            'ego_to_global_space': ego_to_global_space,
            'rgb_frame_index': int(rgb_frame_index),
            'valid_joints': valid_joints
        }
    
    def __init__(self, 
                cfg,
                preprocessed_item_path,
                dataset_item_path,
                bg_preprocessed_item_path,
                bg_dataset_item_path,
                is_train,
                split,
                temporal_bins,
                training_type,
                frame_offset,
                total_frames=None,
                augmentation=False):
        super().__init__(cfg, dataset_item_path, is_train, temporal_bins)

        if training_type == 'pretrain':
            dataset = SyntheticEventSequenceStream(preprocessed_item_path, dataset_item_path, cfg, split, is_train, augmentation, frame_offset, total_frames)
        else:
            dataset = RealEventSequenceStream(preprocessed_item_path, dataset_item_path, cfg, split, is_train, augmentation, frame_offset, total_frames)
        
        self.data_path = dataset_item_path
        self.preprocessed_input_path = preprocessed_item_path
        self.split = split
        self.dataset = dataset  # The underlying stream (one pose sequence)
        self.num_joints = cfg.NUM_JOINTS
        self.temporal_bins = temporal_bins
        # print("Loading data for split:", self.split)
        # print(f'{self.data_path} => load {len(self.dataset)} frames for this pose sequence')
        self.is_train = is_train

        self.bg_dataset_path = bg_dataset_item_path
        self.bg_preprocessed_item_path = bg_preprocessed_item_path

        self.bg_datasets = list()
        # for item in os.listdir(self.bg_dataset_path):
        #     data_path = self.bg_dataset_path / item
        #     preprocessed_input_path = self.bg_preprocessed_item_path / item
            
        #     if os.path.isdir(self.data_path):
        #         # dataset = SingleSequenceDataset(cfg, preprocessed_input_path, data_path, is_train, split, temporal_bins, training_type='Real', augmentation=True)
        #         dataset = RealEventStream(preprocessed_input_path, data_path, cfg, split, is_train, augmentation=True)
        #         self.bg_datasets.append(dataset)

        # for dataset in self.bg_datasets:
        #     print(dataset.data_path)
        # print('Total number of augmented background event sequences:', len(self.bg_datasets))

    def isvalid(self):
        return len(self.dataset) > 0

    def __len__(self):
        # One sample per pose sequence.
        return 1

    def __getitem__(self, idx):
        # Since __len__ returns 1, we expect idx == 0.
        if idx != 0:
            raise IndexError("SingleSequenceDataset only contains one sample (the full pose sequence).")

        return LazySequence(self.dataset, self.bg_datasets, self.transform, self.prepare_anno, self.split)


class EgoEventSequence(Dataset): 
    def __init__(self, 
                cfg, 
                processed_input_path,
                dataset_path,
                bg_input_path,
                bg_dataset_path,
                split,
                training_type,
                use_bg_augmentation,
                temporal_bins,
                fixed_sequence_length,
                finetune=False):
        super().__init__()

        cfg = copy.deepcopy(cfg)
    
        if finetune:
            cfg.DATASET.TYPE = 'Real'
            
        dataset_path = Path(dataset_path)
        processed_input_path = Path(processed_input_path)

        bg_dataset_path = Path(bg_dataset_path)
        bg_input_path = Path(bg_input_path)

        assert split in ['train', 'val', 'test']

        if split == 'train':
            split_path = processed_input_path / 'train.txt'
        elif split == 'val':
            split_path = processed_input_path / 'val.txt'
        elif split == 'test':
            split_path = processed_input_path / 'test.txt'
            
        with open(split_path, 'r') as f:
            self.items = f.read().splitlines()
    
        is_train = (split == 'train')
        self.is_train = is_train

        self.fixed_sequence_length = fixed_sequence_length
    
        original_datasets = []
        for item in tqdm(self.items, desc=f"loading datasets"):
            dataset_item_path = dataset_path / item
            preprocessed_item_path = processed_input_path / split / item

            original_dataset = SingleSequenceDataset(cfg, 
                                            preprocessed_item_path, 
                                            dataset_item_path,
                                            bg_input_path,
                                            bg_dataset_path,
                                            is_train, 
                                            split, 
                                            temporal_bins,
                                            training_type,
                                            frame_offset=0,
                                            total_frames=None)
            if original_dataset.isvalid():
                original_datasets.append(original_dataset)

        self.datasets = original_datasets
        self.total_sequences = len(original_datasets)  # Number of valid pose sequences.

        # TODO: Update real dataset to have the same structure as synthetic dataset
        if training_type in ['pretrain', 'finetune', 'EE3D-W-finetuning']:
            new_datasets = []
            for original_dataset in tqdm(original_datasets, desc=f"creating streaming datasets"):
                dataset_len = len(original_dataset[0])
                total_sub_sequences = dataset_len // self.fixed_sequence_length
                for i in range(total_sub_sequences):
                    frame_offset = i * self.fixed_sequence_length
                    new_dataset = SingleSequenceDataset(cfg,
                                                original_dataset.preprocessed_input_path, 
                                                original_dataset.data_path,
                                                bg_input_path,
                                                bg_dataset_path,
                                                is_train, 
                                                split, 
                                                temporal_bins,
                                                training_type,
                                                frame_offset=frame_offset,
                                                total_frames=self.fixed_sequence_length)
                    if new_dataset.isvalid():
                        new_datasets.append(new_dataset)

            self.datasets = new_datasets
            self.total_sequences = len(new_datasets)  # Number of valid pose sequences.

        
        self.dataset_root = dataset_path
        
        print(f'Dataset root: {self.dataset_root}, split: {split}, finetune: {finetune}')
        # print('Datasets:')
        # for dataset in self.datasets:
        #     print(dataset.data_path)
        print('Total number of pose sequences:', self.total_sequences)

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):    
        # Return the full pose sequence from the corresponding SingleSequenceDataset.
        # Since each SingleSequenceDataset returns one sample, we use index 0.
        return self.datasets[idx][0]

    def evaluate_joints(self, *args, **kwargs):
        return Joints3DDataset.evaluate_joints(*args, **kwargs)

    @classmethod    
    def evaluate_dataset(cls, *args, **kwargs):
        return Joints3DDataset.evaluate_dataset(*args, **kwargs)
