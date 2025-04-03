import platform
import copy
import numpy as np
from rich import print
from pathlib import Path
from torch.utils.data import Dataset

from EventEgoPoseEstimation.dataset.Joints3DDataset import Joints3DDataset
# from EventEgoPoseEstimation.dataset.Joints3DDataset import Joints3DDataset
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
        rgb_indexes = []
        ego_j3ds = []
        ego_j2ds = []
        segmentation_masks = []
        vis_j2ds = []
        vis_j3ds = []
        valid_seg_list = []
        ego_to_global_space_list = []
        # for i in range(self.temporal_bins):
        # for i in range(len(item['ego_j3d'])):

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
            vis_ja = np.zeros((self.num_joints, 1))

            j3d = np.ones((self.num_joints, 3)) * -1
            j2d = np.ones((self.num_joints, 2)) * -1
        else:
            vis_j2d = np.ones_like(j2d)
            vis_j3d = np.ones_like(j3d)
            vis_ja = np.ones((self.num_joints, 1))


        if ego_to_global_space is None:
            ego_to_global_space = np.eye(4)

        # rgb_indexes.append(int(rgb_frame_index))
        # ego_j3ds.append(j3d.astype(np.float32))
        # ego_j2ds.append(j2d.astype(np.float32))
        # segmentation_masks.append(segmentation_mask)
        # valid_seg_list.append(valid_seg)
        # vis_j2ds.append(vis_j2d.astype(np.float32))
        # vis_j3ds.append(vis_j3d.astype(np.float32))
        # ego_to_global_space_list.append(ego_to_global_space)


        return {
                'valid_seg': valid_seg,
                'j2d': j2d.astype(np.float32),
                'j3d': j3d.astype(np.float32),	
                'vis_j2d': vis_j2d.astype(np.float32),
                'vis_j3d': vis_j3d.astype(np.float32),
                'vis_ja': vis_ja.astype(np.float32),
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

        self.dataset = dataset
        # width, height = dataset.width, dataset.height
        self.num_joints = cfg.NUM_JOINTS

        # self.width = width
        # self.height = height

        self.temporal_bins = temporal_bins

        # self.repr = get_representation(cfg, width, height, temporal_bins, augmentation)

        # self.visualize = self.repr.visualize

        print("Loading data for split : ", self.split)
        print(f'{self.data_path} => load {len(self.dataset)} samples')

        self.is_train = is_train

    def isvalid(self):
        return len(self.dataset) > 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, kwargs = idx
        else:
            kwargs = {}
                
        data_batch, frame_id, pose_filename = self.dataset[idx, kwargs]

        # print("Frame ID : ", frame_id)
                
        if len(data_batch) == 0:
            raise StopIteration

        # if self.is_train:
        #     dist = np.random.uniform(0.2, 1.0)
        #     n_events = data_batch.shape[0]
        #     choice_len = np.random.randint(int(n_events * dist), n_events)
        #     choices = np.random.choice(np.arange(n_events), choice_len, replace=False)
        #     data_batch = data_batch[choices, :]

        # data = self.repr(data_batch)
        # frame_index = data['frame_index']
        # frame_indexes = data['frame_index']

        anno = self.dataset.get_annoation(frame_id)
        anno = self.prepare_anno(anno)
        anno['pose_filename'] = pose_filename

        return self.transform(data_batch, frame_id, anno, kwargs)


class EgoEvent(Dataset): 
    def __init__(self, cfg, processed_input_path, dataset_path, split, temporal_bins, finetune=False):
        super().__init__()

        cfg = copy.deepcopy(cfg)
    
        if finetune:
            cfg.DATASET.TYPE = 'Real'
            
        # if cfg.DATASET.TYPE == 'Synthetic':
        #     dataset_root = Path(cfg.DATASET.SYN_ROOT)
        #     # TODO: get this path from config
        #     split_root_path = dataset_root
        #     # test_path = Path("/CT/EventEgo3Dv2/work/EventEgo3Dv2/dataset_splits/test")
        #     # split_root_path = Path("/CT/EventEgo3Dv2/work/EventEgo3Dv2/dataset_splits/test")

        # else:
        #     dataset_root = Path(cfg.DATASET.REAL_ROOT)
        #     # TODO: Update this path
        #     split_root_path = dataset_root

        
        # generate_path_split(split_root_path, cfg)

        dataset_path = Path(dataset_path)
        processed_input_path = Path(processed_input_path)


        assert split in ['train', 'val', 'test']

        if split == 'train':
            split_path = processed_input_path / 'train.txt'
        elif split == 'val':
            split_path = processed_input_path / 'val.txt'
        elif split == 'test':
            split_path = processed_input_path / 'test.txt'
            
        with open(split_path, 'r') as f:
            self.items = f.read().splitlines()
    
        if split == 'train':
            is_train = True
        else:
            is_train = False

        self.is_train = is_train
    
        datasets = list()
        for item in self.items:
            dataset_item_path = dataset_path / item
            preprocessed_item_path = processed_input_path / split / item
            dataset = SingleSequenceDataset(cfg, preprocessed_item_path, dataset_item_path, is_train, split, temporal_bins=temporal_bins)
            if dataset.isvalid():
                # self.visualize = dataset.visualize
                datasets.append(dataset)
            
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.lengths)

        self.dataset_root = dataset_path
        
        print(f'Dataset root: {self.dataset_root}, split: {split}, finetune: {finetune}')
        print('Datasets: ')
        for dataset in datasets:
            print(dataset.data_path)
        print('Total number of events: ', self.total_length)

        # self.indices = generate_indices(self.dataset_root, self.datasets)
        self.indices = generate_indices(Path('/scratch/inf0/user/mdeshmuk/ee3d_indices'), self.datasets)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):    
        if isinstance(idx, tuple):
            idx, kwargs = idx
        else:
            kwargs = {}
        
        dataset_idx, sample_idx = self.indices[idx]   
                
        data, meta = self.datasets[dataset_idx][sample_idx, kwargs]
        
        return data, meta

    def evaluate_joints(self, *args, **kwargs):
        return Joints3DDataset.evaluate_joints(*args, **kwargs)

    @classmethod    
    def evaluate_dataset(cls, *args, **kwargs):
        return Joints3DDataset.evaluate_dataset(*args, **kwargs)