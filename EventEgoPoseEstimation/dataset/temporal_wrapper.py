import numpy as np
import random

from torch.utils.data import Dataset
from EventEgoPoseEstimation.dataset import transforms
from configs.settings import config

class TemoralWrapper(Dataset):
    def __init__(self, dataset, timesteps, split, sample_step, merge_frames) -> None:
        super().__init__()

        self.dataset = dataset
        self.timesteps = timesteps
        self.sample_step = sample_step
        self.split = split
        self.merge_frames = merge_frames
    
    def __len__(self):
        if self.split == 'train':
            return len(self.dataset) // (self.sample_step) # * self.merge_frames)
        else:
            return len(self.dataset)
    
    def __getitem__(self, idx):
        kwargs = {}

        # print("Idx")
        # if self.split == 'train':
        #     real_idx = idx * self.sample_step
        # else:
        #     real_idx = idx

        # if self.merge_frames > 0:
        #     # print("self.sample_step: ", self.sample_step)

        #     real_idx = max(real_idx * self.merge_frames, len(self.dataset) 
        #     kwargs['merge_frames'] = self.merge_frames

        
        # print(f"real_idx: {real_idx}")

        # start_index = max(0, real_idx - self.timesteps)
        # # start_index = max(0, anchor - (self.timesteps - 1) * 10)
        # end_index = real_idx
        
        # end_index += self.timesteps - (end_index - start_index)

        if self.merge_frames > 1:
            kwargs['merge_frames'] = self.merge_frames

        if self.split == 'train':
            start_index = idx * self.sample_step
        else:
            start_index = idx

        end_index = start_index + (self.timesteps * self.merge_frames) - 1

    
        data = []
        # print(f"start_index: {start_index}, end_index: {end_index}")
        for i in range(start_index, end_index + 1):
            try:
                batch_data = self.dataset[i, kwargs]
                data.append(batch_data)
            except Exception as e:
                print("Error in getting data ", e)
                # print("Error in getting data")
                continue

        return data

    def visualize(self, *args, **kwargs):
        return self.dataset.visualize(*args, **kwargs)

    def evaluate_joints(self, *args, **kwargs):
        return self.dataset.evaluate_joints(*args, **kwargs)
    
    def evaluate_dataset(self, *args, **kwargs):
        return self.dataset.evaluate_dataset(*args, **kwargs)
