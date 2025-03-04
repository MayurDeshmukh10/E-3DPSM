import numpy as np
import random

from torch.utils.data import Dataset
from EventEgoPoseEstimation.dataset import transforms
from configs.settings import config

class TemoralWrapper(Dataset):
    def __init__(self, dataset, timesteps, split, sample_step) -> None:
        super().__init__()

        self.dataset = dataset
        self.timesteps = timesteps
        self.sample_step = sample_step
        self.split = split
    
    def __len__(self):
        if self.split == 'train':
            return len(self.dataset) // self.sample_step
        else:
            return len(self.dataset)
    
    def __getitem__(self, idx):

        if self.split == 'train':
            real_idx = idx * self.sample_step
        else:
            real_idx = idx

        start_index = max(0, real_idx - self.timesteps)
        # start_index = max(0, anchor - (self.timesteps - 1) * 10)
        end_index = real_idx
        
        end_index += self.timesteps - (end_index - start_index)

        kwargs = {}
    
        data = []
        # print(f"start_index: {start_index}, end_index: {end_index}")
        # count = 0
        for i in range(start_index, end_index):
            # if i is first then
            # if count == 0:
            #     kwargs['first_step'] = True
            # else:
            #     kwargs['first_step'] = False
            
            data.append(self.dataset[i, kwargs])
            # count += 1

        return data

    def visualize(self, *args, **kwargs):
        return self.dataset.visualize(*args, **kwargs)

    def evaluate_joints(self, *args, **kwargs):
        return self.dataset.evaluate_joints(*args, **kwargs)
    
    def evaluate_dataset(self, *args, **kwargs):
        return self.dataset.evaluate_dataset(*args, **kwargs)
