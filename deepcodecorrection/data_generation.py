"""
This module generate : 
- random noise (0 / 1 randomly)
- random noise injestion in bit values (flipping 0 / 1)
"""


import torch

# import dataset class
from torch.utils.data import Dataset, DataLoader


class NoiseDataset(Dataset):
    """
    Dataset class for noise

    Those data will represent the random input the model will have to transmit
    (and at the end, predict)
    """

    def __init__(self, dim_input, lenght_epoch):
        self.dim_input = dim_input
        self.lenght_epoch = lenght_epoch

    def __len__(self):
        return self.lenght_epoch

    def __getitem__(self, idx):
        return torch.randint(low=0, high=8)
