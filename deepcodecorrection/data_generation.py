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

    def __init__(self, dim_input, lenght_epoch, max_class=8):
        super().__init__()

        self.max_class = max_class
        self.dim_input = dim_input
        self.lenght_epoch = lenght_epoch

    def __len__(self):
        return self.lenght_epoch

    def __getitem__(self, idx):

        choosed_class = torch.randint(low=0, high=self.max_class, size=(self.dim_input,))


        return choosed_class
