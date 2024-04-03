"""
This module generate : 
- random noise (0 / 1 randomly)
- random noise injestion in bit values (flipping 0 / 1)
"""
import math

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

        # corresponding number of bit per class
        self.nb_bit = math.ceil(math.log2(max_class))

    def __len__(self):
        return self.lenght_epoch

    def __getitem__(self, idx):
        choosed_class = torch.randint(
            low=0, high=self.max_class, size=(self.dim_input,)
        )

        bit_array = torch.zeros((self.dim_input, self.nb_bit))

        for index_bit in range(self.nb_bit):
            bit_array[:, index_bit] = (choosed_class >> index_bit) & 1

        return choosed_class, bit_array


class NoiseDatasetWithNoiseInfo(NoiseDataset):
    """
    Dataset class for noise with noise info added
    """

    def __init__(self, dim_input, lenght_epoch, max_class=8, noise_interval=[0, 1]):
        super().__init__(dim_input, lenght_epoch, max_class)

        self.noise_interval = noise_interval

    def __len__(self):
        return self.lenght_epoch

    def __getitem__(self, idx):
        choosed_class, bit_array = super().__getitem__(idx)

        # we also choose uniform noise between 0 and 1 and scale on the interval
        choosed_snr = (
            torch.rand(1) * (self.noise_interval[1] - self.noise_interval[0])
            + self.noise_interval[0]
        )

        # convert SNR to normal value
        snr = 10 ** (choosed_snr / 10.0)

        # compute the noise level
        noise_level = 1.0 / math.sqrt(snr * 2)

        return choosed_class, bit_array, noise_level
