"""
Module in which we test the dataloader
"""

from deepcodecorrection.data_generation import NoiseDataset


def test_dataset():
    data = NoiseDataset(dim_input=100, lenght_epoch=20000, max_class=16)

    assert len(data) == 20000

    # we sample one element from the dataloader
    sample = data[0]

    print(sample)
    assert False
