"""
In this script we will train the model to check the performance
"""

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl

import torchmetrics


from deepcodecorrection.pl_trainer import PLTrainer
from deepcodecorrection.data_generation import NoiseDataset


def main():
    nb_class = 8
    batch_size = 32
    # init dataset
    dataset = NoiseDataset(dim_input=100, lenght_epoch=20000, max_class=nb_class)

    # init trainer
    model = PLTrainer(
        max_dim_input=150, nb_class=nb_class, dim_global=32, coeff_code_rate=1.3
    )

    # train the model
    trainer = pl.Trainer(
        max_epochs=100, accelerator="auto", devices=1, enable_checkpointing=False
    )

    trainer.fit(model, DataLoader(dataset, batch_size=batch_size))


if __name__ == "__main__":
    main()
