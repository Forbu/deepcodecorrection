"""
In this script we will train the model to check the performance
"""

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl

import torchmetrics


from deepcodecorrection.pl_trainer import PLTrainer
from deepcodecorrection.data_generation import NoiseDataset

# seed for reproducibility
torch.manual_seed(42)

def main():
    nb_class = 16
    batch_size = 256
    # init dataset
    dataset = NoiseDataset(dim_input=100, lenght_epoch=20000, max_class=nb_class)
    dataset_val = NoiseDataset(dim_input=100, lenght_epoch=1000, max_class=nb_class)

    # init trainer
    model = PLTrainer(
        max_dim_input=250, nb_class=nb_class, dim_global=32
    )

    # tensorboard logger
    from lightning.pytorch.loggers import TensorBoardLogger

    logger = TensorBoardLogger("tb_logs", name="my_model")

    # train the model
    trainer = pl.Trainer(
        max_epochs=2000,
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        logger=logger,
        gradient_clip_val=0.5,
    )

    trainer.fit(
        model,
        DataLoader(dataset, batch_size=batch_size),
        DataLoader(dataset_val, batch_size=batch_size),
    )


if __name__ == "__main__":
    main()
