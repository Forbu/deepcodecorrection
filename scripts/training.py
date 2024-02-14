"""
In this script we will train the model to check the performance
"""
import os

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl

import torchmetrics

# tensorboard logger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from deepcodecorrection.pl_trainer_noise_conscious import PLTrainer
from deepcodecorrection.data_generation import NoiseDataset

# seed for reproducibility
torch.manual_seed(42)


def load_last_checkpoint(directory):
    """
    A function to load the last checkpoint from the given directory.

    Args:
        directory (str): The directory from which to load the last checkpoint.

    Returns:
        str: The path of the last checkpoint file.
    """
    # we take the last checkpoint
    sorted_files = sorted(os.listdir(directory))

    if len(sorted_files) == 0:
        return None

    last_checkpoint = sorted_files[-1]

    return os.path.join(directory, last_checkpoint)


def main():
    """
    The main function that initializes a dataset, a trainer, logger,
    and checkpoint callback, then trains the model.
    """
    nb_class = 16
    batch_size = 256
    # init dataset
    dataset = NoiseDataset(dim_input=100, lenght_epoch=20000, max_class=nb_class)
    dataset_val = NoiseDataset(dim_input=100, lenght_epoch=1000, max_class=nb_class)

    # init trainer
    model = PLTrainer(
        max_dim_input=250,
        nb_class=nb_class,
        dim_global=32,
    )

    # last_checkpoint = load_last_checkpoint("/home/checkpoints")

    # print(last_checkpoint)

    # model = PLTrainer.load_from_checkpoint(
    #     last_checkpoint,
    #     max_dim_input=250,
    #     nb_class=nb_class,
    #     dim_global=32,
    #     noise_level=0.1
    # )

    logger = TensorBoardLogger("tb_logs", name="my_model")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="my_model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    # train the model
    trainer = pl.Trainer(
        max_epochs=5000,
        accelerator="auto",
        devices=1,
        logger=logger,
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model,
        DataLoader(dataset, batch_size=batch_size),
        DataLoader(dataset_val, batch_size=batch_size),
        # ckpt_path=last_checkpoint,
    )


if __name__ == "__main__":
    main()
