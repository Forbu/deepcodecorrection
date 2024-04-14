"""
In this script we will train the model to check the performance
"""

import os
import math

import numpy as np

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl

import torchmetrics

# tensorboard logger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from deepcodecorrection.pl_trainer_all_noise import PLTrainer
from deepcodecorrection.data_generation import NoiseDatasetWithNoiseInfo

# # seed for reproducibility
# torch.manual_seed(42)

import torch._dynamo

torch._dynamo.config.suppress_errors = True

import argparse


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


def main(args):
    """
    The main function that initializes a dataset, a trainer, logger,
    and checkpoint callback, then trains the model.
    """
    nb_class = args.nb_symbols
    bit_per_class = math.ceil(math.log2(nb_class))

    nb_effective_transmitted_symbol = int(args.code_rate * args.dim_input_global)

    batch_size = 1024
    # init dataset
    dataset = NoiseDatasetWithNoiseInfo(
        dim_input=nb_effective_transmitted_symbol,
        lenght_epoch=20000,
        max_class=nb_class,
    )

    print("code rate : ", args.code_rate)
    print("nb_effective_transmitted_symbol : ", nb_effective_transmitted_symbol)

    # we choose the name according to the SNR and code rate
    name_model = (
        f"deepcode_SNR_code_rate_{round(args.code_rate, 4)}_bit_{bit_per_class}"
    )

    datasets_validation = [
        NoiseDatasetWithNoiseInfo(
            dim_input=nb_effective_transmitted_symbol,
            lenght_epoch=batch_size,
            max_class=nb_class,
            noise_interval=[i, i],
        )
        for i in range(-2, 5)
    ]

    # init trainer
    model = PLTrainer(
        max_dim_input=300,
        nb_class=nb_class,
        dim_global=32,
        coeff_code_rate=args.code_rate,
        dim_global_block=args.dim_input_global,
        nb_block_size=16,
        dataset_train=dataset,
        datasets_validation=datasets_validation,
        batch_size=batch_size,
    )

    # compile the model
    # model = torch.compile(model)

    logger = TensorBoardLogger("tb_logs", name=name_model)

    # use wandb logger instead
    import wandb
    from lightning.pytorch.loggers import WandbLogger

    wandb.init(project="deepcodecorrection", name=name_model)

    logger = WandbLogger(project="deepcodecorrection", name=name_model)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=name_model + "-{epoch:02d}-{train_loss:.2f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
    )

    # train the model
    trainer = pl.Trainer(
        max_time={"minutes": 70 * 3},
        accelerator="auto",
        devices=1,
        logger=logger,
        gradient_clip_val=2.0,
        callbacks=[checkpoint_callback],
        # accumulate_grad_batches=10,
        log_every_n_steps=5,
    )

    trainer.fit(
        model,
        DataLoader(dataset, batch_size=batch_size),
        # DataLoader(dataset_val, batch_size=batch_size),
    )


if __name__ == "__main__":
    # we retrieve the argument
    # 1 SNR value
    parser = argparse.ArgumentParser()

    # 2 code rate
    parser.add_argument("--code_rate", type=float, default=7.0 / 16.0)

    # 3 dim_input of the model
    parser.add_argument("--dim_input_global", type=int, default=240)

    # 4 nb of symbols
    parser.add_argument("--nb_symbols", type=int, default=2)

    args = parser.parse_args()

    main(args)
