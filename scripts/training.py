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

from deepcodecorrection.pl_trainer import PLTrainer
from deepcodecorrection.data_generation import NoiseDataset

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
    dataset = NoiseDataset(
        dim_input=nb_effective_transmitted_symbol,
        lenght_epoch=20000,
        max_class=nb_class,
    )
    dataset_val = NoiseDataset(
        dim_input=nb_effective_transmitted_symbol, lenght_epoch=1000, max_class=nb_class
    )

    snr_db = args.SNR  # this is Eb/N0

    # convert SNR to normal value
    snr = 10 ** (snr_db / 10.0)

    # compute the noise level
    noise_level = 1.0 / math.sqrt(snr)

    # snr per bit
    snr_per_bit = snr / bit_per_class

    print("noise level : ", noise_level)
    print("code rate : ", args.code_rate)
    print("nb_effective_transmitted_symbol : ", nb_effective_transmitted_symbol)
    print("snr : ", snr)
    print("snr per bit : ", snr_per_bit)
    print("shannon limit : ", args.dim_input_global * np.log2(1 + snr))

    # we choose the name according to the SNR and code rate
    name_model = f"deepcode_SNR_{snr_db}_code_rate_{round(args.code_rate, 4)}_bit_{bit_per_class}"

    # init trainer
    model = PLTrainer(
        max_dim_input=300,
        nb_class=nb_class,
        dim_global=32,
        noise_level=noise_level,
        coeff_code_rate=args.code_rate,
        dim_global_block=args.dim_input_global,
        nb_block_size=16,
    )

    # compile the model
    # model = torch.compile(model)

    last_checkpoint = None
    # last_checkpoint = load_last_checkpoint("/home/checkpoints")

    # print(last_checkpoint)
    # last_checkpoint = "/home/checkpoints/my_model-epoch=967-val_loss=0.59.ckpt"

    # model = PLTrainer.load_from_checkpoint(
    #     last_checkpoint,
    #     max_dim_input=250,
    #     nb_class=nb_class,
    #     dim_global=32,
    #     noise_level=0.1
    # )

    logger = TensorBoardLogger("tb_logs", name=name_model)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=name_model + "-{epoch:02d}-{val_loss:.2f}",
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
        gradient_clip_val=2.,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=10,
        log_every_n_steps=5,
    )

    trainer.fit(
        model,
        DataLoader(dataset, batch_size=batch_size),
        DataLoader(dataset_val, batch_size=batch_size),
        ckpt_path=last_checkpoint,
    )


if __name__ == "__main__":
    # we retrieve the argument
    # 1 SNR value
    parser = argparse.ArgumentParser()
    parser.add_argument("--SNR", type=float, default=3.0)

    # 2 code rate
    parser.add_argument("--code_rate", type=float, default=6./16.)

    # 3 dim_input of the model
    parser.add_argument("--dim_input_global", type=int, default=240)

    # 4 nb of symbols
    parser.add_argument("--nb_symbols", type=int, default=2)

    args = parser.parse_args()

    main(args)


    
