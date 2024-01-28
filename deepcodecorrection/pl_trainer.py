"""
Module for training with PyTorch Lightning
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl

import torchmetrics

from vector_quantize_pytorch import FSQ


class PLTrainer(pl.Trainer):
    """
    Class for training with PyTorch Lightning
    """

    def __init__(self, max_dim_input, nb_class, dim_global=32, coeff_code_rate=1.3):
        super().__init__()

        self.dim_global = dim_global
        self.max_dim_input = max_dim_input
        self.nb_class = nb_class
        self.dim_global = dim_global
        self.coeff_code_rate = coeff_code_rate

        self.emitter_transformer = nn.Transformer(
            d_model=dim_global,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=512,
            batch_first=True,
        )

        self.receiver_transformer = nn.Transformer(
            d_model=dim_global,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=512,
            batch_first=True,
        )

        # embedding for the position
        self.position_embedding_encoder_emitter = nn.Embedding(
            max_dim_input, dim_global
        )
        self.position_embedding_encoder_receiver = nn.Embedding(
            max_dim_input, dim_global
        )

        self.position_embedding_decoder_emitter = nn.Embedding(
            max_dim_input, dim_global
        )
        self.position_embedding_decoder_receiver = nn.Embedding(
            max_dim_input, dim_global
        )

        # embedding for the actual input
        self.input_embedding_emitter = nn.Embedding(nb_class, dim_global)

        # FSQ
        levels = [8]  # see 4.1 and A.4.1 in the paper
        self.quantizer = FSQ(levels)

        # three resizing components
        self.resize_emitter = nn.Linear(dim_global, 1)
        self.resize_receiver = nn.Linear(1, dim_global)
        self.resize_output = nn.Linear(dim_global, nb_class)

        self.criterion = nn.CrossEntropyLoss()
        
        # accuracy metric
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x, coeff_code_rate, noise_level):
        """
        Forward pass of the neural network model.

        Args:
            self: the neural network instance
            x: input tensor of shape (batch_size, seq_length)
            coeff_code_rate: coefficient for code rate adjustment

        Returns:
            output: tensor representing the output of the neural network (batch_size, seq_length, nb_class)
        """
        assert coeff_code_rate > 1.0

        batch_size, seq_length = x.shape
        dim_intermediate = torch.round(seq_length * coeff_code_rate).int()

        assert dim_intermediate < self.max_dim_input

        # generate embedding for the input
        x = self.input_embedding_emitter(x)  # batch_size, seq_length, dim_global

        emitter_position = (
            torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        )
        emitter_position = self.position_embedding_encoder_emitter(
            emitter_position
        )  # batch_size, seq_length, dim_global

        x = x + emitter_position

        # now we generate embedding for the emiiter decoder and the receiver encoder
        transmitter_position = (
            torch.arange(dim_intermediate)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to(x.device)
        )

        transmitter_position = self.position_embedding_decoder_emitter(
            transmitter_position
        )  # batch_size, dim_intermediate, dim_global
        receiver_position = self.position_embedding_encoder_receiver(
            transmitter_position
        )  # batch_size, dim_intermediate, dim_global

        # first the emitter transformation
        transmitted_information = self.emitter_transformer(
            x, transmitter_position
        )  # batch_size, dim_intermediate, dim_global

        # resize to batch_size, dim_intermediate, 1
        transmitted_information = self.resize_emitter(transmitted_information)

        # adding noise
        transmitted_information = (
            transmitted_information
            + torch.randn_like(transmitted_information) * noise_level
        )

        # discretization with FSQ
        received_information, _ = self.quantizer(transmitted_information)

        # resize to global dimension
        received_information = self.resize_receiver(received_information)

        # adding position embedding
        received_information = received_information + receiver_position

        # second the receiver transformation
        output = self.receiver_transformer(received_information, emitter_position)

        # final resizing to output logits
        output = self.resize_output(output)

        return output

    def compute_loss(self, x, coeff_code_rate, noise_level):
        """
        Compute the loss function.

        Args:
            self: the neural network instance
            x: input tensor of shape (batch_size, seq_length)
            coeff_code_rate: coefficient for code rate adjustment

        Returns:
            loss: loss value
            output: tensor representing the output of the neural network (batch_size, seq_length, nb_class)
        """
        output = self.forward(x, coeff_code_rate, noise_level)

        # loss (cross entropy)
        loss = self.criterion(output, x)

        # accuracy
        accuracy = self.accuracy(output, x)

        return loss, output, accuracy

    def training_step(self, batch, batch_idx):
        """
        Perform a training step for the given batch and batch index.

        Args:
            batch: The input batch for training.
            batch_idx: The index of the batch.

        Returns:
            The loss value after the training step.
        """
        x = batch

        batch_size, seq_lenght = x.shape

        # choose random noise level
        noise_level = 0.2

        # choose random code rate
        coeff_code_rate = self.coeff_code_rate

        loss, _, accuracy = self.compute_loss(x, coeff_code_rate, noise_level)

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)

        return loss
    