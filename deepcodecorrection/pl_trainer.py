"""
Module for training with PyTorch Lightning
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl


class PLTrainer(pl.Trainer):
    """
    Class for training with PyTorch Lightning
    """

    def __init__(self, dim_global, max_dim_input, nb_class):
        super().__init__()

        self.dim_global = dim_global
        self.max_dim_input = max_dim_input
        self.nb_class = nb_class

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
        self.input_embedding_receiver = nn.Embedding(nb_class, dim_global)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, coeff_code_rate):
        assert coeff_code_rate > 1.0

        batch_size, seq_length = x.shape
        dim_intermediate = torch.round(seq_length * coeff_code_rate).int()

        assert dim_intermediate < self.max_dim_input

        # generate embedding for the input
        x = self.input_embedding_emitter(x)  # batch_size, seq_length, dim_global

        x_position = (
            torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        )
        x_position = self.position_embedding_encoder_emitter(
            x_position
        )  # batch_size, seq_length, dim_global

        x = x + x_position

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

        # discretization with FSQ
        # TODO

        # adding noise
        # TODO
        received_information = transmitted_information

        embedding_receiver = self.input_embedding_receiver(received_information)
        received_information = received_information + embedding_receiver

        # second the receiver transformation
        output = self.receiver_transformer(received_information, receiver_position)

        return output
