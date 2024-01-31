"""
Module for training with PyTorch Lightning
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl

import torchmetrics

from vector_quantize_pytorch import FSQ, VectorQuantize


class PLTrainer(pl.LightningModule):
    """
    Class for training with PyTorch Lightning
    """

    def __init__(
        self,
        max_dim_input,
        nb_class,
        dim_global=32,
        coeff_code_rate=1.3,
        noise_level=1.0,
    ):
        super().__init__()

        self.dim_global = dim_global
        self.max_dim_input = max_dim_input
        self.nb_class = nb_class
        self.dim_global = dim_global
        self.coeff_code_rate = coeff_code_rate
        self.noise_level = noise_level

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_global,
            nhead=8,
            dim_feedforward=dim_global * 4,
            batch_first=True,
        )

        self.emitter_transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_global,
            nhead=8,
            dim_feedforward=dim_global * 4,
            batch_first=True,
        )

        self.receiver_transformer = nn.TransformerEncoder(decoder_layer, num_layers=6)

        # embedding for the position
        self.position_embedding_encoder_emitter = nn.Embedding(
            max_dim_input, dim_global
        )
        self.position_embedding_encoder_receiver = nn.Embedding(
            max_dim_input, dim_global
        )

        # embedding for the actual input
        self.input_embedding_emitter = nn.Embedding(nb_class, dim_global)

        # FSQ
        levels = [8]  # see 4.1 and A.4.1 in the paper
        self.quantizer = FSQ(levels)

        self.vq = VectorQuantize(dim=1, codebook_size=8, freeze_codebook=True)

        # three resizing components
        self.resize_emitter = nn.Linear(dim_global, 1)
        self.resize_receiver = nn.Linear(1, dim_global)
        self.resize_output = nn.Linear(dim_global, nb_class)

        self.criterion = nn.CrossEntropyLoss()

        # accuracy metric
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=nb_class)

        # init the whole model
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

            if isinstance(m, nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

    def forward(self, x, coeff_code_rate, noise_level, inference=False):
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
        dim_intermediate = int(seq_length * coeff_code_rate)

        assert dim_intermediate < self.max_dim_input

        # generate embedding for the input
        x = self.input_embedding_emitter(x)  # batch_size, seq_length, dim_global

        # pad with zeros to obtain dimension of batch_size, dim_intermediate, dim_global
        x = torch.cat(
            [
                x,
                torch.zeros(
                    batch_size, dim_intermediate - seq_length, self.dim_global
                ).to(x.device),
            ],
            dim=1,
        )

        emitter_position_int = (
            torch.arange(dim_intermediate)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to(x.device)
        )
        emitter_position = self.position_embedding_encoder_emitter(
            emitter_position_int
        )  # batch_size, seq_length, dim_global

        x = x + emitter_position

        receiver_position = self.position_embedding_encoder_receiver(
            emitter_position_int
        )  # batch_size, dim_intermediate, dim_global

        # first the emitter transformation
        transmitted_information = self.emitter_transformer(
            x
        )  # batch_size, dim_intermediate, dim_global

        # resize to batch_size, dim_intermediate, 1
        transmitted_information = self.resize_emitter(transmitted_information)

        # adding noise
        noisy_transmitted_information = (
            transmitted_information
            + torch.randn_like(transmitted_information) * noise_level
        )

        # discretization with FSQ
        quantized, indices, commit_loss = self.vq(noisy_transmitted_information)

        if inference == False:
            received_information_quant = noisy_transmitted_information
        else:
            received_information_quant = quantized

            # compute difference between noisy quant and non noisy quantized
            non_noisy_quant, non_noisy_indices, _ = self.vq(transmitted_information)

            print(
                "shuffle level validation: ",
                (non_noisy_indices.long() == indices.long()).float().mean(),
            )

        # resize to global dimension
        received_information = self.resize_receiver(received_information_quant)

        # adding position embedding
        received_information = received_information + receiver_position

        # second the receiver transformation
        output = self.receiver_transformer(received_information)

        # final resizing to output logits
        output = self.resize_output(output)

        return output[:, :seq_length, :], commit_loss

    def compute_loss(self, x, coeff_code_rate, noise_level, inference=False):
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
        output, commit_loss = self.forward(x, coeff_code_rate, noise_level, inference)

        # loss (cross entropy)
        loss = self.criterion(output.permute(0, 2, 1), x)

        # accuracy
        accuracy = self.accuracy(output.permute(0, 2, 1), x)

        return loss, commit_loss, output, accuracy

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

        # choose random code rate
        coeff_code_rate = self.coeff_code_rate

        loss, commit_loss, _, accuracy = self.compute_loss(
            x, coeff_code_rate, self.noise_level
        )

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        self.log("commit_loss", commit_loss)

        return loss + commit_loss / 15.0

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step for the given batch and batch index.

        Args:
            batch: The input batch for validation.
            batch_idx: The index of the batch.

        Returns:
            The loss value after the validation step.
        """
        x = batch

        batch_size, seq_lenght = x.shape

        # choose random code rate
        coeff_code_rate = self.coeff_code_rate

        loss, commit_loss, _, accuracy = self.compute_loss(
            x, coeff_code_rate, self.noise_level, inference=True
        )

        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Args:
            self: the neural network instance

        Returns:
            The optimizer and the learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train_loss",
            },
        }
