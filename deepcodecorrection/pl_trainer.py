"""
Module for training with PyTorch Lightning
"""
import math
import torch
import torch.nn as nn
import lightning.pytorch as pl

import torchmetrics
import matplotlib.pyplot as plt

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
        noise_level=0.0,
        nb_codebook_size=8,
    ):
        super().__init__()

        self.dim_global = dim_global
        self.max_dim_input = max_dim_input
        self.nb_class = nb_class
        self.dim_global = dim_global
        self.coeff_code_rate = coeff_code_rate
        self.noise_level = noise_level
        self.nb_codebook_size = nb_codebook_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_global,
            nhead=8,
            dim_feedforward=dim_global * 4,
            batch_first=True,
        )

        self.emitter_transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.emitter_transformer = torch.compile(self.emitter_transformer)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_global,
            nhead=8,
            dim_feedforward=dim_global * 4,
            batch_first=True,
        )

        self.receiver_transformer = nn.TransformerEncoder(decoder_layer, num_layers=6)
        self.receiver_transformer = torch.compile(self.receiver_transformer)

        # embedding for the position
        self.position_embedding_encoder_emitter = nn.Embedding(
            max_dim_input, dim_global
        )
        self.position_embedding_encoder_receiver = nn.Embedding(
            max_dim_input, dim_global
        )

        # embedding for the actual input
        self.input_embedding_emitter = nn.Embedding(nb_class, dim_global)

        nb_dim_canal = 3

        # three resizing components
        self.resize_emitter = nn.Linear(dim_global, nb_dim_canal)
        self.resize_receiver = nn.Linear(nb_dim_canal - 1, dim_global)
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

    def encode(self, x, coeff_code_rate, noise_level, inference=False):
        """
        Encodes the input data with noise and quantization, and returns the received information,
          receiver position, and sequence length.

        Parameters:
            x: Tensor - The input data to be encoded.
            coeff_code_rate: float - The coefficient for code rate.
            noise_level: float - The level of noise to be added.
            inference: bool - Flag indicating whether the function is used for inference.

        Returns:
            received_information_quant: Tensor - The received information
                                    after quantization and noise addition.
            receiver_position: Tensor - The position of the receiver.
            seq_length: int - The length of the sequence.
        """
        assert coeff_code_rate > 1.0

        batch_size, seq_length = x.shape
        dim_intermediate = int(seq_length * coeff_code_rate)

        assert dim_intermediate < self.max_dim_input

        init_symbol = x.clone()

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

        # only power normalization (discretization leads to instability)
        quantized = (
            transmitted_information
            / transmitted_information.norm(dim=2, keepdim=True)
            * 1.41 # sqrt(2)
        )

        quantized = quantized[:, :, :-1]

        # adding noise
        noisy_transmitted_information = (
            quantized + torch.randn_like(quantized) * noise_level
        )

        # quantized_after_noise, indices_after_noise = self.quantizer_after_noise(
        #     noisy_transmitted_information
        # )
        quantized_after_noise = noisy_transmitted_information

        received_information_quant = quantized_after_noise

        if inference:
            count_init_symbol = torch.bincount(init_symbol.view(-1).long())
            print("count_init_symbol: ", count_init_symbol)

        return received_information_quant, receiver_position, seq_length

    def decode(self, received_information_quant, receiver_position, seq_length):
        """
        Decode the received information by resizing to global dimension, adding position embedding,
          applying receiver transformation, and resizing to output logits.

        Parameters:
            received_information_quant: input information to be decoded
            receiver_position: position embedding for the receiver
            seq_length: length of the output sequence
        Returns:
            output[:, :seq_length, :]: decoded output with specified sequence length
        """

        # resize to global dimension
        received_information = self.resize_receiver(received_information_quant)

        # adding position embedding
        received_information = received_information + receiver_position

        # second the receiver transformation
        output = self.receiver_transformer(received_information)

        # final resizing to output logits
        output = self.resize_output(output)

        return output[:, :seq_length, :], received_information_quant

    def forward(self, x, coeff_code_rate, noise_level, inference=False):
        """
        Forward pass of the neural network model.

        Args:
            self: the neural network instance
            x: input tensor of shape (batch_size, seq_length)
            coeff_code_rate: coefficient for code rate adjustment

        Returns:
            output: tensor representing the output of the neural network
                    (batch_size, seq_length, nb_class)
        """

        received_information_quant, receiver_position, seq_length = self.encode(
            x, coeff_code_rate, noise_level, inference
        )

        output, received_information_quant = self.decode(
            received_information_quant, receiver_position, seq_length
        )

        return output, received_information_quant

    def compute_loss(self, x, coeff_code_rate, noise_level, inference=False):
        """
        Compute the loss function.

        Args:
            self: the neural network instance
            x: input tensor of shape (batch_size, seq_length)
            coeff_code_rate: coefficient for code rate adjustment

        Returns:
            loss: loss value
            output: tensor representing the output of the neural network
            (batch_size, seq_length, nb_class)
        """
        output, quantized_value = self.forward(
            x, coeff_code_rate, noise_level, inference
        )

        # loss (cross entropy)
        loss = self.criterion(output.permute(0, 2, 1), x)

        # accuracy
        accuracy = self.accuracy(output.permute(0, 2, 1), x)

        return loss, quantized_value, accuracy

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

        loss, _, accuracy = self.compute_loss(x, coeff_code_rate, self.noise_level)

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step for the given batch and batch index.

        Args:
            batch: The input batch for validation.
            batch_idx: The index of the batch.

        Returns:
            The loss value after the validation step.
        """
        self.eval()
        x = batch

        batch_size, seq_lenght = x.shape

        # choose random code rate
        coeff_code_rate = self.coeff_code_rate

        loss, quantized_value, accuracy = self.compute_loss(
            x, coeff_code_rate, self.noise_level, inference=True
        )

        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)

        # we take a look at the quantize value
        quantized_value = quantized_value.view(-1, quantized_value.shape[-1])
        quantized_value = quantized_value.detach().cpu().numpy()

        # we plot the quantized value
        self.plot_quantized_value(quantized_value)

        self.train()

    def plot_quantized_value(self, quantized_value):
        """
        Plot the quantized value.

        Args:
            quantized_value: the quantized value
        """
        

        # plot all the value in the quantized value (nb_point, 2)
        plt.scatter(quantized_value[:500, 0], quantized_value[:500, 1])

        path_name_image = "quantized_value_noadd.png"

        # plot the quantized value
        plt.savefig(path_name_image)
        plt.close()

        # we take only the first graph
        img = plt.imread(path_name_image)[:, :, :3]
        img = img.transpose((2, 0, 1))

        # log image
        self.logger.experiment.add_image("quantized_value", img, self.current_epoch)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        We separate the learning for the encoder and the decoder.
        We want a very low learning rate for the encoder (x10 for the decoder).

        Args:
            self: the neural network instance

        Returns:
            The optimizer and the learning rate scheduler.
        """

        optimizer_all = torch.optim.AdamW(self.parameters(), lr=5e-4)

        return (optimizer_all,)
