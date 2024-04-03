"""
Module for training with PyTorch Lightning
"""

import math
import torch
import torch.nn as nn
import lightning.pytorch as pl

import torchmetrics
import matplotlib.pyplot as plt

from deepcodecorrection.utils import generate_random_string

from linear_attention_transformer import LinearAttentionTransformer

torch.set_float32_matmul_precision("high")


class PLTrainer(pl.LightningModule):
    """
    Class for training with PyTorch Lightning
    """

    def __init__(
        self,
        max_dim_input,
        nb_class,
        dim_global=32,
        coeff_code_rate=5.0 / 16.0,
        nb_codebook_size=8,
        dim_global_block=16,
        nb_block_size=16,
        activated_film=False,
    ):
        super().__init__()

        self.dim_global = dim_global
        self.max_dim_input = max_dim_input
        self.nb_class = nb_class
        self.dim_global = dim_global
        self.coeff_code_rate = coeff_code_rate
        self.nb_codebook_size = nb_codebook_size
        self.dim_global_block = dim_global_block
        self.nb_block_size = nb_block_size
        self.activated_film = activated_film

        self.nb_bit = math.ceil(math.log2(nb_class))

        self.random_string = generate_random_string(10)

        # block analysis
        # check if dim_global_block / nb_block_size is an integer
        assert self.dim_global_block % self.nb_block_size == 0, (
            "dim_global_block must be a multiple of nb_block_size"
            + str(self.nb_block_size)
            + " != "
            + str(self.dim_global_block)
        )

        # also check if coeff_code_rate * nb_block_size is an integer
        assert (
            self.coeff_code_rate * self.nb_block_size % 1 == 0
        ), "coeff_code_rate * nb_block_size must be an integer"

        nb_true_bit_per_block = math.ceil(coeff_code_rate * nb_block_size)

        # we compute the index of the block for the true bit to be transfert
        index_block = torch.arange(dim_global_block)
        true_block = index_block % nb_block_size
        true_block = true_block <= nb_true_bit_per_block - 1

        self.index_true_symbol = index_block[true_block]
        self.index_added_symbol = index_block[~true_block]

        # ## encoder
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=dim_global,
        #     nhead=1,
        #     dim_feedforward=dim_global * 4,
        #     batch_first=True,
        # )

        # self.emitter_transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        # self.emitter_transformer = torch.compile(self.emitter_transformer)

        self.emitter_transformer = LinearAttentionTransformer(
            dim=dim_global,
            heads=1,
            depth=6,
            max_seq_len=8192 // 16,
            n_local_attn_heads=1,
            local_attn_window_size=16,
        )

        # decoder layer

        # decoder_layer = nn.TransformerEncoderLayer(
        #     d_model=dim_global,
        #     nhead=1,
        #     dim_feedforward=dim_global * 4,
        #     batch_first=True,
        # )

        # self.receiver_transformer = nn.TransformerEncoder(decoder_layer, num_layers=6)
        # self.receiver_transformer = torch.compile(self.receiver_transformer)

        # film layer for all the projection
        self.film_layer = nn.Sequential(
            nn.Linear(1, dim_global),
            nn.Sigmoid(),
            nn.Linear(dim_global, (dim_global + 1) * 4),
        )

        self.receiver_transformer = LinearAttentionTransformer(
            dim=dim_global,
            heads=1,
            depth=6,
            max_seq_len=8192 // 16,
            n_local_attn_heads=1,
            local_attn_window_size=16,
        )

        # embedding for the position
        self.position_embedding_encoder_emitter = nn.Embedding(
            max_dim_input, dim_global
        )
        self.position_embedding_encoder_receiver = nn.Embedding(
            max_dim_input, dim_global
        )

        # embedding for the actual input
        self.input_embedding_emitter = nn.Embedding(nb_class, dim_global)

        # true bit or added bit
        self.input_embedding_emitter_true_added = nn.Embedding(2, dim_global)

        nb_dim_canal = 3

        # three resizing components
        self.resize_emitter = nn.Linear(dim_global, nb_dim_canal)
        self.resize_receiver = nn.Linear(nb_dim_canal - 1, dim_global)
        self.resize_output = nn.Linear(dim_global, self.nb_bit)

        self.criterion = nn.BCEWithLogitsLoss()

        # accuracy metric
        self.accuracy = torchmetrics.Accuracy(task="binary")

        # init the whole model
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                # m.bias.data.fill_(0.01)

            if isinstance(m, nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

    def encode(self, x, coeff_code_rate, noise_level, noise_film):
        """
        Encodes the input data with noise and quantization, and returns the received information,
          receiver position, and sequence length.

        Parameters:
            x: Tensor - The input data to be encoded.
            coeff_code_rate: float - The coefficient for code rate.
            noise_level: float - The level of noise to be added.
            noise_film: Tensor - The noise information given in film layer

        Returns:
            received_information_quant: Tensor - The received information
                                    after quantization and noise addition.
            receiver_position: Tensor - The position of the receiver.
            seq_length: int - The length of the sequence.
        """
        assert coeff_code_rate <= 1.0

        batch_size, _ = x.shape
        dim_global_block = self.dim_global_block
        assert dim_global_block < self.max_dim_input

        # generate embedding for the input
        x = self.input_embedding_emitter(x)  # batch_size, seq_length, dim_global

        # we create the interleave input
        input_init = torch.zeros(batch_size, dim_global_block, self.dim_global).to(
            x.device
        )
        input_init[:, self.index_true_symbol, :] = x
        input_init[:, self.index_added_symbol, :] = 0

        emitter_position_int = (
            torch.arange(dim_global_block)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to(x.device)
        )
        emitter_position = self.position_embedding_encoder_emitter(
            emitter_position_int
        )  # batch_size, seq_length, dim_global

        true_added_info = torch.zeros(batch_size, dim_global_block).to(x.device)

        true_added_info[:, self.index_true_symbol] = 1
        true_added_info[:, self.index_added_symbol] = 0

        x = (
            input_init
            + emitter_position
            + self.input_embedding_emitter_true_added(true_added_info.long())
        )

        receiver_position = self.position_embedding_encoder_receiver(
            emitter_position_int
        )  # batch_size, dim_global_block, dim_global

        # first the emitter transformation
        transmitted_information = self.emitter_transformer(
            x
        )  # batch_size, dim_global_block, dim_global

        # resize to batch_size, dim_global_block, 1
        transmitted_information = self.resize_emitter(transmitted_information)

        # only power normalization (discretization leads to instability)
        # we impose that the mean of power of transmitted symbol is 1

        # global power normalization
        # quantized = transmitted_information / torch.sqrt(
        #     (transmitted_information.norm(dim=2, keepdim=True, p=2) ** 2).mean(
        #         dim=1, keepdim=True
        #     )
        # )

        quantized = transmitted_information / transmitted_information.norm(
            dim=2, keepdim=True, p=2
        )

        # print("power per batch element :", quantized)

        quantized = quantized[:, :, :-1]

        # adding noise
        noisy_transmitted_information = (
            quantized + torch.randn_like(quantized) * noise_level.unsqueeze(2)
        )

        # quantized_after_noise, indices_after_noise = self.quantizer_after_noise(
        #     noisy_transmitted_information
        # )
        quantized_after_noise = noisy_transmitted_information

        received_information_quant = quantized_after_noise

        # if inference:
        #     count_init_symbol = torch.bincount(init_symbol.view(-1).long())
        #     print("count_init_symbol: ", count_init_symbol)

        return received_information_quant, receiver_position

    def decode(self, received_information_quant, receiver_position, noise_film):
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

        return output[:, self.index_true_symbol, :], received_information_quant

    def forward(self, x, coeff_code_rate, noise_level):
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

        noise_level =noise_level.unsqueeze(1).float()

        # generate embedding for the input
        noise_film = self.film_layer(noise_level)

        received_information_quant, receiver_position = self.encode(
            x, coeff_code_rate, noise_level, noise_film
        )

        output, received_information_quant = self.decode(
            received_information_quant, receiver_position, noise_film
        )

        return output, received_information_quant

    def compute_loss(self, x, coeff_code_rate, noise_level, bit_corresponding):
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
        output, quantized_value = self.forward(x, coeff_code_rate, noise_level)

        # loss (cross entropy)
        loss = self.criterion(output, bit_corresponding)

        # accuracy
        accuracy = self.accuracy(output, bit_corresponding)

        return loss, quantized_value, accuracy

    def training_step(self, batch, _):
        """
        Perform a training step for the given batch and batch index.

        Args:
            batch: The input batch for training.
            batch_idx: The index of the batch.

        Returns:
            The loss value after the training step.
        """

        x, bit_corresponding, noise_level = batch

        # choose random code rate
        coeff_code_rate = self.coeff_code_rate

        loss, _, accuracy = self.compute_loss(
            x, coeff_code_rate, noise_level, bit_corresponding
        )

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        self.log("train_error", 1 - accuracy)
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
        x, bit_corresponding, noise_level = batch

        # choose random code rate
        coeff_code_rate = self.coeff_code_rate

        loss, quantized_value, accuracy = self.compute_loss(
            x, coeff_code_rate, noise_level, bit_corresponding
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

        path_name_image = "/home/images/" + self.random_string + ".png"

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
