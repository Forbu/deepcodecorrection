import torch

from deepcodecorrection.pl_trainer import PLTrainer


# Generates embedding for the input, emitter position, transmitter position, and receiver position
def test_generates_embedding(self):
    # Create an instance of the PLTrainer class
    trainer = PLTrainer(
        max_dim_input=100, nb_class=10, dim_global=32, coeff_code_rate=1.3
    )

    # Create a sample input tensor
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # Set the coefficient for code rate adjustment
    coeff_code_rate = 1.5

    # Set the noise level
    noise_level = 0.1

    # Invoke the forward method
    output = trainer.forward(x, coeff_code_rate, noise_level)

    # Assert that embedding for the input, emitter position, transmitter position, and receiver position are generated
    assert output.shape == (2, 3, 32)
    assert trainer.input_embedding_emitter.weight.shape == (10, 32)
    assert trainer.position_embedding_encoder_emitter.weight.shape == (100, 32)
    assert trainer.position_embedding_decoder_emitter.weight.shape == (100, 32)
    assert trainer.position_embedding_encoder_receiver.weight.shape == (100, 32)


# Performs emitter transformation on the input
def test_emitter_transformation(self):
    # Create an instance of the PLTrainer class
    trainer = PLTrainer(
        max_dim_input=100, nb_class=10, dim_global=32, coeff_code_rate=1.3
    )

    # Create a sample input tensor
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # Set the coefficient for code rate adjustment
    coeff_code_rate = 1.5

    # Set the noise level
    noise_level = 0.1

    # Invoke the forward method
    output = trainer.forward(x, coeff_code_rate, noise_level)

    # Assert that emitter transformation is performed on the input
    assert output.shape == (2, 3, 32)
    assert trainer.emitter_transformer.d_model == 32
    assert trainer.emitter_transformer.nhead == 8
    assert trainer.emitter_transformer.num_encoder_layers == 6
    assert trainer.emitter_transformer.num_decoder_layers == 6
    assert trainer.emitter_transformer.dim_feedforward == 512


# Resizes the output of emitter transformation to batch_size, dim_intermediate, 1
def test_resize_emitter_transformation_output(self):
    # Create an instance of the PLTrainer class
    trainer = PLTrainer(
        max_dim_input=100, nb_class=10, dim_global=32, coeff_code_rate=1.3
    )

    # Create a sample input tensor
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # Set the coefficient for code rate adjustment
    coeff_code_rate = 1.5

    # Set the noise level
    noise_level = 0.1

    # Invoke the forward method
    output = trainer.forward(x, coeff_code_rate, noise_level)

    # Assert that the output of emitter transformation is resized to batch_size, dim_intermediate, 1
    assert output.shape == (2, 3, 1)
    assert trainer.resize_emitter.in_features == 32
    assert trainer.resize_emitter.out_features == 1
