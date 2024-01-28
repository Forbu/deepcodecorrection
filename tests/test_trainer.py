import torch

from deepcodecorrection.pl_trainer import PLTrainer


# Resizes the output of emitter transformation to batch_size, dim_intermediate, 1
def test_forward_transformation_output(self):
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

