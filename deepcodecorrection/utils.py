"""
Module with the different utilities (MLP)
"""


from torch import nn


class MLP(nn.Module):
    """
    Simple MLP with one hidden layer (multi-layer perceptron)
    """

    # MLP with LayerNorm
    def __init__(
        self,
        in_dim,
        out_dim=128,
        hidden_dim=128,
        norm_type="LayerNorm",
    ):
        """
        in_dim: input dimension
        out_dim: output dimension
        hidden_dim: number of nodes in the hidden layer
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', ...
        """

        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, vector):
        """
        Simple forward pass
        """
        return self.model(vector)


import random
import string

def generate_random_string(length):
    """
    Generate a random string of a specified length.

    :param length: The length of the random string to be generated.
    :type length: int
    :return: The randomly generated string.
    :rtype: str
    """
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))
