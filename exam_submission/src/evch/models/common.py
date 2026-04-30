from __future__ import annotations

from torch import nn


def make_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)
