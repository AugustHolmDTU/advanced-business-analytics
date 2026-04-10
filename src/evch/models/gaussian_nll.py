from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from evch.models.common import make_mlp


class GaussianNLLRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.network = make_mlp(input_dim, hidden_dims, 2, dropout=dropout)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.network(inputs)
        mu = outputs[:, :1]
        log_sigma = torch.clamp(outputs[:, 1:], min=-5.0, max=3.0)
        return mu, log_sigma

    @staticmethod
    def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_sigma)
        squared_error = ((targets - mu) / sigma) ** 2
        loss = 0.5 * squared_error + log_sigma
        return loss.mean()

    def predict(self, inputs: np.ndarray, device: str = "cpu") -> dict[str, np.ndarray]:
        self.eval()
        with torch.no_grad():
            tensor_inputs = torch.from_numpy(inputs).to(device)
            mu, log_sigma = self(tensor_inputs)
            sigma = torch.exp(log_sigma)
        return {
            "mu": mu.cpu().numpy().squeeze(-1),
            "sigma": sigma.cpu().numpy().squeeze(-1),
        }

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any]) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "metadata": metadata,
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, map_location: str = "cpu") -> tuple["GaussianNLLRegressor", dict[str, Any]]:
        payload = torch.load(path, map_location=map_location)
        model = cls(
            input_dim=int(payload["input_dim"]),
            hidden_dims=list(payload["hidden_dims"]),
            dropout=float(payload.get("dropout", 0.0)),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload.get("metadata", {})

