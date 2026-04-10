from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from evch.models.common import make_mlp


class QuantileRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], quantiles: list[float], dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.quantiles = quantiles
        self.dropout = dropout
        self.network = make_mlp(input_dim, hidden_dims, len(quantiles), dropout=dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = []
        for index, quantile in enumerate(self.quantiles):
            errors = targets[:, 0] - predictions[:, index]
            losses.append(torch.maximum(quantile * errors, (quantile - 1.0) * errors))
        stacked = torch.stack(losses, dim=1)
        return stacked.mean()

    def predict(self, inputs: np.ndarray, device: str = "cpu") -> dict[str, np.ndarray]:
        self.eval()
        with torch.no_grad():
            tensor_inputs = torch.from_numpy(inputs).to(device)
            preds = self(tensor_inputs).cpu().numpy()
        outputs = {f"q{int(q * 100):02d}": preds[:, idx] for idx, q in enumerate(self.quantiles)}
        outputs["crossing_rate"] = np.mean(np.any(np.diff(preds, axis=1) < 0.0, axis=1))
        return outputs

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any]) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "quantiles": self.quantiles,
            "dropout": self.dropout,
            "metadata": metadata,
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, map_location: str = "cpu") -> tuple["QuantileRegressor", dict[str, Any]]:
        payload = torch.load(path, map_location=map_location)
        model = cls(
            input_dim=int(payload["input_dim"]),
            hidden_dims=list(payload["hidden_dims"]),
            quantiles=list(payload["quantiles"]),
            dropout=float(payload.get("dropout", 0.0)),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload.get("metadata", {})

