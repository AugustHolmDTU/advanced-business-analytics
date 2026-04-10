from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_intervals(
    y_true: np.ndarray,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    path: str | Path,
    title: str,
) -> None:
    order = np.argsort(y_true)
    y_true = y_true[order]
    center = center[order]
    lower = lower[order]
    upper = upper[order]

    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="True demand", linewidth=1.5)
    plt.plot(center, label="Center prediction", linewidth=1.3)
    plt.fill_between(np.arange(len(y_true)), lower, upper, alpha=0.25, label="Prediction interval")
    plt.title(title)
    plt.xlabel("Sorted validation sample")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
