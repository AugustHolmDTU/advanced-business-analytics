from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evch.config.loader import build_config_parser, load_config
from evch.data.city import build_city
from evch.data.features import MODEL_FEATURE_COLUMNS
from evch.envs.demand import DemandGenerator
from evch.models.common import Standardizer, make_regression_loaders
from evch.models.gaussian_nll import GaussianNLLRegressor
from evch.models.plotting import plot_prediction_intervals
from evch.models.quantile import QuantileRegressor
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging
from evch.utils.seeding import set_global_seed
from evch.utils.wandb import init_wandb

LOGGER = logging.getLogger(__name__)


def _train_epoch(model: Any, loader: Any, optimizer: Any, model_name: str, device: str, grad_clip_norm: float) -> float:
    model.train()
    losses: list[float] = []
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        if model_name == "gaussian":
            mu, log_sigma = model(features)
            loss = model.gaussian_nll(mu, log_sigma, targets)
        else:
            predictions = model(features)
            loss = model.quantile_loss(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def _validate(model: Any, loader: Any, model_name: str, device: str) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            if model_name == "gaussian":
                mu, log_sigma = model(features)
                loss = model.gaussian_nll(mu, log_sigma, targets)
            else:
                predictions = model(features)
                loss = model.quantile_loss(predictions, targets)
            losses.append(float(loss.item()))
    return float(np.mean(losses))


def _build_dataset(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    env_cfg = config["environment"]
    demand_cfg = config["demand"]
    data_cfg = config["data"]
    seed = int(config.get("seed", 0))

    city = build_city(env_config=env_cfg, demand_config=demand_cfg, seed=seed)
    generator = DemandGenerator(city, demand_cfg, horizon=int(env_cfg["horizon"]), seed=seed)
    frame = generator.generate_supervised_frame(
        num_days=int(data_cfg["num_days"]),
        include_prev_observation=bool(data_cfg.get("include_prev_observation", True)),
    )
    features = frame[MODEL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = frame["target_demand"].to_numpy(dtype=np.float32).reshape(-1, 1)
    return features, targets, MODEL_FEATURE_COLUMNS


def _split_dataset(features: np.ndarray, targets: np.ndarray, val_split: float, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(features))
    split_idx = int(len(indices) * (1.0 - val_split))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    return features[train_idx], targets[train_idx], features[val_idx], targets[val_idx]


def main() -> None:
    parser = build_config_parser("Train uncertainty-aware demand models on synthetic data.")
    args = parser.parse_args()
    config = load_config(args.config)

    configure_logging(config.get("logging", {}).get("level", "INFO"))
    seed = int(config.get("seed", 0))
    set_global_seed(seed)

    experiment_cfg = config["experiment"]
    model_cfg = config["model"]
    training_cfg = model_cfg["training"]
    output_dir = ensure_dir(Path(experiment_cfg["output_root"]) / experiment_cfg["name"] / "uncertainty")
    run = init_wandb(config=config, job_type="train_uncertainty", run_name=f"{experiment_cfg['name']}_{model_cfg['name']}")

    features, targets, feature_columns = _build_dataset(config)
    x_train, y_train, x_val, y_val = _split_dataset(
        features=features,
        targets=targets,
        val_split=float(config["data"]["val_split"]),
        seed=seed,
    )
    standardizer = Standardizer.fit(x_train)
    x_train_std = standardizer.transform(x_train)
    x_val_std = standardizer.transform(x_val)
    train_loader, val_loader = make_regression_loaders(
        x_train_std,
        y_train.astype(np.float32),
        x_val_std,
        y_val.astype(np.float32),
        batch_size=int(training_cfg["batch_size"]),
    )

    device = str(training_cfg.get("device", "cpu"))
    if model_cfg["name"] == "gaussian":
        model = GaussianNLLRegressor(
            input_dim=x_train_std.shape[1],
            hidden_dims=list(model_cfg["hidden_dims"]),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(device)
    else:
        model = QuantileRegressor(
            input_dim=x_train_std.shape[1],
            hidden_dims=list(model_cfg["hidden_dims"]),
            quantiles=list(model_cfg["quantiles"]),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(int(training_cfg["epochs"])):
        train_loss = _train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            model_name=model_cfg["name"],
            device=device,
            grad_clip_norm=float(training_cfg.get("gradient_clip_norm", 5.0)),
        )
        val_loss = _validate(model=model, loader=val_loader, model_name=model_cfg["name"], device=device)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        run.log({"uncertainty/train_loss": train_loss, "uncertainty/val_loss": val_loss, "epoch": epoch})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / str(model_cfg["checkpoint_name"])
            metadata = {
                "feature_columns": feature_columns,
                "standardizer_mean": standardizer.mean.tolist(),
                "standardizer_std": standardizer.std.tolist(),
                "model_name": model_cfg["name"],
            }
            model.save_checkpoint(checkpoint_path, metadata=metadata)

    if model_cfg["name"] == "gaussian":
        predictions = model.predict(x_val_std, device=device)
        center = predictions["mu"]
        lower = center - 1.645 * predictions["sigma"]
        upper = center + 1.645 * predictions["sigma"]
        metrics = {
            "val_loss": best_val_loss,
            "mae": float(np.mean(np.abs(center - y_val[:, 0]))),
            "rmse": float(np.sqrt(np.mean((center - y_val[:, 0]) ** 2))),
            "interval_coverage_90": float(np.mean((y_val[:, 0] >= lower) & (y_val[:, 0] <= upper))),
            "interval_width_90": float(np.mean(upper - lower)),
        }
        try:
            plot_prediction_intervals(
                y_true=y_val[:, 0],
                center=center,
                lower=lower,
                upper=upper,
                path=output_dir / "gaussian_intervals.png",
                title="Gaussian predictive intervals",
            )
        except Exception as exc:
            LOGGER.warning("Skipping Gaussian interval plot because matplotlib is unavailable: %s", exc)
    else:
        predictions = model.predict(x_val_std, device=device)
        center = predictions["q50"]
        lower = predictions["q05"]
        upper = predictions["q95"]
        metrics = {
            "val_loss": best_val_loss,
            "mae": float(np.mean(np.abs(center - y_val[:, 0]))),
            "rmse": float(np.sqrt(np.mean((center - y_val[:, 0]) ** 2))),
            "interval_coverage_90": float(np.mean((y_val[:, 0] >= lower) & (y_val[:, 0] <= upper))),
            "interval_width_90": float(np.mean(upper - lower)),
            "crossing_rate": float(predictions["crossing_rate"]),
        }
        try:
            plot_prediction_intervals(
                y_true=y_val[:, 0],
                center=center,
                lower=lower,
                upper=upper,
                path=output_dir / "quantile_intervals.png",
                title="Quantile predictive intervals",
            )
        except Exception as exc:
            LOGGER.warning("Skipping quantile interval plot because matplotlib is unavailable: %s", exc)

    run.log({f"uncertainty/{key}": value for key, value in metrics.items()})
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "history.json", {"history": history})
    LOGGER.info("Finished %s training with metrics: %s", model_cfg["name"], metrics)
    run.finish()


if __name__ == "__main__":
    main()
