from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from env.simple_env import SimpleDemandEnv, SimpleEnvConfig

try:
    import wandb  # type: ignore[import-not-found]
except ImportError:
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate EV charging demand over time (no RL).")
    parser.add_argument("--days", type=int, default=3, help="Number of days to simulate.")
    parser.add_argument("--noise-std", type=float, default=4.0, help="Gaussian noise std for demand.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _init_wandb(days: int, steps: int, noise_std: float, mode: str):
    if mode == "disabled":
        return None
    if wandb is None:
        raise RuntimeError("wandb is required for online/offline mode. Install with: pip install wandb")

    return wandb.init(
        project="ev_resilience_rl",
        job_type="simulate_demand",
        mode=mode,
        config={
            "days": days,
            "steps": steps,
            "base_demand": 40,
            "scale": 80,
            "noise_std": noise_std,
            "morning_peak_hour": 8,
            "afternoon_peak_hour": 17,
        },
    )


def simulate(days: int, noise_std: float, wandb_mode: str, seed: int) -> list[dict[str, float | int]]:
    config = SimpleEnvConfig(days=days, noise_std=noise_std)
    env = SimpleDemandEnv(config=config, seed=seed)
    env.reset()

    records: list[dict[str, float | int]] = []
    run = _init_wandb(days=days, steps=config.total_steps, noise_std=noise_std, mode=wandb_mode)

    for _ in range(config.total_steps):
        obs = env.step()
        records.append(obs)

        if run is not None:
            run.log(
                {
                    "env/demand": float(obs["demand"]),
                    "env/hour_of_day": int(obs["hour_of_day"]),
                    "env/day_index": int(obs["day_index"]),
                    "env/global_hour": int(obs["global_hour"]),
                },
                step=int(obs["global_hour"]),
            )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "demand_curve.png"

    global_hours = [int(r["global_hour"]) for r in records]
    demands = [float(r["demand"]) for r in records]

    plt.figure(figsize=(10, 4))
    plt.plot(global_hours, demands, linewidth=2)
    plt.xlabel("Global hour")
    plt.ylabel("Demand")
    plt.title("Synthetic EV charging demand over 3 days")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(image_path, dpi=150)
    plt.close()

    if run is not None:
        table = wandb.Table(columns=["global_hour", "hour_of_day", "day_index", "demand"])
        for r in records:
            table.add_data(int(r["global_hour"]), int(r["hour_of_day"]), int(r["day_index"]), float(r["demand"]))

        run.log(
            {
                "plots/demand_curve": wandb.plot.line(
                    table,
                    "global_hour",
                    "demand",
                    title="Demand curve over 3 days",
                ),
                "tables/demand_timeseries": table,
                "plots/demand_curve_png": wandb.Image(str(image_path)),
            }
        )

        demand_array = np.array(demands, dtype=np.float32)
        run.log(
            {
                "summary/max_demand": float(np.max(demand_array)),
                "summary/min_demand": float(np.min(demand_array)),
                "summary/mean_demand": float(np.mean(demand_array)),
            }
        )
        run.summary["summary/max_demand"] = float(np.max(demand_array))
        run.summary["summary/min_demand"] = float(np.min(demand_array))
        run.summary["summary/mean_demand"] = float(np.mean(demand_array))
        run.finish()

    return records


def main() -> None:
    args = parse_args()
    simulate(days=args.days, noise_std=args.noise_std, wandb_mode=args.wandb_mode, seed=args.seed)


if __name__ == "__main__":
    main()
