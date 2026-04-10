from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from evch.config.loader import build_config_parser, load_config
from evch.data.city import build_city
from evch.envs.demand import DemandGenerator
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging
from evch.utils.seeding import set_global_seed

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = build_config_parser("Generate synthetic demand data for uncertainty modeling.")
    args = parser.parse_args()
    config = load_config(args.config)

    configure_logging(config.get("logging", {}).get("level", "INFO"))
    seed = int(config.get("seed", 0))
    set_global_seed(seed)

    env_cfg = config["environment"]
    demand_cfg = config["demand"]
    data_cfg = config["data"]
    experiment_cfg = config["experiment"]

    output_dir = ensure_dir(Path(experiment_cfg["output_root"]) / experiment_cfg["name"] / "data")
    city = build_city(env_config=env_cfg, demand_config=demand_cfg, seed=seed)
    generator = DemandGenerator(city, demand_cfg, horizon=int(env_cfg["horizon"]), seed=seed)
    frame = generator.generate_supervised_frame(
        num_days=int(data_cfg["num_days"]),
        include_prev_observation=bool(data_cfg.get("include_prev_observation", True)),
    )
    csv_path = output_dir / "synthetic_demand.csv"
    frame.to_csv(csv_path, index=False)
    LOGGER.info("Saved synthetic dataset to %s", csv_path)

    fig_path = output_dir / "synthetic_city.png"
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.scatter(city.zone_coords[:, 0], city.zone_coords[:, 1], s=90, label="Demand zones")
        plt.scatter(city.site_coords[:, 0], city.site_coords[:, 1], s=65, marker="^", label="Candidate sites")
        plt.xlabel("X coordinate (km)")
        plt.ylabel("Y coordinate (km)")
        plt.title("City Layout")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()
    except Exception as exc:
        LOGGER.warning("Skipping city-layout plot because matplotlib is unavailable: %s", exc)

    write_json(
        output_dir / "dataset_metadata.json",
        {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "seed": seed,
        },
    )


if __name__ == "__main__":
    main()
