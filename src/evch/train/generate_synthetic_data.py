from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from evch.config.loader import build_config_parser, load_config
from evch.data.synthetic import make_synthetic_city
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
    city = make_synthetic_city(
        num_sites=int(env_cfg["num_candidate_sites"]),
        num_zones=int(env_cfg["num_demand_zones"]),
        city_extent_km=float(env_cfg["city_extent_km"]),
        seed=seed,
        base_rate_min=float(demand_cfg["base_rate_min"]),
        base_rate_max=float(demand_cfg["base_rate_max"]),
        zone_scale_std=float(demand_cfg["zone_scale_std"]),
    )
    generator = DemandGenerator(city, demand_cfg, horizon=int(env_cfg["horizon"]), seed=seed)
    frame = generator.generate_supervised_frame(
        num_days=int(data_cfg["num_days"]),
        include_prev_observation=bool(data_cfg.get("include_prev_observation", True)),
    )
    csv_path = output_dir / "synthetic_demand.csv"
    frame.to_csv(csv_path, index=False)
    LOGGER.info("Saved synthetic dataset to %s", csv_path)

    fig_path = output_dir / "synthetic_city.png"
    plt.figure(figsize=(6, 6))
    plt.scatter(city.zone_coords[:, 0], city.zone_coords[:, 1], s=90, label="Demand zones")
    plt.scatter(city.site_coords[:, 0], city.site_coords[:, 1], s=65, marker="^", label="Candidate sites")
    plt.xlabel("X coordinate (km)")
    plt.ylabel("Y coordinate (km)")
    plt.title("Synthetic City Layout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

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

