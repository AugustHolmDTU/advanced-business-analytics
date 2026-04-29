from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_reward_value(run_name: str, prefix: str) -> float | None:
    if not run_name.startswith(prefix):
        return None
    token = run_name[len(prefix) :]
    try:
        return float(token.replace("p", "."))
    except ValueError:
        return None


def collect_rows(root: Path, sweep_kind: str) -> list[dict[str, object]]:
    if sweep_kind == "mcs_cost":
        prefix = "mobile_mcs_line_abc_sweep_mcs_cost_"
    elif sweep_kind == "queue_cost":
        prefix = "mobile_mcs_line_abc_sweep_queue_cost_"
    else:
        raise ValueError(f"Unsupported sweep kind: {sweep_kind}")

    rows: list[dict[str, object]] = []
    outputs_dir = root / "outputs"
    for run_dir in sorted(outputs_dir.glob(f"{prefix}*")):
        summary_path = run_dir / "mobile_rl_comparison_multiseed" / "aggregate_summary.json"
        if not summary_path.exists():
            continue
        reward_value = parse_reward_value(run_dir.name, prefix)
        if reward_value is None:
            continue
        payload = json.loads(summary_path.read_text())
        rows.append(
            {
                "sweep_kind": sweep_kind,
                "run_name": run_dir.name,
                "reward_value": reward_value,
                "num_rollouts": payload.get("num_rollouts"),
                "mean_queue_length_mean": payload.get("mean_queue_length_mean"),
                "mean_queue_length_std": payload.get("mean_queue_length_std"),
                "mean_queue_wait_minutes_mean": payload.get("mean_queue_wait_minutes_mean"),
                "mean_queue_wait_minutes_std": payload.get("mean_queue_wait_minutes_std"),
                "peak_queue_length_mean": payload.get("peak_queue_length_mean"),
                "peak_queue_wait_minutes_mean": payload.get("peak_queue_wait_minutes_mean"),
                "mean_active_mobile_stations_mean": payload.get("mean_active_mobile_stations_mean"),
                "summary_path": str(summary_path.relative_to(root)),
            }
        )
    rows.sort(key=lambda row: float(row["reward_value"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sweep_kind",
        "run_name",
        "reward_value",
        "num_rollouts",
        "mean_queue_length_mean",
        "mean_queue_length_std",
        "mean_queue_wait_minutes_mean",
        "mean_queue_wait_minutes_std",
        "peak_queue_length_mean",
        "peak_queue_wait_minutes_mean",
        "mean_active_mobile_stations_mean",
        "summary_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect multi-seed held-out comparison summaries for reward sweeps.")
    parser.add_argument("--sweep-kind", choices=["mcs_cost", "queue_cost", "all"], default="all")
    parser.add_argument("--output-dir", default="outputs/reward_sweep_multiseed_summaries")
    args = parser.parse_args()

    root = Path.cwd()
    output_dir = root / args.output_dir
    sweep_kinds = ["mcs_cost", "queue_cost"] if args.sweep_kind == "all" else [args.sweep_kind]
    for sweep_kind in sweep_kinds:
        rows = collect_rows(root, sweep_kind=sweep_kind)
        output_path = output_dir / f"{sweep_kind}_multiseed_sweep_summary.csv"
        write_csv(output_path, rows)
        print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
