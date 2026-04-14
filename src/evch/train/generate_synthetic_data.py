from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

from evch.config.loader import build_config_parser, load_config
from evch.data.city import build_city
from evch.data.synthetic import SyntheticCity
from evch.envs.demand import DemandGenerator
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging
from evch.utils.seeding import set_global_seed

LOGGER = logging.getLogger(__name__)


def _project_point(
    x_value: float,
    y_value: float,
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> tuple[float, float]:
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    x_px = left + (x_value - min_x) / span_x * (right - left)
    y_px = bottom - (y_value - min_y) / span_y * (bottom - top)
    return x_px, y_px


def _render_site_marker(site_type: str, x_px: float, y_px: float, color: str) -> str:
    if site_type == "service_area":
        return f'<rect x="{x_px - 6:.1f}" y="{y_px - 6:.1f}" width="12" height="12" rx="2" fill="{color}" stroke="#0f172a" stroke-width="1.0" />'
    if site_type == "highway_exit":
        return f'<polygon points="{x_px:.1f},{y_px - 7:.1f} {x_px + 7:.1f},{y_px:.1f} {x_px:.1f},{y_px + 7:.1f} {x_px - 7:.1f},{y_px:.1f}" fill="{color}" stroke="#0f172a" stroke-width="1.0" />'
    return f'<polygon points="{x_px:.1f},{y_px - 8:.1f} {x_px + 8:.1f},{y_px + 6:.1f} {x_px - 8:.1f},{y_px + 6:.1f}" fill="{color}" stroke="#0f172a" stroke-width="1.0" />'


def render_synthetic_layout_svg(city: SyntheticCity, path: Path) -> None:
    width = 1320.0
    height = 720.0
    chart_left = 72.0
    chart_right = 1000.0
    chart_top = 84.0
    chart_bottom = 650.0
    legend_left = 1032.0
    padding = 18.0

    min_x = float(min(city.site_coords[:, 0].min(), city.zone_coords[:, 0].min()))
    max_x = float(max(city.site_coords[:, 0].max(), city.zone_coords[:, 0].max()))
    min_y = float(min(city.site_coords[:, 1].min(), city.zone_coords[:, 1].min()))
    max_y = float(max(city.site_coords[:, 1].max(), city.zone_coords[:, 1].max()))
    if city.layout_type == "corridor":
        min_y = min(min_y, -18.0)
        max_y = max(max_y, 18.0)
        min_x = min(min_x, 0.0)
        max_x = max(max_x, float(city.metadata.get("corridor_length_km", max_x)))

    zone_colors = {
        "city": "#d95f02",
        "suburb": "#7570b3",
        "corridor": "#1b9e77",
        "logistics": "#e7298a",
        "grounded_zone": "#64748b",
        "demand_zone": "#1f77b4",
    }
    site_colors = {
        "city_hub": "#0b7285",
        "service_area": "#495057",
        "highway_exit": "#f08c00",
        "major_hub": "#0b7285",
        "standard_station": "#5c7cfa",
        "candidate_site": "#0b7285",
        "tomtom_station": "#5c7cfa",
    }

    zone_min = float(city.zone_base_demand.min())
    zone_max = float(city.zone_base_demand.max())

    def project(x_value: float, y_value: float) -> tuple[float, float]:
        return _project_point(
            x_value,
            y_value,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            left=chart_left,
            right=chart_right,
            top=chart_top,
            bottom=chart_bottom,
        )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">',
        f'<rect x="0" y="0" width="{int(width)}" height="{int(height)}" fill="#f8fafc" />',
        '<rect x="32" y="26" width="1256" height="666" rx="24" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" />',
        f'<text x="{chart_left:.0f}" y="52" font-size="28" font-family="Helvetica, Arial, sans-serif" fill="#0f172a">{"Synthetic Corridor Layout" if city.layout_type == "corridor" else "Synthetic Layout"}</text>',
        f'<text x="{chart_left:.0f}" y="74" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#475569">Demand zones are colored by type and sized by base demand. Charging candidates are labeled and styled by role.</text>',
        f'<rect x="{chart_left:.1f}" y="{chart_top:.1f}" width="{chart_right - chart_left:.1f}" height="{chart_bottom - chart_top:.1f}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.0" />',
    ]

    for tick_idx in range(6):
        x_value = min_x + (max_x - min_x) * tick_idx / 5.0
        x_px, _ = project(x_value, min_y)
        lines.append(f'<line x1="{x_px:.1f}" y1="{chart_top:.1f}" x2="{x_px:.1f}" y2="{chart_bottom:.1f}" stroke="#e2e8f0" stroke-width="1" />')
        lines.append(f'<text x="{x_px:.1f}" y="{chart_bottom + 22:.1f}" text-anchor="middle" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="#64748b">{x_value:.0f} km</text>')
    for tick_idx in range(5):
        y_value = min_y + (max_y - min_y) * tick_idx / 4.0
        _, y_px = project(min_x, y_value)
        lines.append(f'<line x1="{chart_left:.1f}" y1="{y_px:.1f}" x2="{chart_right:.1f}" y2="{y_px:.1f}" stroke="#eef2f7" stroke-width="1" />')
        lines.append(f'<text x="{chart_left - 12:.1f}" y="{y_px + 4:.1f}" text-anchor="end" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="#64748b">{y_value:.0f}</text>')

    if city.corridor_polyline is not None:
        projected_polyline = [project(float(point[0]), float(point[1])) for point in city.corridor_polyline]
        polyline_points = " ".join(f"{x_px:.1f},{y_px:.1f}" for x_px, y_px in projected_polyline)
        lines.append(f'<polyline points="{polyline_points}" fill="none" stroke="#334155" stroke-width="4.0" stroke-linecap="round" />')

    if city.layout_type == "corridor" and city.metadata.get("city_positions_km"):
        for idx, city_position in enumerate(city.metadata["city_positions_km"]):
            x_px, _ = project(float(city_position), 0.0)
            lines.append(f'<line x1="{x_px:.1f}" y1="{chart_top:.1f}" x2="{x_px:.1f}" y2="{chart_bottom:.1f}" stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="6 6" />')
            lines.append(f'<text x="{x_px + 6:.1f}" y="{chart_top + 18 + idx * 14:.1f}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="#475569">City node {idx + 1}</text>')

    for zone_idx, (coord, zone_type, base_demand) in enumerate(zip(city.zone_coords, city.zone_types or ("demand_zone",) * len(city.zone_coords), city.zone_base_demand, strict=True)):
        x_px, y_px = project(float(coord[0]), float(coord[1]))
        if zone_max > zone_min:
            radius = 5.0 + 9.0 * (float(base_demand) - zone_min) / (zone_max - zone_min)
        else:
            radius = 8.0
        zone_color = zone_colors.get(zone_type, "#1f77b4")
        lines.append(f'<circle cx="{x_px:.1f}" cy="{y_px:.1f}" r="{radius:.1f}" fill="{zone_color}" fill-opacity="0.50" stroke="#ffffff" stroke-width="1.0" />')
        if city.layout_type == "corridor":
            lines.append(f'<text x="{x_px + radius + 4:.1f}" y="{y_px + 3:.1f}" font-size="10" font-family="Helvetica, Arial, sans-serif" fill="#475569">{zone_idx + 1}</text>')

    for site_idx, (coord, site_type) in enumerate(zip(city.site_coords, city.site_types or ("candidate_site",) * len(city.site_coords), strict=True)):
        x_px, y_px = project(float(coord[0]), float(coord[1]))
        site_color = site_colors.get(site_type, "#0b7285")
        lines.append(_render_site_marker(site_type, x_px, y_px, site_color))
        label = city.site_labels[site_idx] if city.site_labels else f"Site {site_idx + 1}"
        label_y = y_px - 10.0 if site_idx % 2 == 0 else y_px + 18.0
        lines.append(f'<text x="{x_px + 10:.1f}" y="{label_y:.1f}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="#0f172a">{escape(label)}</text>')

    lines.extend(
        [
            f'<text x="{chart_left:.1f}" y="{chart_bottom + 48:.1f}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#475569">X-axis: corridor distance or synthetic x-coordinate. Y-axis: lateral offset from the main route.</text>',
            f'<rect x="{legend_left:.1f}" y="92" width="226" height="560" rx="16" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.0" />',
            f'<text x="{legend_left + padding:.1f}" y="120" font-size="18" font-family="Helvetica, Arial, sans-serif" fill="#0f172a">Interpretation</text>',
            f'<text x="{legend_left + padding:.1f}" y="148" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#475569">Demand bubbles</text>',
        ]
    )

    legend_y = 170.0
    for zone_type in sorted(set(city.zone_types or ("demand_zone",))):
        color = zone_colors.get(zone_type, "#1f77b4")
        lines.append(f'<circle cx="{legend_left + padding + 8:.1f}" cy="{legend_y:.1f}" r="7" fill="{color}" fill-opacity="0.6" stroke="#ffffff" stroke-width="1" />')
        lines.append(f'<text x="{legend_left + padding + 24:.1f}" y="{legend_y + 4:.1f}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#334155">{zone_type.replace("_", " ").title()}</text>')
        legend_y += 24.0

    legend_y += 12.0
    lines.append(f'<text x="{legend_left + padding:.1f}" y="{legend_y:.1f}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#475569">Charging sites</text>')
    legend_y += 22.0
    for site_type in sorted(set(city.site_types or ("candidate_site",))):
        color = site_colors.get(site_type, "#0b7285")
        lines.append(_render_site_marker(site_type, legend_left + padding + 8.0, legend_y - 4.0, color))
        lines.append(f'<text x="{legend_left + padding + 24:.1f}" y="{legend_y:.1f}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#334155">{site_type.replace("_", " ").title()}</text>')
        legend_y += 24.0

    legend_y += 12.0
    lines.append(f'<text x="{legend_left + padding:.1f}" y="{legend_y:.1f}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#475569">Stats</text>')
    legend_y += 22.0
    stats = [
        f"Sites: {len(city.site_coords)}",
        f"Demand zones: {len(city.zone_coords)}",
        f"Base demand range: {zone_min:.1f} to {zone_max:.1f}",
    ]
    if city.layout_type == "corridor":
        stats.append(f"Corridor length: {float(city.metadata.get('corridor_length_km', 0.0)):.0f} km")
        stats.append(f"City nodes: {len(city.metadata.get('city_positions_km', []))}")
    for item in stats:
        lines.append(f'<text x="{legend_left + padding:.1f}" y="{legend_y:.1f}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#334155">{escape(item)}</text>')
        legend_y += 20.0

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


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
    city = build_city(env_cfg, demand_cfg, seed=seed)
    generator = DemandGenerator(city, demand_cfg, horizon=int(env_cfg["horizon"]), seed=seed)
    frame = generator.generate_supervised_frame(
        num_days=int(data_cfg["num_days"]),
        include_prev_observation=bool(data_cfg.get("include_prev_observation", True)),
    )
    csv_path = output_dir / "synthetic_demand.csv"
    frame.to_csv(csv_path, index=False)
    LOGGER.info("Saved synthetic dataset to %s", csv_path)

    site_table = pd.DataFrame(
        {
            "site_id": list(range(len(city.site_coords))),
            "label": list(city.site_labels) if city.site_labels else [f"site_{idx}" for idx in range(len(city.site_coords))],
            "type": list(city.site_types) if city.site_types else ["candidate_site"] * len(city.site_coords),
            "x_km": city.site_coords[:, 0],
            "y_km": city.site_coords[:, 1],
            "connector_proxy": (
                city.metadata.get("site_connector_proxy")
                if len(city.metadata.get("site_connector_proxy", [])) == len(city.site_coords)
                else [None] * len(city.site_coords)
            ),
            "power_proxy_kw": (
                city.metadata.get("site_power_proxy_kw")
                if len(city.metadata.get("site_power_proxy_kw", [])) == len(city.site_coords)
                else [None] * len(city.site_coords)
            ),
            "availability_proxy": (
                city.metadata.get("site_availability_proxy")
                if len(city.metadata.get("site_availability_proxy", [])) == len(city.site_coords)
                else [None] * len(city.site_coords)
            ),
        }
    )
    zone_table = pd.DataFrame(
        {
            "zone_id": list(range(len(city.zone_coords))),
            "label": list(city.zone_labels) if city.zone_labels else [f"zone_{idx}" for idx in range(len(city.zone_coords))],
            "type": list(city.zone_types) if city.zone_types else ["demand_zone"] * len(city.zone_coords),
            "x_km": city.zone_coords[:, 0],
            "y_km": city.zone_coords[:, 1],
            "base_demand": city.zone_base_demand,
            "zone_scale": city.zone_scale,
        }
    )
    site_table.to_csv(output_dir / "synthetic_sites.csv", index=False)
    zone_table.to_csv(output_dir / "synthetic_zones.csv", index=False)

    svg_path = output_dir / "synthetic_layout.svg"
    render_synthetic_layout_svg(city, svg_path)

    fig_path = output_dir / "synthetic_city.png"
    plotting_available = False
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt

        plotting_available = True
        plt.figure(figsize=(12, 4.5) if city.layout_type == "corridor" else (6, 6))
        if city.corridor_polyline is not None:
            plt.plot(city.corridor_polyline[:, 0], city.corridor_polyline[:, 1], color="#5b6472", linewidth=3.0, label="Motorway")
        if city.layout_type == "corridor" and city.metadata.get("city_positions_km"):
            for idx, position in enumerate(city.metadata["city_positions_km"]):
                label = "Metro corridor node" if idx == 0 else None
                plt.axvline(float(position), color="#d7dee8", linewidth=0.9, linestyle="--", label=label)

        zone_colors = {
            "city": "#d95f02",
            "suburb": "#7570b3",
            "corridor": "#1b9e77",
            "logistics": "#e7298a",
            "demand_zone": "#1f77b4",
        }
        for zone_type in sorted(set(city.zone_types or ("demand_zone",))):
            mask = [current == zone_type for current in (city.zone_types or ("demand_zone",) * len(city.zone_coords))]
            plt.scatter(
                city.zone_coords[mask, 0],
                city.zone_coords[mask, 1],
                s=90,
                alpha=0.8,
                color=zone_colors.get(zone_type, "#1f77b4"),
                label=f"Demand: {zone_type.replace('_', ' ')}",
            )

        site_styles = {
            "city_hub": ("^", "#0b7285"),
            "service_area": ("s", "#495057"),
            "highway_exit": ("D", "#f08c00"),
            "candidate_site": ("^", "#0b7285"),
        }
        for site_type in sorted(set(city.site_types or ("candidate_site",))):
            marker, color = site_styles.get(site_type, ("^", "#0b7285"))
            mask = [current == site_type for current in (city.site_types or ("candidate_site",) * len(city.site_coords))]
            plt.scatter(
                city.site_coords[mask, 0],
                city.site_coords[mask, 1],
                s=80,
                marker=marker,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                label=f"Site: {site_type.replace('_', ' ')}",
            )

        plt.xlabel("Corridor distance (km)" if city.layout_type == "corridor" else "X coordinate (km)")
        plt.ylabel("Lateral offset (km)" if city.layout_type == "corridor" else "Y coordinate (km)")
        plt.title("Synthetic Corridor Layout" if city.layout_type == "corridor" else "Synthetic City Layout")
        plt.legend(ncol=2 if city.layout_type == "corridor" else 1, fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()
    except Exception as exc:  # pragma: no cover - depends on local plotting stack
        LOGGER.warning("Skipping layout plot because matplotlib is unavailable: %s", exc)
        if fig_path.exists():
            fig_path.unlink()

    write_json(
        output_dir / "dataset_metadata.json",
        {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "seed": seed,
            "layout_type": city.layout_type,
            "site_types": list(city.site_types),
            "zone_types": list(city.zone_types),
            "metadata": city.metadata,
            "plot_created": plotting_available,
            "svg_plot_path": str(svg_path),
        },
    )


if __name__ == "__main__":
    main()
