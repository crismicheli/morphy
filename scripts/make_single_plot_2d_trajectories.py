"""
Plot a static ensemble of 2D trajectories for one scenario in the (E, T) phase plane.

This script is the static counterpart of `make_single_animate_2d_trajectories.py`.
It runs one scenario, projects all simulated trajectories onto the ECM–tension
plane, and saves a single PNG figure instead of a GIF animation.

The input initial condition is given as a 4D point in (C, T, E, O) order.
If `--x0` is omitted, the default center `DEFAULT_SIM["x0_center"]` is used.

Color convention
----------------
- Blue segments/points: trajectory states that remain inside the projected
  viability region in (E, T, O).
- Red segments/points: trajectory states that lie outside the projected
  viability region.

The viability rectangle shown in the background corresponds to the current
bounds on E and T, with O viability incorporated into the point/segment color.

Usage examples
--------------
Default center for a stable regime:
    python scripts/make_single_plot_2d_trajectories.py \
        --filter "Intermediate porosity" \
        --n-traj 28

Custom center for an unstable regime:
    python scripts/make_single_plot_2d_trajectories.py \
        --filter "Hypoxic environment" \
        --n-traj 28 \
        --x0 0.50 1.10 0.95 0.30

Hide the viability rectangle:
    python scripts/make_single_plot_2d_trajectories.py \
        --filter "Enhanced guidance" \
        --hide-box
"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_BOUNDS, DEFAULT_PARAMS, DEFAULT_SIM
from plotting.plot2d_helpers import (
    INSIDE_COLOR,
    OUTSIDE_COLOR,
    add_et_background,
    build_line_segments_2d,
    point_inside_eto_projection,
    segment_colors_for_solution_2d,
)
from plotting.scenario_helpers import choose_scenario, run_single_scenario, scenario_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one simulation scenario in the 2D E-T phase plane and save a static PNG."
    )
    parser.add_argument("--filter", default="Intermediate porosity", help="Substring used to choose a scenario label.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. If omitted, a scenario-based name is created in root/figures.",
    )
    parser.add_argument("--n-traj", type=int, default=DEFAULT_SIM["n_traj"], help="Number of trajectories to simulate.")
    parser.add_argument(
        "--x0",
        type=float,
        nargs=4,
        metavar=("C", "T", "E", "O"),
        default=None,
        help="Explicit initial center as a 4D point in (C, T, E, O) order. If omitted, DEFAULT_SIM['x0_center'] is used.",
    )
    parser.add_argument("--hide-box", action="store_true", help="Hide the ET viability rectangle.")
    return parser.parse_args()


def resolve_output_path(args: argparse.Namespace, label: str) -> Path:
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        return output_path

    slug = scenario_slug(label)
    return ROOT / "figures" / f"{slug}_2d_static.png"


def plot_static_et_ensemble(
    result: dict,
    scenario_cfg: dict,
    output_path: Path,
    *,
    bounds: dict,
    par: dict,
    show_box: bool = True,
    emax_axis: float = 2.0,
    tmax_axis: float = 1.6,
) -> None:
    solutions = result["solutions"]
    reports = result["reports"]
    p = result["p"]
    label = result["label"]

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)

    add_et_background(
        ax,
        bounds=bounds,
        par=par,
        scenario_cfg=scenario_cfg,
        p=p,
        emax_axis=emax_axis,
        tmax_axis=tmax_axis,
        show_box=show_box,
    )

    for sol in solutions:
        segments = build_line_segments_2d(sol, sol.y.shape[1] - 1)
        seg_colors = segment_colors_for_solution_2d(sol, bounds)

        if len(segments) > 0:
            lc = LineCollection(segments, colors=seg_colors[: len(segments)], linewidths=1.5, alpha=0.75)
            ax.add_collection(lc)

        E0 = float(sol.y[2, 0])
        T0 = float(sol.y[1, 0])
        O0 = float(sol.y[3, 0])

        E1 = float(sol.y[2, -1])
        T1 = float(sol.y[1, -1])
        O1 = float(sol.y[3, -1])

        start_inside = point_inside_eto_projection(E0, T0, O0, bounds)
        end_inside = point_inside_eto_projection(E1, T1, O1, bounds)

        ax.plot(E0, T0, "o", ms=4, color=INSIDE_COLOR if start_inside else OUTSIDE_COLOR, alpha=0.9, zorder=5)
        ax.plot(E1, T1, "*", ms=8, color=INSIDE_COLOR if end_inside else OUTSIDE_COLOR, alpha=0.95, zorder=6)

    viable_count = sum(r.viable for r in reports)

    ax.set_title(
        f"{label}\nStatic ensemble in the (E, T) phase plane",
        fontsize=12,
        fontweight="bold",
    )

    ax.text(
        0.02,
        0.98,
        f"Viable trajectories: {viable_count}/{len(reports)} | p={p:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scenario = choose_scenario(args.filter)
    result = run_single_scenario(
        scenario,
        n_traj=args.n_traj,
        x0_center=args.x0,
    )

    output_path = resolve_output_path(args, result["label"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_static_et_ensemble(
        result,
        scenario,
        output_path,
        bounds=DEFAULT_BOUNDS,
        par=DEFAULT_PARAMS,
        show_box=not args.hide_box,
    )

    print(f"Scenario: {result['label']}")
    print(f"Saved PNG: {output_path}")


if __name__ == "__main__":
    main()
