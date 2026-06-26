"""
Figure 1: compare attractor/end-point positions across three scenarios
(stable / unstable / borderline) using the same sparse set of starting points.

Design goals
------------
- Same 5 starting points in all three scenarios.
- Sparse points are intentionally used to emphasize reachability, not basin density.
- Each trajectory is run for enough timepoints to visually approach its asymptotic endpoint.
- Final points are highlighted and annotated with coordinates.
- Endpoints outside the viability box are still shown as reachable but non-viable.

Input row order for x0_matrix is (C, T, E, O).

Usage:
    python scripts/make_figure1_attractor_positions.py --show-box

Optional:
    python scripts/make_figure1_attractor_positions.py \
        --output figures/figure1_attractors.png \
        --classifier-type temporal \
        --show-box
"""
#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_PARENT = REPO_ROOT.parent

for p in (str(PACKAGE_PARENT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import DEFAULT_BOUNDS, DEFAULT_SIM
from plotting.plot_helpers import add_viability_box, build_line_segments_3d, get_axes_limits, point_inside_eto_box
from plotting.scenario_helpers import choose_scenario, run_single_scenario


SCENARIO_FILTERS = [
    ("Stable", "Stiff scaffold"),
    ("Unstable", "Hypoxic environment"),
    ("Borderline", "Fast ECM remodelling"),
]

# Same starting points across all scenarios; order is (C, T, E, O)
# Chosen to be sparse, well-separated, and inside the current admissible bounds.
X0_MATRIX = np.array(
    [
        [0.50, 0.18, 0.10, 0.92],
        [0.50, 0.42, 0.42, 0.72],
        [0.50, 0.72, 0.82, 0.56],
        [0.50, 1.02, 1.18, 0.40],
        [0.50, 1.30, 1.55, 0.26],
    ],
    dtype=float,
)

POINT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
INSIDE_END_COLOR = "#111111"
OUTSIDE_END_COLOR = "#d62728"


def style_axis(ax, title: str, show_box: bool, elev: float = 24.0, azim: float = -58.0) -> None:
    tmax_axis, emax_axis, omax_axis = get_axes_limits(DEFAULT_BOUNDS)
    ax.set_xlim(0.0, tmax_axis)
    ax.set_ylim(0.0, emax_axis)
    ax.set_zlim(0.0, omax_axis)
    ax.set_xlabel("T")
    ax.set_ylabel("E")
    ax.set_zlabel("O")
    ax.set_title(title, fontsize=12, pad=10)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((tmax_axis, emax_axis, omax_axis))
    if show_box:
        add_viability_box(ax, DEFAULT_BOUNDS, omax_axis)


def endpoint_is_viable(sol) -> bool:
    C, T, E, O = (float(v) for v in sol.y[:, -1])
    return (
        C >= DEFAULT_BOUNDS["C_min"]
        and point_inside_eto_box(T, E, O, DEFAULT_BOUNDS)
    )


def plot_single_solution(ax, sol, color: str, idx: int, annotate: bool = True) -> None:
    segments = build_line_segments_3d(sol, sol.y.shape[1] - 1)
    if len(segments) > 0:
        lc = Line3DCollection(segments, colors=color, linewidths=1.5, alpha=0.45)
        ax.add_collection3d(lc)

    start = sol.y[:, 0]
    end = sol.y[:, -1]

    ax.scatter(float(start[1]), float(start[2]), float(start[3]), color=color, s=36, marker="o", alpha=0.95)

    viable = endpoint_is_viable(sol)
    end_color = INSIDE_END_COLOR if viable else OUTSIDE_END_COLOR
    end_marker = "*" if viable else "X"
    ax.scatter(
        float(end[1]), float(end[2]), float(end[3]),
        color=end_color, s=100, marker=end_marker,
        edgecolors="white", linewidths=0.8, alpha=0.98
    )

    if annotate:
        txt = f"x{idx}→({end[1]:.2f}, {end[2]:.2f}, {end[3]:.2f})"
        ax.text(
            float(end[1]), float(end[2]), float(end[3]),
            " " + txt,
            fontsize=8,
            color=end_color,
        )


def run_points_for_scenario(scenario: dict):
    results = []
    for row in X0_MATRIX:
        result = run_single_scenario(
            scenario,
            n_traj=1,
            x0_center=row,
        )
        results.append(result)
    return results


def build_legends(ax):
    start_handles = [
        Line2D([0], [0], color=POINT_COLORS[i], lw=2,
               label=f"x0[{i}] (C,T,E,O) = {np.array2string(X0_MATRIX[i], precision=2, separator=', ')}")
        for i in range(len(X0_MATRIX))
    ]
    state_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="black", label="Start point", markersize=7),
        Line2D([0], [0], marker="*", linestyle="", color=INSIDE_END_COLOR, label="Reachable viable end point", markersize=10),
        Line2D([0], [0], marker="X", linestyle="", color=OUTSIDE_END_COLOR, label="Reachable non-viable end point", markersize=9),
    ]
    leg1 = ax.legend(handles=start_handles, loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=state_handles, loc="lower left", bbox_to_anchor=(0.0, -0.05), frameon=False, fontsize=8)


def main(output: str | None = None, show_box: bool = True) -> None:
    fig = plt.figure(figsize=(18, 6.4), constrained_layout=True)
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]

    for ax, (title, scenario_filter) in zip(axes, SCENARIO_FILTERS):
        scenario = choose_scenario(scenario_filter)
        results = run_points_for_scenario(scenario)

        for i, result in enumerate(results):
            sol = result["solutions"][0]
            plot_single_solution(ax, sol, POINT_COLORS[i], i, annotate=True)

        style_axis(ax, f"{title}: {scenario['label']}", show_box=show_box)

    build_legends(axes[0])

    out = Path(output) if output else (REPO_ROOT / "figures" / "figure1_attractor_positions.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
