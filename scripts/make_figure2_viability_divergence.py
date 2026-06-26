"""
Figure 2: compare two scenarios from the same starting cloud.

Panel A:
    Stable scenario with a family of trajectories started near the viability box
    and remaining viable (blue).

Panel B:
    Same starting cloud, but an unstable scenario in which some trajectories
    leave the viability box. The long-term attractor/endpoint cloud is still
    centered inside the box, but some individual paths become non-viable.

Input state order is (C, T, E, O).

Usage:
    python scripts/make_figure2_viability_divergence.py --show-box
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

from config import DEFAULT_BOUNDS
from plotting.plot_helpers import add_viability_box, build_line_segments_3d, get_axes_limits, point_inside_eto_box
from plotting.scenario_helpers import choose_scenario, run_single_scenario


STABLE_FILTER = "Enhanced guidance"
UNSTABLE_FILTER = "Low porosity"

N_TRAJ = 30

# Same starting center in both scenarios, close to the viable box and within bounds.
# Order: (C, T, E, O)
X0_CENTER = np.array([0.50, 1.10, 1.15, 0.34], dtype=float)

BLUE_TRAJ = "#2166ac"
LIGHT_BLUE = "#67a9cf"
OUTSIDE_RED = "#d73027"
ATTRACTOR_BLACK = "#111111"


def endpoint_viable(sol) -> bool:
    C, T, E, O = (float(v) for v in sol.y[:, -1])
    return (
        C >= DEFAULT_BOUNDS["C_min"]
        and point_inside_eto_box(T, E, O, DEFAULT_BOUNDS)
    )


def ever_outside(sol) -> bool:
    C = sol.y[0]
    T = sol.y[1]
    E = sol.y[2]
    O = sol.y[3]
    for i in range(sol.y.shape[1]):
        if C[i] < DEFAULT_BOUNDS["C_min"]:
            return True
        if not point_inside_eto_box(float(T[i]), float(E[i]), float(O[i]), DEFAULT_BOUNDS):
            return True
    return False


def style_axis(ax, panel_title: str, show_box: bool, elev: float = 24.0, azim: float = -58.0) -> None:
    tmax_axis, emax_axis, omax_axis = get_axes_limits(DEFAULT_BOUNDS)
    ax.set_xlim(0.0, tmax_axis)
    ax.set_ylim(0.0, emax_axis)
    ax.set_zlim(0.0, omax_axis)
    ax.set_xlabel("T")
    ax.set_ylabel("E")
    ax.set_zlabel("O")
    ax.set_title(panel_title, fontsize=12, pad=10)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((tmax_axis, emax_axis, omax_axis))
    if show_box:
        add_viability_box(ax, DEFAULT_BOUNDS, omax_axis)


def plot_solution(ax, sol, *, stable_panel: bool) -> tuple[np.ndarray, bool, bool]:
    segments = build_line_segments_3d(sol, sol.y.shape[1] - 1)
    start = sol.y[:, 0]
    end = sol.y[:, -1]

    outside_any = ever_outside(sol)
    end_is_viable = endpoint_viable(sol)

    if stable_panel:
        seg_color = BLUE_TRAJ
        start_color = BLUE_TRAJ
    else:
        seg_color = OUTSIDE_RED if outside_any else LIGHT_BLUE
        start_color = LIGHT_BLUE

    if len(segments) > 0:
        lc = Line3DCollection(segments, colors=seg_color, linewidths=1.15, alpha=0.36)
        ax.add_collection3d(lc)

    ax.scatter(float(start[1]), float(start[2]), float(start[3]), color=start_color, s=16, alpha=0.55, marker="o")

    end_color = ATTRACTOR_BLACK if end_is_viable else OUTSIDE_RED
    end_marker = "*" if end_is_viable else "X"
    ax.scatter(
        float(end[1]), float(end[2]), float(end[3]),
        color=end_color, s=48, alpha=0.90, marker=end_marker,
        edgecolors="white", linewidths=0.5
    )

    return np.array([float(end[1]), float(end[2]), float(end[3])]), outside_any, end_is_viable


def annotate_mean_endpoint(ax, endpoints: np.ndarray, label: str) -> None:
    mean_pt = endpoints.mean(axis=0)
    ax.scatter(mean_pt[0], mean_pt[1], mean_pt[2], color=ATTRACTOR_BLACK, s=120, marker="D", edgecolors="white", linewidths=0.8)
    ax.text(mean_pt[0], mean_pt[1], mean_pt[2], f" {label}\n({mean_pt[0]:.2f}, {mean_pt[1]:.2f}, {mean_pt[2]:.2f})", fontsize=8, color=ATTRACTOR_BLACK)


def run_panel(scenario_filter: str):
    scenario = choose_scenario(scenario_filter)
    result = run_single_scenario(
        scenario,
        n_traj=N_TRAJ,
        x0_center=X0_CENTER,
    )
    return scenario, result


def build_legends(ax):
    handles = [
        Line2D([0], [0], color=BLUE_TRAJ, lw=2, label="Viable trajectory family"),
        Line2D([0], [0], color=OUTSIDE_RED, lw=2, label="Trajectory leaving viability box"),
        Line2D([0], [0], marker="*", linestyle="", color=ATTRACTOR_BLACK, markersize=10, label="Viable endpoint"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=8)


def main(show_box: bool = True, output: str | None = None) -> None:
    stable_scenario, stable_result = run_panel(STABLE_FILTER)
    unstable_scenario, unstable_result = run_panel(UNSTABLE_FILTER)

    fig = plt.figure(figsize=(12.5, 6.2), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    stable_endpoints = []
    for sol in stable_result["solutions"]:
        end_pt, _, _ = plot_solution(ax1, sol, stable_panel=True)
        stable_endpoints.append(end_pt)
    stable_endpoints = np.vstack(stable_endpoints)

    unstable_endpoints = []
    for sol in unstable_result["solutions"]:
        end_pt, _, _ = plot_solution(ax2, sol, stable_panel=False)
        unstable_endpoints.append(end_pt)
    unstable_endpoints = np.vstack(unstable_endpoints)

    style_axis(ax1, "a) Stable constraints", show_box=show_box)
    style_axis(ax2, "b) Unstable constraints from same start cloud", show_box=show_box)

    annotate_mean_endpoint(ax1, stable_endpoints, "stable mean end point")
    annotate_mean_endpoint(ax2, unstable_endpoints, "unstable mean end point")

    build_legends(ax1)

    out = Path(output) if output else (REPO_ROOT / "figures" / "figure2_viability_divergence.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"Stable scenario: {stable_scenario['label']}")
    print(f"Unstable scenario: {unstable_scenario['label']}")
    print(f"Shared x0 center (C,T,E,O): {X0_CENTER.tolist()}")
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
