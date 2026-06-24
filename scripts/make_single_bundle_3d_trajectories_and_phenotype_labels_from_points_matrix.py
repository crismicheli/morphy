"""
Generate a single 3D bundle with trajectories from multiple initial points and a matching phenotype-label plot.

The left panel overlays trajectories generated from the initial points given as the rows
of an input matrix. The right panel shows the corresponding 3D phenotype labels using the
selected classifier type. The input matrix must be written as
[[c1,t1,e1,o1],[c2,t2,e2,o2],...] and is parsed internally into a NumPy n x 4 array
in (C, T, E, O) order. The script does not assume any special geometric relation between
the input points.

Usage examples:
    python scripts/make_single_bundle_3d_trajectories_and_phenotype_labels_from_points_matrix.py \
        --filter "Intermediate porosity" \
        --x0-matrix "[[0.50,0.90,0.70,0.55],[-0.50,-0.90,-0.70,-0.55]]" \
        --classifier-type temporal --show-box

    python scripts/make_single_bundle_3d_trajectories_and_phenotype_labels_from_points_matrix.py \
        --filter "Enhanced guidance" \
        --x0-matrix "[[0.60,1.00,0.80,0.65],[0.45,0.85,0.68,0.58],[0.72,1.10,0.90,0.62]]" \
        --classifier-type state_machine --show-box
"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

SCRIPTDIR = Path(__file__).resolve().parent
REPOROOT = SCRIPTDIR.parent
PACKAGEPARENT = REPOROOT.parent
for p in (str(PACKAGEPARENT), str(REPOROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import DEFAULT_BOUNDS, DEFAULT_PARAMS, DEFAULT_SIM
from plotting.plot_helpers import (
    add_viability_box,
    build_line_segments_3d,
    classify_all_points,
    get_axes_limits,
)
from plotting.scenario_helpers import choose_scenario, run_single_scenario, scenario_slug, validate_x0_center
from classifiers.classifier_dispatch import get_classifier_components


STATE_ORDER = [
    "Apoptosis",
    "Migration",
    "Proliferation",
    "Quiescence",
    "Diversification",
    "Undetermined",
]

POINT_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a single 3D bundle with trajectories from multiple initial points on the left "
            "and phenotype labels from a selected classifier on the right."
        )
    )
    parser.add_argument("--filter", default="Intermediate porosity", help="Substring used to choose a scenario label.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path. If omitted, a scenario-based name is created in root/figures.",
    )
    parser.add_argument("--prefix", default=None, help="Optional filename prefix; defaults to scenario label slug.")
    parser.add_argument(
        "--classifier-type",
        choices=["static", "temporal", "state_machine"],
        default="temporal",
        help="Classifier backend used for the right-hand phenotype-label panel.",
    )
    parser.add_argument("--n-traj", type=int, default=DEFAULT_SIM["n_traj"], help="Number of trajectories per initial point.")
    parser.add_argument(
        "--x0-matrix",
        type=str,
        required=True,
        help=(
            "Initial points as a matrix written as [[c1,t1,e1,o1],[c2,t2,e2,o2],...] "
            "in (C, T, E, O) order. Parsed internally as a NumPy n x 4 array."
        ),
    )
    parser.add_argument("--elev", type=float, default=24.0, help="3D camera elevation.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth.")
    parser.add_argument("--show-box", action="store_true", help="Show translucent ETO viability box.")
    parser.add_argument("--stride", type=int, default=8, help="Subsample factor for phenotype-label timepoints.")
    return parser.parse_args()


def parse_x0_matrix(text: str) -> np.ndarray:
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            "Could not parse --x0-matrix. Expected format: "
            "[[c1,t1,e1,o1],[c2,t2,e2,o2],...]"
        ) from exc

    arr = np.asarray(value, dtype=float)

    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"--x0-matrix must have shape (n, 4) in (C, T, E, O) order; got shape {arr.shape}"
        )

    if arr.shape[0] < 1:
        raise ValueError("--x0-matrix must contain at least one row.")

    return np.asarray([validate_x0_center(row) for row in arr], dtype=float)


def confirm_overwrite(path: Path) -> bool:
    if not path.exists():
        return True
    reply = input(f"File {path} already exists. Overwrite? [y/N] ").strip().lower()
    return reply in {"y", "yes"}


def resolve_output_path(args: argparse.Namespace, label: str) -> Path:
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = REPOROOT / output_path
        return output_path

    slug = args.prefix or scenario_slug(label)
    return REPOROOT / "figures" / f"{slug}_bundle_3d_points_matrix_clf-{args.classifier_type}.png"


def style_3d_axis(ax, *, title: str, elev: float, azim: float, show_box: bool) -> None:
    tmax_axis, emax_axis, omax_axis = get_axes_limits(DEFAULT_BOUNDS)
    ax.set_title(title, pad=12)
    ax.set_xlabel("T")
    ax.set_ylabel("E")
    ax.set_zlabel("O")
    ax.set_xlim(0.0, tmax_axis)
    ax.set_ylim(0.0, emax_axis)
    ax.set_zlim(0.0, omax_axis)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((tmax_axis, emax_axis, omax_axis))
    if show_box:
        add_viability_box(ax, DEFAULT_BOUNDS, omax_axis)


def color_for_index(i: int) -> str:
    return POINT_COLORS[i % len(POINT_COLORS)]


def plot_trajectory_group(ax, result: dict, color: str) -> None:
    sols = result.get("solutions", [])
    for sol in sols:
        segments = build_line_segments_3d(sol, sol.y.shape[1] - 1)
        if len(segments) == 0:
            continue
        lc = Line3DCollection(segments, colors=color, linewidths=1.2, alpha=0.35)
        ax.add_collection3d(lc)

        start = sol.y[:, 0]
        end = sol.y[:, -1]
        ax.scatter(float(start[1]), float(start[2]), float(start[3]), color=color, s=22, alpha=0.95, marker="o")
        ax.scatter(float(end[1]), float(end[2]), float(end[3]), color=color, s=16, alpha=0.75, marker="^")

    x0 = np.asarray(result["x0_center"], dtype=float)
    ax.scatter(
        float(x0[1]),
        float(x0[2]),
        float(x0[3]),
        color=color,
        s=90,
        marker="*",
        edgecolors="black",
        linewidths=0.5,
    )


def collect_label_points(result: dict, scenario: dict, classifier_fn, reset_fn, color_map: dict, stride: int):
    points = []
    present_states = set()
    for sol in result.get("solutions", []):
        snapshots = classify_all_points(
            sol,
            classifier_fn,
            color_map,
            bounds=DEFAULT_BOUNDS,
            par=DEFAULT_PARAMS,
            scenario_cfg=scenario,
            stride=max(1, stride),
            reset_fn=reset_fn,
        )
        for snap in snapshots:
            points.append((snap["T"], snap["E"], snap["O"], snap["color"], snap["label"]))
            present_states.add(snap["label"])
    return points, present_states


def build_state_legend_handles(color_map: dict, present_states: set[str]) -> list[Line2D]:
    handles = []
    for state in STATE_ORDER:
        if state in present_states:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=color_map[state],
                    markeredgecolor="none",
                    markersize=8,
                    label=state,
                )
            )
    return handles


def build_point_legend_handles(x0_matrix: np.ndarray) -> list[Line2D]:
    handles = []
    for i, row in enumerate(x0_matrix):
        handles.append(
            Line2D(
                [0],
                [0],
                color=color_for_index(i),
                lw=2,
                label=f"x0[{i}] (C,T,E,O) = {np.array2string(row, precision=2, separator=', ')}",
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=10,
            linestyle="",
            label="Initial center",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="gray",
            markersize=7,
            linestyle="",
            label="Trajectory end",
        )
    )
    return handles


def main() -> None:
    args = parse_args()
    scenario = choose_scenario(args.filter)
    x0_matrix = parse_x0_matrix(args.x0_matrix)

    results = [
        run_single_scenario(
            scenario,
            n_traj=args.n_traj,
            x0_center=row,
        )
        for row in x0_matrix
    ]

    classifier_fn, reset_fn, state_colors = get_classifier_components(args.classifier_type)

    output_path = resolve_output_path(args, scenario["label"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not confirm_overwrite(output_path):
        print("Aborted: not overwriting existing file.")
        return

    fig = plt.figure(figsize=(15, 7))
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")

    for i, result in enumerate(results):
        plot_trajectory_group(ax_left, result, color_for_index(i))

    style_3d_axis(
        ax_left,
        title="Trajectories from input initial points",
        elev=args.elev,
        azim=args.azim,
        show_box=args.show_box,
    )
    ax_left.legend(handles=build_point_legend_handles(x0_matrix), loc="upper left", frameon=False, fontsize=8)

    all_points = []
    present_states = set()
    for result in results:
        pts, states = collect_label_points(
            result,
            scenario,
            classifier_fn,
            reset_fn,
            state_colors,
            args.stride,
        )
        all_points.extend(pts)
        present_states |= states

    if all_points:
        coords = np.array([[p[0], p[1], p[2]] for p in all_points], dtype=float)
        colors = [p[3] for p in all_points]
        ax_right.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=18, alpha=0.85, edgecolors="none")

    style_3d_axis(
        ax_right,
        title=f"Phenotype labels ({args.classifier_type})",
        elev=args.elev,
        azim=args.azim,
        show_box=args.show_box,
    )

    state_handles = build_state_legend_handles(state_colors, present_states)
    if state_handles:
        ax_right.legend(handles=state_handles, loc="upper left", frameon=False, fontsize=8)

    fig.suptitle(
        f"{scenario['label']}\n3D trajectories and phenotype labels from an input points matrix",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Scenario: {scenario['label']}")
    print(f"Classifier: {args.classifier_type}")
    print(f"Initial points matrix shape: {x0_matrix.shape}")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
