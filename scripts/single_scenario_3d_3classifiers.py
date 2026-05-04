#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

SCRIPTDIR = Path(__file__).resolve().parent
REPOROOT = SCRIPTDIR.parent
PACKAGEPARENT = REPOROOT.parent
for p in (str(PACKAGEPARENT), str(REPOROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import DEFAULT_BOUNDS, DEFAULT_PARAMS, DEFAULT_SIM
from plotting.scenario_helpers import choose_scenario, run_single_scenario, scenario_slug
from plotting.plot_helpers import draw_eto_box
from classifiers import taxonomy_classifier, temporal_taxonomy_classifier, state_machine_classifier


PANEL_SPECS = [
    (
        "Static classifier",
        taxonomy_classifier.classify_state,
        lambda: None,
        taxonomy_classifier.STATE_COLORS,
    ),
    (
        "Temporal classifier",
        temporal_taxonomy_classifier.classify_state,
        temporal_taxonomy_classifier.reset_classifier_memory,
        temporal_taxonomy_classifier.STATE_COLORS,
    ),
    (
        "State-machine classifier",
        state_machine_classifier.classify_state,
        state_machine_classifier.reset_classifier_memory,
        state_machine_classifier.STATE_COLORS,
    ),
]

STATE_ORDER = [
    "Apoptosis",
    "Migration",
    "Proliferation",
    "Quiescence",
    "Diversification",
    "Undetermined",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a taxonomy-labeled 3D plot for one scenario comparing static, temporal, and state-machine classifiers."
    )
    parser.add_argument("--filter", default="Intermediate porosity", help="Substring used to choose a scenario label.")
    parser.add_argument(
        "--out-dir",
        default=str(REPOROOT / "figures"),
        help="Output directory; defaults to <repo-root>/figures.",
    )
    parser.add_argument("--prefix", default=None, help="Optional filename prefix; defaults to scenario label slug.")
    parser.add_argument("--n-traj", type=int, default=DEFAULT_SIM["n_traj"], help="Number of trajectories.")
    parser.add_argument("--shift-T", type=float, default=1.0, help="Multiplier applied to initial T center.")
    parser.add_argument("--shift-E", type=float, default=1.0, help="Multiplier applied to initial E center.")
    parser.add_argument("--shift-O", type=float, default=1.0, help="Multiplier applied to initial O center.")
    parser.add_argument("--elev", type=float, default=24.0, help="3D camera elevation.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth.")
    parser.add_argument("--show-box", action="store_true", help="Show translucent ETO viability box.")
    parser.add_argument("--stride", type=int, default=8, help="Subsample factor for taxonomy plot timepoints.")
    return parser.parse_args()


def confirm_overwrite(path: Path) -> bool:
    if not path.exists():
        return True
    reply = input(f"File {path} already exists. Overwrite? [y/N] ").strip().lower()
    return reply in {"y", "yes"}


def collect_points(result: dict, stride: int):
    sols = result.get("solutions", [])
    all_points = []
    for sol in sols:
        t = np.asarray(sol.t)
        y = np.asarray(sol.y)
        if y.ndim != 2 or y.shape[1] < 2:
            continue
        dt = max(1e-12, float(np.mean(np.diff(t))))
        dydt = np.gradient(y, dt, axis=1)
        idx = np.arange(0, y.shape[1], max(1, stride))
        for i in idx:
            C, T, E, O = [float(v) for v in y[:, i]]
            dC, dT, dE, dO = [float(v) for v in dydt[:, i]]
            all_points.append((C, T, E, O, dC, dT, dE, dO))
    return all_points


def classify_points(points, classifier_fn, reset_fn, scenario_cfg):
    reset_fn()
    labels = []
    coords = []
    for C, T, E, O, dC, dT, dE, dO in points:
        label = classifier_fn(
            C,
            T,
            E,
            O,
            dC,
            dT,
            dE,
            dO,
            bounds=DEFAULT_BOUNDS,
            par=DEFAULT_PARAMS,
            scenario_cfg=scenario_cfg,
        )
        labels.append(label)
        coords.append((T, E, O))
    return labels, np.asarray(coords)


def style_axis(ax, elev: float, azim: float, show_box: bool):
    ax.set_xlabel("T")
    ax.set_ylabel("E")
    ax.set_zlabel("O")
    ax.set_xlim(float(DEFAULT_BOUNDS["T_min"]) - 0.05, float(DEFAULT_BOUNDS["T_max"]) + 0.05)
    ax.set_ylim(float(DEFAULT_BOUNDS["E_min"]) - 0.05, float(DEFAULT_BOUNDS["E_max"]) + 0.05)
    ax.set_zlim(float(DEFAULT_BOUNDS["O_min"]) - 0.05, 1.05)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)
    if show_box:
        draw_eto_box(ax, DEFAULT_BOUNDS)


def main() -> None:
    args = parse_args()
    scenario = choose_scenario(args.filter)
    result = run_single_scenario(
        scenario,
        n_traj=args.n_traj,
        shift_T=args.shift_T,
        shift_E=args.shift_E,
        shift_O=args.shift_O,
    )

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or scenario_slug(result["label"])
    outpath = outdir / f"{prefix}_taxonomy_3d_3classifiers.png"

    if not confirm_overwrite(outpath):
        print("Aborted: not overwriting existing files.")
        return

    points = collect_points(result, args.stride)

    fig = plt.figure(figsize=(18, 6))
    all_labels = []
    color_map = taxonomy_classifier.STATE_COLORS

    for i, (title, classifier_fn, reset_fn, cmap) in enumerate(PANEL_SPECS, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        labels, coords = classify_points(points, classifier_fn, reset_fn, scenario)
        all_labels.extend(labels)
        color_map = cmap

        if len(coords):
            colors = [cmap.get(label, "#999999") for label in labels]
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=16, alpha=0.9, edgecolors="none")

        ax.set_title(title, fontsize=12, pad=12)
        style_axis(ax, args.elev, args.azim, args.show_box)

    present_states = [state for state in STATE_ORDER if state in set(all_labels)]
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color_map[state], markeredgecolor="none", markersize=8, label=state)
        for state in present_states
    ]

    fig.suptitle(f"3D taxonomy labels across three classifiers\n{result['label']}", fontsize=16, y=0.98)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Scenario: {result['label']}")
    print(f"Taxonomy plot: {outpath}")


if __name__ == "__main__":
    main()
