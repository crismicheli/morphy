#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPTDIR = Path(__file__).resolve().parent
REPOROOT = SCRIPTDIR.parent
PACKAGEPARENT = REPOROOT.parent
for p in (str(PACKAGEPARENT), str(REPOROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import DEFAULT_BOUNDS, DEFAULT_PARAMS, DEFAULT_SIM
from plotting.plot_helpers import compute_point_labels, draw_eto_box
from plotting.scenario_helpers import choose_scenario, run_single_scenario, scenario_slug
from classifiers import taxonomy_classifier, temporal_taxonomy_classifier, state_machine_classifier


CLASSIFIERS = {
    "static": (
        taxonomy_classifier.classify_state,
        lambda: None,
        taxonomy_classifier.STATE_COLORS,
        "Static classifier",
    ),
    "temporal": (
        temporal_taxonomy_classifier.classify_state,
        temporal_taxonomy_classifier.reset_classifier_memory,
        temporal_taxonomy_classifier.STATE_COLORS,
        "Temporal classifier",
    ),
    "state_machine": (
        state_machine_classifier.classify_state,
        state_machine_classifier.reset_classifier_memory,
        state_machine_classifier.STATE_COLORS,
        "State-machine classifier",
    ),
}

PANEL_ORDER = ["static", "temporal", "state_machine"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 3-panel 3D taxonomy plot for static, temporal, and state-machine classifiers on one scenario."
    )
    parser.add_argument("--filter", default="Intermediate porosity", help="Substring used to choose a scenario label.")
    parser.add_argument(
        "--out-dir",
        default=str(REPOROOT / "figures" / "scenario_single_outputs"),
        help="Output directory.",
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
    parser.add_argument(
        "--out-file",
        default=None,
        help="Optional explicit output file path. Defaults to <prefix>_taxonomy_3d_3classifiers.png",
    )
    return parser.parse_args()


def confirm_overwrite(path: Path) -> bool:
    if not path.exists():
        return True
    reply = input(f"File {path} already exists. Overwrite? [y/N] ").strip().lower()
    return reply in {"y", "yes"}


def _style_axis(ax, bounds: dict, elev: float, azim: float, show_box: bool) -> None:
    ax.set_xlabel("T")
    ax.set_ylabel("E")
    ax.set_zlabel("O")
    ax.set_xlim(float(bounds["T_min"]) - 0.05, float(bounds["T_max"]) + 0.05)
    ax.set_ylim(float(bounds["E_min"]) - 0.05, float(bounds["E_max"]) + 0.05)
    ax.set_zlim(float(bounds["O_min"]) - 0.05, 1.05)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)
    if show_box:
        draw_eto_box(ax, bounds)


def _plot_classifier_panel(ax, result: dict, scenario: dict, classifier_key: str, args: argparse.Namespace):
    classifier_fn, reset_fn, color_map, title = CLASSIFIERS[classifier_key]
    labels, coords = compute_point_labels(
        result,
        bounds=DEFAULT_BOUNDS,
        par=DEFAULT_PARAMS,
        classifier_fn=classifier_fn,
        stride=args.stride,
        reset_fn=reset_fn,
        scenario_cfg=scenario,
    )

    T_vals, E_vals, O_vals = coords[:, 0], coords[:, 1], coords[:, 2]
    point_colors = [color_map.get(label, "#999999") for label in labels]

    ax.scatter(T_vals, E_vals, O_vals, c=point_colors, s=18, alpha=0.9, edgecolors="none")
    _style_axis(ax, DEFAULT_BOUNDS, args.elev, args.azim, args.show_box)
    ax.set_title(title, fontsize=12, pad=12)
    return labels, color_map


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
    out_path = Path(args.out_file) if args.out_file else outdir / f"{prefix}_taxonomy_3d_3classifiers.png"

    if not confirm_overwrite(out_path):
        print("Aborted: not overwriting existing file.")
        return

    fig = plt.figure(figsize=(18, 6))
    all_labels = []
    color_map = taxonomy_classifier.STATE_COLORS

    for i, key in enumerate(PANEL_ORDER, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        labels, cmap = _plot_classifier_panel(ax, result, scenario, key, args)
        all_labels.extend(labels)
        color_map = cmap

    ordered_states = [
        "Apoptosis",
        "Migration",
        "Proliferation",
        "Quiescence",
        "Diversification",
        "Undetermined",
    ]
    present_states = [state for state in ordered_states if state in set(all_labels)]
    handles = [Patch(facecolor=color_map[state], edgecolor="none", label=state) for state in present_states]

    fig.suptitle(f"3D taxonomy labels across three classifiers\n{result['label']}", fontsize=16, y=0.98)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Scenario: {result['label']}")
    print(f"3-classifier taxonomy plot: {out_path}")


if __name__ == "__main__":
    main()
