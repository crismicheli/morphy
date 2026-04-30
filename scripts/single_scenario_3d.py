#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTDIR = Path(__file__).resolve().parent
REPOROOT = SCRIPTDIR.parent
PACKAGEPARENT = REPOROOT.parent
for p in (str(PACKAGEPARENT), str(REPOROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import DEFAULT_BOUNDS, DEFAULT_PARAMS, DEFAULT_SIM
from plotting.plot_helpers import save_taxonomy_plot, save_trajectory_animation
from plotting.scenario_helpers import choose_scenario, run_single_scenario, scenario_slug
from classifiers.classifier_dispatch import get_classifier_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate both a 3D trajectory animation and a taxonomy-labeled 3D plot for one scenario."
    )
    parser.add_argument("--filter", default="Intermediate porosity", help="Substring used to choose a scenario label.")
    parser.add_argument("--out-dir", default=str(REPOROOT / "figures" / "scenario_single_outputs"), help="Output directory.")
    parser.add_argument("--prefix", default=None, help="Optional filename prefix; defaults to scenario label slug.")
    parser.add_argument("--classifier-type", choices=["static", "temporal", "state_machine"], default="temporal", help="Classifier backend.")
    parser.add_argument("--n-traj", type=int, default=DEFAULT_SIM["n_traj"], help="Number of trajectories.")
    parser.add_argument("--shift-T", type=float, default=1.0, help="Multiplier applied to initial T center.")
    parser.add_argument("--shift-E", type=float, default=1.0, help="Multiplier applied to initial E center.")
    parser.add_argument("--shift-O", type=float, default=1.0, help="Multiplier applied to initial O center.")
    parser.add_argument("--elev", type=float, default=24.0, help="3D camera elevation.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth.")
    parser.add_argument("--show-box", action="store_true", help="Show translucent ETO viability box.")
    parser.add_argument("--stride", type=int, default=8, help="Subsample factor for taxonomy plot timepoints.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for GIF output.")
    parser.add_argument("--max-frames", type=int, default=160, help="Maximum animation frames.")
    parser.add_argument("--c-stat", choices=["mean", "median", "min", "max"], default="mean", help="How to aggregate C across trajectories for the numeric readout.")
    return parser.parse_args()


def confirm_overwrite(path: Path) -> bool:
    if not path.exists():
        return True
    reply = input(f"File {path} already exists. Overwrite? [y/N] ").strip().lower()
    return reply in {"y", "yes"}


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
    classifier_fn, reset_fn, state_colors = get_classifier_components(args.classifier_type)

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or scenario_slug(result["label"])
    clf_tag = f"clf-{args.classifier_type}"
    animation_path = outdir / f"{prefix}_3d_{clf_tag}.gif"
    taxonomy_path = outdir / f"{prefix}_taxonomy_3d_{clf_tag}.png"

    if not confirm_overwrite(animation_path) or not confirm_overwrite(taxonomy_path):
        print("Aborted: not overwriting existing files.")
        return

    save_trajectory_animation(
        result,
        animation_path,
        bounds=DEFAULT_BOUNDS,
        fps=args.fps,
        max_frames=args.max_frames,
        elev=args.elev,
        azim=args.azim,
        show_box=args.show_box,
        c_stat=args.c_stat,
    )
    save_taxonomy_plot(
        result,
        scenario,
        taxonomy_path,
        bounds=DEFAULT_BOUNDS,
        par=DEFAULT_PARAMS,
        classifier_fn=classifier_fn,
        color_map=state_colors,
        stride=args.stride,
        elev=args.elev,
        azim=args.azim,
        show_box=args.show_box,
        reset_fn=reset_fn,
    )

    print(f"Scenario: {result['label']}")
    print(f"Animation: {animation_path}")
    print(f"Taxonomy plot: {taxonomy_path}")


if __name__ == "__main__":
    main()
