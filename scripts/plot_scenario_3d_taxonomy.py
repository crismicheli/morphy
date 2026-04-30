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

from config import DEFAULTBOUNDS, DEFAULTPARAMS, DEFAULTSIM
from plotting.plot_helpers import save_taxonomy_plot
from plotting.scenario_helpers import choose_scenario, run_single_scenario
from classifiers.classifier_dispatch import get_classifier_components


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot all trajectory points in 3D E,T,O space colored by taxonomy state, without animation.")
    p.add_argument("--filter", default="Intermediate porosity", help="Substring used to choose a scenario label.")
    p.add_argument("--output", default=str(REPOROOT / "figures" / "taxonomy_trajectory_states_3d.png"), help="Output figure path.")
    p.add_argument("--classifier-type", choices=["static", "temporal", "state_machine"], default="static", help="Classifier backend.")
    p.add_argument("--n-traj", type=int, default=DEFAULTSIM["n_traj"], help="Number of trajectories.")
    p.add_argument("--shift-T", type=float, default=1.0, help="Multiplier applied to initial T center.")
    p.add_argument("--shift-E", type=float, default=1.0, help="Multiplier applied to initial E center.")
    p.add_argument("--shift-O", type=float, default=1.0, help="Multiplier applied to initial O center.")
    p.add_argument("--elev", type=float, default=24.0, help="3D camera elevation.")
    p.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth.")
    p.add_argument("--show-box", action="store_true", help="Show translucent ETO viability box.")
    p.add_argument("--stride", type=int, default=8, help="Subsample factor for plotted timepoints to reduce overplotting.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scenario = choose_scenario(args.filter)
    result = run_single_scenario(scenario, n_traj=args.n_traj, shift_T=args.shift_T, shift_E=args.shift_E, shift_O=args.shift_O)
    classifier_fn, reset_fn, state_colors = get_classifier_components(args.classifier_type)
    output_path = Path(args.output)
    save_taxonomy_plot(
        result,
        scenario,
        output_path,
        bounds=DEFAULTBOUNDS,
        par=DEFAULTPARAMS,
        classifier_fn=classifier_fn,
        color_map=state_colors,
        stride=args.stride,
        elev=args.elev,
        azim=args.azim,
        show_box=args.show_box,
        reset_fn=reset_fn,
    )
    print(f"Scenario: {result['label']}")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
