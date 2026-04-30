#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULTBOUNDS, DEFAULTSIM
from plotting.plot_helpers import save_trajectory_animation
from plotting.scenario_helpers import choose_scenario, run_single_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate one scenario in 3D ETO space.")
    parser.add_argument("--filter", default="Intermediate porosity", help="Substring used to choose a scenario label.")
    parser.add_argument("--output", default=str(ROOT / "figures" / "intermediate_porosity_eto_single_clean.gif"), help="Output GIF path.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second.")
    parser.add_argument("--max-frames", type=int, default=160, help="Maximum number of frames.")
    parser.add_argument("--n-traj", type=int, default=DEFAULTSIM["n_traj"], help="Number of trajectories.")
    parser.add_argument("--shift-T", type=float, default=1.0, help="Multiplier applied to initial T center.")
    parser.add_argument("--shift-E", type=float, default=1.0, help="Multiplier applied to initial E center.")
    parser.add_argument("--shift-O", type=float, default=1.0, help="Multiplier applied to initial O center.")
    parser.add_argument("--elev", type=float, default=24.0, help="3D camera elevation.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth.")
    parser.add_argument("--show-box", action="store_true", help="Show translucent ETO viability box.")
    parser.add_argument("--c-stat", choices=["mean", "median", "min", "max"], default="mean", help="How to aggregate C across trajectories for the numeric readout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = choose_scenario(args.filter)
    result = run_single_scenario(scenario, n_traj=args.n_traj, shift_T=args.shift_T, shift_E=args.shift_E, shift_O=args.shift_O)
    output_path = Path(args.output)
    save_trajectory_animation(
        result,
        output_path,
        bounds=DEFAULTBOUNDS,
        fps=args.fps,
        max_frames=args.max_frames,
        elev=args.elev,
        azim=args.azim,
        show_box=args.show_box,
        c_stat=args.c_stat,
    )
    print(f"Scenario: {result['label']}")
    print(f"Saved GIF: {output_path}")


if __name__ == "__main__":
    main()
