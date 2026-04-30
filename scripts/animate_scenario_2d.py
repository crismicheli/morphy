#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_BOUNDS, DEFAULT_PARAMS, DEFAULT_SIM
from plotting.plot2d_helpers import save_et_animation
from plotting.scenario_helpers import choose_scenario, run_single_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate one simulation scenario in the 2D E-T phase plane and save a GIF.")
    parser.add_argument("--filter", default="Intermediate porosity", help="Substring used to choose a scenario label.")
    parser.add_argument("--output", default=str(ROOT / "figures" / "intermediate_porosity_animation.gif"), help="Output GIF path.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the GIF.")
    parser.add_argument("--max-frames", type=int, default=160, help="Maximum number of animation frames to render.")
    parser.add_argument("--n-traj", type=int, default=DEFAULT_SIM["n_traj"], help="Number of trajectories to simulate.")
    parser.add_argument("--shift-T", type=float, default=1.0, help="Multiplier applied to the initial T center.")
    parser.add_argument("--shift-E", type=float, default=1.0, help="Multiplier applied to the initial E center.")
    parser.add_argument("--shift-O", type=float, default=1.0, help="Multiplier applied to the initial O center.")
    parser.add_argument("--hide-box", action="store_true", help="Hide the ET viability rectangle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = choose_scenario(args.filter)
    result = run_single_scenario(scenario, n_traj=args.n_traj, shift_T=args.shift_T, shift_E=args.shift_E, shift_O=args.shift_O)
    output_path = Path(args.output)
    save_et_animation(
        result,
        scenario,
        output_path,
        bounds=DEFAULT_BOUNDS,
        par=DEFAULT_PARAMS,
        fps=args.fps,
        max_frames=args.max_frames,
        show_box=not args.hide_box,
    )
    print(f"Scenario: {result['label']}")
    print(f"Saved GIF: {output_path}")


if __name__ == "__main__":
    main()
