#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_BOUNDS, DEFAULT_SIM
from plotting.plot_helpers import save_trajectory_animation
from plotting.scenario_helpers import choose_scenario, run_single_scenario, scenario_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate one scenario in 3D ETO space.")
    parser.add_argument("--filter", default="Intermediate porosity", help="Substring used to choose a scenario label.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output GIF path. If omitted, a scenario-based name is created in root/figures.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second.")
    parser.add_argument("--max-frames", type=int, default=160, help="Maximum number of frames.")
    parser.add_argument("--n-traj", type=int, default=DEFAULT_SIM["n_traj"], help="Number of trajectories.")
    parser.add_argument(
        "--x0",
        type=float,
        nargs=4,
        metavar=("C", "T", "E", "O"),
        default=None,
        help="Explicit initial center as a 4D point in (C, T, E, O) order. If omitted, DEFAULT_SIM['x0_center'] is used.",
    )
    parser.add_argument("--elev", type=float, default=24.0, help="3D camera elevation.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth.")
    parser.add_argument("--show-box", action="store_true", help="Show translucent viability box.")
    parser.add_argument(
        "--c-stat",
        choices=["mean", "median", "min", "max"],
        default="mean",
        help="How to aggregate C across trajectories for the numeric readout.",
    )
    return parser.parse_args()


def resolve_output_path(result_label: str, output_arg: str | None) -> Path:
    if output_arg:
        output_path = Path(output_arg)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        return output_path
    slug = scenario_slug(result_label)
    return ROOT / "figures" / f"{slug}_3d.gif"


def main() -> None:
    args = parse_args()
    scenario = choose_scenario(args.filter)
    result = run_single_scenario(
        scenario,
        n_traj=args.n_traj,
        x0_center=args.x0,
    )

    output_path = resolve_output_path(result["label"], args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_trajectory_animation(
        result,
        output_path,
        bounds=DEFAULT_BOUNDS,
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
