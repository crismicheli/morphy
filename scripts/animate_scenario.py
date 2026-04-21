#!/usr/bin/env python3
"""
scripts/animate_scenario.py
---------------------------
Animate a representative viability-kernel simulation and save it as a GIF.

This script is written to match the current repo naming:
- top-level config package
- top-level viability_kernels package

Default behavior:
- picks the first scenario whose label contains "Intermediate porosity"
- runs one ensemble simulation using the shared defaults
- animates all trajectories in the (E, T) phase plane
- saves the GIF to figures/intermediate_porosity_animation.gif

Examples
--------
python scripts/animate_scenario.py
python scripts/animate_scenario.py --filter "High porosity"
python scripts/animate_scenario.py --output figures/high_porosity.gif
python scripts/animate_scenario.py --fps 12 --max-frames 180
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from config import DEFAULT_PARAMS, DEFAULT_BOUNDS, DEFAULT_SIM, SCENARIOS
from viability_kernels.simulation import run_scenario
from viability_kernels.phase_plane import ET_field, make_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate one simulation scenario and save a GIF.")
    parser.add_argument(
        "--filter",
        default="Intermediate porosity",
        help="Substring used to choose a scenario label (default: 'Intermediate porosity').",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "figures" / "intermediate_porosity_animation.gif"),
        help="Output GIF path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the GIF (default: 10).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=160,
        help="Maximum number of animation frames to render (default: 160).",
    )
    parser.add_argument(
        "--n-traj",
        type=int,
        default=DEFAULT_SIM["n_traj"],
        help="Number of trajectories to simulate (default from DEFAULT_SIM).",
    )
    return parser.parse_args()


def choose_scenario(keyword: str) -> dict:
    matches = [s for s in SCENARIOS if keyword.lower() in s["label"].lower()]
    if not matches:
        labels = "\n- " + "\n- ".join(s["label"] for s in SCENARIOS)
        raise SystemExit(f"No scenario matched filter '{keyword}'. Available scenarios:{labels}")
    return matches[0]


def main() -> None:
    args = parse_args()
    scenario = choose_scenario(args.filter)

    x0_center = np.array(DEFAULT_SIM["x0_center"])
    result = run_scenario(
        scenario_cfg=scenario,
        par=DEFAULT_PARAMS,
        bounds=DEFAULT_BOUNDS,
        x0_center=x0_center,
        n_traj=args.n_traj,
        t_span=tuple(DEFAULT_SIM["t_span"]),
        n_eval=DEFAULT_SIM["n_eval"],
        rng_seed=DEFAULT_SIM["rng_seed"],
    )

    solutions = result["solutions"]
    reports = result["reports"]
    p = result["p"]
    label = result["label"]

    n_time = min(len(solutions[0].t), args.max_frames)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)

    E_max_axis = 2.0
    T_max_axis = 1.6

    EE, TT = make_grid((0, E_max_axis), (0, T_max_axis), n_points=20)
    dE, dT = ET_field(EE, TT, p, DEFAULT_PARAMS)
    speed = np.sqrt(dE**2 + dT**2) + 1e-9
    ax.quiver(
        EE,
        TT,
        dE / speed,
        dT / speed,
        angles="xy",
        scale_units="xy",
        scale=12,
        alpha=0.25,
        color="grey",
    )

    viability_rect = Rectangle(
        (DEFAULT_BOUNDS["E_min"], DEFAULT_BOUNDS["T_min"]),
        DEFAULT_BOUNDS["E_max"] - DEFAULT_BOUNDS["E_min"],
        DEFAULT_BOUNDS["T_max"] - DEFAULT_BOUNDS["T_min"],
        facecolor="#4dac26",
        alpha=0.10,
        edgecolor="none",
    )
    ax.add_patch(viability_rect)

    colors = ["#2166ac" if r.viable else "#d73027" for r in reports]
    lines = [ax.plot([], [], lw=1.6, alpha=0.85, color=c)[0] for c in colors]
    points = [ax.plot([], [], "o", ms=4, color=c, zorder=5)[0] for c in colors]

    ax.set_title(f"{label}\nAnimated ensemble in the (E, T) phase plane", fontsize=12, fontweight="bold")
    ax.set_xlabel("ECM density E")
    ax.set_ylabel("Cytoskeletal tension T")
    ax.set_xlim(0, E_max_axis)
    ax.set_ylim(0, T_max_axis)
    ax.grid(alpha=0.2)

    viable_count = sum(r.viable for r in reports)
    subtitle = ax.text(
        0.02,
        0.98,
        f"Viable trajectories: {viable_count}/{len(reports)} | p={p:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
    )

    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            point.set_data([], [])
        return [*lines, *points, subtitle]

    def update(frame: int):
        for sol, line, point in zip(solutions, lines, points):
            E_traj = sol.y[2][: frame + 1]
            T_traj = sol.y[1][: frame + 1]
            line.set_data(E_traj, T_traj)
            point.set_data([E_traj[-1]], [T_traj[-1]])
        return [*lines, *points, subtitle]

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_time,
        interval=1000 / max(args.fps, 1),
        blit=True,
    )

    writer = PillowWriter(fps=args.fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)

    print(f"Scenario: {label}")
    print(f"Saved GIF: {output_path}")


if __name__ == "__main__":
    main()
