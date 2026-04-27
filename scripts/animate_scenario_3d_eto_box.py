#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_PARENT = REPO_ROOT.parent

for p in (str(PACKAGE_PARENT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config import DEFAULT_PARAMS, DEFAULT_BOUNDS, DEFAULT_SIM, SCENARIOS
from viabilitykernels.simulation import run_scenario, sample_initial_conditions

BOX_GREEN = "#4dac26"
OUTSIDE_COLOR = "#d73027"
INSIDE_COLOR = "#2166ac"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Animate a 3D ensemble in (E, T, O) space with an optional viability box. Initial conditions are allowed to start outside the box, but a warning is emitted when they do."
    )
    p.add_argument("--filter", default="Intermediate porosity", help="Substring used to choose a scenario label.")
    p.add_argument("--output", default=str(REPO_ROOT / "figures" / "scenario_3d_eto_box.gif"), help="Output GIF path.")
    p.add_argument("--fps", type=int, default=10, help="Frames per second for the GIF.")
    p.add_argument("--max-frames", type=int, default=180, help="Maximum number of animation frames to render.")
    p.add_argument("--n-traj", type=int, default=DEFAULT_SIM["n_traj"], help="Number of trajectories to simulate.")
    p.add_argument("--shift-T", type=float, default=1.0, help="Multiplier applied to the initial T center.")
    p.add_argument("--shift-E", type=float, default=1.0, help="Multiplier applied to the initial E center.")
    p.add_argument("--shift-O", type=float, default=1.0, help="Multiplier applied to the initial O center.")
    p.add_argument("--elev", type=float, default=24.0, help="3D camera elevation.")
    p.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth.")
    p.add_argument("--show-box", action="store_true", help="Show translucent ETO viability box.")
    return p.parse_args()


def choose_scenario(keyword: str) -> dict:
    matches = [s for s in SCENARIOS if keyword.lower() in s["label"].lower()]
    if not matches:
        labels = "\n- " + "\n- ".join(s["label"] for s in SCENARIOS)
        raise SystemExit(f"No scenario matched filter '{keyword}'. Available scenarios:{labels}")
    return matches[0]


def viability_faces(e0: float, e1: float, t0: float, t1: float, o0: float, o1: float):
    v000 = [e0, t0, o0]
    v100 = [e1, t0, o0]
    v110 = [e1, t1, o0]
    v010 = [e0, t1, o0]
    v001 = [e0, t0, o1]
    v101 = [e1, t0, o1]
    v111 = [e1, t1, o1]
    v011 = [e0, t1, o1]
    return [
        [v000, v100, v110, v010],
        [v001, v101, v111, v011],
        [v000, v100, v101, v001],
        [v010, v110, v111, v011],
        [v000, v010, v011, v001],
        [v100, v110, v111, v101],
    ]


def is_inside_viability_box(x0: np.ndarray, bounds: dict) -> bool:
    C, T, E, O = [float(v) for v in x0]
    return (
        C >= bounds["C_min"]
        and T >= bounds["T_min"]
        and T <= bounds["T_max"]
        and E >= bounds["E_min"]
        and E <= bounds["E_max"]
        and O >= bounds["O_min"]
    )


def warn_if_any_initial_conditions_outside(initial_conditions: list[np.ndarray], bounds: dict) -> None:
    outside = [x0 for x0 in initial_conditions if not is_inside_viability_box(x0, bounds)]
    if outside:
        warnings.warn(
            f"{len(outside)}/{len(initial_conditions)} initial conditions start outside the viability box. This is allowed, but please confirm that this is the intended behavior.",
            stacklevel=2,
        )


def main() -> None:
    args = parse_args()
    scenario = choose_scenario(args.filter)

    x0_center = np.array(DEFAULT_SIM["x0_center"], dtype=float)
    noise_scale = (0.03, 0.03, 0.03, 0.05)
    if scenario["expected"] == "borderline":
        x0_center[1] *= 1.5
        x0_center[2] *= 1.7
        noise_scale = (0.04, 0.08, 0.08, 0.06)
    elif scenario["expected"] == "unstable":
        x0_center[1] *= 1.7
        x0_center[2] *= 1.9
        noise_scale = (0.05, 0.10, 0.10, 0.07)

    x0_center[1] *= args.shift_T
    x0_center[2] *= args.shift_E
    x0_center[3] *= args.shift_O

    initial_conditions = sample_initial_conditions(
        x0_center=x0_center,
        n_traj=args.n_traj,
        noise_scale=noise_scale,
        rng_seed=DEFAULT_SIM["rng_seed"],
    )
    warn_if_any_initial_conditions_outside(initial_conditions, DEFAULT_BOUNDS)

    result = run_scenario(
        scenario_cfg=scenario,
        par=DEFAULT_PARAMS,
        bounds=DEFAULT_BOUNDS,
        x0_center=x0_center,
        n_traj=args.n_traj,
        t_span=tuple(DEFAULT_SIM["t_span"]),
        n_eval=DEFAULT_SIM["n_eval"],
        rng_seed=DEFAULT_SIM["rng_seed"],
        noise_scale=noise_scale,
        initial_conditions=initial_conditions,
    )

    solutions = result["solutions"]
    initial_conditions = result["initial_conditions"]
    label = result["label"]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_time = min(len(solutions[0].t), args.max_frames)

    fig = plt.figure(figsize=(8.6, 7.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    E_max_axis = max(2.0, DEFAULT_BOUNDS["E_max"] * 1.08)
    T_max_axis = max(1.6, DEFAULT_BOUNDS["T_max"] * 1.08)
    O_max_axis = 1.4

    ax.set_xlim(0, E_max_axis)
    ax.set_ylim(0, T_max_axis)
    ax.set_zlim(0, O_max_axis)
    ax.set_xlabel("ECM density E")
    ax.set_ylabel("Cytoskeletal tension T")
    ax.set_zlabel("Oxygen O")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.grid(True, alpha=0.25)

    if args.show_box:
        faces = viability_faces(
            DEFAULT_BOUNDS["E_min"], DEFAULT_BOUNDS["E_max"],
            DEFAULT_BOUNDS["T_min"], DEFAULT_BOUNDS["T_max"],
            DEFAULT_BOUNDS["O_min"], O_max_axis,
        )
        box = Poly3DCollection(
            faces,
            facecolors=BOX_GREEN,
            edgecolors=BOX_GREEN,
            linewidths=0.8,
            alpha=0.08,
        )
        ax.add_collection3d(box)

    line_artists = []
    point_artists = []
    for sol, x0 in zip(solutions, initial_conditions):
        color = INSIDE_COLOR if is_inside_viability_box(x0, DEFAULT_BOUNDS) else OUTSIDE_COLOR
        line = ax.plot([], [], [], lw=1.4, alpha=0.75, color=color)[0]
        point = ax.plot([], [], [], "o", ms=4.5, color=color)[0]
        line_artists.append(line)
        point_artists.append(point)

    ax.set_title(
        f"{label}\nAnimated ensemble in 3D (E, T, O)",
        fontsize=13,
        fontweight="bold",
    )

    ax.text2D(
        0.02,
        0.98,
        "Blue = starts inside box | Red = starts outside box",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.94),
    )

    def init():
        for line, point in zip(line_artists, point_artists):
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        return [*line_artists, *point_artists]

    def update(frame: int):
        for sol, line, point in zip(solutions, line_artists, point_artists):
            E_traj = sol.y[2][: frame + 1]
            T_traj = sol.y[1][: frame + 1]
            O_traj = sol.y[3][: frame + 1]
            line.set_data(E_traj, T_traj)
            line.set_3d_properties(O_traj)
            point.set_data([E_traj[-1]], [T_traj[-1]])
            point.set_3d_properties([O_traj[-1]])
        return [*line_artists, *point_artists]

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
