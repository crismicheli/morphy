#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config import DEFAULT_PARAMS, DEFAULT_BOUNDS, DEFAULT_SIM, SCENARIOS
from viabilitykernels.simulation import run_scenario

BLUE = "#2166ac"
RED = "#d73027"
LIGHT_BLUE = "#92c5de"
LIGHT_RED = "#f4a582"
BOX_GREEN = "#4dac26"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate one scenario in 3D ETO space with translucent viability box.")
    parser.add_argument("--filter", default="Intermediate porosity")
    parser.add_argument("--output", default=str(ROOT / "figures" / "intermediate_porosity_eto_3d_box.gif"))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--n-traj", type=int, default=DEFAULT_SIM["n_traj"])
    parser.add_argument("--shift-T", type=float, default=1.0)
    parser.add_argument("--shift-E", type=float, default=1.0)
    parser.add_argument("--shift-O", type=float, default=1.0)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    return parser.parse_args()


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
    )

    solutions = result["solutions"]
    reports = result["reports"]
    effective_par = result["effective_par"]
    n_time = min(len(solutions[0].t), args.max_frames)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9.6, 6.8), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[3.1, 1.25], height_ratios=[3.0, 1.05])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    axc = fig.add_subplot(gs[1, 1])
    axinfo = fig.add_subplot(gs[0, 1])
    axinfo.axis("off")

    E_max_axis = max(2.0, DEFAULT_BOUNDS["E_max"] * 1.08)
    T_max_axis = max(1.6, DEFAULT_BOUNDS["T_max"] * 1.08)
    O_max_axis = 1.4

    ax3d.set_xlim(0, E_max_axis)
    ax3d.set_ylim(0, T_max_axis)
    ax3d.set_zlim(0, O_max_axis)
    ax3d.set_xlabel("ECM density E")
    ax3d.set_ylabel("Cytoskeletal tension T")
    ax3d.set_zlabel("Oxygen O")
    ax3d.view_init(elev=args.elev, azim=args.azim)

    faces = viability_faces(
        DEFAULT_BOUNDS["E_min"], DEFAULT_BOUNDS["E_max"],
        DEFAULT_BOUNDS["T_min"], DEFAULT_BOUNDS["T_max"],
        DEFAULT_BOUNDS["O_min"], O_max_axis,
    )
    box = Poly3DCollection(faces, facecolors=BOX_GREEN, edgecolors=BOX_GREEN, linewidths=0.8, alpha=0.08)
    ax3d.add_collection3d(box)

    colors = [BLUE if r.viable else RED for r in reports]
    fades = [LIGHT_BLUE if r.viable else LIGHT_RED for r in reports]
    lines = [ax3d.plot([], [], [], lw=1.7, alpha=0.9, color=c)[0] for c in colors]
    points = [ax3d.plot([], [], [], "o", ms=4.2, color=c, zorder=6)[0] for c in colors]
    c_lines = [axc.plot([], [], lw=1.4, alpha=0.9, color=f)[0] for f in fades]
    c_points = [axc.plot([], [], "o", ms=3.6, color=c)[0] for c in colors]

    axc.set_xlim(0, solutions[0].t[n_time - 1])
    axc.set_ylim(0, max(1.25, max(sol.y[0][:n_time].max() for sol in solutions) * 1.08))
    axc.axhline(DEFAULT_BOUNDS["C_min"], color="#6b6b6b", ls="--", lw=1.0, alpha=0.85)
    axc.set_title("C trace", fontsize=10)
    axc.set_xlabel("time")
    axc.set_ylabel("C")
    axc.grid(alpha=0.25)

    viable_count = sum(r.viable for r in reports)
    fig.suptitle(f"{result['label']}\nAnimated ensemble in the 3D (E, T, O) phenotype space", fontsize=13, fontweight="bold")
    subtitle = ax3d.text2D(
        0.02, 0.98,
        f"Viable trajectories: {viable_count}/{len(reports)} | p={result['p']:.2f}",
        transform=ax3d.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.92),
    )
    frame_text = ax3d.text2D(
        0.02, 0.90, "t = 0.00",
        transform=ax3d.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.85", alpha=0.88),
    )
    axinfo.text(
        0.02, 0.98,
        "Translucent box = admissible ETO region\n"
        "O is dynamic because it is one row of the integrated state [C,T,E,O]\n"
        f"O follows dO/dt = rho*h(p) - mu*E*O - delta_O*O\n"
        f"Static regime parameters here: a={effective_par['a']:.2f}, beta={effective_par['beta']:.2f}, eta={effective_par['eta']:.2f}, delta_E={effective_par['delta_E']:.2f}, rho={effective_par['rho']:.2f}, s={effective_par['s']:.2f}",
        va="top", ha="left", fontsize=9.2, linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fbfbfb", ec="#d0d0d0", alpha=0.96),
    )

    ax3d.legend(handles=[
        Line2D([0], [0], color=BLUE, lw=2, label="Viable trajectory"),
        Line2D([0], [0], color=RED, lw=2, label="Non-viable trajectory"),
        Line2D([0], [0], color=BOX_GREEN, lw=3, alpha=0.6, label="ETO viability region"),
    ], loc="upper left", frameon=True)

    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        for line, point in zip(c_lines, c_points):
            line.set_data([], [])
            point.set_data([], [])
        frame_text.set_text("t = 0.00")
        return [*lines, *points, *c_lines, *c_points, subtitle, frame_text]

    def update(frame: int):
        t_now = solutions[0].t[frame]
        for sol, line, point, c_line, c_point in zip(solutions, lines, points, c_lines, c_points):
            E = sol.y[2][:frame + 1]
            T = sol.y[1][:frame + 1]
            O = sol.y[3][:frame + 1]
            C = sol.y[0][:frame + 1]
            tt = sol.t[:frame + 1]
            line.set_data(E, T)
            line.set_3d_properties(O)
            point.set_data([E[-1]], [T[-1]])
            point.set_3d_properties([O[-1]])
            c_line.set_data(tt, C)
            c_point.set_data([tt[-1]], [C[-1]])
        frame_text.set_text(f"t = {t_now:.2f}")
        return [*lines, *points, *c_lines, *c_points, subtitle, frame_text]

    anim = FuncAnimation(fig, update, init_func=init, frames=n_time, interval=1000 / max(args.fps, 1), blit=False)
    anim.save(output_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved GIF: {output_path}")


if __name__ == "__main__":
    main()
