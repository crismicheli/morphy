#!/usr/bin/env python3
"""
scripts/animate_scenario_3d_eto.py
---------------------------------
Animate one representative viability-kernel simulation in the 3D
(E, T, O) phenotypic space and save it as a GIF.

Design choices follow the attached 2D animator:
- pick a scenario by label substring
- use the same scenario-dependent initial condition shifting logic
- classify viability using the same ensemble reports
- color viable trajectories blue and non-viable trajectories red
- keep curvature C outside the 3D axes and show it in a bottom-right panel

State convention:
sol.y[0] -> C (curvature guidance / polarity-like organizer)
sol.y[1] -> T (cytoskeletal tension)
sol.y[2] -> E (ECM density)
sol.y[3] -> O (oxygen availability)

Why ETO for phenotype space?
- E captures matrix state
- T captures mechanical state
- O captures metabolic / transport support
- C is still shown because it remains a viability variable and an upstream
  organizer of tension, but it is displayed separately to avoid collapsing the
  metabolic dimension out of the main phenotype view.

Alternative phenotype-space characterizers from static parameters
---------------------------------------------------------------
If you want a parameter-defined phenotype/control space instead of a pure state
space, the most interpretable alternatives from DEFAULT_PARAMS are:
- (p, beta, rho): architecture / mechanocoupling / oxygen support
- (a, eta, delta_E): geometric guidance / damping / remodelling
- (beta, eta, delta_E): mechanical gain / mechanical stabilization / matrix turnover
- (rho, mu, delta_O): oxygen supply / oxygen consumption / oxygen loss
These are better interpreted as control or regime coordinates, not instantaneous
phenotypic coordinates, because they are fixed within a trajectory while E, T,
O, and C evolve over time.
"""

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

from config import DEFAULT_PARAMS, DEFAULT_BOUNDS, DEFAULT_SIM, SCENARIOS
from viabilitykernels.simulation import run_scenario

BLUE = "#2166ac"
RED = "#d73027"
LIGHT_BLUE = "#92c5de"
LIGHT_RED = "#f4a582"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate one scenario in 3D ETO space and save a GIF.")
    parser.add_argument(
        "--filter",
        default="Intermediate porosity",
        help="Substring used to choose a scenario label (default: 'Intermediate porosity').",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "figures" / "intermediate_porosity_eto_3d.gif"),
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
    parser.add_argument(
        "--shift-T",
        type=float,
        default=1.0,
        help="Multiplier applied to the initial T center (default: 1.0).",
    )
    parser.add_argument(
        "--shift-E",
        type=float,
        default=1.0,
        help="Multiplier applied to the initial E center (default: 1.0).",
    )
    parser.add_argument(
        "--shift-O",
        type=float,
        default=1.0,
        help="Multiplier applied to the initial O center (default: 1.0).",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=24.0,
        help="3D camera elevation in degrees.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-58.0,
        help="3D camera azimuth in degrees.",
    )
    return parser.parse_args()


def choose_scenario(keyword: str) -> dict:
    matches = [s for s in SCENARIOS if keyword.lower() in s["label"].lower()]
    if not matches:
        labels = "\n- " + "\n- ".join(s["label"] for s in SCENARIOS)
        raise SystemExit(f"No scenario matched filter '{keyword}'. Available scenarios:{labels}")
    return matches[0]


def make_bottom_right_text(scenario: dict, effective_par: dict) -> str:
    keys = [k for k in ["a", "beta", "eta", "delta_E", "rho", "s"] if k in effective_par]
    params_txt = ", ".join(f"{k}={effective_par[k]:.2f}" for k in keys)
    return (
        f"C shown outside phenotype cube\n"
        f"C is a viability variable and upstream organizer of T\n"
        f"Static regime knobs: p={scenario['p']:.2f}; {params_txt}\n"
        f"Phenotype space here: E (matrix), T (mechanics), O (oxygen)"
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
    p = result["p"]
    label = result["label"]
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

    E_max_axis = max(2.0, DEFAULT_BOUNDS["E_max"] * 1.05)
    T_max_axis = max(1.6, DEFAULT_BOUNDS["T_max"] * 1.05)
    O_max_axis = 1.4

    ax3d.set_xlim(0, E_max_axis)
    ax3d.set_ylim(0, T_max_axis)
    ax3d.set_zlim(0, O_max_axis)
    ax3d.set_xlabel("ECM density E")
    ax3d.set_ylabel("Cytoskeletal tension T")
    ax3d.set_zlabel("Oxygen O")
    ax3d.view_init(elev=args.elev, azim=args.azim)
    ax3d.grid(True, alpha=0.25)

    e0, e1 = DEFAULT_BOUNDS["E_min"], DEFAULT_BOUNDS["E_max"]
    t0, t1 = DEFAULT_BOUNDS["T_min"], DEFAULT_BOUNDS["T_max"]
    o0 = DEFAULT_BOUNDS["O_min"]
    faces = [
        np.array([[e0, t0, o0], [e1, t0, o0], [e1, t1, o0], [e0, t1, o0], [e0, t0, o0]]),
        np.array([[e0, t0, o0], [e1, t0, o0], [e1, t0, O_max_axis], [e0, t0, O_max_axis], [e0, t0, o0]]),
        np.array([[e0, t1, o0], [e1, t1, o0], [e1, t1, O_max_axis], [e0, t1, O_max_axis], [e0, t1, o0]]),
        np.array([[e0, t0, o0], [e0, t1, o0], [e0, t1, O_max_axis], [e0, t0, O_max_axis], [e0, t0, o0]]),
        np.array([[e1, t0, o0], [e1, t1, o0], [e1, t1, O_max_axis], [e1, t0, O_max_axis], [e1, t0, o0]]),
    ]
    for face in faces:
        ax3d.plot(face[:, 0], face[:, 1], face[:, 2], color="#4dac26", alpha=0.30, lw=1.0)

    colors = [BLUE if r.viable else RED for r in reports]
    fades = [LIGHT_BLUE if r.viable else LIGHT_RED for r in reports]
    lines = [ax3d.plot([], [], [], lw=1.7, alpha=0.90, color=c)[0] for c in colors]
    points = [ax3d.plot([], [], [], "o", ms=4.2, color=c, zorder=6)[0] for c in colors]

    c_lines = [axc.plot([], [], lw=1.4, alpha=0.90, color=f)[0] for f in fades]
    c_points = [axc.plot([], [], "o", ms=3.6, color=c)[0] for c in colors]

    axc.set_xlim(0, solutions[0].t[n_time - 1])
    axc.set_ylim(0, max(1.25, max(sol.y[0][:n_time].max() for sol in solutions) * 1.08))
    axc.axhline(DEFAULT_BOUNDS["C_min"], color="#6b6b6b", ls="--", lw=1.0, alpha=0.85)
    axc.set_title("C trace", fontsize=10, pad=8)
    axc.set_xlabel("time")
    axc.set_ylabel("C")
    axc.grid(alpha=0.25)

    viable_count = sum(r.viable for r in reports)
    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=2, label="Viable kernel trajectory"),
        Line2D([0], [0], color=RED, lw=2, label="Non-viable trajectory"),
        Line2D([0], [0], color="#6b6b6b", lw=1, ls="--", label=f"C_min={DEFAULT_BOUNDS['C_min']:.2f}"),
    ]
    ax3d.legend(handles=legend_handles, loc="upper left", frameon=True)

    fig.suptitle(f"{label}\nAnimated ensemble in the 3D (E, T, O) phenotype space", fontsize=13, fontweight="bold")

    subtitle = ax3d.text2D(
        0.02,
        0.98,
        f"Viable trajectories: {viable_count}/{len(reports)} | p={p:.2f}",
        transform=ax3d.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.92),
    )
    frame_text = ax3d.text2D(
        0.02,
        0.90,
        "t = 0.00",
        transform=ax3d.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.85", alpha=0.88),
    )
    axinfo.text(
        0.02,
        0.98,
        make_bottom_right_text(scenario, effective_par),
        va="top",
        ha="left",
        fontsize=9.2,
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fbfbfb", ec="#d0d0d0", alpha=0.96),
    )

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
            E_traj = sol.y[2][: frame + 1]
            T_traj = sol.y[1][: frame + 1]
            O_traj = sol.y[3][: frame + 1]
            C_traj = sol.y[0][: frame + 1]
            tt = sol.t[: frame + 1]

            line.set_data(E_traj, T_traj)
            line.set_3d_properties(O_traj)
            point.set_data([E_traj[-1]], [T_traj[-1]])
            point.set_3d_properties([O_traj[-1]])

            c_line.set_data(tt, C_traj)
            c_point.set_data([tt[-1]], [C_traj[-1]])

        frame_text.set_text(f"t = {t_now:.2f}")
        return [*lines, *points, *c_lines, *c_points, subtitle, frame_text]

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_time,
        interval=1000 / max(args.fps, 1),
        blit=False,
    )

    writer = PillowWriter(fps=args.fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)

    print(f"Scenario: {label}")
    print(f"Saved GIF: {output_path}")


if __name__ == "__main__":
    main()
