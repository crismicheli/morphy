#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config import DEFAULT_PARAMS, DEFAULT_BOUNDS, DEFAULT_SIM, SCENARIOS
from viabilitykernels.simulation import run_scenario
from viabilitykernels.viability import check_trajectory

BLUE = "#2166ac"
RED = "#d73027"
BOX_FACE = (0.301, 0.675, 0.149, 0.16)
BOX_EDGE = (0.301, 0.675, 0.149, 0.70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate one scenario in 3D ETO space.")
    parser.add_argument("--filter", default="Intermediate porosity", help="Substring used to choose a scenario label.")
    parser.add_argument("--output", default=None, help="Output GIF path. If omitted, a scenario-based name is generated.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second.")
    parser.add_argument("--max-frames", type=int, default=160, help="Maximum number of frames.")
    parser.add_argument("--n-traj", type=int, default=DEFAULT_SIM["n_traj"], help="Number of trajectories.")
    parser.add_argument("--shift-T", type=float, default=1.0, help="Multiplier applied to initial T center.")
    parser.add_argument("--shift-E", type=float, default=1.0, help="Multiplier applied to initial E center.")
    parser.add_argument("--shift-O", type=float, default=1.0, help="Multiplier applied to initial O center.")
    parser.add_argument("--elev", type=float, default=24.0, help="3D camera elevation.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth.")
    parser.add_argument("--show-box", action="store_true", help="Show translucent ETO admissible box.")
    parser.add_argument("--c-stat", choices=["mean", "median", "min", "max"], default="mean", help="How to aggregate C across trajectories for the numeric readout.")
    parser.add_argument("--focus-index", type=int, default=0, help="Trajectory index used for exit diagnosis readout.")
    return parser.parse_args()


def choose_scenario(keyword: str) -> dict:
    matches = [s for s in SCENARIOS if keyword.lower() in s["label"].lower()]
    if not matches:
        labels = "\n- " + "\n- ".join(s["label"] for s in SCENARIOS)
        raise SystemExit(f"No scenario matched filter '{keyword}'. Available scenarios:{labels}")
    return matches[0]


def slugify(label: str) -> str:
    s = label.lower()
    s = s.replace("η", "eta").replace("β", "beta").replace("δ", "delta").replace("ρ", "rho")
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "scenario"


def default_output_path(label: str, show_box: bool) -> Path:
    suffix = "_eto_3d_box.gif" if show_box else "_eto_3d.gif"
    return ROOT / "figures" / f"{slugify(label)}{suffix}"


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


def aggregate(values: np.ndarray, mode: str) -> float:
    if mode == "mean":
        return float(np.mean(values))
    if mode == "median":
        return float(np.median(values))
    if mode == "min":
        return float(np.min(values))
    if mode == "max":
        return float(np.max(values))
    raise ValueError(mode)


def first_exit_index(sol, bounds: dict) -> int | None:
    C, T, E, O = sol.y
    inside = (
        (C >= bounds["C_min"]) &
        (T >= bounds["T_min"]) & (T <= bounds["T_max"]) &
        (E >= bounds["E_min"]) & (E <= bounds["E_max"]) &
        (O >= bounds["O_min"])
    )
    bad = np.where(~inside)[0]
    return None if len(bad) == 0 else int(bad[0])


def add_box(ax, bounds: dict, z_top: float) -> None:
    faces = viability_faces(
        bounds["E_min"], bounds["E_max"],
        bounds["T_min"], bounds["T_max"],
        bounds["O_min"], z_top,
    )
    box = Poly3DCollection(faces, facecolors=BOX_FACE, edgecolors=BOX_EDGE, linewidths=1.1)
    box.set_zsort("min")
    ax.add_collection3d(box)

    e0, e1 = bounds["E_min"], bounds["E_max"]
    t0, t1 = bounds["T_min"], bounds["T_max"]
    o0, o1 = bounds["O_min"], z_top
    edges = [
        ([e0, e1], [t0, t0], [o0, o0]), ([e0, e1], [t1, t1], [o0, o0]),
        ([e0, e0], [t0, t1], [o0, o0]), ([e1, e1], [t0, t1], [o0, o0]),
        ([e0, e0], [t0, t0], [o0, o1]), ([e1, e1], [t0, t0], [o0, o1]),
        ([e0, e0], [t1, t1], [o0, o1]), ([e1, e1], [t1, t1], [o0, o1]),
        ([e0, e1], [t0, t0], [o1, o1]), ([e0, e1], [t1, t1], [o1, o1]),
        ([e0, e0], [t0, t1], [o1, o1]), ([e1, e1], [t0, t1], [o1, o1]),
    ]
    for ex, ty, oz in edges:
        ax.plot(ex, ty, oz, color=BOX_EDGE[:3], alpha=0.85, lw=1.1)


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
    label = result["label"]

    diagnostics = []
    for sol in solutions:
        rep = check_trajectory(sol, DEFAULT_BOUNDS)
        diagnostics.append({
            "report": rep,
            "exit_idx": first_exit_index(sol, DEFAULT_BOUNDS),
        })

    n_time = min(len(solutions[0].t), args.max_frames)
    output_path = Path(args.output) if args.output else default_output_path(label, args.show_box)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.6, 6.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    E_max_axis = max(2.0, DEFAULT_BOUNDS["E_max"] * 1.10)
    T_max_axis = max(1.6, DEFAULT_BOUNDS["T_max"] * 1.10)
    O_max_axis = max(1.4, DEFAULT_BOUNDS["O_min"] * 4.5)

    ax.set_xlim(0, E_max_axis)
    ax.set_ylim(0, T_max_axis)
    ax.set_zlim(0, O_max_axis)
    ax.set_xlabel("ECM density E")
    ax.set_ylabel("Cytoskeletal tension T")
    ax.set_zlabel("Oxygen O")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.grid(True, alpha=0.20)

    if args.show_box:
        add_box(ax, DEFAULT_BOUNDS, O_max_axis)

    lines = [ax.plot([], [], [], lw=1.8, alpha=0.95, color=BLUE)[0] for _ in solutions]
    points = [ax.plot([], [], [], "o", ms=4.4, color=BLUE, zorder=6)[0] for _ in solutions]

    fig.suptitle(f"{label}\nAnimated ensemble in the 3D (E, T, O) phenotype space", fontsize=13, fontweight="bold")
    frame_text = ax.text2D(
        0.02, 0.92, "t = 0.00",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.85", alpha=0.88),
    )
    c_text = ax.text2D(
        0.98, 0.06, f"C ({args.c_stat}) = 0.000",
        transform=ax.transAxes, va="bottom", ha="right", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", alpha=0.94),
    )
    focus_idx = max(0, min(args.focus_index, len(solutions) - 1))
    exit_text = ax.text2D(
        0.98, 0.14, "",
        transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.94),
    )

    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            line.set_3d_properties([])
            line.set_color(BLUE)
            point.set_data([], [])
            point.set_3d_properties([])
            point.set_color(BLUE)
        frame_text.set_text("t = 0.00")
        c_text.set_text(f"C ({args.c_stat}) = 0.000")
        exit_text.set_text("")
        return [*lines, *points, frame_text, c_text, exit_text]

    def update(frame: int):
        t_now = solutions[0].t[frame]
        c_now = np.array([sol.y[0][frame] for sol in solutions], dtype=float)
        c_val = aggregate(c_now, args.c_stat)

        for sol, line, point, diag in zip(solutions, lines, points, diagnostics):
            E = sol.y[2][: frame + 1]
            T = sol.y[1][: frame + 1]
            O = sol.y[3][: frame + 1]
            line.set_data(E, T)
            line.set_3d_properties(O)
            point.set_data([E[-1]], [T[-1]])
            point.set_3d_properties([O[-1]])

            exit_idx = diag["exit_idx"]
            current_color = BLUE if exit_idx is None or frame < exit_idx else RED
            line.set_color(current_color)
            point.set_color(current_color)

        rep = diagnostics[focus_idx]["report"]
        exit_idx = diagnostics[focus_idx]["exit_idx"]
        if rep.viable or exit_idx is None or frame < exit_idx:
            exit_msg = f"traj {focus_idx}: viable so far"
        else:
            vars_txt = ",".join(rep.violated_vars) if rep.violated_vars else "?"
            exit_msg = f"traj {focus_idx}: exit at t={rep.first_exit_time:.2f} via {vars_txt}"

        frame_text.set_text(f"t = {t_now:.2f}")
        c_text.set_text(f"C ({args.c_stat}) = {c_val:.3f}")
        exit_text.set_text(exit_msg)
        return [*lines, *points, frame_text, c_text, exit_text]

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_time,
        interval=1000 / max(args.fps, 1),
        blit=False,
    )
    anim.save(output_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    print(f"Scenario: {label}")
    print(f"Saved GIF: {output_path}")
    for i, d in enumerate(diagnostics):
        rep = d["report"]
        status = "VIABLE" if rep.viable else f"EXIT t={rep.first_exit_time:.2f} via {','.join(rep.violated_vars)}"
        print(f"traj {i:02d}: {status}")


if __name__ == "__main__":
    main()
