
#!/usr/bin/env python3
"""
Single-panel 3D ETO animation with optional translucent viability box.
Fixes:
- automatic output naming based on scenario label when --output is not provided
- visibly rendered box when --show-box is enabled
- clean single-panel layout with only time and C readout
"""

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

BLUE = "#2166ac"
RED = "#d73027"
BOX_GREEN = "#4dac26"


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
    parser.add_argument("--show-box", action="store_true", help="Show translucent ETO viability box.")
    parser.add_argument("--c-stat", choices=["mean", "median", "min", "max"], default="mean", help="How to aggregate C across trajectories for the numeric readout.")
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

    n_time = min(len(solutions[0].t), args.max_frames)
    output_path = Path(args.output) if args.output else default_output_path(label, args.show_box)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.4, 6.6), constrained_layout=True)
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
        faces = viability_faces(
            DEFAULT_BOUNDS["E_min"], DEFAULT_BOUNDS["E_max"],
            DEFAULT_BOUNDS["T_min"], DEFAULT_BOUNDS["T_max"],
            DEFAULT_BOUNDS["O_min"], O_max_axis,
        )
        box = Poly3DCollection(
            faces,
            facecolors=(0.301, 0.675, 0.149, 0.16),
            edgecolors=(0.301, 0.675, 0.149, 0.65),
            linewidths=1.1,
        )
        box.set_zsort('min')
        ax.add_collection3d(box)
        # reinforce visibility with explicit boundary edges
        edge_sets = [
            ([DEFAULT_BOUNDS["E_min"], DEFAULT_BOUNDS["E_max"]], [DEFAULT_BOUNDS["T_min"], DEFAULT_BOUNDS["T_min"]], [DEFAULT_BOUNDS["O_min"], DEFAULT_BOUNDS["O_min"]]),
            ([DEFAULT_BOUNDS["E_min"], DEFAULT_BOUNDS["E_max"]], [DEFAULT_BOUNDS["T_max"], DEFAULT_BOUNDS["T_max"]], [DEFAULT_BOUNDS["O_min"], DEFAULT_BOUNDS["O_min"]]),
            ([DEFAULT_BOUNDS["E_min"], DEFAULT_BOUNDS["E_min"]], [DEFAULT_BOUNDS["T_min"], DEFAULT_BOUNDS["T_max"]], [DEFAULT_BOUNDS["O_min"], DEFAULT_BOUNDS["O_min"]]),
            ([DEFAULT_BOUNDS["E_max"], DEFAULT_BOUNDS["E_max"]], [DEFAULT_BOUNDS["T_min"], DEFAULT_BOUNDS["T_max"]], [DEFAULT_BOUNDS["O_min"], DEFAULT_BOUNDS["O_min"]]),
            ([DEFAULT_BOUNDS["E_min"], DEFAULT_BOUNDS["E_min"]], [DEFAULT_BOUNDS["T_min"], DEFAULT_BOUNDS["T_min"]], [DEFAULT_BOUNDS["O_min"], O_max_axis]),
            ([DEFAULT_BOUNDS["E_max"], DEFAULT_BOUNDS["E_max"]], [DEFAULT_BOUNDS["T_min"], DEFAULT_BOUNDS["T_min"]], [DEFAULT_BOUNDS["O_min"], O_max_axis]),
            ([DEFAULT_BOUNDS["E_min"], DEFAULT_BOUNDS["E_min"]], [DEFAULT_BOUNDS["T_max"], DEFAULT_BOUNDS["T_max"]], [DEFAULT_BOUNDS["O_min"], O_max_axis]),
            ([DEFAULT_BOUNDS["E_max"], DEFAULT_BOUNDS["E_max"]], [DEFAULT_BOUNDS["T_max"], DEFAULT_BOUNDS["T_max"]], [DEFAULT_BOUNDS["O_min"], O_max_axis]),
            ([DEFAULT_BOUNDS["E_min"], DEFAULT_BOUNDS["E_max"]], [DEFAULT_BOUNDS["T_min"], DEFAULT_BOUNDS["T_min"]], [O_max_axis, O_max_axis]),
            ([DEFAULT_BOUNDS["E_min"], DEFAULT_BOUNDS["E_max"]], [DEFAULT_BOUNDS["T_max"], DEFAULT_BOUNDS["T_max"]], [O_max_axis, O_max_axis]),
            ([DEFAULT_BOUNDS["E_min"], DEFAULT_BOUNDS["E_min"]], [DEFAULT_BOUNDS["T_min"], DEFAULT_BOUNDS["T_max"]], [O_max_axis, O_max_axis]),
            ([DEFAULT_BOUNDS["E_max"], DEFAULT_BOUNDS["E_max"]], [DEFAULT_BOUNDS["T_min"], DEFAULT_BOUNDS["T_max"]], [O_max_axis, O_max_axis]),
        ]
        for ex, ty, oz in edge_sets:
            ax.plot(ex, ty, oz, color=BOX_GREEN, alpha=0.75, lw=1.0)

    colors = [BLUE if r.viable else RED for r in reports]
    lines = [ax.plot([], [], [], lw=1.7, alpha=0.90, color=c)[0] for c in colors]
    points = [ax.plot([], [], [], "o", ms=4.2, color=c, zorder=6)[0] for c in colors]

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

    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        frame_text.set_text("t = 0.00")
        c_text.set_text(f"C ({args.c_stat}) = 0.000")
        return [*lines, *points, frame_text, c_text]

    def update(frame: int):
        t_now = solutions[0].t[frame]
        c_now = np.array([sol.y[0][frame] for sol in solutions], dtype=float)
        c_val = aggregate(c_now, args.c_stat)
        for sol, line, point in zip(solutions, lines, points):
            E = sol.y[2][: frame + 1]
            T = sol.y[1][: frame + 1]
            O = sol.y[3][: frame + 1]
            line.set_data(E, T)
            line.set_3d_properties(O)
            point.set_data([E[-1]], [T[-1]])
            point.set_3d_properties([O[-1]])
        frame_text.set_text(f"t = {t_now:.2f}")
        c_text.set_text(f"C ({args.c_stat}) = {c_val:.3f}")
        return [*lines, *points, frame_text, c_text]

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


if __name__ == "__main__":
    main()
