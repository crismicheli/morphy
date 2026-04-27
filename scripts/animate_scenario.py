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
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from config import DEFAULT_PARAMS, DEFAULT_BOUNDS, DEFAULT_SIM, SCENARIOS
from viabilitykernels.simulation import run_scenario
from viabilitykernels.phase_plane import ET_field, make_grid

INSIDE_COLOR = "#2166ac"
OUTSIDE_COLOR = "#d73027"
BOX_GREEN = "#4dac26"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate one simulation scenario and save a GIF.")
    parser.add_argument("--filter", default="Intermediate porosity", help="Substring used to choose a scenario label.")
    parser.add_argument("--output", default=str(ROOT / "figures" / "intermediate_porosity_animation.gif"), help="Output GIF path.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the GIF.")
    parser.add_argument("--max-frames", type=int, default=160, help="Maximum number of animation frames to render.")
    parser.add_argument("--n-traj", type=int, default=DEFAULT_SIM["n_traj"], help="Number of trajectories to simulate.")
    parser.add_argument("--shift-T", type=float, default=1.0, help="Multiplier applied to the initial T center.")
    parser.add_argument("--shift-E", type=float, default=1.0, help="Multiplier applied to the initial E center.")
    return parser.parse_args()


def choose_scenario(keyword: str) -> dict:
    matches = [s for s in SCENARIOS if keyword.lower() in s["label"].lower()]
    if not matches:
        labels = "\n- " + "\n- ".join(s["label"] for s in SCENARIOS)
        raise SystemExit(f"No scenario matched filter '{keyword}'. Available scenarios:{labels}")
    return matches[0]


def point_inside_et_box(E: float, T: float, bounds: dict) -> bool:
    return (
        bounds["E_min"] <= E <= bounds["E_max"]
        and bounds["T_min"] <= T <= bounds["T_max"]
    )


def segment_colors_for_solution(sol, bounds: dict) -> list[str]:
    E = sol.y[2]
    T = sol.y[1]
    colors = []
    for i in range(len(sol.t) - 1):
        inside = point_inside_et_box(float(E[i + 1]), float(T[i + 1]), bounds)
        colors.append(INSIDE_COLOR if inside else OUTSIDE_COLOR)
    return colors


def build_line_segments_2d(sol, frame: int):
    E = sol.y[2][: frame + 1]
    T = sol.y[1][: frame + 1]
    if len(E) < 2:
        return np.empty((0, 2, 2))
    points = np.column_stack([E, T])
    return np.stack([points[:-1], points[1:]], axis=1)


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

    n_time = min(len(solutions[0].t), args.max_frames)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)

    E_max_axis = 2.0
    T_max_axis = 1.6

    EE, TT = make_grid((0, E_max_axis), (0, T_max_axis), n_points=20)
    scenario_params = dict(DEFAULT_PARAMS)
    scenario_params.update(scenario.get("param_overrides", {}))
    dE, dT = ET_field(EE, TT, p, scenario_params)
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
        facecolor=BOX_GREEN,
        alpha=0.10,
        edgecolor="none",
    )
    ax.add_patch(viability_rect)

    segment_color_lists = [segment_colors_for_solution(sol, DEFAULT_BOUNDS) for sol in solutions]
    line_collections = []
    points = []
    for sol in solutions:
        E0 = float(sol.y[2][0])
        T0 = float(sol.y[1][0])
        inside0 = point_inside_et_box(E0, T0, DEFAULT_BOUNDS)
        lc = LineCollection([], linewidths=1.6, alpha=0.85)
        ax.add_collection(lc)
        point = ax.plot([], [], "o", ms=4, color=INSIDE_COLOR if inside0 else OUTSIDE_COLOR, zorder=5)[0]
        line_collections.append(lc)
        points.append(point)

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
        for lc, point in zip(line_collections, points):
            lc.set_segments([])
            lc.set_color([])
            point.set_data([], [])
        return [*line_collections, *points, subtitle]

    def update(frame: int):
        for sol, lc, point, seg_colors in zip(solutions, line_collections, points, segment_color_lists):
            segments = build_line_segments_2d(sol, frame)
            lc.set_segments(segments)
            if len(segments) > 0:
                lc.set_color(seg_colors[: len(segments)])
            E_now = float(sol.y[2][frame])
            T_now = float(sol.y[1][frame])
            point.set_data([E_now], [T_now])
            point.set_color(INSIDE_COLOR if point_inside_et_box(E_now, T_now, DEFAULT_BOUNDS) else OUTSIDE_COLOR)
        return [*line_collections, *points, subtitle]

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
