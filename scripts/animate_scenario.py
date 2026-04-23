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
from matplotlib.patches import Rectangle

from config import DEFAULT_PARAMS, DEFAULT_BOUNDS, DEFAULT_SIM, SCENARIOS
from viabilitykernels.simulation import integrate_trajectory
from viabilitykernels.viability import check_trajectory
from viabilitykernels.phase_plane import ET_field, make_grid

BLUE = "#2166ac"
RED = "#d73027"
BOX_GREEN = "#4dac26"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate one simulation scenario and save a GIF.")
    parser.add_argument("--filter", default="Intermediate porosity", help="Substring used to choose a scenario label.")
    parser.add_argument("--output", default=None, help="Output GIF path. If omitted, a scenario-based name is generated.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the GIF.")
    parser.add_argument("--max-frames", type=int, default=160, help="Maximum number of animation frames.")
    parser.add_argument("--n-traj", type=int, default=DEFAULT_SIM["n_traj"], help="Number of trajectories.")
    parser.add_argument("--shift-T", type=float, default=1.0, help="Multiplier applied to the initial T center.")
    parser.add_argument("--shift-E", type=float, default=1.0, help="Multiplier applied to the initial E center.")
    parser.add_argument("--hide-box", action="store_true", help="Hide the 2D viability box overlay.")
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


def default_output_path(label: str) -> Path:
    return ROOT / "figures" / f"{slugify(label)}_animation.gif"


def sample_viable_initial_conditions(
    x0_center: np.ndarray,
    n_traj: int,
    bounds: dict,
    noise_scale=(0.03, 0.03, 0.03, 0.05),
    rng_seed: int = 42,
    clip_min: float = 0.01,
    max_tries: int = 20000,
):
    rng = np.random.default_rng(rng_seed)
    ics = []
    tries = 0
    while len(ics) < n_traj and tries < max_tries:
        tries += 1
        x0 = np.clip(np.asarray(x0_center, dtype=float) + rng.normal(scale=noise_scale, size=4), clip_min, None)
        C, T, E, O = x0
        inside = (
            (C >= bounds["C_min"]) and
            (T >= bounds["T_min"]) and (T <= bounds["T_max"]) and
            (E >= bounds["E_min"]) and (E <= bounds["E_max"]) and
            (O >= bounds["O_min"])
        )
        if inside:
            ics.append(x0)
    if len(ics) < n_traj:
        raise RuntimeError(
            f"Could only sample {len(ics)}/{n_traj} viable initial conditions; reduce noise or move the center deeper inside the viable region."
        )
    return ics


def first_exit_index(sol, bounds: dict):
    C, T, E, O = sol.y
    inside = (
        (C >= bounds["C_min"]) &
        (T >= bounds["T_min"]) & (T <= bounds["T_max"]) &
        (E >= bounds["E_min"]) & (E <= bounds["E_max"]) &
        (O >= bounds["O_min"])
    )
    bad = np.where(~inside)[0]
    return None if len(bad) == 0 else int(bad[0])


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

    scenario_params = {**DEFAULT_PARAMS, **scenario.get("param_overrides", {})}
    initial_conditions = sample_viable_initial_conditions(
        x0_center=x0_center,
        n_traj=args.n_traj,
        bounds=DEFAULT_BOUNDS,
        noise_scale=noise_scale,
        rng_seed=DEFAULT_SIM["rng_seed"],
    )
    solutions = [
        integrate_trajectory(
            x0,
            p=scenario["p"],
            par=scenario_params,
            t_span=tuple(DEFAULT_SIM["t_span"]),
            n_eval=DEFAULT_SIM["n_eval"],
        )
        for x0 in initial_conditions
    ]
    reports = [check_trajectory(sol, DEFAULT_BOUNDS) for sol in solutions]
    diagnostics = [{"report": rep, "exit_idx": first_exit_index(sol, DEFAULT_BOUNDS)} for sol, rep in zip(solutions, reports)]

    p = scenario["p"]
    label = scenario["label"]
    viable_count = sum(r.viable for r in reports)

    n_time = min(len(solutions[0].t), args.max_frames)
    output_path = Path(args.output) if args.output else default_output_path(label)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 5.8), constrained_layout=True)

    E_max_axis = max(2.0, DEFAULT_BOUNDS["E_max"] * 1.12)
    T_max_axis = max(1.6, DEFAULT_BOUNDS["T_max"] * 1.08)

    EE, TT = make_grid((0, E_max_axis), (0, T_max_axis), n_points=20)
    dE, dT = ET_field(EE, TT, p, scenario_params)
    speed = np.sqrt(dE**2 + dT**2) + 1e-9
    ax.quiver(
        EE, TT, dE / speed, dT / speed,
        angles="xy", scale_units="xy", scale=12,
        alpha=0.25, color="grey",
    )

    if not args.hide_box:
        viability_rect = Rectangle(
            (DEFAULT_BOUNDS["E_min"], DEFAULT_BOUNDS["T_min"]),
            DEFAULT_BOUNDS["E_max"] - DEFAULT_BOUNDS["E_min"],
            DEFAULT_BOUNDS["T_max"] - DEFAULT_BOUNDS["T_min"],
            facecolor=BOX_GREEN,
            alpha=0.12,
            edgecolor="none",
            zorder=0.6,
        )
        ax.add_patch(viability_rect)

    lines = [ax.plot([], [], lw=1.7, alpha=0.90, color=BLUE)[0] for _ in solutions]
    points = [ax.plot([], [], "o", ms=4.2, color=BLUE, zorder=5)[0] for _ in solutions]

    ax.set_title(f"{label}\nAnimated ensemble in the (E, T) phase plane", fontsize=12, fontweight="bold")
    ax.set_xlabel("ECM density E")
    ax.set_ylabel("Cytoskeletal tension T")
    ax.set_xlim(0, E_max_axis)
    ax.set_ylim(0, T_max_axis)
    ax.grid(alpha=0.2)

    subtitle = ax.text(
        0.02, 0.98,
        f"Viable trajectories: {viable_count}/{len(reports)} | p={p:.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.92),
    )
    frame_text = ax.text(
        0.02, 0.89, "t = 0.00",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.85", alpha=0.90),
    )
    focus_idx = max(0, min(args.focus_index, len(solutions) - 1))
    exit_text = ax.text(
        0.98, 0.04, "",
        transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.8", alpha=0.92),
    )

    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            line.set_color(BLUE)
            point.set_data([], [])
            point.set_color(BLUE)
        frame_text.set_text("t = 0.00")
        exit_text.set_text("")
        return [*lines, *points, subtitle, frame_text, exit_text]

    def update(frame: int):
        for sol, line, point, diag in zip(solutions, lines, points, diagnostics):
            E_traj = sol.y[2][: frame + 1]
            T_traj = sol.y[1][: frame + 1]
            line.set_data(E_traj, T_traj)
            point.set_data([E_traj[-1]], [T_traj[-1]])

            exit_idx = diag["exit_idx"]
            color = BLUE if exit_idx is None or frame < exit_idx else RED
            line.set_color(color)
            point.set_color(color)

        t_now = solutions[0].t[frame]
        rep = diagnostics[focus_idx]["report"]
        exit_idx = diagnostics[focus_idx]["exit_idx"]
        if rep.viable or exit_idx is None or frame < exit_idx:
            exit_msg = f"traj {focus_idx}: viable so far"
        else:
            vars_txt = ",".join(rep.violated_vars) if rep.violated_vars else "?"
            exit_msg = f"traj {focus_idx}: exit at t={rep.first_exit_time:.2f} via {vars_txt}"

        frame_text.set_text(f"t = {t_now:.2f}")
        exit_text.set_text(exit_msg)
        return [*lines, *points, subtitle, frame_text, exit_text]

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_time,
        interval=1000 / max(args.fps, 1),
        blit=True,
    )

    anim.save(output_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    print(f"Scenario: {label}")
    print(f"Saved GIF: {output_path}")
    print(f"Viable trajectories: {viable_count}/{len(reports)}")


if __name__ == "__main__":
    main()
