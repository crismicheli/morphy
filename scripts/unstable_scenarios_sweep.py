#!/usr/bin/env python3
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_PARAMS, DEFAULT_BOUNDS, DEFAULT_SIM, SCENARIOS
from viabilitykernels.simulation import run_scenario, sample_initial_conditions
from classifiers.taxonomy_classifier import STATE_COLORS, classify_state


BOX_GREEN = "#4dac26"
OUTSIDE_COLOR = "#d73027"
INSIDE_COLOR = "#2166ac"


def choose_unstable_scenarios() -> list[dict]:
    unstable = [s for s in SCENARIOS if s.get("expected") == "unstable"]
    if len(unstable) != 3:
        warnings.warn(
            f"Expected 3 unstable scenarios, found {len(unstable)}. Proceeding with all found.",
            stacklevel=2,
        )
    return unstable


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
        and bounds["T_min"] <= T <= bounds["T_max"]
        and bounds["E_min"] <= E <= bounds["E_max"]
        and O >= bounds["O_min"]
    )


def warn_if_any_initial_conditions_outside(initial_conditions: list[np.ndarray], bounds: dict, tag: str) -> None:
    outside = [x0 for x0 in initial_conditions if not is_inside_viability_box(x0, bounds)]
    if outside:
        warnings.warn(
            f"{tag}: {len(outside)}/{len(initial_conditions)} initial conditions start outside the viability box. "
            f"This is allowed, but please confirm that this is the intended behavior.",
            stacklevel=2,
        )


def classify_all_points(sol, bounds: dict, stride: int = 8) -> list[dict]:
    t = sol.t
    y = sol.y
    dt = max(1e-12, float(np.mean(np.diff(t))))
    dydt = np.gradient(y, dt, axis=1)

    snapshots = []
    for i in range(0, y.shape[1], stride):
        C, T, E, O = [float(v) for v in y[:, i]]
        dC, dT, dE, dO = [float(v) for v in dydt[:, i]]
        label = classify_state(C, T, E, O, dC, dT, dE, dO, bounds)
        snapshots.append(
            {
                "t": float(t[i]),
                "C": C,
                "T": T,
                "E": E,
                "O": O,
                "label": label,
                "color": STATE_COLORS[label],
            }
        )
    return snapshots


def make_regime_center(regime: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    if regime == "inside":
        center = np.array([0.22, 0.45, 0.35, 0.65], dtype=float)
        noise = (0.02, 0.03, 0.03, 0.03)
    elif regime == "inside_near_boundary":
        center = np.array([0.16, 1.40, 1.65, 0.24], dtype=float)
        noise = (0.01, 0.04, 0.05, 0.02)
    elif regime == "outside_near_boundary":
        center = np.array([0.14, 1.56, 1.86, 0.18], dtype=float)
        noise = (0.01, 0.04, 0.05, 0.02)
    else:
        raise ValueError(f"Unknown regime: {regime}")
    return center, noise


def save_taxonomy_plot(result: dict, output_path: Path, show_box: bool = True, stride: int = 8) -> None:
    solutions = result["solutions"]
    label = result["label"]

    all_points = []
    for sol in solutions:
        all_points.extend(classify_all_points(sol, bounds=DEFAULT_BOUNDS, stride=stride))

    fig = plt.figure(figsize=(9.6, 7.4), constrained_layout=True)
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
    ax.view_init(elev=24.0, azim=-58.0)
    ax.grid(True, alpha=0.25)

    if show_box:
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

    for cls, color in STATE_COLORS.items():
        pts = [p for p in all_points if p["label"] == cls]
        if not pts:
            continue
        E = [p["E"] for p in pts]
        T = [p["T"] for p in pts]
        O = [p["O"] for p in pts]
        ax.scatter(E, T, O, s=12, alpha=0.55, color=color, label=cls)

    ax.set_title(
        f"{label}\nAll trajectory points in 3D (E, T, O), colored by taxonomy state",
        fontsize=13,
        fontweight="bold",
    )

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=10,
        title="Taxonomy state",
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor("0.75")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_eto_animation(result: dict, output_path: Path, fps: int = 10, max_frames: int = 180, show_box: bool = True) -> None:
    solutions = result["solutions"]
    initial_conditions = result["initial_conditions"]
    label = result["label"]

    n_time = min(len(solutions[0].t), max_frames)

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
    ax.view_init(elev=24.0, azim=-58.0)
    ax.grid(True, alpha=0.25)

    if show_box:
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

    ax.set_title(f"{label}\nAnimated ensemble in 3D (E, T, O)", fontsize=13, fontweight="bold")

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
        interval=1000 / max(fps, 1),
        blit=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)


def main() -> None:
    unstable_scenarios = choose_unstable_scenarios()
    regimes = ["inside", "inside_near_boundary", "outside_near_boundary"]

    out_dir = ROOT / "figures" / "unstable_sweep_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_ensemble_runs = 0

    for scenario in unstable_scenarios:
        for regime in regimes:
            x0_center, noise_scale = make_regime_center(regime)

            initial_conditions = sample_initial_conditions(
                x0_center=x0_center,
                n_traj=DEFAULT_SIM["n_traj"],
                noise_scale=noise_scale,
                rng_seed=DEFAULT_SIM["rng_seed"],
            )

            tag = f"{scenario['label']} | {regime}"
            warn_if_any_initial_conditions_outside(initial_conditions, DEFAULT_BOUNDS, tag)

            result = run_scenario(
                scenario_cfg=scenario,
                par=DEFAULT_PARAMS,
                bounds=DEFAULT_BOUNDS,
                x0_center=x0_center,
                n_traj=DEFAULT_SIM["n_traj"],
                t_span=tuple(DEFAULT_SIM["t_span"]),
                n_eval=DEFAULT_SIM["n_eval"],
                rng_seed=DEFAULT_SIM["rng_seed"],
                noise_scale=noise_scale,
                initial_conditions=initial_conditions,
            )

            slug = scenario["label"].lower().replace(" ", "_").replace("-", "_")
            plot_path = out_dir / f"{slug}__{regime}__taxonomy_3d.png"
            anim_path = out_dir / f"{slug}__{regime}__eto_3d.gif"

            save_taxonomy_plot(result, plot_path, show_box=True, stride=8)
            save_eto_animation(result, anim_path, fps=10, max_frames=180, show_box=True)

            total_ensemble_runs += 1
            print(f"Done: {scenario['label']} | {regime}")

    total_trajectories = total_ensemble_runs * DEFAULT_SIM["n_traj"]
    print(f"Ensemble runs: {total_ensemble_runs}")
    print(f"Trajectory simulations: {total_trajectories}")
    print(f"Output files: {total_ensemble_runs * 2}")


if __name__ == "__main__":
    main()
