#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PACKAGE_PARENT = REPO_ROOT.parent

for p in (str(PACKAGE_PARENT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config import DEFAULT_PARAMS, DEFAULT_BOUNDS, DEFAULT_SIM, SCENARIOS
from viabilitykernels.simulation import run_scenario
from morphy.classifiers.taxonomy_classifier import STATE_COLORS, classify_state

BOX_GREEN = "#4dac26"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot all trajectory points in 3D (E,T,O) space colored by taxonomy state, without animation."
    )
    p.add_argument("--filter", default="Intermediate porosity", help="Substring used to choose a scenario label.")
    p.add_argument("--output", default=str(REPO_ROOT / "figures" / "taxonomy_trajectory_states_3d.png"), help="Output figure path.")
    p.add_argument("--n-traj", type=int, default=DEFAULT_SIM["n_traj"], help="Number of trajectories.")
    p.add_argument("--shift-T", type=float, default=1.0, help="Multiplier applied to initial T center.")
    p.add_argument("--shift-E", type=float, default=1.0, help="Multiplier applied to initial E center.")
    p.add_argument("--shift-O", type=float, default=1.0, help="Multiplier applied to initial O center.")
    p.add_argument("--elev", type=float, default=24.0, help="3D camera elevation.")
    p.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth.")
    p.add_argument("--show-box", action="store_true", help="Show translucent ETO viability box.")
    p.add_argument("--stride", type=int, default=8, help="Subsample factor for plotted timepoints to reduce overplotting.")
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


def classify_all_points(sol, bounds: dict, stride: int = 8) -> list[dict]:
    t = sol.t
    y = sol.y
    dt = max(1e-12, float(np.mean(np.diff(t))))
    dydt = np.gradient(y, dt, axis=1)

    step = max(1, int(stride))
    snapshots = []
    for i in range(0, y.shape[1], step):
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
                "dC": dC,
                "dT": dT,
                "dE": dE,
                "dO": dO,
                "label": label,
                "color": STATE_COLORS[label],
            }
        )
    return snapshots


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
    label = result["label"]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_points = []
    for sol in solutions:
        all_points.extend(classify_all_points(sol, bounds=DEFAULT_BOUNDS, stride=args.stride))

    counts = Counter(p["label"] for p in all_points)

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

    for cls, color in STATE_COLORS.items():
        pts = [p for p in all_points if p["label"] == cls]
        if not pts:
            continue
        E = [p["E"] for p in pts]
        T = [p["T"] for p in pts]
        O = [p["O"] for p in pts]
        ax.scatter(E, T, O, s=12, alpha=0.55, color=color, label=f"{cls} ({len(pts)})")

    legend = ax.legend(
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    frameon=True,
    fontsize=10,
    title="Taxonomy state",
    borderpad=0.8,
    labelspacing=0.6,
    handletextpad=0.6,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor("0.75")

    ax.set_title(
        f"{label}\nAll trajectory points in 3D (E, T, O), colored by taxonomy state",
        fontsize=13,
        fontweight="bold",
    )

    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=9, title="Taxonomy state")
    legend.get_frame().set_alpha(0.94)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Scenario: {label}")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
