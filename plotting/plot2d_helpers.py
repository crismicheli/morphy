from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from viabilitykernels.phase_plane import ETfield, makegrid, plotallscenarios

INSIDE_COLOR = "#2166ac"
OUTSIDE_COLOR = "#d73027"
BOX_GREEN = "#4dac26"


def point_inside_et_box(E: float, T: float, bounds: Dict[str, float]) -> bool:
    return bounds["Emin"] <= E <= bounds["Emax"] and bounds["Tmin"] <= T <= bounds["Tmax"]


def build_line_segments_2d(sol, frame: int) -> np.ndarray:
    E = sol.y[2, : frame + 1]
    T = sol.y[1, : frame + 1]
    if len(E) < 2:
        return np.empty((0, 2, 2))
    points = np.column_stack([E, T])
    return np.stack([points[:-1], points[1:]], axis=1)


def segment_colors_for_solution_2d(sol, bounds: Dict[str, float]) -> list[str]:
    E = sol.y[2]
    T = sol.y[1]
    return [INSIDE_COLOR if point_inside_et_box(float(E[i + 1]), float(T[i + 1]), bounds) else OUTSIDE_COLOR for i in range(len(sol.t) - 1)]


def add_et_background(ax, *, bounds: Dict[str, float], par: Dict, scenario_cfg: Dict, p: float, emax_axis: float = 2.0, tmax_axis: float = 1.6, show_box: bool = True, grid_points: int = 20) -> None:
    EE, TT = makegrid((0, emax_axis), (0, tmax_axis), npoints=grid_points)
    scenario_params = dict(par)
    scenario_params.update(scenario_cfg.get("param_overrides", {}))
    dE, dT = ETfield(EE, TT, p, scenario_params)
    speed = np.sqrt(dE**2 + dT**2) + 1e-9
    ax.quiver(EE, TT, dE / speed, dT / speed, angles="xy", scale_units="xy", scale=12, alpha=0.25, color="grey")
    if show_box:
        rect = Rectangle((bounds["Emin"], bounds["Tmin"]), bounds["Emax"] - bounds["Emin"], bounds["Tmax"] - bounds["Tmin"], facecolor=BOX_GREEN, alpha=0.10, edgecolor="none")
        ax.add_patch(rect)
    ax.set_xlabel("ECM density E")
    ax.set_ylabel("Cytoskeletal tension T")
    ax.set_xlim(0, emax_axis)
    ax.set_ylim(0, tmax_axis)
    ax.grid(alpha=0.2)


def save_et_animation(result: Dict, scenario_cfg: Dict, output_path: Path, *, bounds: Dict, par: Dict, fps: int = 10, max_frames: int = 160, show_box: bool = True, emax_axis: float = 2.0, tmax_axis: float = 1.6) -> None:
    solutions = result["solutions"]
    reports = result["reports"]
    p = result["p"]
    label = result["label"]
    n_time = min(len(solutions[0].t), max_frames)
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    add_et_background(ax, bounds=bounds, par=par, scenario_cfg=scenario_cfg, p=p, emax_axis=emax_axis, tmax_axis=tmax_axis, show_box=show_box)
    segment_color_lists = [segment_colors_for_solution_2d(sol, bounds) for sol in solutions]
    line_collections = []
    points = []
    for sol in solutions:
        E0 = float(sol.y[2, 0])
        T0 = float(sol.y[1, 0])
        inside0 = point_inside_et_box(E0, T0, bounds)
        lc = LineCollection([], linewidths=1.6, alpha=0.85)
        ax.add_collection(lc)
        point = ax.plot([], [], "o", ms=4, color=INSIDE_COLOR if inside0 else OUTSIDE_COLOR, zorder=5)[0]
        line_collections.append(lc)
        points.append(point)
    ax.set_title(f"{label}
Animated ensemble in the (E, T) phase plane", fontsize=12, fontweight="bold")
    viable_count = sum(r.viable for r in reports)
    subtitle = ax.text(0.02, 0.98, f"Viable trajectories: {viable_count}/{len(reports)} | p={p:.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))

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
            E_now = float(sol.y[2, frame])
            T_now = float(sol.y[1, frame])
            point.set_data([E_now], [T_now])
            point.set_color(INSIDE_COLOR if point_inside_et_box(E_now, T_now, bounds) else OUTSIDE_COLOR)
        return [*line_collections, *points, subtitle]

    anim = FuncAnimation(fig, update, init_func=init, frames=n_time, interval=1000 / max(fps, 1), blit=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def print_summary_table(results: Sequence[Dict]) -> None:
    header = f"{'Scenario':<50} {'p':>6} {'Viable':>8} {'Expected':>12}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in results:
        vf = r["viable_fraction"]
        pct = f"{vf:.0%}"
        expected = r.get("expected", "—")
        print(f" {r['label']:<48} {r['p']:>6.2f} {pct:>8} {expected:>12}")
    print(sep)
    print()


def save_all_scenarios_figure(results: Sequence[Dict], output_path: str | Path | None, *, par: Dict, bounds: Dict, suptitle: str) -> object:
    save_path = None if output_path is None else str(output_path)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    return plotallscenarios(scenarioresults=list(results), par=par, bounds=bounds, suptitle=suptitle, savepath=save_path)
