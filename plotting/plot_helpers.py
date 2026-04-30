from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

BOX_GREEN = "#4dac26"
INSIDE_COLOR = "#2166ac"
OUTSIDE_COLOR = "#d73027"


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


def get_axes_limits(bounds: Dict[str, float]) -> Tuple[float, float, float]:
    return max(2.0, bounds["Emax"] * 1.08), max(1.6, bounds["Tmax"] * 1.08), 1.4


def add_viability_box(ax, bounds: Dict[str, float], omax_axis: float) -> None:
    faces = viability_faces(
        bounds["Emin"], bounds["Emax"], bounds["Tmin"], bounds["Tmax"], bounds["Omin"], omax_axis
    )
    box = Poly3DCollection(
        faces,
        facecolors=BOX_GREEN,
        edgecolors=BOX_GREEN,
        linewidths=0.8,
        alpha=0.08,
    )
    ax.add_collection3d(box)


def point_inside_eto_box(E: float, T: float, O: float, bounds: Dict[str, float]) -> bool:
    return bounds["Emin"] <= E <= bounds["Emax"] and bounds["Tmin"] <= T <= bounds["Tmax"] and O >= bounds["Omin"]


def compute_solution_derivatives(sol) -> np.ndarray:
    dt = max(1e-12, float(np.mean(np.diff(sol.t))))
    return np.gradient(sol.y, dt, axis=1)


def classify_all_points(sol, classifier_fn: Callable, color_map: Dict[str, str], *, bounds: Dict, par: Dict, scenario_cfg: Dict, stride: int = 8, reset_fn: Callable | None = None) -> List[Dict[str, float]]:
    if reset_fn is not None:
        reset_fn()
    dydt = compute_solution_derivatives(sol)
    snapshots = []
    for i in range(0, sol.y.shape[1], stride):
        C, T, E, O = (float(v) for v in sol.y[:, i])
        dC, dT, dE, dO = (float(v) for v in dydt[:, i])
        label = classifier_fn(C, T, E, O, dC, dT, dE, dO, bounds=bounds, par=par, scenariocfg=scenario_cfg)
        snapshots.append({"t": float(sol.t[i]), "C": C, "T": T, "E": E, "O": O, "label": label, "color": color_map[label]})
    return snapshots


def segment_colors_for_solution(sol, bounds: Dict[str, float]) -> List[str]:
    E = sol.y[2]
    T = sol.y[1]
    O = sol.y[3]
    colors = []
    for i in range(len(sol.t) - 1):
        inside = point_inside_eto_box(float(E[i + 1]), float(T[i + 1]), float(O[i + 1]), bounds)
        colors.append(INSIDE_COLOR if inside else OUTSIDE_COLOR)
    return colors


def build_line_segments_3d(sol, frame: int) -> np.ndarray:
    E = sol.y[2, : frame + 1]
    T = sol.y[1, : frame + 1]
    O = sol.y[3, : frame + 1]
    if len(E) < 2:
        return np.empty((0, 2, 3))
    points = np.column_stack([E, T, O])
    return np.stack([points[:-1], points[1:]], axis=1)


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


def save_taxonomy_plot(result: Dict, scenario_cfg: Dict, output_path: Path, *, bounds: Dict, par: Dict, classifier_fn: Callable, color_map: Dict[str, str], stride: int = 8, elev: float = 24.0, azim: float = -58.0, show_box: bool = False, reset_fn: Callable | None = None) -> None:
    solutions = result["solutions"]
    label = result["label"]
    all_points = []
    for sol in solutions:
        all_points.extend(
            classify_all_points(sol, classifier_fn, color_map, bounds=bounds, par=par, scenario_cfg=scenario_cfg, stride=stride, reset_fn=reset_fn)
        )
    fig = plt.figure(figsize=(9.6, 7.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    emax_axis, tmax_axis, omax_axis = get_axes_limits(bounds)
    ax.set_xlim(0, emax_axis)
    ax.set_ylim(0, tmax_axis)
    ax.set_zlim(0, omax_axis)
    ax.set_xlabel("ECM density E")
    ax.set_ylabel("Cytoskeletal tension T")
    ax.set_zlabel("Oxygen O")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)
    if show_box:
        add_viability_box(ax, bounds, omax_axis)
    for cls, color in color_map.items():
        pts = [p for p in all_points if p["label"] == cls]
        if not pts:
            continue
        E = [p["E"] for p in pts]
        T = [p["T"] for p in pts]
        O = [p["O"] for p in pts]
        ax.scatter(E, T, O, s=12, alpha=0.55, color=color, label=cls)
    ax.set_title(f"{label} trajectory points in 3D E, T, O, colored by taxonomy state", fontsize=13, fontweight="bold")
    legend = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=10, title="Taxonomy state")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor("0.75")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_trajectory_animation(result: Dict, output_path: Path, *, bounds: Dict, fps: int = 10, max_frames: int = 160, elev: float = 24.0, azim: float = -58.0, show_box: bool = False, c_stat: str = "mean") -> None:
    solutions = result["solutions"]
    label = result["label"]
    ntime = min(len(solutions[0].t), max_frames)
    fig = plt.figure(figsize=(8.4, 6.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    emax_axis, tmax_axis, omax_axis = get_axes_limits(bounds)
    ax.set_xlim(0, emax_axis)
    ax.set_ylim(0, tmax_axis)
    ax.set_zlim(0, omax_axis)
    ax.set_xlabel("ECM density E")
    ax.set_ylabel("Cytoskeletal tension T")
    ax.set_zlabel("Oxygen O")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)
    if show_box:
        add_viability_box(ax, bounds, omax_axis)
    segment_color_lists = [segment_colors_for_solution(sol, bounds) for sol in solutions]
    line_collections = []
    points = []
    for sol in solutions:
        E0 = float(sol.y[2, 0])
        T0 = float(sol.y[1, 0])
        O0 = float(sol.y[3, 0])
        inside0 = point_inside_eto_box(E0, T0, O0, bounds)
        seed_segments = np.array([[[E0, T0, O0], [E0, T0, O0]]])
        lc = Line3DCollection(seed_segments, linewidths=1.7, alpha=0.90)
        lc.set_color(INSIDE_COLOR if inside0 else OUTSIDE_COLOR)
        ax.add_collection3d(lc)
        point, = ax.plot([E0], [T0], [O0], "o", ms=4.2, color=INSIDE_COLOR if inside0 else OUTSIDE_COLOR, zorder=6)
        line_collections.append(lc)
        points.append(point)
    fig.suptitle(f"{label} ensemble in the 3D E, T, O phenotype space", fontsize=13, fontweight="bold")
    frame_text = ax.text2D(0.02, 0.92, "t = 0.00", transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.85", alpha=0.88))
    c_text = ax.text2D(0.98, 0.06, f"C {c_stat} = 0.000", transform=ax.transAxes, va="bottom", ha="right", fontsize=10, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", alpha=0.94))

    def init():
        for sol, lc, point in zip(solutions, line_collections, points):
            E0 = float(sol.y[2, 0])
            T0 = float(sol.y[1, 0])
            O0 = float(sol.y[3, 0])
            inside0 = point_inside_eto_box(E0, T0, O0, bounds)
            seed_segments = np.array([[[E0, T0, O0], [E0, T0, O0]]])
            lc.set_segments(seed_segments)
            lc.set_color(INSIDE_COLOR if inside0 else OUTSIDE_COLOR)
            point.set_data([E0], [T0])
            point.set_3d_properties([O0])
            point.set_color(INSIDE_COLOR if inside0 else OUTSIDE_COLOR)
        frame_text.set_text("t = 0.00")
        c_text.set_text(f"C {c_stat} = 0.000")
        return [*line_collections, *points, frame_text, c_text]

    def update(frame: int):
        t_now = solutions[0].t[frame]
        c_now = np.array([sol.y[0, frame] for sol in solutions], dtype=float)
        c_val = aggregate(c_now, c_stat)
        for sol, lc, point, segcolors in zip(solutions, line_collections, points, segment_color_lists):
            segments = build_line_segments_3d(sol, frame)
            if len(segments) == 0:
                E0 = float(sol.y[2, 0])
                T0 = float(sol.y[1, 0])
                O0 = float(sol.y[3, 0])
                inside0 = point_inside_eto_box(E0, T0, O0, bounds)
                seed_segments = np.array([[[E0, T0, O0], [E0, T0, O0]]])
                lc.set_segments(seed_segments)
                lc.set_color(INSIDE_COLOR if inside0 else OUTSIDE_COLOR)
            else:
                lc.set_segments(segments)
                lc.set_color(segcolors[: len(segments)])
            E_now = float(sol.y[2, frame])
            T_now = float(sol.y[1, frame])
            O_now = float(sol.y[3, frame])
            point.set_data([E_now], [T_now])
            point.set_3d_properties([O_now])
            point.set_color(INSIDE_COLOR if point_inside_eto_box(E_now, T_now, O_now, bounds) else OUTSIDE_COLOR)
        frame_text.set_text(f"t = {t_now:.2f}")
        c_text.set_text(f"C {c_stat} = {c_val:.3f}")
        return [*line_collections, *points, frame_text, c_text]

    anim = FuncAnimation(fig, update, init_func=init, frames=ntime, interval=1000 / max(fps, 1), blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
