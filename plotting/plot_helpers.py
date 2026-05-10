from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

BOX_GREEN = "#4dac26"
INSIDE_COLOR = "#2166ac"
OUTSIDE_COLOR = "#d73027"


def viability_faces(t0: float, t1: float, e0: float, e1: float, o0: float, o1: float):
    v000 = [t0, e0, o0]
    v100 = [t1, e0, o0]
    v110 = [t1, e1, o0]
    v010 = [t0, e1, o0]
    v001 = [t0, e0, o1]
    v101 = [t1, e0, o1]
    v111 = [t1, e1, o1]
    v011 = [t0, e1, o1]
    return [
        [v000, v100, v110, v010],
        [v001, v101, v111, v011],
        [v000, v100, v101, v001],
        [v010, v110, v111, v011],
        [v000, v010, v011, v001],
        [v100, v110, v111, v101],
    ]


def get_axes_limits(bounds: Dict[str, float]) -> Tuple[float, float, float]:
    return max(1.6, bounds["T_max"] * 1.08), max(2.0, bounds["E_max"] * 1.08), 1.4


def add_viability_box(ax, bounds: Dict[str, float], omax_axis: float) -> None:
    faces = viability_faces(
        bounds["T_min"], bounds["T_max"],
        bounds["E_min"], bounds["E_max"],
        bounds["O_min"], omax_axis,
    )
    box = Poly3DCollection(
        faces,
        facecolors=BOX_GREEN,
        edgecolors=BOX_GREEN,
        linewidths=0.8,
        alpha=0.08,
    )
    ax.add_collection3d(box)


def point_inside_eto_box(T: float, E: float, O: float, bounds: Dict[str, float]) -> bool:
    return (
        bounds["T_min"] <= T <= bounds["T_max"]
        and bounds["E_min"] <= E <= bounds["E_max"]
        and O >= bounds["O_min"]
    )


def compute_solution_derivatives(sol) -> np.ndarray:
    dt = max(1e-12, float(np.mean(np.diff(sol.t))))
    return np.gradient(sol.y, dt, axis=1)


def classify_all_points(
    sol,
    classifier_fn: Callable,
    color_map: Dict[str, str],
    *,
    bounds: Dict,
    par: Dict,
    scenario_cfg: Dict,
    stride: int = 8,
    reset_fn: Callable | None = None,
) -> List[Dict[str, float]]:
    if reset_fn is not None:
        reset_fn()
    dydt = compute_solution_derivatives(sol)
    snapshots = []
    for i in range(0, sol.y.shape[1], stride):
        C, T, E, O = (float(v) for v in sol.y[:, i])
        dC, dT, dE, dO = (float(v) for v in dydt[:, i])
        label = classifier_fn(C, T, E, O, dC, dT, dE, dO, bounds=bounds, par=par, scenario_cfg=scenario_cfg)
        snapshots.append(
            {
                "t": float(sol.t[i]),
                "C": C,
                "T": T,
                "E": E,
                "O": O,
                "label": label,
                "color": color_map[label],
            }
        )
    return snapshots


def segment_colors_for_solution(sol, bounds: Dict[str, float]) -> List[str]:
    T = sol.y[1]
    E = sol.y[2]
    O = sol.y[3]
    colors = []
    for i in range(len(sol.t) - 1):
        inside = point_inside_eto_box(float(T[i + 1]), float(E[i + 1]), float(O[i + 1]), bounds)
        colors.append(INSIDE_COLOR if inside else OUTSIDE_COLOR)
    return colors


def build_line_segments_3d(sol, frame: int) -> np.ndarray:
    T = sol.y[1, : frame + 1]
    E = sol.y[2, : frame + 1]
    O = sol.y[3, : frame + 1]
    if len(T) < 2:
        return np.empty((0, 2, 3))
    points = np.column_stack([T, E, O])
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


def _trajectory_points_3d(sol) -> np.ndarray:
    return np.column_stack([sol.y[1], sol.y[2], sol.y[3]])


def _collect_quiver_samples(
    solutions,
    *,
    step: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    origins_all = []
    vectors_all = []

    step = max(1, int(step))

    for sol in solutions:
        pts = _trajectory_points_3d(sol)
        if pts.shape[0] < 2:
            continue

        origins = pts[:-1:step]
        ends = pts[1::step]
        n = min(len(origins), len(ends))
        if n == 0:
            continue

        origins = origins[:n]
        vectors = ends[:n] - origins

        keep = np.linalg.norm(vectors, axis=1) > 1e-12
        if np.any(keep):
            origins_all.append(origins[keep])
            vectors_all.append(vectors[keep])

    if not origins_all:
        return np.empty((0, 3)), np.empty((0, 3))

    return np.vstack(origins_all), np.vstack(vectors_all)


def _normalize_quiver_vectors(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def _bin_quiver_field(
    origins: np.ndarray,
    vectors: np.ndarray,
    *,
    bins: int | Tuple[int, int, int] = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(origins) == 0:
        return origins, vectors

    if isinstance(bins, int):
        bins = (bins, bins, bins)

    bins_arr = np.asarray(bins, dtype=int)
    mins = origins.min(axis=0)
    maxs = origins.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-12)

    idx = np.floor((origins - mins) / spans * bins_arr).astype(int)
    idx = np.clip(idx, 0, bins_arr - 1)

    bucket: Dict[Tuple[int, int, int], Dict[str, List[np.ndarray]]] = {}
    for p, v, key in zip(origins, vectors, map(tuple, idx)):
        if key not in bucket:
            bucket[key] = {"p": [], "v": []}
        bucket[key]["p"].append(p)
        bucket[key]["v"].append(v)

    out_origins = []
    out_vectors = []
    for item in bucket.values():
        p = np.mean(np.vstack(item["p"]), axis=0)
        v = np.mean(np.vstack(item["v"]), axis=0)
        if np.linalg.norm(v) > 1e-12:
            out_origins.append(p)
            out_vectors.append(v)

    if not out_origins:
        return np.empty((0, 3)), np.empty((0, 3))

    return np.vstack(out_origins), np.vstack(out_vectors)


def add_quiver_overlay(
    ax,
    solutions,
    *,
    mode: str = "normalized",
    step: int = 5,
    bins: int | Tuple[int, int, int] = 10,
    length: float = 0.10,
    color: str = "0.25",
    alpha: float = 0.35,
    linewidth: float = 0.6,
):
    origins, vectors = _collect_quiver_samples(solutions, step=step)
    if len(origins) == 0:
        return None

    mode = mode.lower()

    if mode == "raw":
        plot_vectors = vectors
    elif mode == "normalized":
        plot_vectors = _normalize_quiver_vectors(vectors)
    elif mode == "binned":
        norm_vectors = _normalize_quiver_vectors(vectors)
        origins, plot_vectors = _bin_quiver_field(origins, norm_vectors, bins=bins)
        plot_vectors = _normalize_quiver_vectors(plot_vectors)
    else:
        raise ValueError(f"Unsupported quiver mode: {mode}")

    if len(origins) == 0:
        return None

    return ax.quiver(
        origins[:, 0],
        origins[:, 1],
        origins[:, 2],
        plot_vectors[:, 0],
        plot_vectors[:, 1],
        plot_vectors[:, 2],
        length=length,
        normalize=False,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
    )


def save_taxonomy_plot(
    result: Dict,
    scenario_cfg: Dict,
    output_path: Path,
    *,
    bounds: Dict,
    par: Dict,
    classifier_fn: Callable,
    color_map: Dict[str, str],
    stride: int = 8,
    elev: float = 24.0,
    azim: float = -58.0,
    show_box: bool = False,
    reset_fn: Callable | None = None,
) -> None:
    solutions = result["solutions"]
    label = result["label"]

    all_points = []
    for sol in solutions:
        all_points.extend(
            classify_all_points(
                sol,
                classifier_fn,
                color_map,
                bounds=bounds,
                par=par,
                scenario_cfg=scenario_cfg,
                stride=stride,
                reset_fn=reset_fn,
            )
        )

    fig = plt.figure(figsize=(9.6, 7.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    tmax_axis, emax_axis, omax_axis = get_axes_limits(bounds)
    ax.set_xlim(0, tmax_axis)
    ax.set_ylim(0, emax_axis)
    ax.set_zlim(0, omax_axis)

    ax.set_xlabel("Cytoskeletal tension T")
    ax.set_ylabel("ECM density E")
    ax.set_zlabel("Oxygen O")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)

    if show_box:
        add_viability_box(ax, bounds, omax_axis)

    for cls, color in color_map.items():
        pts = [p for p in all_points if p["label"] == cls]
        if not pts:
            continue
        T = [p["T"] for p in pts]
        E = [p["E"] for p in pts]
        O = [p["O"] for p in pts]
        ax.scatter(T, E, O, s=12, alpha=0.55, color=color, label=cls)

    ax.set_title(
        f"{label} trajectory points in 3D T, E, O, colored by taxonomy state",
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


def save_trajectory_animation(
    result: Dict,
    output_path: Path,
    *,
    bounds: Dict,
    fps: int = 10,
    max_frames: int = 160,
    elev: float = 24.0,
    azim: float = -58.0,
    show_box: bool = False,
    c_stat: str = "mean",
    show_quiver: bool = False,
    quiver_mode: str = "normalized",
    quiver_step: int = 5,
    quiver_bins: int | Tuple[int, int, int] = 10,
    quiver_length: float = 0.10,
    quiver_color: str = "0.25",
    quiver_alpha: float = 0.35,
    quiver_linewidth: float = 0.6,
) -> None:
    solutions = result["solutions"]
    label = result["label"]
    ntime = min(len(solutions[0].t), max_frames)

    fig = plt.figure(figsize=(8.4, 6.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    tmax_axis, emax_axis, omax_axis = get_axes_limits(bounds)
    ax.set_xlim(0, tmax_axis)
    ax.set_ylim(0, emax_axis)
    ax.set_zlim(0, omax_axis)

    ax.set_xlabel("Cytoskeletal tension T")
    ax.set_ylabel("ECM density E")
    ax.set_zlabel("Oxygen O")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)

    if show_box:
        add_viability_box(ax, bounds, omax_axis)

    quiver_artist = None
    if show_quiver:
        quiver_artist = add_quiver_overlay(
            ax,
            solutions,
            mode=quiver_mode,
            step=quiver_step,
            bins=quiver_bins,
            length=quiver_length,
            color=quiver_color,
            alpha=quiver_alpha,
            linewidth=quiver_linewidth,
        )

    segment_color_lists = [segment_colors_for_solution(sol, bounds) for sol in solutions]

    line_collections = []
    points = []

    for sol in solutions:
        T0 = float(sol.y[1, 0])
        E0 = float(sol.y[2, 0])
        O0 = float(sol.y[3, 0])

        inside0 = point_inside_eto_box(T0, E0, O0, bounds)
        seed_segments 
