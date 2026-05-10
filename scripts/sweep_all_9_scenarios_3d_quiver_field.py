from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
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
        bounds["T_min"],
        bounds["T_max"],
        bounds["E_min"],
        bounds["E_max"],
        bounds["O_min"],
        omax_axis,
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


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def bin_field(
    origins: np.ndarray,
    vectors: np.ndarray,
    *,
    bins: int,
    renormalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(origins) == 0:
        return origins, vectors

    bins_arr = np.array([bins, bins, bins], dtype=int)
    mins = origins.min(axis=0)
    maxs = origins.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-12)

    idx = np.floor((origins - mins) / spans * bins_arr).astype(int)
    idx = np.clip(idx, 0, bins_arr - 1)

    bucket = {}
    for p0, v0, key in zip(origins, vectors, map(tuple, idx)):
        if key not in bucket:
            bucket[key] = {"p": [], "v": []}
        bucket[key]["p"].append(p0)
        bucket[key]["v"].append(v0)

    p_out = []
    v_out = []
    for item in bucket.values():
        p_mean = np.mean(np.vstack(item["p"]), axis=0)
        v_mean = np.mean(np.vstack(item["v"]), axis=0)
        if np.linalg.norm(v_mean) > 1e-12:
            p_out.append(p_mean)
            v_out.append(v_mean)

    if not p_out:
        return np.empty((0, 3)), np.empty((0, 3))

    p_out = np.vstack(p_out)
    v_out = np.vstack(v_out)

    if renormalize:
        v_out = normalize_vectors(v_out)

    return p_out, v_out


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


def save_quiver_field_plot(
    scenario_cfg: Dict,
    origins: np.ndarray,
    vectors: np.ndarray,
    output_path: Path,
    *,
    bounds: Dict,
    c_slice: float,
    elev: float = 24.0,
    azim: float = -58.0,
    show_box: bool = False,
    quiver_mode: str = "normalized",
    quiver_length: float = 0.10,
    quiver_alpha: float = 0.45,
    quiver_linewidth: float = 0.7,
    bins: int = 8,
    binned_norm: bool = False,
    dpi: int = 220,
    cmap_name: str = "jet",
) -> None:
    keep = np.linalg.norm(vectors, axis=1) > 1e-12
    origins = origins[keep]
    vectors = vectors[keep]

    raw_speed = np.linalg.norm(vectors, axis=1)

    mode = quiver_mode.lower()
    if mode == "raw":
        plot_origins = origins
        plot_vectors = vectors
        color_values = raw_speed
    elif mode == "normalized":
        plot_origins = origins
        plot_vectors = normalize_vectors(vectors)
        color_values = raw_speed
    elif mode == "binned":
        binned_origins, binned_vectors = bin_field(
            origins,
            vectors,
            bins=bins,
            renormalize=False,
        )
        if len(binned_vectors) == 0:
            plot_origins = binned_origins
            plot_vectors = binned_vectors
            color_values = np.empty((0,), dtype=float)
        else:
            color_values = np.linalg.norm(binned_vectors, axis=1)
            plot_vectors = normalize_vectors(binned_vectors) if binned_norm else binned_vectors
            plot_origins = binned_origins
    else:
        raise ValueError(f"Unsupported quiver_mode: {quiver_mode}")

    fig = plt.figure(figsize=(8.8, 6.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    tmax_axis, emax_axis, omax_axis = get_axes_limits(bounds)
    t_upper = max(tmax_axis, float(plot_origins[:, 0].max()) * 1.05 if len(plot_origins) else tmax_axis)
    e_upper = max(emax_axis, float(plot_origins[:, 1].max()) * 1.05 if len(plot_origins) else emax_axis)
    o_upper = max(omax_axis, float(plot_origins[:, 2].max()) * 1.05 if len(plot_origins) else omax_axis)

    ax.set_xlim(0, t_upper)
    ax.set_ylim(0, e_upper)
    ax.set_zlim(0, o_upper)

    ax.set_xlabel("Cytoskeletal tension T")
    ax.set_ylabel("ECM density E")
    ax.set_zlabel("Oxygen O")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)

    if show_box:
        add_viability_box(ax, bounds, ax.get_zlim()[1])

    if len(plot_origins) > 0:
        vmin = float(np.min(color_values))
        vmax = float(np.max(color_values))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-12

        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(cmap_name)
        arrow_colors = cmap(norm(color_values))

        ax.quiver(
            plot_origins[:, 0],
            plot_origins[:, 1],
            plot_origins[:, 2],
            plot_vectors[:, 0],
            plot_vectors[:, 1],
            plot_vectors[:, 2],
            length=quiver_length,
            normalize=False,
            colors=arrow_colors,
            alpha=quiver_alpha,
            linewidth=quiver_linewidth,
        )

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.08, shrink=0.78)
        cbar.set_label("Raw speed ||(dT, dE, dO)||", rotation=90)

    ax.set_title(
        f"{scenario_cfg['label']} | 3D quiver field in T-E-O at C={c_slice:.2f} ({mode})",
        fontsize=13,
        fontweight="bold",
    )

    info = (
        f"p = {scenario_cfg['p']:.3f}\n"
        f"vectors = {len(plot_origins)}\n"
        f"colormap = {cmap_name}"
    )
    ax.text2D(
        0.98,
        0.04,
        info,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.82", alpha=0.92),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
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

    segment_color_lists = [segment_colors_for_solution(sol, bounds) for sol in solutions]

    line_collections = []
    points = []

    for sol in solutions:
        T0 = float(sol.y[1, 0])
        E0 = float(sol.y[2, 0])
        O0 = float(sol.y[3, 0])

        inside0 = point_inside_eto_box(T0, E0, O0, bounds)
        seed_segments = np.array([[[T0, E0, O0], [T0, E0, O0]]])

        lc = Line3DCollection(seed_segments, linewidths=1.7, alpha=0.90)
        lc.set_color(INSIDE_COLOR if inside0 else OUTSIDE_COLOR)
        ax.add_collection3d(lc)

        point, = ax.plot(
            [T0],
            [E0],
            [O0],
            "o",
            ms=4.2,
            color=INSIDE_COLOR if inside0 else OUTSIDE_COLOR,
            zorder=6,
        )
        line_collections.append(lc)
        points.append(point)

    fig.suptitle(
        f"{label} ensemble in the 3D T, E, O phenotype space",
        fontsize=13,
        fontweight="bold",
    )

    frame_text = ax.text2D(
        0.02,
        0.92,
        "t = 0.00",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.85", alpha=0.88),
    )
    c_text = ax.text2D(
        0.98,
        0.06,
        f"C {c_stat} = 0.000",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", alpha=0.94),
    )

    def init():
        for sol, lc, point in zip(solutions, line_collections, points):
            T0 = float(sol.y[1, 0])
            E0 = float(sol.y[2, 0])
            O0 = float(sol.y[3, 0])

            inside0 = point_inside_eto_box(T0, E0, O0, bounds)
            seed_segments = np.array([[[T0, E0, O0], [T0, E0, O0]]])

            lc.set_segments(seed_segments)
            lc.set_color(INSIDE_COLOR if inside0 else OUTSIDE_COLOR)

            point.set_data([T0], [E0])
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
                T0 = float(sol.y[1, 0])
                E0 = float(sol.y[2, 0])
                O0 = float(sol.y[3, 0])

                inside0 = point_inside_eto_box(T0, E0, O0, bounds)
                seed_segments = np.array([[[T0, E0, O0], [T0, E0, O0]]])

                lc.set_segments(seed_segments)
                lc.set_color(INSIDE_COLOR if inside0 else OUTSIDE_COLOR)
            else:
                lc.set_segments(segments)
                lc.set_color(segcolors[: len(segments)])

            T_now = float(sol.y[1, frame])
            E_now = float(sol.y[2, frame])
            O_now = float(sol.y[3, frame])

            point.set_data([T_now], [E_now])
            point.set_3d_properties([O_now])
            point.set_color(
                INSIDE_COLOR if point_inside_eto_box(T_now, E_now, O_now, bounds) else OUTSIDE_COLOR
            )

        frame_text.set_text(f"t = {t_now:.2f}")
        c_text.set_text(f"C {c_stat} = {c_val:.3f}")
        return [*line_collections, *points, frame_text, c_text]

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=ntime,
        interval=1000 / max(fps, 1),
        blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
