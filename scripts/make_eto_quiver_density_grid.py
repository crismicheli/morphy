#!/usr/bin/env python3
"""
Generate 3D ETO quiver plots for one scenario at different arrow densities.

This script renders only the 3D ETO quiver field view (axes T, E, O) at a fixed
curvature slice C = c_slice. It does not render trajectories, taxonomy labels,
or animations.

Arrow density is controlled by the number of grid points used to sample the
field. Arrows use a single fixed color, so vector magnitude is conveyed only by
arrow length, not by a colormap.

State order in the model is always (C, T, E, O). The quiver plot shows the
projected field (dT, dE, dO) at fixed C.

Examples
--------
python scripts/make_eto_quiver_density_grid.py --filter "High porosity" --densities 5 8 12 --show-box
python scripts/make_eto_quiver_density_grid.py --filter "Intermediate porosity" --densities 6 10 14
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_BOUNDS, DEFAULT_PARAMS
from plotting.plot_helpers import add_viability_box, get_axes_limits
from plotting.scenario_helpers import choose_scenario, scenario_slug
from viabilitykernels.odes import rhs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3D ETO quiver plots for one scenario at different arrow densities."
    )
    parser.add_argument(
        "--filter",
        default="Intermediate porosity",
        help="Substring used to choose a scenario label.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output PNG files. If omitted, files are written to ROOT/figures/quiver_density.",
    )
    parser.add_argument(
        "--densities",
        type=int,
        nargs="+",
        default=[5, 8, 12],
        help="Grid densities (number of grid points per axis) used to sample the field.",
    )
    parser.add_argument(
        "--c-slice",
        type=float,
        default=0.50,
        help="Fixed curvature slice C at which the ETO field is sampled.",
    )
    parser.add_argument(
        "--quiver-length",
        type=float,
        default=0.10,
        help="Arrow length scaling for matplotlib 3D quiver.",
    )
    parser.add_argument(
        "--arrow-color",
        default="#2166ac",
        help="Fixed arrow color. No colormap is used.",
    )
    parser.add_argument(
        "--arrow-alpha",
        type=float,
        default=0.55,
        help="Arrow transparency.",
    )
    parser.add_argument(
        "--arrow-linewidth",
        type=float,
        default=0.7,
        help="Arrow linewidth.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize vectors before plotting. If omitted, magnitude is encoded by arrow length.",
    )
    parser.add_argument(
        "--show-box",
        action="store_true",
        help="Show the viability box in the 3D ETO plot.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=24.0,
        help="3D elevation angle.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-58.0,
        help="3D azimuth angle.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output PNG resolution.",
    )
    return parser.parse_args()


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def build_eto_field_samples(
    scenario_cfg: dict,
    *,
    c_slice: float,
    density: int,
    bounds: dict,
    par: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample the vector field on a regular grid in (T, E, O) at fixed C.

    Returns
    -------
    origins : ndarray, shape (n, 3)
        Sampled points in (T, E, O).
    vectors : ndarray, shape (n, 3)
        Corresponding derivatives (dT, dE, dO).
    """
    scenario_params = dict(par)
    scenario_params.update(scenario_cfg.get("param_overrides", {}))
    p = float(scenario_cfg["p"])

    tmax_axis, emax_axis, omax_axis = get_axes_limits(bounds)

    T_vals = np.linspace(0.0, tmax_axis, density)
    E_vals = np.linspace(0.0, emax_axis, density)
    O_vals = np.linspace(0.0, omax_axis, density)

    origins = []
    vectors = []

    for T in T_vals:
        for E in E_vals:
            for O in O_vals:
                x = np.array([c_slice, T, E, O], dtype=float)
                dx = np.asarray(rhs(0.0, x, p, scenario_params), dtype=float)
                dT, dE, dO = float(dx[1]), float(dx[2]), float(dx[3])
                origins.append([T, E, O])
                vectors.append([dT, dE, dO])

    origins_arr = np.asarray(origins, dtype=float)
    vectors_arr = np.asarray(vectors, dtype=float)

    keep = np.linalg.norm(vectors_arr, axis=1) > 1e-12
    return origins_arr[keep], vectors_arr[keep]


def save_eto_quiver_plot(
    scenario_cfg: dict,
    origins: np.ndarray,
    vectors: np.ndarray,
    output_path: Path,
    *,
    bounds: dict,
    c_slice: float,
    elev: float = 24.0,
    azim: float = -58.0,
    show_box: bool = False,
    quiver_length: float = 0.10,
    arrow_color: str = "#2166ac",
    arrow_alpha: float = 0.55,
    arrow_linewidth: float = 0.7,
    normalize: bool = False,
    dpi: int = 220,
) -> None:
    plot_origins = origins
    plot_vectors = normalize_vectors(vectors) if normalize else vectors

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
        ax.quiver(
            plot_origins[:, 0],
            plot_origins[:, 1],
            plot_origins[:, 2],
            plot_vectors[:, 0],
            plot_vectors[:, 1],
            plot_vectors[:, 2],
            length=quiver_length,
            normalize=normalize,
            color=arrow_color,
            alpha=arrow_alpha,
            linewidths=arrow_linewidth,
        )

    density_tag = int(round(len(origins) ** (1.0 / 3.0))) if len(origins) else 0
    magnitude_mode = "normalized direction only" if normalize else "length encodes speed"
    ax.set_title(
        f"{scenario_cfg['label']}\nETO quiver field at C={c_slice:.2f}, density={density_tag} ({magnitude_mode})",
        fontsize=12,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scenario = choose_scenario(args.filter)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    else:
        output_dir = ROOT / "figures" / "quiver_density"

    output_dir.mkdir(parents=True, exist_ok=True)

    slug = scenario_slug(scenario["label"])

    for density in args.densities:
        origins, vectors = build_eto_field_samples(
            scenario,
            c_slice=args.c_slice,
            density=density,
            bounds=DEFAULT_BOUNDS,
            par=DEFAULT_PARAMS,
        )

        output_path = output_dir / f"{slug}_eto_quiver_density_{density}.png"

        save_eto_quiver_plot(
            scenario,
            origins,
            vectors,
            output_path,
            bounds=DEFAULT_BOUNDS,
            c_slice=args.c_slice,
            elev=args.elev,
            azim=args.azim,
            show_box=args.show_box,
            quiver_length=args.quiver_length,
            arrow_color=args.arrow_color,
            arrow_alpha=args.arrow_alpha,
            arrow_linewidth=args.arrow_linewidth,
            normalize=args.normalize,
            dpi=args.dpi,
        )

        print(f"Saved density={density}: {output_path}")


if __name__ == "__main__":
    main()
