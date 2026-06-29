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
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_BOUNDS, DEFAULT_PARAMS, DEFAULT_SIM
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
        help="Base directory for output PNG files. Scenario subfolder is created inside it.",
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
        "--show-attractor",
        action="store_true",
        help="Overlay the attractor as a black star.",
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
    parser.add_argument(
        "--attractor-time",
        type=float,
        default=float(DEFAULT_SIM["t_span"][1]),
        help="Integration horizon used to estimate the attractor point.",
    )
    parser.add_argument(
        "--attractor-n-eval",
        type=int,
        default=int(DEFAULT_SIM["n_eval"]),
        help="Number of time samples used when integrating toward the attractor.",
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


def estimate_attractor_eto(
    scenario_cfg: dict,
    *,
    par: dict,
    t_final: float,
    n_eval: int,
) -> np.ndarray:
    scenario_params = dict(par)
    scenario_params.update(scenario_cfg.get("param_overrides", {}))
    p = float(scenario_cfg["p"])

    x0 = np.asarray(DEFAULT_SIM["x0_center"], dtype=float)
    t_eval = np.linspace(0.0, t_final, n_eval)

    sol = solve_ivp(
        lambda t, x: rhs(t, x, p, scenario_params),
        (0.0, t_final),
        x0,
        t_eval=t_eval,
        rtol=float(DEFAULT_SIM["rtol"]),
        atol=float(DEFAULT_SIM["atol"]),
    )

    if not sol.success:
        raise RuntimeError(f"Attractor integration failed: {sol.message}")

    return np.asarray(sol.y[1:4, -1], dtype=float)


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
    attractor_eto: np.ndarray | None = None,
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

    if attractor_eto is not None:
        t_upper = max(t_upper, float(attractor_eto[0]) * 1.05)
        e_upper = max(e_upper, float(attractor_eto[1]) * 1.05)
        o_upper = max(o_upper, float(attractor_eto[2]) * 1.05)

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

    if attractor_eto is not None:
        ax.scatter(
            [attractor_eto[0]],
            [attractor_eto[1]],
            [attractor_eto[2]],
            color="black",
            marker="*",
            s=220,
            depthshade=False,
            label="Attractor",
            zorder=10,
        )
        ax.legend(loc="upper right", frameon=True)

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
    slug = scenario_slug(scenario["label"])

    if args.output_dir:
        base_output_dir = Path(args.output_dir)
        if not base_output_dir.is_absolute():
            base_output_dir = ROOT / base_output_dir
    else:
        base_output_dir = ROOT / "figures" / "quiver_density"

    scenario_output_dir = base_output_dir / slug
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    attractor_eto = None
    if args.show_attractor:
        attractor_eto = estimate_attractor_eto(
            scenario,
            par=DEFAULT_PARAMS,
            t_final=args.attractor_time,
            n_eval=args.attractor_n_eval,
        )

    for density in args.densities:
        origins, vectors = build_eto_field_samples(
            scenario,
            c_slice=args.c_slice,
            density=density,
            bounds=DEFAULT_BOUNDS,
            par=DEFAULT_PARAMS,
        )

        output_path = scenario_output_dir / f"{slug}_eto_quiver_density_{density}.png"

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
            attractor_eto=attractor_eto,
            dpi=args.dpi,
        )

        print(f"Saved density={density}: {output_path}")


if __name__ == "__main__":
    main()
