"""
Generate 3D ETO quiver plots for one scenario at different arrow densities.

This script focuses only on the ETO quiver field view (axes T, E, O) at a fixed
curvature slice C = c_slice. It does not render trajectories, taxonomy labels,
or animations. Instead, it samples the vector field on a 3D grid and saves
multiple quiver plots that differ only in arrow density.

Arrow density is controlled by the number of grid points used to sample the
field and, optionally, by binning in the plotting helper. This makes it easy
to visually compare sparse, medium, and dense field representations for the
same scenario.

State order is always (C, T, E, O), but the quiver plot projects the vector
field into (T, E, O) at fixed C.

Usage examples
--------------
Default scenario with three densities:
    python scripts/make_eto_quiver_density_grid.py --filter "Intermediate porosity"

Specify custom densities:
    python scripts/make_eto_quiver_density_grid.py \
        --filter "High porosity" \
        --densities 5 8 12

Show the viability box:
    python scripts/make_eto_quiver_density_grid.py \
        --filter "Enhanced guidance" \
        --show-box
"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_BOUNDS, DEFAULT_PARAMS
from plotting.plot_helpers import save_quiver_field_plot
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
        help="Directory for output PNG files. If omitted, files are written to root/figures/quiver_density.",
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
        "--quiver-mode",
        choices=["raw", "normalized", "binned"],
        default="binned",
        help="Rendering mode passed to save_quiver_field_plot.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=8,
        help="Binning level used only when quiver-mode='binned'.",
    )
    parser.add_argument(
        "--quiver-length",
        type=float,
        default=0.10,
        help="Arrow length scaling passed to save_quiver_field_plot.",
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
    return parser.parse_args()


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

    tmax_axis = max(1.6, bounds["T_max"] * 1.08)
    emax_axis = max(2.0, bounds["E_max"] * 1.08)
    omax_axis = 1.4

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

    return np.asarray(origins, dtype=float), np.asarray(vectors, dtype=float)


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
            arrow_color: str = "#2166ac",
        ) -> None:

        print(f"Saved density={density}: {output_path}")


if __name__ == "__main__":
    main()
