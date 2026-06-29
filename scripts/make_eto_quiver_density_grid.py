#!/usr/bin/env python3
"""
Generate 3D ETO quiver-field plots for one selected morphodynamic scenario,
sweeping over multiple arrow densities while keeping the same underlying
dynamical system, camera view, and fixed curvature slice.

This script is intentionally specialized for the ETO projection only:

- Full model state order is always (C, T, E, O).
- The rendered 3D plot uses axes (T, E, O).
- The field is sampled at a fixed curvature slice C = --c-slice.
- Arrows are drawn in a single fixed color, so local speed is encoded only
  by vector length unless --normalize is requested.
- A black star can mark an estimated attractor in ETO space.
- A sparse set of seeded trajectories can be overlaid independently of the
  quiver density, so the figure remains readable for both sparse and dense
  arrow fields.

Compared with the more general quiver utilities in the repository, this script
is aimed at one practical task: generate a small family of static ETO quiver
plots for the same scenario at different sampling densities, optionally with
clean trajectory overlays.

Typical usage
-------------
1) High-porosity scenario, three densities, viability box, attractor marker:
   python scripts/make_eto_quiver_density_grid.py \
       --filter "High porosity" \
       --densities 5 8 12 \
       --show-box \
       --show-attractor

2) Same scenario, but also overlay sparse seeded trajectories:
   python scripts/make_eto_quiver_density_grid.py \
       --filter "High porosity" \
       --densities 5 8 12 \
       --show-box \
       --show-attractor \
       --show-trajectories

3) Intermediate porosity with slightly longer arrows:
   python scripts/make_eto_quiver_density_grid.py \
       --filter "Intermediate porosity" \
       --densities 6 10 14 \
       --quiver-length 0.14 \
       --show-attractor

4) Enhanced-guidance scenario, normalized direction field only:
   python scripts/make_eto_quiver_density_grid.py \
       --filter "Enhanced guidance" \
       --densities 7 11 \
       --normalize \
       --show-box

5) Hypoxic environment written into a custom output root:
   python scripts/make_eto_quiver_density_grid.py \
       --filter "Hypoxic environment" \
       --densities 5 9 13 \
       --output-dir figures/custom_quiver_runs \
       --show-attractor

What gets saved
---------------
Outputs are written into a compact scenario-specific folder such as:

    figures/high_porosity_eto_quiver/

or, if --output-dir is provided:

    <output-dir>/high_porosity_eto_quiver/

The saved files follow the pattern:

    high_porosity_eto_quiver_density_5.png
    high_porosity_eto_quiver_density_8.png
    high_porosity_eto_quiver_density_12.png

Notes
-----
- The ODE field is evaluated through viabilitykernels.odes.rhs.
- Scenario-specific parameter overrides are merged onto DEFAULT_PARAMS.
- The attractor marker is estimated by integrating from
  DEFAULT_SIM["x0_center"] and projecting the terminal point to (T, E, O).
- The seeded trajectories are deliberately sparse and are not tied to the
  quiver density, so they keep a consistent visual weight across all outputs.
- This script does not use a speed colormap by design.
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_BOUNDS, DEFAULT_PARAMS, DEFAULT_SIM
from plotting.plot_helpers import add_viability_box, get_axes_limits
from plotting.scenario_helpers import choose_scenario
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
        help=(
            "Base directory for output PNG files. If omitted, outputs are written "
            "under ROOT/figures/<minimal-scenario-name>_eto_quiver/."
        ),
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
        "--show-trajectories",
        action="store_true",
        help="Overlay a sparse set of seeded trajectories in ETO space.",
    )
    parser.add_argument(
        "--traj-time",
        type=float,
        default=20.0,
        help="Integration horizon for seeded trajectories.",
    )
    parser.add_argument(
        "--traj-n-eval",
        type=int,
        default=300,
        help="Number of time samples for each seeded trajectory.",
    )
    parser.add_argument(
        "--traj-color",
        default="black",
        help="Seeded trajectory color.",
    )
    parser.add_argument(
        "--traj-alpha",
        type=float,
        default=0.35,
        help="Seeded trajectory transparency.",
    )
    parser.add_argument(
        "--traj-linewidth",
        type=float,
        default=1.0,
        help="Seeded trajectory linewidth.",
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


def minimal_scenario_name(label: str) -> str:
    """
    Convert a verbose scenario label into a compact, filesystem-friendly stem.

    Examples
    --------
    "High porosity  (p=0.75) — borderline" -> "high_porosity"
    "Hypoxic environment  (ρ=0.3, s=0.4) — unstable" -> "hypoxic_environment"
    "Enhanced guidance  (a=6.0) — stable" -> "enhanced_guidance"
    "Near-critical asymmetric regime (β=2.6, η=0.7, δ_E=0.9, a=6.2, ρ=0.85, s=0.9) — borderline"
        -> "near_critical_asymmetric_regime"
    """
    head = label.split("(")[0]
    head = head.split("—")[0]
    head = head.strip().lower()

    head = unicodedata.normalize("NFKD", head)
    head = head.encode("ascii", "ignore").decode("ascii")

    head = head.replace("-", "_")
    head = re.sub(r"[^a-z0-9_ ]+", "", head)
    head = re.sub(r"\s+", "_", head)
    head = re.sub(r"_+", "_", head).strip("_")

    return head or "scenario"


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

    Parameters
    ----------
    scenario_cfg : dict
        Scenario dictionary selected from the configured scenario catalogue.
    c_slice : float
        Fixed curvature value used to define the ETO slice.
    density : int
        Number of sample points per axis for T, E, and O.
    bounds : dict
        Default viability bounds used to derive axis extents.
    par : dict
        Baseline parameter dictionary, before applying scenario overrides.

    Returns
    -------
    origins : ndarray, shape (n, 3)
        Sampled points in ETO coordinates (T, E, O).
    vectors : ndarray, shape (n, 3)
        Corresponding projected derivatives (dT, dE, dO).
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


def build_sparse_trajectory_seeds(
    *,
    c_slice: float,
    bounds: dict,
) -> np.ndarray:
    """
    Build a small, density-independent seed set in full state space (C, T, E, O).

    The seed cloud is intentionally sparse so that trajectory overlays remain
    readable regardless of the quiver density.
    """
    tmax_axis, emax_axis, omax_axis = get_axes_limits(bounds)

    T_vals = np.array([0.20, 0.55, 0.90]) * tmax_axis
    E_vals = np.array([0.20, 0.55, 0.90]) * emax_axis
    O_vals = np.array([0.20, 0.75]) * omax_axis

    seeds = []
    for T in T_vals:
        for E in E_vals:
            for O in O_vals:
                seeds.append([c_slice, T, E, O])

    return np.asarray(seeds, dtype=float)


def integrate_seeded_trajectories(
    scenario_cfg: dict,
    *,
    seeds_4d: np.ndarray,
    par: dict,
    t_final: float,
    n_eval: int,
) -> list[np.ndarray]:
    """
    Integrate full 4D trajectories from sparse seeds and return ETO projections.

    Each returned curve has shape (3, n_eval_like), corresponding to (T, E, O).
    """
    scenario_params = dict(par)
    scenario_params.update(scenario_cfg.get("param_overrides", {}))
    p = float(scenario_cfg["p"])

    t_eval = np.linspace(0.0, t_final, n_eval)
    curves: list[np.ndarray] = []

    for x0 in seeds_4d:
        try:
            sol = solve_ivp(
                lambda t, x: rhs(t, x, p, scenario_params),
                (0.0, t_final),
                np.asarray(x0, dtype=float),
                t_eval=t_eval,
                rtol=float(DEFAULT_SIM["rtol"]),
                atol=float(DEFAULT_SIM["atol"]),
            )
        except ValueError:
            continue

        if sol.success and sol.y.shape[0] == 4:
            curves.append(np.asarray(sol.y[1:4, :], dtype=float))

    return curves


def estimate_attractor_eto(
    scenario_cfg: dict,
    *,
    par: dict,
    t_final: float,
    n_eval: int,
) -> np.ndarray:
    """
    Estimate an attractor-like terminal point in ETO space by integrating the
    selected scenario from the repository's default initial center.

    The full model state is (C, T, E, O), but the returned point is the
    projected terminal coordinate (T, E, O).
    """
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
    trajectory_curves: list[np.ndarray] | None = None,
    traj_color: str = "black",
    traj_alpha: float = 0.35,
    traj_linewidth: float = 1.0,
    dpi: int = 220,
) -> None:
    """
    Save one static 3D ETO quiver plot for a given scenario and density.

    If normalize=False, vector length conveys local speed.
    If normalize=True, all arrows are scaled uniformly and show direction only.
    """
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

    if trajectory_curves:
        for curve in trajectory_curves:
            if curve.size:
                t_upper = max(t_upper, float(np.max(curve[0, :])) * 1.05)
                e_upper = max(e_upper, float(np.max(curve[1, :])) * 1.05)
                o_upper = max(o_upper, float(np.max(curve[2, :])) * 1.05)

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

    if trajectory_curves:
        for curve in trajectory_curves:
            ax.plot(
                curve[0, :],
                curve[1, :],
                curve[2, :],
                color=traj_color,
                alpha=traj_alpha,
                linewidth=traj_linewidth,
                zorder=6,
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
    scenario_name = minimal_scenario_name(scenario["label"])

    if args.output_dir:
        base_output_dir = Path(args.output_dir)
        if not base_output_dir.is_absolute():
            base_output_dir = ROOT / base_output_dir
        scenario_output_dir = base_output_dir / f"{scenario_name}_eto_quiver"
    else:
        scenario_output_dir = ROOT / "figures" / f"{scenario_name}_eto_quiver"

    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    attractor_eto = None
    if args.show_attractor:
        attractor_eto = estimate_attractor_eto(
            scenario,
            par=DEFAULT_PARAMS,
            t_final=args.attractor_time,
            n_eval=args.attractor_n_eval,
        )

    trajectory_curves = None
    if args.show_trajectories:
        seeds_4d = build_sparse_trajectory_seeds(
            c_slice=args.c_slice,
            bounds=DEFAULT_BOUNDS,
        )
        trajectory_curves = integrate_seeded_trajectories(
            scenario,
            seeds_4d=seeds_4d,
            par=DEFAULT_PARAMS,
            t_final=args.traj_time,
            n_eval=args.traj_n_eval,
        )

    for density in args.densities:
        origins, vectors = build_eto_field_samples(
            scenario,
            c_slice=args.c_slice,
            density=density,
            bounds=DEFAULT_BOUNDS,
            par=DEFAULT_PARAMS,
        )

        output_path = scenario_output_dir / f"{scenario_name}_eto_quiver_density_{density}.png"

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
            trajectory_curves=trajectory_curves,
            traj_color=args.traj_color,
            traj_alpha=args.traj_alpha,
            traj_linewidth=args.traj_linewidth,
            dpi=args.dpi,
        )

        print(f"Saved density={density}: {output_path}")


if __name__ == "__main__":
    main()
