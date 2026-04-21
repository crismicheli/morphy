"""
phase_plane.py
--------------
Reduced vector field on the (E, T) plane and plotting helpers.

The full model is 4-dimensional.  To visualise it we project the dynamics
onto the (ECM density, cytoskeletal tension) plane using quasi-steady
approximations for the two fast variables C and O (see :mod:`odes`).

This module provides:

* :func:`ET_field` — the projected 2-D vector field.
* :func:`make_grid`  — helper to create a regular meshgrid over (E, T).
* :func:`plot_phase_portrait` — draw a single-axis phase portrait panel.
* :func:`plot_all_scenarios` — multi-panel figure for a list of scenarios.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.integrate import OdeResult

from viability_kernels.odes import quasi_steady_C, quasi_steady_O
from viability_kernels.viability import ViabilityReport


# ---------------------------------------------------------------------------
# Vector field
# ---------------------------------------------------------------------------

def ET_field(
    E: np.ndarray,
    T: np.ndarray,
    p: float,
    par: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Projected vector field on the (E, T) plane.

    Slow variables E and T are evolved with C and O replaced by their
    quasi-steady approximations (see :func:`~viability_kernels.odes.quasi_steady_C`
    and :func:`~viability_kernels.odes.quasi_steady_O`).  This is purely
    a *visualisation device* — it is not a separate sub-model.

    Parameters
    ----------
    E : array-like
        ECM density values (grid or scalar).
    T : array-like
        Cytoskeletal tension values (matching shape to E).
    p : float
        Scaffold porosity.
    par : dict
        Model parameters.

    Returns
    -------
    dE : array-like
        Rate of change of ECM density on the projected plane.
    dT : array-like
        Rate of change of tension on the projected plane.
    """
    Cstar = quasi_steady_C(p, par)
    Ostar = quasi_steady_O(E, p, par)

    dT = par["beta"] * Cstar - par["delta_T"] * T - par["eta"] * E * T
    dE = par["kappa"] * T * Ostar - par["delta_E"] * E

    return dE, dT


# ---------------------------------------------------------------------------
# Grid helper
# ---------------------------------------------------------------------------

def make_grid(
    E_range: Tuple[float, float] = (0.0, 2.0),
    T_range: Tuple[float, float] = (0.0, 1.6),
    n_points: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a 2-D meshgrid over the (E, T) plane.

    Parameters
    ----------
    E_range : (float, float)
        (min, max) for the ECM axis.
    T_range : (float, float)
        (min, max) for the tension axis.
    n_points : int
        Number of sample points along each axis.

    Returns
    -------
    EE, TT : ndarray
        Coordinate arrays of shape ``(n_points, n_points)``.
    """
    E_grid = np.linspace(*E_range, n_points)
    T_grid = np.linspace(*T_range, n_points)
    return np.meshgrid(E_grid, T_grid)


# ---------------------------------------------------------------------------
# Single-panel portrait
# ---------------------------------------------------------------------------

def plot_phase_portrait(
    ax: plt.Axes,
    solutions: List[OdeResult],
    reports: List[ViabilityReport],
    p: float,
    par: dict,
    bounds: Dict[str, float],
    title: str = "",
    T_max_axis: float = 1.6,
    E_max_axis: float = 2.0,
    arrow_scale: float = 12,
    show_quiver: bool = True,
    color_viable: str = "#2166ac",
    color_nonviable: str = "#d73027",
) -> None:
    """Draw one phase-portrait panel on *ax*.

    Each trajectory is plotted in the (E, T) plane.  Viable trajectories
    use *color_viable* at full opacity; non-viable ones use *color_nonviable*
    at reduced opacity.  A shaded rectangle indicates the projected viability
    region.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw on.
    solutions : list of OdeResult
        Simulated trajectories for this scenario.
    reports : list of ViabilityReport
        Viability reports (same order as *solutions*).
    p : float
        Porosity used for this scenario (needed for the vector field).
    par : dict
        Model parameters.
    bounds : dict
        Viability thresholds.
    title : str
        Panel title.
    T_max_axis : float
        Upper limit of the tension axis.
    E_max_axis : float
        Upper limit of the ECM axis.
    arrow_scale : float
        Passed to ``quiver`` as the scale parameter (larger = shorter arrows).
    show_quiver : bool
        Toggle the background vector field.
    color_viable : str
        Hex colour for viable trajectories.
    color_nonviable : str
        Hex colour for non-viable trajectories.
    """
    if show_quiver:
        EE, TT = make_grid((0, E_max_axis), (0, T_max_axis))
        dE, dT = ET_field(EE, TT, p, par)
        speed = np.sqrt(dE**2 + dT**2) + 1e-9
        ax.quiver(
            EE, TT,
            dE / speed, dT / speed,
            angles="xy",
            scale_units="xy",
            scale=arrow_scale,
            alpha=0.30,
            color="grey",
        )

    # Projected viability window
    ax.axvspan(
        bounds["E_min"], bounds["E_max"],
        ymin=bounds["T_min"] / T_max_axis,
        ymax=bounds["T_max"] / T_max_axis,
        alpha=0.10,
        color="#4dac26",
        label="Viability region",
    )

    for sol, report in zip(solutions, reports):
        E_traj = sol.y[2]
        T_traj = sol.y[1]

        if report.viable:
            ax.plot(E_traj, T_traj, lw=1.4, alpha=0.85, color=color_viable)
        else:
            ax.plot(E_traj, T_traj, lw=1.2, alpha=0.45, color=color_nonviable)

        # Mark initial condition
        ax.plot(E_traj[0], T_traj[0], "o", ms=2.5, color="black", zorder=5)

    # Legend patches
    viable_patch = mpatches.Patch(color=color_viable, label="Viable")
    nonviable_patch = mpatches.Patch(color=color_nonviable, label="Non-viable")
    ax.legend(handles=[viable_patch, nonviable_patch], fontsize=7, loc="upper right")

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("ECM density  E", fontsize=9)
    ax.set_xlim(0, E_max_axis)
    ax.set_ylim(0, T_max_axis)


# ---------------------------------------------------------------------------
# Multi-panel figure
# ---------------------------------------------------------------------------

def plot_all_scenarios(
    scenario_results: List[dict],
    par: dict,
    bounds: Dict[str, float],
    suptitle: str = "Viability kernel simulations",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Produce a multi-panel phase-portrait figure for several scenarios.

    Parameters
    ----------
    scenario_results : list of dict
        Each entry must have keys:
        ``{"label": str, "p": float, "solutions": list, "reports": list}``.
    par : dict
        Model parameters (shared across all scenarios in this call).
    bounds : dict
        Viability thresholds.
    suptitle : str
        Figure-level super-title.
    figsize : (float, float) or None
        If None a sensible default is computed from the number of panels.
    save_path : str or None
        If provided the figure is saved to this path before returning.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(scenario_results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    if figsize is None:
        figsize = (5.0 * cols, 4.5 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    for i, result in enumerate(scenario_results):
        ax = axes_flat[i]
        plot_phase_portrait(
            ax=ax,
            solutions=result["solutions"],
            reports=result["reports"],
            p=result["p"],
            par=par,
            bounds=bounds,
            title=result["label"],
        )
        axes_flat[i].set_ylabel("Cytoskeletal tension  T", fontsize=9)

    # Hide any unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
