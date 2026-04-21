"""
viability.py
------------
Viability kernel utilities: threshold checks, classification, and
trajectory-level diagnostics.

A trajectory is *viable* if, for every time-point in the solution, every
state variable remains within the biologically meaningful range defined
in a viability dictionary.  Trajectories that leave this region are
classified as *non-viable* (or *escaping*).

The viability dictionary always follows this schema::

    {
        "C_min": float,    # minimum curvature
        "T_min": float,    # minimum cytoskeletal tension
        "T_max": float,    # maximum cytoskeletal tension
        "E_min": float,    # minimum ECM density
        "E_max": float,    # maximum ECM density
        "O_min": float,    # minimum oxygen availability
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.integrate import OdeResult


# ---------------------------------------------------------------------------
# Data class for a trajectory's viability report
# ---------------------------------------------------------------------------

@dataclass
class ViabilityReport:
    """Summary of a single trajectory's viability status.

    Attributes
    ----------
    viable : bool
        True when all time-points satisfy all constraints.
    first_exit_time : float or None
        Simulation time at which the trajectory first leaves the kernel;
        None if the trajectory remains viable throughout.
    violated_vars : list of str
        Names of state variables that violated a bound at some point.
    fraction_viable : float
        Proportion of sampled time-points inside the viable region.
    """

    viable: bool
    first_exit_time: Optional[float] = None
    violated_vars: List[str] = field(default_factory=list)
    fraction_viable: float = 1.0

    def __repr__(self) -> str:  # pragma: no cover
        status = "VIABLE" if self.viable else f"EXITS at t={self.first_exit_time:.2f}"
        return (
            f"ViabilityReport({status}, "
            f"fraction_viable={self.fraction_viable:.2%}, "
            f"violated={self.violated_vars})"
        )


# ---------------------------------------------------------------------------
# Core check function
# ---------------------------------------------------------------------------

def check_trajectory(sol: OdeResult, bounds: Dict[str, float]) -> ViabilityReport:
    """Evaluate whether a simulated trajectory remains inside the viability kernel.

    Parameters
    ----------
    sol : OdeResult
        Output of ``scipy.integrate.solve_ivp``.  Must have ``sol.y`` of
        shape ``(4, n_timepoints)`` with rows ``[C, T, E, O]``.
    bounds : dict
        Viability thresholds.  Required keys:
        ``C_min, T_min, T_max, E_min, E_max, O_min``.

    Returns
    -------
    ViabilityReport
        Full diagnostic report for the trajectory.

    Examples
    --------
    >>> from viability_kernels.viability import check_trajectory
    >>> report = check_trajectory(sol, bounds)
    >>> print(report.viable, report.first_exit_time)
    """
    C, T, E, O = sol.y
    t = sol.t

    # Build boolean mask: True means the point is inside the kernel
    masks = {
        "C": C >= bounds["C_min"],
        "T_lo": T >= bounds["T_min"],
        "T_hi": T <= bounds["T_max"],
        "E_lo": E >= bounds["E_min"],
        "E_hi": E <= bounds["E_max"],
        "O": O >= bounds["O_min"],
    }

    # Map internal mask keys to human-readable variable names
    var_labels = {
        "C": "C",
        "T_lo": "T",
        "T_hi": "T",
        "E_lo": "E",
        "E_hi": "E",
        "O": "O",
    }

    combined_mask = np.ones(len(t), dtype=bool)
    for mask in masks.values():
        combined_mask &= mask

    fraction_viable = combined_mask.mean()
    violated_vars = sorted(
        {var_labels[k] for k, m in masks.items() if not m.all()}
    )

    if combined_mask.all():
        return ViabilityReport(
            viable=True,
            fraction_viable=float(fraction_viable),
        )

    # Find the first time-point where the trajectory exits
    first_exit_idx = int(np.argmin(combined_mask))
    first_exit_time = float(t[first_exit_idx])

    return ViabilityReport(
        viable=False,
        first_exit_time=first_exit_time,
        violated_vars=violated_vars,
        fraction_viable=float(fraction_viable),
    )


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def classify_ensemble(
    solutions: List[OdeResult],
    bounds: Dict[str, float],
) -> List[ViabilityReport]:
    """Apply :func:`check_trajectory` to every solution in a list.

    Parameters
    ----------
    solutions : list of OdeResult
        All trajectories from a simulation ensemble.
    bounds : dict
        Viability thresholds (same format as :func:`check_trajectory`).

    Returns
    -------
    list of ViabilityReport
        One report per trajectory, in the same order as *solutions*.
    """
    return [check_trajectory(sol, bounds) for sol in solutions]


def viable_fraction(reports: List[ViabilityReport]) -> float:
    """Return the fraction of trajectories that are fully viable.

    Parameters
    ----------
    reports : list of ViabilityReport

    Returns
    -------
    float
        Value in [0, 1].
    """
    if not reports:
        return 0.0
    return sum(r.viable for r in reports) / len(reports)
