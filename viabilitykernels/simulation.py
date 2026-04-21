"""
simulation.py
-------------
High-level simulation runners: initial condition sampling, ODE integration,
and ensemble execution.

This module wraps ``scipy.integrate.solve_ivp`` with convenience functions
for generating initial condition clouds and running full scenario ensembles.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import OdeResult, solve_ivp

from viability_kernels.odes import rhs
from viability_kernels.viability import ViabilityReport, classify_ensemble


# ---------------------------------------------------------------------------
# Initial condition helpers
# ---------------------------------------------------------------------------

def sample_initial_conditions(
    x0_center: np.ndarray,
    n_traj: int,
    noise_scale: Tuple[float, float, float, float] = (0.03, 0.03, 0.03, 0.05),
    rng_seed: Optional[int] = 42,
    clip_min: float = 0.01,
) -> List[np.ndarray]:
    """Sample a cloud of initial conditions around a central point.

    Gaussian noise is added independently to each state variable.  The
    resulting conditions are clipped from below at *clip_min* to avoid
    non-physical (negative) starting states.

    Parameters
    ----------
    x0_center : array-like, shape (4,)
        Central initial condition ``[C, T, E, O]``.
    n_traj : int
        Number of initial conditions to sample.
    noise_scale : (float, float, float, float)
        Standard deviation of the Gaussian perturbation for each variable.
    rng_seed : int or None
        Seed for reproducibility.  Pass ``None`` for a random seed.
    clip_min : float
        Lower bound applied after perturbation.

    Returns
    -------
    list of ndarray, shape (4,)
        List of *n_traj* initial condition arrays.
    """
    rng = np.random.default_rng(rng_seed)
    ics = []
    for _ in range(n_traj):
        perturb = rng.normal(scale=noise_scale, size=4)
        x0 = np.clip(np.asarray(x0_center) + perturb, clip_min, None)
        ics.append(x0)
    return ics


# ---------------------------------------------------------------------------
# Single trajectory integration
# ---------------------------------------------------------------------------

def integrate_trajectory(
    x0: np.ndarray,
    p: float,
    par: dict,
    t_span: Tuple[float, float] = (0.0, 30.0),
    n_eval: int = 800,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> OdeResult:
    """Integrate one trajectory of the scaffold-cell-matrix ODE.

    Parameters
    ----------
    x0 : array-like, shape (4,)
        Initial condition ``[C, T, E, O]``.
    p : float
        Scaffold porosity.
    par : dict
        Model parameters.
    t_span : (float, float)
        Integration interval ``(t_start, t_end)``.
    n_eval : int
        Number of evenly-spaced evaluation points.
    rtol : float
        Relative tolerance for the ODE solver.
    atol : float
        Absolute tolerance for the ODE solver.

    Returns
    -------
    OdeResult
        Solution object from ``solve_ivp``.

    Raises
    ------
    RuntimeError
        If the ODE solver fails to converge.
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    sol = solve_ivp(
        rhs,
        t_span,
        x0,
        t_eval=t_eval,
        args=(p, par),
        rtol=rtol,
        atol=atol,
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError(
            f"ODE solver failed at p={p}: {sol.message}"
        )
    return sol


# ---------------------------------------------------------------------------
# Ensemble runner
# ---------------------------------------------------------------------------

def run_ensemble(
    initial_conditions: List[np.ndarray],
    p: float,
    par: dict,
    bounds: Dict[str, float],
    t_span: Tuple[float, float] = (0.0, 30.0),
    n_eval: int = 800,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> Tuple[List[OdeResult], List[ViabilityReport]]:
    """Run the full ODE system for each initial condition in an ensemble.

    Parameters
    ----------
    initial_conditions : list of ndarray
        Starting states for each trajectory.
    p : float
        Scaffold porosity (constant across the ensemble).
    par : dict
        Model parameters (constant across the ensemble).
    bounds : dict
        Viability thresholds used to classify each trajectory.
    t_span : (float, float)
        Integration interval.
    n_eval : int
        Number of output time-points.
    rtol, atol : float
        Solver tolerances.

    Returns
    -------
    solutions : list of OdeResult
        One per initial condition.
    reports : list of ViabilityReport
        Viability diagnosis for each trajectory.
    """
    solutions = [
        integrate_trajectory(x0, p, par, t_span, n_eval, rtol, atol)
        for x0 in initial_conditions
    ]
    reports = classify_ensemble(solutions, bounds)
    return solutions, reports


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(
    scenario_cfg: dict,
    par: dict,
    bounds: Dict[str, float],
    x0_center: Optional[np.ndarray] = None,
    n_traj: int = 18,
    t_span: Tuple[float, float] = (0.0, 30.0),
    n_eval: int = 800,
    rng_seed: int = 42,
) -> dict:
    """Run a named scenario defined by a configuration dictionary.

    Parameters
    ----------
    scenario_cfg : dict
        Must contain at minimum:
        ``{"label": str, "p": float}``.
        May optionally contain ``"param_overrides"`` (dict) to override
        specific entries in *par*.
    par : dict
        Base model parameters.
    bounds : dict
        Viability thresholds.
    x0_center : ndarray or None
        Central initial condition.  Defaults to ``[0.20, 0.15, 0.10, 0.60]``.
    n_traj : int
        Number of trajectories in the ensemble.
    t_span : (float, float)
        Integration time span.
    n_eval : int
        Number of output time-points.
    rng_seed : int
        Seed for initial condition sampling.

    Returns
    -------
    dict
        ``{"label", "p", "solutions", "reports", "viable_fraction"}``
    """
    if x0_center is None:
        x0_center = np.array([0.20, 0.15, 0.10, 0.60])

    # Apply any parameter overrides defined in the scenario config
    effective_par = {**par, **scenario_cfg.get("param_overrides", {})}

    ics = sample_initial_conditions(x0_center, n_traj, rng_seed=rng_seed)
    solutions, reports = run_ensemble(
        ics,
        p=scenario_cfg["p"],
        par=effective_par,
        bounds=bounds,
        t_span=t_span,
        n_eval=n_eval,
    )

    vf = sum(r.viable for r in reports) / len(reports)

    return {
        "label": scenario_cfg["label"],
        "p": scenario_cfg["p"],
        "solutions": solutions,
        "reports": reports,
        "viable_fraction": vf,
        "effective_par": effective_par,
    }
