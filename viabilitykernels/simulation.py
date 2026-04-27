from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scipy.integrate._ivp.ivp import OdeResult

from viabilitykernels.odes import rhs
from viabilitykernels.viability import ViabilityReport, classify_ensemble


def sample_initial_conditions(
    x0_center: np.ndarray,
    n_traj: int,
    noise_scale: Tuple[float, float, float, float] = (0.03, 0.03, 0.03, 0.05),
    rng_seed: Optional[int] = 42,
    clip_min: float = 0.01,
) -> List[np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    ics = []
    for _ in range(n_traj):
        perturb = rng.normal(scale=noise_scale, size=4)
        x0 = np.clip(np.asarray(x0_center) + perturb, clip_min, None)
        ics.append(x0)
    return ics


def integrate_trajectory(
    x0: np.ndarray,
    p: float,
    par: dict,
    t_span: Tuple[float, float] = (0.0, 30.0),
    n_eval: int = 800,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> OdeResult:
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
        raise RuntimeError(f"ODE solver failed at p={p}: {sol.message}")
    return sol


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
    solutions = [
        integrate_trajectory(x0, p, par, t_span, n_eval, rtol, atol)
        for x0 in initial_conditions
    ]
    reports = classify_ensemble(solutions, bounds)
    return solutions, reports


def run_scenario(
    scenario_cfg: dict,
    par: dict,
    bounds: Dict[str, float],
    x0_center: Optional[np.ndarray] = None,
    n_traj: int = 18,
    t_span: Tuple[float, float] = (0.0, 30.0),
    n_eval: int = 800,
    rng_seed: int = 42,
    noise_scale: Tuple[float, float, float, float] = (0.03, 0.03, 0.03, 0.05),
    initial_conditions: Optional[List[np.ndarray]] = None,
) -> dict:
    if x0_center is None:
        x0_center = np.array([0.20, 0.15, 0.10, 0.60])

    effective_par = {**par, **scenario_cfg.get("param_overrides", {})}

    if initial_conditions is None:
        ics = sample_initial_conditions(
            x0_center,
            n_traj,
            noise_scale=noise_scale,
            rng_seed=rng_seed,
        )
    else:
        ics = [np.asarray(x0, dtype=float) for x0 in initial_conditions]

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
        "initial_conditions": ics,
    }
