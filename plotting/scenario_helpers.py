from __future__ import annotations

import warnings
from typing import Dict

import numpy as np

from config import DEFAULT_BOUNDS, DEFAULT_PARAMS, DEFAULT_SIM, SCENARIOS
from viabilitykernels.simulation import run_scenario, sample_initial_conditions


def choose_scenario(keyword: str) -> Dict:
    matches = [s for s in SCENARIOS if keyword.lower() in s["label"].lower()]
    if not matches:
        labels = ", ".join(s["label"] for s in SCENARIOS)
        raise SystemExit(f"No scenario matched filter '{keyword}'. Available scenarios: {labels}")
    return matches[0]


def scenario_slug(label: str) -> str:
    return label.lower().replace(" ", "_").replace("-", "_")


def is_inside_viability_box(x0: np.ndarray, bounds: Dict) -> bool:
    C, T, E, O = (float(v) for v in x0)
    return (
        bounds["C_min"] <= C
        and bounds["T_min"] <= T <= bounds["T_max"]
        and bounds["E_min"] <= E <= bounds["E_max"]
        and O >= bounds["O_min"]
    )


def warn_if_any_initial_conditions_outside(initial_conditions, bounds: Dict) -> None:
    outside = [x0 for x0 in initial_conditions if not is_inside_viability_box(x0, bounds)]
    if outside:
        warnings.warn(
            f"{len(outside)}/{len(initial_conditions)} initial conditions start outside the viability box. "
            "This is allowed, but please confirm that this is the intended behavior.",
            stacklevel=2,
        )


def get_base_x0_center(scenario: Dict) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Return the default/reference center and noise scale for this scenario.

    This remains useful as a fallback and as metadata for plotting/reporting,
    but it is no longer used as the mandatory basis for shift-based input.
    """
    x0_center = np.array(DEFAULT_SIM["x0_center"], dtype=float)
    noise_scale = (0.03, 0.03, 0.03, 0.05)
    return x0_center, noise_scale


def validate_x0_center(x0_center) -> np.ndarray:
    """
    Validate and normalize a user-provided 4D initial point.

    Expected state order is (C, T, E, O).
    """
    arr = np.asarray(x0_center, dtype=float)
    if arr.shape != (4,):
        raise ValueError(
            f"x0_center must be a 4D point in (C, T, E, O) order; got shape {arr.shape}"
        )
    return arr


def resolve_x0_center(
    scenario: Dict,
    *,
    x0_center=None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """
    Resolve the effective center used for sampling trajectories.

    Returns:
        base_x0_center: the default/reference point from config
        resolved_x0_center: the actual center used for this run
        noise_scale: sampling noise scale
    """
    base_x0_center, noise_scale = get_base_x0_center(scenario)

    if x0_center is None:
        resolved_x0_center = np.array(base_x0_center, dtype=float, copy=True)
    else:
        resolved_x0_center = validate_x0_center(x0_center)

    return base_x0_center, resolved_x0_center, noise_scale


def compute_initial_conditions(
    scenario: Dict,
    *,
    n_traj: int,
    x0_center=None,
):
    """
    Build the actual initial-condition cloud from an explicit 4D center point.

    If x0_center is None, fall back to the default/reference center from config.
    """
    base_x0_center, resolved_x0_center, noise_scale = resolve_x0_center(
        scenario,
        x0_center=x0_center,
    )

    initial_conditions = sample_initial_conditions(
        x0_center=resolved_x0_center,
        n_traj=n_traj,
        noise_scale=noise_scale,
        rng_seed=DEFAULT_SIM["rng_seed"],
    )

    return base_x0_center, resolved_x0_center, noise_scale, initial_conditions


def run_single_scenario(
    scenario: Dict,
    *,
    n_traj: int,
    x0_center=None,
):
    """
    Run one scenario from a direct 4D initial center point.

    If x0_center is None, fall back to the default/reference center from config.
    """
    base_x0_center, resolved_x0_center, noise_scale, initial_conditions = compute_initial_conditions(
        scenario,
        n_traj=n_traj,
        x0_center=x0_center,
    )

    warn_if_any_initial_conditions_outside(initial_conditions, DEFAULT_BOUNDS)

    result = run_scenario(
        scenario_cfg=scenario,
        par=DEFAULT_PARAMS,
        bounds=DEFAULT_BOUNDS,
        x0_center=resolved_x0_center,
        n_traj=n_traj,
        t_span=tuple(DEFAULT_SIM["t_span"]),
        n_eval=DEFAULT_SIM["n_eval"],
        rng_seed=DEFAULT_SIM["rng_seed"],
        noise_scale=noise_scale,
        initial_conditions=initial_conditions,
    )

    result["base_x0_center"] = base_x0_center
    result["x0_center"] = resolved_x0_center
    result["noise_scale"] = noise_scale
    return result
