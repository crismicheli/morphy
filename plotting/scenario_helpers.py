from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Tuple

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
    return bounds["Cmin"] <= C and bounds["Tmin"] <= T <= bounds["Tmax"] and bounds["Emin"] <= E <= bounds["Emax"] and O >= bounds["Omin"]


def warn_if_any_initial_conditions_outside(initial_conditions, bounds: Dict) -> None:
    outside = [x0 for x0 in initial_conditions if not is_inside_viability_box(x0, bounds)]
    if outside:
        warnings.warn(
            f"{len(outside)}/{len(initial_conditions)} initial conditions start outside the viability box. This is allowed, but please confirm that this is the intended behavior.",
            stacklevel=2,
        )


def compute_initial_conditions(scenario: Dict, *, n_traj: int, shift_T: float = 1.0, shift_E: float = 1.0, shift_O: float = 1.0) -> Tuple[np.ndarray, Tuple[float, float, float, float], list]:
    x0_center = np.array(DEFAULTSIM["x0_center"], dtype=float)
    noise_scale = (0.03, 0.03, 0.03, 0.05)
    expected = scenario.get("expected", "")
    if expected in {"borderline", "boundary"}:
        x0_center[1] = 1.5
        x0_center[2] = 1.7
        noise_scale = (0.04, 0.08, 0.08, 0.06)
    elif expected == "unstable":
        x0_center[1] = 1.7
        x0_center[2] = 1.9
        noise_scale = (0.05, 0.10, 0.10, 0.07)
    x0_center[1] *= shift_T
    x0_center[2] *= shift_E
    x0_center[3] *= shift_O
    initial_conditions = sampleinitialconditions(
        x0center=x0_center,
        ntraj=n_traj,
        noisescale=noise_scale,
        rngseed=DEFAULTSIM["rng_seed"],
    )
    return x0_center, noise_scale, initial_conditions


def run_single_scenario(scenario: Dict, *, n_traj: int, shift_T: float = 1.0, shift_E: float = 1.0, shift_O: float = 1.0):
    x0_center, noise_scale, initial_conditions = compute_initial_conditions(
        scenario, n_traj=n_traj, shift_T=shift_T, shift_E=shift_E, shift_O=shift_O
    )
    warn_if_any_initial_conditions_outside(initial_conditions, DEFAULTBOUNDS)
    result = runscenario(
        scenariocfg=scenario,
        par=DEFAULTPARAMS,
        bounds=DEFAULTBOUNDS,
        x0center=x0_center,
        ntraj=n_traj,
        tspan=tuple(DEFAULTSIM["t_span"]),
        neval=DEFAULTSIM["n_eval"],
        rngseed=DEFAULTSIM["rng_seed"],
        noisescale=noise_scale,
        initialconditions=initial_conditions,
    )
    return result
