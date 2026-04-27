from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional

import numpy as np

STATE_COLORS: Dict[str, str] = {
    "Apoptosis": "#D81B60",
    "Migration": "#1E88E5",
    "Proliferation": "#43A047",
    "Quiescence": "#FB8C00",
    "Diversification": "#8E24AA",
    "Undetermined": "#00ACC1",
}

# Biologically interpretable static parameters intentionally used by the classifier.
# We do NOT use the full parameter vector; we only keep the subset that maps cleanly
# to phenotype-like labels.
INTERPRETABLE_STATIC_PARAMS = (
    "p",         # scaffold porosity / structural context
    "beta",      # curvature-to-tension coupling
    "eta",       # ECM-mediated tension damping
    "kappa",     # ECM deposition rate
    "mu",        # ECM-mediated oxygen consumption
    "delta_T",   # intrinsic tension decay
    "delta_E",   # ECM remodelling / degradation
    "delta_O",   # baseline oxygen loss
    "rho",       # porosity-to-oxygen scaling
    "s",         # oxygen-supply scaling with porosity
)


def _merge_effective_parameters(
    par: Optional[dict] = None,
    scenario_cfg: Optional[dict] = None,
) -> Dict[str, float]:
    effective = dict(par or {})
    if scenario_cfg is not None:
        effective.update(scenario_cfg.get("param_overrides", {}))
        if "p" in scenario_cfg:
            effective["p"] = scenario_cfg["p"]
    return effective


def _extract_parameter_context(
    par: Optional[dict] = None,
    scenario_cfg: Optional[dict] = None,
) -> Dict[str, float]:
    eff = _merge_effective_parameters(par=par, scenario_cfg=scenario_cfg)

    ctx = {key: float(eff.get(key, np.nan)) for key in INTERPRETABLE_STATIC_PARAMS}

    p = 0.0 if np.isnan(ctx["p"]) else ctx["p"]
    rho = 1.0 if np.isnan(ctx["rho"]) else ctx["rho"]
    s = 1.0 if np.isnan(ctx["s"]) else ctx["s"]
    beta = 1.0 if np.isnan(ctx["beta"]) else ctx["beta"]
    eta = 1.0 if np.isnan(ctx["eta"]) else ctx["eta"]
    kappa = 1.0 if np.isnan(ctx["kappa"]) else ctx["kappa"]
    mu = 1.0 if np.isnan(ctx["mu"]) else ctx["mu"]
    delta_T = 1.0 if np.isnan(ctx["delta_T"]) else ctx["delta_T"]
    delta_E = 0.5 if np.isnan(ctx["delta_E"]) else ctx["delta_E"]
    delta_O = 0.4 if np.isnan(ctx["delta_O"]) else ctx["delta_O"]

    # Composite interpretable axes used by the heuristic rules below.
    ctx.update(
        {
            "oxygen_supply": rho * s * p,
            "tension_drive": beta,
            "tension_damping": eta,
            "matrix_drive": kappa,
            "oxygen_burden": mu,
            "decay_burden": delta_T + delta_E + delta_O,
        }
    )
    return ctx


def _near_lower(x: float, lo: float, frac: float = 0.25, abs_pad: float = 0.05) -> bool:
    return x <= max(lo * (1.0 + frac), lo + abs_pad)


def _near_upper(x: float, hi: float, frac: float = 0.10, abs_pad: float = 0.05) -> bool:
    return x >= min(hi * (1.0 - frac), hi - abs_pad)


def classify_state(
    C: float,
    T: float,
    E: float,
    O: float,
    dC: float,
    dT: float,
    dE: float,
    dO: float,
    bounds: dict,
    par: Optional[dict] = None,
    scenario_cfg: Optional[dict] = None,
) -> str:
    """Classify one instantaneous state into a coarse biological taxonomy.

    The classifier combines:
    1. current state variables (C, T, E, O),
    2. local derivatives (dC, dT, dE, dO),
    3. a deliberately small, biologically interpretable subset of static parameters.

    The static subset is explicit in INTERPRETABLE_STATIC_PARAMS and is intended to
    bias the taxonomy toward phenotype-like interpretations without making the logic
    opaque or overfit to the full ODE parameter vector.
    """
    ctx = _extract_parameter_context(par=par, scenario_cfg=scenario_cfg)

    C_min = bounds["C_min"]
    T_min = bounds["T_min"]
    T_max = bounds["T_max"]
    E_min = bounds["E_min"]
    E_max = bounds["E_max"]
    O_min = bounds["O_min"]

    near_C_low = _near_lower(C, C_min, frac=0.20, abs_pad=0.03)
    near_T_low = _near_lower(T, T_min, frac=0.25, abs_pad=0.05)
    near_T_high = _near_upper(T, T_max, frac=0.10, abs_pad=0.08)
    near_E_low = _near_lower(E, E_min, frac=0.50, abs_pad=0.05)
    near_E_high = _near_upper(E, E_max, frac=0.10, abs_pad=0.12)
    near_O_low = _near_lower(O, O_min, frac=0.25, abs_pad=0.05)

    low_oxygen_supply = ctx["oxygen_supply"] < 0.75
    high_oxygen_burden = ctx["oxygen_burden"] > 1.0
    strong_tension_drive = ctx["tension_drive"] > 2.2
    weak_tension_drive = ctx["tension_drive"] < 1.25
    strong_tension_damping = ctx["tension_damping"] > 1.0
    strong_matrix_drive = ctx["matrix_drive"] > 1.35
    high_decay_burden = ctx["decay_burden"] > 2.35

    # 1) APOPTOSIS
    # Educated guess: low oxygen, collapsing curvature, falling oxygen, or high
    # oxygen-consumption / decay burden indicates cell loss / collapse.
    if O < O_min or C < C_min:
        if dO < 0 or dC < 0 or high_oxygen_burden or low_oxygen_supply or high_decay_burden:
            return "Apoptosis"
    if near_O_low and (dO < -0.02) and (high_oxygen_burden or low_oxygen_supply):
        return "Apoptosis"
    if near_C_low and dC < -0.02 and (dE <= 0 or dT <= 0):
        return "Apoptosis"

    # 2) PROLIFERATION
    # Educated guess: viable oxygen, increasing matrix, non-collapsing tension,
    # and strong matrix drive support growth / biosynthetic expansion.
    if O > max(0.35, O_min + 0.10) and dE > 0.03:
        if strong_matrix_drive or (T > max(0.30, T_min + 0.10) and dT >= -0.02):
            return "Proliferation"
    if near_E_high and O > O_min and dE >= 0:
        return "Proliferation"

    # 3) MIGRATION
    # Educated guess: migration corresponds to viable but remodelling states with
    # matrix loosening/turnover, preserved oxygen, and non-terminal tension.
    if O > O_min and T >= T_min and dE < -0.02:
        if abs(dT) < 0.05 or (strong_tension_damping and not strong_tension_drive):
            return "Migration"
    if E > E_min and dE < 0 and dT <= 0.03 and dO >= -0.02 and not near_O_low:
        return "Migration"

    # 4) QUIESCENCE
    # Educated guess: moderate viable state with small derivatives and low drive.
    if (
        T_min <= T <= 0.75 * T_max
        and E_min <= E <= 0.70 * E_max
        and O > O_min + 0.08
        and abs(dT) < 0.03
        and abs(dE) < 0.02
        and abs(dO) < 0.03
    ):
        return "Quiescence"
    if near_T_low and weak_tension_drive and dT <= 0 and dE <= 0.02 and O > O_min:
        return "Quiescence"

    # 5) DIVERSIFICATION
    # Educated guess: this label captures non-terminal, non-homeostatic branching
    # states with mixed signals - e.g. adequate oxygen but rising tension and/or
    # ECM, suggesting directional commitment rather than stasis.
    if O > O_min and C > C_min:
        if (dT > 0.04 and dE > 0) or (near_T_high and dE >= 0) or (dC > 0.02 and dE > 0):
            return "Diversification"
    if strong_tension_drive and O > O_min and not (T > T_max):
        if dT >= 0 and dE >= -0.01:
            return "Diversification"

    # 6) UNDETERMINED
    # Fallback for mixed or ambiguous states, including near-boundary regions.
    if near_T_high or near_E_high or near_O_low or near_E_low or near_C_low:
        return "Undetermined"
    return "Undetermined"


def classify_solution(
    sol,
    bounds: dict,
    par: Optional[dict] = None,
    scenario_cfg: Optional[dict] = None,
    n_samples: int = 7,
) -> str:
    t = sol.t
    y = sol.y
    dt = max(1e-12, float(np.mean(np.diff(t))))
    dydt = np.gradient(y, dt, axis=1)

    idx = np.linspace(0, y.shape[1] - 1, num=min(n_samples, y.shape[1]), dtype=int)
    labels: List[str] = []
    for i in idx:
        C, T, E, O = [float(v) for v in y[:, i]]
        dC, dT, dE, dO = [float(v) for v in dydt[:, i]]
        labels.append(
            classify_state(
                C, T, E, O, dC, dT, dE, dO,
                bounds=bounds,
                par=par,
                scenario_cfg=scenario_cfg,
            )
        )

    if not labels:
        return "Undetermined"

    return Counter(labels).most_common(1)[0][0]


def classify_solutions(
    solutions: Iterable,
    bounds: dict,
    par: Optional[dict] = None,
    scenario_cfg: Optional[dict] = None,
    n_samples: int = 7,
) -> List[str]:
    return [
        classify_solution(sol, bounds=bounds, par=par, scenario_cfg=scenario_cfg, n_samples=n_samples)
        for sol in solutions
    ]
