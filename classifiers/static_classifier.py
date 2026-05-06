from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


STATE_COLORS = {
    "Apoptosis": "#111111",
    "Migration": "#00B7FF",
    "Proliferation": "#FF7A00",
    "Quiescence": "#2ECC40",
    "Diversification": "#8E44AD",
    "Undetermined": "#FFD400",
}

INTERPRETABLE_STATIC_PARAMS = (
    "p",
    "beta",
    "eta",
    "kappa",
    "mu",
    "delta_T",
    "delta_E",
    "delta_O",
    "rho",
    "s",
)


@dataclass
class StateHeuristics:
    # near-bound fractions and absolute pads
    frac_C_low: float = 0.20
    abs_C_low_pad: float = 0.03

    frac_T_low: float = 0.25
    abs_T_low_pad: float = 0.05
    frac_T_high: float = 0.10
    abs_T_high_pad: float = 0.08

    frac_E_low: float = 0.50
    abs_E_low_pad: float = 0.05
    frac_E_high: float = 0.10
    abs_E_high_pad: float = 0.12

    frac_O_low: float = 0.25
    abs_O_low_pad: float = 0.05

    # derivative thresholds
    dO_apoptosis_cut: float = -0.02
    dC_apoptosis_cut: float = -0.02

    dE_prolif_cut: float = 0.03

    dE_migration_cut: float = -0.02
    dT_migration_abs_cut: float = 0.05

    dT_quiescence_abs_cut: float = 0.03
    dE_quiescence_abs_cut: float = 0.02
    dO_quiescence_abs_cut: float = 0.03

    dT_diversification_cut: float = 0.04
    dC_diversification_cut: float = 0.02
    dE_diversification_min: float = -0.01  # for contextual diversification branch

    # oxygen thresholds used in rules
    O_prolif_min_abs: float = 0.35
    O_prolif_min_offset: float = 0.10
    O_quiescence_min_offset: float = 0.08

    # tension thresholds used in rules
    T_prolif_min_abs: float = 0.30
    T_prolif_min_offset: float = 0.10

    # contextual gates
    oxygen_supply_low: float = 0.75
    oxygen_burden_high: float = 1.0
    tension_drive_strong: float = 2.2
    tension_drive_weak: float = 1.25
    tension_damping_strong: float = 1.0
    matrix_drive_strong: float = 1.35
    decay_burden_high: float = 2.35


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


def _near_lower(x: float, lo: float, frac: float, abs_pad: float) -> bool:
    return x <= max(lo * (1.0 + frac), lo + abs_pad)


def _near_upper(x: float, hi: float, frac: float, abs_pad: float) -> bool:
    return x >= min(hi * (1.0 - frac), hi - abs_pad)


def _compute_boundary_flags(
    C: float,
    T: float,
    E: float,
    O: float,
    bounds: dict,
    cfg: StateHeuristics,
):
    C_min = float(bounds["C_min"])
    T_min = float(bounds["T_min"])
    T_max = float(bounds["T_max"])
    E_min = float(bounds["E_min"])
    E_max = float(bounds["E_max"])
    O_min = float(bounds["O_min"])

    near_C_low = _near_lower(C, C_min, frac=cfg.frac_C_low, abs_pad=cfg.abs_C_low_pad)
    near_T_low = _near_lower(T, T_min, frac=cfg.frac_T_low, abs_pad=cfg.abs_T_low_pad)
    near_T_high = _near_upper(
        T, T_max, frac=cfg.frac_T_high, abs_pad=cfg.abs_T_high_pad
    )
    near_E_low = _near_lower(E, E_min, frac=cfg.frac_E_low, abs_pad=cfg.abs_E_low_pad)
    near_E_high = _near_upper(
        E, E_max, frac=cfg.frac_E_high, abs_pad=cfg.abs_E_high_pad
    )
    near_O_low = _near_lower(O, O_min, frac=cfg.frac_O_low, abs_pad=cfg.abs_O_low_pad)

    return (
        C_min,
        T_min,
        T_max,
        E_min,
        E_max,
        O_min,
        near_C_low,
        near_T_low,
        near_T_high,
        near_E_low,
        near_E_high,
        near_O_low,
    )


def _is_apoptosis_state(
    C: float,
    O: float,
    dC: float,
    dO: float,
    dT: float,
    dE: float,
    C_min: float,
    O_min: float,
    near_C_low: bool,
    near_O_low: bool,
    ctx: Dict[str, float],
    cfg: StateHeuristics,
) -> bool:
    low_oxygen_supply = ctx["oxygen_supply"] < cfg.oxygen_supply_low
    high_oxygen_burden = ctx["oxygen_burden"] > cfg.oxygen_burden_high
    high_decay_burden = ctx["decay_burden"] > cfg.decay_burden_high

    # original main apoptosis rule
    if (O < O_min or C < C_min) and (
        dO < 0
        or dC < 0
        or high_oxygen_burden
        or low_oxygen_supply
        or high_decay_burden
    ):
        return True

    # near-oxygen-low apoptosis rule
    if near_O_low and dO < cfg.dO_apoptosis_cut and (
        high_oxygen_burden or low_oxygen_supply
    ):
        return True

    # near-curvature-low apoptosis rule
    if near_C_low and dC < cfg.dC_apoptosis_cut and (dE <= 0 or dT <= 0):
        return True

    return False


def _is_proliferation_state(
    T: float,
    E: float,
    O: float,
    dT: float,
    dE: float,
    T_min: float,
    E_max: float,
    O_min: float,
    near_E_high: bool,
    ctx: Dict[str, float],
    cfg: StateHeuristics,
) -> bool:
    strong_matrix_drive = ctx["matrix_drive"] > cfg.matrix_drive_strong

    # main proliferation rule
    O_thresh = max(cfg.O_prolif_min_abs, O_min + cfg.O_prolif_min_offset)
    if O > O_thresh and dE > cfg.dE_prolif_cut:
        if strong_matrix_drive or (
            T > max(cfg.T_prolif_min_abs, T_min + cfg.T_prolif_min_offset)
            and dT >= -0.02
        ):
            return True

    # near-ECM-high proliferation rule
    if near_E_high and O > O_min and dE >= 0:
        return True

    return False


def _is_migration_state(
    T: float,
    E: float,
    O: float,
    dT: float,
    dE: float,
    dO: float,
    T_min: float,
    E_min: float,
    near_O_low: bool,
    ctx: Dict[str, float],
    cfg: StateHeuristics,
) -> bool:
    strong_tension_damping = ctx["tension_damping"] > cfg.tension_damping_strong
    strong_tension_drive = ctx["tension_drive"] > cfg.tension_drive_strong

    # primary migration rule
    if O > O_min and T >= T_min and dE < cfg.dE_migration_cut:
        if abs(dT) < cfg.dT_migration_abs_cut or (
            strong_tension_damping and not strong_tension_drive
        ):
            return True

    # broader ECM-loss pattern
    if (
        E > E_min
        and dE < 0
        and dT <= 0.03
        and dO >= -0.02
        and not near_O_low
    ):
        return True

    return False


def _is_quiescence_state(
    T: float,
    E: float,
    O: float,
    dT: float,
    dE: float,
    dO: float,
    T_min: float,
    T_max: float,
    E_min: float,
    E_max: float,
    O_min: float,
    near_T_low: bool,
    ctx: Dict[str, float],
    cfg: StateHeuristics,
) -> bool:
    weak_tension_drive = ctx["tension_drive"] < cfg.tension_drive_weak

    # main quiescence rule
    if (
        T_min <= T <= 0.75 * T_max
        and E_min <= E <= 0.70 * E_max
        and O > O_min + cfg.O_quiescence_min_offset
        and abs(dT) < cfg.dT_quiescence_abs_cut
        and abs(dE) < cfg.dE_quiescence_abs_cut
        and abs(dO) < cfg.dO_quiescence_abs_cut
    ):
        return True

    # weak-tension-drive alternative
    if (
        near_T_low
        and weak_tension_drive
        and dT <= 0
        and dE <= 0.02
        and O > O_min
    ):
        return True

    return False


def _is_diversification_state(
    C: float,
    T: float,
    E: float,
    O: float,
    dC: float,
    dT: float,
    dE: float,
    C_min: float,
    T_max: float,
    O_min: float,
    near_T_high: bool,
    ctx: Dict[str, float],
    cfg: StateHeuristics,
) -> bool:
    strong_tension_drive = ctx["tension_drive"] > cfg.tension_drive_strong

    if O > O_min and C > C_min:
        if (dT > cfg.dT_diversification_cut and dE > 0) or (
            near_T_high and dE >= 0
        ) or (dC > cfg.dC_diversification_cut and dE > 0):
            return True

    # contextual diversification branch
    if strong_tension_drive and O > O_min and not (T > T_max):
        if dT >= 0 and dE >= cfg.dE_diversification_min:
            return True

    return False


def _is_near_any_bound(
    near_C_low: bool,
    near_T_high: bool,
    near_E_high: bool,
    near_O_low: bool,
    near_E_low: bool,
) -> bool:
    return (
        near_T_high
        or near_E_high
        or near_O_low
        or near_E_low
        or near_C_low
    )


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
    cfg: Optional[StateHeuristics] = None,
) -> str:
    """
    Classify one instantaneous state into a coarse biological taxonomy.
    """

    if cfg is None:
        cfg = StateHeuristics()

    ctx = _extract_parameter_context(par=par, scenario_cfg=scenario_cfg)

    (
        C_min,
        T_min,
        T_max,
        E_min,
        E_max,
        O_min,
        near_C_low,
        near_T_low,
        near_T_high,
        near_E_low,
        near_E_high,
        near_O_low,
    ) = _compute_boundary_flags(C, T, E, O, bounds, cfg)

    # 1. Apoptosis (hard override near boundaries and collapse)
    if _is_apoptosis_state(
        C=C,
        O=O,
        dC=dC,
        dO=dO,
        dT=dT,
        dE=dE,
        C_min=C_min,
        O_min=O_min,
        near_C_low=near_C_low,
        near_O_low=near_O_low,
        ctx=ctx,
        cfg=cfg,
    ):
        return "Apoptosis"

    # 2. Proliferation
    if _is_proliferation_state(
        T=T,
        E=E,
        O=O,
        dT=dT,
        dE=dE,
        T_min=T_min,
        E_max=E_max,
        O_min=O_min,
        near_E_high=near_E_high,
        ctx=ctx,
        cfg=cfg,
    ):
        return "Proliferation"

    # 3. Migration
    if _is_migration_state(
        T=T,
        E=E,
        O=O,
        dT=dT,
        dE=dE,
        dO=dO,
        T_min=T_min,
        E_min=E_min,
        near_O_low=near_O_low,
        ctx=ctx,
        cfg=cfg,
    ):
        return "Migration"

    # 4. Quiescence
    if _is_quiescence_state(
        T=T,
        E=E,
        O=O,
        dT=dT,
        dE=dE,
        dO=dO,
        T_min=T_min,
        T_max=T_max,
        E_min=E_min,
        E_max=E_max,
        O_min=O_min,
        near_T_low=near_T_low,
        ctx=ctx,
        cfg=cfg,
    ):
        return "Quiescence"

    # 5. Diversification
    if _is_diversification_state(
        C=C,
        T=T,
        E=E,
        O=O,
        dC=dC,
        dT=dT,
        dE=dE,
        C_min=C_min,
        T_max=T_max,
        O_min=O_min,
        near_T_high=near_T_high,
        ctx=ctx,
        cfg=cfg,
    ):
        return "Diversification"

    # 6. Undetermined (boundary-adjacent or unmatched)
    if _is_near_any_bound(
        near_C_low=near_C_low,
        near_T_high=near_T_high,
        near_E_high=near_E_high,
        near_O_low=near_O_low,
        near_E_low=near_E_low,
    ):
        return "Undetermined"

    return "Undetermined"
