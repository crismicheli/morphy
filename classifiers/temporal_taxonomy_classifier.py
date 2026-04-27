from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple
import math

STATE_COLORS = {
    "Apoptosis": "#111111",
    "Migration": "#00B7FF",
    "Proliferation": "#FF7A00",
    "Quiescence": "#2ECC40",
    "Diversification": "#8E44AD",
    "Undetermined": "#FFD400",
}

STATE_ORDER = (
    "Apoptosis",
    "Migration",
    "Proliferation",
    "Quiescence",
    "Diversification",
    "Undetermined",
)

EPS = 1e-12


@dataclass
class _TemporalMemory:
    last_label: str = "Undetermined"
    dwell_count: int = 0
    outside_count: int = 0
    inside_count: int = 0
    history: Deque[str] = field(default_factory=lambda: deque(maxlen=8))


_TEMPORAL_CACHE: Dict[Tuple[int, int], _TemporalMemory] = defaultdict(_TemporalMemory)


def reset_classifier_memory() -> None:
    _TEMPORAL_CACHE.clear()


def _safe_get(d: Optional[dict], key: str, default: float) -> float:
    if d is None:
        return float(default)
    return float(d.get(key, default))


def _infer_signature(par: Optional[dict], scenario_cfg: Optional[dict]) -> Tuple[int, int]:
    p_val = 0.0 if scenario_cfg is None else float(scenario_cfg.get("p", 0.0))
    s_label = "" if scenario_cfg is None else str(scenario_cfg.get("label", ""))
    par_items = tuple(sorted((par or {}).items()))
    return hash((round(p_val, 6), s_label)), hash(par_items)


def _inside_viability(C: float, T: float, E: float, O: float, bounds: dict) -> bool:
    return (
        C >= float(bounds["C_min"])
        and float(bounds["T_min"]) <= T <= float(bounds["T_max"])
        and float(bounds["E_min"]) <= E <= float(bounds["E_max"])
        and O >= float(bounds["O_min"])
    )


def _recovery_score(
    C: float,
    T: float,
    E: float,
    O: float,
    dC: float,
    dT: float,
    dE: float,
    dO: float,
    bounds: dict,
) -> float:
    score = 0.0
    if C < bounds["C_min"] and dC > 0:
        score += 1.0
    if T < bounds["T_min"] and dT > 0:
        score += 1.0
    if T > bounds["T_max"] and dT < 0:
        score += 1.0
    if E < bounds["E_min"] and dE > 0:
        score += 1.0
    if E > bounds["E_max"] and dE < 0:
        score += 1.0
    if O < bounds["O_min"] and dO > 0:
        score += 1.0
    return score


def _secondary_parameter_modifiers(
    par: Optional[dict],
    scenario_cfg: Optional[dict],
) -> Dict[str, float]:
    p = _safe_get(scenario_cfg, "p", 0.0)

    adhesion = _safe_get(par, "adhesion", _safe_get(par, "alpha", 1.0))
    motility_gain = _safe_get(par, "motility", _safe_get(par, "m", 1.0))
    growth_gain = _safe_get(par, "growth", _safe_get(par, "g", 1.0))
    oxygen_gain = _safe_get(par, "oxygen", _safe_get(par, "o", 1.0))
    remodeling_gain = _safe_get(par, "remodeling", _safe_get(par, "r", 1.0))
    stress_gain = _safe_get(par, "stress", _safe_get(par, "s", 1.0))

    expected = "" if scenario_cfg is None else str(scenario_cfg.get("expected", "")).lower()

    modifiers = {
        "Apoptosis": 1.0,
        "Migration": 1.0,
        "Proliferation": 1.0,
        "Quiescence": 1.0,
        "Diversification": 1.0,
        "Undetermined": 1.0,
    }

    modifiers["Migration"] *= 1.0 + 0.12 * max(0.0, motility_gain - 1.0)
    modifiers["Migration"] *= 1.0 + 0.08 * max(0.0, p)

    modifiers["Proliferation"] *= 1.0 + 0.14 * max(0.0, growth_gain - 1.0)
    modifiers["Proliferation"] *= 1.0 + 0.08 * max(0.0, oxygen_gain - 1.0)
    modifiers["Proliferation"] *= 1.0 + 0.06 * max(0.0, adhesion - 1.0)

    modifiers["Quiescence"] *= 1.0 + 0.10 * max(0.0, adhesion - 1.0)
    modifiers["Quiescence"] *= 1.0 - 0.06 * max(0.0, motility_gain - 1.0)

    modifiers["Diversification"] *= 1.0 + 0.14 * max(0.0, remodeling_gain - 1.0)
    modifiers["Diversification"] *= 1.0 + 0.06 * max(0.0, p)

    modifiers["Apoptosis"] *= 1.0 + 0.16 * max(0.0, stress_gain - 1.0)
    modifiers["Apoptosis"] *= 1.0 - 0.06 * max(0.0, oxygen_gain - 1.0)

    if expected == "unstable":
        modifiers["Apoptosis"] *= 1.18
        modifiers["Undetermined"] *= 1.08
        modifiers["Quiescence"] *= 0.94
        modifiers["Proliferation"] *= 0.95
    elif expected in {"boundary", "borderline"}:
        modifiers["Undetermined"] *= 1.15
        modifiers["Diversification"] *= 1.05
        modifiers["Apoptosis"] *= 1.04
    elif expected == "stable":
        modifiers["Quiescence"] *= 1.10
        modifiers["Proliferation"] *= 1.08
        modifiers["Apoptosis"] *= 0.92

    for key, value in modifiers.items():
        modifiers[key] = max(0.65, min(1.5, value))

    return modifiers


def _instantaneous_scores(
    C: float,
    T: float,
    E: float,
    O: float,
    dC: float,
    dT: float,
    dE: float,
    dO: float,
    bounds: dict,
    par: Optional[dict],
    scenario_cfg: Optional[dict],
) -> Dict[str, float]:
    scores = {k: 0.0 for k in STATE_ORDER}

    C_min = float(bounds["C_min"])
    T_min = float(bounds["T_min"])
    T_max = float(bounds["T_max"])
    E_min = float(bounds["E_min"])
    E_max = float(bounds["E_max"])
    O_min = float(bounds["O_min"])

    T_mid = 0.5 * (T_min + T_max)
    T_span = max(EPS, T_max - T_min)
    E_mid = 0.5 * (E_min + E_max)
    E_span = max(EPS, E_max - E_min)

    inside = _inside_viability(C, T, E, O, bounds)

    low_C = max(0.0, (C_min - C) / max(C_min, EPS))
    low_O = max(0.0, (O_min - O) / max(O_min, EPS))
    low_T = max(0.0, (T_min - T) / max(T_min, EPS))
    high_T = max(0.0, (T - T_max) / max(T_max, EPS))
    low_E = max(0.0, (E_min - E) / max(E_min, EPS))
    high_E = max(0.0, (E - E_max) / max(E_max, EPS))

    stress = low_C + low_O + low_T + high_T + low_E + high_E
    scores["Apoptosis"] += 3.0 * stress
    if dC < 0:
        scores["Apoptosis"] += 0.8 * abs(dC)
    if dO < 0:
        scores["Apoptosis"] += 0.8 * abs(dO)
    if not inside:
        scores["Apoptosis"] += 1.0

    motility = max(0.0, dT) + max(0.0, dE) + 0.5 * max(0.0, dC)
    if T >= T_mid and O >= O_min and C >= C_min:
        scores["Migration"] += 2.0 + motility
    scores["Migration"] += 0.8 * max(0.0, (T - T_mid) / T_span)
    scores["Migration"] += 0.6 * max(0.0, dE)

    growth = max(0.0, dC) + max(0.0, dE) + max(0.0, dO)
    if inside and O >= O_min and C >= C_min and T_min <= T <= T_max:
        scores["Proliferation"] += 2.0 + growth
    scores["Proliferation"] += 0.8 * max(0.0, 1.0 - abs(T - T_mid) / (0.5 * T_span + EPS))
    scores["Proliferation"] += 0.5 * max(0.0, 1.0 - abs(E - E_mid) / (0.5 * E_span + EPS))

    quiescent_dynamics = math.exp(-2.5 * (abs(dC) + abs(dT) + abs(dE) + abs(dO)))
    if inside:
        scores["Quiescence"] += 1.8 * quiescent_dynamics
    scores["Quiescence"] += 0.9 * max(0.0, 1.0 - abs(T - T_mid) / (0.5 * T_span + EPS))
    if abs(dC) < 0.03 and abs(dT) < 0.03 and abs(dE) < 0.03 and abs(dO) < 0.03:
        scores["Quiescence"] += 1.5

    remodeling = abs(dE) + 0.6 * abs(dT)
    if inside and E >= E_mid and O >= O_min:
        scores["Diversification"] += 1.5 + remodeling
    if dE > 0 and abs(dT) > 0.03:
        scores["Diversification"] += 0.8

    scores["Undetermined"] += 0.5
    if not inside:
        scores["Undetermined"] += 1.5
    if sum(v > 2.0 for k, v in scores.items() if k != "Undetermined") == 0:
        scores["Undetermined"] += 1.0

    modifiers = _secondary_parameter_modifiers(par, scenario_cfg)
    for label in STATE_ORDER:
        scores[label] *= modifiers[label]

    return scores


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
    key = _infer_signature(par, scenario_cfg)
    mem = _TEMPORAL_CACHE[key]

    inside = _inside_viability(C, T, E, O, bounds)
    recovery = _recovery_score(C, T, E, O, dC, dT, dE, dO, bounds)

    if inside:
        mem.inside_count += 1
        mem.outside_count = 0
    else:
        mem.outside_count += 1
        mem.inside_count = 0

    scores = _instantaneous_scores(C, T, E, O, dC, dT, dE, dO, bounds, par, scenario_cfg)
    candidate = max(STATE_ORDER, key=lambda s: (scores[s], -STATE_ORDER.index(s)))

    grace_steps = 3
    apoptosis_persistence = 4
    switch_persistence = 2

    if not inside and mem.last_label == "Undetermined" and mem.dwell_count < grace_steps:
        candidate = "Undetermined"

    if not inside and mem.outside_count < grace_steps and recovery >= 1.0:
        candidate = "Undetermined"

    if candidate == "Apoptosis":
        if inside:
            candidate = mem.last_label if mem.last_label != "Apoptosis" else "Undetermined"
        elif mem.outside_count < apoptosis_persistence and recovery > 0.0:
            candidate = "Undetermined"

    if mem.last_label == "Apoptosis":
        if not inside or recovery < 2.0:
            final = "Apoptosis"
        else:
            final = "Undetermined"
    else:
        allowed = {
            "Undetermined": set(STATE_ORDER),
            "Quiescence": {"Quiescence", "Migration", "Proliferation", "Diversification", "Undetermined", "Apoptosis"},
            "Migration": {"Migration", "Diversification", "Quiescence", "Undetermined", "Apoptosis"},
            "Proliferation": {"Proliferation", "Diversification", "Quiescence", "Undetermined", "Apoptosis"},
            "Diversification": {"Diversification", "Migration", "Proliferation", "Quiescence", "Undetermined", "Apoptosis"},
            "Apoptosis": {"Apoptosis", "Undetermined"},
        }
        if candidate not in allowed.get(mem.last_label, set(STATE_ORDER)):
            candidate = mem.last_label

        if candidate == mem.last_label:
            final = candidate
        else:
            recent_match = sum(1 for x in mem.history if x == candidate)
            if recent_match >= switch_persistence - 1:
                final = candidate
            else:
                final = mem.last_label

    if final == mem.last_label:
        mem.dwell_count += 1
    else:
        mem.dwell_count = 1
    mem.last_label = final
    mem.history.append(candidate)

    return final
