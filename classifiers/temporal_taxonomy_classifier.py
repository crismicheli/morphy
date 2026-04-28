from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple
import math

STATE_COLORS = {
    "Apoptosis": "#111111",
    "Migration": "#00B7FF",from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from taxonomy_classifier import classify_state as classify_static_state
from taxonomy_classifier import STATE_COLORS


STATE_ORDER = (
    "Apoptosis",
    "Proliferation",
    "Migration",
    "Quiescence",
    "Diversification",
    "Undetermined",
)


@dataclass
class TemporalMemory:
    last_label: str = "Undetermined"
    last_base_label: str = "Undetermined"
    dwell_count: int = 0
    history: Deque[str] = field(default_factory=lambda: deque(maxlen=8))
    base_history: Deque[str] = field(default_factory=lambda: deque(maxlen=8))
    outside_viability_count: int = 0
    inside_viability_count: int = 0


_TEMPORAL_CACHE: Dict[Tuple[int, int], TemporalMemory] = defaultdict(TemporalMemory)


def reset_classifier_memory() -> None:
    """Clear all cached temporal memory states."""
    _TEMPORAL_CACHE.clear()


def _infer_signature(par: Optional[dict], scenario_cfg: Optional[dict]) -> Tuple[int, int]:
    p_val = 0.0 if scenario_cfg is None else float(scenario_cfg.get("p", 0.0))
    s_label = "" if scenario_cfg is None else str(scenario_cfg.get("label", ""))
    par_items = tuple(sorted((par or {}).items()))
    return hash((round(p_val, 6), s_label)), hash(par_items)


def _inside_viability(C: float, T: float, E: float, O: float, bounds: dict) -> bool:
    """Evaluate viability-bounds membership.

    This function is intentionally separate from taxonomy logic.
    It is used only by the temporal wrapper to detect persistence outside bounds,
    never to redefine the underlying static taxonomy rules.
    """
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
    """Measure whether an out-of-bounds state is moving back toward viability.

    This score is a temporal helper only. It does not define taxonomy labels.
    """
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


def _majority_label(labels: List[str]) -> str:
    if not labels:
        return "Undetermined"
    counts = Counter(labels)
    best_n = max(counts.values())
    tied = {label for label, n in counts.items() if n == best_n}
    for label in STATE_ORDER:
        if label in tied:
            return label
    return "Undetermined"


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
    *,
    grace_outside_steps: int = 2,
    apoptosis_persistence_steps: int = 3,
    switch_persistence_steps: int = 2,
) -> str:
    """Temporal wrapper around the static taxonomy classifier.

    Design philosophy:
    1. Inherit instantaneous taxonomy from taxonomy_classifier.classify_state.
    2. Keep viability-bounds checks separate from taxonomy rules.
    3. Add only explicit temporal post-processing:
       - short grace period for transient excursions,
       - recovery-aware suppression of premature terminal labels,
       - persistence requirement for rapid state switching,
       - sticky terminal behavior for sustained apoptosis.

    The temporal layer does NOT re-score labels from scratch and does NOT replace
    the static classifier's explicit dynamic-first rule structure.
    """
    key = _infer_signature(par, scenario_cfg)
    mem = _TEMPORAL_CACHE[key]

    base_label = classify_static_state(
        C, T, E, O, dC, dT, dE, dO,
        bounds=bounds,
        par=par,
        scenario_cfg=scenario_cfg,
    )

    inside = _inside_viability(C, T, E, O, bounds)
    recovery = _recovery_score(C, T, E, O, dC, dT, dE, dO, bounds)

    if inside:
        mem.inside_viability_count += 1
        mem.outside_viability_count = 0
    else:
        mem.outside_viability_count += 1
        mem.inside_viability_count = 0

    candidate = base_label

    if not inside and base_label != "Apoptosis":
        if mem.outside_viability_count <= grace_outside_steps and recovery > 0.0:
            candidate = "Undetermined"

    if base_label == "Apoptosis":
        if inside:
            candidate = "Undetermined"
        elif mem.outside_viability_count < apoptosis_persistence_steps and recovery > 0.0:
            candidate = "Undetermined"

    if mem.last_label == "Apoptosis":
        if not inside or recovery < 2.0:
            final = "Apoptosis"
        else:
            final = "Undetermined"
    else:
        if candidate == mem.last_label:
            final = candidate
        else:
            recent_match = sum(1 for x in mem.history if x == candidate)
            if recent_match >= switch_persistence_steps - 1:
                final = candidate
            else:
                final = mem.last_label if mem.last_label != "Undetermined" else candidate

    if final == mem.last_label:
        mem.dwell_count += 1
    else:
        mem.dwell_count = 1

    mem.last_label = final
    mem.last_base_label = base_label
    mem.history.append(final)
    mem.base_history.append(base_label)

    return final


def classify_solution(
    sol,
    bounds: dict,
    par: Optional[dict] = None,
    scenario_cfg: Optional[dict] = None,
    n_samples: int = 7,
    *,
    grace_outside_steps: int = 2,
    apoptosis_persistence_steps: int = 3,
    switch_persistence_steps: int = 2,
    reset_memory_before_run: bool = True,
) -> str:
    """Classify a full solution using the temporal wrapper over sampled points.

    The workflow is:
    1. compute local derivatives from the trajectory,
    2. sample evenly spaced points,
    3. classify each point with the temporal wrapper,
    4. return the majority temporal label.

    This preserves inheritance from the static classifier while adding explicit,
    limited temporal consistency rules.
    """
    if reset_memory_before_run:
        reset_classifier_memory()

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
                grace_outside_steps=grace_outside_steps,
                apoptosis_persistence_steps=apoptosis_persistence_steps,
                switch_persistence_steps=switch_persistence_steps,
            )
        )

    return _majority_label(labels)


def classify_solutions(
    solutions: Iterable,
    bounds: dict,
    par: Optional[dict] = None,
    scenario_cfg: Optional[dict] = None,
    n_samples: int = 7,
    *,
    grace_outside_steps: int = 2,
    apoptosis_persistence_steps: int = 3,
    switch_persistence_steps: int = 2,
) -> List[str]:
    return [
        classify_solution(
            sol,
            bounds=bounds,
            par=par,
            scenario_cfg=scenario_cfg,
            n_samples=n_samples,
            grace_outside_steps=grace_outside_steps,
            apoptosis_persistence_steps=apoptosis_persistence_steps,
            switch_persistence_steps=switch_persistence_steps,
            reset_memory_before_run=True,
        )
        for sol in solutions
    ]
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
