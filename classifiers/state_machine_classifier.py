from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Optional, Tuple

from .taxonomy_classifier import STATE_COLORS
from .taxonomy_classifier import classify_state as classify_static_state


class FSMState(str, Enum):
    APOPTOSIS = "Apoptosis"
    PROLIFERATION = "Proliferation"
    MIGRATION = "Migration"
    QUIESCENCE = "Quiescence"
    DIVERSIFICATION = "Diversification"
    UNDETERMINED = "Undetermined"


@dataclass
class TemporalFSMMemory:
    current_state: FSMState = FSMState.UNDETERMINED
    last_base_label: str = "Undetermined"
    dwell_count: int = 0
    history: Deque[str] = field(default_factory=lambda: deque(maxlen=8))
    base_history: Deque[str] = field(default_factory=lambda: deque(maxlen=8))
    outside_viability_count: int = 0
    inside_viability_count: int = 0


_FSM_CACHE: Dict[Tuple[int, int], TemporalFSMMemory] = defaultdict(TemporalFSMMemory)


def reset_classifier_memory() -> None:
    """Clear all cached FSM memory states."""
    _FSM_CACHE.clear()


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


def _coerce_state(label: str) -> FSMState:
    try:
        return FSMState(label)
    except ValueError:
        return FSMState.UNDETERMINED


def _transition_from_non_apoptosis(
    candidate: str,
    current_state: FSMState,
    history: Deque[str],
    switch_persistence_steps: int,
) -> FSMState:
    if candidate == current_state.value:
        return current_state

    recent_match = sum(1 for x in history if x == candidate)
    if recent_match >= switch_persistence_steps - 1:
        return _coerce_state(candidate)

    if current_state != FSMState.UNDETERMINED:
        return current_state
    return _coerce_state(candidate)


def _transition_from_apoptosis(
    inside: bool,
    recovery: float,
) -> FSMState:
    if not inside or recovery < 2.0:
        return FSMState.APOPTOSIS
    return FSMState.UNDETERMINED


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
    """FSM formulation of the temporal classifier.

    This classifier is written as an explicit finite-state machine but preserves
    the same output behavior as the current temporal classifier:
    - static classifier defines the instantaneous base label,
    - viability/recovery define guarded transitions,
    - switching persistence uses recent label support,
    - apoptosis is treated as a sticky state with guarded release.
    """
    key = _infer_signature(par, scenario_cfg)
    mem = _FSM_CACHE[key]

    base_label = classify_static_state(
        C,
        T,
        E,
        O,
        dC,
        dT,
        dE,
        dO,
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

    if mem.current_state == FSMState.APOPTOSIS:
        final_state = _transition_from_apoptosis(inside=inside, recovery=recovery)
    else:
        final_state = _transition_from_non_apoptosis(
            candidate=candidate,
            current_state=mem.current_state,
            history=mem.history,
            switch_persistence_steps=switch_persistence_steps,
        )

    if final_state == mem.current_state:
        mem.dwell_count += 1
    else:
        mem.dwell_count = 1

    mem.current_state = final_state
    mem.last_base_label = base_label
    mem.history.append(final_state.value)
    mem.base_history.append(base_label)

    return final_state.value
