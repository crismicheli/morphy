from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .taxonomy_classifier import classify_state as classify_static_state
from .taxonomy_classifier import STATE_COLORS


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
