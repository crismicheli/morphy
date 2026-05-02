from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .taxonomy_classifier import STATE_COLORS
from .taxonomy_classifier import classify_state as classify_static_state
from .temporal_taxonomy_classifier import classify_state as classify_temporal_state

STATE_ORDER = (
    "Apoptosis",
    "Proliferation",
    "Migration",
    "Quiescence",
    "Diversification",
    "Undetermined",
)


@dataclass
class StateMachineMemory:
    """Minimal memory for enforcing allowed transitions only.

    This layer keeps track of the last emitted state-machine label so that
    it can apply an explicit transition graph on top of the temporal label.

    It does not add extra stickiness beyond that graph and does not
    re-implement temporal smoothing.
    """

    last_state_machine_label: str = "Undetermined"


_STATE_MACHINE_CACHE: Dict[Tuple[int, int], StateMachineMemory] = defaultdict(StateMachineMemory)


def reset_classifier_memory() -> None:
    """Clear cached state-machine memory.

    This should be called when starting a new trajectory classification
    so that the allowed-transition logic does not leak across scenarios.
    """
    _STATE_MACHINE_CACHE.clear()


def _infer_signature(par: Optional[dict], scenario_cfg: Optional[dict]) -> Tuple[int, int]:
    """Infer a hashable signature for caching per parameter/scenario pair."""
    p_val = 0.0 if scenario_cfg is None else float(scenario_cfg.get("p", 0.0))
    s_label = "" if scenario_cfg is None else str(scenario_cfg.get("label", ""))
    par_items = tuple(sorted((par or {}).items()))
    return hash((round(p_val, 6), s_label)), hash(par_items)


def _allowed_transitions() -> Dict[str, set]:
    """Explicit transition graph between coarse biological regimes.

    The graph is intentionally sparse: it encodes which mode changes are
    considered biologically plausible in one step, while leaving the
    temporal classifier in charge of local smoothing.
    """
    return {
        # From an unresolved state, any label is allowed.
        "Undetermined": set(STATE_ORDER),
        # Quiet regimes can move into activity or collapse.
        "Quiescence": {
            "Quiescence",
            "Migration",
            "Proliferation",
            "Diversification",
            "Undetermined",
            "Apoptosis",
        },
        # Migration and proliferation can interconvert, rest, diversify, or collapse.
        "Migration": {
            "Migration",
            "Diversification",
            "Quiescence",
            "Proliferation",
            "Undetermined",
            "Apoptosis",
        },
        "Proliferation": {
            "Proliferation",
            "Diversification",
            "Quiescence",
            "Migration",
            "Undetermined",
            "Apoptosis",
        },
        # Diversification can move back to active/quiet modes or collapse.
        "Diversification": {
            "Diversification",
            "Migration",
            "Proliferation",
            "Quiescence",
            "Undetermined",
            "Apoptosis",
        },
        # Once in apoptosis, only staying apoptotic or becoming unresolved is allowed.
        "Apoptosis": {"Apoptosis", "Undetermined"},
    }


def _apply_allowed_transition(candidate: str, previous: str) -> str:
    """Project a candidate label onto the allowed transition graph.

    If the candidate transition is not allowed from the previous label,
    the previous label is retained. This adds a lightweight regime graph
    on top of the temporal classifier without additional stickiness.
    """
    allowed = _allowed_transitions()
    if candidate not in allowed.get(previous, set(STATE_ORDER)):
        return previous
    return candidate


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
    temporal_grace_outside_steps: int = 2,
    temporal_apoptosis_persistence_steps: int = 3,
    temporal_switch_persistence_steps: int = 2,
) -> str:
    """State-machine wrapper with only an explicit transition graph.

    Layer 1 (static): `taxonomy_classifier.classify_state` defines
    instantaneous label semantics.

    Layer 2 (temporal): `temporal_taxonomy_classifier.classify_state`
    adds local sequential smoothing, viability-aware grace periods,
    and recovery-sensitive handling of apoptosis.

    Layer 3 (this module): applies a simple allowed-transition graph on
    top of the temporal label, using only the last emitted state-machine
    label as context. No extra stickiness, streaks, or scores are added
    beyond what the temporal classifier already implements.
    """
    key = _infer_signature(par, scenario_cfg)
    mem = _STATE_MACHINE_CACHE[key]

    # Compute the static and temporal labels but do not alter their logic.
    static_label = classify_static_state(
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

    temporal_label = classify_temporal_state(
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
        grace_outside_steps=temporal_grace_outside_steps,
        apoptosis_persistence_steps=temporal_apoptosis_persistence_steps,
        switch_persistence_steps=temporal_switch_persistence_steps,
    )

    # Apply only the explicit transition graph.
    previous = mem.last_state_machine_label
    final = _apply_allowed_transition(temporal_label, previous)

    # Update memory with the new label.
    mem.last_state_machine_label = final

    return final
