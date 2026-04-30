from __future__ import annotations

from typing import Callable, Dict, Optional

from taxonomy_classifier import STATE_COLORS
from taxonomy_classifier import classify_state as classify_static_state
from temporal_taxonomy_classifier import (
    classify_state as classify_temporal_state,
    reset_classifier_memory as reset_temporal_classifier_memory,
)
from state_machine_classifier import (
    classify_state as classify_state_machine_state,
    reset_classifier_memory as reset_state_machine_classifier_memory,
)

CLASSIFIER_TYPES = ("static", "temporal", "state_machine")


ClassifierFn = Callable[..., str]


_CLASSIFIER_FUNCS: Dict[str, ClassifierFn] = {
    "static": classify_static_state,
    "temporal": classify_temporal_state,
    "state_machine": classify_state_machine_state,
}


def validate_classifier_type(classifier_type: str) -> str:
    normalized = str(classifier_type).strip().lower()
    if normalized not in _CLASSIFIER_FUNCS:
        valid = ", ".join(CLASSIFIER_TYPES)
        raise ValueError(f"Unknown classifier_type={classifier_type!r}. Expected one of: {valid}.")
    return normalized


def get_state_colors() -> dict:
    return STATE_COLORS


def reset_classifier_memory(classifier_type: str) -> None:
    classifier_type = validate_classifier_type(classifier_type)
    if classifier_type == "temporal":
        reset_temporal_classifier_memory()
    elif classifier_type == "state_machine":
        reset_state_machine_classifier_memory()


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
    classifier_type: str = "static",
    **classifier_kwargs,
) -> str:
    classifier_type = validate_classifier_type(classifier_type)
    fn = _CLASSIFIER_FUNCS[classifier_type]
    return fn(
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
        **classifier_kwargs,
    )
