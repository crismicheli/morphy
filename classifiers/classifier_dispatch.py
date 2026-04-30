from __future__ import annotations

from .taxonomy_classifier import STATE_COLORS as STATIC_STATE_COLORS
from .taxonomy_classifier import classify_state as classify_static_state
from .temporal_taxonomy_classifier import classify_state as classify_temporal_state
from .temporal_taxonomy_classifier import reset_classifier_memory as reset_temporal_memory
from .state_machine_classifier import classify_state as classify_state_machine
from .state_machine_classifier import reset_classifier_memory as reset_state_machine_memory


def get_classifier_components(classifier_type: str):
    classifier_type = str(classifier_type).strip().lower()

    if classifier_type == "static":
        return classify_static_state, None, STATIC_STATE_COLORS

    if classifier_type == "temporal":
        return classify_temporal_state, reset_temporal_memory, STATIC_STATE_COLORS

    if classifier_type == "state_machine":
        return classify_state_machine, reset_state_machine_memory, STATIC_STATE_COLORS

    raise ValueError(
        f"Unknown classifier_type={classifier_type!r}. "
        "Expected one of: static, temporal, state_machine."
    )
