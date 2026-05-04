
# Finite state machine classifier

This document describes the current `state_machine_classifier.py` as an **explicit finite-state-machine reformulation** of the temporal classifier.

The classifier stack is:

1. **Static classifier**: defines the instantaneous biological meaning of the taxonomy labels.
2. **State machine classifier**: re-expresses the temporal logic as guarded state transitions so that the emitted labels match the temporal classifier.

In this revised design, the state machine is **not** an extra regime-management layer on top of the temporal classifier. Instead, it is a different implementation form of the same temporal logic.

## Core principle

The current state machine does not introduce new biological labels, new transition penalties, new persistence layers, or new regime-level constraints beyond the temporal classifier.

Its purpose is only to make the temporal logic explicit in state-machine form:

- the current emitted label is treated as the current FSM state,
- the static classifier provides the instantaneous proposed label,
- viability and recovery act as guarded transition signals,
- recent label history controls whether a switch is accepted,
- apoptosis remains a special sticky state with guarded release.

The intended result is behavioral equivalence: for the same input sequence and the same reset pattern, the state-machine classifier should emit the same labels as the temporal classifier.

## State set

The FSM uses the same label set as the static and temporal classifiers:

- `Apoptosis`
- `Proliferation`
- `Migration`
- `Quiescence`
- `Diversification`
- `Undetermined`

These are not new states invented by the state machine. They are the same taxonomy labels, now interpreted as explicit FSM states.

## Inputs to the state machine

For each sampled point, the state machine reads:

- `base_label` from the static classifier,
- `inside` from the viability-membership check,
- `recovery` from the recovery-score helper,
- the current FSM state,
- recent emitted-label history,
- inside/outside viability counters.

This means the state machine still inherits its biological meaning from the static classifier. The FSM only governs how the current label evolves over time.

## Memory

For each parameter/scenario signature, the state machine stores a `TemporalFSMMemory` object containing:

- `current_state`
- `last_base_label`
- `dwell_count`
- `history`
- `base_history`
- `outside_viability_count`
- `inside_viability_count`

Compared with the older state-machine design, this memory is intentionally minimal. It removes the extra regime-management machinery such as ambiguity streaks, stress streaks, recovery streaks, and independent state-machine persistence counters.

## Viability and recovery helpers

The FSM uses the same two helper concepts as the temporal classifier.

### Viability membership

A point is considered inside viability if:

- `C >= C_min`
- `T_min <= T <= T_max`
- `E_min <= E <= E_max`
- `O >= O_min`

This viability test is not itself a taxonomy rule. It is only a transition-management signal.

### Recovery score

The FSM computes the same recovery score used by the temporal classifier. Recovery increases when an out-of-bounds variable is moving back toward the viable region, for example:

- `dC > 0` while `C < C_min`
- `dT > 0` while `T < T_min`
- `dT < 0` while `T > T_max`
- `dE > 0` while `E < E_min`
- `dE < 0` while `E > E_max`
- `dO > 0` while `O < O_min`

The recovery score does not define labels by itself. It only guards certain transitions.

## Decision workflow

The current state machine follows the same overall logic as the temporal classifier, but expressed as explicit state transitions.

1. Compute the instantaneous `base_label` using the static classifier.
2. Evaluate viability membership.
3. Compute the recovery score.
4. Update inside/outside viability counters.
5. Build a candidate label from the base label using the same temporal caution rules.
6. Apply a state-dependent transition rule:
   - if the current state is `Apoptosis`, use the apoptosis retention/release rule,
   - otherwise, use the non-apoptotic switching rule.
7. Update memory.
8. Return the new FSM state as the emitted label.

## Candidate construction

The candidate label starts as the `base_label` from the static classifier.

### 1. Brief-excursion grace rule

If the point is outside viability, the base label is not `Apoptosis`, the outside excursion is still recent, and recovery is already present, the candidate is softened to `Undetermined`.

In code-like terms:

```python
if not inside and base_label != "Apoptosis":
    if outside_viability_count <= grace_outside_steps and recovery > 0.0:
        candidate = "Undetermined"
```

This prevents overreaction to short-lived excursions.

### 2. Early-apoptosis caution rule

If the base label is `Apoptosis`, the FSM does not always accept it immediately.

- If the point is actually inside viability, the candidate becomes `Undetermined`.
- If the system has been outside viability for fewer than `apoptosis_persistence_steps` and recovery is already present, the candidate also becomes `Undetermined`.

In code-like terms:

```python
if base_label == "Apoptosis":
    if inside:
        candidate = "Undetermined"
    elif outside_viability_count < apoptosis_persistence_steps and recovery > 0.0:
        candidate = "Undetermined"
```

This preserves the temporal classifier's recovery-aware caution around premature terminal labeling.

## State-dependent transitions

Once the candidate label is built, the FSM applies one of two transition rules depending on the current state.

### 1. Transition from non-apoptotic states

If the current FSM state is not `Apoptosis`, the machine uses the anti-flicker switching rule.

- If the candidate matches the current state, remain in the current state.
- Otherwise, count how often the candidate appears in recent emitted-label history.
- If that recent support is large enough, switch to the candidate state.
- If that support is too weak, keep the current state, unless the current state is `Undetermined`, in which case the candidate is allowed.

This is equivalent to the temporal classifier's persistence-based switching logic.

In simplified form:

```python
if candidate == current_state:
    final = current_state
else:
    recent_match = count_recent_history_matches(candidate)
    if recent_match >= switch_persistence_steps - 1:
        final = candidate
    else:
        final = current_state if current_state != "Undetermined" else candidate
```

### 2. Transition from apoptosis

If the current FSM state is already `Apoptosis`, the machine uses the sticky-apoptosis rule.

- If the system is still outside viability, remain in `Apoptosis`.
- If recovery is still weak (`recovery < 2.0`), remain in `Apoptosis`.
- Only if the point is back inside viability and recovery is strong enough does the FSM leave `Apoptosis`, and then it releases to `Undetermined`, not directly to another positive label.

In simplified form:

```python
if current_state == "Apoptosis":
    if not inside or recovery < 2.0:
        final = "Apoptosis"
    else:
        final = "Undetermined"
```

This is the key asymmetric transition in the machine.

## Transition interpretation

The FSM therefore has a simple interpretation:

- most labels behave like ordinary states with persistence-based switching,
- `Apoptosis` behaves like a sticky guarded state,
- viability and recovery do not define labels, but they decide whether certain state changes are allowed,
- the state machine is a transparent encoding of the temporal classifier's short-memory logic.

## What changed relative to the older state-machine design

The previous state-machine document described a heavier regime-management layer with:

- ambiguity persistence,
- stress lock,
- recovery unlock,
- explicit allowed-transition maps,
- an additional switch-persistence layer on top of the temporal classifier,
- sustained apoptosis retention at a separate regime level.

That is no longer the current implementation.

The revised state machine has been simplified so that it no longer adds independent regime logic on top of the temporal classifier. Instead, it reuses the temporal logic itself and presents it as an FSM.

## Pseudocode summary

```python
base_label = classify_static_state(...)
inside = inside_viability(...)
recovery = recovery_score(...)
update_inside_outside_counters(...)

candidate = base_label

if not inside and base_label != "Apoptosis":
    if outside_viability_count <= grace_outside_steps and recovery > 0.0:
        candidate = "Undetermined"

if base_label == "Apoptosis":
    if inside or (outside_viability_count < apoptosis_persistence_steps and recovery > 0.0):
        candidate = "Undetermined"

if current_state == "Apoptosis":
    if not inside or recovery < 2.0:
        final = "Apoptosis"
    else:
        final = "Undetermined"
else:
    if candidate == current_state:
        final = current_state
    else:
        recent_match = count_recent_history_matches(candidate)
        if recent_match >= switch_persistence_steps - 1:
            final = candidate
        else:
            final = current_state if current_state != "Undetermined" else candidate

update_memory(...)
return final
```

## Scope of the classifier

Each call returns the label for the current point, interpreted in the context of short memory from previous points.

So although the code is now written as a state machine, the classifier remains local-and-sequential rather than trajectory-global. It does not label a point by sampling the whole trajectory. It reads the current point, consults its small memory, updates its state, and returns the current label.

## Recommended interpretation

The best short description of the current implementation is:

> The state machine classifier is an explicit FSM reformulation of the temporal classifier.

It should be read as a semantics-preserving rewrite, not as an added regime layer. Its role is to make the temporal classifier easier to reason about by expressing the same logic as guarded transitions between explicit label states.
