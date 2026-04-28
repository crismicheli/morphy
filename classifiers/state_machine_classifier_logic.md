# Deterministic state machine classifier design rules

This document describes a three-layer state machine classifier whose transition logic is expressed through **explicit named rules**, not score biases.

The classifier stack is:

1. **Static classifier**: defines instantaneous taxonomy label meaning.
2. **Temporal classifier**: stabilizes short-horizon label fluctuations.
3. **State machine classifier**: applies explicit transition rules to produce regime-level labels.

## State Machine logic   

![image](state_machine_logic.png?v=2)


## Core principle

The state machine does not construct a new scoring system and does not perturb hidden label scores. It inherits labels from the static and temporal classifiers, then applies a fixed sequence of explicit transition rules.

This makes the code auditable: every final transition can be explained by a named rule rather than by a numerical bias.

## Inputs inherited from lower layers

For each sampled point, the state machine reads:

- `static_label` from the static classifier,
- `temporal_label` from the temporal classifier,
- `inside` from a separate viability-membership check,
- `recovery` from a separate recovery-score helper,
- current memory state and streak counters.

The state machine does not redefine taxonomy semantics. It only controls how inherited labels persist or transition over time.

## Memory and streaks

The state machine keeps the following memory variables:

- `last_state_machine_label`
- `last_temporal_label`
- `last_static_label`
- `dwell_count`
- `history`
- `temporal_history`
- `static_history`
- `outside_viability_count`
- `inside_viability_count`
- `recovery_streak`
- `stress_streak`
- `ambiguity_streak`

These variables support explicit transition rules rather than score manipulation.

## Rule order

The state machine applies its rules in the following order:

1. Ambiguity persistence rule
2. Stress lock rule
3. Recovery unlock rule
4. Allowed transition rule
5. Switch persistence rule
6. Sustained apoptosis retention rule

The order matters. Earlier rules may alter the candidate label before later rules evaluate it.

## Transition rules

| Rule | Condition | Action |
|---|---|---|
| **Ambiguity persistence** | `ambiguity_streak >= state_ambiguity_window` | Set candidate to `Undetermined` |
| **Stress lock** | `stress_streak >= state_stress_lock_steps`, outside viability, and `temporal_label` is `Apoptosis` or `Undetermined` | Set candidate to `Apoptosis` |
| **Recovery unlock** | `recovery_streak >= state_recovery_lock_steps` | If candidate is `Apoptosis`, replace with `Undetermined`; if candidate is `Undetermined` and `temporal_label` is `Quiescence` or `Proliferation`, restore that temporal label |
| **Allowed transition** | Candidate transition is not permitted by the transition map from the previous state-machine label | Reject the candidate and keep the previous label |
| **Switch persistence** | Candidate differs from the previous label but lacks enough recent support in history | Keep the previous label, unless the previous label is `Undetermined`, in which case allow the candidate |
| **Sustained apoptosis retention** | Previous state-machine label is `Apoptosis` | Keep `Apoptosis` while still outside viability or while recovery remains weak; only relax to `Undetermined` after sustained recovery |

## Transition states

[!image](state_machine_v2.png)  


## Allowed transition map

The explicit allowed transition map is:

| Previous label | Allowed next labels |
|---|---|
| `Undetermined` | `Apoptosis`, `Proliferation`, `Migration`, `Quiescence`, `Diversification`, `Undetermined` |
| `Quiescence` | `Quiescence`, `Migration`, `Proliferation`, `Diversification`, `Undetermined`, `Apoptosis` |
| `Migration` | `Migration`, `Diversification`, `Quiescence`, `Undetermined`, `Apoptosis` |
| `Proliferation` | `Proliferation`, `Diversification`, `Quiescence`, `Undetermined`, `Apoptosis` |
| `Diversification` | `Diversification`, `Migration`, `Proliferation`, `Quiescence`, `Undetermined`, `Apoptosis` |
| `Apoptosis` | `Apoptosis`, `Undetermined` |

This map is explicit and should be treated as part of the model definition.

## Workflow pseudocode

```python
static_label = static_classifier.classify_state(...)
temporal_label = temporal_classifier.classify_state(...)
inside = inside_viability(...)
recovery = recovery_score(...)

update_streaks(...)

candidate = temporal_label
candidate = ambiguity_persistence_rule(candidate)
candidate = stress_lock_rule(candidate)
candidate = recovery_unlock_rule(candidate)
candidate = allowed_transition_rule(candidate, previous_label)
candidate = switch_persistence_rule(candidate, previous_label)
final = sustained_apoptosis_retention_rule(candidate, previous_label)

update_memory()
return final
```

## Separation of taxonomy and viability

The state machine preserves the same separation used in the lower layers.

- Taxonomy meaning comes from the static classifier.
- Short-term label stabilization comes from the temporal classifier.
- Viability membership and recovery are not label definitions; they are only transition-management signals.

This means the state machine does not collapse taxonomy into viability status.

## What is new on top of the lower layers

The genuinely new contribution of the state machine is:

- explicit regime-level transition rules,
- explicit allowed-transition constraints,
- explicit persistent-ambiguity handling,
- explicit sustained-stress escalation,
- explicit sustained-recovery relaxation,
- explicit regime-level persistence beyond the temporal wrapper.

These additions operate on inherited labels rather than redefining taxonomy.

## Solution-level use

At the solution level, the classifier:

1. computes derivatives,
2. samples evenly spaced points,
3. applies the full three-layer stack point by point,
4. returns the majority state-machine label.

This preserves continuity with the existing sampling-and-voting design while making transition logic more explicit.

## Recommended interpretation

The state machine should be interpreted as a regime-management layer, not as a replacement taxonomy model. Its role is to decide when inherited labels should persist, when they may transition, and when ambiguity, sustained stress, or sustained recovery should dominate the regime-level interpretation.
