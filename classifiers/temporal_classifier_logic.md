# Temporal taxonomy classifier design rules

This document defines a temporal taxonomy classifier that explicitly **inherits from** the current static classifier in `taxonomy_classifier.py` rather than replacing it.

The temporal classifier is designed to preserve the static classifier philosophy:

1. **Dynamic-first instantaneous evidence remains primary**.
2. **Static contextual parameters remain those already defined and used by the static classifier**.
3. **Viability bounds remain separate from taxonomy semantics**.
4. **Temporal logic is added only as a clearly delimited post-processing layer**.

## Inheritance from the static classifier

The temporal classifier does **not** introduce a new instantaneous scoring system. It first calls the existing static classifier:

```python
classify_static_state(C, T, E, O, dC, dT, dE, dO, bounds, par=None, scenario_cfg=None)
```

This means the temporal model inherits all of the following from the static classifier:

- the same dynamic-first interpretation of `C`, `T`, `E`, `O` and their derivatives,
- the same interpretable static parameter subset,
- the same contextual gates,
- the same label set,
- the same ordered first-match rule structure.

Therefore, the temporal layer is not a new taxonomy definition. It is a temporal wrapper around the already defined taxonomy.

## What is new in the temporal classifier

The temporal classifier adds only a small number of explicit time-related computational elements.

### 1. Memory of previous assigned labels

The classifier stores a short memory object per parameter/scenario signature. That memory includes:

- the last temporal label,
- the last base static label,
- dwell count for the current temporal label,
- short recent history of temporal labels,
- short recent history of base static labels,
- counts of consecutive samples inside and outside viability bounds.

This memory is used only to stabilize label transitions over time.

### 2. Viability-membership tracking

The temporal classifier computes whether the current point lies inside the viability bounds:

```python
_inside_viability(C, T, E, O, bounds)
```

This is **not** used to redefine taxonomy labels. It is used only to detect whether a point has spent several sampled steps outside the admissible region.

### 3. Recovery score

The temporal wrapper computes a simple recovery score based on whether an out-of-bounds variable is moving back toward its admissible range.

Examples:

- if `C < C_min` but `dC > 0`, recovery increases,
- if `O < O_min` but `dO > 0`, recovery increases,
- if `T > T_max` but `dT < 0`, recovery increases.

This score is not a taxonomy rule. It is only a temporal hint used to avoid prematurely locking a state into a terminal interpretation while it is still recovering.

### 4. Grace period for transient excursions

When a trajectory has only recently moved outside viability bounds and is already showing recovery, the temporal wrapper may temporarily output `Undetermined` rather than immediately forcing a strong state conclusion.

This prevents brief excursions from being overinterpreted.

### 5. Persistence requirement for state switching

A new candidate label is not always accepted immediately. If the candidate differs from the current temporal label, the wrapper requires short recent support before switching.

This is a temporal smoothing rule, not a taxonomy-definition rule.

### 6. Sticky handling of sustained apoptosis

Once sustained apoptosis has been established, the temporal wrapper keeps the label unless the trajectory shows sufficiently strong recovery and re-entry toward viable behavior.

This reflects the idea that terminal collapse should be harder to reverse at the label level than ordinary mode switching.

## Explicit separation of taxonomy and viability

A central design requirement is that taxonomy and viability remain separate.

- **Taxonomy** answers: what biological mode does the current state most resemble?
- **Viability** answers: is the current state inside the admissible bounds?

The temporal classifier preserves that separation by using the static taxonomy classifier as the source of biological labels and using viability membership only as a temporal-stability signal.

Viability does **not** directly assign Apoptosis, Migration, Proliferation, Quiescence, or Diversification. It only helps the temporal wrapper decide whether a label should be stabilized, delayed, or temporarily replaced by `Undetermined` during transient excursions.

## Temporal decision workflow

The temporal classifier follows this sequence.

1. Compute the base instantaneous taxonomy label using the static classifier.
2. Compute whether the state is inside viability bounds.
3. Compute the recovery score.
4. Apply explicit temporal wrapper rules:
   - grace outside bounds,
   - recovery-aware suppression of premature apoptosis,
   - persistence requirement for label switching,
   - sticky sustained apoptosis.
5. Update temporal memory.
6. Return the final temporal label.

## Temporal pseudocode

```python
base_label = classify_static_state(...)
inside = inside_viability(...)
recovery = recovery_score(...)

candidate = base_label

if transiently_outside_bounds and recovering and base_label != "Apoptosis":
    candidate = "Undetermined"

if base_label == "Apoptosis":
    if inside_bounds:
        candidate = "Undetermined"
    elif outside_duration_is_short and recovering:
        candidate = "Undetermined"

if last_temporal_label == "Apoptosis":
    if still_outside_bounds or weak_recovery:
        final = "Apoptosis"
    else:
        final = "Undetermined"
else:
    if candidate_is_supported_by_recent_history:
        final = candidate
    else:
        final = previous_label_or_candidate

update_memory()
return final
```

This pseudocode should be read as a wrapper around the static classifier, not as a replacement classifier.

## Relationship to the previous temporal implementation

This rewritten temporal classifier intentionally avoids the problems of the earlier temporal implementation.

It does **not**:

- construct a new per-label score system,
- redefine label semantics through generic parameter multipliers,
- mix taxonomy definition and viability scoring into a single opaque decision layer,
- replace the explicit rule structure of the static classifier.

Instead, it inherits the already clarified static taxonomy and adds only explicit temporal consistency rules.

## Solution-level use

The temporal classifier also provides a `classify_solution(...)` helper.

That function:

1. computes derivatives from the full trajectory,
2. samples evenly spaced trajectory points,
3. applies the temporal point classifier in sequence,
4. returns the majority temporal label.

This preserves the same sampling-based trajectory summary style already used in the static classifier ecosystem, while making the temporal additions explicit and limited.

## Recommended interpretation

The correct interpretation of this temporal classifier is:

- the **static classifier** defines what each biological label means,
- the **temporal wrapper** decides how quickly those labels are allowed to change across sampled trajectory points,
- the **viability checks** contribute only to temporal caution and recovery handling, not to taxonomy semantics themselves.

That separation keeps the code readable, auditable, and faithful to the current static classifier philosophy.
