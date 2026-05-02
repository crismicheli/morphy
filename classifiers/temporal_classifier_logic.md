# Temporal taxonomy classifier

The temporal taxonomy classifier is best understood as the static classifier plus a small memory-and-caution layer. It does not redefine the taxonomy, introduce a new scoring system, or replace the biological meaning of the existing labels. Instead, it first asks what the static classifier thinks the current point looks like, then adjusts that answer using short-term memory, viability tracking, recovery detection, and a few simple persistence rules.

## Core idea

The static classifier labels what the system looks like right now from the current state, local derivatives, and a limited set of interpretable parameters. The temporal classifier keeps that exact instantaneous logic by calling the static classifier as its first step. After that, it adds a wrapper that makes the final label less jumpy and more aware of whether the system is briefly wobbling, steadily recovering, or staying in a bad regime for long enough to count as sustained collapse.

In simple terms, the static classifier is a snapshot reader, while the temporal classifier is a snapshot reader with short-term memory and caution layers on top.

## What stays the same as the static classifier

The temporal classifier inherits the same label set as the static classifier: Apoptosis, Proliferation, Migration, Quiescence, Diversification, and Undetermined. It also inherits the same dynamic-first interpretation of `C`, `T`, `E`, `O` and their derivatives, the same contextual parameter subset, the same contextual gates, and the same ordered rule-based philosophy at the instantaneous level.

It first calls the existing static classifier:

```python
classify_static_state(C, T, E, O, dC, dT, dE, dO, bounds, par=None, scenario_cfg=None)
```

This means the temporal classifier is not a new taxonomy definition. The biological meaning of the labels still comes from the static classifier.

## What the temporal layer adds

The temporal layer adds a small set of extra features on top of the static label.

### 1. Short memory

For each parameter/scenario signature, the temporal classifier stores a `TemporalMemory` object in an internal cache. That memory keeps the last temporal label, the last static base label, a dwell count, a short recent history of temporal labels, a short recent history of static labels, and counters for how many consecutive previous steps were inside or outside viability bounds.

This memory is there to stabilize interpretation over time, not to create a hidden-state model.

### 2. Viability tracking

The temporal wrapper separately checks whether the current point is inside the viability region. That check uses the bounds on `C`, `T`, `E`, and `O`, but it is intentionally kept separate from taxonomy definition.

In other words, viability is not used here to say “this is Migration” or “this is Apoptosis.” It is used only to detect whether the system has been staying outside the admissible region for several steps in a row.

### 3. Recovery detection

The temporal classifier computes a recovery score that asks whether an out-of-bounds variable is moving back toward the viable region. For example, recovery increases if `C` is too low but `dC > 0`, if `T` is too high but `dT < 0`, or if `O` is too low but `dO > 0`.

This score does not define labels on its own. It acts as a caution signal telling the wrapper that the current bad-looking state may actually be on the way back.

### 4. Grace period for brief excursions

If the current point is outside viability bounds, but the static classifier does not say Apoptosis and the system has only recently moved outside while already showing recovery, the temporal wrapper temporarily changes the candidate label to `Undetermined`.

This feature stops the classifier from overreacting to short-lived excursions.

### 5. Recovery-aware caution around apoptosis

If the static classifier says `Apoptosis`, the temporal wrapper does not always accept that immediately. If the state is actually inside viability bounds, or if it has been outside for only a short time and recovery is already present, the wrapper changes the candidate label to `Undetermined` instead of committing straight away to apoptosis.

This is one of the main ways the temporal version becomes more conservative than the static one during transient collapse-like episodes.

### 6. Slower switching between labels

If the new candidate label differs from the previous temporal label, the wrapper does not automatically switch. It checks recent temporal history and only accepts the new label if there is enough recent support.

If that support is missing, the wrapper keeps the previous non-Undetermined temporal label, or falls back to the candidate if there was no meaningful previous label. This is the main anti-flicker mechanism.

### 7. Sticky sustained apoptosis

Once the temporal label is already `Apoptosis`, the wrapper makes that state harder to leave than an ordinary label. If the system is still outside viability bounds or recovery is still weak, the final label remains `Apoptosis`.

Only when the point is back inside viability and recovery is strong enough does the wrapper release that sticky terminal state, and even then it releases it to `Undetermined`, not directly to a positive label. This reflects the idea that sustained collapse should be harder to reverse at the label level than ordinary mode switching.

## Decision workflow

The temporal classifier follows a simple sequence.

1. Compute the base label using the static classifier.
2. Check whether the current point is inside viability bounds.
3. Compute the recovery score.
4. Adjust the candidate label using temporal rules: grace period, recovery-aware caution, slower switching, and sticky apoptosis.
5. Update temporal memory.
6. Return the final temporal label.

A compact pseudocode view is:

```python
base_label = classify_static_state(...)
inside = inside_viability(...)
recovery = recovery_score(...)

candidate = base_label

if outside_briefly and recovering and base_label != "Apoptosis":
    candidate = "Undetermined"

if base_label == "Apoptosis" and (inside or recovering_too_early):
    candidate = "Undetermined"

if last_label == "Apoptosis":
    final = keep_or_release_apoptosis_based_on_inside_and_recovery(...)
else:
    final = switch_only_if_recent_history_supports_candidate(...)

update_memory()
return final
```

This pseudocode is only a simplification, but it matches the structure of the actual implementation.

## Scope of the classifier

Each call to the temporal classifier returns the label for the current state. The key difference from the static classifier is that the current call is interpreted in light of short retained memory from previous steps.

That means the temporal classifier is local-and-sequential rather than trajectory-aggregating. It reads the current point, consults its short memory, updates that memory, and returns the current temporal label.

## Relationship to viability

Taxonomy and viability are still separate concepts in the temporal implementation. Taxonomy answers what biological mode the system most resembles, while viability answers whether the state is inside the allowed region defined by the bounds.

The temporal classifier preserves that separation. Viability is used only as a timing and confidence signal for smoothing transitions, delaying premature conclusions, and deciding whether sustained apoptosis should remain sticky.

## Important implementation details

A few details are worth stating explicitly because they matter for a precise reading of the code.

- Memory is cached per inferred signature, built from scenario `p`, scenario `label`, and the contents of `par`, rather than from every possible scenario field.
- Switching support is implemented through recent-label counting in temporal history, not through a separate probabilistic confidence model.
- Leaving sticky apoptosis requires being back inside viability and having sufficiently strong recovery.
- The temporal wrapper uses only explicit sequential memory and rule-based persistence checks; it does not contain built-in whole-trajectory summarization logic.

These details are compatible with the simple framing above, but they are also useful when documenting the implementation faithfully.

## Recommended interpretation

The best short description is:

> The temporal taxonomy classifier is the static classifier plus a small memory-and-caution layer.

The static classifier decides what the current point looks like biologically. The temporal layer then decides whether that label should be trusted immediately, softened to `Undetermined`, held for a bit longer, or made harder to reverse because the recent sequence suggests a transient wobble, an ongoing recovery, or sustained collapse.
