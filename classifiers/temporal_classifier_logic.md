# Temporal Classifier Logic

## Overview

The temporal classifier is a hybrid decision system. It combines a pointwise phenotype scorer with short-term memory, recovery-aware boundary logic, transition constraints, and a secondary layer of static-parameter modulation. The result is a classifier that still reacts to the current values of `C`, `T`, `E`, `O` and their derivatives, but no longer treats each time point as fully independent.

The classifier outputs one of six labels:

- Apoptosis
- Migration
- Proliferation
- Quiescence
- Diversification
- Undetermined

Its decision process can be understood as two stacked levels:

1. An instantaneous scoring layer that evaluates the current state.
2. A temporal arbitration layer that decides whether the currently favored state is strong and persistent enough to become the emitted label.

In the revised implementation, the instantaneous layer itself has two subparts:

- primary dynamic and geometric scoring from the current state and derivatives;
- secondary score modulation from static parameters and scenario metadata.

## Inputs

For each time point, the classifier receives:

- `C`, `T`, `E`, `O`
- `dC`, `dT`, `dE`, `dO`
- `bounds`
- optional `par`
- optional `scenario_cfg`

The four state variables define the current biological system state. The four derivatives describe the local direction of motion in state space. The bounds define the viability box. The optional parameter dictionaries provide static contextual information.

## High-level structure

The classifier executes the following conceptual pipeline:

1. Identify the memory context from static parameters and scenario metadata.
2. Determine whether the current point lies inside the viability region.
3. Update temporal counters for time spent inside or outside viability.
4. Compute a recovery score based on whether out-of-range variables are moving back toward viability.
5. Compute raw instantaneous scores for all six labels from the current state and local dynamics.
6. Apply static-parameter and scenario-based score modifiers to those raw scores.
7. Select the highest-scoring provisional label.
8. Apply temporal safety rules, especially near viability boundaries.
9. Apply transition restrictions and hysteresis.
10. Emit the final label and update memory.

## The six labels

### Apoptosis

Represents persistent failure, severe stress, or nonviability. It is designed to be conservative both when entered and when exited.

### Migration

Represents a motile, high-tension, actively moving or remodeling state. It is favored when tension and matrix-related dynamics suggest directed activity.

### Proliferation

Represents growth-supportive dynamics inside a viable region. It is favored when the state is balanced, supported, and trending upward in constructive variables.

### Quiescence

Represents a viable but dynamically quiet regime. It is favored when derivatives are small and the system sits in a stable part of the viable box.

### Diversification

Represents remodeling, branching, or structurally changing behavior. It is favored when the matrix and tension are evolving in a viable environment without mapping cleanly to pure migration or pure growth.

### Undetermined

Represents ambiguity, transitional behavior, weak evidence, buffering after perturbation, or deliberately delayed commitment. It acts as a neutral safe state.

## Memory context and signature

The classifier does not keep one global state for all trajectories. Instead, it builds a memory key from:

- the scenario label;
- the scenario parameter `p`;
- the sorted contents of the static parameter dictionary.

This means temporal memory is partitioned by scenario and parameter context. Two trajectories that differ in scenario identity or static parameter values do not share the same short-term classification memory.

This memory context is important, but it is not the same thing as score influence. The current revised version uses static parameters in two ways:

- to define memory separation;
- to modulate the biological scores themselves.

## Internal memory state

Each memory context stores:

- `last_label`
- `dwell_count`
- `outside_count`
- `inside_count`
- `history`

### `last_label`

The most recent emitted label.

### `dwell_count`

How many consecutive classification steps the current emitted label has persisted.

### `outside_count`

How many consecutive steps the trajectory has remained outside the viability region.

### `inside_count`

How many consecutive steps the trajectory has remained inside the viability region.

### `history`

A short deque of recent candidate labels. This stores proposed state changes, even when they are not yet accepted as final.

## Viability logic

A point is viable if all of the following are true:

- `C` is above its lower bound;
- `T` is between its lower and upper bounds;
- `E` is between its lower and upper bounds;
- `O` is above its lower bound.

If all conditions hold, the point is inside viability. Otherwise it is outside viability.

This binary result influences the classifier in multiple ways:

- it contributes directly to some class scores;
- it updates the inside and outside counters;
- it affects grace periods;
- it affects apoptosis entry and exit rules.

## Recovery score

The recovery score measures whether the current trajectory is moving back toward viability.

A variable contributes to recovery when it is currently on the wrong side of a viability threshold but its derivative points inward:

- low `C` with positive `dC`;
- low `T` with positive `dT`;
- high `T` with negative `dT`;
- low `E` with positive `dE`;
- high `E` with negative `dE`;
- low `O` with positive `dO`.

Each such recovery-consistent condition contributes one unit. The recovery score is therefore a directional measure rather than a distance measure.

This score helps distinguish active correction from passive deterioration.

## Instantaneous scoring: primary layer

The primary instantaneous layer computes raw scores for all six labels using only current state geometry and local derivatives.

### Geometric reference values

The classifier derives these helper quantities from the viability bounds:

- `T_mid`, midpoint of the viable tension interval;
- `T_span`, width of the viable tension interval;
- `E_mid`, midpoint of the viable matrix interval;
- `E_span`, width of the viable matrix interval.

These quantities define what it means for `T` and `E` to be central, elevated, depressed, or extreme.

### Stress decomposition

The classifier computes six nonnegative stress components:

- `low_C`
- `low_O`
- `low_T`
- `high_T`
- `low_E`
- `high_E`

Each component is zero inside the allowed interval and increases continuously as the variable moves outside its viability range. Their sum is the total stress level.

### Apoptosis raw score

Apoptosis increases with overall stress. It is further increased when:

- `dC` is negative;
- `dO` is negative;
- the point lies outside viability.

The intent is to favor apoptosis when the system is deprived, decaying, or structurally failing.

### Migration raw score

Migration measures a high-activity motile regime. It increases when:

- `T` is above its midpoint;
- `O` and `C` are sufficient;
- `dT` is positive;
- `dE` is positive;
- `dC` is positive, with weaker weight;
- `T` is elevated above the viable midpoint;
- `E` is actively increasing.

This score emphasizes dynamic motion-related behavior.

### Proliferation raw score

Proliferation is only favored in viable conditions. It increases when:

- the point is inside viability;
- oxygen and `C` are sufficient;
- `T` lies in the viable interval;
- `dC`, `dE`, or `dO` are positive;
- `T` is near the midpoint of the viable tension interval;
- `E` is near the midpoint of the viable matrix interval.

This score favors balanced, constructive growth.

### Quiescence raw score

Quiescence increases when:

- the point is inside viability;
- the total derivative magnitude is small;
- `T` is close to its viable midpoint;
- all four derivatives are very small simultaneously.

The derivative term is passed through an exponential decay, so quiescence rises as the dynamics become quieter.

### Diversification raw score

Diversification increases when:

- the point is inside viability;
- `E` is at or above the midpoint of the viable matrix interval;
- oxygen is sufficient;
- matrix dynamics are active;
- tension dynamics are also active;
- `dE` is positive while `T` is changing meaningfully.

This creates a score for structured remodeling rather than stillness or simple growth.

### Undetermined raw score

Undetermined starts from a baseline and gains strength when:

- the point is outside viability;
- no other label has become decisively supported.

This allows the classifier to preserve ambiguity instead of overcommitting.

## Instantaneous scoring: secondary parameter layer

The updated classifier now adds a second instantaneous layer that uses static parameters and scenario metadata as score modifiers.

This does not replace the biological dynamic logic. Instead, it rescales the raw class scores after they are computed. The purpose is to let static context bias the classification toward some phenotypes and away from others while keeping the temporal machinery intact.

## Parameter extraction strategy

The classifier reads static signals from `par` and `scenario_cfg` using a tolerant lookup scheme.

From `scenario_cfg`, it reads:

- `p`
- `expected`

From `par`, it attempts to read these biological tendencies, with fallbacks for alternate names:

- adhesion, falling back to `alpha`
- motility, falling back to `m`
- growth, falling back to `g`
- oxygen support, falling back to `o`
- remodeling, falling back to `r`
- stress, falling back to `s`

If a key is missing, a neutral default of `1.0` is used.

This makes the classifier robust to slightly different parameter naming conventions.

## Meaning of the static modifiers

The secondary parameter layer constructs multiplicative modifiers for each output class.

### Migration modifier

Migration is increased by:

- motility gain above neutral;
- positive scenario parameter `p`.

This means a system configured for stronger motility or more migration-favoring scenario structure will amplify the migration score.

### Proliferation modifier

Proliferation is increased by:

- growth support above neutral;
- oxygen support above neutral;
- adhesion above neutral.

This reflects the idea that a more growth-supportive and better-supported static environment should bias the classifier toward proliferative interpretations when the dynamics are compatible.

### Quiescence modifier

Quiescence is increased by stronger adhesion and mildly decreased by higher motility.

This means quiescence becomes more likely in statically stabilizing environments and less likely in environments predisposed toward movement.

### Diversification modifier

Diversification is increased by stronger remodeling support and also mildly by larger `p`.

This gives structural remodeling parameters a direct role in phenotype interpretation.

### Apoptosis modifier

Apoptosis is increased by stronger stress support and mildly decreased by stronger oxygen support.

This allows the same dynamic state to be interpreted as more or less failure-prone depending on static system fragility or resilience.

### Undetermined modifier

Undetermined is not heavily shaped by the static parameter dictionary itself, but it can still be affected by scenario expectations.

## Scenario-expectation modifiers

In addition to generic static parameters, the classifier uses the scenario’s expected regime as a weak contextual prior.

### Unstable scenarios

If `expected` is `unstable`, the classifier:

- boosts apoptosis;
- mildly boosts undetermined;
- slightly suppresses quiescence;
- slightly suppresses proliferation.

This makes unstable scenarios more failure-prone and less likely to settle too easily into calm labels.

### Boundary or borderline scenarios

If `expected` is `boundary` or `borderline`, the classifier:

- boosts undetermined;
- mildly boosts diversification;
- mildly boosts apoptosis.

This encourages ambiguity and transitional interpretations near regime boundaries.

### Stable scenarios

If `expected` is `stable`, the classifier:

- boosts quiescence;
- boosts proliferation;
- suppresses apoptosis.

This gives stable scenarios a contextual prior toward viability-preserving states.

## Why multiplicative modulation is used

The static parameter layer acts multiplicatively rather than additively. This has three advantages:

- it preserves the relative structure of the dynamic score layer;
- it does not manufacture a phenotype from zero evidence;
- it lets static context strengthen or weaken an already supported interpretation.

In other words, static parameters tilt the competition among labels rather than replacing the biological evidence from the current state.

## Candidate selection

After primary raw scores are computed and static modifiers are applied, the classifier chooses the highest-scoring label as the provisional candidate.

This candidate is still not final. The temporal arbitration layer can replace it.

## Grace rules near boundaries

The classifier includes explicit grace rules to avoid premature commitment after brief exits from the viability region.

Two rules are used.

### First grace rule

If the point is outside viability, the previous final label is `Undetermined`, and the current label has not persisted long enough, the candidate is forced to `Undetermined`.

This avoids immediate overinterpretation at the beginning of a trajectory or just after a disturbance.

### Second grace rule

If the point is outside viability for only a short time and the recovery score is already positive, the candidate is also forced to `Undetermined`.

This means a brief boundary excursion with inward-pointing dynamics is interpreted as unresolved rather than failed.

## Conservative apoptosis entry

Even if apoptosis wins the score competition, it still must satisfy extra temporal rules.

If the candidate is apoptosis:

- and the point is inside viability, apoptosis is rejected;
- or if the point is outside viability but not for long enough and recovery is present, apoptosis is downgraded to `Undetermined`.

So apoptosis is never entered on one alarming point alone.

## Sticky apoptosis exit

Once the last emitted label is apoptosis, the classifier becomes very conservative about leaving it.

It remains in apoptosis unless both conditions hold:

- the current point is inside viability;
- the recovery score is sufficiently strong.

Even then, it does not jump directly into a specific healthy phenotype. It first returns to `Undetermined`.

This creates a biologically cautious buffer between failure and recovery.

## Allowed transition graph

For all non-apoptotic states, the classifier checks whether a proposed transition is allowed from the previous final label.

The transition map is intentionally structured.

- `Undetermined` may move to any label.
- `Quiescence` may move to migration, proliferation, diversification, undetermined, or apoptosis.
- `Migration` may move to diversification, quiescence, undetermined, or apoptosis.
- `Proliferation` may move to diversification, quiescence, undetermined, or apoptosis.
- `Diversification` may move to migration, proliferation, quiescence, undetermined, or apoptosis.
- `Apoptosis` may only remain apoptosis or move to undetermined.

If a candidate transition is forbidden, the candidate is replaced by the previous final label.

## Hysteresis and persistence

Even allowed transitions must persist before they are accepted.

If the candidate differs from the previous final label, the classifier counts how often that candidate appears in recent candidate history.

- If the candidate appears often enough, the switch is accepted.
- Otherwise, the classifier keeps the previous final label.

This creates hysteresis: regime changes need repeated evidence rather than a single fluctuation.

## Final memory update

After the final label has been determined, memory is updated.

- If the final label matches the previous final label, `dwell_count` increases.
- Otherwise `dwell_count` resets to one.
- `last_label` is updated.
- The provisional candidate is appended to the history deque.

The next classification call will interpret the state through this updated context.

## Why the history stores candidates

The history buffer stores provisional candidates rather than only final labels. This is a crucial design choice.

A label can repeatedly win the instantaneous competition but still be blocked temporarily by hysteresis. By storing candidates, the classifier allows a new regime to accumulate evidence across time until the switch threshold is met.

Without this mechanism, the classifier would become too inert and transitions would be excessively delayed.

## Interaction between dynamic and static layers

The revised classifier should be interpreted as a dynamic scorer modulated by static context.

- The dynamic layer asks: what phenotype is supported by the current state and local direction of change?
- The static layer asks: given the scenario and parameter regime, which of those biologically plausible phenotypes should be comparatively more favored?
- The temporal layer asks: has that interpretation persisted long enough to be trusted?

This three-level architecture is the core logic of the revised classifier.

## Practical interpretation

A label emitted by this classifier means more than a pointwise threshold crossing.

- `Apoptosis` means sustained nonviability, reinforced by temporal persistence and insufficient recovery.
- `Migration` means dynamic motility-compatible behavior, possibly strengthened by motility-related static parameters.
- `Proliferation` means viable growth-compatible behavior, possibly strengthened by growth, oxygen, and adhesion support.
- `Quiescence` means stable viable behavior, especially in statically stabilizing contexts.
- `Diversification` means viable remodeling behavior, especially when remodeling-support parameters are strong.
- `Undetermined` means ambiguity, early recovery, boundary behavior, or temporal buffering.

## Reset behavior

The classifier exposes `reset_classifier_memory()` for situations where a new independent trajectory or batch should not inherit temporal context from a previous one.

This reset should be called whenever continuity across trajectories is not intended.

## Decision flow summary

For each time point, the classifier performs this sequence:

1. Build the scenario-parameter signature.
2. Retrieve the corresponding memory state.
3. Determine whether the current point lies inside viability.
4. Update inside and outside counters.
5. Compute the recovery score.
6. Compute raw dynamic scores for all labels.
7. Compute static and scenario-based score modifiers.
8. Multiply raw scores by those modifiers.
9. Select the provisional candidate.
10. Apply outside-boundary grace rules.
11. Apply apoptosis-entry safeguards.
12. Apply sticky apoptosis-exit logic.
13. Enforce the allowed transition graph.
14. Apply hysteresis using recent candidate history.
15. Emit the final label.
16. Update memory.

## Conceptual summary

The revised temporal classifier is not simply a temporal smoother and not simply a rule-based scorer. It is a three-part decision architecture:

- a primary biological scoring layer based on current state and derivatives;
- a secondary contextual modulation layer based on static parameters and expected scenario regime;
- a temporal arbitration layer based on persistence, recovery, hysteresis, and transition structure.

That combination allows the classifier to remain sensitive to current dynamics, aware of static context, and resistant to noisy label flicker.
