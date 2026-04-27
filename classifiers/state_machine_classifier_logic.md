# Deterministic State Machine Classifier

## Overview

This classifier is a deterministic finite-state machine built on the scaffolding of the temporal classifier. It keeps the same state set, the same viability logic, the same recovery logic, the same instantaneous biological score layer, the same static-parameter modulation layer, and the same transition graph and hysteresis structure. The new part is that it adds explicit deterministic transition biases that act on top of the temporal classifier before the final transition arbitration is resolved.

The result is no longer a purely temporal score smoother. It becomes an explicit state machine with memory, persistence, deterministic bias rules, and controlled transition behavior.

The six states are:

- Apoptosis
- Migration
- Proliferation
- Quiescence
- Diversification
- Undetermined

## What is inherited from the temporal classifier

The deterministic state machine inherits the following components directly from the temporal classifier design:

- viability-box membership check;
- recovery score from inward-pointing derivatives;
- primary instantaneous score layer based on `C`, `T`, `E`, `O`, `dC`, `dT`, `dE`, `dO`;
- secondary static-parameter modulation from `par` and `scenario_cfg`;
- inside and outside counters;
- sticky apoptosis behavior;
- transition constraints among the six labels;
- hysteresis via recent candidate history.

That means the state machine does not replace the temporal classifier logic. It extends it.

## What is new compared with the temporal classifier

The new state machine introduces deterministic transition biases on top of the temporal score scaffold.

These are the genuinely new mechanisms:

- ambiguity-driven biasing;
- recovery-streak biasing;
- stress-streak biasing;
- state-retention biases tied to static parameters;
- scenario-dependent persistence biases.

These biases are applied after the instantaneous dynamic scores and static parameter modifiers have been computed, but before the candidate state is finalized.

## Core design principle

The temporal classifier mainly asks whether a state is currently supported and whether that support is persistent enough to survive hysteresis.

The deterministic state machine goes one step further. It asks whether the system should be nudged toward or away from certain transitions based on explicit rule-based trajectory context.

In other words:

- the temporal classifier scores states;
- the state machine biases transitions.

## Memory structure

The state machine keeps all memory variables from the temporal classifier and adds three more:

- `recovery_streak`
- `stress_streak`
- `ambiguity_streak`

### `recovery_streak`

Counts how many consecutive steps the system has been inside viability with a sufficiently strong recovery signal.

### `stress_streak`

Counts how many consecutive steps the system has been outside viability with no recovery evidence.

### `ambiguity_streak`

Counts how many consecutive steps the score competition among labels has been too close to call clearly.

These extra memory variables are the main new deterministic state-machine mechanism.

## Base score layer

The base score layer is unchanged in structure relative to the temporal classifier.

It still computes:

- apoptosis from stress and decaying variables;
- migration from elevated tension and forward dynamic activity;
- proliferation from viable balanced growth;
- quiescence from viable low-dynamics stability;
- diversification from viable remodeling activity;
- undetermined from ambiguity and outside-viability buffering.

## Static-parameter layer

The static-parameter layer is also preserved. It still uses:

- adhesion or `alpha`;
- motility or `m`;
- growth or `g`;
- oxygen support or `o`;
- remodeling or `r`;
- stress or `s`;
- scenario parameter `p`;
- scenario expected regime.

These quantities multiplicatively modulate the base scores before state-machine-specific biases are applied.

## New mechanism 1: ambiguity bias

The classifier first measures the gap between the highest and second-highest score.

If this top-gap is small, the system is interpreted as ambiguous. In that case:

- `Undetermined` receives a positive deterministic bias;
- the ambiguity streak counter increases.

If ambiguity persists for multiple consecutive steps, the `Undetermined` bias becomes even stronger.

### Why this is new

The temporal classifier already had a weak undetermined fallback, but it did not explicitly track persistent score ambiguity as its own memory process. The state machine does.

### Effect on transitions

This mechanism makes the system more reluctant to jump prematurely into a committed biological state when the evidence is indecisive for several steps in a row.

## New mechanism 2: recovery-streak bias

If the system is inside viability and the recovery score is sufficiently strong, the recovery streak increases.

When this recovery streak persists:

- Quiescence receives a positive bias;
- Proliferation receives a positive bias;
- Apoptosis receives a negative bias.

### Why this is new

The temporal classifier used recovery mainly as a protective condition near the boundary and as a gate for leaving apoptosis. The state machine upgrades recovery into an explicit trajectory-level transition bias.

### Effect on transitions

Sustained recovery gradually tilts the machine toward viable stable or constructive states, even before a transition is fully locked in.

## New mechanism 3: stress-streak bias

If the system is outside viability and has no recovery evidence, the stress streak increases.

When this stress streak persists:

- Apoptosis receives a strong positive bias;
- Undetermined receives a weaker positive bias.

### Why this is new

The temporal classifier already penalized nonviability, but it did not separately accumulate unresolved stress as a deterministic transition driver. The state machine does.

### Effect on transitions

Repeated failed states without signs of correction will increasingly push the system toward apoptosis, while still allowing a short unresolved buffer through `Undetermined`.

## New mechanism 4: state-retention biases from static parameters

The state machine adds state-retention biases based on both the previous label and the static parameter regime.

Examples:

- if the current label is Migration and motility support is high, Migration gets an extra positive bias;
- if the current label is Proliferation and growth support is high, Proliferation gets an extra positive bias;
- if the current label is Diversification and remodeling support is high, Diversification gets an extra positive bias.

### Why this is new

The temporal classifier already used static parameters to rescale scores, but it did not combine them with the previously occupied state to create deterministic retention tendencies. The state machine does.

### Effect on transitions

This creates path-dependent persistence. A state that is already active and well-supported by the static environment is harder to leave.

## New mechanism 5: scenario-dependent persistence biases

The state machine also adds deterministic context-sensitive biases based on the expected scenario regime.

### Unstable scenarios

If the scenario is unstable and outside-viability persistence is building up, apoptosis is biased upward.

### Boundary or borderline scenarios

If the scenario is boundary-like and the top score gap is small, Undetermined is biased upward and Diversification receives a mild boost.

### Stable scenarios

If the scenario is stable, the point is inside viability, and inside persistence is long enough, Quiescence and Proliferation are biased upward.

### Why this is new

The temporal classifier already had scenario-level score priors, but it did not combine them explicitly with temporal persistence conditions to drive transitions. The state machine does.

## Order of operations

For each time point, the deterministic state machine executes the following sequence:

1. Determine the memory key from `par` and `scenario_cfg`.
2. Retrieve the memory state.
3. Check whether the current point is inside viability.
4. Update inside and outside counters.
5. Compute the recovery score.
6. Compute the base instantaneous class scores.
7. Apply static-parameter multiplicative modifiers.
8. Compute deterministic transition biases.
9. Add these biases to the class scores.
10. Select the provisional candidate label.
11. Apply boundary grace rules.
12. Apply apoptosis-entry safeguards.
13. Apply sticky apoptosis exit rules.
14. Enforce the allowed transition graph.
15. Apply hysteresis.
16. Emit the final label.
17. Update all memory counters and streaks.

## Relation between biases and hysteresis

The new biases do not replace hysteresis. They operate before hysteresis.

That means the architecture now has two different stabilization mechanisms:

- **biases**, which shape which candidate is proposed;
- **hysteresis**, which decides whether the proposed change has enough recent support to be accepted.

This is one of the most important differences from the temporal classifier.

### Temporal classifier

The temporal classifier mainly stabilizes after the score competition has already produced a candidate.

### State machine classifier

The state machine classifier influences the competition itself before hysteresis is applied.

So the state machine is not just “more hysteretic.” It is biased upstream of hysteresis.

## Why this is still deterministic

Every output is determined by the current observable state, the static parameter context, and the finite memory variables. No random sampling, transition probabilities, or stochastic draws are used.

If the same input sequence is replayed from the same initial memory state, the output state sequence will be identical.

## Why this is a state machine

This model qualifies as a deterministic state machine because:

- it has a finite set of discrete states;
- it keeps an explicit internal memory state;
- it updates that memory by deterministic rules;
- its next emitted state depends on the current state, current observables, and explicit transition logic.

It is not a Markov chain because transition behavior depends on counters and streaks, not only on the current emitted label.

## Transition structure

The allowed transition structure is inherited from the temporal classifier.

- `Undetermined` can move to any state.
- `Quiescence` can remain quiescent or move to Migration, Proliferation, Diversification, Undetermined, or Apoptosis.
- `Migration` can remain migratory or move to Diversification, Quiescence, Undetermined, or Apoptosis.
- `Proliferation` can remain proliferative or move to Diversification, Quiescence, Undetermined, or Apoptosis.
- `Diversification` can remain diversified or move to Migration, Proliferation, Quiescence, Undetermined, or Apoptosis.
- `Apoptosis` can only remain Apoptosis or move to Undetermined.

The new deterministic biases act inside this graph. They do not add new edges. They alter which allowed edge becomes favored.

## Interpretation of the new behavior

The revised classifier should be interpreted as a rule-driven controller on top of the temporal scaffold.

- ambiguity pushes the machine toward neutral holding states;
- sustained recovery pushes it toward viable stabilizing or constructive states;
- sustained unresolved stress pushes it toward failure;
- state-environment agreement increases persistence in the current state;
- scenario context shapes how fast these tendencies accumulate.

## Main conceptual difference from the temporal classifier

The temporal classifier answers: “given the current evidence and recent history, which label should be emitted?”

The state machine classifier answers: “given the current evidence, recent history, and ongoing transition tendencies, which state transition should the machine commit to now?”

That is the core conceptual upgrade.

## Summary of what changed

Compared with the temporal classifier, the deterministic state machine adds:

- explicit ambiguity tracking;
- explicit recovery streak tracking;
- explicit stress streak tracking;
- retention biases tied to the current occupied state and static support parameters;
- scenario-dependent persistence biases;
- upstream transition shaping before hysteresis.

These additions make the classifier behave more like a designed state machine and less like a passive temporal smoother.
