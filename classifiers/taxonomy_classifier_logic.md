# Taxonomy classifier design rules

This document describes the current biological taxonomy classifier used to annotate scaffold-cell-matrix simulations. The classifier assigns one of six labels — Apoptosis, Migration, Proliferation, Quiescence, Diversification, or Undetermined — from the instantaneous system state, its local derivatives, and a deliberately limited subset of interpretable static parameters.

The classifier is designed to be readable, biologically interpretable, and auditable. It is not a machine-learned model, not a hidden-state model, and not a temporal Markov process. It is an ordered rule-based classifier that reads what the system is doing at a given point in time and uses static context only through explicit, named gates.

## Classifier philosophy

The implementation follows three principles.

1. **Dynamic-first evidence**. The current state variables `C`, `T`, `E`, and `O`, together with their local derivatives `dC`, `dT`, `dE`, and `dO`, provide the primary biological signal.
2. **Context-aware refinement**. A small subset of static parameters is used to define biologically meaningful contextual tendencies such as oxygen support, matrix drive, or mechanical damping.
3. **Ordered first-match logic**. Labels are evaluated in a fixed order, and the first satisfied rule determines the output label.

This means the classifier is dynamic-first, but not dynamic-only. Static parameters do not appear merely at the end as tie-breakers; they are embedded explicitly inside several label rules where they help decide whether a dynamic pattern is biologically plausible.

## Inputs to the classifier

The main classifier interface is:

```python
classify_state(C, T, E, O, dC, dT, dE, dO, bounds, par=None, scenario_cfg=None)
```

The inputs have the following roles:

- `C`: curvature-like structural state
- `T`: cytoskeletal tension
- `E`: ECM density
- `O`: oxygen availability
- `dC`, `dT`, `dE`, `dO`: local time derivatives of those variables
- `bounds`: viability-style biological reference thresholds
- `par`: base model parameter dictionary
- `scenario_cfg`: scenario-specific parameter overrides and porosity

The classifier uses the state variables and derivatives as the main phenotype evidence. It uses the bounds as biological reference ranges, and it uses the effective parameter context to decide whether a given pattern should be interpreted as collapse-like, growth-like, remodeling-like, quiet, or branching-like.

## Static parameters used explicitly

The classifier does not use the full ODE parameter vector. It uses only the following interpretable subset:

- `p`
- `beta`
- `eta`
- `kappa`
- `mu`
- `delta_T`
- `delta_E`
- `delta_O`
- `rho`
- `s`

These parameters are used because each has a direct interpretation that maps reasonably well onto phenotype-like labels. The goal is to keep the classifier explainable rather than making it depend on every parameter in the dynamical system.

## Effective parameter context

Before any label is assigned, the classifier builds an effective parameter context.

1. Start from `par`.
2. Merge `scenario_cfg["param_overrides"]` on top when available.
3. If `scenario_cfg["p"]` is present, overwrite `p` with that value.

This ensures that the classifier always evaluates the current scenario-specific regime rather than the base parameter set alone.

## Derived contextual axes

The classifier constructs a small set of explicit helper quantities from the interpretable parameter subset:

- `oxygen_supply = rho * s * p`
- `tension_drive = beta`
- `tension_damping = eta`
- `matrix_drive = kappa`
- `oxygen_burden = mu`
- `decay_burden = delta_T + delta_E + delta_O`

These are not latent states. They are readable composite quantities used to make the rules biologically explicit.

## Boundary-awareness logic

The classifier uses helper functions to detect whether a variable is near a lower or upper reference threshold.

- `_near_lower(x, lo, frac, abs_pad)`
- `_near_upper(x, hi, frac, abs_pad)`

These helpers are used to define the following flags:

- `near_C_low = _near_lower(C, C_min, frac=0.20, abs_pad=0.03)`
- `near_T_low = _near_lower(T, T_min, frac=0.25, abs_pad=0.05)`
- `near_T_high = _near_upper(T, T_max, frac=0.10, abs_pad=0.08)`
- `near_E_low = _near_lower(E, E_min, frac=0.50, abs_pad=0.05)`
- `near_E_high = _near_upper(E, E_max, frac=0.10, abs_pad=0.12)`
- `near_O_low = _near_lower(O, O_min, frac=0.25, abs_pad=0.05)`

This design lets the classifier distinguish between clearly safe values, clearly pathological values, and biologically suspicious near-boundary values.

## Context gates used in the rules

The current implementation turns the contextual axes into explicit boolean gates:

- `low_oxygen_supply = oxygen_supply < 0.75`
- `high_oxygen_burden = oxygen_burden > 1.0`
- `strong_tension_drive = tension_drive > 2.2`
- `weak_tension_drive = tension_drive < 1.25`
- `strong_tension_damping = tension_damping > 1.0`
- `strong_matrix_drive = matrix_drive > 1.35`
- `high_decay_burden = decay_burden > 2.35`

These gates are part of the actual classifier logic and should be treated as implementation-level thresholds rather than vague narrative descriptions.

## Decision order

The classifier is evaluated in the following order:

1. Apoptosis
2. Proliferation
3. Migration
4. Quiescence
5. Diversification
6. Undetermined

This ordering matters. The classifier is a first-match system, so if a state satisfies an earlier label block, later blocks are not evaluated.

## Dynamic-first pseudocode

The implementation is most easily understood as a dynamic-first rule engine with explicit contextual gates inside the label checks. A simplified version of that logic is:

```python
if strong_dynamic_signature_for_apoptosis:
    return "Apoptosis"
elif strong_dynamic_signature_for_proliferation:
    return "Proliferation"
elif strong_dynamic_signature_for_migration:
    return "Migration"
elif strong_dynamic_signature_for_quiescence:
    return "Quiescence"
elif strong_dynamic_signature_for_diversification:
    return "Diversification"
else:
    use_static_context_to_bias_ambiguous_case()
    if still_ambiguous:
        return "Undetermined"
```

This pseudocode is intentionally schematic. In the actual implementation, static context is not applied only at the very end; it is embedded inside several of the rule blocks themselves, but the pseudocode still captures the intended precedence and the dynamic-first reading of the classifier.

## State-label determination rules

The table below summarizes the implemented rules used to assign each label.

| Label | Primary dynamic signature | Contextual gates or supporting conditions | Return condition |
|---|---|---|---|
| **Apoptosis** | `O < O_min` or `C < C_min`; or `near_O_low` with `dO < -0.02`; or `near_C_low` with `dC < -0.02` | Collapse support from `dO < 0`, `dC < 0`, `high_oxygen_burden`, `low_oxygen_supply`, `high_decay_burden`, or nonconstructive `dE <= 0` / `dT <= 0` | Return first if any apoptosis rule matches |
| **Proliferation** | Viable oxygen and matrix gain: `O > max(0.35, O_min + 0.10)` and `dE > 0.03`; or `near_E_high` with `dE >= 0` | `strong_matrix_drive`, or supportive tension with `T > max(0.30, T_min + 0.10)` and `dT >= -0.02` | Return if no apoptosis rule matched and a proliferation rule matches |
| **Migration** | Viable remodeling: `O > O_min`, `T >= T_min`, `dE < -0.02`; or broader ECM-loss pattern with `E > E_min`, `dE < 0`, `dT <= 0.03`, `dO >= -0.02`, `not near_O_low` | Stable or damped tension via `abs(dT) < 0.05` or `strong_tension_damping and not strong_tension_drive` | Return if earlier labels did not match and a migration rule matches |
| **Quiescence** | Viable moderate state with small derivatives: `T_min <= T <= 0.75*T_max`, `E_min <= E <= 0.70*E_max`, `O > O_min + 0.08`, `abs(dT) < 0.03`, `abs(dE) < 0.02`, `abs(dO) < 0.03` | Low-drive alternative: `near_T_low`, `weak_tension_drive`, `dT <= 0`, `dE <= 0.02`, `O > O_min` | Return if earlier labels did not match and a quiescence rule matches |
| **Diversification** | Active non-terminal constructive change with `O > O_min` and `C > C_min`, plus one of: `dT > 0.04 and dE > 0`, `near_T_high and dE >= 0`, or `dC > 0.02 and dE > 0` | Alternative contextual branch: `strong_tension_drive`, `not (T > T_max)`, `dT >= 0`, `dE >= -0.01` | Return if earlier labels did not match and a diversification rule matches |
| **Undetermined** | Mixed, weak, boundary-adjacent, or unmatched dynamics | `near_T_high`, `near_E_high`, `near_O_low`, `near_E_low`, or `near_C_low`; otherwise global fallback | Return when no earlier state rule matched |

## Per-label explanation

### Apoptosis

Apoptosis is intended to capture collapse-like or terminal behavior. The strongest evidence is low oxygen, collapsing curvature, worsening oxygen, or strong depletion burden. In the implementation, apoptosis is evaluated first so that clearly terminal patterns take precedence over productive or remodeling interpretations.

### Proliferation

Proliferation captures viable, constructive matrix-building behavior. The classifier looks primarily for adequate oxygen together with positive ECM growth, then uses matrix-drive or non-collapsing tension to support the interpretation.

### Migration

Migration captures viable ECM remodeling or matrix loss without immediate collapse. The most important signal is `dE < 0` under still-viable oxygen and tension, especially when tension remains stable or is strongly damped.

### Quiescence

Quiescence captures quiet, viable, low-activity states. The main signature is a moderate operating range together with small derivatives. A weak-tension-drive alternative also supports quiescence when the system sits near low tension without showing signs of collapse.

### Diversification

Diversification captures non-terminal, non-quiescent activity suggestive of directional commitment or branching. The classifier looks for rising tension and constructive ECM behavior, or for positive curvature-plus-ECM growth patterns, with strong tension drive as an alternative contextual route.

### Undetermined

Undetermined is the ambiguity label and the global fallback. It is used when the state sits near problematic boundaries, when the evidence is mixed, or when none of the earlier state definitions fits cleanly.

## Relationship to viability

Taxonomy and viability are related but not identical.

Viability asks whether a point or a trajectory stays inside a predefined admissible region defined by thresholds such as `C_min`, `T_min`, `T_max`, `E_min`, `E_max`, and `O_min`. Taxonomy asks what kind of biological mode the system appears to be expressing.

The taxonomy classifier does use these thresholds and their near-boundary neighborhoods as biological reference values. However, it does not produce viability reports and should not be interpreted as a direct viability classifier. The same reference bounds help anchor both systems, but the outputs answer different questions.

## Instantaneous and trajectory-level use

The central classifier is instantaneous. It takes one sampled point and returns one label. It has no built-in memory of previous states.

Trajectory-level labeling is handled separately by:

```python
classify_solution(sol, bounds, par=None, scenario_cfg=None, n_samples=7)
```

This helper:

1. Computes local derivatives from the trajectory.
2. Selects a fixed number of evenly spaced sample indices.
3. Classifies each sampled point independently with `classify_state`.
4. Returns the majority label across those sampled points.

This is a sampling-and-voting strategy, not temporal smoothing and not a hidden-state model.

## Practical coding guidance

The implementation should continue to follow these constraints:

- Keep `classify_state(...)` as the central interface.
- Use derivatives as part of the primary evidence, not as optional decoration.
- Merge scenario overrides before extracting contextual axes.
- Keep the static parameter subset explicit in `INTERPRETABLE_STATIC_PARAMS`.
- Document helper thresholds and contextual gates in the code and in the accompanying design document.
- Preserve the ordered first-match rule structure so rule precedence stays auditable.
- Keep taxonomy labels separate from viability reports.

## Recommended interpretation

The classifier should be understood as an interpretable biological annotation layer built on top of the dynamical system. Its job is to read what the system is doing locally, then use a limited amount of mechanistic context to resolve whether that local behavior is better interpreted as collapse, growth, remodeling, quiescence, branching, or ambiguity.

That balance — dynamic-first evidence, explicit contextual gates, and auditable rule order — is the central design philosophy of the current implementation.
