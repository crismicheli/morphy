# Static taxonomy classifier

This document describes the biological taxonomy classifier used to annotate scaffold–cell–matrix simulations. The classifier assigns one of six labels — Apoptosis, Migration, Proliferation, Quiescence, Diversification, or Undetermined — from the instantaneous system state, its local derivatives, a deliberately limited subset of interpretable static parameters, and an explicit configuration of heuristic thresholds.

The classifier is designed to be readable, biologically interpretable, and auditable. It is not a machine‑learned model, not a hidden‑state model, and not a temporal Markov process. It is an ordered, rule‑based classifier that reads what the system is doing at a given point in time and uses static context and threshold configuration only through explicit, named gates.

## Classifier philosophy

The implementation follows three principles.

1. **Dynamic‑first evidence**  
   The current state variables `C`, `T`, `E`, and `O`, together with their local derivatives `dC`, `dT`, `dE`, and `dO`, provide the primary biological signal.

2. **Context‑aware refinement**  
   A small subset of static parameters is used to define biologically meaningful contextual tendencies such as oxygen support, matrix drive, or mechanical damping.

3. **Ordered first‑match logic with explicit thresholds**  
   Labels are evaluated in a fixed order via dedicated predicates, and the first satisfied rule determines the output label. All numerical thresholds used in these predicates are collected in a single configuration object, so calibration means adjusting that configuration rather than editing code.

The classifier is dynamic‑first, but not dynamic‑only. Static parameters and heuristic thresholds are embedded explicitly inside the label rules where they help decide whether a dynamic pattern is biologically plausible.

## Inputs to the classifier

The main classifier interface is:

```python
classify_state(
    C, T, E, O,
    dC, dT, dE, dO,
    bounds,
    par=None,
    scenario_cfg=None,
    cfg: Optional[StateHeuristics] = None,
)
```

The inputs have the following roles:

- `C`: curvature‑like structural state  
- `T`: cytoskeletal tension  
- `E`: ECM density  
- `O`: oxygen availability  
- `dC`, `dT`, `dE`, `dO`: local time derivatives of those variables  
- `bounds`: viability‑style biological reference thresholds (`C_min`, `T_min`, `T_max`, `E_min`, `E_max`, `O_min`)  
- `par`: base model parameter dictionary  
- `scenario_cfg`: scenario‑specific parameter overrides and porosity  
- `cfg`: optional `StateHeuristics` instance collecting all heuristic thresholds used in the rules  

The classifier uses the state variables and derivatives as the main phenotype evidence. It uses the bounds as biological reference ranges, and it uses the effective parameter context, together with `cfg`, to decide whether a given pattern should be interpreted as collapse‑like, growth‑like, remodeling‑like, quiet, or branching‑like.

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

These parameters are used because each has a direct interpretation that maps reasonably well onto phenotype‑like labels. The goal is to keep the classifier explainable rather than making it depend on every parameter in the dynamical system.

## Effective parameter context

Before any label is assigned, the classifier builds an effective parameter context.

1. Start from `par`.  
2. Merge `scenario_cfg["param_overrides"]` on top when available.  
3. If `scenario_cfg["p"]` is present, overwrite `p` with that value.

This ensures that the classifier always evaluates the current scenario‑specific regime rather than the base parameter set alone.

## Derived contextual axes

From the interpretable static parameters, the classifier constructs a small set of explicit helper quantities:

- `oxygen_supply = rho * s * p`  
- `tension_drive = beta`  
- `tension_damping = eta`  
- `matrix_drive = kappa`  
- `oxygen_burden = mu`  
- `decay_burden = delta_T + delta_E + delta_O`  

These are not hidden states. They are readable composite quantities used to make the rules biologically explicit and to provide calibration‑friendly gates.

## Heuristic configuration (`StateHeuristics`)

All numerical thresholds that define the rule surfaces are collected in a calibration configuration class, `StateHeuristics`. This includes:

- **Near‑boundary pads**  
  Fractions and absolute paddings used by `_near_lower` and `_near_upper` for each variable, e.g. `frac_C_low`, `abs_C_low_pad`, `frac_T_high`, `abs_T_high_pad`, etc.

- **Derivative thresholds**  
  Thresholds on derivatives, e.g. `dO_apoptosis_cut`, `dC_apoptosis_cut`, `dE_prolif_cut`, `dE_migration_cut`, `dT_quiescence_abs_cut`, `dE_quiescence_abs_cut`, `dO_quiescence_abs_cut`, `dT_diversification_cut`, `dC_diversification_cut`.

- **Absolute and offset oxygen/tension thresholds**  
  Terms like `O_prolif_min_abs`, `O_prolif_min_offset`, `O_quiescence_min_offset`, `T_prolif_min_abs`, `T_prolif_min_offset`.

- **Contextual gates**  
  Thresholds on derived axes such as `oxygen_supply_low`, `oxygen_burden_high`, `tension_drive_strong`, `tension_drive_weak`, `tension_damping_strong`, `matrix_drive_strong`, and `decay_burden_high`.

The classifier never uses “magic numbers” directly. Instead, it obtains all thresholds from `cfg`. This makes the classifier calibration‑ready: fitting the taxonomy to experiments and simulated ensembles is equivalent to adjusting `StateHeuristics`.

## Boundary‑awareness logic

The classifier uses helper functions to detect whether a variable is near a lower or upper reference threshold:

- `_near_lower(x, lo, frac, abs_pad)`  
- `_near_upper(x, hi, frac, abs_pad)`

Using `cfg`, these helpers define the following flags:

- `near_C_low = _near_lower(C, C_min, frac=cfg.frac_C_low, abs_pad=cfg.abs_C_low_pad)`  
- `near_T_low = _near_lower(T, T_min, frac=cfg.frac_T_low, abs_pad=cfg.abs_T_low_pad)`  
- `near_T_high = _near_upper(T, T_max, frac=cfg.frac_T_high, abs_pad=cfg.abs_T_high_pad)`  
- `near_E_low = _near_lower(E, E_min, frac=cfg.frac_E_low, abs_pad=cfg.abs_E_low_pad)`  
- `near_E_high = _near_upper(E, E_max, frac=cfg.frac_E_high, abs_pad=cfg.abs_E_high_pad)`  
- `near_O_low = _near_lower(O, O_min, frac=cfg.frac_O_low, abs_pad=cfg.abs_O_low_pad)`  

This design lets the classifier distinguish between clearly safe values, clearly pathological values, and biologically suspicious near‑boundary values, while keeping the definitions of “near” calibratable.

## Context gates used in the rules

The derived axes are turned into explicit boolean gates using `cfg`:

- `low_oxygen_supply = oxygen_supply < cfg.oxygen_supply_low`  
- `high_oxygen_burden = oxygen_burden > cfg.oxygen_burden_high`  
- `strong_tension_drive = tension_drive > cfg.tension_drive_strong`  
- `weak_tension_drive = tension_drive < cfg.tension_drive_weak`  
- `strong_tension_damping = tension_damping > cfg.tension_damping_strong`  
- `strong_matrix_drive = matrix_drive > cfg.matrix_drive_strong`  
- `high_decay_burden = decay_burden > cfg.decay_burden_high`  

These gates are part of the actual classifier logic and should be treated as calibratable, implementation‑level thresholds rather than vague narrative descriptions.

## Decision order and predicate structure

The classifier is evaluated in the following order:

1. Apoptosis  
2. Proliferation  
3. Migration  
4. Quiescence  
5. Diversification  
6. Undetermined  

This ordering matters. The classifier is a first‑match system, so if a state satisfies an earlier label predicate, later predicates are not evaluated.

In code, each label is implemented as a dedicated predicate:

- `_is_apoptosis_state(...)`  
- `_is_proliferation_state(...)`  
- `_is_migration_state(...)`  
- `_is_quiescence_state(...)`  
- `_is_diversification_state(...)`  
- `_is_near_any_bound(...)`  

`classify_state` calls these in sequence on the current state, derivatives, boundary flags, context, and `cfg`. This refactor makes the decision tree explicit and modular without changing the original semantics.

## Dynamic‑first pseudocode

A schematic view of the logic is:

```python
if _is_apoptosis_state(..., ctx, cfg):
    return "Apoptosis"
elif _is_proliferation_state(..., ctx, cfg):
    return "Proliferation"
elif _is_migration_state(..., ctx, cfg):
    return "Migration"
elif _is_quiescence_state(..., ctx, cfg):
    return "Quiescence"
elif _is_diversification_state(..., ctx, cfg):
    return "Diversification"
elif _is_near_any_bound(...):
    return "Undetermined"
else:
    return "Undetermined"
```

Static context is embedded directly inside the individual predicates (for example via `oxygen_supply`, `tension_drive`, `matrix_drive`, and the calibration thresholds), but the dynamic‑first precedence and ordered matching remain unchanged.

## State‑label determination rules

The table below summarizes the implemented rules used to assign each label under the refactored, parameterized scheme.

| Label | Primary dynamic signature | Contextual gates or supporting conditions | Return condition |
|---|---|---|---|
| **Apoptosis** | One of: (i) `O < O_min` or `C < C_min` together with at least one adverse trend or burden; (ii) `near_O_low` with `dO < cfg.dO_apoptosis_cut`; (iii) `near_C_low` with `dC < cfg.dC_apoptosis_cut` and nonconstructive `dE`/`dT` | Collapse support from `dO < 0`, `dC < 0`, `oxygen_burden > cfg.oxygen_burden_high`, `oxygen_supply < cfg.oxygen_supply_low`, `decay_burden > cfg.decay_burden_high`, or `(dE <= 0 or dT <= 0)` | Returned first if any apoptosis predicate is satisfied |
| **Proliferation** | Viable oxygen and ECM gain: `O > max(cfg.O_prolif_min_abs, O_min + cfg.O_prolif_min_offset)` and `dE > cfg.dE_prolif_cut`, or `near_E_high` with `dE >= 0` | Main branch uses `matrix_drive > cfg.matrix_drive_strong` or supportive tension `T > max(cfg.T_prolif_min_abs, T_min + cfg.T_prolif_min_offset)` and `dT >= -0.02` as contextual support | Returned if no apoptosis rule matched and a proliferation predicate matches |
| **Migration** | Viable ECM loss/remodeling: `O > O_min`, `T >= T_min`, `dE < cfg.dE_migration_cut`, or broader ECM‑loss pattern `E > E_min`, `dE < 0`, `dT <= 0.03`, `dO >= -0.02`, and `not near_O_low` | Stable or damped tension via `abs(dT) < cfg.dT_migration_abs_cut`, or `tension_damping > cfg.tension_damping_strong` together with `not (tension_drive > cfg.tension_drive_strong)` | Returned if earlier labels did not match and a migration predicate matches |
| **Quiescence** | Viable moderate state with small derivatives: `T_min <= T <= 0.75*T_max`, `E_min <= E <= 0.70*E_max`, `O > O_min + cfg.O_quiescence_min_offset`, `abs(dT) < cfg.dT_quiescence_abs_cut`, `abs(dE) < cfg.dE_quiescence_abs_cut`, `abs(dO) < cfg.dO_quiescence_abs_cut` | Low‑drive alternative: `near_T_low` with `tension_drive < cfg.tension_drive_weak`, `dT <= 0`, `dE <= 0.02`, and `O > O_min` | Returned if earlier labels did not match and a quiescence predicate matches |
| **Diversification** | Active, non‑terminal constructive change with `O > O_min` and `C > C_min`, plus one of: `dT > cfg.dT_diversification_cut and dE > 0`, `near_T_high and dE >= 0`, or `dC > cfg.dC_diversification_cut and dE > 0` | Alternative contextual branch: `tension_drive > cfg.tension_drive_strong`, `not (T > T_max)`, `dT >= 0`, and `dE >= cfg.dE_diversification_min` | Returned if earlier labels did not match and a diversification predicate matches |
| **Undetermined** | Mixed, weak, boundary‑adjacent, or unmatched dynamics | One of the boundary flags `near_T_high`, `near_E_high`, `near_O_low`, `near_E_low`, or `near_C_low` via `_is_near_any_bound`; otherwise global fallback when no predicate matches | Returned when no earlier state rule matched |

## Per‑label explanation

### Apoptosis

Apoptosis is intended to capture collapse‑like or terminal behavior. The strongest evidence is low oxygen, collapsing curvature, worsening oxygen, or strong depletion burden. The classifier also treats near‑boundary states with clearly adverse derivatives as apoptotic. Apoptosis is evaluated first so that clearly terminal patterns take precedence over productive or remodeling interpretations.

### Proliferation

Proliferation captures viable, constructive matrix‑building behavior. The classifier looks primarily for sufficient oxygen together with sustained ECM growth (`dE > cfg.dE_prolif_cut`), then uses matrix drive and favorable tension as contextual support. Near‑maximal ECM with nonnegative growth under viable oxygen also supports a proliferative interpretation.

### Migration

Migration captures viable ECM remodeling or loss without immediate collapse. The most important signal is a patterned decrease in ECM (`dE` below a migration cut) under still‑viable oxygen and tension, with either small tension changes or strong damping relative to drive. A broader ECM‑loss pattern with moderate tension and non‑worsening oxygen also supports migration when not near hypoxic boundaries.

### Quiescence

Quiescence captures quiet, viable, low‑activity states. The main signature is a moderate operating range in tension and ECM together with small derivatives in tension, ECM, and oxygen. A weak‑tension‑drive alternative supports quiescence when the system sits near low tension, has limited ECM change, and remains oxygen‑viable, without showing clear signs of collapse or active remodeling.

### Diversification

Diversification captures non‑terminal, non‑quiescent activity suggestive of directional commitment or branching. The classifier looks for positive curvature‑plus‑ECM growth or rising tension plus constructive ECM change under viable oxygen and non‑minimal curvature. Strong tension drive with non‑terminal tension and near‑neutral ECM change provides a contextual route to Diversification when other labels do not fit.

### Undetermined

Undetermined is the ambiguity label and the global fallback. It is used when the state sits near problematic boundaries, when the evidence is mixed, or when none of the earlier state definitions fits cleanly. `_is_near_any_bound` encodes this behavior by collapsing several `near_*` flags into a single boundary‑adjacent predicate.

## Relationship to viability

Taxonomy and viability are related but not identical.

Viability asks whether a point stays inside a predefined admissible region defined by thresholds such as `C_min`, `T_min`, `T_max`, `E_min`, `E_max`, and `O_min`. Taxonomy asks what kind of biological mode the system appears to be expressing.

The taxonomy classifier uses these thresholds and their near‑boundary neighborhoods as biological reference values. However, it does not produce viability reports and should not be interpreted as a direct viability classifier. The same reference bounds help anchor both systems, but the outputs answer different questions.

## Instantaneous use

The classifier is instantaneous. It takes one sampled point and returns one label. It has no built‑in memory of previous states and it does not aggregate information across multiple points.

Any downstream analysis that examines trajectories, endpoints, or temporal persistence should be treated as a separate layer built on top of `classify_state(...)`, not as part of the static taxonomy definition itself.

## Practical coding guidance

The implementation should continue to follow these constraints:

- Keep `classify_state(...)` as the central interface.  
- Use derivatives as part of the primary evidence, not as optional decoration.  
- Merge scenario overrides before extracting contextual axes.  
- Keep the static parameter subset explicit in `INTERPRETABLE_STATIC_PARAMS`.  
- Document helper thresholds and contextual gates in the code and keep them centralized in `StateHeuristics`.  
- Preserve the ordered first‑match rule structure so rule precedence stays auditable.  
- Keep taxonomy labels separate from viability reports.  
- Do not reintroduce trajectory‑level aggregation logic into the static classifier module.

## Recommended interpretation

The classifier should be understood as an interpretable biological annotation layer built on top of the dynamical system. Its job is to read what the system is doing locally, then use a limited amount of mechanistic context and a calibratable set of thresholds to resolve whether that local behavior is better interpreted as collapse, growth, remodeling, quiescence, branching, or ambiguity.

The balance — dynamic‑first evidence, explicit contextual gates, explicit calibration parameters, and auditable rule order — is the central design philosophy of the current implementation.
