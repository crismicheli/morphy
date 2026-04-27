# Taxonomy classifier design rules

This document defines an explicit rule structure for the biological taxonomy classifier used on scaffold-cell-matrix trajectories. The classifier assigns one of six labels — Apoptosis, Migration, Proliferation, Quiescence, Diversification, or Undetermined — from the instantaneous system state, its local derivatives, and a deliberately restricted subset of static parameters.

The rule design treats the evolving state variables as the primary source of truth and the selected static parameters as secondary contextual modifiers. This separation is intentional: the state and derivative values describe what the system is doing now, while the static parameters describe the mechanistic regime under which that behavior occurs.

## Core principle

The model evolves four dynamic variables, curvature `C`, cytoskeletal tension `T`, ECM density `E`, and oxygen `O`, with corresponding time derivatives `dC/dt`, `dT/dt`, `dE/dt`, and `dO/dt`. These dynamic quantities are the most direct signals of whether the system is collapsing, remodeling, growing, or remaining near a stable operating point.

For that reason, the classifier should be written in a dynamic-first order. Static parameters should only refine the interpretation of ambiguous or near-boundary states, not overwrite an otherwise clear dynamic signature.

## Primary inputs

The primary inputs are the current state coordinates and their local derivatives:

- `C, T, E, O`
- `dC, dT, dE, dO`

These inputs are primary because they directly encode the observed phenotype of the state. For example, low oxygen with a falling oxygen derivative is already strong evidence for collapse-like behavior, while decreasing ECM under viable oxygen and tension is strong evidence for remodeling or migration-like behavior.

### Why the dynamic variables are primary

The governing equations directly define phenotype-relevant mechanisms through these variables:

- `dC/dt = alpha * (g(p) - C)` links curvature loss or recovery to polarity-like structural state.
- `dT/dt = beta * C - delta_T * T - eta * E * T` makes tension a central readout of activation versus damping.
- `dE/dt = kappa * T * O - delta_E * E` makes ECM accumulation versus remodeling directly visible in `E` and `dE`.
- `dO/dt = rho * h(p) - mu * E * O - delta_O * O` makes oxygen sufficiency versus depletion directly visible in `O` and `dO`.

Because each biological label is meant to describe a state of the system rather than just a parameter regime, the dynamic variables should dominate classification decisions.

## Secondary inputs

Only a small, biologically interpretable subset of static parameters should be used as contextual modifiers:

| Parameter | Biological interpretation | Why include it |
|---|---|---|
| `p` | Scaffold porosity / structural context | Alters supply and guidance context. |
| `rho`, `s` | Oxygen supply scaling | Together determine effective oxygen availability from porosity. |
| `beta` | Curvature-to-tension coupling | Indicates how strongly structure loads tension. |
| `eta` | ECM-mediated tension damping | Indicates how strongly ECM restrains tension. |
| `kappa` | ECM deposition rate | Biases toward matrix accumulation or growth-like states. |
| `mu` | ECM-mediated oxygen consumption | Biases toward oxygen stress and collapse risk. |
| `delta_T` | Tension decay | Shapes persistence versus relaxation of active tension. |
| `delta_E` | ECM remodeling / degradation | Shapes persistence versus erosion of matrix. |
| `delta_O` | Baseline oxygen loss | Shapes vulnerability to low-oxygen states. |

This subset is preferable to using the full parameter vector because it preserves interpretability and maps cleanly to the biological semantics of the labels. Parameters such as `a`, `b`, and `alpha` affect the dynamics, but they are less directly tied to the meaning of labels like Apoptosis, Migration, or Quiescence.

### Derived contextual axes

The classifier should make its secondary logic explicit by constructing a few composite axes from the selected static parameters:

- `oxygen_supply = rho * s * p`, representing effective porosity-linked oxygen support.
- `tension_drive = beta`, representing the tendency to convert curvature into tension.
- `tension_damping = eta`, representing ECM-mediated restraint of tension.
- `matrix_drive = kappa`, representing anabolic ECM production tendency.
- `oxygen_burden = mu`, representing ECM-dependent oxygen stress burden.
- `decay_burden = delta_T + delta_E + delta_O`, representing aggregate relaxation / depletion burden.

These axes keep the code readable and make the biological rationale transparent.

## Rule order

The recommended rule order is:

1. Evaluate clearly terminal or collapse-like states first.
2. Evaluate clearly productive or remodeling states next.
3. Evaluate quiet or low-activity states after that.
4. Use contextual static parameters only to resolve uncertainty or bias interpretation when the dynamic evidence is mixed.
5. Fall back to Undetermined for ambiguous boundary cases.

This order prevents static context from overwhelming an obviously interpretable dynamic signature.

## Label definitions

### Apoptosis

**Primary evidence** should be low oxygen, falling oxygen, low or collapsing curvature, or simultaneous decline in core viability-supporting variables. In code, this means prioritizing cases where `O` is below or near its lower range and `dO < 0`, or where `C` is near collapse and still decreasing.

**Secondary evidence** should be poor oxygen supply, high oxygen burden, or high aggregate decay burden. These parameters do not define apoptosis on their own, but they strengthen the interpretation that a low-oxygen or collapsing state is a genuine death-like trajectory rather than a transient dip.

### Migration

**Primary evidence** should be ECM loss or remodeling under still-viable oxygen and tension, typically `dE < 0` while `O > O_min` and `T >= T_min`. This corresponds to a state that is active and viable, but not retaining or building matrix.

**Secondary evidence** should be stronger tension damping without excessive tension drive, because that supports matrix loosening and restrained force transmission rather than escalating stress. These static inputs should only reinforce the migration interpretation when the dynamic signal already points toward remodeling.

### Proliferation

**Primary evidence** should be viable oxygen together with increasing matrix, for example `dE > 0` and non-collapsing tension. This indicates biosynthetic or expansion-like behavior rather than merely survival.

**Secondary evidence** should be strong matrix drive `kappa` and adequate oxygen supply. These parameters support the interpretation that positive matrix growth is part of a constructive proliferative regime rather than a transient fluctuation.

### Quiescence

**Primary evidence** should be moderate viable values of `T`, `E`, and `O` combined with small derivatives, meaning the system is neither collapsing nor actively remodeling. In practice, this means all relevant derivatives are near zero and the state remains away from extreme boundaries.

**Secondary evidence** should be weak tension drive or generally low forcing. These parameters help explain why the system remains quiet, but they should not create a quiescence label if the dynamic variables show active remodeling or collapse.

### Diversification

**Primary evidence** should be non-terminal, non-quiescent mixed activity such as rising tension with increasing matrix, or rising curvature with constructive ECM behavior. This label captures directional change or commitment-like behavior rather than a fully settled phenotype.

**Secondary evidence** should be stronger tension drive in the presence of adequate oxygen and nonnegative matrix tendency. This suggests an active mechanobiological regime capable of branching toward differentiated or diversified outcomes.

### Undetermined

This label should be used when the state is ambiguous, mixed, or near multiple boundaries without a sufficiently clear dominant interpretation. It is especially useful for cases where dynamic evidence is weak or contradictory and the static context does not resolve the ambiguity.

## Dynamic-first pseudocode

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

This structure makes the intended precedence explicit and preserves biological interpretability.

## Independence from viability

The taxonomy classification should remain **independent from viability classification**. Viability answers whether a trajectory stays inside a predefined admissible region, while the taxonomy answers what kind of biological state a point or trajectory appears to represent.

The viability logic is threshold-based and uses bounds such as `C_min`, `T_min`, `T_max`, `E_min`, `E_max`, and `O_min` to determine whether a state remains inside a kernel-like acceptable region. A trajectory is viable when it remains within those bounds across time.

By contrast, taxonomy labels should describe biological mode even when a state is outside viability bounds. For example:

- A state can be classified as Apoptosis precisely because it has crossed into low-oxygen or collapse-like territory.
- A state can be Migration-like while still viable, because it may be remodeling matrix without violating any bound.
- A state can be Undetermined even if it is technically viable, because viability does not imply clear biological identity.

This separation is important because otherwise taxonomy would collapse into a restatement of the viability test rather than a biologically meaningful interpretation layer.

## Practical coding guidance

The classifier implementation should therefore follow these design constraints:

- Keep `classify_state(C, T, E, O, dC, dT, dE, dO, bounds, par=None, scenario_cfg=None)` as the central interface.
- Compute derivatives from the simulated trajectory before classification, because local flow is part of the primary evidence.
- Read static parameter overrides from `scenario_cfg["param_overrides"]` and porosity from `scenario_cfg["p"]`, merged on top of base parameters.
- Store the chosen static subset in a named constant such as `INTERPRETABLE_STATIC_PARAMS` so the model assumptions are visible in code.
- Keep taxonomy labels and viability reports separate objects or separate function outputs.

## Summary of intended philosophy

The classifier should be interpreted as a biological state annotator layered on top of the dynamical system. Its first task is to read what the system is doing from `C`, `T`, `E`, `O` and their derivatives; its second task is to use a small, explicit subset of mechanistically meaningful static parameters to clarify uncertain cases.

That philosophy keeps the classifier readable, biologically interpretable, and distinct from the independent viability kernel logic already implemented in the simulation stack.
