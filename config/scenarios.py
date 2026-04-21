"""
config/scenarios.py
-------------------
Predefined simulation scenarios covering a range of scaffold and biophysical
conditions — some that produce stable, viable trajectories and others that
drive the system out of the viability kernel.

Each scenario is a dictionary with the following schema::

    {
        "label"          : str          # human-readable name shown in plots
        "p"              : float        # scaffold porosity in [0, 1]
        "description"    : str          # one-paragraph biological rationale
        "expected"       : str          # "stable" | "unstable" | "borderline"
        "param_overrides": dict         # optional; overrides DEFAULT_PARAMS
    }

The ``param_overrides`` field allows individual scenarios to change specific
model parameters without modifying the shared defaults.

Scenario catalogue
------------------
Porosity sweep (base parameters):
    1. Low porosity (p=0.15)       — weak guidance, poor oxygen  → unstable
    2. Intermediate porosity (p=0.40) — optimal regime          → stable
    3. High porosity (p=0.75)      — faded guidance, high O₂    → borderline

Biophysical extremes:
    4. Stiff scaffold / high ECM damping (eta high)             → stable
    5. Hypoxic environment (low rho, low s)                     → unstable
    6. Over-tensioned (high beta)                               → unstable
    7. Optimal + aggressive ECM remodelling (high delta_E)      → borderline
    8. Near-optimal with enhanced curvature (high a)            → stable
"""

SCENARIOS: list = [
    # ------------------------------------------------------------------
    # 1. LOW POROSITY — stable regime is not reached
    # ------------------------------------------------------------------
    {
        "label": "Low porosity  (p=0.15) — unstable",
        "p": 0.15,
        "description": (
            "Very low porosity severely restricts both curvature guidance "
            "and oxygen transport.  The curvature target g(0.15) is small, "
            "so C never rises enough to sustain meaningful tension.  Oxygen "
            "supply h(0.15) is also poor.  Most trajectories exit the "
            "viability kernel through the T_min or O_min boundary."
        ),
        "expected": "unstable",
        "param_overrides": {},
    },

    # ------------------------------------------------------------------
    # 2. INTERMEDIATE POROSITY — optimal regime
    # ------------------------------------------------------------------
    {
        "label": "Intermediate porosity  (p=0.40) — stable",
        "p": 0.40,
        "description": (
            "Intermediate porosity sits near the peak of the hump-shaped "
            "guidance function g(p), providing strong curvature signals "
            "while h(p) delivers adequate oxygen.  This combination allows "
            "tension to rise to a moderate level and ECM to accumulate "
            "gradually, keeping the system inside the viable region."
        ),
        "expected": "stable",
        "param_overrides": {},
    },

    # ------------------------------------------------------------------
    # 3. HIGH POROSITY — borderline
    # ------------------------------------------------------------------
    {
        "label": "High porosity  (p=0.75) — borderline",
        "p": 0.75,
        "description": (
            "At high porosity the curvature guidance g(p) has decayed well "
            "past its peak, so the mechanical cue is weakened.  Oxygen "
            "supply is generous.  Some trajectories converge to a viable "
            "steady state driven by oxygen, while others drift out as ECM "
            "accumulates and depletes O below the threshold."
        ),
        "expected": "borderline",
        "param_overrides": {},
    },

    # ------------------------------------------------------------------
    # 4. STIFF SCAFFOLD — high ECM damping keeps tension in range
    # ------------------------------------------------------------------
    {
        "label": "Stiff scaffold  (η=1.8) — stable",
        "p": 0.40,
        "description": (
            "A stiffer scaffold is modelled by increasing the ECM-mediated "
            "damping coefficient η.  Dense crosslinked ECM strongly attenuates "
            "mechanical fluctuations, preventing tension from reaching the "
            "T_max ceiling.  Combined with intermediate porosity this yields "
            "a very reliable, compact attractor well inside the viability kernel."
        ),
        "expected": "stable",
        "param_overrides": {"eta": 1.8},
    },

    # ------------------------------------------------------------------
    # 5. HYPOXIC ENVIRONMENT — poor oxygen supply
    # ------------------------------------------------------------------
    {
        "label": "Hypoxic environment  (ρ=0.3, s=0.4) — unstable",
        "p": 0.40,
        "description": (
            "Hypoxia is modelled by reducing both the porosity-oxygen scaling "
            "s and the supply rate ρ.  Even at intermediate porosity the "
            "oxygen level O quickly falls below O_min, halting ECM deposition "
            "(which requires O > 0) and causing the system to collapse.  "
            "Essentially no trajectory remains viable."
        ),
        "expected": "unstable",
        "param_overrides": {"rho": 0.3, "s": 0.4},
    },

    # ------------------------------------------------------------------
    # 6. OVER-TENSIONED — high beta drives T above ceiling
    # ------------------------------------------------------------------
    {
        "label": "Over-tensioned  (β=3.5) — unstable",
        "p": 0.40,
        "description": (
            "A large curvature-to-tension coupling β means that even a "
            "moderate curvature signal rapidly amplifies cytoskeletal tension "
            "past the T_max threshold.  The system enters a hyper-contractile "
            "regime that is biologically associated with excessive cell "
            "contraction and scaffold compaction."
        ),
        "expected": "unstable",
        "param_overrides": {"beta": 3.5},
    },

    # ------------------------------------------------------------------
    # 7. FAST ECM REMODELLING — borderline stability
    # ------------------------------------------------------------------
    {
        "label": "Fast ECM remodelling  (δ_E=1.2) — borderline",
        "p": 0.40,
        "description": (
            "A high ECM degradation rate δ_E prevents the matrix from "
            "accumulating enough to anchor the mechanical state.  Trajectories "
            "oscillate near the E_min boundary.  Whether a trajectory stays "
            "viable depends sensitively on the initial oxygen level."
        ),
        "expected": "borderline",
        "param_overrides": {"delta_E": 1.2},
    },

    # ------------------------------------------------------------------
    # 8. ENHANCED CURVATURE GUIDANCE — robust stable
    # ------------------------------------------------------------------
    {
        "label": "Enhanced guidance  (a=6.0) — stable",
        "p": 0.35,
        "description": (
            "Increasing the amplitude a of the hump-shaped guidance function "
            "g(p) enhances the geometric cue the scaffold provides to cells. "
            "Combined with slightly sub-optimal porosity (p=0.35), the larger "
            "peak amplitude still delivers strong curvature, resulting in "
            "robust convergence to a viable attractor across all tested "
            "initial conditions."
        ),
        "expected": "stable",
        "param_overrides": {"a": 6.0},
    },
]
