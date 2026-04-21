"""
viability_kernels
=================
A modular toolkit for simulating viability kernels in the scaffold-cell-matrix
dynamical system described in Supplementary Appendix A/B.

Sub-modules
-----------
odes
    ODE right-hand side and porosity-dependent auxiliary functions.
viability
    Trajectory classification and viability-kernel diagnostics.
phase_plane
    Projected vector field and phase-portrait plotting helpers.
simulation
    Initial condition sampling, trajectory integration, and ensemble runners.
"""

from viability_kernels.odes import (
    g_porosity,
    h_porosity,
    rhs,
    quasi_steady_C,
    quasi_steady_O,
)
from viability_kernels.viability import (
    ViabilityReport,
    check_trajectory,
    classify_ensemble,
    viable_fraction,
)
from viability_kernels.simulation import (
    sample_initial_conditions,
    integrate_trajectory,
    run_ensemble,
    run_scenario,
)

__all__ = [
    # odes
    "g_porosity",
    "h_porosity",
    "rhs",
    "quasi_steady_C",
    "quasi_steady_O",
    # viability
    "ViabilityReport",
    "check_trajectory",
    "classify_ensemble",
    "viable_fraction",
    # simulation
    "sample_initial_conditions",
    "integrate_trajectory",
    "run_ensemble",
    "run_scenario",
]
