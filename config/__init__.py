"""
config
======
Configuration package for viability-kernel simulations.

Exports
-------
DEFAULT_PARAMS
    Base model parameters for the scaffold-cell-matrix ODE system.
DEFAULT_BOUNDS
    Default viability thresholds for the four state variables.
DEFAULT_SIM
    Default simulation settings (time span, tolerances, IC noise, …).
SCENARIOS
    List of predefined scenario configurations (stable, unstable, borderline).
"""

from config.default_params import DEFAULT_PARAMS, DEFAULT_BOUNDS, DEFAULT_SIM
from config.scenarios import SCENARIOS

__all__ = ["DEFAULT_PARAMS", "DEFAULT_BOUNDS", "DEFAULT_SIM", "SCENARIOS"]
