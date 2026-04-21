"""
odes.py
-------
Right-hand side of the scaffold-cell-matrix ODE system and the two
porosity-dependent auxiliary functions.

State vector
~~~~~~~~~~~~
x = [C, T, E, O]

    C  : curvature state adopted by cells
    T  : cytoskeletal tension
    E  : extracellular matrix (ECM) density
    O  : oxygen availability

The four equations are::

    dC/dt = alpha * (g(p) - C)
    dT/dt = beta*C - delta_T*T - eta*E*T
    dE/dt = kappa*T*O - delta_E*E
    dO/dt = rho*h(p) - mu*E*O - delta_O*O

where g(p) and h(p) encode how scaffold porosity p enters the dynamics.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Porosity-dependent auxiliary functions
# ---------------------------------------------------------------------------

def g_porosity(p: float, a: float, b: float) -> float:
    """Effective curvature guidance imposed by scaffold porosity.

    This hump-shaped function captures the geometric interaction between
    the scaffold's pore structure and adherent cells:

    * Low porosity  → weak guidance (cells too confined)
    * Intermediate  → maximal guidance
    * High porosity → guidance fades again (cells underconfined)

    The function is ``g(p) = a * p * exp(-b * p)``.

    Parameters
    ----------
    p : float
        Scaffold porosity (dimensionless, typically in [0, 1]).
    a : float
        Amplitude parameter controlling the peak guidance level.
    b : float
        Decay parameter controlling how fast guidance falls off with p.

    Returns
    -------
    float
        Curvature guidance level (same units as C).
    """
    return a * p * np.exp(-b * p)


def h_porosity(p: float, s: float) -> float:
    """Effective oxygen supply promoted by porosity.

    A simple monotonically increasing relation: more open pores allow
    better nutrient transport.  The function is ``h(p) = s * p``.

    Parameters
    ----------
    p : float
        Scaffold porosity (dimensionless, typically in [0, 1]).
    s : float
        Scaling parameter (units: oxygen supply per unit porosity).

    Returns
    -------
    float
        Oxygen supply level (same units as O).
    """
    return s * p


# ---------------------------------------------------------------------------
# Full 4-D ODE right-hand side
# ---------------------------------------------------------------------------

def rhs(t: float, x: np.ndarray, p: float, par: dict) -> list:
    """Right-hand side of the scaffold-cell-matrix ODE system.

    This function is compatible with ``scipy.integrate.solve_ivp``.

    Parameters
    ----------
    t : float
        Current time (required by solve_ivp, not used explicitly because
        the system is autonomous).
    x : array-like, shape (4,)
        Current state vector ``[C, T, E, O]``.
    p : float
        Scaffold porosity value for this simulation run.
    par : dict
        Dictionary of model parameters.  Expected keys:
        ``a, b, s, alpha, beta, delta_T, eta, kappa, delta_E,
        rho, mu, delta_O``.

    Returns
    -------
    list of float, length 4
        Time derivatives ``[dC/dt, dT/dt, dE/dt, dO/dt]``.

    Raises
    ------
    ValueError
        If any state variable becomes negative (non-physical).
    """
    C, T, E, O = x

    # Non-negativity guard (soft; integration error will still propagate)
    if C < 0 or T < 0 or E < 0 or O < 0:
        raise ValueError(
            f"Non-physical state encountered at t={t:.4f}: "
            f"C={C:.4f}, T={T:.4f}, E={E:.4f}, O={O:.4f}"
        )

    # Porosity-dependent inputs
    g = g_porosity(p, par["a"], par["b"])
    h = h_porosity(p, par["s"])

    # Curvature: relaxes toward the scaffold-imposed target g(p)
    dCdt = par["alpha"] * (g - C)

    # Tension: driven by curvature, damped intrinsically and by ECM
    dTdt = par["beta"] * C - par["delta_T"] * T - par["eta"] * E * T

    # ECM deposition: requires both tension and oxygen; balanced by degradation
    dEdt = par["kappa"] * T * O - par["delta_E"] * E

    # Oxygen: supplied via porosity, consumed by ECM accumulation and baseline loss
    dOdt = par["rho"] * h - par["mu"] * E * O - par["delta_O"] * O

    return [dCdt, dTdt, dEdt, dOdt]


# ---------------------------------------------------------------------------
# Quasi-steady approximations (used for phase-plane projection only)
# ---------------------------------------------------------------------------

def quasi_steady_C(p: float, par: dict) -> float:
    """Quasi-steady curvature approximation.

    Because alpha is typically large, C relaxes quickly to the scaffold
    target g(p).  This function returns that fast-time fixed point.

    .. math::
        C^* = g(p)

    Parameters
    ----------
    p : float
        Scaffold porosity.
    par : dict
        Model parameters (must contain ``a`` and ``b``).

    Returns
    -------
    float
        Approximate curvature at quasi-steady state.
    """
    return g_porosity(p, par["a"], par["b"])


def quasi_steady_O(E: np.ndarray, p: float, par: dict) -> np.ndarray:
    """Quasi-steady oxygen approximation given ECM density E.

    Setting ``dO/dt = 0`` and solving for O gives::

        O*(E) = rho * h(p) / (mu * E + delta_O)

    Parameters
    ----------
    E : array-like
        ECM density (scalar or array).
    p : float
        Scaffold porosity.
    par : dict
        Model parameters (must contain ``rho, s, mu, delta_O``).

    Returns
    -------
    array-like
        Approximate oxygen level(s) at quasi-steady state.
    """
    h = h_porosity(p, par["s"])
    denom = par["mu"] * E + par["delta_O"]
    return (par["rho"] * h) / denom
