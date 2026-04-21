"""
config/default_params.py
------------------------
Default model parameters and viability bounds for the scaffold-cell-matrix
ODE system.

All values are dimensionless and qualitative — they are chosen to produce
biologically plausible dynamics, not to match quantitative experimental data.

Usage::

    from config.default_params import DEFAULT_PARAMS, DEFAULT_BOUNDS

Parameter reference
-------------------
Curvature guidance function  g(p) = a * p * exp(-b * p):
    a   : amplitude of the hump (peak guidance level)
    b   : decay rate (location of the peak ~ 1/b)

Oxygen supply function  h(p) = s * p:
    s   : linear scaling of oxygen supply with porosity

Curvature equation  dC/dt = alpha*(g(p) - C):
    alpha  : relaxation rate of C toward the scaffold target

Tension equation  dT/dt = beta*C - delta_T*T - eta*E*T:
    beta    : curvature-to-tension coupling
    delta_T : intrinsic tension decay rate
    eta     : ECM-mediated damping of tension

ECM equation  dE/dt = kappa*T*O - delta_E*E:
    kappa   : ECM deposition rate (requires T and O)
    delta_E : ECM degradation / remodelling rate

Oxygen equation  dO/dt = rho*h(p) - mu*E*O - delta_O*O:
    rho     : porosity-to-oxygen scaling
    mu      : ECM-mediated oxygen consumption
    delta_O : baseline oxygen loss / consumption
"""

# ---------------------------------------------------------------------------
# Base model parameters
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict = {
    # --- Curvature guidance hump-shaped function g(p) = a*p*exp(-b*p) ---
    "a": 4.0,    # amplitude
    "b": 3.0,    # decay

    # --- Oxygen supply h(p) = s*p ---
    "s": 1.2,    # linear scaling

    # --- Curvature relaxation ---
    "alpha": 1.5,

    # --- Tension dynamics ---
    "beta": 1.8,
    "delta_T": 1.0,
    "eta": 0.8,

    # --- ECM dynamics ---
    "kappa": 1.2,
    "delta_E": 0.5,

    # --- Oxygen dynamics ---
    "rho": 0.9,
    "mu": 0.8,
    "delta_O": 0.4,
}


# ---------------------------------------------------------------------------
# Default viability bounds
# ---------------------------------------------------------------------------

DEFAULT_BOUNDS: dict = {
    # Curvature must not collapse to near-zero (cells have lost polarity)
    "C_min": 0.15,

    # Tension must remain in a functional window (too low or too high is bad)
    "T_min": 0.10,
    "T_max": 1.50,

    # ECM density must be meaningful and not overly dense
    "E_min": 0.05,
    "E_max": 1.80,

    # Oxygen must not drop below a minimum survival threshold
    "O_min": 0.20,
}


# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------

DEFAULT_SIM: dict = {
    "t_span": (0.0, 30.0),
    "n_eval": 800,
    "n_traj": 18,
    "rtol": 1e-6,
    "atol": 1e-8,
    "rng_seed": 42,
    # Central initial condition: moderate curvature, low T, low E, moderate O
    "x0_center": [0.20, 0.15, 0.10, 0.60],
    "noise_scale": [0.03, 0.03, 0.03, 0.05],
}
