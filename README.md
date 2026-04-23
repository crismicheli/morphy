# Morphy

Morphy is a Python repository for simulating and visualizing viability behavior in a scaffold–cell–matrix dynamical system. The current codebase provides an installable core package, reusable configuration presets, a script entry point for running predefined scenarios, and project folders for figures, notebooks, tests, and documentation.

At the moment, the internal Python package is named `viabilitykernels`, while the repository name remains `morphy`. This keeps the current import paths stable while leaving open the option to rename the package later if desired.

## Features

- ODE definitions for the scaffold–cell–matrix system via `odes.py`.
- Viability classification utilities and trajectory diagnostics via `viability.py`.
- Ensemble simulation runners and initial-condition sampling via `simulation.py`.
- Phase-plane plotting and multi-scenario visualization helpers via `phase_plane.py`.
- Predefined simulation defaults and scenario presets via the `config` package.
- A command-line script for running scenario batches and optionally saving figures.

## Repository structure

```text
morphy/
├── pyproject.toml
├── README.md
├── .gitignore
├── viabilitykernels/
│   ├── __init__.py
│   ├── odes.py
│   ├── viability.py
│   ├── simulation.py
│   └── phase_plane.py
├── config/
│   ├── __init__.py
│   ├── default_params.py
│   └── scenarios.py
├── scripts/
│   └── run_scenarios.py
├── figures/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── tests/
│   └── .gitkeep
└── docs/
    └── .gitkeep
```

The top-level structure follows a common Python-project pattern: importable source code lives in package directories, executable utilities live in `scripts/`, and exploratory or generated material lives in dedicated project folders rather than inside the package itself.

## Package layout

### `viabilitykernels/`

This is the main Python package. It contains the reusable scientific code: the ODE right-hand side, quasi-steady approximations, viability checks, ensemble simulation logic, and phase-plane plotting helpers.

### `config/`

This package contains reusable configuration objects for the project, including default model parameters, default viability bounds, default simulation settings, and predefined scenarios. Its `__init__.py` file re-exports the main configuration constants to simplify imports such as `from config import DEFAULT_PARAMS, SCENARIOS`.

### `scripts/`

This folder contains runnable entry points that orchestrate package code and config presets. The current main script is `run_scenarios.py`, which parses CLI arguments, runs the predefined scenarios, prints a summary table, and optionally saves a multi-panel figure.

### `figures/`, `notebooks/`, `tests/`, `docs/`

- `figures/` stores saved output plots and paper-ready graphics.
- `notebooks/` is reserved for exploratory analysis, prototyping, and narrative computational work.
- `tests/` is reserved for automated tests as the package matures.
- `docs/` is reserved for longer technical or user documentation beyond this README.

## Installation

Clone the repository and create a virtual environment before installing dependencies. A root-level `pyproject.toml` is the standard place for Python project metadata and installation configuration.

```bash
git clone https://github.com/crismicheli/morphy.git
cd morphy
python -m venv .venv
source .venv/bin/activate
OR
python -m venv .venv
.\.venv\Scripts\Activate.ps1
IF IN ANACONDA POWER SHELL
python -m pip install --upgrade pip
python -m pip install -e .
```

If you have not yet written `pyproject.toml`, you can still run the code directly from the repo root during the transition period, but moving to editable installation is the cleaner long-term setup for a package-oriented repository.

## Usage

Run the scenario batch script from the repository root:

```bash
python scripts/run_scenarios.py
```

Save a figure to the `figures/` folder:

```bash
python scripts/run_scenarios.py --save figures/all_scenarios.png
```

Filter scenarios by keyword:

```bash
python scripts/run_scenarios.py --filter stable
```

Skip plotting and print only the summary table:

```bash
python scripts/run_scenarios.py --no-plot
```

## Import examples

You can import configuration constants from the `config` package:

```python
from config import DEFAULT_PARAMS, DEFAULT_BOUNDS, DEFAULT_SIM, SCENARIOS
```

You can also import core simulation functions from the scientific package:

```python
from viabilitykernels.simulation import runscenario
from viabilitykernels.viability import classifyensemble
```

## Roadmap

Potential next steps for the repository include adding automated tests, defining package metadata and dependencies in `pyproject.toml`, documenting the mathematical model more fully in `docs/`, and identifying a layer of simple inference that connects mutlicellular phenotypical states with kernels trajectories in time.

## License

Add your preferred license in a `LICENSE` file at the repository root so downstream users know how they can use and adapt the code. Including license information in the root repository files is standard README practice for Python projects.
