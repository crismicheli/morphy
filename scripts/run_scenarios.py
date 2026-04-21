#!/usr/bin/env python3
"""
scripts/run_scenarios.py
------------------------
Command-line entry point: run all predefined scenarios, print a summary
table, and optionally save a multi-panel phase-portrait figure.

Usage
-----
Run all scenarios and display the figure interactively::

    python scripts/run_scenarios.py

Save the figure to a file::

    python scripts/run_scenarios.py --save figures/all_scenarios.png

Run only scenarios whose label contains a keyword (case-insensitive)::

    python scripts/run_scenarios.py --filter stable

Show this help message::

    python scripts/run_scenarios.py --help
"""

import argparse
import sys
import os
import time

# Allow running from the repository root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from config.default_params import DEFAULT_PARAMS, DEFAULT_BOUNDS, DEFAULT_SIM
from config.scenarios import SCENARIOS
from viability_kernels.simulation import run_scenario
from viability_kernels.phase_plane import plot_all_scenarios


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run viability-kernel scenario simulations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the figure to this path (e.g., figures/all.png).",
    )
    parser.add_argument(
        "--filter",
        metavar="KEYWORD",
        default=None,
        help="Only run scenarios whose label contains KEYWORD (case-insensitive).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting; print the summary table only.",
    )
    parser.add_argument(
        "--n-traj",
        type=int,
        default=DEFAULT_SIM["n_traj"],
        help=f"Number of trajectories per ensemble (default: {DEFAULT_SIM['n_traj']}).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list) -> None:
    """Print a formatted summary table to stdout."""
    header = f"{'Scenario':<50} {'p':>6} {'Viable':>8} {'Expected':>12}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in results:
        vf = r["viable_fraction"]
        pct = f"{vf:.0%}"
        expected = r.get("expected", "—")
        print(f"  {r['label']:<48} {r['p']:>6.2f} {pct:>8}  {expected:>12}")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Filter scenarios if requested
    scenarios = SCENARIOS
    if args.filter:
        kw = args.filter.lower()
        scenarios = [s for s in scenarios if kw in s["label"].lower()]
        if not scenarios:
            print(f"No scenarios match the filter '{args.filter}'. Exiting.")
            sys.exit(1)

    x0_center = np.array(DEFAULT_SIM["x0_center"])

    print(f"\nRunning {len(scenarios)} scenario(s)...\n")
    all_results = []

    for cfg in scenarios:
        t0 = time.perf_counter()
        result = run_scenario(
            scenario_cfg=cfg,
            par=DEFAULT_PARAMS,
            bounds=DEFAULT_BOUNDS,
            x0_center=x0_center,
            n_traj=args.n_traj,
            t_span=tuple(DEFAULT_SIM["t_span"]),
            n_eval=DEFAULT_SIM["n_eval"],
            rng_seed=DEFAULT_SIM["rng_seed"],
        )
        # Carry the expected label into the result dict
        result["expected"] = cfg.get("expected", "—")
        elapsed = time.perf_counter() - t0
        print(f"  ✓  {cfg['label']:<50}  vf={result['viable_fraction']:.0%}  ({elapsed:.2f}s)")
        all_results.append(result)

    print_summary(all_results)

    if args.no_plot:
        return

    if args.save and not os.path.exists(os.path.dirname(args.save)):
        os.makedirs(os.path.dirname(args.save), exist_ok=True)

    # For plotting we use the default params (shared across all scenarios)
    fig = plot_all_scenarios(
        scenario_results=all_results,
        par=DEFAULT_PARAMS,   # used only for the vector field direction
        bounds=DEFAULT_BOUNDS,
        suptitle="Viability kernel simulations — porosity and biophysical scenarios",
        save_path=args.save,
    )

    if args.save:
        print(f"Figure saved to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
