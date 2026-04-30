#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from config import DEFAULTBOUNDS, DEFAULTPARAMS, DEFAULTSIM, SCENARIOS
from viabilitykernels.simulation import runscenario
from plotting.plot2d_helpers import print_summary_table, save_all_scenarios_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run viability-kernel scenario simulations.")
    parser.add_argument("--save", metavar="PATH", default=None, help="Save the figure to this path (e.g., figures/all.png).")
    parser.add_argument("--filter", metavar="KEYWORD", default=None, help="Only run scenarios whose label contains KEYWORD (case-insensitive).")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting; print the summary table only.")
    parser.add_argument("--n-traj", type=int, default=DEFAULTSIM["n_traj"], help=f"Number of trajectories per ensemble (default: {DEFAULTSIM['n_traj']}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = SCENARIOS
    if args.filter:
        kw = args.filter.lower()
        scenarios = [s for s in scenarios if kw in s["label"].lower()]
    if not scenarios:
        print(f"No scenarios match the filter '{args.filter}'. Exiting.")
        raise SystemExit(1)
    print(f"\nRunning {len(scenarios)} scenario(s)...\n")
    all_results = []
    for cfg in scenarios:
        t0 = time.perf_counter()
        result = runscenario(
            scenariocfg=cfg,
            par=DEFAULTPARAMS,
            bounds=DEFAULTBOUNDS,
            x0center=DEFAULTSIM["x0_center"],
            ntraj=args.n_traj,
            tspan=tuple(DEFAULTSIM["t_span"]),
            neval=DEFAULTSIM["n_eval"],
            rngseed=DEFAULTSIM["rng_seed"],
        )
        result["expected"] = cfg.get("expected", "—")
        elapsed = time.perf_counter() - t0
        print(f" ✓ {cfg['label']:<50} vf={result['viable_fraction']:.0%} ({elapsed:.2f}s)")
        all_results.append(result)
    print_summary_table(all_results)
    if args.no_plot:
        return
    fig = save_all_scenarios_figure(
        all_results,
        args.save,
        par=DEFAULTPARAMS,
        bounds=DEFAULTBOUNDS,
        suptitle="Viability kernel simulations — porosity and biophysical scenarios",
    )
    if args.save:
        print(f"Figure saved to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
