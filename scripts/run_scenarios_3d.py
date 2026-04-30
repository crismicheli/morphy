#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULTBOUNDS, DEFAULTPARAMS, DEFAULTSIM, SCENARIOS
from viabilitykernels.simulation import runscenario, sampleinitialconditions
from plotting.plot_helpers import save_taxonomy_plot, save_trajectory_animation
from plotting.scenario_helpers import scenario_slug, warn_if_any_initial_conditions_outside
from classifiers.classifier_dispatch import get_classifier_components


def choose_scenarios(expected: str) -> list[dict]:
    scenarios = [s for s in SCENARIOS if str(s.get("expected", "")).lower() == expected]
    if not scenarios:
        available = sorted({str(s.get("expected")) for s in SCENARIOS if s.get("expected") is not None})
        raise ValueError(f"No scenarios found for expected={expected!r}. Available values: {', '.join(available)}")
    return scenarios


def make_regime_center(regime: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    if regime == "inside":
        center = np.array([0.22, 0.45, 0.35, 0.65], dtype=float)
        noise = (0.02, 0.03, 0.03, 0.03)
    elif regime == "inside_near_boundary":
        center = np.array([0.16, 1.40, 1.65, 0.24], dtype=float)
        noise = (0.01, 0.04, 0.05, 0.02)
    elif regime == "outside_near_boundary":
        center = np.array([0.14, 1.56, 1.86, 0.18], dtype=float)
        noise = (0.01, 0.04, 0.05, 0.02)
    else:
        raise ValueError(f"Unknown regime: {regime}")
    return center, noise


def run_regime_scenario(scenario: dict, regime: str, *, n_traj: int):
    x0_center, noise_scale = make_regime_center(regime)
    initial_conditions = sampleinitialconditions(
        x0center=x0_center,
        ntraj=n_traj,
        noisescale=noise_scale,
        rngseed=DEFAULTSIM["rng_seed"],
    )
    warn_if_any_initial_conditions_outside(initial_conditions, DEFAULTBOUNDS)
    result = runscenario(
        scenariocfg=scenario,
        par=DEFAULTPARAMS,
        bounds=DEFAULTBOUNDS,
        x0center=x0_center,
        ntraj=n_traj,
        tspan=tuple(DEFAULTSIM["t_span"]),
        neval=DEFAULTSIM["n_eval"],
        rngseed=DEFAULTSIM["rng_seed"],
        noisescale=noise_scale,
        initialconditions=initial_conditions,
    )
    result["regime"] = regime
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep scenarios by expected class and generate taxonomy plots plus 3D ETO animations.")
    parser.add_argument("scenario_class", help="Scenario expected class to sweep, e.g. unstable, boundary, or stable.")
    parser.add_argument("--classifier-type", choices=["static", "temporal", "state_machine"], default="static", help="Classifier backend for taxonomy plots.")
    parser.add_argument("--n-traj", type=int, default=DEFAULTSIM["n_traj"], help="Number of trajectories per sweep run.")
    parser.add_argument("--stride", type=int, default=8, help="Subsample factor for taxonomy plot timepoints.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for animations.")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum number of animation frames.")
    parser.add_argument("--show-box", action="store_true", help="Show translucent viability box in both outputs.")
    parser.add_argument("--out-dir", default=None, help="Optional output directory. Defaults to figures/<scenario_class>_sweep_3d.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_class = args.scenario_class.strip().lower()
    selected_scenarios = choose_scenarios(scenario_class)
    classifier_fn, reset_fn, state_colors = get_classifier_components(args.classifier_type)
    regimes = ["inside", "inside_near_boundary", "outside_near_boundary"]
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "figures" / f"{scenario_class}_sweep_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_ensemble_runs = 0
    for scenario in selected_scenarios:
        for regime in regimes:
            result = run_regime_scenario(scenario, regime, n_traj=args.n_traj)
            slug = scenario_slug(scenario["label"])
            plot_path = out_dir / f"{slug}__{regime}__taxonomy_3d.png"
            anim_path = out_dir / f"{slug}__{regime}__eto_3d.gif"
            save_taxonomy_plot(
                result,
                scenario,
                plot_path,
                bounds=DEFAULTBOUNDS,
                par=DEFAULTPARAMS,
                classifier_fn=classifier_fn,
                color_map=state_colors,
                stride=args.stride,
                show_box=args.show_box,
                reset_fn=reset_fn,
            )
            save_trajectory_animation(
                result,
                anim_path,
                bounds=DEFAULTBOUNDS,
                fps=args.fps,
                max_frames=args.max_frames,
                show_box=args.show_box,
            )
            total_ensemble_runs += 1
            print(f"Done: {scenario['label']} | {regime}")

    total_trajectories = total_ensemble_runs * args.n_traj
    print(f"Scenario class: {scenario_class}")
    print(f"Classifier type: {args.classifier_type}")
    print(f"Ensemble runs: {total_ensemble_runs}")
    print(f"Trajectory simulations: {total_trajectories}")
    print(f"Output files: {total_ensemble_runs * 2}")


if __name__ == "__main__":
    main()
