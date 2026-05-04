#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_BOUNDS, DEFAULT_PARAMS, DEFAULT_SIM, SCENARIOS
from viabilitykernels.simulation import run_scenario, sample_initial_conditions
from plotting.plot_helpers import save_taxonomy_plot, save_trajectory_animation
from plotting.scenario_helpers import scenario_slug
from classifiers.classifier_dispatch import get_classifier_components

TARGET_SCENARIOS = [
    "Low porosity",
    "Intermediate porosity",
    "High porosity",
    "Stiff scaffold",
    "Hypoxic environment",
    "Over-tensioned",
    "Fast ECM remodelling",
    "Enhanced guidance",
    "Near-critical asymmetric regime",
]

REGIME_ORDER = [
    "outside",
    "inside",
    "near",
]


def choose_target_scenarios() -> list[dict]:
    selected = []
    missing = []
    for needle in TARGET_SCENARIOS:
        matches = [s for s in SCENARIOS if needle.lower() in str(s.get("label", "")).lower()]
        if not matches:
            missing.append(needle)
            continue
        selected.append(matches[0])
    if missing:
        available = ", ".join(str(s.get("label", "<unnamed>")) for s in SCENARIOS)
        raise ValueError(
            "Could not find scenario labels for: "
            + ", ".join(missing)
            + f". Available scenarios: {available}"
        )
    return selected


def make_regime_center(regime: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    if regime == "inside":
        center = np.array([0.22, 0.45, 0.35, 0.65], dtype=float)
        noise = (0.02, 0.03, 0.03, 0.03)
    elif regime == "near":
        center = np.array([0.16, 1.40, 1.65, 0.24], dtype=float)
        noise = (0.01, 0.04, 0.05, 0.02)
    elif regime == "outside":
        center = np.array([0.14, 1.56, 1.86, 0.18], dtype=float)
        noise = (0.01, 0.04, 0.05, 0.02)
    else:
        raise ValueError(f"Unknown regime: {regime}")
    return center, noise


def is_inside_viability_box(x0: np.ndarray, bounds: dict) -> bool:
    C, T, E, O = [float(v) for v in x0]
    return (
        C >= bounds["C_min"]
        and bounds["T_min"] <= T <= bounds["T_max"]
        and bounds["E_min"] <= E <= bounds["E_max"]
        and O >= bounds["O_min"]
    )


def warn_initial_conditions(initial_conditions: list[np.ndarray], bounds: dict, tag: str, regime: str) -> None:
    outside_count = sum(0 if is_inside_viability_box(x0, bounds) else 1 for x0 in initial_conditions)
    if regime == "outside" and outside_count == 0:
        warnings.warn(f"{tag}: expected outside starts, but all sampled initial conditions are inside the viability box.")
    if regime == "inside" and outside_count > 0:
        warnings.warn(f"{tag}: expected inside starts, but {outside_count}/{len(initial_conditions)} initial conditions are outside the viability box.")
    if regime == "near":
        warnings.warn(
            f"{tag}: near-boundary regime produced {outside_count}/{len(initial_conditions)} initial conditions outside the viability box; mixed starts are acceptable."
        )


def run_regime_scenario(scenario: dict, regime: str, *, n_traj: int, rng_seed: int):
    x0_center, noise_scale = make_regime_center(regime)
    initial_conditions = sample_initial_conditions(
        x0_center=x0_center,
        n_traj=n_traj,
        noise_scale=noise_scale,
        rng_seed=rng_seed,
    )
    warn_initial_conditions(initial_conditions, DEFAULT_BOUNDS, f"{scenario['label']} | {regime}", regime)
    result = run_scenario(
        scenario_cfg=scenario,
        par=DEFAULT_PARAMS,
        bounds=DEFAULT_BOUNDS,
        x0_center=x0_center,
        n_traj=n_traj,
        t_span=tuple(DEFAULT_SIM["t_span"]),
        n_eval=DEFAULT_SIM["n_eval"],
        rng_seed=rng_seed,
        noise_scale=noise_scale,
        initial_conditions=initial_conditions,
    )
    result["regime"] = regime
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep all 9 named scenarios across outside, inside, and near-boundary initial-condition regimes."
    )
    parser.add_argument(
        "--classifier-type",
        choices=["static", "temporal", "state_machine"],
        default="state_machine",
        help="Classifier backend for taxonomy plots.",
    )
    parser.add_argument(
        "--n-traj",
        type=int,
        default=DEFAULT_SIM["n_traj"],
        help="Number of trajectories per run.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Subsample factor for taxonomy plot timepoints.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for animations.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Maximum animation frames.",
    )
    parser.add_argument(
        "--show-box",
        action="store_true",
        help="Show translucent viability box in outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory. Defaults to figures/all_9_scenarios_sweep_3d.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=DEFAULT_SIM["rng_seed"],
        help="Base random seed for initial-condition sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = choose_target_scenarios()
    classifier_fn, reset_fn, state_colors = get_classifier_components(args.classifier_type)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "figures" / "all_9_scenarios_sweep_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_runs = 0
    for scenario_idx, scenario in enumerate(scenarios):
        slug = scenario_slug(scenario["label"])
        for regime_idx, regime in enumerate(REGIME_ORDER):
            rng_seed = args.rng_seed + 100 * scenario_idx + regime_idx
            result = run_regime_scenario(scenario, regime, n_traj=args.n_traj, rng_seed=rng_seed)
            plot_path = out_dir / f"{slug}__{regime}__taxonomy_3d__clf-{args.classifier_type}.png"
            anim_path = out_dir / f"{slug}__{regime}__eto_3d.gif"

            save_taxonomy_plot(
                result,
                scenario,
                plot_path,
                bounds=DEFAULT_BOUNDS,
                par=DEFAULT_PARAMS,
                classifier_fn=classifier_fn,
                color_map=state_colors,
                stride=args.stride,
                show_box=args.show_box,
                reset_fn=reset_fn,
            )
            save_trajectory_animation(
                result,
                anim_path,
                bounds=DEFAULT_BOUNDS,
                fps=args.fps,
                max_frames=args.max_frames,
                show_box=args.show_box,
            )

            total_runs += 1
            print(f"Done: {scenario['label']} | {regime}")

    print(f"Scenarios: {len(scenarios)}")
    print(f"Regimes per scenario: {len(REGIME_ORDER)}")
    print(f"Total ensemble runs: {total_runs}")
    print(f"Total output files: {2 * total_runs}")


if __name__ == "__main__":
    main()
