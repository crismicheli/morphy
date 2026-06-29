#!/usr/bin/env python3
"""
Export point-only trajectory records for one selected morphodynamic scenario.

This script generates an arbitrary number of seeded trajectories, integrates
them through the full 4D scaffold-cell-matrix ODE system, deduplicates
trajectories by their initial conditions, estimates a reference attractor, and
writes a flat pointwise dataset in which each row corresponds to one sampled
time point from one deduplicated trajectory.

The exported table is intentionally point-centric. Trajectory objects are used
internally only for:
- initial-condition deduplication,
- time integration,
- and point timestamp provenance.

No trajectory summary table is written. The only output is a point-record CSV.

Per-point fields
----------------
Each exported point record contains:
- full 4D position: C, T, E, O
- viability label: viable / non_viable
- phenotypical label: classifier output for that point
- 4D Euclidean distance from the attractor
- timestamp within its parent trajectory

Model conventions
-----------------
- Full state order is always (C, T, E, O).
- Each trajectory is integrated in the full 4D system.
- The attractor distance is computed in full 4D, not in ETO only.
- Per-point labels come from one classifier selected by --classifier-type.

Typical usage
-------------
1) Export 200 trajectories around the default initial center using the static classifier:
   python scripts/export_trajectory_point_records.py \
       --filter "High porosity" \
       --n-traj 200 \
       --classifier-type static

2) Export 500 trajectories with temporal classification:
   python scripts/export_trajectory_point_records.py \
       --filter "Intermediate porosity" \
       --n-traj 500 \
       --classifier-type temporal

3) Export trajectories from a custom center point:
   python scripts/export_trajectory_point_records.py \
       --filter "Enhanced guidance" \
       --n-traj 300 \
       --x0-center 0.20 0.15 0.10 0.60 \
       --classifier-type state_machine

4) Export to a custom output directory and deduplicate more aggressively:
   python scripts/export_trajectory_point_records.py \
       --filter "Hypoxic environment" \
       --n-traj 250 \
       --dedup-decimals 6 \
       --output-dir output/traj_exports

What gets written
-----------------
One CSV file is written into a scenario-specific folder such as:

    output/high_porosity_point_records/

with a filename such as:

    high_porosity_point_records_temporal.csv

Notes
-----
- Initial conditions are sampled the same way the repository samples scenario
  ensembles: perturbations around a center point with configurable noise scale.
- Deduplication happens at the seed / initial-condition level after rounding to
  --dedup-decimals decimal places.
- For temporal or state-machine classifiers, classifier memory is reset once per
  trajectory so labels are trajectory-local rather than leaking across runs.
- The attractor is estimated by integrating from the chosen reference center and
  taking the terminal 4D state.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifiers.classifier_dispatch import get_classifier_components
from config import DEFAULT_BOUNDS, DEFAULT_PARAMS, DEFAULT_SIM
from plotting.scenario_helpers import choose_scenario
from viabilitykernels.simulation import integrate_trajectory, sample_initial_conditions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export point-only trajectory records for one scenario."
    )
    parser.add_argument(
        "--filter",
        default="Intermediate porosity",
        help="Substring used to choose a scenario label.",
    )
    parser.add_argument(
        "--classifier-type",
        choices=["static", "temporal", "state_machine"],
        default="static",
        help="Classifier used to label every trajectory point.",
    )
    parser.add_argument(
        "--n-traj",
        type=int,
        default=200,
        help="Requested number of sampled trajectories before deduplication.",
    )
    parser.add_argument(
        "--x0-center",
        type=float,
        nargs=4,
        default=None,
        metavar=("C0", "T0", "E0", "O0"),
        help="Optional explicit initial-condition center in (C, T, E, O) order.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        nargs=4,
        default=None,
        metavar=("C_SIG", "T_SIG", "E_SIG", "O_SIG"),
        help="Optional noise scale for initial-condition sampling in (C, T, E, O) order.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=int(DEFAULT_SIM["rng_seed"]),
        help="Random seed used for initial-condition sampling.",
    )
    parser.add_argument(
        "--dedup-decimals",
        type=int,
        default=8,
        help="Decimal precision used when deduplicating initial conditions.",
    )
    parser.add_argument(
        "--t-final",
        type=float,
        default=float(DEFAULT_SIM["t_span"][1]),
        help="Final integration time.",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=int(DEFAULT_SIM["n_eval"]),
        help="Number of sampled time points per trajectory.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where the point-record CSV is written.",
    )
    return parser.parse_args()


def minimal_scenario_name(label: str) -> str:
    """
    Convert a verbose scenario label into a compact, filesystem-friendly stem.

    Examples
    --------
    "High porosity  (p=0.75) — borderline" -> "high_porosity"
    "Hypoxic environment  (ρ=0.3, s=0.4) — unstable" -> "hypoxic_environment"
    "Enhanced guidance  (a=6.0) — stable" -> "enhanced_guidance"
    """
    head = label.split("(")[0]
    head = head.split("—")[0]
    head = head.strip().lower()

    head = unicodedata.normalize("NFKD", head)
    head = head.encode("ascii", "ignore").decode("ascii")

    head = head.replace("-", "_")
    head = re.sub(r"[^a-z0-9_ ]+", "", head)
    head = re.sub(r"\s+", "_", head)
    head = re.sub(r"_+", "_", head).strip("_")

    return head or "scenario"


def inside_viability(C: float, T: float, E: float, O: float, bounds: dict) -> bool:
    """
    Pointwise viability check in full 4D state space.
    """
    return (
        C >= float(bounds["C_min"])
        and float(bounds["T_min"]) <= T <= float(bounds["T_max"])
        and float(bounds["E_min"]) <= E <= float(bounds["E_max"])
        and O >= float(bounds["O_min"])
    )


def deduplicate_initial_conditions(
    initial_conditions,
    *,
    decimals: int,
) -> list[np.ndarray]:
    """
    Deduplicate initial conditions by rounding and hashing the 4D seed vector.

    Deduplication is done before integration so that repeated or near-identical
    seeds do not produce redundant trajectories.
    """
    seen = set()
    unique = []

    for x0 in initial_conditions:
        arr = np.asarray(x0, dtype=float)
        key = tuple(np.round(arr, decimals=decimals).tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append(arr)

    return unique


def effective_parameters_for_scenario(scenario_cfg: dict, par: dict) -> dict:
    """
    Merge scenario-specific parameter overrides onto the default parameter set.
    """
    effective = dict(par)
    effective.update(scenario_cfg.get("param_overrides", {}))
    return effective


def estimate_attractor_4d(
    scenario_cfg: dict,
    *,
    par: dict,
    x0_center: np.ndarray,
    t_final: float,
    n_eval: int,
) -> np.ndarray:
    """
    Estimate a reference attractor by integrating from the chosen center point
    and returning the terminal 4D state.
    """
    effective_par = effective_parameters_for_scenario(scenario_cfg, par)
    sol = integrate_trajectory(
        x0=np.asarray(x0_center, dtype=float),
        p=float(scenario_cfg["p"]),
        par=effective_par,
        t_span=(0.0, float(t_final)),
        n_eval=int(n_eval),
        rtol=float(DEFAULT_SIM["rtol"]),
        atol=float(DEFAULT_SIM["atol"]),
    )
    return np.asarray(sol.y[:, -1], dtype=float)


def compute_derivatives(sol) -> np.ndarray:
    """
    Approximate pointwise derivatives from a sampled solution array.
    """
    dt = max(1e-12, float(np.mean(np.diff(sol.t))))
    return np.gradient(sol.y, dt, axis=1)


def build_point_records(
    *,
    scenario_cfg: dict,
    classifier_type: str,
    classify_fn,
    reset_fn,
    effective_par: dict,
    bounds: dict,
    attractor_4d: np.ndarray,
    trajectories: list,
    initial_conditions: list[np.ndarray],
) -> list[dict]:
    """
    Build the flat point-record table.

    One row is emitted per sampled time point. Trajectory objects are not
    exported; only pointwise data are written.
    """
    point_records: list[dict] = []
    scenario_name = minimal_scenario_name(scenario_cfg["label"])

    for traj_idx, (x0, sol) in enumerate(zip(initial_conditions, trajectories), start=1):
        if reset_fn is not None:
            reset_fn()

        dydt = compute_derivatives(sol)

        for i in range(sol.y.shape[1]):
            timestamp = float(sol.t[i])
            C, T, E, O = (float(v) for v in sol.y[:, i])
            dC, dT, dE, dO = (float(v) for v in dydt[:, i])

            point_viable = inside_viability(C, T, E, O, bounds)
            viability_label = "viable" if point_viable else "non_viable"

            phenotypical_label = classify_fn(
                C, T, E, O, dC, dT, dE, dO,
                bounds=bounds,
                par=effective_par,
                scenario_cfg=scenario_cfg,
            )

            point_state_4d = np.array([C, T, E, O], dtype=float)
            distance_from_attractor_4d = float(np.linalg.norm(point_state_4d - attractor_4d))

            point_records.append(
                {
                    "scenario_label": scenario_cfg["label"],
                    "scenario_name": scenario_name,
                    "scenario_expected": scenario_cfg.get("expected", ""),
                    "classifier_type": classifier_type,
                    "trajectory_id": traj_idx,
                    "point_index": i,
                    "timestamp": timestamp,
                    "x0_C": float(x0[0]),
                    "x0_T": float(x0[1]),
                    "x0_E": float(x0[2]),
                    "x0_O": float(x0[3]),
                    "C": C,
                    "T": T,
                    "E": E,
                    "O": O,
                    "viability_label": viability_label,
                    "phenotypical_label": phenotypical_label,
                    "distance_from_attractor_4d": distance_from_attractor_4d,
                }
            )

    return point_records


def write_csv(rows: list[dict], output_path: Path) -> None:
    """
    Write a list of dict rows to CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {output_path}")

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    scenario_cfg = choose_scenario(args.filter)
    scenario_name = minimal_scenario_name(scenario_cfg["label"])

    x0_center = (
        np.asarray(args.x0_center, dtype=float)
        if args.x0_center is not None
        else np.asarray(DEFAULT_SIM["x0_center"], dtype=float)
    )
    noise_scale = (
        tuple(float(v) for v in args.noise_scale)
        if args.noise_scale is not None
        else tuple(float(v) for v in DEFAULT_SIM["noise_scale"])
    )

    sampled_ics = sample_initial_conditions(
        x0_center=x0_center,
        n_traj=int(args.n_traj),
        noise_scale=noise_scale,
        rng_seed=int(args.rng_seed),
    )
    unique_ics = deduplicate_initial_conditions(
        sampled_ics,
        decimals=int(args.dedup_decimals),
    )

    effective_par = effective_parameters_for_scenario(scenario_cfg, DEFAULT_PARAMS)
    classify_fn, reset_fn, _ = get_classifier_components(args.classifier_type)

    attractor_4d = estimate_attractor_4d(
        scenario_cfg,
        par=DEFAULT_PARAMS,
        x0_center=x0_center,
        t_final=float(args.t_final),
        n_eval=int(args.n_eval),
    )

    trajectories = [
        integrate_trajectory(
            x0=np.asarray(x0, dtype=float),
            p=float(scenario_cfg["p"]),
            par=effective_par,
            t_span=(0.0, float(args.t_final)),
            n_eval=int(args.n_eval),
            rtol=float(DEFAULT_SIM["rtol"]),
            atol=float(DEFAULT_SIM["atol"]),
        )
        for x0 in unique_ics
    ]

    point_records = build_point_records(
        scenario_cfg=scenario_cfg,
        classifier_type=args.classifier_type,
        classify_fn=classify_fn,
        reset_fn=reset_fn,
        effective_par=effective_par,
        bounds=DEFAULT_BOUNDS,
        attractor_4d=attractor_4d,
        trajectories=trajectories,
        initial_conditions=unique_ics,
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    scenario_dir = output_dir / f"{scenario_name}_point_records"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    point_csv = scenario_dir / f"{scenario_name}_point_records_{args.classifier_type}.csv"
    write_csv(point_records, point_csv)

    print(f"Scenario: {scenario_cfg['label']}")
    print(f"Requested trajectories: {args.n_traj}")
    print(f"Unique trajectories after deduplication: {len(unique_ics)}")
    print(f"Point records written: {len(point_records)}")
    print(f"Point CSV: {point_csv}")


if __name__ == "__main__":
    main()
