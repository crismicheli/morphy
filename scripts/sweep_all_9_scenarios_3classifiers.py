#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SINGLE_SCRIPT = SCRIPTS_DIR / "single_scenario_3d_3classifiers.py"
SUMMARY_NAME = "sweep_summary_calls.txt"
DEFAULT_OUTDIR = ROOT / "figures" / "all_9_scenarios_3classifiers_strict"

TARGET_SCENARIOS = [
    ("Low porosity", "low_porosity_p015"),
    ("Intermediate porosity", "intermediate_porosity_p040"),
    ("High porosity", "high_porosity_p075"),
    ("Stiff scaffold", "stiff_scaffold_eta18"),
    ("Hypoxic environment", "hypoxic_environment"),
    ("Over-tensioned", "over_tensioned_beta35"),
    ("Fast ECM remodelling", "fast_ecm_remodelling_deltae12"),
    ("Enhanced guidance", "enhanced_guidance_a60"),
    ("Near-critical asymmetric regime", "near_critical_asymmetric"),
]

REGIME_ORDER = ["outside", "inside", "near"]

BASE_NEAR = {"T": 1.40, "E": 1.65, "O": 0.24}
STRICT_TARGETS = {
    "outside": {"T": 1.56, "E": 1.86, "O": 0.18},
    "inside": {"T": 0.45, "E": 0.35, "O": 0.65},
    "near": {"T": 1.40, "E": 1.65, "O": 0.24},
}

BOUNDS = {
    "T_min": 0.2,
    "T_max": 1.5,
    "E_min": 0.1,
    "E_max": 1.8,
    "O_min": 0.2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep 9 scenarios across strict outside/inside/near start-point regimes using single_scenario_3d_3classifiers.py and save a summary text file."
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTDIR), help="Directory for saved figures and summary file.")
    parser.add_argument("--n-traj", type=int, default=40, help="Number of trajectories per run.")
    parser.add_argument("--stride", type=int, default=8, help="Subsample factor for taxonomy plots.")
    parser.add_argument("--elev", type=float, default=24.0, help="3D camera elevation.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth.")
    parser.add_argument("--show-box", action="store_true", help="Show translucent viability box.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--random-seed", type=int, default=7, help="Seed for regime-point selection.")
    return parser.parse_args()


def classify_start_regime(T: float, E: float, O: float) -> str:
    inside = (
        BOUNDS["T_min"] <= T <= BOUNDS["T_max"]
        and BOUNDS["E_min"] <= E <= BOUNDS["E_max"]
        and O >= BOUNDS["O_min"]
    )
    if not inside:
        return "outside"

    near_boundary = (
        abs(T - BOUNDS["T_min"]) <= 0.12
        or abs(T - BOUNDS["T_max"]) <= 0.12
        or abs(E - BOUNDS["E_min"]) <= 0.12
        or abs(E - BOUNDS["E_max"]) <= 0.12
        or abs(O - BOUNDS["O_min"]) <= 0.08
    )
    return "near" if near_boundary else "inside"


def sample_start_point(regime: str, rng: np.random.Generator) -> dict[str, float]:
    target = STRICT_TARGETS[regime]
    scales = {
        "outside": np.array([0.03, 0.04, 0.025]),
        "inside": np.array([0.06, 0.08, 0.06]),
        "near": np.array([0.025, 0.03, 0.015]),
    }[regime]

    for _ in range(2000):
        candidate = np.array([target["T"], target["E"], target["O"]], dtype=float) + rng.normal(0.0, scales)
        T, E, O = map(float, candidate)
        if classify_start_regime(T, E, O) == regime:
            return {"T": T, "E": E, "O": O}
    raise RuntimeError(f"Failed to sample a valid start point for regime={regime}")


def point_to_shifts(point: dict[str, float]) -> dict[str, float]:
    return {
        "shift_T": point["T"] / BASE_NEAR["T"],
        "shift_E": point["E"] / BASE_NEAR["E"],
        "shift_O": point["O"] / BASE_NEAR["O"],
    }


def build_command(args: argparse.Namespace, scenario_filter: str, prefix: str, shifts: dict[str, float]) -> list[str]:
    cmd = [
        args.python,
        str(SINGLE_SCRIPT),
        "--filter",
        scenario_filter,
        "--prefix",
        prefix,
        "--out-dir",
        str(args.out_dir),
        "--n-traj",
        str(args.n_traj),
        "--shift-T",
        f"{shifts['shift_T']:.10f}",
        "--shift-E",
        f"{shifts['shift_E']:.10f}",
        "--shift-O",
        f"{shifts['shift_O']:.10f}",
        "--stride",
        str(args.stride),
        "--elev",
        str(args.elev),
        "--azim",
        str(args.azim),
    ]
    if args.show_box:
        cmd.append("--show-box")
    return cmd


def expected_output(prefix: str, out_dir: Path) -> Path:
    return out_dir / f"{prefix}_taxonomy_3d_3classifiers.png"


def main() -> None:
    args = parse_args()
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not SINGLE_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find target script: {SINGLE_SCRIPT}")

    rng = np.random.default_rng(args.random_seed)
    summary_path = args.out_dir / SUMMARY_NAME
    summary_lines: list[str] = []
    total = 0

    for scenario_filter, slug in TARGET_SCENARIOS:
        for regime in REGIME_ORDER:
            start_point = sample_start_point(regime, rng)
            shifts = point_to_shifts(start_point)
            prefix = f"{slug}__{regime}"
            cmd = build_command(args, scenario_filter, prefix, shifts)
            out_png = expected_output(prefix, args.out_dir)
            total += 1

            header = (
                f"[{total:02d}/27] {scenario_filter} | requested_start_regime={regime} | "
                f"start_point=(T={start_point['T']:.4f}, E={start_point['E']:.4f}, O={start_point['O']:.4f}) | "
                f"classified_start={classify_start_regime(start_point['T'], start_point['E'], start_point['O'])}"
            )
            cmd_str = " ".join(cmd)
            saved_line = f"Expected output: {out_png}"

            print(header)
            print(cmd_str)
            print(saved_line)

            summary_lines.extend([header, cmd_str, saved_line, ""])

    summary_lines.append(f"Completed planned runs: {total}")
    summary_lines.append(f"Summary file: {summary_path}")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    if args.dry_run:
        print(f"Dry run complete. Summary written to: {summary_path}")
        return

    for line in summary_lines:
        if not line.startswith("["):
            continue
        pass

    run_index = 0
    for scenario_filter, slug in TARGET_SCENARIOS:
        for regime in REGIME_ORDER:
            start_point = sample_start_point(regime, rng)
            shifts = point_to_shifts(start_point)
            prefix = f"{slug}__{regime}"
            cmd = build_command(args, scenario_filter, prefix, shifts)
            run_index += 1
            completed = subprocess.run(cmd, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Command failed for scenario={scenario_filter!r}, regime={regime!r} with exit code {completed.returncode}"
                )

    print(f"Completed {run_index} runs.")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
