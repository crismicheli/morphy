#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SINGLE_SCRIPT = SCRIPTS_DIR / "single_scenario_3d_3classifiers.py"
DEFAULT_OUTDIR = ROOT / "figures" / "all_9_scenarios_3classifiers"

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

REGIMES = [
    ("outside", {"shift_T": 1.56 / 1.40, "shift_E": 1.86 / 1.65, "shift_O": 0.18 / 0.24}),
    ("inside", {"shift_T": 0.45 / 1.40, "shift_E": 0.35 / 1.65, "shift_O": 0.65 / 0.24}),
    ("near", {"shift_T": 1.0, "shift_E": 1.0, "shift_O": 1.0}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrapper over single_scenario_3d_3classifiers.py to sweep 9 scenarios across outside/inside/near starts and save 27 taxonomy figures."
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTDIR), help="Directory for saved figures.")
    parser.add_argument("--n-traj", type=int, default=40, help="Number of trajectories per run.")
    parser.add_argument("--stride", type=int, default=8, help="Subsample factor for taxonomy plots.")
    parser.add_argument("--elev", type=float, default=24.0, help="3D camera elevation.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth.")
    parser.add_argument("--show-box", action="store_true", help="Show translucent viability box.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def expected_output(prefix: str, out_dir: Path) -> Path:
    return out_dir / f"{prefix}_taxonomy_3d_3classifiers.png"


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
        str(shifts["shift_T"]),
        "--shift-E",
        str(shifts["shift_E"]),
        "--shift-O",
        str(shifts["shift_O"]),
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


def main() -> None:
    args = parse_args()
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not SINGLE_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find wrapper target script: {SINGLE_SCRIPT}")

    total = 0
    for scenario_filter, slug in TARGET_SCENARIOS:
        for regime_name, shifts in REGIMES:
            prefix = f"{slug}__{regime_name}"
            cmd = build_command(args, scenario_filter, prefix, shifts)
            out_png = expected_output(prefix, args.out_dir)
            total += 1

            print(f"[{total:02d}/27] {scenario_filter} | {regime_name}")
            print(" ".join(cmd))

            if args.dry_run:
                continue

            completed = subprocess.run(cmd, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Command failed for scenario={scenario_filter!r}, regime={regime_name!r} with exit code {completed.returncode}"
                )
            print(f"Saved: {out_png}")

    print(f"Completed {total} runs.")
    print(f"Expected taxonomy figures: {total}")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
