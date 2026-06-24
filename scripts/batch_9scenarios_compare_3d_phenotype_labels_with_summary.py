"""
Generate 3D phenotype-label figures for 27 cases across 9 scenarios and save the visualization settings.
FIXME: do it for any 3 vars and bounds

For each of the 9 scenarios, the script automatically selects three initial-point regimes:
inside the viability bounds, inside but near the viability boundary, and outside
the viability boundary. This produces 27 cases in total. For each case, it creates one
figure that shows the phenotype labels from all 3 classifiers side by side, and it saves
the selected initial points and visualization parameters to an output text file.

Usage example:
    python scripts/batch_9scenarios_compare_3d_phenotype_labels_with_summary.py \
        --out-dir figures/all_9_compare_3d --show-box
"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_BOUNDS
from plotting.scenario_helpers import choose_scenario, get_base_x0_center

SCRIPTS_DIR = ROOT / "scripts"
SINGLE_SCRIPT = SCRIPTS_DIR / "single_scenario_3d_3classifiers.py"
SUMMARY_NAME = "sweep_summary_calls.txt"
DEFAULT_OUTDIR = ROOT / "figures" / "all_9_scenarios_3classifiers_from_x0"

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


def classify_eto_regime(T: float, E: float, O: float) -> str:
    inside = (
        DEFAULT_BOUNDS["T_min"] <= T <= DEFAULT_BOUNDS["T_max"]
        and DEFAULT_BOUNDS["E_min"] <= E <= DEFAULT_BOUNDS["E_max"]
        and O >= DEFAULT_BOUNDS["O_min"]
    )
    if not inside:
        return "outside"
    near = (
        abs(T - DEFAULT_BOUNDS["T_min"]) <= 0.12
        or abs(T - DEFAULT_BOUNDS["T_max"]) <= 0.12
        or abs(E - DEFAULT_BOUNDS["E_min"]) <= 0.12
        or abs(E - DEFAULT_BOUNDS["E_max"]) <= 0.12
        or abs(O - DEFAULT_BOUNDS["O_min"]) <= 0.08
    )
    return "near" if near else "inside"


def find_point_from_base(
    base_x0_center,
    regime: str,
) -> tuple[float, float, float, float]:
    base_C, _, _, _ = [float(v) for v in base_x0_center]

    candidates = {
        "inside": [
            (base_C, 0.45, 0.35, 0.65),
            (base_C, 0.70, 0.60, 0.55),
        ],
        "near": [
            (base_C, 1.40, 1.65, 0.24),
            (base_C, 1.46, 1.74, 0.22),
        ],
        "outside": [
            (base_C, 1.56, 1.86, 0.18),
            (base_C, 1.62, 1.92, 0.16),
            (base_C, 1.52, 1.84, 0.19),
        ],
    }

    for x0_center in candidates[regime]:
        _, T, E, O = x0_center
        if classify_eto_regime(T, E, O) == regime:
            return x0_center

    raise RuntimeError(f"Could not build a valid {regime} x0_center from base x0_center={base_x0_center}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep 9 scenarios using exported base x0_center values and explicit 4D initial points, then save a summary of all calls."
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTDIR), help="Directory for saved figures and summary file.")
    parser.add_argument("--n-traj", type=int, default=40, help="Number of trajectories per run.")
    parser.add_argument("--stride", type=int, default=8, help="Subsample factor for taxonomy plots.")
    parser.add_argument("--elev", type=float, default=24.0, help="3D camera elevation.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth.")
    parser.add_argument("--show-box", action="store_true", help="Show translucent viability box.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def build_command(
    args: argparse.Namespace,
    scenario_filter: str,
    prefix: str,
    x0_center: tuple[float, float, float, float],
) -> list[str]:
    C, T, E, O = x0_center
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
        "--x0",
        f"{C:.10f}",
        f"{T:.10f}",
        f"{E:.10f}",
        f"{O:.10f}",
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

    summary_path = args.out_dir / SUMMARY_NAME
    summary_lines: list[str] = []
    planned_cmds: list[list[str]] = []

    total = 0
    for scenario_filter, slug in TARGET_SCENARIOS:
        scenario = choose_scenario(scenario_filter)
        base_x0_center, _ = get_base_x0_center(scenario)
        base_C, base_T, base_E, base_O = [float(v) for v in base_x0_center]

        for regime in REGIME_ORDER:
            x0_center = find_point_from_base(base_x0_center, regime)
            C, T, E, O = x0_center
            prefix = f"{slug}__{regime}"
            cmd = build_command(args, scenario_filter, prefix, x0_center)
            planned_cmds.append(cmd)
            total += 1

            header = (
                f"[{total:02d}/27] {scenario['label']} | requested_start_regime={regime} | "
                f"base_x0=(C={base_C:.4f}, T={base_T:.4f}, E={base_E:.4f}, O={base_O:.4f}) | "
                f"x0=(C={C:.4f}, T={T:.4f}, E={E:.4f}, O={O:.4f}) | "
                f"classified_start={classify_eto_regime(T, E, O)}"
            )
            cmd_str = " ".join(cmd)
            saved_line = f"Expected output: {expected_output(prefix, args.out_dir)}"
            summary_lines.extend([header, cmd_str, saved_line, ""])

    summary_lines.append(f"Completed planned runs: {total}")
    summary_lines.append(f"Summary file: {summary_path}")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    if args.dry_run:
        print(f"Dry run complete. Summary written to: {summary_path}")
        return

    for cmd in planned_cmds:
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")

    print(f"Completed {len(planned_cmds)} runs.")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
