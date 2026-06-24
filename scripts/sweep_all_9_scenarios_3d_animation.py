#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
ANIMATE_SCRIPT = SCRIPTS_DIR / "animate_scenario_3d.py"
DEFAULT_SUMMARY = ROOT / "figures" / "all_9_scenarios_3classifiers_from_x0" / "sweep_summary_calls.txt"
DEFAULT_OUTDIR = ROOT / "figures" / "all_9_scenarios_3d_animations_from_calls"
DEFAULT_LOG = "animation_calls_log.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse sweep_summary_calls.txt and run the matching 3D animation sweep with corrected naming and max_frames capped at 100."
    )
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Path to sweep_summary_calls.txt.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTDIR), help="Directory for generated GIFs and call log.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for animations.")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum frames per animation; capped at 100.")
    parser.add_argument("--c-stat", choices=["mean", "median", "min", "max"], default="mean", help="C aggregation method passed to animate_scenario_3d.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def parse_summary_calls(summary_text: str) -> list[dict[str, str | list[str]]]:
    entries: list[dict[str, str | list[str]]] = []
    lines = [line.strip() for line in summary_text.splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("["):
            i += 1
            continue
        if i + 1 >= len(lines):
            break

        call_line = lines[i + 1]
        filter_match = re.search(r"--filter\s+(.+?)\s+--prefix\s+", call_line)
        prefix_match = re.search(r"--prefix\s+([^\s]+)", call_line)
        ntraj_match = re.search(r"--n-traj\s+([^\s]+)", call_line)
        x0_match = re.search(r"--x0\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)", call_line)
        elev_match = re.search(r"--elev\s+([^\s]+)", call_line)
        azim_match = re.search(r"--azim\s+([^\s]+)", call_line)
        show_box = "--show-box" in call_line

        if not all([filter_match, prefix_match, ntraj_match, x0_match, elev_match, azim_match]):
            raise ValueError(f"Could not parse summary call line: {call_line}")

        entries.append(
            {
                "filter": filter_match.group(1),
                "prefix": prefix_match.group(1),
                "n_traj": ntraj_match.group(1),
                "x0": [
                    x0_match.group(1),
                    x0_match.group(2),
                    x0_match.group(3),
                    x0_match.group(4),
                ],
                "elev": elev_match.group(1),
                "azim": azim_match.group(1),
                "show_box": "1" if show_box else "0",
            }
        )
        i += 2
    return entries


def build_animation_command(args: argparse.Namespace, entry: dict[str, str | list[str]]) -> tuple[list[str], Path]:
    output_path = Path(args.out_dir) / f"{entry['prefix']}_3d.gif"
    x0 = entry["x0"]
    assert isinstance(x0, list)

    cmd = [
        args.python,
        str(ANIMATE_SCRIPT),
        "--filter",
        str(entry["filter"]),
        "--output",
        str(output_path),
        "--fps",
        str(args.fps),
        "--max-frames",
        str(min(args.max_frames, 100)),
        "--n-traj",
        str(entry["n_traj"]),
        "--x0",
        *x0,
        "--elev",
        str(entry["elev"]),
        "--azim",
        str(entry["azim"]),
        "--c-stat",
        args.c_stat,
    ]
    if entry["show_box"] == "1":
        cmd.append("--show-box")
    return cmd, output_path


def main() -> None:
    args = parse_args()
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find summary file: {summary_path}")

    entries = parse_summary_calls(summary_path.read_text(encoding="utf-8"))
    if not entries:
        raise RuntimeError("No calls were parsed from the summary file.")

    log_path = args.out_dir / DEFAULT_LOG
    log_lines: list[str] = []

    for idx, entry in enumerate(entries, start=1):
        cmd, output_path = build_animation_command(args, entry)
        header = f"[{idx:02d}/{len(entries):02d}] {entry['filter']} -> {output_path.name}"
        cmd_str = " ".join(cmd)
        print(header)
        print(cmd_str)
        log_lines.extend([header, cmd_str, ""])

        if args.dry_run:
            continue

        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Animation command failed with exit code {completed.returncode}: {cmd_str}")

    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Parsed calls: {len(entries)}")
    print(f"Animation call log: {log_path}")


if __name__ == "__main__":
    main()
