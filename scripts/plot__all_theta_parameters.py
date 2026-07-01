
#!/usr/bin/env python3
"""
Plot theta-like scenario parameters across all configured scenarios.

This script loads SCENARIOS from the project's config package, extracts all
numeric scenario-level parameters and param_overrides, keeps only parameter names
starting with 'theta' by default, and generates:

1) A heatmap of parameter values by scenario.
2) A line plot comparing parameter values across scenarios.
3) A CSV table of the extracted values.

Usage:
python scripts/plot_all_theta_parameters_v2.py
python scripts/plot_all_theta_parameters_v2.py --include-defaults
python scripts/plot_all_theta_parameters_v2.py --include-defaults --normalize
python scripts/plot_all_theta_parameters_v2.py --prefix=

The script is designed to be tolerant of mild variation in scenario structure.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_PARAMS, SCENARIOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot theta parameters for all scenarios.")
    parser.add_argument("--output-dir", default="output/theta_parameter_plots", help="Directory for output files.")
    parser.add_argument("--prefix", default="theta", help="Keep only parameter names that start with this prefix. Use empty string to keep all numeric parameters.")
    parser.add_argument("--include-defaults", action="store_true", help="Include numeric values from DEFAULT_PARAMS in the extracted table, overridden by each scenario's param_overrides where applicable.")
    parser.add_argument("--normalize", action="store_true", help="Also save a normalized heatmap using column-wise z-scores when possible.")
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI.")
    return parser.parse_args()


def is_number(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(float(x))


def scenario_label(s: dict, idx: int) -> str:
    return str(s.get("label") or s.get("name") or s.get("scenario") or f"scenario_{idx+1}")


def extract_scenario_numeric_params(s: dict, include_defaults: bool) -> dict:
    values = {}
    if include_defaults:
        for k, v in dict(DEFAULT_PARAMS).items():
            if is_number(v):
                values[str(k)] = float(v)

    for k, v in s.items():
        if k == "param_overrides":
            continue
        if is_number(v):
            values[str(k)] = float(v)

    overrides = s.get("param_overrides", {})
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            if is_number(v):
                values[str(k)] = float(v)
    return values


def build_parameter_table(prefix: str, include_defaults: bool) -> pd.DataFrame:
    rows = []
    for idx, scenario in enumerate(SCENARIOS):
        label = scenario_label(scenario, idx)
        params = extract_scenario_numeric_params(scenario, include_defaults=include_defaults)
        if prefix:
            params = {k: v for k, v in params.items() if str(k).startswith(prefix)}
        row = {"scenario": label, **params}
        rows.append(row)

    df = pd.DataFrame(rows)
    if "scenario" not in df.columns:
        raise ValueError("No scenario labels could be extracted.")
    df = df.set_index("scenario")
    df = df.dropna(axis=1, how="all")
    if df.empty or df.shape[1] == 0:
        raise ValueError("No matching numeric parameters were found. Try a different --prefix or use --prefix '' equivalent by editing the command to --prefix=.")
    return df.sort_index(axis=1)


def plot_heatmap(df: pd.DataFrame, outpath: Path, title: str, dpi: int) -> None:
    data = df.to_numpy(dtype=float)
    n_rows, n_cols = data.shape
    fig_w = max(9, 0.7 * n_cols + 4)
    fig_h = max(5, 0.45 * n_rows + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(df.index)
    ax.set_title(title, fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Parameter value")
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_normalized_heatmap(df: pd.DataFrame, outpath: Path, title: str, dpi: int) -> None:
    work = df.copy()
    for col in work.columns:
        vals = work[col].to_numpy(dtype=float)
        mu = np.nanmean(vals)
        sd = np.nanstd(vals)
        if sd > 0:
            work[col] = (vals - mu) / sd
        else:
            work[col] = 0.0
    data = work.to_numpy(dtype=float)
    n_rows, n_cols = data.shape
    fig_w = max(9, 0.7 * n_cols + 4)
    fig_h = max(5, 0.45 * n_rows + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(work.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(work.index)
    ax.set_title(title, fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Column-wise z-score")
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def plot_lines(df: pd.DataFrame, outpath: Path, title: str, dpi: int) -> None:
    fig_w = max(10, 0.7 * len(df.columns) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5), constrained_layout=True)
    x = np.arange(len(df.columns))
    for scenario in df.index:
        ax.plot(x, df.loc[scenario].to_numpy(dtype=float), marker="o", linewidth=1.8, label=scenario)
    ax.set_xticks(x)
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_ylabel("Parameter value")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=8, ncol=2)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_parameter_table(prefix=args.prefix, include_defaults=args.include_defaults)
    safe_prefix = args.prefix if args.prefix else "all_numeric"

    csv_path = output_dir / f"{safe_prefix}_parameters_by_scenario.csv"
    heatmap_path = output_dir / f"{safe_prefix}_parameters_heatmap.png"
    lines_path = output_dir / f"{safe_prefix}_parameters_lines.png"

    df.to_csv(csv_path)
    plot_heatmap(df, heatmap_path, title=f"Scenario parameters ({safe_prefix})", dpi=args.dpi)
    plot_lines(df, lines_path, title=f"Scenario parameter profiles ({safe_prefix})", dpi=args.dpi)

    print(f"Scenarios: {len(df.index)}")
    print(f"Parameters: {len(df.columns)}")
    print(f"CSV: {csv_path}")
    print(f"Heatmap: {heatmap_path}")
    print(f"Line plot: {lines_path}")

    if args.normalize:
        norm_path = output_dir / f"{safe_prefix}_parameters_heatmap_zscore.png"
        plot_normalized_heatmap(df, norm_path, title=f"Scenario parameters ({safe_prefix}, z-score)", dpi=args.dpi)
        print(f"Normalized heatmap: {norm_path}")


if __name__ == "__main__":
    main()
