
#!/usr/bin/env python3
"""
Plot all static simulation parameters used by selected scenarios.

Interpretation
--------------
"Static parameters" means the effective parameter set used to simulate each
scenario: start from DEFAULT_PARAMS, then overwrite with each scenario's
param_overrides. The resulting table therefore represents the actual parameter
values used in each simulation.

Scenario selection behavior
---------------------------
This version is deliberately forgiving:
- By default, it uses the first 9 scenarios in SCENARIOS.
- `--list-scenarios` prints all available labels.
- `--scenario-labels` performs case-insensitive substring matching by default.
- `--exact-labels` switches `--scenario-labels` to exact matching.

Outputs
-------
1) CSV table of effective static parameter values by scenario.
2) Heatmap of parameter values by scenario.
3) Line plot comparing parameter profiles across scenarios.
4) Optional normalized heatmap using column-wise z-scores.

Typical usage
-------------
python scripts/plot_all_static_parameters_for_9_scenarios.py --list-scenarios
python scripts/plot_all_static_parameters_for_9_scenarios.py --scenario-labels "High porosity" "Intermediate porosity" "Low porosity"
python scripts/plot_all_static_parameters_for_9_scenarios.py --scenario-labels porosity --drop-constant-columns
python scripts/plot_all_static_parameters_for_9_scenarios.py --scenario-labels "High porosity  (p=0.75) — borderline" --exact-labels
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
    parser = argparse.ArgumentParser(description="Plot all static simulation parameters across scenarios.")
    parser.add_argument("--output-dir", default="output/static_parameter_plots", help="Directory for output files.")
    parser.add_argument("--scenario-labels", nargs="*", default=None, help="Scenario selectors. By default these are treated as case-insensitive substrings, not exact labels.")
    parser.add_argument("--exact-labels", action="store_true", help="Interpret --scenario-labels as exact labels instead of substring matches.")
    parser.add_argument("--list-scenarios", action="store_true", help="Print all available scenario labels and exit.")
    parser.add_argument("--drop-constant-columns", action="store_true", help="Drop parameters that have the same value in every selected scenario.")
    parser.add_argument("--normalize", action="store_true", help="Also save a normalized heatmap using column-wise z-scores when possible.")
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI.")
    return parser.parse_args()


def is_number(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(float(x))


def scenario_label(s: dict, idx: int) -> str:
    return str(s.get("label") or s.get("name") or s.get("scenario") or f"scenario_{idx+1}")


def effective_parameters_for_scenario(scenario: dict) -> dict:
    params = {}
    for k, v in dict(DEFAULT_PARAMS).items():
        if is_number(v):
            params[str(k)] = float(v)
    overrides = scenario.get("param_overrides", {})
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            if is_number(v):
                params[str(k)] = float(v)
    return params


def available_scenarios() -> list[tuple[str, dict]]:
    return [(scenario_label(s, i), s) for i, s in enumerate(SCENARIOS)]


def print_available_scenarios() -> None:
    for i, (lab, _s) in enumerate(available_scenarios(), start=1):
        print(f"{i:02d}. {lab}")


def select_scenarios(selectors: list[str] | None, exact_labels: bool) -> list[dict]:
    labeled = available_scenarios()
    if not selectors:
        return [s for _, s in labeled[:9]]

    if exact_labels:
        available = {lab: s for lab, s in labeled}
        missing = [lab for lab in selectors if lab not in available]
        if missing:
            all_labels = [lab for lab, _ in labeled]
            raise ValueError(
                "Unknown scenario labels: " + str(missing) + "\nAvailable labels are:\n- " + "\n- ".join(all_labels)
            )
        return [available[lab] for lab in selectors]

    selected = []
    selected_labels = set()
    misses = []
    all_labels = [lab for lab, _ in labeled]
    for selector in selectors:
        needle = selector.lower()
        matches = [(lab, s) for lab, s in labeled if needle in lab.lower()]
        if not matches:
            misses.append(selector)
            continue
        for lab, s in matches:
            if lab not in selected_labels:
                selected.append(s)
                selected_labels.add(lab)
    if misses:
        raise ValueError(
            "No scenario labels matched these substring selectors: " + str(misses) + "\nAvailable labels are:\n- " + "\n- ".join(all_labels)
        )
    if not selected:
        raise ValueError("No scenarios were selected.")
    return selected


def build_parameter_table(selected_scenarios: list[dict], drop_constant_columns: bool) -> pd.DataFrame:
    rows = []
    for idx, scenario in enumerate(selected_scenarios):
        label = scenario_label(scenario, idx)
        params = effective_parameters_for_scenario(scenario)
        rows.append({"scenario": label, **params})

    df = pd.DataFrame(rows).set_index("scenario")
    df = df.dropna(axis=1, how="all")
    if df.empty or df.shape[1] == 0:
        raise ValueError("No numeric static parameters were found after merging DEFAULT_PARAMS with scenario overrides.")

    if drop_constant_columns:
        keep_cols = []
        for col in df.columns:
            vals = df[col].to_numpy(dtype=float)
            if not np.allclose(vals, vals[0], equal_nan=True):
                keep_cols.append(col)
        df = df[keep_cols]
        if df.shape[1] == 0:
            raise ValueError("All selected parameters were constant across the chosen scenarios.")

    return df.sort_index(axis=1)


def plot_heatmap(df: pd.DataFrame, outpath: Path, title: str, dpi: int) -> None:
    data = df.to_numpy(dtype=float)
    n_rows, n_cols = data.shape
    fig_w = max(10, 0.55 * n_cols + 4)
    fig_h = max(5.5, 0.45 * n_rows + 3)
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
        work[col] = (vals - mu) / sd if sd > 0 else 0.0
    data = work.to_numpy(dtype=float)
    n_rows, n_cols = data.shape
    fig_w = max(10, 0.55 * n_cols + 4)
    fig_h = max(5.5, 0.45 * n_rows + 3)
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
    fig_w = max(11, 0.55 * len(df.columns) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 6.8), constrained_layout=True)
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

    if args.list_scenarios:
        print_available_scenarios()
        return

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_scenarios = select_scenarios(args.scenario_labels, args.exact_labels)
    df = build_parameter_table(selected_scenarios, drop_constant_columns=args.drop_constant_columns)

    if args.scenario_labels:
        scenario_tag = "exact_labels" if args.exact_labels else "substring_labels"
    else:
        scenario_tag = "first9"
    const_tag = "varying_only" if args.drop_constant_columns else "all_static"

    csv_path = output_dir / f"static_parameters_{scenario_tag}_{const_tag}.csv"
    heatmap_path = output_dir / f"static_parameters_{scenario_tag}_{const_tag}_heatmap.png"
    lines_path = output_dir / f"static_parameters_{scenario_tag}_{const_tag}_lines.png"

    df.to_csv(csv_path)
    plot_heatmap(df, heatmap_path, title=f"Static simulation parameters ({scenario_tag}, {const_tag})", dpi=args.dpi)
    plot_lines(df, lines_path, title=f"Static parameter profiles ({scenario_tag}, {const_tag})", dpi=args.dpi)

    print(f"Scenarios selected: {len(df.index)}")
    print(f"Parameters plotted: {len(df.columns)}")
    print(f"CSV: {csv_path}")
    print(f"Heatmap: {heatmap_path}")
    print(f"Line plot: {lines_path}")

    if args.normalize:
        norm_path = output_dir / f"static_parameters_{scenario_tag}_{const_tag}_heatmap_zscore.png"
        plot_normalized_heatmap(df, norm_path, title=f"Static simulation parameters ({scenario_tag}, {const_tag}, z-score)", dpi=args.dpi)
        print(f"Normalized heatmap: {norm_path}")


if __name__ == "__main__":
    main()
