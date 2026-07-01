
#!/usr/bin/env python3
"""
Plot all static simulation parameters used by selected scenarios.

Interpretation
--------------
"Static parameters" means the effective parameter set used to simulate each
scenario: start from DEFAULT_PARAMS, then overwrite with each scenario's
param_overrides. The resulting table therefore represents the actual parameter
values used in each simulation.

Plot style
----------
This script makes one single figure:
- x-axis: scenarios
- within each scenario: one colored bar per parameter
- bar color identifies the parameter

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
    parser = argparse.ArgumentParser(description="Plot static simulation parameters as grouped bars across scenarios.")
    parser.add_argument("--output-dir", default="output/static_parameter_barplots", help="Directory for output files.")
    parser.add_argument("--scenario-labels", nargs="*", default=None, help="Scenario selectors. By default these are treated as case-insensitive substrings, not exact labels.")
    parser.add_argument("--exact-labels", action="store_true", help="Interpret --scenario-labels as exact labels instead of substring matches.")
    parser.add_argument("--list-scenarios", action="store_true", help="Print all available scenario labels and exit.")
    parser.add_argument("--drop-constant-columns", action="store_true", help="Drop parameters that have the same value in every selected scenario.")
    parser.add_argument("--normalize-columns", action="store_true", help="Scale each parameter column independently to [0, 1] before plotting, useful when parameters have very different magnitudes.")
    parser.add_argument("--max-params", type=int, default=None, help="Optional cap on number of parameters plotted, after filtering.")
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


def select_scenarios(selectors: list[str] | None, exact_labels: bool) -> list[tuple[str, dict]]:
    labeled = available_scenarios()
    if not selectors:
        return labeled[:9]

    if exact_labels:
        available = {lab: s for lab, s in labeled}
        missing = [lab for lab in selectors if lab not in available]
        if missing:
            all_labels = [lab for lab, _ in labeled]
            raise ValueError(
                "Unknown scenario labels: " + str(missing) + "\nAvailable labels are:\n- " + "\n- ".join(all_labels)
            )
        return [(lab, available[lab]) for lab in selectors]

    selected = []
    seen = set()
    misses = []
    all_labels = [lab for lab, _ in labeled]
    for selector in selectors:
        needle = selector.lower()
        matches = [(lab, s) for lab, s in labeled if needle in lab.lower()]
        if not matches:
            misses.append(selector)
            continue
        for lab, s in matches:
            if lab not in seen:
                selected.append((lab, s))
                seen.add(lab)
    if misses:
        raise ValueError(
            "No scenario labels matched these substring selectors: " + str(misses) + "\nAvailable labels are:\n- " + "\n- ".join(all_labels)
        )
    return selected


def build_parameter_table(selected_scenarios: list[tuple[str, dict]], drop_constant_columns: bool, max_params: int | None) -> pd.DataFrame:
    rows = []
    for label, scenario in selected_scenarios:
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

    df = df.reindex(sorted(df.columns), axis=1)
    if max_params is not None:
        df = df.iloc[:, :max_params]
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        vals = out[col].to_numpy(dtype=float)
        lo = np.nanmin(vals)
        hi = np.nanmax(vals)
        if hi > lo:
            out[col] = (vals - lo) / (hi - lo)
        else:
            out[col] = 1.0
    return out


def plot_grouped_bars(df_plot: pd.DataFrame, outpath: Path, title: str, ylabel: str, dpi: int) -> None:
    scenarios = list(df_plot.index)
    params = list(df_plot.columns)
    n_scen = len(scenarios)
    n_params = len(params)

    x = np.arange(n_scen)
    total_group_width = 0.86
    bar_width = total_group_width / max(n_params, 1)
    offsets = (np.arange(n_params) - (n_params - 1) / 2.0) * bar_width

    fig_w = max(11, 1.3 * n_scen + 0.28 * n_params + 4)
    fig_h = 7.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n_params)]

    for j, param in enumerate(params):
        heights = df_plot[param].to_numpy(dtype=float)
        ax.bar(x + offsets[j], heights, width=bar_width * 0.95, label=param, color=colors[j], edgecolor="black", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Parameter", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, frameon=True, fontsize=8)

    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
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
    df = build_parameter_table(selected_scenarios, args.drop_constant_columns, args.max_params)
    df_plot = normalize_columns(df) if args.normalize_columns else df

    if args.scenario_labels:
        scenario_tag = "exact_labels" if args.exact_labels else "substring_labels"
    else:
        scenario_tag = "first9"
    value_tag = "normalized" if args.normalize_columns else "raw"
    const_tag = "varying_only" if args.drop_constant_columns else "all_static"

    csv_path = output_dir / f"static_parameters_{scenario_tag}_{const_tag}_{value_tag}.csv"
    barplot_path = output_dir / f"static_parameters_{scenario_tag}_{const_tag}_{value_tag}_grouped_bars.png"

    df.to_csv(csv_path)
    ylabel = "Normalized parameter value (0-1 within parameter)" if args.normalize_columns else "Parameter value"
    title = f"Static simulation parameters by scenario ({scenario_tag}, {const_tag}, {value_tag})"
    plot_grouped_bars(df_plot, barplot_path, title=title, ylabel=ylabel, dpi=args.dpi)

    print(f"Scenarios selected: {len(df.index)}")
    print(f"Parameters plotted: {len(df.columns)}")
    print(f"CSV: {csv_path}")
    print(f"Grouped bar plot: {barplot_path}")


if __name__ == "__main__":
    main()
