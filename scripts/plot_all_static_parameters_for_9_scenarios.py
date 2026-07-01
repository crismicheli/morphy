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
- an asterisk is drawn above bars whose effective value differs from the default
  value for that parameter

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
import re
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


def clean_label(label: str) -> str:
    cleaned = re.sub(r"\b(stable|borderline|unstable)\b", "", label, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_,;:")
    return cleaned or label


def scenario_label(s: dict, idx: int) -> str:
    raw = str(s.get("label") or s.get("name") or s.get("scenario") or f"scenario_{idx+1}")
    return clean_label(raw)


def extract_p_from_label(label: str) -> float | None:
    m = re.search(r"\(\s*p\s*=\s*([0-9]*\.?[0-9]+)", label)
    if m:
        return float(m.group(1))
    return None


def default_parameter_values() -> dict:
    defaults = {}
    for k, v in dict(DEFAULT_PARAMS).items():
        if is_number(v):
            defaults[str(k)] = float(v)
    defaults["p"] = float(DEFAULT_PARAMS.get("p")) if is_number(DEFAULT_PARAMS.get("p")) else np.nan
    return defaults


def effective_parameters_for_scenario(scenario: dict, label: str) -> dict:
    params = {}
    for k, v in dict(DEFAULT_PARAMS).items():
        if is_number(v):
            params[str(k)] = float(v)

    if "p" not in params:
        params["p"] = np.nan

    overrides = scenario.get("param_overrides", {})
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            if is_number(v):
                params[str(k)] = float(v)

    if is_number(scenario.get("p")):
        params["p"] = float(scenario["p"])
    else:
        p_from_label = extract_p_from_label(label)
        if p_from_label is not None:
            params["p"] = p_from_label

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
        params = effective_parameters_for_scenario(scenario, label)
        rows.append({"scenario": label, **params})

    df = pd.DataFrame(rows).set_index("scenario")
    df = df.dropna(axis=1, how="all")
    if df.empty or df.shape[1] == 0:
        raise ValueError("No numeric static parameters were found after merging DEFAULT_PARAMS with scenario overrides and p.")

    if "p" not in df.columns:
        raise ValueError("Parameter 'p' could not be recovered from defaults, scenario['p'], or the scenario label.")

    if drop_constant_columns:
        keep_cols = []
        for col in df.columns:
            vals = df[col].to_numpy(dtype=float)
            if not np.allclose(vals, vals[0], equal_nan=True):
                keep_cols.append(col)
        if "p" not in keep_cols and "p" in df.columns:
            keep_cols = ["p"] + keep_cols
        df = df[keep_cols]
        if df.shape[1] == 0:
            raise ValueError("All selected parameters were constant across the chosen scenarios.")

    ordered_cols = ["p"] + sorted([c for c in df.columns if c != "p"])
    df = df[ordered_cols]

    if max_params is not None:
        remaining = [c for c in df.columns if c != "p"]
        df = df[["p"] + remaining[: max(0, max_params - 1)]]
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


def build_changed_mask(df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        default_val = defaults.get(col, np.nan)
        if np.isnan(default_val):
            if col == "p":
                mask[col] = True
            continue
        vals = df[col].to_numpy(dtype=float)
        mask[col] = ~np.isclose(vals, default_val, equal_nan=True)
    return mask


def plot_grouped_bars(df_raw: pd.DataFrame, df_plot: pd.DataFrame, changed_mask: pd.DataFrame, outpath: Path, title: str, ylabel: str, dpi: int) -> None:
    scenarios = list(df_plot.index)
    params = list(df_plot.columns)
    n_scen = len(scenarios)
    n_params = len(params)

    x = np.arange(n_scen)
    total_group_width = 0.86
    bar_width = total_group_width / max(n_params, 1)
    offsets = (np.arange(n_params) - (n_params - 1) / 2.0) * bar_width

    fig_w = max(11, 1.3 * n_scen + 0.28 * n_params + 4)
    fig_h = 7.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n_params)]

    ymax = np.nanmax(df_plot.to_numpy(dtype=float)) if df_plot.size else 1.0
    ymin = np.nanmin(df_plot.to_numpy(dtype=float)) if df_plot.size else 0.0
    yrange = ymax - ymin if ymax > ymin else max(abs(ymax), 1.0)
    pad = 0.06 * yrange

    for j, param in enumerate(params):
        heights = df_plot[param].to_numpy(dtype=float)
        bars = ax.bar(
            x + offsets[j],
            heights,
            width=bar_width * 0.95,
            label=param,
            color=colors[j],
            edgecolor="black",
            linewidth=0.3,
        )
        changed = changed_mask[param].to_numpy(dtype=bool)
        for bar, is_changed in zip(bars, changed):
            if not is_changed:
                continue
            y = bar.get_height()
            star_y = y + pad if y >= 0 else y - pad
            va = "bottom" if y >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                star_y,
                "*",
                ha="center",
                va=va,
                fontsize=14,
                fontweight="bold",
                color="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Parameter", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, frameon=True, fontsize=8)

    top_limit = ymax + 3 * pad
    bottom_limit = min(0, ymin - 2 * pad)
    ax.set_ylim(bottom_limit, top_limit)

    ax.text(0.0, 1.02, "* differs from default value", transform=ax.transAxes, ha="left", va="bottom", fontsize=9)

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
    defaults = default_parameter_values()
    changed_mask = build_changed_mask(df, defaults)
    df_plot = normalize_columns(df) if args.normalize_columns else df

    if args.scenario_labels:
        scenario_tag = "exact_labels" if args.exact_labels else "substring_labels"
    else:
        scenario_tag = "first9"
    value_tag = "normalized" if args.normalize_columns else "raw"
    const_tag = "varying_only" if args.drop_constant_columns else "all_static"

    csv_path = output_dir / f"static_parameters_{scenario_tag}_{const_tag}_{value_tag}.csv"
    changed_csv_path = output_dir / f"static_parameters_{scenario_tag}_{const_tag}_{value_tag}_changed_vs_default.csv"
    barplot_path = output_dir / f"static_parameters_{scenario_tag}_{const_tag}_{value_tag}_grouped_bars.png"

    df.to_csv(csv_path)
    changed_mask.astype(int).to_csv(changed_csv_path)
    ylabel = "Normalized parameter value (0-1 within parameter)" if args.normalize_columns else "Parameter value"
    title = f"Static simulation parameters by scenario ({scenario_tag}, {const_tag}, {value_tag})"
    plot_grouped_bars(df, df_plot, changed_mask, barplot_path, title=title, ylabel=ylabel, dpi=args.dpi)

    print(f"Scenarios selected: {len(df.index)}")
    print(f"Parameters plotted: {len(df.columns)}")
    print(f"Includes p: {'p' in df.columns}")
    print(f"Parameter list: {', '.join(df.columns)}")
    print(f"CSV: {csv_path}")
    print(f"Changed mask CSV: {changed_csv_path}")
    print(f"Grouped bar plot: {barplot_path}")


if __name__ == "__main__":
    main()
