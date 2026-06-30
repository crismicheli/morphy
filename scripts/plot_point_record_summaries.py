#!/usr/bin/env python3
"""
Read a point-record CSV exported by export_trajectory_point_records.py and
produce two summary plots:

1) Time-resolved smoothed histograms of all phenotype labels.
2) Viable-point distance histograms in either raw counts or percentages.

The input CSV is assumed to contain pointwise records only, with at least the
following columns:
    - timestamp
    - time_delta
    - viability_label
    - phenotypical_label
    - distance_from_attractor_4d

Optional columns such as scenario_label, scenario_name, classifier_type,
trajectory_id, point_index, and state coordinates are ignored by the core
aggregation logic but can be used in figure titles when present.

The time unit is read from the CSV column `time_unit` when available and can be
overridden from the command line.

Typical usage
-------------
1) Basic run (raw viable-point counts by distance):
   python scripts/plot_point_record_summaries.py \
       --csv output/high_porosity_point_records/high_porosity_point_records_temporal.csv

2) Custom time unit override and smoothing:
   python scripts/plot_point_record_summaries.py \
       --csv output/high_porosity_point_records/high_porosity_point_records_temporal.csv \
       --time-unit days \
       --time-bins 80 \
       --smooth-window 7

3) Viable-point distance as percent of viable points:
   python scripts/plot_point_record_summaries.py \
       --csv output/high_porosity_point_records/high_porosity_point_records_temporal.csv \
       --distance-mode percent \
       --distance-percent-denominator viable

4) Viable-point distance as percent of all exported points:
   python scripts/plot_point_record_summaries.py \
       --csv output/high_porosity_point_records/high_porosity_point_records_temporal.csv \
       --distance-mode percent \
       --distance-percent-denominator all
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifiers.static_classifier import STATE_COLORS

FALLBACK_COLOR = "#7f7f7f"
DISTANCE_COLOR = "#2166ac"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot phenotype-time and viable-distance summaries from a point-record CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to the point-record CSV produced by export_trajectory_point_records.py.")
    parser.add_argument("--output-dir", default=None, help="Output directory for PNG figures. Defaults to a sibling 'figures' folder next to the CSV.")
    parser.add_argument("--time-bins", type=int, default=60, help="Number of equally spaced time bins for the phenotype histogram plot.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Centered moving-average window size used to smooth phenotype counts over time.")
    parser.add_argument("--distance-bins", type=int, default=40, help="Number of equally spaced bins for the viable-point distance histogram.")
    parser.add_argument("--distance-mode", choices=["count", "percent"], default="count", help="Plot viable-point distance histogram as raw counts or percentages.")
    parser.add_argument("--distance-percent-denominator", choices=["viable", "all"], default="viable", help="When --distance-mode percent is used, normalize by number of viable points or all points in the CSV.")
    parser.add_argument("--time-unit", default=None, help="Override the model time unit label. If omitted, the script uses the CSV time_unit column when present, otherwise 'a.u.'.")
    parser.add_argument("--dpi", type=int, default=220, help="Output figure DPI.")
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.astype(float, copy=True)
    window = int(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values.astype(float), (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def load_point_records(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"timestamp", "time_delta", "viability_label", "phenotypical_label", "distance_from_attractor_4d"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["time_delta"] = pd.to_numeric(df["time_delta"], errors="coerce")
    df["distance_from_attractor_4d"] = pd.to_numeric(df["distance_from_attractor_4d"], errors="coerce")
    df = df.dropna(subset=["timestamp", "distance_from_attractor_4d", "phenotypical_label", "viability_label"])
    return df


def resolve_time_unit(df: pd.DataFrame, cli_time_unit: str | None) -> str:
    if cli_time_unit:
        return cli_time_unit
    if "time_unit" in df.columns and not df["time_unit"].dropna().empty:
        return str(df["time_unit"].dropna().iloc[0])
    return "a.u."


def infer_title_stub(df: pd.DataFrame, csv_path: Path) -> str:
    scenario = df["scenario_label"].iloc[0] if "scenario_label" in df.columns and not df.empty else csv_path.stem
    classifier = df["classifier_type"].iloc[0] if "classifier_type" in df.columns and not df.empty else "classifier"
    return f"{scenario} | {classifier}"


def build_time_label_counts(df: pd.DataFrame, time_bins: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    t_min = float(df["timestamp"].min())
    t_max = float(df["timestamp"].max())
    if not np.isfinite(t_min) or not np.isfinite(t_max):
        raise ValueError("Timestamp range is not finite.")
    if t_max <= t_min:
        t_max = t_min + 1e-9
    edges = np.linspace(t_min, t_max, int(time_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    labels = sorted(df["phenotypical_label"].astype(str).unique().tolist())
    counts = np.zeros((len(labels), len(centers)), dtype=float)
    for idx, label in enumerate(labels):
        tvals = df.loc[df["phenotypical_label"].astype(str) == label, "timestamp"].to_numpy(dtype=float)
        hist, _ = np.histogram(tvals, bins=edges)
        counts[idx, :] = hist.astype(float)
    return centers, counts, labels


def plot_smoothed_phenotype_histograms(df: pd.DataFrame, output_path: Path, *, time_bins: int, smooth_window: int, time_unit: str, dpi: int) -> None:
    centers, counts, labels = build_time_label_counts(df, time_bins=time_bins)
    fig, ax = plt.subplots(figsize=(10.2, 6.4), constrained_layout=True)
    for idx, label in enumerate(labels):
        smoothed = moving_average(counts[idx, :], smooth_window)
        color = STATE_COLORS.get(label, FALLBACK_COLOR)
        ax.plot(centers, smoothed, linewidth=2.2, color=color, label=label)
    title_stub = infer_title_stub(df, output_path)
    ax.set_title(f"Time-resolved smoothed phenotype histograms\n{title_stub}", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"Trajectory time [{time_unit}]")
    ax.set_ylabel("Smoothed point count")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def compute_viable_distance_histogram(df: pd.DataFrame, *, distance_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    viable = df.loc[df["viability_label"].astype(str) == "viable"].copy()
    if viable.empty:
        raise ValueError("No viable points found in the CSV; cannot build viable-distance histogram.")
    distances = viable["distance_from_attractor_4d"].to_numpy(dtype=float)
    d_min = float(np.min(distances))
    d_max = float(np.max(distances))
    if not np.isfinite(d_min) or not np.isfinite(d_max):
        raise ValueError("Distance range is not finite.")
    if d_max <= d_min:
        d_max = d_min + 1e-9
    edges = np.linspace(d_min, d_max, int(distance_bins) + 1)
    hist, _ = np.histogram(distances, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    return centers, widths, hist.astype(float), len(viable), len(df)


def plot_viable_distance_histogram(
    df: pd.DataFrame,
    output_path: Path,
    *,
    distance_bins: int,
    distance_mode: str,
    distance_percent_denominator: str,
    dpi: int,
) -> None:
    centers, widths, hist, n_viable, n_all = compute_viable_distance_histogram(df, distance_bins=distance_bins)
    title_stub = infer_title_stub(df, output_path)

    if distance_mode == "count":
        yvals = hist
        ylabel = "Number of viable points"
        legend_label = f"Viable points (N={n_viable})"
    else:
        denom = n_viable if distance_percent_denominator == "viable" else n_all
        if denom <= 0:
            raise ValueError("Cannot normalize histogram because denominator is zero.")
        yvals = 100.0 * hist / float(denom)
        if distance_percent_denominator == "viable":
            ylabel = "Viable points [% of viable points]"
            legend_label = f"Viable points, % of viable (N={n_viable})"
        else:
            ylabel = "Viable points [% of all points]"
            legend_label = f"Viable points, % of all (N_all={n_all}, N_viable={n_viable})"

    fig, ax = plt.subplots(figsize=(9.6, 6.0), constrained_layout=True)
    ax.bar(
        centers,
        yvals,
        width=widths,
        color=DISTANCE_COLOR,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
        align="center",
        label=legend_label,
    )

    ax.set_title(
        f"Viable points versus 4D distance from attractor\n{title_stub}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("4D distance from attractor")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    if distance_mode == "percent":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
    ax.legend(frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    df = load_point_records(csv_path)
    time_unit = resolve_time_unit(df, args.time_unit)

    if args.output_dir is None:
        output_dir = csv_path.parent / "figures"
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = csv_path.stem
    phenotype_png = output_dir / f"{stem}_phenotype_time_smoothed.png"
    suffix = f"{args.distance_mode}"
    if args.distance_mode == "percent":
        suffix += f"_{args.distance_percent_denominator}"
    viable_distance_png = output_dir / f"{stem}_viable_distance_hist_{suffix}.png"

    plot_smoothed_phenotype_histograms(
        df,
        phenotype_png,
        time_bins=args.time_bins,
        smooth_window=args.smooth_window,
        time_unit=time_unit,
        dpi=args.dpi,
    )
    plot_viable_distance_histogram(
        df,
        viable_distance_png,
        distance_bins=args.distance_bins,
        distance_mode=args.distance_mode,
        distance_percent_denominator=args.distance_percent_denominator,
        dpi=args.dpi,
    )

    print(f"Loaded rows: {len(df)}")
    print(f"Resolved time unit: {time_unit}")
    print(f"Phenotype plot: {phenotype_png}")
    print(f"Viable-distance plot: {viable_distance_png}")


if __name__ == "__main__":
    main()
