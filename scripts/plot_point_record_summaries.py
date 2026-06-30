
#!/usr/bin/env python3
"""
Read a point-record CSV exported by export_trajectory_point_records.py and
produce two distance-based summary plots:

1) Phenotype labels versus 4D distance from the attractor, shown as a stacked
   histogram in either raw counts or 100% stacked percentages.
2) Viable-point distance histogram in either raw counts or percentages.

The input CSV is assumed to contain pointwise records only, with at least the
following columns:
    - trajectory_id
    - timestamp
    - time_delta
    - viability_label
    - phenotypical_label
    - distance_from_attractor_4d

Optional columns such as scenario_label, scenario_name, classifier_type,
point_index, and state coordinates are ignored by the core aggregation logic
but can be used in figure titles when present.

Summary metadata
----------------
Both plots annotate the dataset with:
- total number of trajectories,
- total number of points,
- maximum 4D distance from attractor.
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
        description="Plot phenotype-distance and viable-distance summaries from a point-record CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to the point-record CSV produced by export_trajectory_point_records.py.")
    parser.add_argument("--output-dir", default=None, help="Output directory for PNG figures. Defaults to a sibling 'figures' folder next to the CSV.")
    parser.add_argument("--phenotype-distance-bins", type=int, default=40, help="Number of equally spaced distance bins for the phenotype-distance stacked histogram.")
    parser.add_argument("--phenotype-distance-mode", choices=["count", "percent"], default="percent", help="Show phenotype-by-distance as raw counts or 100%% stacked percentages.")
    parser.add_argument("--distance-bins", type=int, default=40, help="Number of equally spaced bins for the viable-point distance histogram.")
    parser.add_argument("--distance-mode", choices=["count", "percent"], default="count", help="Plot viable-point distance histogram as raw counts or percentages.")
    parser.add_argument("--distance-percent-denominator", choices=["viable", "all"], default="viable", help="When --distance-mode percent is used, normalize by number of viable points or all points in the CSV.")
    parser.add_argument("--time-unit", default=None, help="Override the model time unit label. If omitted, the script uses the CSV time_unit column when present, otherwise 'a.u.'.")
    parser.add_argument("--dpi", type=int, default=220, help="Output figure DPI.")
    return parser.parse_args()


def load_point_records(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"trajectory_id", "timestamp", "time_delta", "viability_label", "phenotypical_label", "distance_from_attractor_4d"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    df = df.copy()
    df["trajectory_id"] = pd.to_numeric(df["trajectory_id"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["time_delta"] = pd.to_numeric(df["time_delta"], errors="coerce")
    df["distance_from_attractor_4d"] = pd.to_numeric(df["distance_from_attractor_4d"], errors="coerce")
    df = df.dropna(subset=["trajectory_id", "timestamp", "distance_from_attractor_4d", "phenotypical_label", "viability_label"])
    df["trajectory_id"] = df["trajectory_id"].astype(int)
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


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "n_trajectories": int(df["trajectory_id"].nunique()),
        "n_points": int(len(df)),
        "max_distance": float(df["distance_from_attractor_4d"].max()),
    }


def summary_text(summary: dict) -> str:
    return (
        f"Trajectories={summary['n_trajectories']}, "
        f"Points={summary['n_points']}, "
        f"Max dist={summary['max_distance']:.3g}"
    )


def distance_edges(df: pd.DataFrame, n_bins: int) -> np.ndarray:
    d_min = float(df["distance_from_attractor_4d"].min())
    d_max = float(df["distance_from_attractor_4d"].max())
    if not np.isfinite(d_min) or not np.isfinite(d_max):
        raise ValueError("Distance range is not finite.")
    if d_max <= d_min:
        d_max = d_min + 1e-9
    return np.linspace(d_min, d_max, int(n_bins) + 1)


def build_phenotype_distance_counts(df: pd.DataFrame, n_bins: int) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    edges = distance_edges(df, n_bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    labels = sorted(df["phenotypical_label"].astype(str).unique().tolist())
    counts = np.zeros((len(labels), len(centers)), dtype=float)

    for idx, label in enumerate(labels):
        dvals = df.loc[df["phenotypical_label"].astype(str) == label, "distance_from_attractor_4d"].to_numpy(dtype=float)
        hist, _ = np.histogram(dvals, bins=edges)
        counts[idx, :] = hist.astype(float)

    return centers, widths, labels, counts


def plot_phenotype_distance_histogram(
    df: pd.DataFrame,
    output_path: Path,
    *,
    phenotype_distance_bins: int,
    phenotype_distance_mode: str,
    dpi: int,
) -> None:
    centers, widths, labels, counts = build_phenotype_distance_counts(df, n_bins=phenotype_distance_bins)
    summary = dataset_summary(df)
    title_stub = infer_title_stub(df, output_path)

    fig, ax = plt.subplots(figsize=(10.8, 6.6), constrained_layout=True)
    bottoms = np.zeros(len(centers), dtype=float)

    if phenotype_distance_mode == "percent":
        column_sums = counts.sum(axis=0)
        safe_den = np.where(column_sums > 0, column_sums, 1.0)
        plot_counts = 100.0 * counts / safe_den
        ylabel = "Phenotype composition [% within distance bin]"
    else:
        plot_counts = counts
        ylabel = "Number of points"

    for idx, label in enumerate(labels):
        color = STATE_COLORS.get(label, FALLBACK_COLOR)
        ax.bar(
            centers,
            plot_counts[idx, :],
            width=widths,
            bottom=bottoms,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            align="center",
            label=label,
        )
        bottoms += plot_counts[idx, :]

    ax.set_title(
        f"Phenotype labels versus 4D distance from attractor\n{title_stub}\n{summary_text(summary)}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("4D distance from attractor")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    if phenotype_distance_mode == "percent":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
    ax.legend(frameon=True, ncol=2, title="Phenotype labels")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def compute_viable_distance_histogram(df: pd.DataFrame, *, distance_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    viable = df.loc[df["viability_label"].astype(str) == "viable"].copy()
    if viable.empty:
        raise ValueError("No viable points found in the CSV; cannot build viable-distance histogram.")
    edges = distance_edges(viable, n_bins=distance_bins)
    distances = viable["distance_from_attractor_4d"].to_numpy(dtype=float)
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
    summary = dataset_summary(df)
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

    fig, ax = plt.subplots(figsize=(9.8, 6.1), constrained_layout=True)
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
        f"Viable points versus 4D distance from attractor\n{title_stub}\n{summary_text(summary)}",
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
    _time_unit = resolve_time_unit(df, args.time_unit)

    if args.output_dir is None:
        output_dir = csv_path.parent / "figures"
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = csv_path.stem
    phenotype_png = output_dir / f"{stem}_phenotype_distance_stacked_{args.phenotype_distance_mode}.png"
    suffix = f"{args.distance_mode}"
    if args.distance_mode == "percent":
        suffix += f"_{args.distance_percent_denominator}"
    viable_distance_png = output_dir / f"{stem}_viable_distance_hist_{suffix}.png"

    plot_phenotype_distance_histogram(
        df,
        phenotype_png,
        phenotype_distance_bins=args.phenotype_distance_bins,
        phenotype_distance_mode=args.phenotype_distance_mode,
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

    summary = dataset_summary(df)
    print(f"Loaded rows: {len(df)}")
    print(f"Trajectories: {summary['n_trajectories']}")
    print(f"Points: {summary['n_points']}")
    print(f"Max distance from attractor: {summary['max_distance']}")
    print(f"Phenotype-distance plot: {phenotype_png}")
    print(f"Viable-distance plot: {viable_distance_png}")


if __name__ == "__main__":
    main()
