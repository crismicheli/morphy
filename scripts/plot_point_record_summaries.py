#!/usr/bin/env python3
"""
Read a point-record CSV exported by export_trajectory_point_records.py and
produce two distance-based summary plots using only points from trajectories
that remain viable throughout their sampled lifetime.

Plots
-----
1) Phenotype labels versus 4D distance from the attractor, shown as a stacked
   histogram in either raw counts or 100% stacked percentages.
2) Viable-point distance histogram, defaulting to percentage of viable points.

Viability filtering
-------------------
Before plotting, the script excludes every point belonging to a trajectory that
contains any non-viable point. In other words, only trajectories whose sampled
points are all labeled `viable` are retained for both plots.

Complementary x-axis
--------------------
The phenotype-distance plot includes a secondary x-axis showing a complementary
measure: an estimated minimum signed distance to the viability-box boundary for
points at the corresponding attractor-distance bin center.

Interpretation of the secondary axis:
- positive values: inside the viability box (distance to nearest boundary face)
- zero: on the viability boundary
- negative values: outside the viability box (magnitude indicates overshoot)

The secondary axis is a binned summary, not an exact pointwise coordinate map.
For each attractor-distance bin center on the primary x-axis, the script finds
points in the nearest bin and reports the median signed distance-to-box value
for those points.
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
from config import DEFAULT_BOUNDS

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
    parser.add_argument("--distance-mode", choices=["count", "percent"], default="percent", help="Plot viable-point distance histogram as raw counts or percentages.")
    parser.add_argument("--distance-percent-denominator", choices=["viable", "all"], default="viable", help="When --distance-mode percent is used, normalize by number of retained viable points or all retained points.")
    parser.add_argument("--time-unit", default=None, help="Unused for distance plots but accepted for interface compatibility.")
    parser.add_argument("--dpi", type=int, default=220, help="Output figure DPI.")
    return parser.parse_args()


def load_point_records(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"trajectory_id", "viability_label", "phenotypical_label", "distance_from_attractor_4d", "C", "T", "E", "O"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    df = df.copy()
    df["trajectory_id"] = pd.to_numeric(df["trajectory_id"], errors="coerce")
    df["distance_from_attractor_4d"] = pd.to_numeric(df["distance_from_attractor_4d"], errors="coerce")
    for col in ["C", "T", "E", "O"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["trajectory_id", "distance_from_attractor_4d", "phenotypical_label", "viability_label", "C", "T", "E", "O"])
    df["trajectory_id"] = df["trajectory_id"].astype(int)
    return df


def infer_title_stub(df: pd.DataFrame, csv_path: Path) -> str:
    scenario = df["scenario_label"].iloc[0] if "scenario_label" in df.columns and not df.empty else csv_path.stem
    classifier = df["classifier_type"].iloc[0] if "classifier_type" in df.columns and not df.empty else "classifier"
    return f"{scenario} | {classifier}"


def retain_only_fully_viable_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    good_ids = []
    for traj_id, group in df.groupby("trajectory_id"):
        labels = group["viability_label"].astype(str)
        if bool((labels == "viable").all()):
            good_ids.append(traj_id)
    kept = df[df["trajectory_id"].isin(good_ids)].copy()
    if kept.empty:
        raise ValueError("No fully viable trajectories remain after filtering.")
    return kept


def signed_distance_to_viability_box(row: pd.Series, bounds: dict) -> float:
    C = float(row["C"])
    T = float(row["T"])
    E = float(row["E"])
    O = float(row["O"])
    C_min = float(bounds["C_min"])
    T_min = float(bounds["T_min"])
    T_max = float(bounds["T_max"])
    E_min = float(bounds["E_min"])
    E_max = float(bounds["E_max"])
    O_min = float(bounds["O_min"])

    inside = (
        C >= C_min and
        T_min <= T <= T_max and
        E_min <= E <= E_max and
        O >= O_min
    )

    if inside:
        distances_to_faces = [
            C - C_min,
            T - T_min,
            T_max - T,
            E - E_min,
            E_max - E,
            O - O_min,
        ]
        return float(min(distances_to_faces))

    violations = []
    if C < C_min:
        violations.append(C_min - C)
    if T < T_min:
        violations.append(T_min - T)
    if T > T_max:
        violations.append(T - T_max)
    if E < E_min:
        violations.append(E_min - E)
    if E > E_max:
        violations.append(E - E_max)
    if O < O_min:
        violations.append(O_min - O)
    return -float(max(violations))


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "n_trajectories": int(df["trajectory_id"].nunique()),
        "n_points": int(len(df)),
        "max_distance": float(df["distance_from_attractor_4d"].max()),
    }


def summary_text(summary: dict) -> str:
    return (
        f"Retained trajectories={summary['n_trajectories']}, "
        f"Retained points={summary['n_points']}, "
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


def attach_signed_box_distance(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    out = df.copy()
    out["signed_box_distance"] = out.apply(lambda row: signed_distance_to_viability_box(row, bounds), axis=1)
    return out


def build_secondary_axis_lookup(df: pd.DataFrame, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_ids = np.digitize(df["distance_from_attractor_4d"].to_numpy(dtype=float), edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, len(centers) - 1)
    signed_vals = df["signed_box_distance"].to_numpy(dtype=float)

    medians = np.full(len(centers), np.nan, dtype=float)
    for i in range(len(centers)):
        vals = signed_vals[bin_ids == i]
        if len(vals) > 0:
            medians[i] = float(np.median(vals))

    valid = np.where(np.isfinite(medians))[0]
    if len(valid) == 0:
        return centers, np.zeros(len(centers), dtype=float)

    first_valid = valid[0]
    last_valid = valid[-1]
    medians[:first_valid] = medians[first_valid]
    medians[last_valid + 1:] = medians[last_valid]
    for a, b in zip(valid[:-1], valid[1:]):
        if b - a > 1:
            medians[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [medians[a], medians[b]])
    return centers, medians


def build_phenotype_distance_counts(df: pd.DataFrame, n_bins: int) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    edges = distance_edges(df, n_bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    labels = sorted(df["phenotypical_label"].astype(str).unique().tolist())
    counts = np.zeros((len(labels), len(centers)), dtype=float)

    for idx, label in enumerate(labels):
        dvals = df.loc[df["phenotypical_label"].astype(str) == label, "distance_from_attractor_4d"].to_numpy(dtype=float)
        hist, _ = np.histogram(dvals, bins=edges)
        counts[idx, :] = hist.astype(float)

    sec_x, sec_vals = build_secondary_axis_lookup(df, edges)
    return centers, widths, labels, counts, sec_x, sec_vals


def add_secondary_box_distance_axis(ax, primary_centers: np.ndarray, signed_box_medians: np.ndarray) -> None:
    secax = ax.secondary_xaxis('top')
    secax.set_xticks(primary_centers)
    secax.set_xticklabels([f"{v:.2g}" for v in signed_box_medians], rotation=0)
    secax.set_xlabel("Median signed min distance to viability box boundary at matching attractor-distance bin")


def plot_phenotype_distance_histogram(
    df: pd.DataFrame,
    output_path: Path,
    *,
    phenotype_distance_bins: int,
    phenotype_distance_mode: str,
    dpi: int,
) -> None:
    centers, widths, labels, counts, sec_x, sec_vals = build_phenotype_distance_counts(df, n_bins=phenotype_distance_bins)
    summary = dataset_summary(df)
    title_stub = infer_title_stub(df, output_path)

    fig, ax = plt.subplots(figsize=(11.6, 7.0), constrained_layout=True)
    bottoms = np.zeros(len(centers), dtype=float)

    if phenotype_distance_mode == "percent":
        column_sums = counts.sum(axis=0)
        safe_den = np.where(column_sums > 0, column_sums, 1.0)
        plot_counts = 100.0 * counts / safe_den
        ylabel = "Phenotype composition [% within distance bin]"
    else:
        plot_counts = counts
        ylabel = "Number of retained points"

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
    add_secondary_box_distance_axis(ax, sec_x, sec_vals)
    ax.legend(frameon=True, ncol=2, title="Phenotype labels")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def compute_viable_distance_histogram(df: pd.DataFrame, *, distance_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    edges = distance_edges(df, n_bins=distance_bins)
    distances = df["distance_from_attractor_4d"].to_numpy(dtype=float)
    hist, _ = np.histogram(distances, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    return centers, widths, hist.astype(float), len(df), len(df)


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
        ylabel = "Number of retained viable points"
        legend_label = f"Retained viable points (N={n_viable})"
    else:
        denom = n_viable if distance_percent_denominator == "viable" else n_all
        if denom <= 0:
            raise ValueError("Cannot normalize histogram because denominator is zero.")
        yvals = 100.0 * hist / float(denom)
        if distance_percent_denominator == "viable":
            ylabel = "Retained viable points [% of retained viable points]"
            legend_label = f"Retained viable points, % of viable (N={n_viable})"
        else:
            ylabel = "Retained viable points [% of retained points]"
            legend_label = f"Retained viable points, % of all retained (N={n_all})"

    fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)
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
        f"Retained viable points versus 4D distance from attractor\n{title_stub}\n{summary_text(summary)}",
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

    raw_df = load_point_records(csv_path)
    kept_df = retain_only_fully_viable_trajectories(raw_df)
    kept_df = attach_signed_box_distance(kept_df, DEFAULT_BOUNDS)

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
        kept_df,
        phenotype_png,
        phenotype_distance_bins=args.phenotype_distance_bins,
        phenotype_distance_mode=args.phenotype_distance_mode,
        dpi=args.dpi,
    )
    plot_viable_distance_histogram(
        kept_df,
        viable_distance_png,
        distance_bins=args.distance_bins,
        distance_mode=args.distance_mode,
        distance_percent_denominator=args.distance_percent_denominator,
        dpi=args.dpi,
    )

    summary = dataset_summary(kept_df)
    print(f"Original trajectories: {raw_df['trajectory_id'].nunique()}")
    print(f"Retained trajectories: {summary['n_trajectories']}")
    print(f"Retained points: {summary['n_points']}")
    print(f"Max distance from attractor (retained set): {summary['max_distance']}")
    print(f"Phenotype-distance plot: {phenotype_png}")
    print(f"Viable-distance plot: {viable_distance_png}")


if __name__ == "__main__":
    main()
