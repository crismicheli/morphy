
#!/usr/bin/env python3
"""
Read a point-record CSV exported by export_trajectory_point_records.py and
produce two distance-based summary plots.

Plots
-----
1) Phenotype labels versus 4D distance from the attractor, shown as a stacked
   histogram in either raw counts or 100% stacked percentages. Only points from
   fully viable trajectories are retained.
2) Cumulative percentage of retained viable points versus 4D distance from the
   attractor.

Filtering rule
--------------
A trajectory survives if all of its sampled points have viability_label ==
"viable". The phenotype plot and cumulative viable-distance plot are both built
from the retained set only.

Secondary x-axis
----------------
The phenotype-distance plot includes a sparse top x-axis showing a complementary
summary measure: the median signed minimum distance to the viability-box
boundary for the corresponding attractor-distance bin.

Signed box distance interpretation:
- positive: inside the box
- zero: on the boundary
- negative: outside the box
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
CUMULATIVE_COLOR = "#2166ac"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot phenotype-distance and cumulative viable-distance summaries from a point-record CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to the point-record CSV produced by export_trajectory_point_records.py.")
    parser.add_argument("--output-dir", default=None, help="Output directory for PNG figures. Defaults to a sibling 'figures' folder next to the CSV.")
    parser.add_argument("--phenotype-distance-bins", type=int, default=40, help="Number of equally spaced distance bins for the phenotype-distance stacked histogram.")
    parser.add_argument("--phenotype-distance-mode", choices=["count", "percent"], default="percent", help="Show phenotype-by-distance as raw counts or 100%% stacked percentages.")
    parser.add_argument("--cumulative-distance-bins", type=int, default=80, help="Number of equally spaced bins for the cumulative viable-distance curve.")
    parser.add_argument("--top-axis-ticks", type=int, default=6, help="Maximum number of labeled ticks on the top complementary x-axis.")
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


def split_survival_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    surviving_ids = []
    for traj_id, group in df.groupby("trajectory_id"):
        labels = group["viability_label"].astype(str)
        if bool((labels == "viable").all()):
            surviving_ids.append(traj_id)
    surviving = df[df["trajectory_id"].isin(surviving_ids)].copy()
    dropped = df[~df["trajectory_id"].isin(surviving_ids)].copy()
    if surviving.empty:
        raise ValueError("No fully viable trajectories remain after filtering.")
    return surviving, dropped


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
        distances_to_faces = [C - C_min, T - T_min, T_max - T, E - E_min, E_max - E, O - O_min]
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


def attach_signed_box_distance(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    out = df.copy()
    out["signed_box_distance"] = out.apply(lambda row: signed_distance_to_viability_box(row, bounds), axis=1)
    return out


def summarize_sets(raw_df: pd.DataFrame, surviving_df: pd.DataFrame) -> dict:
    total_traj = int(raw_df["trajectory_id"].nunique())
    surviving_traj = int(surviving_df["trajectory_id"].nunique())
    total_points = int(len(raw_df))
    surviving_points = int(len(surviving_df))
    max_distance_surviving = float(surviving_df["distance_from_attractor_4d"].max())
    return {
        "total_trajectories": total_traj,
        "surviving_trajectories": surviving_traj,
        "total_points": total_points,
        "surviving_points": surviving_points,
        "max_distance_surviving": max_distance_surviving,
    }


def summary_text(summary: dict) -> str:
    return (
        f"Traj: total={summary['total_trajectories']}, surviving={summary['surviving_trajectories']} | "
        f"Points: total={summary['total_points']}, surviving={summary['surviving_points']} | "
        f"Max surviving dist={summary['max_distance_surviving']:.3g}"
    )


def distance_edges(df: pd.DataFrame, n_bins: int) -> np.ndarray:
    d_min = float(df["distance_from_attractor_4d"].min())
    d_max = float(df["distance_from_attractor_4d"].max())
    if not np.isfinite(d_min) or not np.isfinite(d_max):
        raise ValueError("Distance range is not finite.")
    if d_max <= d_min:
        d_max = d_min + 1e-9
    return np.linspace(d_min, d_max, int(n_bins) + 1)


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


def choose_sparse_ticks(values: np.ndarray, max_ticks: int) -> np.ndarray:
    if len(values) <= max_ticks:
        return np.arange(len(values), dtype=int)
    return np.linspace(0, len(values) - 1, max_ticks, dtype=int)


def add_secondary_box_distance_axis(ax, primary_centers: np.ndarray, signed_box_medians: np.ndarray, max_ticks: int) -> None:
    secax = ax.secondary_xaxis('top')
    tick_idx = choose_sparse_ticks(primary_centers, max_ticks=max_ticks)
    secax.set_xticks(primary_centers[tick_idx])
    secax.set_xticklabels([f"{signed_box_medians[i]:.2g}" for i in tick_idx])
    secax.set_xlabel("Median signed min distance to viability-box boundary")


def plot_phenotype_distance_histogram(
    df: pd.DataFrame,
    output_path: Path,
    *,
    phenotype_distance_bins: int,
    phenotype_distance_mode: str,
    top_axis_ticks: int,
    summary: dict,
    title_stub: str,
    dpi: int,
) -> None:
    centers, widths, labels, counts, sec_x, sec_vals = build_phenotype_distance_counts(df, n_bins=phenotype_distance_bins)

    fig, ax = plt.subplots(figsize=(11.8, 7.1), constrained_layout=True)
    bottoms = np.zeros(len(centers), dtype=float)

    if phenotype_distance_mode == "percent":
        column_sums = counts.sum(axis=0)
        safe_den = np.where(column_sums > 0, column_sums, 1.0)
        plot_counts = 100.0 * counts / safe_den
        ylabel = "Phenotype composition [% within distance bin]"
    else:
        plot_counts = counts
        ylabel = "Number of surviving points"

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
    add_secondary_box_distance_axis(ax, sec_x, sec_vals, max_ticks=top_axis_ticks)
    ax.legend(frameon=True, ncol=2, title="Phenotype labels")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def build_cumulative_viable_curve(df: pd.DataFrame, n_bins: int) -> tuple[np.ndarray, np.ndarray, int]:
    edges = distance_edges(df, n_bins=n_bins)
    distances = np.sort(df["distance_from_attractor_4d"].to_numpy(dtype=float))
    hist, _ = np.histogram(distances, bins=edges)
    cumulative = np.cumsum(hist).astype(float)
    n = len(distances)
    y = 100.0 * cumulative / float(n)
    x = edges[1:]
    return x, y, n


def plot_cumulative_viable_distance(
    df: pd.DataFrame,
    output_path: Path,
    *,
    cumulative_distance_bins: int,
    summary: dict,
    title_stub: str,
    dpi: int,
) -> None:
    x, y, n = build_cumulative_viable_curve(df, n_bins=cumulative_distance_bins)

    fig, ax = plt.subplots(figsize=(10.4, 6.2), constrained_layout=True)
    ax.step(x, y, where="post", color=CUMULATIVE_COLOR, linewidth=2.4, label=f"Cumulative % of surviving viable points (N={n})")
    ax.set_title(
        f"Cumulative surviving viable points versus 4D distance from attractor\n{title_stub}\n{summary_text(summary)}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("4D distance from attractor")
    ax.set_ylabel("Cumulative [% of surviving viable points]")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
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
    surviving_df, _dropped_df = split_survival_sets(raw_df)
    surviving_df = attach_signed_box_distance(surviving_df, DEFAULT_BOUNDS)

    if args.output_dir is None:
        output_dir = csv_path.parent / "figures"
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    title_stub = infer_title_stub(surviving_df, csv_path)
    summary = summarize_sets(raw_df, surviving_df)

    stem = csv_path.stem
    phenotype_png = output_dir / f"{stem}_phenotype_distance_stacked_{args.phenotype_distance_mode}.png"
    cumulative_png = output_dir / f"{stem}_viable_distance_cumulative_percent.png"

    plot_phenotype_distance_histogram(
        surviving_df,
        phenotype_png,
        phenotype_distance_bins=args.phenotype_distance_bins,
        phenotype_distance_mode=args.phenotype_distance_mode,
        top_axis_ticks=args.top_axis_ticks,
        summary=summary,
        title_stub=title_stub,
        dpi=args.dpi,
    )
    plot_cumulative_viable_distance(
        surviving_df,
        cumulative_png,
        cumulative_distance_bins=args.cumulative_distance_bins,
        summary=summary,
        title_stub=title_stub,
        dpi=args.dpi,
    )

    print(f"Total trajectories: {summary['total_trajectories']}")
    print(f"Surviving trajectories: {summary['surviving_trajectories']}")
    print(f"Total points: {summary['total_points']}")
    print(f"Surviving points: {summary['surviving_points']}")
    print(f"Max surviving distance from attractor: {summary['max_distance_surviving']}")
    print(f"Phenotype-distance plot: {phenotype_png}")
    print(f"Cumulative viable-distance plot: {cumulative_png}")


if __name__ == "__main__":
    main()
