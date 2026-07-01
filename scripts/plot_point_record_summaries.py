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

Optional spatial resampling
---------------------------
To reduce oversampling near the attractor, trajectories can be rediscretized in
CETO state space using approximately uniform 4D arc-length spacing. This turns
strongly time-dense trajectories into more spatially even samples before the
plots are computed.

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

# from classifiers.classifier_dispatch import get_classifier_callable
from classifiers.static_classifier import STATE_COLORS
from config import DEFAULT_BOUNDS

FALLBACK_COLOR = "#7f7f7f"
CUMULATIVE_COLOR = "#2166ac"
STATE_COLS = ["C", "T", "E", "O"]


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
    parser.add_argument("--print-attractor-summary", action="store_true", help="Print the estimated attractor position in CETO coordinates and its signed minimum distances to the viability-box faces.")
    parser.add_argument("--sampling-mode", choices=["raw", "arclength"], default="raw", help="Use original time-sampled points or arc-length-rediscretized state-space samples.")
    parser.add_argument("--space-step", type=float, default=0.02, help="Target 4D CETO arc-length spacing when --sampling-mode arclength is used.")
    parser.add_argument("--classifier-type", default=None, help="Classifier name used to relabel resampled points. If omitted, uses the CSV classifier_type column when present, otherwise 'static'.")
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
    for col in STATE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["trajectory_id", "distance_from_attractor_4d", "phenotypical_label", "viability_label", *STATE_COLS])
    df["trajectory_id"] = df["trajectory_id"].astype(int)
    return df


def infer_title_stub(df: pd.DataFrame, csv_path: Path) -> str:
    scenario = df["scenario_label"].iloc[0] if "scenario_label" in df.columns and not df.empty else csv_path.stem
    classifier = df["classifier_type"].iloc[0] if "classifier_type" in df.columns and not df.empty else "classifier"
    return f"{scenario} | {classifier}"


def infer_classifier_type(df: pd.DataFrame, cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    if "classifier_type" in df.columns and not df["classifier_type"].dropna().empty:
        return str(df["classifier_type"].dropna().iloc[0])
    return "static"


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


def signed_distance_to_viability_box_row(values: np.ndarray, bounds: dict) -> float:
    C, T, E, O = map(float, values)
    C_min = float(bounds["C_min"])
    T_min = float(bounds["T_min"])
    T_max = float(bounds["T_max"])
    E_min = float(bounds["E_min"])
    E_max = float(bounds["E_max"])
    O_min = float(bounds["O_min"])
    inside = C >= C_min and T_min <= T <= T_max and E_min <= E <= E_max and O >= O_min
    if inside:
        return float(min([C - C_min, T - T_min, T_max - T, E - E_min, E_max - E, O - O_min]))
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
    vals = out[STATE_COLS].to_numpy(dtype=float)
    out["signed_box_distance"] = np.array([signed_distance_to_viability_box_row(v, bounds) for v in vals], dtype=float)
    return out


def estimate_attractor_from_retained_points(df: pd.DataFrame) -> dict:
    work = df.copy()
    if "timestamp" in work.columns and work["timestamp"].notna().any():
        work = work.dropna(subset=["timestamp"]).sort_values(["trajectory_id", "timestamp"])
        tail = work.groupby("trajectory_id", as_index=False).tail(1)
    else:
        tail = work.groupby("trajectory_id", as_index=False).tail(1)
    C = float(tail["C"].median())
    T = float(tail["T"].median())
    E = float(tail["E"].median())
    O = float(tail["O"].median())
    distances = {
        "C_to_Cmin": C - float(DEFAULT_BOUNDS["C_min"]),
        "T_to_Tmin": T - float(DEFAULT_BOUNDS["T_min"]),
        "T_to_Tmax": float(DEFAULT_BOUNDS["T_max"]) - T,
        "E_to_Emin": E - float(DEFAULT_BOUNDS["E_min"]),
        "E_to_Emax": float(DEFAULT_BOUNDS["E_max"]) - E,
        "O_to_Omin": O - float(DEFAULT_BOUNDS["O_min"]),
    }
    return {
        "C": C,
        "T": T,
        "E": E,
        "O": O,
        "distances": distances,
        "min_signed_distance": float(min(distances.values())),
        "n_terminal_points": int(len(tail)),
    }


def summarize_sets(raw_df: pd.DataFrame, surviving_df: pd.DataFrame) -> dict:
    return {
        "total_trajectories": int(raw_df["trajectory_id"].nunique()),
        "surviving_trajectories": int(surviving_df["trajectory_id"].nunique()),
        "total_points": int(len(raw_df)),
        "surviving_points": int(len(surviving_df)),
        "max_distance_surviving": float(surviving_df["distance_from_attractor_4d"].max()),
    }


def summary_text(summary: dict) -> str:
    return (
        f"Traj: total={summary['total_trajectories']}, surviving={summary['surviving_trajectories']} | "
        f"Points: total={summary['total_points']}, surviving={summary['surviving_points']} | "
        f"Max surviving dist={summary['max_distance_surviving']:.3g}"
    )


def resample_trajectory_arclength(group: pd.DataFrame, *, space_step: float, attractor: np.ndarray, classifier_name: str) -> pd.DataFrame:
    g = group.sort_values("timestamp") if "timestamp" in group.columns and group["timestamp"].notna().any() else group.copy()
    xyz = g[STATE_COLS].to_numpy(dtype=float)
    if len(xyz) == 0:
        return g.iloc[0:0].copy()
    if len(xyz) == 1:
        out = g.iloc[[0]].copy()
        out["distance_from_attractor_4d"] = np.linalg.norm(out[STATE_COLS].to_numpy(dtype=float) - attractor, axis=1)
        out["signed_box_distance"] = np.array([signed_distance_to_viability_box_row(out[STATE_COLS].iloc[0].to_numpy(dtype=float), DEFAULT_BOUNDS)])
        return out

    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0:
        out = g.iloc[[0]].copy()
        out["distance_from_attractor_4d"] = np.linalg.norm(out[STATE_COLS].to_numpy(dtype=float) - attractor, axis=1)
        out["signed_box_distance"] = np.array([signed_distance_to_viability_box_row(out[STATE_COLS].iloc[0].to_numpy(dtype=float), DEFAULT_BOUNDS)])
        return out

    targets = np.arange(0.0, total + 0.5 * space_step, space_step)
    if targets[-1] < total:
        targets = np.append(targets, total)

    coords = {}
    for idx, col in enumerate(STATE_COLS):
        coords[col] = np.interp(targets, s, xyz[:, idx])

    out = pd.DataFrame({
        "trajectory_id": int(g["trajectory_id"].iloc[0]),
        **coords,
    })
    if "scenario_label" in g.columns:
        out["scenario_label"] = g["scenario_label"].iloc[0]
    if "classifier_type" in g.columns:
        out["classifier_type"] = classifier_name
    out["distance_from_attractor_4d"] = np.linalg.norm(out[STATE_COLS].to_numpy(dtype=float) - attractor, axis=1)
    out["signed_box_distance"] = np.array([signed_distance_to_viability_box_row(v, DEFAULT_BOUNDS) for v in out[STATE_COLS].to_numpy(dtype=float)], dtype=float)
    out["viability_label"] = np.where(out["signed_box_distance"] >= 0.0, "viable", "non_viable")

    classifier = get_classifier_callable(classifier_name)
    phenos = []
    for _, row in out.iterrows():
        state = row[STATE_COLS].to_numpy(dtype=float)
        try:
            label = classifier(state, {}) if classifier_name != "static" else classifier(state)
        except TypeError:
            label = classifier(state)
        phenos.append(str(label))
    out["phenotypical_label"] = phenos
    return out


def maybe_spatially_resample(df: pd.DataFrame, *, sampling_mode: str, space_step: float, classifier_name: str) -> pd.DataFrame:
    if sampling_mode == "raw":
        return df.copy()
    if space_step <= 0:
        raise ValueError("--space-step must be positive when using arc-length resampling.")
    attractor_info = estimate_attractor_from_retained_points(df)
    attractor = np.array([attractor_info["C"], attractor_info["T"], attractor_info["E"], attractor_info["O"]], dtype=float)
    pieces = []
    for _, group in df.groupby("trajectory_id"):
        pieces.append(resample_trajectory_arclength(group, space_step=space_step, attractor=attractor, classifier_name=classifier_name))
    out = pd.concat(pieces, ignore_index=True)
    out = out[out["viability_label"].astype(str) == "viable"].copy()
    if out.empty:
        raise ValueError("Arc-length resampling produced no viable points.")
    return out


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
    first_valid, last_valid = valid[0], valid[-1]
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


def plot_phenotype_distance_histogram(df: pd.DataFrame, output_path: Path, *, phenotype_distance_bins: int, phenotype_distance_mode: str, top_axis_ticks: int, summary: dict, title_stub: str, sampling_mode: str, dpi: int) -> None:
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
        ax.bar(centers, plot_counts[idx, :], width=widths, bottom=bottoms, color=color, edgecolor="white", linewidth=0.5, align="center", label=label)
        bottoms += plot_counts[idx, :]
    ax.set_title(f"Phenotype labels versus 4D distance from attractor\n{title_stub} | sampling={sampling_mode}\n{summary_text(summary)}", fontsize=13, fontweight="bold")
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


def plot_cumulative_viable_distance(df: pd.DataFrame, output_path: Path, *, cumulative_distance_bins: int, summary: dict, title_stub: str, sampling_mode: str, dpi: int) -> None:
    x, y, n = build_cumulative_viable_curve(df, n_bins=cumulative_distance_bins)
    fig, ax = plt.subplots(figsize=(10.4, 6.2), constrained_layout=True)
    ax.step(x, y, where="post", color=CUMULATIVE_COLOR, linewidth=2.4, label=f"Cumulative % of surviving viable points (N={n})")
    ax.set_title(f"Cumulative surviving viable points versus 4D distance from attractor\n{title_stub} | sampling={sampling_mode}\n{summary_text(summary)}", fontsize=13, fontweight="bold")
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
    classifier_name = infer_classifier_type(raw_df, args.classifier_type)
    sampled_df = maybe_spatially_resample(surviving_df, sampling_mode=args.sampling_mode, space_step=args.space_step, classifier_name=classifier_name)
    sampled_df = attach_signed_box_distance(sampled_df, DEFAULT_BOUNDS)

    if args.output_dir is None:
        output_dir = csv_path.parent / "figures"
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    title_stub = infer_title_stub(sampled_df, csv_path)
    summary = summarize_sets(raw_df, sampled_df)

    stem = csv_path.stem
    suffix = args.sampling_mode if args.sampling_mode == "raw" else f"{args.sampling_mode}_step_{args.space_step:g}"
    phenotype_png = output_dir / f"{stem}_phenotype_distance_stacked_{args.phenotype_distance_mode}_{suffix}.png"
    cumulative_png = output_dir / f"{stem}_viable_distance_cumulative_percent_{suffix}.png"

    plot_phenotype_distance_histogram(
        sampled_df,
        phenotype_png,
        phenotype_distance_bins=args.phenotype_distance_bins,
        phenotype_distance_mode=args.phenotype_distance_mode,
        top_axis_ticks=args.top_axis_ticks,
        summary=summary,
        title_stub=title_stub,
        sampling_mode=args.sampling_mode,
        dpi=args.dpi,
    )
    plot_cumulative_viable_distance(
        sampled_df,
        cumulative_png,
        cumulative_distance_bins=args.cumulative_distance_bins,
        summary=summary,
        title_stub=title_stub,
        sampling_mode=args.sampling_mode,
        dpi=args.dpi,
    )

    print(f"Sampling mode: {args.sampling_mode}")
    if args.sampling_mode == "arclength":
        print(f"Space step: {args.space_step}")
    print(f"Total trajectories: {int(raw_df['trajectory_id'].nunique())}")
    print(f"Surviving trajectories: {int(surviving_df['trajectory_id'].nunique())}")
    print(f"Total points: {len(raw_df)}")
    print(f"Points used in plots: {len(sampled_df)}")
    print(f"Max plotted distance from attractor: {float(sampled_df['distance_from_attractor_4d'].max())}")
    if args.print_attractor_summary:
        attractor = estimate_attractor_from_retained_points(sampled_df)
        print("Estimated attractor CETO position (from median terminal plotted points):")
        print(f"  C={attractor['C']:.6g}, T={attractor['T']:.6g}, E={attractor['E']:.6g}, O={attractor['O']:.6g}")
        print("Signed distances to viability-box faces:")
        for key, value in attractor['distances'].items():
            print(f"  {key}={value:.6g}")
        print(f"Minimum signed distance to viability box: {attractor['min_signed_distance']:.6g}")
        print(f"Terminal points used for attractor estimate: {attractor['n_terminal_points']}")
    print(f"Phenotype-distance plot: {phenotype_png}")
    print(f"Cumulative viable-distance plot: {cumulative_png}")


if __name__ == "__main__":
    main()
