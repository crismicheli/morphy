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

Trajectory filtering
--------------------
The plotted point set can be chosen with:
- fully-viable: only trajectories whose points are all viability_label == viable
- all: all trajectories
- non-fully-viable: only trajectories containing at least one non-viable point

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

from classifiers.static_classifier import STATE_COLORS
from config import DEFAULT_BOUNDS

FALLBACK_COLOR = "#7f7f7f"
CUMULATIVE_COLOR = "#2166ac"
STATE_COLS = ["C", "T", "E", "O"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot phenotype-distance and cumulative distance summaries from a point-record CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to the point-record CSV produced by export_trajectory_point_records.py.")
    parser.add_argument("--output-dir", default=None, help="Output directory for PNG figures. Defaults to a sibling 'figures' folder next to the CSV.")
    parser.add_argument("--phenotype-distance-bins", type=int, default=40, help="Number of equally spaced distance bins for the phenotype-distance stacked histogram.")
    parser.add_argument("--phenotype-distance-mode", choices=["count", "percent"], default="percent", help="Show phenotype-by-distance as raw counts or 100%% stacked percentages.")
    parser.add_argument("--cumulative-distance-bins", type=int, default=80, help="Number of equally spaced bins for the cumulative distance curve.")
    parser.add_argument("--top-axis-ticks", type=int, default=6, help="Maximum number of labeled ticks on the top complementary x-axis.")
    parser.add_argument("--print-attractor-summary", action="store_true", help="Print the estimated attractor position in CETO coordinates and its signed minimum distances to the viability-box faces.")
    parser.add_argument("--trajectory-filter", choices=["fully-viable", "all", "non-fully-viable"], default="fully-viable", help="Choose which trajectories contribute points to the plots.")
    parser.add_argument("--sampling-mode", choices=["raw", "arclength"], default="raw", help="Use original time-sampled points or arc-length-rediscretized state-space samples.")
    parser.add_argument("--space-step", type=float, default=0.02, help="Target 4D CETO arc-length spacing when --sampling-mode arclength is used.")
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


def classify_trajectory_ids(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    fully_viable_ids = []
    non_fully_viable_ids = []
    for traj_id, group in df.groupby("trajectory_id"):
        labels = group["viability_label"].astype(str)
        if bool((labels == "viable").all()):
            fully_viable_ids.append(int(traj_id))
        else:
            non_fully_viable_ids.append(int(traj_id))
    return fully_viable_ids, non_fully_viable_ids


def select_trajectories(df: pd.DataFrame, trajectory_filter: str) -> tuple[pd.DataFrame, dict]:
    fully_viable_ids, non_fully_viable_ids = classify_trajectory_ids(df)
    if trajectory_filter == "fully-viable":
        selected_ids = fully_viable_ids
    elif trajectory_filter == "non-fully-viable":
        selected_ids = non_fully_viable_ids
    else:
        selected_ids = sorted(set(fully_viable_ids).union(non_fully_viable_ids))

    selected = df[df["trajectory_id"].isin(selected_ids)].copy()
    if selected.empty:
        raise ValueError(f"No trajectories remain after applying --trajectory-filter {trajectory_filter!r}.")

    counts = {
        "total_trajectories": int(df["trajectory_id"].nunique()),
        "fully_viable_trajectories": int(len(fully_viable_ids)),
        "non_fully_viable_trajectories": int(len(non_fully_viable_ids)),
        "selected_trajectories": int(len(selected_ids)),
        "total_points": int(len(df)),
        "selected_points_raw": int(len(selected)),
    }
    return selected, counts


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


def summarize_sets(raw_df: pd.DataFrame, selected_df: pd.DataFrame, plotted_df: pd.DataFrame, counts: dict) -> dict:
    return {
        "total_trajectories": counts["total_trajectories"],
        "fully_viable_trajectories": counts["fully_viable_trajectories"],
        "non_fully_viable_trajectories": counts["non_fully_viable_trajectories"],
        "selected_trajectories": counts["selected_trajectories"],
        "total_points": counts["total_points"],
        "selected_points_raw": counts["selected_points_raw"],
        "plotted_points": int(len(plotted_df)),
        "max_distance_plotted": float(plotted_df["distance_from_attractor_4d"].max()),
        "selected_points_viable": int((selected_df["viability_label"].astype(str) == "viable").sum()),
        "selected_points_nonviable": int((selected_df["viability_label"].astype(str) != "viable").sum()),
    }


def summary_text(summary: dict) -> str:
    return (
        f"Traj selected={summary['selected_trajectories']} of {summary['total_trajectories']} | "
        f"Raw selected points={summary['selected_points_raw']} | plotted={summary['plotted_points']} | "
        f"Max plotted dist={summary['max_distance_plotted']:.3g}"
    )


def resample_trajectory_arclength(group: pd.DataFrame, *, space_step: float, attractor: np.ndarray) -> pd.DataFrame:
    g = group.sort_values("timestamp") if "timestamp" in group.columns and g_has_timestamp(group) else group.copy()
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
    if len(targets) == 0 or targets[-1] < total:
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
        out["classifier_type"] = g["classifier_type"].iloc[0]

    nearest_idx = np.searchsorted(s, targets, side="left")
    nearest_idx = np.clip(nearest_idx, 0, len(s) - 1)
    prev_idx = np.clip(nearest_idx - 1, 0, len(s) - 1)
    choose_prev = np.abs(targets - s[prev_idx]) <= np.abs(s[nearest_idx] - targets)
    nearest_idx = np.where(choose_prev, prev_idx, nearest_idx)

    out["phenotypical_label"] = g["phenotypical_label"].astype(str).to_numpy()[nearest_idx]
    out["viability_label"] = g["viability_label"].astype(str).to_numpy()[nearest_idx]
    out["distance_from_attractor_4d"] = np.linalg.norm(out[STATE_COLS].to_numpy(dtype=float) - attractor, axis=1)
    out["signed_box_distance"] = np.array([signed_distance_to_viability_box_row(v, DEFAULT_BOUNDS) for v in out[STATE_COLS].to_numpy(dtype=float)], dtype=float)
    if "timestamp" in g.columns and g_has_timestamp(g):
        t = g["timestamp"].to_numpy(dtype=float)
        out["timestamp"] = np.interp(targets, s, t)
    return out


def g_has_timestamp(g: pd.DataFrame) -> bool:
    return "timestamp" in g.columns and g["timestamp"].notna().any()


def maybe_spatially_resample(df: pd.DataFrame, *, sampling_mode: str, space_step: float) -> pd.DataFrame:
    if sampling_mode == "raw":
        return df.copy()
    if space_step <= 0:
        raise ValueError("--space-step must be positive when using arc-length resampling.")
    attractor_info = estimate_attractor_from_retained_points(df)
    attractor = np.array([attractor_info["C"], attractor_info["T"], attractor_info["E"], attractor_info["O"]], dtype=float)
    pieces = []
    for _, group in df.groupby("trajectory_id"):
        pieces.append(resample_trajectory_arclength(group, space_step=space_step, attractor=attractor))
    out = pd.concat(pieces, ignore_index=True)
    if out.empty:
        raise ValueError("Arc-length resampling produced no points.")
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


def plot_phenotype_distance_histogram(df: pd.DataFrame, output_path: Path, *, phenotype_distance_bins: int, phenotype_distance_mode: str, top_axis_ticks: int, summary: dict, title_stub: str, sampling_mode: str, trajectory_filter: str, dpi: int) -> None:
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
        ylabel = "Number of plotted points"
    for idx, label in enumerate(labels):
        color = STATE_COLORS.get(label, FALLBACK_COLOR)
        ax.bar(centers, plot_counts[idx, :], width=widths, bottom=bottoms, color=color, edgecolor="white", linewidth=0.5, align="center", label=label)
        bottoms += plot_counts[idx, :]
    ax.set_title(f"Phenotype labels versus 4D distance from attractor\n{title_stub} | filter={trajectory_filter} | sampling={sampling_mode}\n{summary_text(summary)}", fontsize=13, fontweight="bold")
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


def plot_cumulative_viable_distance(df: pd.DataFrame, output_path: Path, *, cumulative_distance_bins: int, summary: dict, title_stub: str, sampling_mode: str, trajectory_filter: str, dpi: int) -> None:
    x, y, n = build_cumulative_viable_curve(df, n_bins=cumulative_distance_bins)
    fig, ax = plt.subplots(figsize=(10.4, 6.2), constrained_layout=True)
    ax.step(x, y, where="post", color=CUMULATIVE_COLOR, linewidth=2.4, label=f"Cumulative % of plotted points (N={n})")
    ax.set_title(f"Cumulative plotted points versus 4D distance from attractor\n{title_stub} | filter={trajectory_filter} | sampling={sampling_mode}\n{summary_text(summary)}", fontsize=13, fontweight="bold")
    ax.set_xlabel("4D distance from attractor")
    ax.set_ylabel("Cumulative [% of plotted points]")
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
    selected_df, counts = select_trajectories(raw_df, args.trajectory_filter)
    plotted_df = maybe_spatially_resample(selected_df, sampling_mode=args.sampling_mode, space_step=args.space_step)
    plotted_df = attach_signed_box_distance(plotted_df, DEFAULT_BOUNDS)

    if args.output_dir is None:
        output_dir = csv_path.parent / "figures"
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    title_stub = infer_title_stub(plotted_df, csv_path)
    summary = summarize_sets(raw_df, selected_df, plotted_df, counts)

    stem = csv_path.stem
    suffix = args.sampling_mode if args.sampling_mode == "raw" else f"{args.sampling_mode}_step_{args.space_step:g}"
    phenotype_png = output_dir / f"{stem}_phenotype_distance_stacked_{args.phenotype_distance_mode}_{args.trajectory_filter}_{suffix}.png"
    cumulative_png = output_dir / f"{stem}_distance_cumulative_percent_{args.trajectory_filter}_{suffix}.png"

    plot_phenotype_distance_histogram(
        plotted_df,
        phenotype_png,
        phenotype_distance_bins=args.phenotype_distance_bins,
        phenotype_distance_mode=args.phenotype_distance_mode,
        top_axis_ticks=args.top_axis_ticks,
        summary=summary,
        title_stub=title_stub,
        sampling_mode=args.sampling_mode,
        trajectory_filter=args.trajectory_filter,
        dpi=args.dpi,
    )
    plot_cumulative_viable_distance(
        plotted_df,
        cumulative_png,
        cumulative_distance_bins=args.cumulative_distance_bins,
        summary=summary,
        title_stub=title_stub,
        sampling_mode=args.sampling_mode,
        trajectory_filter=args.trajectory_filter,
        dpi=args.dpi,
    )

    print(f"Trajectory filter: {args.trajectory_filter}")
    print(f"Sampling mode: {args.sampling_mode}")
    if args.sampling_mode == "arclength":
        print(f"Space step: {args.space_step}")
    print(f"Total trajectories: {summary['total_trajectories']}")
    print(f"Fully viable trajectories: {summary['fully_viable_trajectories']}")
    print(f"Non-fully-viable trajectories: {summary['non_fully_viable_trajectories']}")
    print(f"Selected trajectories: {summary['selected_trajectories']}")
    print(f"Total points: {summary['total_points']}")
    print(f"Raw selected points: {summary['selected_points_raw']}")
    print(f"Selected viable points: {summary['selected_points_viable']}")
    print(f"Selected non-viable points: {summary['selected_points_nonviable']}")
    print(f"Plotted points: {summary['plotted_points']}")
    print(f"Max plotted distance from attractor: {summary['max_distance_plotted']}")
    if args.print_attractor_summary:
        attractor = estimate_attractor_from_retained_points(plotted_df)
        print("Estimated attractor CETO position (from median terminal plotted points):")
        print(f"  C={attractor['C']:.6g}, T={attractor['T']:.6g}, E={attractor['E']:.6g}, O={attractor['O']:.6g}")
        print("Signed distances to viability-box faces:")
        for key, value in attractor['distances'].items():
            print(f"  {key}={value:.6g}")
        print(f"Minimum signed distance to viability box: {attractor['min_signed_distance']:.6g}")
        print(f"Terminal points used for attractor estimate: {attractor['n_terminal_points']}")
    print(f"Phenotype-distance plot: {phenotype_png}")
    print(f"Cumulative distance plot: {cumulative_png}")


if __name__ == "__main__":
    main()
