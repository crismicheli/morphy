
#!/usr/bin/env python3
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

from config import DEFAULT_BOUNDS, SCENARIOS

STATE_COLS = ["C", "T", "E", "O"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot minimum signed attractor distance to boundaries versus porosity and tabulate all scenarios.")
    parser.add_argument("--csv-dir", required=True, help="Directory containing one point-record CSV per scenario.")
    parser.add_argument("--output-dir", default="output/min_signed_distance_from_scenarios", help="Output directory.")
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI.")
    return parser.parse_args()


def scenario_label(s: dict, idx: int) -> str:
    raw = str(s.get("label") or s.get("name") or s.get("scenario") or f"scenario_{idx+1}")
    raw = re.sub(r"\b(stable|borderline|unstable)\b", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+", " ", raw).strip(" -_,;:")
    return raw or f"scenario_{idx+1}"


def extract_p(label: str, scenario: dict) -> float | None:
    if isinstance(scenario, dict) and scenario.get("p") is not None:
        try:
            return float(scenario.get("p"))
        except Exception:
            pass
    m = re.search(r"\(\s*p\s*=\s*([0-9]*\.?[0-9]+)", label)
    return float(m.group(1)) if m else None


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


def load_point_records(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"trajectory_id", *STATE_COLS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    out = df.copy()
    out["trajectory_id"] = pd.to_numeric(out["trajectory_id"], errors="coerce")
    for col in STATE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["trajectory_id", *STATE_COLS])
    out["trajectory_id"] = out["trajectory_id"].astype(int)
    return out


def estimate_attractor(df: pd.DataFrame) -> dict:
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
    values = np.array([C, T, E, O], dtype=float)
    min_signed = signed_distance_to_viability_box_row(values, DEFAULT_BOUNDS)
    return {
        "C": C,
        "T": T,
        "E": E,
        "O": O,
        "min_signed_distance": float(min_signed),
        "n_terminal_points": int(len(tail)),
        "C_to_Cmin": C - float(DEFAULT_BOUNDS["C_min"]),
        "T_to_Tmin": T - float(DEFAULT_BOUNDS["T_min"]),
        "T_to_Tmax": float(DEFAULT_BOUNDS["T_max"]) - T,
        "E_to_Emin": E - float(DEFAULT_BOUNDS["E_min"]),
        "E_to_Emax": float(DEFAULT_BOUNDS["E_max"]) - E,
        "O_to_Omin": O - float(DEFAULT_BOUNDS["O_min"]),
    }


def find_csv_for_label(csv_dir: Path, label: str) -> Path:
    candidates = sorted(csv_dir.glob("*.csv"))
    label_tokens = re.sub(r"[^a-z0-9]+", " ", label.lower()).split()
    best = None
    best_score = -1
    for path in candidates:
        stem_tokens = re.sub(r"[^a-z0-9]+", " ", path.stem.lower()).split()
        score = sum(tok in stem_tokens for tok in label_tokens)
        if score > best_score:
            best = path
            best_score = score
    if best is None or best_score <= 0:
        raise FileNotFoundError(f"Could not match scenario label '{label}' to a CSV in {csv_dir}")
    return best


def build_all_scenarios_table(csv_dir: Path) -> pd.DataFrame:
    rows = []
    for i, scenario in enumerate(SCENARIOS):
        label = scenario_label(scenario, i)
        p = extract_p(label, scenario)
        csv_path = find_csv_for_label(csv_dir, label)
        df = load_point_records(csv_path)
        attractor = estimate_attractor(df)
        rows.append({
            "scenario": label,
            "csv_file": csv_path.name,
            "p": p,
            "attractor_C": attractor["C"],
            "attractor_T": attractor["T"],
            "attractor_E": attractor["E"],
            "attractor_O": attractor["O"],
            "C_to_Cmin": attractor["C_to_Cmin"],
            "T_to_Tmin": attractor["T_to_Tmin"],
            "T_to_Tmax": attractor["T_to_Tmax"],
            "E_to_Emin": attractor["E_to_Emin"],
            "E_to_Emax": attractor["E_to_Emax"],
            "O_to_Omin": attractor["O_to_Omin"],
            "min_signed_distance": attractor["min_signed_distance"],
            "n_terminal_points": attractor["n_terminal_points"],
        })
    out = pd.DataFrame(rows)
    return out


def select_porosity_three(df: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["p"]).copy()
    if len(work) < 3:
        raise ValueError("Fewer than 3 scenarios have a porosity value p.")
    ordered = work.sort_values("p").reset_index(drop=True)
    mid = len(ordered) // 2
    subset = ordered.iloc[[0, mid, len(ordered) - 1]].copy()
    subset = subset.drop_duplicates(subset=["p"]).sort_values("p").reset_index(drop=True)
    if len(subset) != 3:
        raise ValueError("Could not identify three distinct low/intermediate/high porosity scenarios.")
    return subset


def quadratic_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.poly1d]:
    coeffs = np.polyfit(x, y, deg=2)
    poly = np.poly1d(coeffs)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = poly(x_smooth)
    return x_smooth, y_smooth, poly


def plot_porosity_curve(df_three: pd.DataFrame, outpath: Path, dpi: int) -> None:
    x = df_three["p"].to_numpy(dtype=float)
    y = df_three["min_signed_distance"].to_numpy(dtype=float)
    x_smooth, y_smooth, _poly = quadratic_curve(x, y)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax.scatter(x, y, s=80, color="tab:blue", zorder=3, label="Selected porosity scenarios")
    ax.plot(x_smooth, y_smooth, color="tab:orange", linewidth=2.2, label="Smooth quadratic curve")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)

    for _, row in df_three.iterrows():
        ax.annotate(row["scenario"], (row["p"], row["min_signed_distance"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    ax.set_xlabel("Porosity p")
    ax.set_ylabel("Minimum signed distance to boundaries")
    ax.set_title("Minimum signed attractor distance to boundaries vs porosity")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_df = build_all_scenarios_table(csv_dir)
    porosity_df = select_porosity_three(all_df)

    all_csv = output_dir / "all_9_scenarios_boundary_distances.csv"
    three_csv = output_dir / "three_porosity_scenarios_boundary_distances.csv"
    png = output_dir / "min_signed_distance_vs_porosity_three_scenarios.png"

    all_df.to_csv(all_csv, index=False)
    porosity_df.to_csv(three_csv, index=False)
    plot_porosity_curve(porosity_df, png, dpi=args.dpi)

    print(f"All-scenarios table: {all_csv}")
    print(f"Three-scenario table: {three_csv}")
    print(f"Plot: {png}")


if __name__ == "__main__":
    main()
