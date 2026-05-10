#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_BOUNDS, DEFAULT_PARAMS
from plotting.scenario_helpers import choose_scenario
from viabilitykernels.odes import rhs
from plotting.plot_helpers import save_quiver_field_plot

DEFAULT_OUTDIR = ROOT / "figures" / "all_9_scenarios_3d_quiver_field"
SUMMARY_NAME = "sweep_quiver_field_summary.txt"

TARGET_SCENARIOS = [
    ("Low porosity", "low_porosity_p015"),
    ("Intermediate porosity", "intermediate_porosity_p040"),
    ("High porosity", "high_porosity_p075"),
    ("Stiff scaffold", "stiff_scaffold_eta18"),
    ("Hypoxic environment", "hypoxic_environment"),
    ("Over-tensioned", "over_tensioned_beta35"),
    ("Fast ECM remodelling", "fast_ecm_remodelling_deltae12"),
    ("Enhanced guidance", "enhanced_guidance_a60"),
    ("Near-critical asymmetric regime", "near_critical_asymmetric"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep 9 scenarios and save static 3D colored quiver field plots in T-E-O space."
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTDIR), help="Directory for saved figures.")
    parser.add_argument("--c-slice", type=float, default=0.20, help="Fixed C value used for the T-E-O field slice.")
    parser.add_argument("--nT", type=int, default=9, help="Grid resolution along T.")
    parser.add_argument("--nE", type=int, default=9, help="Grid resolution along E.")
    parser.add_argument("--nO", type=int, default=7, help="Grid resolution along O.")
    parser.add_argument("--t-pad", type=float, default=0.12, help="Padding around T bounds.")
    parser.add_argument("--e-pad", type=float, default=0.12, help="Padding around E bounds.")
    parser.add_argument("--o-min", type=float, default=0.05, help="Minimum O for field grid.")
    parser.add_argument("--o-max", type=float, default=1.20, help="Maximum O for field grid.")
    parser.add_argument("--elev", type=float, default=24.0, help="3D camera elevation.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth.")
    parser.add_argument("--show-box", action="store_true", help="Show translucent viability box.")
    parser.add_argument("--quiver-mode", choices=["raw", "normalized", "binned"], default="normalized")
    parser.add_argument("--quiver-length", type=float, default=0.10, help="Displayed arrow length.")
    parser.add_argument("--quiver-alpha", type=float, default=0.45, help="Quiver transparency.")
    parser.add_argument("--quiver-linewidth", type=float, default=0.7, help="Quiver linewidth.")
    parser.add_argument("--binned-norm", action="store_true", help="Normalize averaged binned vectors after aggregation.")
    parser.add_argument("--bins", type=int, default=8, help="Binning resolution per axis for binned mode.")
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI.")
    parser.add_argument("--cmap-name", default="jet", help="Matplotlib colormap name for quiver coloring.")
    parser.add_argument("--dry-run", action="store_true", help="Plan outputs without rendering.")
    return parser.parse_args()


def make_field_grid(
    bounds: Dict[str, float],
    *,
    c_slice: float,
    nT: int,
    nE: int,
    nO: int,
    t_pad: float,
    e_pad: float,
    o_min: float,
    o_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    t_vals = np.linspace(max(0.01, bounds["T_min"] - t_pad), bounds["T_max"] + t_pad, nT)
    e_vals = np.linspace(max(0.01, bounds["E_min"] - e_pad), bounds["E_max"] + e_pad, nE)
    o_vals = np.linspace(max(0.01, o_min), o_max, nO)

    TT, EE, OO = np.meshgrid(t_vals, e_vals, o_vals, indexing="ij")
    C = np.full_like(TT, float(c_slice))

    points_4d = np.column_stack([
        C.ravel(),
        TT.ravel(),
        EE.ravel(),
        OO.ravel(),
    ])
    teo_origins = np.column_stack([
        TT.ravel(),
        EE.ravel(),
        OO.ravel(),
    ])
    return points_4d, teo_origins


def evaluate_field(points_4d: np.ndarray, *, p: float, par: Dict) -> np.ndarray:
    vecs = np.zeros((points_4d.shape[0], 3), dtype=float)
    for i, x in enumerate(points_4d):
        dx = np.asarray(rhs(0.0, x, p, par), dtype=float)
        vecs[i] = dx[1:4]
    return vecs


def plot_quiver_field(
    scenario_cfg: dict,
    output_path: Path,
    *,
    bounds: Dict[str, float],
    par: Dict,
    c_slice: float,
    nT: int,
    nE: int,
    nO: int,
    t_pad: float,
    e_pad: float,
    o_min: float,
    o_max: float,
    elev: float,
    azim: float,
    show_box: bool,
    quiver_mode: str,
    quiver_length: float,
    quiver_alpha: float,
    quiver_linewidth: float,
    bins: int,
    binned_norm: bool,
    dpi: int,
    cmap_name: str,
) -> None:
    effective_par = {**par, **scenario_cfg.get("param_overrides", {})}
    p = scenario_cfg["p"]

    points_4d, origins = make_field_grid(
        bounds,
        c_slice=c_slice,
        nT=nT,
        nE=nE,
        nO=nO,
        t_pad=t_pad,
        e_pad=e_pad,
        o_min=o_min,
        o_max=o_max,
    )
    vectors = evaluate_field(points_4d, p=p, par=effective_par)

    keep = np.linalg.norm(vectors, axis=1) > 1e-12
    origins = origins[keep]
    vectors = vectors[keep]

    save_quiver_field_plot(
        scenario_cfg,
        origins,
        vectors,
        output_path,
        bounds=bounds,
        c_slice=c_slice,
        elev=elev,
        azim=azim,
        show_box=show_box,
        quiver_mode=quiver_mode,
        quiver_length=quiver_length,
        quiver_alpha=quiver_alpha,
        quiver_linewidth=quiver_linewidth,
        bins=bins,
        binned_norm=binned_norm,
        dpi=dpi,
        cmap_name=cmap_name,
    )


def expected_output(out_dir: Path, slug: str, mode: str) -> Path:
    return out_dir / f"{slug}_quiver_field_3d_{mode}.png"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    total = 0

    for scenario_filter, slug in TARGET_SCENARIOS:
        scenario_cfg = choose_scenario(scenario_filter)
        output_path = expected_output(out_dir, slug, args.quiver_mode)
        total += 1

        header = (
            f"[{total:02d}/09] {scenario_cfg['label']} | "
            f"p={scenario_cfg['p']:.4f} | "
            f"mode={args.quiver_mode} | "
            f"C-slice={args.c_slice:.3f} | "
            f"cmap={args.cmap_name}"
        )
        saved_line = f"Expected output: {output_path}"
        print(header)
        print(saved_line)
        summary_lines.extend([header, saved_line, ""])

        if args.dry_run:
            continue

        plot_quiver_field(
            scenario_cfg,
            output_path,
            bounds=DEFAULT_BOUNDS,
            par=DEFAULT_PARAMS,
            c_slice=args.c_slice,
            nT=args.nT,
            nE=args.nE,
            nO=args.nO,
            t_pad=args.t_pad,
            e_pad=args.e_pad,
            o_min=args.o_min,
            o_max=args.o_max,
            elev=args.elev,
            azim=args.azim,
            show_box=args.show_box,
            quiver_mode=args.quiver_mode,
            quiver_length=args.quiver_length,
            quiver_alpha=args.quiver_alpha,
            quiver_linewidth=args.quiver_linewidth,
            bins=args.bins,
            binned_norm=args.binned_norm,
            dpi=args.dpi,
            cmap_name=args.cmap_name,
        )

    summary_path = out_dir / SUMMARY_NAME
    summary_lines.append(f"Completed planned runs: {total}")
    summary_lines.append(f"Summary file: {summary_path}")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    if args.dry_run:
        print(f"Dry run complete. Summary written to: {summary_path}")
    else:
        print(f"Completed {total} runs.")
        print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
