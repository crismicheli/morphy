#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

STATE_COLORS = {
    "Apoptosis": "#d73027",
    "Migration": "#4575b4",
    "Proliferation": "#1a9850",
    "Quiescence": "#66bd63",
    "Diversification": "#984ea3",
    "Undetermined": "#808080",
}


def terminal_snapshot(sol, window: int = 40) -> dict:
    w = max(5, min(window, sol.y.shape[1]))
    tail = sol.y[:, -w:]
    tail_t = sol.t[-w:]
    means = tail.mean(axis=1)
    dt = max(1e-12, float(np.mean(np.diff(tail_t))))
    deriv = np.gradient(tail, dt, axis=1)
    dmeans = deriv.mean(axis=1)
    spreads = np.ptp(tail, axis=1)
    return {
        "C": float(means[0]),
        "T": float(means[1]),
        "E": float(means[2]),
        "O": float(means[3]),
        "dC": float(dmeans[0]),
        "dT": float(dmeans[1]),
        "dE": float(dmeans[2]),
        "dO": float(dmeans[3]),
        "spread": float(np.max(spreads)),
    }


def classify_state(C, T, E, O, dC, dT, dE, dO, bounds):
    c_margin = 0.05
    t_margin = 0.08
    e_margin = 0.05
    o_margin = 0.06
    small = 0.015

    near_low_resource = (
        C < bounds["C_min"] + c_margin
        or T < bounds["T_min"] + t_margin
        or O < bounds["O_min"] + o_margin
    )

    near_viable_core = (
        C >= bounds["C_min"] + c_margin
        and T >= bounds["T_min"] + t_margin
        and T <= bounds["T_max"] - 2 * t_margin
        and E >= bounds["E_min"] + e_margin
        and E <= bounds["E_max"] - 0.15
        and O >= bounds["O_min"] + o_margin
    )

    if O < bounds["O_min"] or C < 0.8 * bounds["C_min"] or T < 0.8 * bounds["T_min"]:
        return "Apoptosis"
    if abs(dC) < small and abs(dT) < small and abs(dE) < small and abs(dO) < small and near_viable_core:
        return "Quiescence"
    if dE > small and dC >= -small and dT >= -small and O >= bounds["O_min"] and T >= bounds["T_min"]:
        return "Proliferation"
    if dC > small and abs(dE) < 2 * small and T >= bounds["T_min"] and O >= bounds["O_min"]:
        return "Migration"
    if (abs(dC) > small and abs(dT) > small and abs(dE) > small) or (E > bounds["E_max"]) or (T > bounds["T_max"]):
        return "Diversification"
    if near_low_resource:
        return "Apoptosis"
    return "Undetermined"


def classify_solution(sol, bounds, window: int = 40) -> dict:
    s = terminal_snapshot(sol, window=window)
    label = classify_state(s["C"], s["T"], s["E"], s["O"], s["dC"], s["dT"], s["dE"], s["dO"], bounds)
    s["label"] = label
    s["color"] = STATE_COLORS[label]
    return s


def classify_solutions(solutions, bounds, window: int = 40) -> list[dict]:
    return [classify_solution(sol, bounds=bounds, window=window) for sol in solutions]
