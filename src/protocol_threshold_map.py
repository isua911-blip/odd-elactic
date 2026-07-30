#!/usr/bin/env python3
"""Recompute protocol-work and independent bond-threshold comparisons.

Unlike the submitted entry point, this version does not depend on an omitted
``cache/crack_advance`` directory. It calls the archived scientific solver
``crack_advance_work.protocol_unit_result`` directly and writes every CSV/JSON
file used by the manuscript and verification checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from active_tip_scan import ActiveCrackedStrip
from crack_advance_work import protocol_unit_result

DELTA_C = 0.02
LOADS = (0.85, 0.90, 0.95, 0.98)
CASES = (
    {
        "nx": 48,
        "ny": 36,
        "a": 6.0,
        "t_end": 70000.0,
        "ko_values": (0.0, 0.02, 0.05, 0.10, 0.115, 0.12, 0.15, 0.20, 0.25, 0.30),
    },
    {
        "nx": 64,
        "ny": 48,
        "a": 8.0,
        "t_end": 125000.0,
        "ko_values": (0.0, 0.02, 0.05, 0.10, 0.115, 0.12, 0.15, 0.20, 0.25),
    },
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_units(
    case: dict[str, object], worker_root: Path, refresh_cache: bool = False
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    case_dir = worker_root / f"{int(case['nx'])}x{int(case['ny'])}"
    case_dir.mkdir(parents=True, exist_ok=True)
    for ko in case["ko_values"]:
        token = f"{float(ko):+.12f}".replace("+", "p").replace("-", "m").replace(".", "p")
        cache_file = case_dir / f"unit_{token}.json"
        if cache_file.exists() and not refresh_cache:
            row = json.loads(cache_file.read_text(encoding="utf-8"))
            print(f"protocol unit Nx={case['nx']} ko={ko} [cached]", flush=True)
        else:
            print(f"protocol unit Nx={case['nx']} ko={ko}", flush=True)
            row = asdict(
                protocol_unit_result(
                    int(case["nx"]),
                    int(case["ny"]),
                    float(case["a"]),
                    float(ko),
                    "right",
                    float(case["t_end"]),
                )
            )
            cache_file.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
    return rows


def static_diag(nx: int, ny: int, a: float, ko: float) -> dict[str, float]:
    return ActiveCrackedStrip(nx=nx, ny=ny, crack_half_length=a, k=1.0, k_o=float(ko)).static_diagnostics(DELTA_C)


def linear_crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    for i in range(len(x) - 1):
        if (y[i] - level) * (y[i + 1] - level) <= 0:
            if y[i + 1] == y[i]:
                return float(0.5 * (x[i] + x[i + 1]))
            return float(x[i] + (level - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PACKAGE_ROOT / "data" / "protocol_threshold_results")
    parser.add_argument("--refresh-cache", action="store_true", help="recompute resumable protocol-unit worker files")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    threshold_rows: list[dict[str, object]] = []
    cache_rows: list[dict[str, object]] = []
    slope_rows: list[dict[str, object]] = []
    direct_rows: list[dict[str, object]] = []
    curves: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for case in CASES:
        nx, ny, a = int(case["nx"]), int(case["ny"]), float(case["a"])
        data = compute_units(
            case, PACKAGE_ROOT / "data_recomputed" / "protocol_threshold_workers", args.refresh_cache
        )
        ko = np.array([d["k_o"] for d in data], float)
        stress = np.array([d["remote_stress_unit"] for d in data], float)
        work = np.array([d["protocol_work_unit"] for d in data], float)
        ext = np.array([d["candidate_extension_unit"] for d in data], float)
        i0 = int(np.argmin(np.abs(ko)))
        s0, A0, e0 = stress[i0], work[i0], ext[i0]
        r = s0**2 * work / (stress**2 * A0)
        b = s0 * ext / (stress * e0)
        rp = PchipInterpolator(ko, r)
        bp = PchipInterpolator(ko, b)
        rlin = interp1d(ko, r, kind="linear")
        dr0 = float(rp.derivative()(0.0))
        db0 = float(bp.derivative()(0.0))
        slope_rows.append(
            {
                "nx": nx,
                "ny": ny,
                "r_protocol_prime_zero": dr0,
                "bond_prime_zero": db0,
                "rprime_over_2bprime": dr0 / (2 * db0),
            }
        )

        passive = static_diag(nx, ny, a, 0.0)
        PG = 0.5 * (passive["initiation_sigma_left"] + passive["initiation_sigma_right"])
        deltaG = 0.5 * (passive["initiation_delta_left"] + passive["initiation_delta_right"])
        GcP = deltaG**2 * A0

        for d, ri, bi in zip(data, r, b):
            cache_rows.append(
                {
                    "nx": nx,
                    "ny": ny,
                    "crack_half_length": a,
                    "k_o": d["k_o"],
                    "r_protocol_unit": ri,
                    "b_bond_unit": bi,
                    "protocol_work_unit": d["protocol_work_unit"],
                    "viscous_dissipation_unit": d["viscous_dissipation_unit"],
                    "remote_stress_unit": d["remote_stress_unit"],
                    "candidate_extension_unit": d["candidate_extension_unit"],
                    "full_balance_residual_unit": d["full_balance_residual_unit"],
                    "final_state_relative_norm": d["final_state_relative_norm"],
                    "t_end": d["t_end"],
                }
            )

        for p in LOADS:
            target_r = 1 / p**2
            target_b = 1 / p
            kp = brentq(lambda x: float(rp(x)) - target_r, 0.0, float(ko.max()), xtol=1e-12)
            kp_lin = linear_crossing(ko, r, target_r)

            def bond_eq(x: float) -> float:
                return static_diag(nx, ny, a, x)["initiation_sigma_right"] / PG - p

            kb = brentq(bond_eq, 0.0, 0.45, xtol=2e-10, rtol=2e-10)
            kb_interp = brentq(lambda x: float(bp(x)) - target_b, 0.0, float(ko.max()), xtol=1e-12)
            A_at_bond = float(p**2 * rp(kb))
            A_at_bond_lin = float(p**2 * rlin(kb))
            bond_at_protocol = float(p * bp(kp))
            threshold_rows.append(
                {
                    "nx": nx,
                    "ny": ny,
                    "crack_half_length": a,
                    "load_fraction": p,
                    "passive_initiation_stress": PG,
                    "protocol_resistance": GcP,
                    "protocol_threshold_k_o": kp,
                    "bond_threshold_k_o": kb,
                    "protocol_minus_bond": kp - kb,
                    "critical_k_o_relative_error": abs(kp - kb) / kb,
                    "protocol_work_ratio_at_bond_threshold": A_at_bond,
                    "protocol_work_ratio_at_bond_threshold_linear_interp": A_at_bond_lin,
                    "bond_extension_ratio_at_protocol_threshold": bond_at_protocol,
                    "protocol_root_pchip_minus_linear": kp - kp_lin,
                    "bond_root_interp_abs_error": abs(kb_interp - kb),
                    "relative_dissipation_shift_at_bond_threshold": A_at_bond - 1.0,
                }
            )

            if abs(p - 0.85) < 1e-12:
                direct = asdict(protocol_unit_result(nx, ny, a, kb, "right", float(case["t_end"])))
                (out / f"direct_bondroot_{nx}x{ny}_p085.json").write_text(
                    json.dumps(direct, indent=2), encoding="utf-8"
                )
                delta = p * PG / float(direct["remote_stress_unit"])
                direct_ratio = delta**2 * float(direct["protocol_work_unit"]) / GcP
                direct_rows.append(
                    {
                        "nx": nx,
                        "ny": ny,
                        "load_fraction": p,
                        "bond_threshold_k_o": kb,
                        "direct_protocol_work_ratio_at_bond_threshold": direct_ratio,
                        "pchip_protocol_work_ratio_at_bond_threshold": A_at_bond,
                        "abs_interpolation_error": abs(direct_ratio - A_at_bond),
                        "full_balance_residual_unit": direct["full_balance_residual_unit"],
                        "final_state_relative_norm": direct["final_state_relative_norm"],
                    }
                )
        curves[nx] = (ko, r, b)

    write_rows(out / "protocol_threshold_systematic.csv", threshold_rows)
    write_rows(out / "protocol_unit_cache.csv", cache_rows)
    write_rows(out / "small_ko_slope_match.csv", slope_rows)
    write_rows(out / "direct_protocol_validation.csv", direct_rows)

    markers = {48: "o", 64: "s"}
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for nx in (48, 64):
        rr = sorted([r for r in threshold_rows if r["nx"] == nx], key=lambda z: z["load_fraction"])
        p = np.array([r["load_fraction"] for r in rr])
        kp = np.array([r["protocol_threshold_k_o"] for r in rr])
        kb = np.array([r["bond_threshold_k_o"] for r in rr])
        ax.plot(p, kb, marker=markers[nx], linestyle="-", label=f"{nx} bond threshold")
        ax.plot(p, kp, marker=markers[nx], linestyle="--", label=f"{nx} protocol threshold")
    ax.set_xlabel(r"load fraction $P/P_G^{\rm lat}$")
    ax.set_ylabel(r"critical odd coefficient $k_{o,c}/k$")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "protocol_vs_bond_threshold_map.pdf")
    plt.close(fig)

    ko48, r48, b48 = curves[48]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].plot(ko48, 0.9**2 * r48, "o-", label=r"$A^{\mathcal{P}}/G_c^{\mathcal{P}}$")
    axes[0].plot(ko48, 0.9 * b48, "s-", label=r"$\delta_{\rm tip}/\delta_c$")
    axes[0].axhline(1.0, linestyle="--", linewidth=1.0)
    axes[0].set_xlim(0, 0.25)
    axes[0].set_xlabel(r"$k_o/k$")
    axes[0].set_ylabel("normalized initiation measure")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    for nx in (48, 64):
        rr = sorted([r for r in threshold_rows if r["nx"] == nx], key=lambda z: z["load_fraction"])
        p = np.array([r["load_fraction"] for r in rr])
        kp = np.array([r["protocol_threshold_k_o"] for r in rr])
        kb = np.array([r["bond_threshold_k_o"] for r in rr])
        axes[1].plot(p, kb, marker=markers[nx], linestyle="-", label=f"{nx} bond")
        axes[1].plot(p, kp, marker=markers[nx], linestyle="--", label=f"{nx} protocol")
    axes[1].set_xlabel(r"$P/P_G^{\rm lat}$")
    axes[1].set_ylabel(r"$k_{o,c}/k$")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    for label, axis in zip(("a", "b"), axes):
        axis.text(0.02, 0.96, f"({label})", transform=axis.transAxes, va="top", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig3_protocol_threshold_systematic.pdf")
    plt.close(fig)

    summary = {
        "loads": list(LOADS),
        "sizes": ["48x36", "64x48"],
        "maximum_protocol_work_ratio_deviation_at_bond_threshold": max(
            abs(r["protocol_work_ratio_at_bond_threshold"] - 1) for r in threshold_rows
        ),
        "maximum_critical_ko_relative_error": max(r["critical_k_o_relative_error"] for r in threshold_rows),
        "near_passive_slope_ratios": slope_rows,
        "maximum_direct_interpolation_abs_error": max(r["abs_interpolation_error"] for r in direct_rows),
        "interpretation": (
            "The passively calibrated work measure remains close to unity at the independent bond threshold, "
            "but its inferred critical k_o becomes increasingly biased deeper below the passive initiation load."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
