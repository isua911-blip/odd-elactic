#!/usr/bin/env python3
"""Recompute abrupt-cut work for a family of anisotropic mobilities."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from crack_advance_work import even_energy
from same_endpoint_scaling import ScalingConfig, full_odd_work, prepare

KO_VALUES = (0.0, 0.12, 0.222271)
RATIOS = (0.5, 1.0, 2.0, 4.0)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_case(cfg: ScalingConfig, ko: float, ratio: float) -> dict[str, object]:
    result = full_odd_work(cfg, ko, ratio, 0.0)
    _model, Ee_old, Ee_new, _Eo_new, _Knew, u0, _u1, _c, _cv, _f = prepare(cfg, ko)
    cut_energy = even_energy(Ee_old, u0) - even_energy(Ee_new, u0)
    balance = (
        float(result["odd_work"])
        - float(result["delta_even_energy"])
        - float(result["dissipation"])
        - cut_energy
    )
    return {
        "k_o": ko,
        "mobility_ratio": ratio,
        "odd_work": result["odd_work"],
        "dissipation": result["dissipation"],
        "protocol_work": result["protocol_work"],
        "cut_energy": cut_energy,
        "balance": balance,
        "final_relative_norm": result["final_relative_norm"],
        "steps": result["steps"],
        "t_end": cfg.t_end,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PACKAGE_ROOT / "data" / "protocol_family_results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = ScalingConfig(
        nx=48,
        ny=36,
        crack_half_length=6.0,
        t_end=70000.0,
        rtol=2.0e-8,
        relative_atol=1.0e-10,
    )
    rows: list[dict[str, object]] = []
    for ko in KO_VALUES:
        for ratio in RATIOS:
            print(f"mobility protocol ko={ko} ratio={ratio}", flush=True)
            rows.append(compute_case(cfg, ko, ratio))
    write_rows(args.out / "same_endpoint_mobility_protocol.csv", rows)

    normalized: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for ko in KO_VALUES:
        group = [row for row in rows if abs(float(row["k_o"]) - ko) < 1e-14]
        iso = next(row for row in group if abs(float(row["mobility_ratio"]) - 1.0) < 1e-14)
        for row in group:
            out = dict(row)
            out["protocol_work_over_isotropic"] = float(row["protocol_work"]) / float(iso["protocol_work"])
            out["odd_work_over_isotropic"] = (
                float(row["odd_work"]) / float(iso["odd_work"])
                if abs(float(iso["odd_work"])) > 1e-30
                else ""
            )
            normalized.append(out)
        pvals = np.array([float(row["protocol_work"]) for row in group])
        ovals = np.array([float(row["odd_work"]) for row in group])
        summaries.append(
            {
                "k_o": ko,
                "protocol_work_min": float(pvals.min()),
                "protocol_work_max": float(pvals.max()),
                "protocol_work_span_relative": float((pvals.max() - pvals.min()) / abs(float(iso["protocol_work"]))),
                "odd_work_min": float(ovals.min()),
                "odd_work_max": float(ovals.max()),
                "odd_work_span_relative": (
                    float((ovals.max() - ovals.min()) / abs(float(iso["odd_work"]))) if abs(float(iso["odd_work"])) > 1e-30 else 0.0
                ),
                "maximum_balance_abs": max(abs(float(row["balance"])) for row in group),
                "maximum_final_relative_norm": max(float(row["final_relative_norm"]) for row in group),
            }
        )
    write_rows(args.out / "same_endpoint_mobility_protocol_normalized.csv", normalized)
    write_rows(args.out / "same_endpoint_mobility_summary.csv", summaries)
    print(json.dumps({"rows": len(rows), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
