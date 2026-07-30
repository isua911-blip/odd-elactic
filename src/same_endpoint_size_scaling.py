#!/usr/bin/env python3
"""Finite-size scaling of the same-endpoint mobility-path functional.

This module contains only post-processing helpers.  The executable wrapper runs
individual mobility paths in isolated subprocesses because sparse BDF factor
memory can grow when many large lattice paths are solved in one interpreter.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    out_dir: Path,
    first_rows: list[dict[str, object]],
    full_rows: list[dict[str, object]],
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    first_rows.sort(key=lambda r: (int(r["nx"]), float(r["mobility_ratio"]), float(r["theta"])))
    full_rows.sort(key=lambda r: (int(r["nx"]), float(r["k_o"]), str(r["extreme"])))
    write_csv(out_dir / "first_order_size_map.csv", first_rows)
    write_csv(out_dir / "finite_ko_extreme_validation.csv", full_rows)

    sizes = sorted({int(r["nx"]) for r in first_rows})
    size_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for nx in sizes:
        rows = [r for r in first_rows if int(r["nx"]) == nx and not bool(r.get("extra_symmetry_point", False))]
        extra = [r for r in first_rows if int(r["nx"]) == nx and bool(r.get("extra_symmetry_point", False))]
        phi_values = np.array([float(r["phi_first_order"]) for r in rows])
        minimum = rows[int(np.argmin(phi_values))]
        maximum = rows[int(np.argmax(phi_values))]
        dphi = float(maximum["phi_first_order"]) - float(minimum["phi_first_order"])
        resistance = float(rows[0]["passive_resistance"])
        r1 = [r for r in rows if math.isclose(float(r["mobility_ratio"]), 1.0)]
        r1_span = max(float(r["phi_first_order"]) for r in r1) - min(float(r["phi_first_order"]) for r in r1)
        pair = next(r for r in rows if math.isclose(float(r["mobility_ratio"]), 2.0) and math.isclose(float(r["theta"]), 0.0))
        reciprocal = extra[0]
        reciprocal_error = abs(float(pair["phi_first_order"]) - float(reciprocal["phi_first_order"]))
        size_rows.append({
            "nx": nx,
            "ny": int(rows[0]["ny"]),
            "a_lat_over_L": 1.0 / nx,
            "crack_half_over_L": float(rows[0]["crack_half_length"]) / nx,
            "passive_resistance": resistance,
            "phi_min": float(minimum["phi_first_order"]),
            "phi_max": float(maximum["phi_first_order"]),
            "phi_span": dphi,
            "phi_span_over_passive_resistance": dphi / resistance,
            "min_ratio": float(minimum["mobility_ratio"]),
            "min_theta_over_pi": float(minimum["theta_over_pi"]),
            "max_ratio": float(maximum["mobility_ratio"]),
            "max_theta_over_pi": float(maximum["theta_over_pi"]),
            "r1_angle_invariance_abs": r1_span,
            "reciprocal_rotation_symmetry_abs": reciprocal_error,
            "max_first_order_final_relative_norm": max(float(r["final_relative_norm"]) for r in rows + extra),
        })

        frows = [r for r in full_rows if int(r["nx"]) == nx]
        for ko in sorted({float(r["k_o"]) for r in frows}):
            pair_rows = [r for r in frows if math.isclose(float(r["k_o"]), ko)]
            fmin = next(r for r in pair_rows if r["extreme"] == "min_phi")
            fmax = next(r for r in pair_rows if r["extreme"] == "max_phi")
            actual = float(fmax["protocol_work"]) - float(fmin["protocol_work"])
            predicted = ko * dphi
            validation_rows.append({
                "nx": nx,
                "ny": int(fmin["ny"]),
                "a_lat_over_L": 1.0 / nx,
                "k_o": ko,
                "passive_resistance": resistance,
                "actual_delta_A_max_minus_min": actual,
                "first_order_prediction": predicted,
                "actual_over_passive_resistance": actual / resistance,
                "prediction_over_passive_resistance": predicted / resistance,
                "prediction_abs_error": abs(actual - predicted),
                "prediction_relative_error": abs(actual - predicted) / max(abs(actual), 1.0e-30),
                "abs_error_over_ko_squared": abs(actual - predicted) / (ko * ko),
                "endpoint_even_energy_difference_between_paths": abs(float(fmax["delta_even_energy"]) - float(fmin["delta_even_energy"])),
                "max_final_relative_norm": max(float(fmax["final_relative_norm"]), float(fmin["final_relative_norm"])),
            })

    write_csv(out_dir / "size_normalized_summary.csv", size_rows)
    write_csv(out_dir / "finite_ko_pair_differences.csv", validation_rows)

    x = np.array([float(r["a_lat_over_L"]) for r in size_rows])
    y = np.array([float(r["phi_span_over_passive_resistance"]) for r in size_rows])
    design = np.column_stack([np.ones_like(x), x])
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coeff
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    small = [r for r in validation_rows if abs(float(r["k_o"])) <= 0.02]
    summary = {
        "sizes": sizes,
        "geometric_family": "ny/nx=3/4, crack_half_length/nx=1/8",
        "first_order_formula": "Delta A(M1,M2)=k_o[Phi(M1)-Phi(M2)]+O(k_o^2)",
        "normalization": "passive virtual-cut resistance = -Delta W_e at k_o=0",
        "phi_span_over_passive_resistance_by_size": [
            {"nx": int(r["nx"]), "value": float(r["phi_span_over_passive_resistance"])} for r in size_rows
        ],
        "linear_in_a_lat_over_L_continuum_intercept": float(coeff[0]),
        "linear_in_a_lat_over_L_slope": float(coeff[1]),
        "linear_fit_r2": r2,
        "relative_change_32_to_largest": float(abs(y[-1] - y[0]) / abs(y[0])),
        "max_r1_angle_invariance_abs": max(float(r["r1_angle_invariance_abs"]) for r in size_rows),
        "max_reciprocal_rotation_symmetry_abs": max(float(r["reciprocal_rotation_symmetry_abs"]) for r in size_rows),
        "max_small_ko_prediction_relative_error": max(float(r["prediction_relative_error"]) for r in small),
        "max_endpoint_even_energy_path_difference": max(float(r["endpoint_even_energy_difference_between_paths"]) for r in validation_rows),
        "max_final_relative_norm": max(
            max(float(r["max_first_order_final_relative_norm"]) for r in size_rows),
            max(float(r["max_final_relative_norm"]) for r in validation_rows),
        ),
    }
    summary["pass"] = bool(
        len(sizes) >= 3
        and summary["max_r1_angle_invariance_abs"] < 1.0e-11
        and summary["max_reciprocal_rotation_symmetry_abs"] < 1.0e-10
        and summary["max_small_ko_prediction_relative_error"] < 0.12
        and summary["max_endpoint_even_energy_path_difference"] < 1.0e-10
        and summary["max_final_relative_norm"] < 2.0e-6
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
