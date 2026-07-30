#!/usr/bin/env python3
"""Additional checks introduced during the major-comment revision.

The checks address: irreducible-sign conventions, lattice-to-continuum field
refinement, mobility-dependent stability, passive lattice-trapping baselines,
and uncertainty of the localization-gauge refinement exponent.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.sparse import block_diag
from scipy.sparse.linalg import eigs, eigsh
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from crack_tip_asymptotics import (
    OddModuli,
    stiffness_matrix,
    tensor_to_irreducible_stress,
)
from crack_tip_lattice_fit import analyze_case
from same_endpoint_scaling import ScalingConfig, mobility_matrix, prepare


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def convention_audit() -> dict[str, object]:
    rng = np.random.default_rng(20260726)
    moduli = OddModuli(B=1.3, mu=0.61, A_o=-0.22, K_o=-0.09)
    C = stiffness_matrix(moduli)
    pairing_error = 0.0
    inverse_error = 0.0
    pde_error = 0.0
    for _ in range(50):
        H = rng.normal(size=(2, 2))
        sigma = (C @ H.reshape(-1)).reshape(2, 2)
        e = np.array(
            [
                H[0, 0] + H[1, 1],
                H[1, 0] - H[0, 1],
                H[0, 0] - H[1, 1],
                H[0, 1] + H[1, 0],
            ]
        )
        s = tensor_to_irreducible_stress(sigma)
        pairing_error = max(pairing_error, abs(float(sigma.ravel() @ H.ravel() - s @ e)))
        reconstructed = np.array(
            [[s[0] + s[2], s[3] - s[1]], [s[3] + s[1], s[0] - s[2]]]
        )
        inverse_error = max(inverse_error, float(np.max(np.abs(reconstructed - sigma))))

        q = rng.normal(size=2)
        u = rng.normal(size=2)
        Hq = np.outer(u, q)
        sigmaq = (C @ Hq.reshape(-1)).reshape(2, 2)
        force = sigmaq @ q
        div_force = float(q @ force)
        curl_force = float(q[0] * force[1] - q[1] * force[0])
        d = float(q @ u)
        c = float(q[0] * u[1] - q[1] * u[0])
        q2 = float(q @ q)
        div_pred = q2 * ((moduli.B + moduli.mu) * d - moduli.K_o * c)
        curl_pred = q2 * ((moduli.K_o + moduli.A_o) * d + moduli.mu * c)
        pde_error = max(pde_error, abs(div_force - div_pred), abs(curl_force - curl_pred))

    k, ko = 1.0, 0.37
    B = math.sqrt(3.0) * k / 2.0
    mu = math.sqrt(3.0) * k / 4.0
    Ao = -math.sqrt(3.0) * ko / 2.0
    Ko = -math.sqrt(3.0) * ko / 4.0
    determinant = mu * (B + mu) + Ko * (Ko + Ao)
    determinant_exact = 9.0 / 16.0 * (k * k + ko * ko)
    return {
        "pairing": "sigma:H=sum_alpha sigma_alpha e_alpha",
        "sigma_1_definition": "(sigma_yx-sigma_xy)/2",
        "pde_equations": [
            "(B+mu) Laplacian(d)-K_o Laplacian(c)=0",
            "(K_o+A_o) Laplacian(d)+mu Laplacian(c)=0",
        ],
        "max_pairing_error": pairing_error,
        "max_inverse_error": inverse_error,
        "max_fourier_pde_error": pde_error,
        "lattice_ray_determinant": determinant,
        "lattice_ray_determinant_exact": determinant_exact,
        "pass": bool(
            pairing_error < 1.0e-12
            and inverse_error < 1.0e-12
            and pde_error < 1.0e-11
            and abs(determinant - determinant_exact) < 1.0e-14
        ),
    }


def crack_tip_refinement(out_dir: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for nx in (32, 48, 64, 80, 96):
        ny = 3 * nx // 4
        crack = nx / 8.0
        r_inner = 0.05 * nx
        r_outer = 0.09 * nx
        radii = np.linspace(r_inner, r_outer, 7)
        angles = np.linspace(-math.pi + 0.10, math.pi - 0.10, 73)
        fit_radius = max(2.0, 0.05 * nx)
        for ko in (0.0, 0.15):
            fit_rows, metadata, *_ = analyze_case(
                nx=nx,
                ny=ny,
                crack_half_length=crack,
                k_o=ko,
                tip="right",
                radii=radii,
                angles=angles,
                fit_radius=fit_radius,
                annuli=((r_inner, r_outer),),
            )
            matched = next(row for row in fit_rows if row["basis"] == "matched_odd")
            passive_basis = next(row for row in fit_rows if row["basis"] == "passive")
            rows.append(
                {
                    "nx": nx,
                    "ny": ny,
                    "a_lat_over_L": 1.0 / nx,
                    "crack_half_over_L": crack / nx,
                    "r_inner_over_L": r_inner / nx,
                    "r_outer_over_L": r_outer / nx,
                    "k_o": ko,
                    "matched_relative_residual": matched["relative_L2_residual"],
                    "passive_basis_relative_residual": passive_basis["relative_L2_residual"],
                    "wrong_basis_penalty": passive_basis["relative_L2_residual"]
                    - matched["relative_L2_residual"],
                    "equilibrium_residual_inf": metadata["equilibrium_residual_inf"],
                }
            )
    write_csv(out_dir / "crack_tip_refinement.csv", rows)
    active = [r for r in rows if math.isclose(float(r["k_o"]), 0.15)]
    passive = [r for r in rows if math.isclose(float(r["k_o"]), 0.0)]
    summary = {
        "fixed_annulus_R_over_L": [0.05, 0.09],
        "sizes": [int(r["nx"]) for r in active],
        "active_matched_residual_32": float(active[0]["matched_relative_residual"]),
        "active_matched_residual_96": float(active[-1]["matched_relative_residual"]),
        "active_wrong_basis_residual_32": float(active[0]["passive_basis_relative_residual"]),
        "active_wrong_basis_residual_96": float(active[-1]["passive_basis_relative_residual"]),
        "passive_residual_32": float(passive[0]["matched_relative_residual"]),
        "passive_residual_96": float(passive[-1]["matched_relative_residual"]),
        "active_matched_relative_reduction": 1.0
        - float(active[-1]["matched_relative_residual"])
        / float(active[0]["matched_relative_residual"]),
        "passive_relative_reduction": 1.0
        - float(passive[-1]["matched_relative_residual"])
        / float(passive[0]["matched_relative_residual"]),
        "minimum_wrong_basis_penalty": min(float(r["wrong_basis_penalty"]) for r in active),
        "max_equilibrium_residual": max(float(r["equilibrium_residual_inf"]) for r in rows),
    }
    summary["pass"] = bool(
        summary["active_matched_residual_96"] < summary["active_matched_residual_32"]
        and summary["passive_residual_96"] < summary["passive_residual_32"]
        and summary["minimum_wrong_basis_penalty"] > 0.025
        and summary["max_equilibrium_residual"] < 1.0e-11
    )
    (out_dir / "crack_tip_refinement_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def mobility_sqrt_matrix(ndof: int, ratio: float, theta: float):
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    diagonal = np.diag([ratio ** 0.25, ratio ** -0.25])
    block = rotation @ diagonal @ rotation.T
    return block_diag([block] * (ndof // 2), format="csr")


def _stability_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for ko in (0.0, 0.12, 0.222271):
        for ratio in (0.5, 1.0, 2.0, 4.0):
            cases.append(
                dict(source="main_same_endpoint", nx=48, ny=36, crack=6.0, k_o=ko, ratio=ratio, theta=0.0)
            )
    small = pd.read_csv(ROOT / "data" / "same_endpoint_scaling_results" / "finite_ko_validation.csv")
    for row in small[["k_o", "mobility_ratio", "theta"]].drop_duplicates().itertuples(index=False):
        cases.append(
            dict(source="small_ko_validation", nx=32, ny=24, crack=5.0, k_o=float(row.k_o), ratio=float(row.mobility_ratio), theta=float(row.theta))
        )
    size = pd.read_csv(ROOT / "data" / "same_endpoint_size_scaling_results" / "finite_ko_extreme_validation.csv")
    for row in size[["nx", "ny", "crack_half_length", "k_o", "mobility_ratio", "theta"]].drop_duplicates().itertuples(index=False):
        cases.append(
            dict(source="size_extreme_validation", nx=int(row.nx), ny=int(row.ny), crack=float(row.crack_half_length), k_o=float(row.k_o), ratio=float(row.mobility_ratio), theta=float(row.theta))
        )
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for case in cases:
        key = (case["nx"], case["ny"], case["crack"], case["k_o"], case["ratio"], round(float(case["theta"]), 14))
        if key in unique:
            unique[key]["source"] = str(unique[key]["source"]) + ";" + str(case["source"])
        else:
            unique[key] = case
    return list(unique.values())


def stability_spectrum(out_dir: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    prepared: dict[tuple[int, int, float, float], tuple[object, object, np.ndarray]] = {}
    for case in _stability_cases():
        key = (int(case["nx"]), int(case["ny"]), float(case["crack"]), float(case["k_o"]))
        if key not in prepared:
            config = ScalingConfig(nx=key[0], ny=key[1], crack_half_length=key[2], t_end=1.0)
            model, _Ee_old, _Ee_new, _Eo_new, Knew, _u0, _u1, _c, _cv, free = prepare(config, key[3])
            prepared[key] = (model, Knew, free)
        model, Knew, free = prepared[key]
        ratio = float(case["ratio"])
        theta = float(case["theta"])
        Kff = Knew[free][:, free]
        Msqrt = mobility_sqrt_matrix(model.ndof, ratio, theta)[free][:, free]
        transformed = Msqrt @ Kff @ Msqrt
        hermitian = 0.5 * (transformed + transformed.T)
        min_hermitian = float(
            eigsh(hermitian, k=1, which="SA", return_eigenvectors=False, tol=2.0e-8, maxiter=20000)[0]
        )
        soft = eigs(
            transformed,
            k=6,
            which="SR",
            return_eigenvectors=False,
            tol=2.0e-8,
            maxiter=20000,
        )
        rows.append(
            {
                **case,
                "ndof_free": len(free),
                "minimum_hermitian_part_eigenvalue": min_hermitian,
                "minimum_soft_mode_real_part": float(np.min(np.real(soft))),
                "maximum_soft_mode_abs_imaginary_part": float(np.max(np.abs(np.imag(soft)))),
            }
        )
    write_csv(out_dir / "mobility_stability_spectrum.csv", rows)
    summary = {
        "number_of_unique_tested_combinations": len(rows),
        "minimum_hermitian_part_eigenvalue": min(float(r["minimum_hermitian_part_eigenvalue"]) for r in rows),
        "minimum_soft_mode_real_part": min(float(r["minimum_soft_mode_real_part"]) for r in rows),
        "maximum_soft_mode_abs_imaginary_part": max(float(r["maximum_soft_mode_abs_imaginary_part"]) for r in rows),
        "interpretation": (
            "All tested post-cut operators are strictly dissipative in the M^{-1}-weighted norm. "
            "Some higher modes are complex, so componentwise monotonicity is not asserted."
        ),
    }
    summary["pass"] = bool(
        summary["minimum_hermitian_part_eigenvalue"] > 0.0
        and summary["minimum_soft_mode_real_part"] > 0.0
    )
    (out_dir / "mobility_stability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def passive_step_comparison(out_dir: Path) -> dict[str, object]:
    steps = pd.read_csv(ROOT / "data" / "advance_resistance_results" / "advance_scaled_steps.csv")
    selected = steps[
        (steps.nx == 48)
        & np.isclose(steps.crack_fraction, 0.125)
        & (steps.boundary_condition == "fixed_grip")
        & np.isclose(steps.load_fraction, 0.90)
        & (np.isclose(steps.k_o, 0.0) | np.isclose(steps.k_o, 0.20))
    ].sort_values(["k_o", "step"])
    cols = [
        "k_o",
        "step",
        "work_ratio",
        "extension_ratio",
        "odd_work",
        "effective_resistance",
        "stability_margin",
    ]
    selected[cols].to_csv(out_dir / "passive_active_step_comparison.csv", index=False)
    passive = selected[np.isclose(selected.k_o, 0.0)].sort_values("step")
    active = selected[np.isclose(selected.k_o, 0.20)].sort_values("step")
    passive_extension_span = float(passive.extension_ratio.max() - passive.extension_ratio.min())
    active_extension_span = float(active.extension_ratio.max() - active.extension_ratio.min())
    summary = {
        "passive_extension_ratios": [float(x) for x in passive.extension_ratio],
        "active_extension_ratios": [float(x) for x in active.extension_ratio],
        "passive_work_ratios": [float(x) for x in passive.work_ratio],
        "active_work_ratios": [float(x) for x in active.work_ratio],
        "passive_extension_periodic_span": passive_extension_span,
        "active_extension_periodic_span": active_extension_span,
        "active_to_passive_extension_span_ratio": active_extension_span / passive_extension_span,
        "interpretation": (
            "A weak passive registry/lattice-trapping oscillation is present. Odd work superposes a much larger "
            "+,-,+,- modulation and produces the one-step accessibility pattern."
        ),
        "pass": bool(passive_extension_span > 0.0 and active_extension_span > 5.0 * passive_extension_span),
    }
    (out_dir / "passive_active_step_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def gauge_exponent_uncertainty(out_dir: Path) -> dict[str, object]:
    data = json.loads(
        (ROOT / "data" / "gauge_convergence_results" / "summary.json").read_text(encoding="utf-8")
    )
    x = np.array([float(r["a_lat_over_L"]) for r in data["target_active_relative_spans"]])
    y = np.array([float(r["span"]) for r in data["target_active_relative_spans"]])
    lx, ly = np.log(x), np.log(y)
    design = np.column_stack([np.ones_like(lx), lx])
    coefficient, *_ = np.linalg.lstsq(design, ly, rcond=None)
    residual = ly - design @ coefficient
    dof = len(x) - 2
    variance = float(residual @ residual / dof)
    covariance = variance * np.linalg.inv(design.T @ design)
    slope_se = math.sqrt(float(covariance[1, 1]))
    critical = float(student_t.ppf(0.975, dof))
    interval = [float(coefficient[1] - critical * slope_se), float(coefficient[1] + critical * slope_se)]
    rows = [
        {
            "a_lat_over_L": float(xi),
            "relative_gauge_span": float(yi),
            "log_fit_value": float(math.exp(coefficient[0]) * xi ** coefficient[1]),
        }
        for xi, yi in zip(x, y)
    ]
    write_csv(out_dir / "gauge_exponent_fit_with_uncertainty.csv", rows)
    summary = {
        "exponent": float(coefficient[1]),
        "standard_error": slope_se,
        "confidence_level": 0.95,
        "confidence_interval": interval,
        "degrees_of_freedom": dof,
        "heuristic": (
            "Changing the adjacent-cell share is a bounded discrete superpotential. "
            "Summation by parts against a smooth domain weight adds one discrete derivative, "
            "giving an expected O(a_lat/L) relative ambiguity away from the process zone."
        ),
        "pass": bool(interval[0] < 1.0 < interval[1]),
    }
    (out_dir / "gauge_exponent_uncertainty_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_all(out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        "convention_audit": convention_audit(),
        "crack_tip_refinement": crack_tip_refinement(out_dir),
        "mobility_stability": stability_spectrum(out_dir),
        "passive_step_comparison": passive_step_comparison(out_dir),
        "gauge_exponent_uncertainty": gauge_exponent_uncertainty(out_dir),
    }
    summaries["pass"] = all(bool(v.get("pass", False)) for v in summaries.values() if isinstance(v, dict))
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


if __name__ == "__main__":
    result = run_all(ROOT / "data" / "refinement_validation_results")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit("Revision checks failed")
