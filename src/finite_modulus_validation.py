#!/usr/bin/env python3
"""Fourth-round IJSS revision checks.

Audits requested in the minor revision:
- exact finite-K_o apparent-flux formula in the A_o=0 sector;
- confirmation that the continuum/lattice bridge uses full finite-modulus G,
  not the first-order expansion, with an explicit truncation audit;
- all-intact-bond tensile audit of the cleavage-constrained arrest states;
- explicit q=b0+b2(k_o/k)^2 fits and component decomposition of the
  unprotected complete-period contribution.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from crack_tip_asymptotics import (
    J_matrix,
    OddModuli,
    analytic_first_order_J_derivatives,
)
from crack_tip_lattice_fit import homogenized_moduli
from crack_advance_work import bond_extension
from propagation_limit_analysis import (
    DeadLoadCascade,
    FixedGripCascade,
    crossing_frontier,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_finite_Ko_flux(out_dir: Path) -> dict[str, object]:
    """Verify G(A_o=0,K_o)=g(K_o) I over 288 K/modulus cases."""
    B_values = (0.7, 1.0, 1.4, 2.0)
    mu_values = (0.25, 0.4, 0.6)
    Ko_values = (-0.8, -0.3, -0.08, 0.0, 0.17, 0.65)
    K_pairs = ((1.0, 0.0), (0.0, 1.0), (0.73, -0.28), (-0.41, 0.82))
    rows: list[dict[str, object]] = []
    matrix_errors: list[float] = []
    flux_errors: list[float] = []
    for B in B_values:
        for mu in mu_values:
            for Ko in Ko_values:
                G_num = J_matrix(OddModuli(B=B, mu=mu, A_o=0.0, K_o=Ko), n_theta=401)
                scalar = 1.0 / (4.0 * B) + mu / (4.0 * (mu * mu + Ko * Ko))
                matrix_error = float(np.max(np.abs(G_num - scalar * np.eye(2))))
                matrix_errors.append(matrix_error)
                for KI, KII in K_pairs:
                    numeric = float(np.array([KI, KII]) @ G_num @ np.array([KI, KII]))
                    exact = (KI * KI + KII * KII) * scalar
                    rel = abs(numeric - exact) / max(abs(exact), 1.0e-30)
                    flux_errors.append(rel)
                    rows.append(
                        {
                            "B": B,
                            "mu": mu,
                            "K_o": Ko,
                            "K_I": KI,
                            "K_II": KII,
                            "J_numeric": numeric,
                            "J_exact": exact,
                            "relative_error": rel,
                            "G_offdiag_numeric": float(G_num[0, 1]),
                            "G_diagonal_difference": float(G_num[0, 0] - G_num[1, 1]),
                        }
                    )
    write_csv(out_dir / "exact_finite_Ko_flux_288_cases.csv", rows)
    summary = {
        "exact_formula": "J=(K_I^2+K_II^2)[1/(4B)+mu/(4(mu^2+K_o^2))] for A_o=0",
        "number_of_flux_cases": len(rows),
        "number_of_G_evaluations": len(B_values) * len(mu_values) * len(Ko_values),
        "maximum_G_matrix_absolute_error": max(matrix_errors),
        "maximum_flux_relative_error": max(flux_errors),
        "consequence": (
            "For A_o=0, G is exactly a scalar multiple of the identity at every K_o; "
            "K_o changes the deviatoric compliance but cannot couple Modes I and II."
        ),
    }
    summary["pass"] = bool(
        summary["number_of_flux_cases"] == 288
        and summary["maximum_G_matrix_absolute_error"] < 2.0e-12
        and summary["maximum_flux_relative_error"] < 2.0e-12
    )
    (out_dir / "exact_finite_Ko_flux_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def finite_modulus_bridge_audit(out_dir: Path) -> dict[str, object]:
    """Show that the archived bridge used full G and quantify truncation error."""
    fits = pd.read_csv(
        ROOT / "data" / "crack_tip_lattice_fit_results" / "stress_fit_annulus_scan.csv"
    )
    fits = fits[(fits.tip == "right") & (fits.basis == "matched_odd")].copy()
    jdata = pd.read_csv(
        ROOT / "data" / "discrete_configurational_results" / "discrete_domain_radius_scan.csv"
    )
    measured = (
        jdata.groupby("k_o", as_index=False).J_total.mean().rename(columns={"J_total": "J_h"})
    )
    J_pass = float(measured[np.isclose(measured.k_o, 0.0)].J_h.iloc[0])

    m0 = homogenized_moduli(1.0, 0.0)
    G0 = J_matrix(m0, n_theta=1201, energy_choice="micro_hessian")
    K0 = fits[np.isclose(fits.global_k_o, 0.0)][["K_I", "K_II"]].to_numpy(float)
    q0 = float(np.mean(np.einsum("ni,ij,nj->n", K0, G0, K0)))
    alpha = J_pass / q0

    rows: list[dict[str, object]] = []
    for ko, group in fits.groupby("global_k_o"):
        if ko < -1.0e-14:
            continue
        ko = float(ko)
        K = group[["K_I", "K_II"]].to_numpy(float)
        moduli = homogenized_moduli(1.0, ko)
        dGdA, _ = analytic_first_order_J_derivatives(moduli.B, moduli.mu)
        matrices = {
            "full_microenergetic": J_matrix(moduli, 1201, "micro_hessian"),
            "full_major_symmetric": J_matrix(moduli, 1201, "major_symmetric_projection"),
            "first_order_microenergetic": G0 + moduli.A_o * dGdA,
        }
        Ko_scalar = 1.0 / (4.0 * moduli.B) + moduli.mu / (
            4.0 * (moduli.mu * moduli.mu + moduli.K_o * moduli.K_o)
        )
        matrices["exact_Ao0_Ko_only"] = Ko_scalar * np.eye(2)
        measured_inc = float(measured[np.isclose(measured.k_o, ko)].J_h.iloc[0] / J_pass - 1.0)
        for name, G in matrices.items():
            q = float(np.mean(np.einsum("ni,ij,nj->n", K, G, K)))
            pred_inc = alpha * q / J_pass - 1.0
            rows.append(
                {
                    "k_o_over_k": ko,
                    "prediction": name,
                    "passive_calibration_scalar": alpha,
                    "measured_increment": measured_inc,
                    "predicted_increment": pred_inc,
                    "prediction_minus_measurement": pred_inc - measured_inc,
                }
            )
    write_csv(out_dir / "finite_modulus_bridge_truncation_audit.csv", rows)

    g20 = fits[np.isclose(fits.global_k_o, 0.20)].copy()
    ratios = (g20.K_II / g20.K_I).to_numpy(float)

    def value(name: str, ko: float = 0.20) -> float:
        r = next(
            row
            for row in rows
            if row["prediction"] == name and math.isclose(float(row["k_o_over_k"]), ko)
        )
        return float(r["predicted_increment"])

    full_micro = value("full_microenergetic")
    first_micro = value("first_order_microenergetic")
    full_ms = value("full_major_symmetric")
    exact_ko = value("exact_Ao0_Ko_only")
    measured20 = next(
        float(row["measured_increment"])
        for row in rows
        if row["prediction"] == "full_microenergetic"
        and math.isclose(float(row["k_o_over_k"]), 0.20)
    )
    passive_scalar = float(G0[0, 0])
    m20 = homogenized_moduli(1.0, 0.20)
    Ko_scalar20 = 1.0 / (4.0 * m20.B) + m20.mu / (
        4.0 * (m20.mu * m20.mu + m20.K_o * m20.K_o)
    )
    summary = {
        "bridge_uses": "full finite-modulus numerical G(A_o,K_o) for both energy gauges",
        "passive_calibration_scalar": alpha,
        "measured_increment_ko0p20": measured20,
        "full_microenergetic_increment_ko0p20": full_micro,
        "full_major_symmetric_increment_ko0p20": full_ms,
        "first_order_microenergetic_increment_ko0p20": first_micro,
        "first_order_minus_full_microenergetic_percentage_points": 100.0 * (first_micro - full_micro),
        "full_gauge_separation_percentage_points": 100.0 * (full_micro - full_ms),
        "exact_Ko_only_G_relative_change_at_ko0p20": Ko_scalar20 / passive_scalar - 1.0,
        "exact_Ko_only_prediction_with_fitted_K_increment": exact_ko,
        "KII_mean_ko0p20": float(g20.K_II.mean()),
        "KII_annulus_sd_ko0p20": float(g20.K_II.std(ddof=1)),
        "KII_over_KI_mean_ko0p20": float(ratios.mean()),
        "KII_over_KI_annulus_sd_ko0p20": float(ratios.std(ddof=1)),
        "KII_over_KI_range_ko0p20": [float(ratios.min()), float(ratios.max())],
        "KII_sign_resolved": bool(np.all(ratios > 0.0) or np.all(ratios < 0.0)),
        "calibration_assumption": (
            "The passive MLS amplitude factor is assumed to apply equally to the even and odd active increments."
        ),
        "interpretation": (
            "The 9.35% bridge value was not obtained from the first-order expansion. The full finite-modulus "
            "calculation already contains the exact K_o^2 compliance correction and all numerical A_o^2/A_oK_o "
            "terms. Subtracting the isolated K_o^2 term from it would double count that contribution."
        ),
    }
    summary["pass"] = bool(
        abs(summary["first_order_minus_full_microenergetic_percentage_points"]) < 0.5
        and summary["full_gauge_separation_percentage_points"] > 3.0
        and summary["KII_sign_resolved"] is False
    )
    (out_dir / "finite_modulus_bridge_truncation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _audit_one_arrest_state(row: dict[str, object]) -> dict[str, object]:
    nx = int(row["nx"])
    ny = int(row["ny"])
    target = float(row["target_half_length"])
    ko = float(row["k_o"])
    bc = str(row["boundary_condition"])
    amplitude = float(row["applied_amplitude"])
    broken = int(row["broken_bonds"])
    delta_c = 0.02
    cascade = (
        FixedGripCascade(nx, ny, target, ko)
        if bc == "fixed_grip"
        else DeadLoadCascade(nx, ny, target, ko)
    )
    if broken:
        frontier0 = crossing_frontier(cascade.model, cascade.removed)
        side = str(row["first_break_side"])
        cascade.removed.add(frontier0[side])
    displacement = cascade.solve(amplitude)[0]
    frontier = crossing_frontier(cascade.model, cascade.removed)
    frontier_ids = set(frontier.values())
    values: list[tuple[float, int, object]] = []
    for bond_id, bond in enumerate(cascade.model.all_bonds):
        if bond_id in cascade.removed:
            continue
        values.append((bond_extension(bond, displacement) / delta_c, bond_id, bond))
    values.sort(key=lambda item: item[0], reverse=True)
    max_all, max_all_id, max_all_bond = values[0]
    nonfront = [item for item in values if item[1] not in frontier_ids]
    max_non, max_non_id, max_non_bond = nonfront[0]
    max_front = max((item[0] for item in values if item[1] in frontier_ids), default=float("nan"))
    return {
        **row,
        "max_all_intact_ratio": max_all,
        "max_frontier_ratio_recomputed": max_front,
        "max_nonfrontier_ratio": max_non,
        "max_all_bond_id": max_all_id,
        "max_all_crosses_crack_plane": bool(max_all_bond.crosses_crack_plane),
        "max_all_bond_nx": float(max_all_bond.n[0]),
        "max_all_bond_ny": float(max_all_bond.n[1]),
        "max_nonfrontier_bond_id": max_non_id,
        "max_nonfrontier_crosses_crack_plane": bool(max_non_bond.crosses_crack_plane),
        "max_nonfrontier_bond_nx": float(max_non_bond.n[0]),
        "max_nonfrontier_bond_ny": float(max_non_bond.n[1]),
    }


def all_bond_arrest_audit(out_dir: Path) -> dict[str, object]:
    cascade = pd.read_csv(
        ROOT / "data" / "advance_resistance_results" / "cascade_robustness_scan.csv"
    )
    records = cascade.to_dict("records")
    workers = min(8, os.cpu_count() or 2)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        audited = list(pool.map(_audit_one_arrest_state, records, chunksize=2))
    write_csv(out_dir / "all_bond_arrest_audit.csv", audited)
    df = pd.DataFrame(audited)
    over = df.max_nonfrontier_ratio >= 1.0 - 1.0e-10
    p09 = np.isclose(df.load_fraction, 0.90)
    summary = {
        "number_of_cleavage_constrained_arrest_states": len(df),
        "maximum_all_intact_bond_ratio": float(df.max_all_intact_ratio.max()),
        "maximum_nonfrontier_bond_ratio": float(df.max_nonfrontier_ratio.max()),
        "states_with_nonfrontier_bond_at_or_above_threshold": int(over.sum()),
        "states_with_nonfrontier_bond_at_or_above_threshold_p0p90": int((over & p09).sum()),
        "maximum_nonfrontier_ratio_p0p90": float(df.loc[p09, "max_nonfrontier_ratio"].max()),
        "all_threshold_exceedances_occur_after_one_cleavage_break": bool(
            np.all(df.loc[over, "broken_bonds"].to_numpy(int) == 1)
        ),
        "all_threshold_exceedances_are_off_cleavage_plane": bool(
            np.all(~df.loc[over, "max_nonfrontier_crosses_crack_plane"].to_numpy(bool))
        ),
        "threshold_exceedance_orientation": sorted(
            {
                (round(float(r.max_nonfrontier_bond_nx), 6), round(float(r.max_nonfrontier_bond_ny), 6))
                for r in df.loc[over].itertuples(index=False)
            }
        ),
        "conclusion": (
            "The previous one-bond arrest statement is valid only for the prescribed cleavage-front candidate set. "
            "An unrestricted tensile-bond rule would admit off-plane branching candidates in part of the scan."
        ),
    }
    summary["pass"] = bool(
        summary["number_of_cleavage_constrained_arrest_states"] == 224
        and summary["states_with_nonfrontier_bond_at_or_above_threshold"] > 0
        and summary["all_threshold_exceedances_occur_after_one_cleavage_break"]
        and summary["all_threshold_exceedances_are_off_cleavage_plane"]
    )
    (out_dir / "all_bond_arrest_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def period_even_component_audit(out_dir: Path) -> dict[str, object]:
    steps = pd.read_csv(
        ROOT / "data" / "advance_resistance_results" / "advance_scaled_steps.csv"
    )
    steps = steps[np.isclose(steps.load_fraction, 0.90)].copy()
    fit_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    group_cols = ["nx", "ny", "crack_fraction", "boundary_condition", "period"]
    for key, group in steps.groupby(group_cols):
        sums = (
            group.groupby("k_o", as_index=False)
            .agg(
                available_work=("available_work", "sum"),
                effective_resistance=("effective_resistance", "sum"),
                cut_energy=("cut_energy", "sum"),
                viscous_dissipation=("viscous_dissipation", "sum"),
                odd_work=("odd_work", "sum"),
                external_work=("external_work", "sum"),
                even_energy_change=("even_energy_change", "sum"),
            )
            .sort_values("k_o")
        )
        x = sums.k_o.to_numpy(float) ** 2
        resistance = float(sums.effective_resistance.iloc[0])
        ratios = {
            "total_ratio": sums.available_work.to_numpy(float) / sums.effective_resistance.to_numpy(float),
            "cut_energy_over_R": sums.cut_energy.to_numpy(float) / sums.effective_resistance.to_numpy(float),
            "dissipation_over_R": sums.viscous_dissipation.to_numpy(float) / sums.effective_resistance.to_numpy(float),
            "odd_work_over_R": sums.odd_work.to_numpy(float) / sums.effective_resistance.to_numpy(float),
        }
        coefficients: dict[str, tuple[float, float, float]] = {}
        for name, y in ratios.items():
            X = np.column_stack((np.ones_like(x), x))
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ coef
            coefficients[name] = (float(coef[0]), float(coef[1]), float(np.max(np.abs(resid))))
        nx, ny, frac, bc, period = key
        fit_rows.append(
            {
                "nx": int(nx),
                "ny": int(ny),
                "crack_fraction": float(frac),
                "boundary_condition": bc,
                "period": int(period),
                "fit_form": "q=b0+b2*(k_o/k)^2",
                "b0": coefficients["total_ratio"][0],
                "b2": coefficients["total_ratio"][1],
                "maximum_abs_fit_residual": coefficients["total_ratio"][2],
            }
        )
        component_rows.append(
            {
                "nx": int(nx),
                "ny": int(ny),
                "crack_fraction": float(frac),
                "boundary_condition": bc,
                "period": int(period),
                "resistance": resistance,
                "total_b2": coefficients["total_ratio"][1],
                "cut_energy_b2": coefficients["cut_energy_over_R"][1],
                "dissipation_b2": coefficients["dissipation_over_R"][1],
                "odd_work_b2": coefficients["odd_work_over_R"][1],
            }
        )
    write_csv(out_dir / "period_ratio_linear_in_ko_squared_fits.csv", fit_rows)
    write_csv(out_dir / "period_ratio_component_slopes.csv", component_rows)
    fit_df = pd.DataFrame(fit_rows)
    comp_df = pd.DataFrame(component_rows)
    upper = fit_df.loc[fit_df.b0.idxmax()]
    upper_comp = comp_df[
        (comp_df.nx == upper.nx)
        & (comp_df.ny == upper.ny)
        & np.isclose(comp_df.crack_fraction, upper.crack_fraction)
        & (comp_df.boundary_condition == upper.boundary_condition)
        & (comp_df.period == upper.period)
    ].iloc[0]
    summary = {
        "fit_form": "q(k_o)=b0+b2*(k_o/k)^2",
        "number_of_fits": len(fit_df),
        "number_of_positive_b2": int((fit_df.b2 > 0.0).sum()),
        "b2_range": [float(fit_df.b2.min()), float(fit_df.b2.max())],
        "maximum_fit_residual": float(fit_df.maximum_abs_fit_residual.max()),
        "upper_envelope_case": {
            "nx": int(upper.nx),
            "crack_fraction": float(upper.crack_fraction),
            "boundary_condition": str(upper.boundary_condition),
            "period": int(upper.period),
            "b0": float(upper.b0),
            "b2": float(upper.b2),
            "cut_energy_b2": float(upper_comp.cut_energy_b2),
            "dissipation_b2": float(upper_comp.dissipation_b2),
            "odd_work_b2": float(upper_comp.odd_work_b2),
        },
        "sign_status": (
            "The nonpositive b2 values are an empirical result for this lattice, loading range and cleavage rule. "
            "D_eta>=0 does not determine the sign of its k_o^2 derivative."
        ),
        "traction_period2_explanation": (
            "For the upper dead-load period-2 curve, the negative b2 is dominated by a decrease of the pre-cut "
            "bond energy; the dissipation term is nearly flat and the positive odd-work contribution is insufficient "
            "to offset the reduced candidate-bond extension."
        ),
    }
    summary["pass"] = bool(
        summary["number_of_fits"] == 28
        and summary["number_of_positive_b2"] == 0
        and summary["maximum_fit_residual"] < 1.0e-3
        and summary["upper_envelope_case"]["cut_energy_b2"] < 0.0
    )
    (out_dir / "period_even_component_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_all(out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        "exact_finite_Ko_flux": exact_finite_Ko_flux(out_dir),
        "finite_modulus_bridge": finite_modulus_bridge_audit(out_dir),
        "all_bond_arrest": all_bond_arrest_audit(out_dir),
        "period_even_component": period_even_component_audit(out_dir),
    }
    summaries["pass"] = all(bool(v.get("pass", False)) for v in summaries.values())
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


if __name__ == "__main__":
    result = run_all(ROOT / "data" / "finite_modulus_validation_results")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit("Fourth-round checks failed")
