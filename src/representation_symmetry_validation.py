#!/usr/bin/env python3
"""Third-round IJSS revision checks.

Audits requested in the third review:
- closed-form cancellation of the first-order A_o flux under the nonobjective
  major-symmetric energy split;
- increment-normalized continuum/lattice J bridge with one actual passive
  calibration scalar and K_II uncertainty across fitting annuli;
- O(k_o^2) unprotected period-sum contribution under glide symmetry;
- absence of demonstrated convergence of the directional bias chi_A;
- a constructive comparison of state-function, operational-work and local
  tensile criteria under fixed grip.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from active_tip_scan import ActiveCrackedStrip
from crack_advance_work import assemble_components, even_energy, solve_equilibrium
from crack_tip_asymptotics import J_matrix
from crack_tip_lattice_fit import homogenized_moduli


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def closed_form_energy_gauge_cancellation(out_dir: Path) -> dict[str, object]:
    B, mu = 1.0, 0.5
    KI, KII = 0.73, -0.28
    kappa = 1.0 + 2.0 * mu / B
    micro_matrix_cross = (B + mu) / (8.0 * B * B * mu)
    # K^T G K contains twice the off-diagonal matrix entry.
    micro_flux_cross_coefficient = 2.0 * micro_matrix_cross
    delta_flux_cross_coefficient = -(kappa * kappa - 1.0) / (16.0 * mu * mu)
    identity_form = -(B + mu) / (4.0 * B * B * mu)

    # Independent angular quadrature of the Muskhelishvili expression.
    n = 500001
    theta = np.linspace(-math.pi, math.pi, n)
    r = 1.0
    phi_sq = (KI - 1j * KII) ** 2 / (8.0 * math.pi * r * np.exp(1j * theta))
    angular_integral = float(np.trapz(np.imag(phi_sq) * np.cos(theta) * r, theta))
    expected_integral = -KI * KII / 4.0

    rows = [
        {
            "B": B,
            "mu": mu,
            "kappa": kappa,
            "K_I": KI,
            "K_II": KII,
            "angular_integral_numeric": angular_integral,
            "angular_integral_exact": expected_integral,
            "micro_flux_cross_coefficient": micro_flux_cross_coefficient,
            "gauge_increment_cross_coefficient": delta_flux_cross_coefficient,
            "gauge_increment_identity_form": identity_form,
            "net_first_order_cross_coefficient": micro_flux_cross_coefficient + delta_flux_cross_coefficient,
        }
    ]
    write_csv(out_dir / "closed_form_energy_gauge_cancellation.csv", rows)
    summary = {
        "muskhelishvili_identity": "d+i c=(kappa phi'-conj(phi'))/mu, kappa=1+2mu/B",
        "angular_integral_numeric": angular_integral,
        "angular_integral_exact": expected_integral,
        "angular_integral_abs_error": abs(angular_integral - expected_integral),
        "microenergetic_first_order_flux_coefficient": micro_flux_cross_coefficient,
        "major_symmetric_gauge_increment_coefficient": delta_flux_cross_coefficient,
        "major_symmetric_gauge_increment_identity_form": identity_form,
        "net_major_symmetric_first_order_coefficient": micro_flux_cross_coefficient + delta_flux_cross_coefficient,
        "conclusion": (
            "The nonobjective major-symmetric energy split cancels the complete O(A_o) K_I K_II term exactly; "
            "G_ms=I/E_2D+O(A_o^2,A_o K_o,K_o^2)."
        ),
    }
    summary["pass"] = bool(
        summary["angular_integral_abs_error"] < 2.0e-10
        and abs(delta_flux_cross_coefficient - identity_form) < 1.0e-14
        and abs(summary["net_major_symmetric_first_order_coefficient"]) < 1.0e-14
    )
    (out_dir / "closed_form_energy_gauge_cancellation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def continuum_lattice_increment_bridge(out_dir: Path) -> dict[str, object]:
    fits = pd.read_csv(ROOT / "data" / "crack_tip_lattice_fit_results" / "stress_fit_annulus_scan.csv")
    fits = fits[(fits.tip == "right") & (fits.basis == "matched_odd")].copy()
    jdata = pd.read_csv(ROOT / "data" / "discrete_configurational_results" / "discrete_domain_radius_scan.csv")
    measured = jdata.groupby("k_o", as_index=False).J_total.mean().rename(columns={"J_total": "J_h_radius_mean"})
    passive_measured = float(measured[np.isclose(measured.k_o, 0.0)].J_h_radius_mean.iloc[0])

    q_by_choice: dict[str, dict[float, np.ndarray]] = {}
    for choice in ("micro_hessian", "major_symmetric_projection"):
        q_by_choice[choice] = {}
        for ko, group in fits.groupby("global_k_o"):
            G = J_matrix(homogenized_moduli(1.0, float(ko)), n_theta=1201, energy_choice=choice)
            vals = []
            for row in group.itertuples(index=False):
                K = np.array([float(row.K_I), float(row.K_II)])
                vals.append(float(K @ G @ K))
            q_by_choice[choice][float(ko)] = np.asarray(vals)

    # One actual scalar, common to every annulus and both gauges because their passive G is identical.
    q0_mean = float(np.mean(q_by_choice["micro_hessian"][0.0]))
    calibration = passive_measured / q0_mean
    measured_increment_by_ko = {
        float(row.k_o): float(row.J_h_radius_mean / passive_measured - 1.0)
        for row in measured.itertuples(index=False)
    }

    rows: list[dict[str, object]] = []
    for choice in ("micro_hessian", "major_symmetric_projection"):
        for ko in sorted(q_by_choice[choice]):
            if ko < -1.0e-12:
                continue
            q = q_by_choice[choice][ko]
            predicted_abs = calibration * float(np.mean(q))
            predicted_inc = predicted_abs / passive_measured - 1.0
            measured_abs = float(measured[np.isclose(measured.k_o, ko)].J_h_radius_mean.iloc[0])
            measured_inc = measured_increment_by_ko[ko]
            annulus_inc = calibration * q / passive_measured - 1.0
            inc_error = predicted_inc - measured_inc
            rows.append(
                {
                    "energy_split": choice,
                    "k_o": ko,
                    "passive_calibration_scalar": calibration,
                    "J_h_radius_mean": measured_abs,
                    "J_pred_mean": predicted_abs,
                    "measured_increment_over_passive": measured_inc,
                    "predicted_increment_over_passive": predicted_inc,
                    "increment_difference": inc_error,
                    "increment_relative_error": (inc_error / measured_inc if abs(measured_inc) > 1.0e-14 else 0.0),
                    "predicted_increment_annulus_min": float(np.min(annulus_inc)),
                    "predicted_increment_annulus_max": float(np.max(annulus_inc)),
                }
            )
    write_csv(out_dir / "continuum_lattice_J_increment_bridge.csv", rows)

    k_rows: list[dict[str, object]] = []
    for ko, group in fits.groupby("global_k_o"):
        KI = group.K_I.to_numpy(float)
        KII = group.K_II.to_numpy(float)
        k_rows.append(
            {
                "k_o": float(ko),
                "n_annuli": len(group),
                "K_I_mean": float(KI.mean()),
                "K_I_annulus_sd": float(KI.std(ddof=1)),
                "K_I_min": float(KI.min()),
                "K_I_max": float(KI.max()),
                "K_II_mean": float(KII.mean()),
                "K_II_annulus_sd": float(KII.std(ddof=1)),
                "K_II_min": float(KII.min()),
                "K_II_max": float(KII.max()),
                "K_II_sd_over_abs_mean": float(KII.std(ddof=1) / max(abs(KII.mean()), 1.0e-30)),
            }
        )
    write_csv(out_dir / "fitted_K_annulus_uncertainty.csv", k_rows)

    def row_for(choice: str, ko: float) -> dict[str, object]:
        return next(r for r in rows if r["energy_split"] == choice and math.isclose(float(r["k_o"]), ko))

    micro20 = row_for("micro_hessian", 0.2)
    major20 = row_for("major_symmetric_projection", 0.2)
    k20 = next(r for r in k_rows if math.isclose(float(r["k_o"]), 0.2))
    measured20 = float(micro20["measured_increment_over_passive"])
    summary = {
        "calibration_definition": "alpha=mean_R[J_h(k_o=0)]/mean_annulus[K^T G_0 K]",
        "passive_calibration_scalar": calibration,
        "calibration_distance_from_unity": calibration - 1.0,
        "measured_increment_ko0p2": measured20,
        "microenergetic_predicted_increment_ko0p2": float(micro20["predicted_increment_over_passive"]),
        "major_symmetric_predicted_increment_ko0p2": float(major20["predicted_increment_over_passive"]),
        "microenergetic_increment_relative_error_ko0p2": float(micro20["increment_relative_error"]),
        "major_symmetric_increment_relative_error_ko0p2": float(major20["increment_relative_error"]),
        "microenergetic_annulus_increment_range_ko0p2": [float(micro20["predicted_increment_annulus_min"]), float(micro20["predicted_increment_annulus_max"])],
        "major_symmetric_annulus_increment_range_ko0p2": [float(major20["predicted_increment_annulus_min"]), float(major20["predicted_increment_annulus_max"])],
        "KII_ko0p2_mean": float(k20["K_II_mean"]),
        "KII_ko0p2_annulus_sd": float(k20["K_II_annulus_sd"]),
        "KII_ko0p2_range": [float(k20["K_II_min"]), float(k20["K_II_max"])],
        "KII_ko0p2_sd_over_abs_mean": float(k20["K_II_sd_over_abs_mean"]),
        "interpretation": (
            "Increment normalization exposes a 37% microenergetic overprediction and a 25% major-symmetric "
            "underprediction at k_o/k=0.20. Both annulus ranges bracket the measured increment, and K_II varies "
            "by an amount comparable to its mean. The bridge supports sign and order of magnitude but is not a "
            "quantitative gauge discriminator; objectivity and microscopic energetics select the gauge."
        ),
    }
    summary["pass"] = bool(
        1.20 < calibration < 1.35
        and 0.30 < summary["microenergetic_increment_relative_error_ko0p2"] < 0.45
        and -0.35 < summary["major_symmetric_increment_relative_error_ko0p2"] < -0.15
        and summary["KII_ko0p2_sd_over_abs_mean"] > 0.5
    )
    (out_dir / "continuum_lattice_J_increment_bridge_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def glide_even_part_and_period_extrapolation(out_dir: Path) -> dict[str, object]:
    periods = pd.read_csv(ROOT / "data" / "advance_resistance_results" / "advance_period_summary.csv")
    p90 = periods[np.isclose(periods.load_fraction, 0.90)].copy()
    fit_rows: list[dict[str, object]] = []
    roots: list[float] = []
    for keys, group in p90.groupby(["nx", "ny", "crack_fraction", "boundary_condition", "period"]):
        group = group.sort_values("k_o")
        x = group.k_o.to_numpy(float) ** 2
        y = group.work_ratio.to_numpy(float)
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ coef
        intercept, slope = map(float, coef)
        root = float("nan")
        if slope > 0.0 and intercept < 1.0:
            root = math.sqrt((1.0 - intercept) / slope)
            roots.append(root)
        fit_rows.append(
            {
                "nx": int(keys[0]),
                "ny": int(keys[1]),
                "crack_fraction": float(keys[2]),
                "boundary_condition": str(keys[3]),
                "period": int(keys[4]),
                "intercept_at_ko0": intercept,
                "slope_vs_ko_squared": slope,
                "maximum_abs_fit_residual": float(np.max(np.abs(residual))),
                "positive_crossing_estimate": root,
            }
        )
    write_csv(out_dir / "period_sum_quadratic_fits_p0p90.csv", fit_rows)

    envelope = p90.groupby("k_o", as_index=False).work_ratio.max().sort_values("k_o")
    x = envelope.k_o.to_numpy(float) ** 2
    y = envelope.work_ratio.to_numpy(float)
    X = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    env_res = y - X @ c

    nominal = p90[(p90.nx == 48) & np.isclose(p90.crack_fraction, 0.125)].copy()
    nominal_rows = nominal[["boundary_condition", "k_o", "period", "work_ratio", "odd_work"]].to_dict("records")
    write_csv(out_dir / "nominal_period_sum_vs_ko.csv", nominal_rows)
    summary = {
        "symmetry_scope": "glide protects only the k_o-odd O(k_o) contribution; pair-summed even terms begin at O(k_o^2)",
        "number_of_quadratic_fits": len(fit_rows),
        "number_of_positive_crossing_estimates_at_p0p90": len(roots),
        "minimum_positive_crossing_estimate": min(roots) if roots else None,
        "envelope_intercept": float(c[0]),
        "envelope_slope_vs_ko_squared": float(c[1]),
        "envelope_max_abs_residual": float(np.max(np.abs(env_res))),
        "maximum_observed_period_ratio_p0p90": float(p90.work_ratio.max()),
        "fixed_grip_period_ratio_range_at_ko0p30": [
            float(p90[(p90.boundary_condition == "fixed_grip") & np.isclose(p90.k_o, 0.30)].work_ratio.min()),
            float(p90[(p90.boundary_condition == "fixed_grip") & np.isclose(p90.k_o, 0.30)].work_ratio.max()),
        ],
        "dead_load_period_ratio_range_at_ko0p30": [
            float(p90[(p90.boundary_condition == "dead_load") & np.isclose(p90.k_o, 0.30)].work_ratio.min()),
            float(p90[(p90.boundary_condition == "dead_load") & np.isclose(p90.k_o, 0.30)].work_ratio.max()),
        ],
        "interpretation": (
            "The unprotected pair-summed contribution is quadratic but ensemble dependent. At P/P_init=0.90, "
            "every fitted slope is nonpositive and no positive-k_o crossing of unity is predicted; therefore the "
            "available data do not support a finite critical k_o estimate. Extrapolation beyond 0.30 would be unjustified."
        ),
    }
    summary["pass"] = bool(
        summary["number_of_positive_crossing_estimates_at_p0p90"] == 0
        and summary["maximum_observed_period_ratio_p0p90"] < 1.0
        and summary["envelope_slope_vs_ko_squared"] < 0.0
    )
    (out_dir / "glide_even_part_period_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def directional_bias_convergence_audit(out_dir: Path) -> dict[str, object]:
    summary0 = json.loads((ROOT / "data" / "directional_driving_results" / "summary.json").read_text())
    vals = summary0["right_tip_ko0p20_normalized_odd_excess_bias_by_size"]
    nx = np.array(sorted(int(k) for k in vals), dtype=float)
    y = np.array([float(vals[str(int(n))]) for n in nx])
    x = 1.0 / nx
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ coef
    r2 = 1.0 - float(residual @ residual) / float((y - y.mean()) @ (y - y.mean()))
    increments = np.diff(y)
    xsteps = -np.diff(x)
    apparent_slopes = increments / xsteps
    rows = [
        {
            "nx": int(n),
            "a_lat_over_L": float(xx),
            "chi_A": float(yy),
            "linear_fit": float(pp),
            "linear_fit_residual": float(rr),
        }
        for n, xx, yy, pp, rr in zip(nx, x, y, X @ coef, residual)
    ]
    write_csv(out_dir / "directional_bias_size_audit.csv", rows)
    summary = {
        "values": {str(int(n)): float(v) for n, v in zip(nx, y)},
        "successive_chi_increments": increments.tolist(),
        "successive_a_over_L_step_magnitudes": xsteps.tolist(),
        "successive_secant_slopes": apparent_slopes.tolist(),
        "linear_in_a_over_L_intercept": float(coef[0]),
        "linear_in_a_over_L_slope": float(coef[1]),
        "linear_fit_R2": r2,
        "maximum_fit_residual": float(np.max(np.abs(residual))),
        "convergence_claim_supported": False,
        "interpretation": (
            "The three values establish sign and order of magnitude only. Successive chi_A increments do not shrink "
            "with the halved a/L step, so no converged continuum value is claimed."
        ),
    }
    summary["pass"] = bool(
        increments[1] >= 0.9 * increments[0]
        and apparent_slopes[1] > apparent_slopes[0]
        and summary["convergence_claim_supported"] is False
    )
    (out_dir / "directional_bias_convergence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _state_function_unit(nx: int, ny: int, crack: float, ko: float) -> dict[str, float]:
    model = ActiveCrackedStrip(nx, ny, crack, 1.0, ko)
    cut = model.tip_candidates()["right"]
    old_ids = [b for b in range(len(model.all_bonds)) if b not in model.removed_ids]
    new_ids = [b for b in old_ids if b != cut]
    Ee0, Eo0 = assemble_components(model, old_ids)
    Ee1, Eo1 = assemble_components(model, new_ids)
    u0, c0, cv0, f0, r0 = solve_equilibrium(Ee0 + Eo0, model, 1.0)
    u1, c1, cv1, f1, r1 = solve_equilibrium(Ee1 + Eo1, model, 1.0)
    if not (np.array_equal(c0, c1) and np.array_equal(f0, f1) and np.allclose(cv0, cv1)):
        raise RuntimeError("boundary partition changed")
    state_work = even_energy(Ee0, u0) - even_energy(Ee1, u1)
    bond = model.all_bonds[cut]
    du = u0[2 * bond.j:2 * bond.j + 2] - u0[2 * bond.i:2 * bond.i + 2]
    extension = float(du @ bond.n)
    reactions = np.asarray((Ee0 + Eo0) @ u0)
    top_y = np.array([2 * model.node_id(i, model.ny - 1) + 1 for i in range(model.nx)])
    remote_stress = float(np.sum(reactions[top_y]) / model.period)
    return {
        "k_o": ko,
        "state_function_work_unit": state_work,
        "candidate_extension_unit": extension,
        "remote_stress_unit": remote_stress,
        "max_force_residual": max(float(r0), float(r1)),
    }


def constructive_state_function_criterion(out_dir: Path) -> dict[str, object]:
    unit = pd.read_csv(ROOT / "data" / "protocol_family_results" / "protocol_family_unit_scan.csv")
    if "fixed_remote_stress_unit" in unit.columns and "remote_stress_unit" not in unit.columns:
        unit = unit.rename(columns={"fixed_remote_stress_unit": "remote_stress_unit"})
    unit = unit[unit.nx == 48].sort_values("k_o")
    ko_values = unit.k_o.to_numpy(float)
    static_rows = [_state_function_unit(48, 36, 6.0, float(ko)) for ko in ko_values]
    static = pd.DataFrame(static_rows)
    merged = unit.merge(static, on="k_o", suffixes=("_protocol", "_state"))
    load = 0.90
    passive = merged.iloc[np.argmin(np.abs(merged.k_o.to_numpy(float)))]
    rows: list[dict[str, object]] = []
    for r in merged.itertuples(index=False):
        s = float(r.remote_stress_unit_protocol)
        state = load**2 * float(passive.remote_stress_unit_protocol) ** 2 * float(r.state_function_work_unit) / (
            s**2 * float(passive.state_function_work_unit)
        )
        operational = load**2 * float(passive.remote_stress_unit_protocol) ** 2 * float(r.fixed_protocol_work_unit) / (
            s**2 * float(passive.fixed_protocol_work_unit)
        )
        local = load * float(passive.remote_stress_unit_protocol) * float(r.candidate_extension_unit_protocol) / (
            s * float(passive.candidate_extension_unit_protocol)
        )
        rows.append(
            {
                "k_o": float(r.k_o),
                "load_fraction": load,
                "state_function_measure": state,
                "operational_work_measure": operational,
                "local_extension_measure": local,
                "state_minus_operational": state - operational,
                "state_minus_local": state - local,
                "max_force_residual": float(r.max_force_residual),
            }
        )
    write_csv(out_dir / "constructive_state_function_criterion.csv", rows)

    def crossing(name: str) -> float | None:
        x = np.array([float(r["k_o"]) for r in rows])
        y = np.array([float(r[name]) for r in rows]) - 1.0
        for i in range(len(x) - 1):
            if y[i] == 0.0:
                return float(x[i])
            if y[i] * y[i + 1] < 0.0:
                return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
        return None

    summary = {
        "load_fraction": load,
        "state_function_threshold_ko_linear": crossing("state_function_measure"),
        "operational_threshold_ko_linear": crossing("operational_work_measure"),
        "local_extension_threshold_ko_linear": crossing("local_extension_measure"),
        "maximum_state_vs_operational_difference": max(abs(float(r["state_minus_operational"])) for r in rows),
        "maximum_state_vs_local_difference": max(abs(float(r["state_minus_local"])) for r in rows),
        "maximum_static_force_residual": max(float(r["max_force_residual"]) for r in rows),
        "interpretation": (
            "-Delta U_e provides a state-function coarse-grained measure if active supply is booked separately. "
            "It is a constructive alternative, not an identity with the local tensile rule or with protocol-indexed total work."
        ),
    }
    summary["pass"] = bool(summary["maximum_static_force_residual"] < 1.0e-12)
    (out_dir / "constructive_state_function_criterion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_all(out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        "closed_form_energy_gauge": closed_form_energy_gauge_cancellation(out_dir),
        "increment_bridge": continuum_lattice_increment_bridge(out_dir),
        "glide_even_part": glide_even_part_and_period_extrapolation(out_dir),
        "directional_convergence": directional_bias_convergence_audit(out_dir),
        "constructive_state_criterion": constructive_state_function_criterion(out_dir),
    }
    summaries["pass"] = all(bool(v.get("pass", False)) for v in summaries.values())
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


if __name__ == "__main__":
    result = run_all(ROOT / "data" / "representation_symmetry_results")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit("Third-round checks failed")
