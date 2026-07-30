#!/usr/bin/env python3
"""Second-round reviewer checks and quantitative bridges.

This module audits the recoverable-energy split, glide-reflection symmetry,
quasistatic bond softening, continuum/lattice crack-tip flux closure, statistical
fits, and relaxation-tail accuracy requested in the second IJSS review.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigs
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from active_tip_scan import ActiveCrackedStrip
from crack_advance_work import (
    assemble_components,
    even_energy,
    integrate_relaxation,
    solve_equilibrium,
)
from crack_tip_asymptotics import J_matrix, OddModuli
from crack_tip_lattice_fit import homogenized_moduli
from same_endpoint_scaling import ScalingConfig, first_order_map, full_odd_work, mobility_matrix, prepare


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def energetic_split_audit(out_dir: Path) -> dict[str, object]:
    B, mu = 1.0, 0.5
    eps = 1.0e-5
    rows: list[dict[str, object]] = []
    derivatives: dict[str, np.ndarray] = {}
    for choice in ("micro_hessian", "major_symmetric_projection"):
        G0 = J_matrix(OddModuli(B, mu, 0.0, 0.0), n_theta=4001, energy_choice=choice)
        GpA = J_matrix(OddModuli(B, mu, eps, 0.0), n_theta=4001, energy_choice=choice)
        GmA = J_matrix(OddModuli(B, mu, -eps, 0.0), n_theta=4001, energy_choice=choice)
        GpK = J_matrix(OddModuli(B, mu, 0.0, eps), n_theta=4001, energy_choice=choice)
        GmK = J_matrix(OddModuli(B, mu, 0.0, -eps), n_theta=4001, energy_choice=choice)
        dA = (GpA - GmA) / (2.0 * eps)
        dK = (GpK - GmK) / (2.0 * eps)
        derivatives[choice] = dA
        for i in range(2):
            for j in range(2):
                rows.append(
                    {
                        "energy_split": choice,
                        "matrix_i": i,
                        "matrix_j": j,
                        "G0": float(G0[i, j]),
                        "dG_dAo_at_0": float(dA[i, j]),
                        "dG_dKo_at_0": float(dK[i, j]),
                    }
                )
    write_csv(out_dir / "continuous_energy_split_J_derivatives.csv", rows)
    analytic = (B + mu) / (8.0 * B * B * mu)
    micro = derivatives["micro_hessian"]
    projected = derivatives["major_symmetric_projection"]
    summary = {
        "definition_used_in_lattice": "C_e = Hessian(U_e), the conservative central-spring modulus",
        "alternative_split": "C_e=(C+C^T)/2 in the full-gradient pairing",
        "alternative_energy_increment": "Delta W=(A_o/2)e_0 e_1",
        "micro_hessian_dG12_dAo": float(micro[0, 1]),
        "analytic_micro_hessian_dG12_dAo": analytic,
        "major_projection_dG12_dAo": float(projected[0, 1]),
        "maximum_abs_dG_dKo": float(
            max(
                abs(float(r["dG_dKo_at_0"]))
                for r in rows
            )
        ),
        "interpretation": (
            "The recoverable-energy split is a continuum gauge. The microscopic Hessian is objective and "
            "matches the discrete bond energy. The major-symmetric projection adds (A_o/2)e0e1 and cancels "
            "the micro-Hessian first-order A_o cross-mode coefficient."
        ),
    }
    summary["pass"] = bool(
        abs(summary["micro_hessian_dG12_dAo"] - analytic) < 2.0e-8
        and abs(summary["major_projection_dG12_dAo"]) < 2.0e-8
        and summary["maximum_abs_dG_dKo"] < 2.0e-8
    )
    (out_dir / "continuous_energy_split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _glide_node_map(model: ActiveCrackedStrip) -> np.ndarray:
    """Node permutation for G:(x,y)->(x+a/2, h-y)."""
    mapping = np.empty(model.n_nodes, dtype=int)
    jl = model.j_lower
    for j in range(model.ny):
        jp = model.ny - 1 - j
        for i in range(model.nx):
            ip = (i + j - jl) % model.nx
            mapping[model.node_id(i, j)] = model.node_id(ip, jp)
    return mapping


def _glide_dof_matrix(model: ActiveCrackedStrip, node_map: np.ndarray) -> sparse.csr_matrix:
    # Polar vector reflection Q=diag(1,-1): (P u)_{G(i)}=Q u_i.
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for source, dest in enumerate(node_map):
        rows.extend([2 * dest, 2 * dest + 1])
        cols.extend([2 * source, 2 * source + 1])
        data.extend([1.0, -1.0])
    return sparse.coo_matrix((data, (rows, cols)), shape=(model.ndof, model.ndof)).tocsr()


def _glide_bond_map(model: ActiveCrackedStrip, node_map: np.ndarray) -> np.ndarray:
    lookup = {
        frozenset((bond.i, bond.j)): bid for bid, bond in enumerate(model.all_bonds)
    }
    mapping = np.empty(len(model.all_bonds), dtype=int)
    for bid, bond in enumerate(model.all_bonds):
        key = frozenset((int(node_map[bond.i]), int(node_map[bond.j])))
        mapping[bid] = lookup[key]
    return mapping


def _solve_with_constraints(matrix, model, constrained, values):
    mask = np.ones(model.ndof, dtype=bool)
    mask[constrained] = False
    free = np.arange(model.ndof)[mask]
    uf = sparse.linalg.spsolve(
        matrix[free][:, free], -(matrix[free][:, constrained] @ values)
    )
    u = np.zeros(model.ndof)
    u[constrained] = values
    u[free] = uf
    residual = np.asarray(matrix @ u)
    return u, free, float(np.max(np.abs(residual[free])))


def _arbitrary_topology_protocol(
    model: ActiveCrackedStrip,
    removed_before: set[int],
    cut_id: int,
    t_end: float = 50000.0,
    constrained: np.ndarray | None = None,
    constrained_values: np.ndarray | None = None,
) -> dict[str, float]:
    old = [bid for bid in range(len(model.all_bonds)) if bid not in removed_before]
    if cut_id not in old:
        raise ValueError("cut bond is already removed")
    new = [bid for bid in old if bid != cut_id]
    Ee0, Eo0 = assemble_components(model, old)
    Ee1, Eo1 = assemble_components(model, new)
    K0, K1 = Ee0 + Eo0, Ee1 + Eo1
    if constrained is None:
        c, cv = model.constrained_dofs(1.0)
    else:
        c = np.asarray(constrained, dtype=int)
        cv = np.asarray(constrained_values, dtype=float)
    u0, f, r0 = _solve_with_constraints(K0, model, c, cv)
    u1, f1, r1 = _solve_with_constraints(K1, model, c, cv)
    if not np.array_equal(f, f1):
        raise RuntimeError("constraint mismatch")
    Wodd, diss, rel, nsteps = integrate_relaxation(
        K1, Eo1, u0, u1, c, cv, f, t_end=t_end,
        rtol=1.0e-9, relative_atol=1.0e-11,
    )
    dU = even_energy(Ee1, u1) - even_energy(Ee0, u0)
    return {
        "odd_work": float(Wodd),
        "protocol_work": float(Wodd - dU),
        "delta_even_energy": float(dU),
        "dissipation": float(diss),
        "final_relative_norm": float(rel),
        "n_steps": int(nsteps),
        "force_residual": max(float(r0), float(r1)),
    }


def glide_reflection_audit(out_dir: Path) -> dict[str, object]:
    nx, ny, crack = 24, 18, 4.0
    ko = 0.2
    base = ActiveCrackedStrip(nx, ny, crack, 1.0, ko)
    node_map = _glide_node_map(base)
    bond_map = _glide_bond_map(base, node_map)
    P = _glide_dof_matrix(base, node_map)
    removed = set(base.removed_ids)
    removed_g = {int(bond_map[b]) for b in removed}
    cut = base.tip_candidates()["right"]
    cut_g = int(bond_map[cut])
    active = [b for b in range(len(base.all_bonds)) if b not in removed]
    active_g = [b for b in range(len(base.all_bonds)) if b not in removed_g]
    Ee, Eo = assemble_components(base, active)
    Eeg, Eog = assemble_components(base, active_g)
    even_error = float(sparse.linalg.norm(P.T @ Eeg @ P - Ee, ord=np.inf))
    odd_sign_error = float(sparse.linalg.norm(P.T @ Eog @ P + Eo, ord=np.inf))

    # Exact finite-domain relation between a topology and its glide image.
    original_minus = ActiveCrackedStrip(nx, ny, crack, 1.0, -ko)
    transformed_plus = ActiveCrackedStrip(nx, ny, crack, 1.0, ko)
    c0, cv0 = original_minus.constrained_dofs(1.0)
    # Transform every prescribed component by the polar-vector glide reflection.
    transformed_constraints = []
    transformed_values = []
    for dof, value in zip(c0, cv0):
        node, comp = divmod(int(dof), 2)
        dest = int(node_map[node])
        transformed_constraints.append(2 * dest + comp)
        transformed_values.append(float(value) * (1.0 if comp == 0 else -1.0))
    order = np.argsort(transformed_constraints)
    cg = np.asarray(transformed_constraints, dtype=int)[order]
    cvg = np.asarray(transformed_values, dtype=float)[order]
    p_minus = _arbitrary_topology_protocol(original_minus, removed, cut, constrained=c0, constrained_values=cv0)
    p_plus_g = _arbitrary_topology_protocol(transformed_plus, removed_g, cut_g, constrained=cg, constrained_values=cvg)

    # Small-k_o reversal error is expected to be quadratic because the path also changes.
    small = 0.02
    p_small_plus = _arbitrary_topology_protocol(
        ActiveCrackedStrip(nx, ny, crack, 1.0, small), removed, cut
    )
    p_small_minus = _arbitrary_topology_protocol(
        ActiveCrackedStrip(nx, ny, crack, 1.0, -small), removed, cut
    )
    rows = [
        {"case": "original_-ko", **p_minus},
        {"case": "glide_image_+ko", **p_plus_g},
        {"case": "original_small_+ko", **p_small_plus},
        {"case": "original_small_-ko", **p_small_minus},
    ]
    write_csv(out_dir / "glide_reflection_protocol_check.csv", rows)
    exact_work_error = abs(p_plus_g["odd_work"] - p_minus["odd_work"])
    exact_A_error = abs(p_plus_g["protocol_work"] - p_minus["protocol_work"])
    small_reversal_residual = abs(p_small_plus["odd_work"] + p_small_minus["odd_work"])
    summary = {
        "glide_map": "G:(x,y)->(x+a_lat/2,h-y)",
        "operator_even_conjugacy_inf": even_error,
        "operator_odd_sign_conjugacy_inf": odd_sign_error,
        "exact_glide_odd_work_abs_error": exact_work_error,
        "exact_glide_protocol_work_abs_error": exact_A_error,
        "small_ko_odd_reversal_residual": small_reversal_residual,
        "small_ko_reversal_residual_over_ko_squared": small_reversal_residual / (small * small),
        "interpretation": (
            "The glide maps the finite strip to a translated reflected topology exactly. For a semi-infinite "
            "cleavage crack, this translated topology is the next deleted-bond registry, yielding the period-two "
            "relation. Finite two-tip strips deviate through the remote tip and boundaries."
        ),
    }
    summary["pass"] = bool(
        even_error < 1.0e-12
        and odd_sign_error < 1.0e-12
        and exact_work_error < 2.0e-10
        and exact_A_error < 2.0e-10
    )
    (out_dir / "glide_reflection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _softening_path_case(
    nx: int,
    ny: int,
    crack: float,
    ko: float,
    n_steps: int,
) -> dict[str, float]:
    model = ActiveCrackedStrip(nx, ny, crack, 1.0, ko)
    cut = model.tip_candidates()["right"]
    old = [b for b in range(len(model.all_bonds)) if b not in model.removed_ids]
    rest = [b for b in old if b != cut]
    Ee_rest, Eo_rest = assemble_components(model, rest)
    Ee_cut, Eo_cut = assemble_components(model, [cut])
    c, cv = model.constrained_dofs(1.0)
    mask = np.ones(model.ndof, dtype=bool)
    mask[c] = False
    f = np.arange(model.ndof)[mask]

    # Cosine-clustered values resolve both endpoints of the equilibrium branch.
    q = np.linspace(0.0, 1.0, n_steps)
    s_values = 0.5 * (1.0 + np.cos(math.pi * q))
    states: list[np.ndarray] = []
    energies: list[float] = []
    odd_forces: list[np.ndarray] = []
    max_residual = 0.0
    for s in s_values:
        Ee = Ee_rest + float(s) * Ee_cut
        Eo = Eo_rest + float(s) * Eo_cut
        K = Ee + Eo
        u, c2, cv2, f2, residual = solve_equilibrium(K, model, 1.0)
        if not (np.array_equal(c, c2) and np.array_equal(f, f2) and np.allclose(cv, cv2)):
            raise RuntimeError("softening constraints changed")
        states.append(u)
        energies.append(even_energy(Ee, u))
        odd_forces.append(np.asarray(-(Eo[f][:, :] @ u)))
        max_residual = max(max_residual, float(residual))
    Wodd = 0.0
    for i in range(1, len(states)):
        du = states[i][f] - states[i - 1][f]
        Wodd += 0.5 * float((odd_forces[i] + odd_forces[i - 1]) @ du)
    dU = energies[-1] - energies[0]
    return {
        "n_softening_points": n_steps,
        "k_o": ko,
        "odd_work": Wodd,
        "delta_even_energy": dU,
        "quasistatic_protocol_work": Wodd - dU,
        "maximum_equilibrium_residual": max_residual,
    }


def quasistatic_softening_audit(out_dir: Path) -> dict[str, object]:
    instant = pd.read_csv(
        ROOT / "data" / "protocol_family_results" / "same_endpoint_mobility_protocol.csv"
    )
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for ko in (0.12, 0.222271):
        cases = [_softening_path_case(48, 36, 6.0, ko, n) for n in (65, 129, 257)]
        rows.extend(cases)
        final = cases[-1]
        inst = instant[np.isclose(instant.k_o, ko)]
        amin = float(inst.protocol_work.min())
        amax = float(inst.protocol_work.max())
        Aqs = float(final["quasistatic_protocol_work"])
        summaries.append(
            {
                "k_o": ko,
                "quasistatic_protocol_work": Aqs,
                "instantaneous_cut_min_over_M": amin,
                "instantaneous_cut_max_over_M": amax,
                "location_relative_to_instant_interval": (
                    "inside" if amin <= Aqs <= amax else ("below" if Aqs < amin else "above")
                ),
                "distance_to_interval": 0.0 if amin <= Aqs <= amax else min(abs(Aqs - amin), abs(Aqs - amax)),
                "relative_129_to_257_change": abs(cases[-1]["quasistatic_protocol_work"] - cases[-2]["quasistatic_protocol_work"]) / max(abs(Aqs), 1.0e-30),
            }
        )
    write_csv(out_dir / "quasistatic_softening_convergence.csv", rows)
    write_csv(out_dir / "quasistatic_vs_instantaneous_protocol.csv", summaries)
    summary = {
        "softening_law": "the conservative and odd interactions of the selected bond are multiplied by the same scalar s from 1 to 0",
        "viscous_limit": "equilibrium branch; D_eta -> 0",
        "cases": summaries,
        "maximum_relative_quadrature_change": max(float(r["relative_129_to_257_change"]) for r in summaries),
        "maximum_interval_violation": max(float(r["distance_to_interval"]) for r in summaries),
        "interpretation": (
            "Quasistatic softening selects a mobility-independent value only after a bond-deactivation law is specified. "
            "The instantaneous-cut mobility span remains irreducible within that protocol family, not across all possible debonding laws."
        ),
    }
    summary["pass"] = bool(
        summary["maximum_relative_quadrature_change"] < 2.0e-5
        and all(float(r["distance_to_interval"]) < 0.01 for r in summaries)
    )
    (out_dir / "quasistatic_softening_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def continuum_lattice_flux_bridge(out_dir: Path) -> dict[str, object]:
    fits = pd.read_csv(
        ROOT / "data" / "crack_tip_lattice_fit_results" / "stress_fit_annulus_scan.csv"
    )
    fits = fits[(fits.tip == "right") & (fits.basis == "matched_odd")].copy()
    jdata = pd.read_csv(
        ROOT / "data" / "discrete_configurational_results" / "discrete_domain_radius_scan.csv"
    )
    measured = jdata.groupby("k_o", as_index=False).J_total.mean().rename(columns={"J_total": "J_h_radius_mean"})
    rows: list[dict[str, object]] = []
    for choice in ("micro_hessian", "major_symmetric_projection"):
        q_by_annulus: dict[tuple[float, float, float], float] = {}
        G_cache = {
            float(ko): J_matrix(
                homogenized_moduli(1.0, float(ko)),
                n_theta=801,
                energy_choice=choice,
            )
            for ko in sorted(fits.global_k_o.unique())
        }
        for r in fits.itertuples(index=False):
            K = np.array([float(r.K_I), float(r.K_II)])
            G = G_cache[float(r.global_k_o)]
            q_by_annulus[(float(r.global_k_o), float(r.r_inner), float(r.r_outer))] = float(K @ G @ K)
        passive_measured = float(measured[np.isclose(measured.k_o, 0.0)].J_h_radius_mean.iloc[0])
        for ko in sorted(fits.global_k_o.unique()):
            if ko < -1.0e-12:
                continue
            pred_annuli = []
            for ann in fits[np.isclose(fits.global_k_o, ko)][["r_inner", "r_outer"]].itertuples(index=False):
                key = (float(ko), float(ann.r_inner), float(ann.r_outer))
                key0 = (0.0, float(ann.r_inner), float(ann.r_outer))
                pred_annuli.append(passive_measured * q_by_annulus[key] / q_by_annulus[key0])
            actual = float(measured[np.isclose(measured.k_o, ko)].J_h_radius_mean.iloc[0])
            rows.append(
                {
                    "energy_split": choice,
                    "k_o": float(ko),
                    "J_h_radius_mean": actual,
                    "J_pred_passive_calibrated_mean": float(np.mean(pred_annuli)),
                    "J_pred_annulus_min": float(np.min(pred_annuli)),
                    "J_pred_annulus_max": float(np.max(pred_annuli)),
                    "relative_prediction_error": float((np.mean(pred_annuli) - actual) / actual),
                    "measured_ratio_to_passive": actual / passive_measured,
                    "predicted_ratio_to_passive": float(np.mean(pred_annuli)) / passive_measured,
                }
            )
    write_csv(out_dir / "continuum_lattice_J_bridge.csv", rows)
    micro = [r for r in rows if r["energy_split"] == "micro_hessian"]
    proj = [r for r in rows if r["energy_split"] == "major_symmetric_projection"]
    summary = {
        "comparison": "fitted (K_I,K_II) inserted into the continuum G matrix; one passive scalar calibration removes reconstruction amplitude bias",
        "micro_hessian_max_abs_relative_error": max(abs(float(r["relative_prediction_error"])) for r in micro),
        "major_projection_max_abs_relative_error": max(abs(float(r["relative_prediction_error"])) for r in proj),
        "micro_hessian_ko0p2_predicted_ratio": next(float(r["predicted_ratio_to_passive"]) for r in micro if math.isclose(float(r["k_o"]), 0.2)),
        "measured_ko0p2_ratio": next(float(r["measured_ratio_to_passive"]) for r in micro if math.isclose(float(r["k_o"]), 0.2)),
        "interpretation": (
            "The continuum and lattice calculations are now quantitatively connected. Agreement is at the few-percent "
            "level after passive calibration, commensurate with the 7-8% stress-fit residual; the comparison also exposes "
            "the recoverable-energy split dependence."
        ),
    }
    summary["pass"] = bool(summary["micro_hessian_max_abs_relative_error"] < 0.05)
    (out_dir / "continuum_lattice_J_bridge_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _linear_fit_with_se(x: np.ndarray, y: np.ndarray, through_origin: bool) -> dict[str, float]:
    if through_origin:
        X = x[:, None]
    else:
        X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ coef
    dof = len(y) - X.shape[1]
    variance = float(residual @ residual / dof)
    cov = variance * np.linalg.inv(X.T @ X)
    slope = float(coef[-1])
    slope_se = math.sqrt(float(cov[-1, -1]))
    if through_origin:
        r2_origin = 1.0 - float(residual @ residual) / float(y @ y)
        intercept = 0.0
    else:
        r2_origin = 1.0 - float(residual @ residual) / float((y - y.mean()) @ (y - y.mean()))
        intercept = float(coef[0])
    return {
        "slope": slope,
        "slope_standard_error": slope_se,
        "intercept": intercept,
        "degrees_of_freedom": dof,
        "r2_definition_value": r2_origin,
        "maximum_abs_residual": float(np.max(np.abs(residual))),
        "root_mean_square_residual": float(math.sqrt(np.mean(residual**2))),
    }


def statistical_audit(out_dir: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    # Through-origin continuum keyhole/source slope used in Eq. (Jfit).
    d = pd.read_csv(ROOT / "data" / "apparent_j_results" / "annulus_balance.csv")
    d = d[d.tip == "right"]
    xcol = "Q_odd"
    ycol = "active_excess_contour_drift"
    fit1 = _linear_fit_with_se(d[xcol].to_numpy(float), d[ycol].to_numpy(float), True)
    rows.append({"fit": "continuum_keyhole_source", **fit1})

    db = pd.read_csv(ROOT / "data" / "discrete_configurational_results" / "discrete_configurational_balance.csv")
    fit2 = _linear_fit_with_se(db.direct_odd_term.to_numpy(float), db.excess_total.to_numpy(float), True)
    rows.append({"fit": "discrete_odd_term", **fit2})

    # Passive initiation exponent, no exponent imposed.
    pb = pd.read_csv(ROOT / "data" / "baseline_results" / "passive_crack_scan.csv")
    sel = pb[(pb.effective_half_length >= 6.5) & (pb.effective_half_length <= 12.0)]
    lx = np.log(sel.effective_half_length.to_numpy(float))
    ly = np.log(sel.initiation_remote_stress.to_numpy(float))
    X = np.column_stack([np.ones_like(lx), lx])
    coef, *_ = np.linalg.lstsq(X, ly, rcond=None)
    res = ly - X @ coef
    dof = len(lx) - 2
    var = float(res @ res / dof)
    cov = var * np.linalg.inv(X.T @ X)
    se = math.sqrt(float(cov[1, 1]))
    crit = float(student_t.ppf(0.975, dof))
    exponent = float(coef[1])
    exponent_ci = [exponent - crit * se, exponent + crit * se]

    # Gauge systematic interval across radii and excess definitions.
    gf = pd.read_csv(ROOT / "data" / "gauge_convergence_results" / "gauge_convergence_fits.csv")
    active = gf[np.isclose(gf.k_o, 0.15)]
    systematic = [
        float(active.active_relative_exponent.min()),
        float(max(active.active_relative_exponent.max(), active.excess_relative_exponent.max())),
    ]

    # Four-size path extrapolation.  The first three sizes retain the complete
    # mobility map; N_x=80 adds only the two extrema that carry the intercept.
    ss = pd.read_csv(ROOT / "data" / "same_endpoint_size_scaling_results" / "size_normalized_summary.csv")
    cfg80 = ScalingConfig(nx=80, ny=60, crack_half_length=10.0, t_end=250000.0, rtol=2.0e-8, relative_atol=1.0e-10)
    ext80, delta80 = first_order_map(cfg80, [4.0], [0.0, 3.0 * math.pi / 8.0])
    phi80 = [float(r["phi_first_order"]) for r in ext80]
    row80 = {
        "nx": 80,
        "ny": 60,
        "a_lat_over_L": 1.0 / 80.0,
        "passive_resistance": -float(delta80),
        "phi_min": min(phi80),
        "phi_max": max(phi80),
        "phi_span_over_passive_resistance": (max(phi80) - min(phi80)) / (-float(delta80)),
        "min_ratio": 4.0,
        "min_theta_over_pi": 0.0,
        "max_ratio": 4.0,
        "max_theta_over_pi": 0.375,
        "maximum_final_relative_norm": max(float(r["final_relative_norm"]) for r in ext80),
    }
    path_rows = [
        {
            "nx": int(r.nx),
            "ny": int(r.ny),
            "a_lat_over_L": float(r.a_lat_over_L),
            "passive_resistance": float(r.passive_resistance),
            "phi_min": float(r.phi_min),
            "phi_max": float(r.phi_max),
            "phi_span_over_passive_resistance": float(r.phi_span_over_passive_resistance),
            "min_ratio": float(r.min_ratio),
            "min_theta_over_pi": float(r.min_theta_over_pi),
            "max_ratio": float(r.max_ratio),
            "max_theta_over_pi": float(r.max_theta_over_pi),
            "maximum_final_relative_norm": float(r.max_first_order_final_relative_norm),
        }
        for r in ss.itertuples(index=False)
    ] + [row80]
    path_rows.sort(key=lambda r: int(r["nx"]))
    write_csv(out_dir / "path_coefficient_four_size.csv", path_rows)
    fit3 = _linear_fit_with_se(
        np.array([float(r["a_lat_over_L"]) for r in path_rows]),
        np.array([float(r["phi_span_over_passive_resistance"]) for r in path_rows]),
        False,
    )
    rows.append({"fit": "path_coefficient_vs_a_over_L_four_size", **fit3})
    write_csv(out_dir / "regression_definitions_and_uncertainty.csv", rows)
    summary = {
        "R0_squared_definition": "R0^2=1-sum(residual^2)/sum(y^2) for a regression constrained through the origin",
        "through_origin_fits": rows[:2],
        "passive_initiation_exponent": exponent,
        "passive_initiation_exponent_standard_error": se,
        "passive_initiation_exponent_95pct_CI": exponent_ci,
        "passive_initiation_fit_range_a_over_a_lat": [6.5, 12.0],
        "passive_initiation_fit_range_a_over_L": [6.5 / 64.0, 12.0 / 64.0],
        "gauge_nominal_statistical_CI": [0.915, 1.026],
        "gauge_systematic_exponent_range": systematic,
        "path_four_size_linear_fit": fit3,
        "path_nx80": row80,
    }
    summary["pass"] = bool(
        exponent_ci[0] < -0.5 < exponent_ci[1]
        and systematic[0] <= 0.84
        and systematic[1] >= 1.16
    )
    (out_dir / "statistical_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def relaxation_tail_audit(out_dir: Path) -> dict[str, object]:
    # Recompute the flagship finite-modulus mobility spans at doubled terminal time.
    existing = pd.read_csv(
        ROOT / "data" / "protocol_family_results" / "same_endpoint_mobility_protocol.csv"
    )
    rows: list[dict[str, object]] = []
    mode_rows: list[dict[str, object]] = []
    for ko in (0.12, 0.222271):
        cfg = ScalingConfig(nx=48, ny=36, crack_half_length=6.0, t_end=140000.0, rtol=5.0e-10, relative_atol=1.0e-12)
        for ratio in (0.5, 1.0, 2.0, 4.0):
            res = full_odd_work(cfg, ko, ratio, 0.0)
            rows.append({"k_o": ko, "mobility_ratio": ratio, **res})

        # Modal audit for the slowest main-scan mobility at this k_o.
        ratio = 0.5
        model, _Ee0, _Ee1, _Eo1, Knew, u0, u1, c, _cv, free = prepare(cfg, ko)
        M = mobility_matrix(model.ndof, ratio, 0.0)
        A = M[free][:, free] @ Knew[free][:, free]
        # Sixteen soft modes are enough to identify the first materially excited one.
        vals, vecs = eigs(A, k=16, which="SR", tol=2.0e-8, maxiter=30000)
        x0 = u0[free] - u1[free]
        norm0 = np.linalg.norm(x0)
        candidates = []
        for val, vec in zip(vals, vecs.T):
            amp = abs(np.vdot(vec, x0)) / max(np.linalg.norm(vec) * norm0, 1.0e-30)
            candidates.append((float(np.real(val)), float(abs(np.imag(val))), float(amp)))
        excited = [c for c in candidates if c[2] > 1.0e-7 and c[0] > 0.0]
        selected = min(excited, key=lambda z: z[0]) if excited else min(candidates, key=lambda z: z[0])
        mode_rows.append(
            {
                "k_o": ko,
                "mobility_ratio": ratio,
                "mode_real_part": selected[0],
                "mode_abs_imaginary_part": selected[1],
                "right_vector_overlap_fraction": selected[2],
                "tau": 1.0 / selected[0],
                "original_t_end": 70000.0,
                "original_t_end_over_tau": 70000.0 * selected[0],
                "refined_t_end_over_tau": 140000.0 * selected[0],
            }
        )
    write_csv(out_dir / "mobility_long_time_recalculation.csv", rows)
    write_csv(out_dir / "slowest_excited_mode.csv", mode_rows)
    comparisons = []
    for ko in (0.12, 0.222271):
        old = existing[np.isclose(existing.k_o, ko)]
        new = [r for r in rows if math.isclose(float(r["k_o"]), ko)]
        old_span = float(old.protocol_work.max() - old.protocol_work.min())
        new_values = np.array([float(r["protocol_work"]) for r in new])
        new_span = float(new_values.max() - new_values.min())
        comparisons.append(
            {
                "k_o": ko,
                "original_span": old_span,
                "refined_span": new_span,
                "absolute_span_change": abs(new_span - old_span),
                "relative_span_change": abs(new_span - old_span) / max(abs(new_span), 1.0e-30),
            }
        )
    write_csv(out_dir / "mobility_span_tail_convergence.csv", comparisons)
    summary = {
        "slowest_excited_modes": mode_rows,
        "span_convergence": comparisons,
        "maximum_relative_span_change": max(float(r["relative_span_change"]) for r in comparisons),
        "interpretation": "The advertised quantity max_M A-min_M A is directly converged, not inferred from endpoint offsets alone.",
    }
    summary["pass"] = bool(summary["maximum_relative_span_change"] < 5.0e-4)
    (out_dir / "relaxation_tail_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary



def independent_bond_sum_audit(out_dir: Path) -> dict[str, object]:
    """Cross-check sparse assembly against an independent direct bond-force sum."""
    rng = np.random.default_rng(20260726)
    model = ActiveCrackedStrip(32, 24, 4.0, 1.0, 0.2)
    active = [bid for bid in range(len(model.all_bonds)) if bid not in model.removed_ids]
    Ee, Eo = assemble_components(model, active)
    u = rng.normal(size=model.ndof)
    f_even_direct = np.zeros(model.ndof)
    f_odd_direct = np.zeros(model.ndof)
    energy_direct = 0.0
    odd_power_direct = 0.0
    v = rng.normal(size=model.ndof)
    for bid in active:
        bond = model.all_bonds[bid]
        du = u[2*bond.j:2*bond.j+2] - u[2*bond.i:2*bond.i+2]
        ext = float(du @ bond.n)
        t = np.array([-bond.n[1], bond.n[0]])
        fe = model.k * ext * bond.n
        fo = -model.k_o * ext * t
        f_even_direct[2*bond.i:2*bond.i+2] += fe
        f_even_direct[2*bond.j:2*bond.j+2] -= fe
        f_odd_direct[2*bond.i:2*bond.i+2] += fo
        f_odd_direct[2*bond.j:2*bond.j+2] -= fo
        energy_direct += 0.5 * model.k * ext * ext
        odd_power_direct += float(fo @ (v[2*bond.i:2*bond.i+2] - v[2*bond.j:2*bond.j+2]))
    f_even_matrix = -(Ee @ u)
    f_odd_matrix = -(Eo @ u)
    energy_matrix = 0.5 * float(u @ (Ee @ u))
    odd_power_matrix = float(f_odd_matrix @ v)
    summary = {
        "even_force_max_abs_error": float(np.max(np.abs(f_even_direct-f_even_matrix))),
        "odd_force_max_abs_error": float(np.max(np.abs(f_odd_direct-f_odd_matrix))),
        "even_energy_abs_error": abs(energy_direct-energy_matrix),
        "odd_virtual_power_abs_error": abs(odd_power_direct-odd_power_matrix),
        "interpretation": "Independent explicit bond loops reproduce sparse assembly forces, conservative energy, and odd virtual power.",
    }
    summary["pass"] = bool(max(summary[k] for k in (
        "even_force_max_abs_error", "odd_force_max_abs_error", "even_energy_abs_error", "odd_virtual_power_abs_error")) < 2e-11)
    (out_dir / "independent_bond_sum_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_all(out_dir: Path, recompute_all: bool = False) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if recompute_all:
        summaries = {
            "energetic_split": energetic_split_audit(out_dir),
            "glide_reflection": glide_reflection_audit(out_dir),
            "quasistatic_softening": quasistatic_softening_audit(out_dir),
            "continuum_lattice_flux_bridge": continuum_lattice_flux_bridge(out_dir),
            "statistics": statistical_audit(out_dir),
            "relaxation_tail": relaxation_tail_audit(out_dir),
            "independent_bond_sum": independent_bond_sum_audit(out_dir),
        }
    else:
        files = {
            "energetic_split": "continuous_energy_split_summary.json",
            "glide_reflection": "glide_reflection_summary.json",
            "quasistatic_softening": "quasistatic_softening_summary.json",
            "continuum_lattice_flux_bridge": "continuum_lattice_J_bridge_summary.json",
            "statistics": "statistical_audit_summary.json",
            "relaxation_tail": "relaxation_tail_summary.json",
            "independent_bond_sum": "independent_bond_sum_summary.json",
        }
        missing = [name for name in files.values() if not (out_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing archived second-round audit outputs: {missing}. Run with recompute_all=True.")
        summaries = {
            key: json.loads((out_dir / name).read_text(encoding="utf-8"))
            for key, name in files.items()
        }
    summaries["pass"] = all(bool(v.get("pass", False)) for v in summaries.values())
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


if __name__ == "__main__":
    result = run_all(ROOT / "data" / "continuum_lattice_validation_results", recompute_all=True)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit("Second-round checks failed")
