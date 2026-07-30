#!/usr/bin/env python3
"""Small-odd-modulus scaling for same-endpoint crack-advance work.

For the fixed-grip virtual bond cut, the odd stiffness is linear in k_o.  At
first order, endpoint and trajectory corrections do not enter the odd work:

    W_odd(M;k_o) = k_o Phi(M;K_e,O,Delta u) + O(k_o^2),

where Phi is the unit-odd-force line integral along the passive relaxation path
selected by the positive-definite mobility M.  This module evaluates Phi for a
rotated unit-determinant mobility family and verifies the expansion against the
full odd dynamics.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import block_diag, csr_matrix

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from active_tip_scan import ActiveCrackedStrip
from crack_advance_work import assemble_components, even_energy, solve_equilibrium


@dataclass(frozen=True)
class ScalingConfig:
    nx: int = 32
    ny: int = 24
    crack_half_length: float = 5.0
    tip: str = "right"
    t_end: float = 40000.0
    rtol: float = 2.0e-8
    relative_atol: float = 1.0e-10


def active_ids(model: ActiveCrackedStrip) -> list[int]:
    return [i for i in range(len(model.all_bonds)) if i not in model.removed_ids]


def mobility_matrix(ndof: int, ratio: float, theta: float) -> csr_matrix:
    """Rotated SPD mobility with determinant one in every nodal 2x2 block."""
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    D = np.diag([math.sqrt(ratio), 1.0 / math.sqrt(ratio)])
    block = R @ D @ R.T
    return block_diag([block] * (ndof // 2), format="csr")


def _integrate_path(
    Kff,
    Kfc,
    Off,
    Ofc,
    u1f: np.ndarray,
    uc: np.ndarray,
    x0: np.ndarray,
    Mff,
    t_end: float,
    rtol: float,
    relative_atol: float,
) -> dict[str, float]:
    A = Mff @ Kff
    scale = max(float(np.max(np.abs(x0))), 1.0e-12)
    sol = solve_ivp(
        lambda _t, x: -(A @ x),
        (0.0, t_end),
        x0,
        method="BDF",
        jac=-A,
        rtol=rtol,
        atol=relative_atol * scale,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    def fields(x: np.ndarray):
        uf = u1f + x
        fodd = -(Off @ uf + Ofc @ uc)
        fnet = -(Kff @ uf + Kfc @ uc)
        return np.asarray(fodd), np.asarray(fnet)

    Wodd = 0.0
    diss = 0.0
    fodd0, fnet0 = fields(sol.y[:, 0])
    for j in range(1, len(sol.t)):
        du = sol.y[:, j] - sol.y[:, j - 1]
        fodd1, fnet1 = fields(sol.y[:, j])
        Wodd += 0.5 * float((fodd0 + fodd1) @ du)
        diss += 0.5 * float((fnet0 + fnet1) @ du)
        fodd0, fnet0 = fodd1, fnet1
    rel = float(np.linalg.norm(sol.y[:, -1]) / max(np.linalg.norm(x0), 1.0e-30))
    return {"odd_work": Wodd, "dissipation": diss, "final_relative_norm": rel, "steps": len(sol.t)}


def prepare(config: ScalingConfig, k_o: float):
    model = ActiveCrackedStrip(config.nx, config.ny, config.crack_half_length, 1.0, k_o)
    cut = model.tip_candidates()[config.tip]
    old = active_ids(model)
    new = [i for i in old if i != cut]
    Ee_old, Eo_old = assemble_components(model, old)
    Ee_new, Eo_new = assemble_components(model, new)
    Kold = Ee_old + Eo_old
    Knew = Ee_new + Eo_new
    u0, c, cv, f, _ = solve_equilibrium(Kold, model, 1.0)
    u1, c1, cv1, f1, _ = solve_equilibrium(Knew, model, 1.0)
    if not (np.array_equal(c, c1) and np.array_equal(f, f1) and np.allclose(cv, cv1)):
        raise RuntimeError("Endpoint constraint sets changed across the virtual cut")
    return model, Ee_old, Ee_new, Eo_new, Knew, u0, u1, c, cv, f


def first_order_map(config: ScalingConfig, ratios: list[float], angles: list[float]):
    # Passive endpoints and trajectory operator.
    model, Ee_old, Ee_new, _zero, Knew, u0, u1, c, cv, f = prepare(config, 0.0)
    # Unit odd matrix on the post-cut topology.
    model1 = ActiveCrackedStrip(config.nx, config.ny, config.crack_half_length, 1.0, 1.0)
    cut1 = model1.tip_candidates()[config.tip]
    new1 = [i for i in active_ids(model1) if i != cut1]
    _even1, Ounit = assemble_components(model1, new1)

    Kff = Knew[f][:, f]
    Kfc = Knew[f][:, c]
    Off = Ounit[f][:, f]
    Ofc = Ounit[f][:, c]
    x0 = u0[f] - u1[f]
    rows = []
    for ratio in ratios:
        for theta in angles:
            M = mobility_matrix(model.ndof, ratio, theta)
            res = _integrate_path(
                Kff, Kfc, Off, Ofc, u1[f], cv, x0, M[f][:, f],
                config.t_end, config.rtol, config.relative_atol,
            )
            rows.append({
                "mobility_ratio": ratio,
                "theta": theta,
                "theta_over_pi": theta / math.pi,
                "phi_first_order": res["odd_work"],
                "passive_dissipation": res["dissipation"],
                "final_relative_norm": res["final_relative_norm"],
                "steps": res["steps"],
            })
    endpoint_energy = even_energy(Ee_new, u1) - even_energy(Ee_old, u0)
    return rows, float(endpoint_energy)


def full_odd_work(config: ScalingConfig, k_o: float, ratio: float, theta: float):
    model, Ee_old, Ee_new, Eo_new, Knew, u0, u1, c, cv, f = prepare(config, k_o)
    M = mobility_matrix(model.ndof, ratio, theta)
    res = _integrate_path(
        Knew[f][:, f], Knew[f][:, c], Eo_new[f][:, f], Eo_new[f][:, c],
        u1[f], cv, u0[f] - u1[f], M[f][:, f],
        config.t_end, config.rtol, config.relative_atol,
    )
    delta_even = even_energy(Ee_new, u1) - even_energy(Ee_old, u0)
    res.update({
        "k_o": k_o,
        "mobility_ratio": ratio,
        "theta": theta,
        "theta_over_pi": theta / math.pi,
        "protocol_work": res["odd_work"] - delta_even,
        "delta_even_energy": delta_even,
    })
    return res


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(out_dir: Path, config: ScalingConfig = ScalingConfig()) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = [0.5, 1.0, 2.0, 4.0]
    angles = [0.0, math.pi / 8.0, math.pi / 4.0, 3.0 * math.pi / 8.0]
    map_rows, passive_delta_even = first_order_map(config, ratios, angles)
    ref = next(r for r in map_rows if r["mobility_ratio"] == 1.0 and abs(r["theta"]) < 1e-15)
    for r in map_rows:
        r["delta_phi_vs_isotropic"] = r["phi_first_order"] - ref["phi_first_order"]
    write_csv(out_dir / "mobility_first_order_map.csv", map_rows)

    # Validate finite-k_o scaling on a representative anisotropic subset.
    cases = [(0.5, 0.0), (2.0, 0.0), (4.0, math.pi / 8.0), (4.0, math.pi / 4.0)]
    ko_values = [-0.04, -0.02, -0.01, 0.01, 0.02, 0.04]
    full_rows = []
    ref_full = {ko: full_odd_work(config, ko, 1.0, 0.0) for ko in ko_values}
    phi_lookup = {(r["mobility_ratio"], r["theta"]): r["phi_first_order"] for r in map_rows}
    for ratio, theta in cases:
        dphi = phi_lookup[(ratio, theta)] - ref["phi_first_order"]
        for ko in ko_values:
            res = full_odd_work(config, ko, ratio, theta)
            actual = res["protocol_work"] - ref_full[ko]["protocol_work"]
            pred = ko * dphi
            full_rows.append({
                **res,
                "reference_protocol_work": ref_full[ko]["protocol_work"],
                "delta_protocol_work_vs_isotropic": actual,
                "first_order_prediction": pred,
                "prediction_abs_error": abs(actual - pred),
                "prediction_relative_error": abs(actual - pred) / max(abs(actual), 1.0e-30),
            })
    write_csv(out_dir / "finite_ko_validation.csv", full_rows)

    # Exact mobility-family symmetries: r=1 is angle independent and
    # M(r,theta+pi/2)=M(1/r,theta).
    sym_rows = []
    r1 = [r for r in map_rows if r["mobility_ratio"] == 1.0]
    r1_span = max(r["phi_first_order"] for r in r1) - min(r["phi_first_order"] for r in r1)
    sym_rows.append({"check": "r=1 angle invariance", "abs_difference": r1_span})
    lookup = {(r["mobility_ratio"], round(r["theta"], 14)): r["phi_first_order"] for r in map_rows}
    # Available paired case: (r=2,theta=0) equals (r=0.5,theta=pi/2), not in the
    # base angle list. Evaluate that extra point directly.
    extra_rows, _ = first_order_map(config, [0.5], [math.pi / 2.0])
    pair_diff = abs(phi_lookup[(2.0, 0.0)] - extra_rows[0]["phi_first_order"])
    sym_rows.append({"check": "M(r,theta)=M(1/r,theta+pi/2)", "abs_difference": pair_diff})
    write_csv(out_dir / "symmetry_checks.csv", sym_rows)

    small = [r for r in full_rows if abs(r["k_o"]) <= 0.02]
    max_small_rel = max(r["prediction_relative_error"] for r in small)
    max_small_quadratic_remainder = max(
        r["prediction_abs_error"] / (r["k_o"] ** 2) for r in small
    )
    summary = {
        "config": config.__dict__,
        "first_order_formula": "Delta A(M1,M2)=k_o[Phi(M1)-Phi(M2)]+O(k_o^2)",
        "passive_endpoint_even_energy_change": passive_delta_even,
        "isotropic_phi": ref["phi_first_order"],
        "phi_span_over_mobility_family": max(r["phi_first_order"] for r in map_rows) - min(r["phi_first_order"] for r in map_rows),
        "r1_angle_invariance_abs": r1_span,
        "reciprocal_rotation_symmetry_abs": pair_diff,
        "max_small_ko_prediction_relative_error": max_small_rel,
        "max_small_ko_abs_error_over_ko_squared": max_small_quadratic_remainder,
        "max_final_relative_norm": max([r["final_relative_norm"] for r in map_rows] + [r["final_relative_norm"] for r in full_rows]),
    }
    summary["pass"] = bool(
        summary["r1_angle_invariance_abs"] < 1e-12
        and summary["reciprocal_rotation_symmetry_abs"] < 1e-8
        and summary["max_small_ko_prediction_relative_error"] < 0.08
        and summary["max_final_relative_norm"] < 2e-6
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_analysis(PACKAGE_ROOT / "data" / "same_endpoint_scaling_results"), indent=2))
