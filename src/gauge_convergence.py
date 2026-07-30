#!/usr/bin/env python3
"""Localization-gauge convergence for the discrete configurational flux.

The triangular lattice is refined by increasing ``nx`` at fixed geometric
ratios.  The lattice spacing is the unit length, so ``a_lat/L = 1/nx``.  Crack
half-length, strip height, domain radius and weight-transition width all scale
with ``L=nx``.  For every size, odd modulus and localization parameter alpha,
the module evaluates the same piecewise-affine material-force functional.

The calculation is intentionally agnostic about the outcome.  It reports the
absolute gauge span, a span normalized by the reflection-symmetric flux, and
power-law fits versus a_lat/L.  The final classification is:

A: relative gauge span decreases consistently and the fitted exponent is
   positive;
B: the span approaches a nonzero plateau within the tested range;
C: the sequence is irregular or unresolved.

This is a finite-size diagnosis, not a theorem about the continuum limit.
"""
from __future__ import annotations

import csv
import gc
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from active_tip_scan import ActiveCrackedStrip
from discrete_configurational_analysis import (
    AREA_TRI,
    affine_gradient,
    build_triangles,
    localize_coords,
    q_value,
    tip_geometry,
)
from lattice_baselines import R90


@dataclass(frozen=True)
class GaugeConvergenceConfig:
    nx_values: tuple[int, ...] = (32, 48, 64, 80, 96)
    aspect_ny_over_nx: float = 0.75
    crack_half_over_nx: float = 0.125
    radius_over_nx: tuple[float, ...] = (0.075, 0.10, 0.125)
    width_over_nx: float = 0.025
    ko_values: tuple[float, ...] = (0.05, 0.15)
    alpha_values: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    tip: str = "right"
    q_lp_power: float = 4.0


def _even_integer(value: float) -> int:
    n = int(round(value))
    return n if n % 2 == 0 else n + 1


def geometry(nx: int, config: GaugeConvergenceConfig) -> tuple[int, float]:
    ny = max(8, _even_integer(config.aspect_ny_over_nx * nx))
    crack_half = config.crack_half_over_nx * nx
    return ny, crack_half


def partition_weights(model, triangles, alpha: float, axis: str = "y") -> dict[tuple[int, int], float]:
    """Assign each bond to adjacent triangles with an upper/lower split alpha."""
    adjacent: dict[int, list[int]] = defaultdict(list)
    for ti, tri in enumerate(triangles):
        for bid in tri.bond_ids:
            adjacent[bid].append(ti)

    weights: dict[tuple[int, int], float] = {}
    for bid, triangle_ids in adjacent.items():
        if len(triangle_ids) == 1:
            weights[(triangle_ids[0], bid)] = 1.0
        elif len(triangle_ids) == 2:
            values = []
            for ti in triangle_ids:
                centroid = np.mean(triangles[ti].coords, axis=0)
                values.append(float(centroid[1] if axis == "y" else centroid[0]))
            high = 0 if values[0] >= values[1] else 1
            low = 1 - high
            weights[(triangle_ids[high], bid)] = float(alpha)
            weights[(triangle_ids[low], bid)] = float(1.0 - alpha)
        else:
            raise RuntimeError(f"Bond {bid} belongs to {len(triangle_ids)} triangles")
    return weights


def weighted_fields(model, u, triangles, tip: str, support_radius: float, weights):
    active = set(range(len(model.all_bonds))) - set(model.removed_ids)
    _, _, direction = tip_geometry(model, tip)
    transform = np.diag([direction, 1.0])
    fields = []
    local_nodes = []

    for ti, tri in enumerate(triangles):
        xloc = localize_coords(tri.coords, model, tip)
        centroid = np.mean(xloc, axis=0)
        if np.max(np.abs(centroid)) > support_radius + 2.0:
            continue

        uloc = (transform @ u[np.asarray(tri.nodes)].T).T
        grad_u = affine_gradient(xloc, uloc)
        stress_even = np.zeros((2, 2))
        stress_odd = np.zeros((2, 2))
        energy_even = 0.0
        for bid in tri.bond_ids:
            if bid not in active:
                continue
            weight = weights[(ti, bid)]
            bond = model.all_bonds[bid]
            n = transform @ bond.n
            t = R90 @ n
            du = transform @ (u[bond.j] - u[bond.i])
            extension = float(du @ n)
            factor = weight / AREA_TRI
            stress_even += factor * model.k * extension * np.outer(n, n)
            stress_odd += factor * (-direction * model.k_o) * extension * np.outer(t, n)
            energy_even += factor * 0.5 * model.k * extension * extension
        fields.append((grad_u, stress_even, stress_odd, float(energy_even)))
        local_nodes.append(xloc)
    return fields, local_nodes


def evaluate_J(fields, local_nodes, radius: float, width: float, p: float) -> dict[str, float]:
    even = odd = 0.0
    for (grad_u, stress_even, stress_odd, energy_even), xnodes in zip(fields, local_nodes):
        qnod = np.array([q_value(x, radius, width, p) for x in xnodes])[:, None]
        grad_q = affine_gradient(xnodes, qnod)[0]
        ux = grad_u[:, 0]
        P_even = np.array([energy_even, 0.0]) - stress_even.T @ ux
        P_odd = -stress_odd.T @ ux
        even += -AREA_TRI * float(P_even @ grad_q)
        odd += -AREA_TRI * float(P_odd @ grad_q)
    return {"J_even": even, "J_odd": odd, "J_total": even + odd}


def force_reproduction_error(model, u, triangles, weights) -> float:
    """Interior nodal residual reproduced by the localized triangle stresses."""
    active = set(range(len(model.all_bonds))) - set(model.removed_ids)
    reconstructed = np.zeros_like(u)
    for ti, tri in enumerate(triangles):
        stress_even = np.zeros((2, 2))
        stress_odd = np.zeros((2, 2))
        for bid in tri.bond_ids:
            if bid not in active:
                continue
            weight = weights[(ti, bid)]
            bond = model.all_bonds[bid]
            n = bond.n
            t = R90 @ n
            extension = float((u[bond.j] - u[bond.i]) @ n)
            factor = weight / AREA_TRI
            stress_even += factor * model.k * extension * np.outer(n, n)
            stress_odd += factor * (-model.k_o) * extension * np.outer(t, n)
        interpolation = np.column_stack([np.ones(3), tri.coords[:, 0], tri.coords[:, 1]])
        coeff = np.linalg.inv(interpolation)
        for a, node in enumerate(tri.nodes):
            reconstructed[node] += AREA_TRI * (stress_even + stress_odd) @ coeff[1:, a]

    model_residual = (model.K @ u.ravel()).reshape(-1, 2)
    interior = np.ones(model.n_nodes, dtype=bool)
    for i in range(model.nx):
        interior[model.node_id(i, 0)] = False
        interior[model.node_id(i, model.ny - 1)] = False
    return float(np.max(np.abs((reconstructed - model_residual)[interior])))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _power_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) < 3:
        return {"prefactor": float("nan"), "exponent": float("nan"), "r2_log": float("nan")}
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    design = np.column_stack([np.ones_like(lx), lx])
    coeff, *_ = np.linalg.lstsq(design, ly, rcond=None)
    pred = design @ coeff
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {"prefactor": float(math.exp(coeff[0])), "exponent": float(coeff[1]), "r2_log": r2}


def _classify(summary_rows: list[dict[str, object]], fit_rows: list[dict[str, object]]) -> tuple[str, str]:
    target = [r for r in summary_rows if math.isclose(float(r["k_o"]), 0.15) and math.isclose(float(r["R_over_L"]), 0.10)]
    target.sort(key=lambda r: int(r["nx"]))
    fit = next((r for r in fit_rows if math.isclose(float(r["k_o"]), 0.15) and math.isclose(float(r["R_over_L"]), 0.10)), None)
    if len(target) < 4 or fit is None:
        return "C", "insufficient finite-size sequence"

    values = np.array([float(r["active_relative_gauge_span"]) for r in target])
    exponent = float(fit["active_relative_exponent"])
    r2 = float(fit["active_relative_r2_log"])
    monotone_fraction = float(np.mean(np.diff(values) < 0.0))
    reduction = float(values[-1] / values[0]) if values[0] > 0 else float("inf")

    if exponent > 0.25 and r2 > 0.80 and monotone_fraction >= 0.75 and reduction < 0.75:
        return "A", "relative gauge span decreases with a positive, well-resolved refinement exponent"
    if abs(exponent) < 0.15 and reduction > 0.75:
        return "B", "relative gauge span is approximately size independent over the tested range"
    return "C", "finite-size trend is irregular or not yet asymptotic"



def compute_size_rows(nx: int, config: GaugeConvergenceConfig) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compute one refinement level; isolated workers can release solver memory."""
    raw_rows: list[dict[str, object]] = []
    force_rows: list[dict[str, object]] = []
    ny, crack_half = geometry(nx, config)
    radii = [fraction * nx for fraction in config.radius_over_nx]
    width = config.width_over_nx * nx
    support = max(radii) + 0.5 * width + 1.0

    passive_model = ActiveCrackedStrip(nx, ny, crack_half, 1.0, 0.0)
    passive_u, _, passive_free_residual = passive_model.solve(1.0)
    triangles = build_triangles(passive_model)

    active_cases = {}
    for ko in config.ko_values:
        active_model = ActiveCrackedStrip(nx, ny, crack_half, 1.0, ko)
        active_u, _, active_free_residual = active_model.solve(1.0)
        active_cases[ko] = (active_model, active_u, active_free_residual)

    # Force reproduction is an exact algebraic audit.  Evaluate it on the two
    # extreme gauges and the reflection-symmetric gauge; intermediate alpha
    # values are convex combinations and need not repeat the full-domain sum.
    force_audit_alphas = {0.0, 0.5, 1.0}

    for alpha in config.alpha_values:
        passive_weights = partition_weights(passive_model, triangles, alpha)
        passive_fields, passive_nodes = weighted_fields(
            passive_model, passive_u, triangles, config.tip, support, passive_weights
        )
        passive_force_error = (
            force_reproduction_error(passive_model, passive_u, triangles, passive_weights)
            if alpha in force_audit_alphas else float("nan")
        )

        active_field_cache = {}
        for ko, (active_model, active_u, active_free_residual) in active_cases.items():
            active_weights = partition_weights(active_model, triangles, alpha)
            active_fields, active_nodes = weighted_fields(
                active_model, active_u, triangles, config.tip, support, active_weights
            )
            active_force_error = (
                force_reproduction_error(active_model, active_u, triangles, active_weights)
                if alpha in force_audit_alphas else float("nan")
            )
            active_field_cache[ko] = (active_fields, active_nodes)
            if alpha in force_audit_alphas:
                force_rows.append({
                    "nx": nx,
                    "ny": ny,
                    "a_lat_over_L": 1.0 / nx,
                    "crack_half_over_L": crack_half / nx,
                    "k_o": ko,
                    "alpha_upper": alpha,
                    "passive_equilibrium_residual_inf": passive_free_residual,
                    "active_equilibrium_residual_inf": active_free_residual,
                    "passive_force_reproduction_error": passive_force_error,
                    "active_force_reproduction_error": active_force_error,
                })

        for R_over_L, radius in zip(config.radius_over_nx, radii):
            passive_J = evaluate_J(passive_fields, passive_nodes, radius, width, config.q_lp_power)
            for ko in config.ko_values:
                active_fields, active_nodes = active_field_cache[ko]
                active_J = evaluate_J(active_fields, active_nodes, radius, width, config.q_lp_power)
                row: dict[str, object] = {
                    "nx": nx,
                    "ny": ny,
                    "L": float(nx),
                    "a_lat_over_L": 1.0 / nx,
                    "crack_half_length": crack_half,
                    "crack_half_over_L": crack_half / nx,
                    "R": radius,
                    "R_over_L": R_over_L,
                    "transition_width": width,
                    "width_over_L": width / nx,
                    "contour_radius_layers": radius,
                    "k_o": ko,
                    "alpha_upper": alpha,
                }
                for comp in ("J_even", "J_odd", "J_total"):
                    row[f"passive_{comp}"] = passive_J[comp]
                    row[f"active_{comp}"] = active_J[comp]
                    row[f"excess_{comp}"] = active_J[comp] - passive_J[comp]
                raw_rows.append(row)

    return raw_rows, force_rows


def summarize_rows(
    out_dir: Path,
    config: GaugeConvergenceConfig,
    raw_rows: list[dict[str, object]],
    force_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Write merged data, finite-size fits and the A/B/C diagnosis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "gauge_convergence_raw.csv", raw_rows)
    _write_csv(out_dir / "gauge_force_reproduction.csv", force_rows)

    grouped: dict[tuple[int, float, float], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(int(row["nx"]), float(row["k_o"]), float(row["R_over_L"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (nx, ko, R_over_L), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: float(r["alpha_upper"]))
        sym = min(rows, key=lambda r: abs(float(r["alpha_upper"]) - 0.5))
        active_values = np.array([float(r["active_J_total"]) for r in rows])
        passive_values = np.array([float(r["passive_J_total"]) for r in rows])
        excess_values = np.array([float(r["excess_J_total"]) for r in rows])
        active_span = float(np.ptp(active_values))
        passive_span = float(np.ptp(passive_values))
        excess_span = float(np.ptp(excess_values))
        active_scale = max(abs(float(sym["active_J_total"])), 1.0e-30)
        passive_scale = max(abs(float(sym["passive_J_total"])), 1.0e-30)
        excess_scale = max(abs(float(sym["excess_J_total"])), abs(ko) * active_scale, 1.0e-30)
        summary_rows.append({
            "nx": nx,
            "ny": int(float(sym["ny"])),
            "a_lat_over_L": 1.0 / nx,
            "k_o": ko,
            "R_over_L": R_over_L,
            "R_layers": float(sym["R"]),
            "active_symmetric_J": float(sym["active_J_total"]),
            "passive_symmetric_J": float(sym["passive_J_total"]),
            "excess_symmetric_J": float(sym["excess_J_total"]),
            "active_absolute_gauge_span": active_span,
            "passive_absolute_gauge_span": passive_span,
            "excess_absolute_gauge_span": excess_span,
            "active_relative_gauge_span": active_span / active_scale,
            "passive_relative_gauge_span": passive_span / passive_scale,
            "excess_relative_gauge_span": excess_span / excess_scale,
            "active_min_J": float(np.min(active_values)),
            "active_max_J": float(np.max(active_values)),
            "excess_min_J": float(np.min(excess_values)),
            "excess_max_J": float(np.max(excess_values)),
        })
    _write_csv(out_dir / "gauge_convergence_summary.csv", summary_rows)

    fit_rows: list[dict[str, object]] = []
    for ko in config.ko_values:
        for R_over_L in config.radius_over_nx:
            rows = [r for r in summary_rows if math.isclose(float(r["k_o"]), ko) and math.isclose(float(r["R_over_L"]), R_over_L)]
            rows.sort(key=lambda r: int(r["nx"]))
            x = np.array([float(r["a_lat_over_L"]) for r in rows])
            active_abs = _power_fit(x, np.array([float(r["active_absolute_gauge_span"]) for r in rows]))
            active_rel = _power_fit(x, np.array([float(r["active_relative_gauge_span"]) for r in rows]))
            excess_rel = _power_fit(x, np.array([float(r["excess_relative_gauge_span"]) for r in rows]))
            fit_rows.append({
                "k_o": ko,
                "R_over_L": R_over_L,
                "active_absolute_prefactor": active_abs["prefactor"],
                "active_absolute_exponent": active_abs["exponent"],
                "active_absolute_r2_log": active_abs["r2_log"],
                "active_relative_prefactor": active_rel["prefactor"],
                "active_relative_exponent": active_rel["exponent"],
                "active_relative_r2_log": active_rel["r2_log"],
                "excess_relative_prefactor": excess_rel["prefactor"],
                "excess_relative_exponent": excess_rel["exponent"],
                "excess_relative_r2_log": excess_rel["r2_log"],
            })
    _write_csv(out_dir / "gauge_convergence_fits.csv", fit_rows)

    classification, rationale = _classify(summary_rows, fit_rows)
    force_max = max(
        max(float(r["passive_force_reproduction_error"]), float(r["active_force_reproduction_error"]))
        for r in force_rows
    )
    target_fit = next(
        r for r in fit_rows
        if math.isclose(float(r["k_o"]), 0.15) and math.isclose(float(r["R_over_L"]), 0.10)
    )
    target_rows = sorted(
        [r for r in summary_rows if math.isclose(float(r["k_o"]), 0.15) and math.isclose(float(r["R_over_L"]), 0.10)],
        key=lambda r: int(r["nx"]),
    )
    summary = {
        "config": asdict(config),
        "continuum_limit_classification": classification,
        "classification_rationale": rationale,
        "classification_scope": "finite-size evidence for the tested geometrically similar family; not a continuum theorem",
        "target_case": {"k_o": 0.15, "R_over_L": 0.10},
        "target_active_relative_spans": [
            {"nx": int(r["nx"]), "a_lat_over_L": float(r["a_lat_over_L"]), "span": float(r["active_relative_gauge_span"])}
            for r in target_rows
        ],
        "target_active_relative_fit_exponent": float(target_fit["active_relative_exponent"]),
        "target_active_relative_fit_r2_log": float(target_fit["active_relative_r2_log"]),
        "target_excess_relative_fit_exponent": float(target_fit["excess_relative_exponent"]),
        "max_force_reproduction_error": force_max,
        "all_flux_values_finite": bool(all(np.isfinite(float(r["active_J_total"])) and np.isfinite(float(r["passive_J_total"])) for r in raw_rows)),
    }
    summary["pass"] = bool(
        summary["all_flux_values_finite"]
        and summary["max_force_reproduction_error"] < 1.0e-11
        and classification in {"A", "B", "C"}
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_analysis(out_dir: Path, config: GaugeConvergenceConfig = GaugeConvergenceConfig()) -> dict[str, object]:
    raw_rows: list[dict[str, object]] = []
    force_rows: list[dict[str, object]] = []
    for nx in config.nx_values:
        rr, fr = compute_size_rows(nx, config)
        raw_rows.extend(rr)
        force_rows.extend(fr)
        gc.collect()
    return summarize_rows(out_dir, config, raw_rows, force_rows)


if __name__ == "__main__":
    result = run_analysis(PACKAGE_ROOT / "data" / "gauge_convergence_results")
    print(json.dumps(result, indent=2))
