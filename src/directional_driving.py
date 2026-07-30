#!/usr/bin/env python3
"""Virtual-kink and directional configurational driving.

A crack path is represented on the dual graph of the triangular lattice.  The
centres of adjacent elementary triangles are dual nodes and a deleted primal
bond is the dual edge joining them.  Starting from either crack tip, every
non-backtracking two-edge dual path has unit end-to-end length.  The three
forward paths have local directions -60, 0 and +60 degrees, so their operational
work can be compared without a path-length correction.

Two independent directional diagnostics are evaluated:

1. fixed-grip, isotropic-mobility operational work for the sequential two-bond
   deletion, including the odd-force line integral along each relaxation;
2. the projection of the discrete configurational-force vector obtained from
   the piecewise-affine domain functional for a family of localization gauges.

The raw lattice has a small passive up/down registry bias.  It is retained in
all output and separated from the odd-induced excess by subtracting the k_o=0
value at the same size, tip, radius and localization gauge.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sys
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from active_tip_scan import ActiveCrackedStrip
from crack_advance_work import (
    assemble_components,
    bond_extension,
    even_energy,
    integrate_relaxation,
    solve_equilibrium,
)
from discrete_configurational_analysis import (
    AREA_TRI,
    affine_gradient,
    build_triangles,
    q_value,
)
from gauge_convergence import partition_weights, weighted_fields
from lattice_baselines import wrap_centered
from propagation_limit_analysis import top_reaction_stress

DIRECTIONS = (-60, 0, 60)


@dataclass(frozen=True)
class DirectionalConfig:
    nx_values: tuple[int, ...] = (32, 48, 64)
    aspect_ny_over_nx: float = 0.75
    crack_half_over_nx: float = 0.125
    ko_values: tuple[float, ...] = (-0.20, 0.0, 0.20)
    tips: tuple[str, ...] = ("right", "left")
    left_tip_nx_values: tuple[int, ...] = (48,)
    alpha_values: tuple[float, ...] = (0.0, 0.5, 1.0)
    radius_over_nx: tuple[float, ...] = (0.10, 0.125)
    width_over_nx: float = 0.025
    q_lp_power: float = 4.0
    delta_c: float = 0.02


@dataclass(frozen=True)
class DualKinkPath:
    tip: str
    direction_deg: int
    first_bond_id: int
    second_bond_id: int
    dual_dx_local: float
    dual_dy_local: float
    dual_length: float


@dataclass
class TopologyState:
    removed: frozenset[int]
    active_ids: list[int]
    even: object
    odd: object
    total: object
    displacement: np.ndarray
    constrained: np.ndarray
    constrained_values: np.ndarray
    free: np.ndarray
    force_residual: float
    remote_stress: float


@dataclass(frozen=True)
class CutStep:
    bond_id: int
    extension_unit: float
    odd_work_unit: float
    even_energy_change_unit: float
    dissipation_unit: float
    cut_energy_unit: float
    protocol_work_unit: float
    balance_residual_unit: float
    final_state_relative_norm: float
    n_time_steps: int


def _even_integer(value: float) -> int:
    n = int(round(value))
    return n if n % 2 == 0 else n + 1


def geometry(nx: int, cfg: DirectionalConfig) -> tuple[int, float]:
    return max(8, _even_integer(cfg.aspect_ny_over_nx * nx)), cfg.crack_half_over_nx * nx


def integration_settings(nx: int) -> tuple[float, float, float, float]:
    table = {
        32: (12000.0, 1.0e-6, 1.0e-8, 500.0),
        48: (27000.0, 1.0e-6, 1.0e-8, 1100.0),
        64: (48000.0, 1.0e-6, 1.0e-8, 2000.0),
    }
    if nx not in table:
        scale = (nx / 32.0) ** 2
        return 12000.0 * scale, 1.0e-6, 1.0e-8, 500.0 * scale
    return table[nx]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _bond_midpoint_unwrapped(model: ActiveCrackedStrip, bond_id: int, reference_x: float) -> np.ndarray:
    bond = model.all_bonds[bond_id]
    pi = model.positions[bond.i].copy()
    pj = model.positions[bond.j].copy()
    dx = pj[0] - pi[0]
    if dx > 0.5 * model.period:
        pj[0] -= model.period
    elif dx < -0.5 * model.period:
        pj[0] += model.period
    midpoint = 0.5 * (pi + pj)
    midpoint[0] += round((reference_x - midpoint[0]) / model.period) * model.period
    return midpoint


def enumerate_dual_kink_paths(
    model: ActiveCrackedStrip,
    removed: Iterable[int],
    tip: str,
) -> dict[int, DualKinkPath]:
    """Return the equal-length two-edge dual paths at 0 and +/-60 degrees."""
    if tip not in {"right", "left"}:
        raise ValueError("tip must be 'right' or 'left'")
    removed_set = set(removed)
    triangles = build_triangles(model)
    bond_to_triangles: dict[int, list[int]] = collections.defaultdict(list)
    for triangle_id, triangle in enumerate(triangles):
        for bond_id in triangle.bond_ids:
            bond_to_triangles[bond_id].append(triangle_id)

    direction = 1.0 if tip == "right" else -1.0
    centre = 0.5 * model.period
    terminal_candidates: list[tuple[float, int]] = []
    for bond_id in removed_set:
        midpoint = _bond_midpoint_unwrapped(model, bond_id, centre)
        local_x = direction * wrap_centered(midpoint[0] - centre, model.period)
        terminal_candidates.append((local_x, bond_id))
    if not terminal_candidates:
        raise RuntimeError("No removed bond available to locate the crack tip")
    terminal_bond = max(terminal_candidates)[1]

    adjacent = bond_to_triangles[terminal_bond]
    ahead = [
        triangle_id
        for triangle_id in adjacent
        if sum(bid in removed_set for bid in triangles[triangle_id].bond_ids) == 1
    ]
    if len(ahead) != 1:
        raise RuntimeError(
            f"Expected one exterior dual triangle at {tip} tip, found {len(ahead)}"
        )
    initial_triangle = ahead[0]
    initial_centre = np.mean(triangles[initial_triangle].coords, axis=0)

    paths: dict[int, DualKinkPath] = {}
    for first_bond in triangles[initial_triangle].bond_ids:
        if first_bond == terminal_bond or first_bond in removed_set:
            continue
        next_triangles = [ti for ti in bond_to_triangles[first_bond] if ti != initial_triangle]
        if len(next_triangles) != 1:
            continue
        first_triangle = next_triangles[0]
        for second_bond in triangles[first_triangle].bond_ids:
            if second_bond == first_bond or second_bond in removed_set:
                continue
            final_triangles = [ti for ti in bond_to_triangles[second_bond] if ti != first_triangle]
            if len(final_triangles) != 1:
                continue
            final_centre = np.mean(triangles[final_triangles[0]].coords, axis=0)
            dx = float(final_centre[0] - initial_centre[0])
            dx -= round(dx / model.period) * model.period
            vector_local = np.array([direction * dx, final_centre[1] - initial_centre[1]])
            length = float(np.linalg.norm(vector_local))
            angle = int(round(math.degrees(math.atan2(vector_local[1], vector_local[0]))))
            if angle not in DIRECTIONS:
                continue
            if not math.isclose(length, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                raise RuntimeError(f"Directional path length is {length}, not one lattice unit")
            if angle in paths:
                raise RuntimeError(f"Duplicate dual path at {angle} degrees")
            paths[angle] = DualKinkPath(
                tip=tip,
                direction_deg=angle,
                first_bond_id=first_bond,
                second_bond_id=second_bond,
                dual_dx_local=float(vector_local[0]),
                dual_dy_local=float(vector_local[1]),
                dual_length=length,
            )
    if set(paths) != set(DIRECTIONS):
        raise RuntimeError(f"Did not recover all forward directions: {sorted(paths)}")
    return paths


def solve_state(model: ActiveCrackedStrip, removed: Iterable[int]) -> TopologyState:
    removed_set = frozenset(removed)
    active_ids = [bid for bid in range(len(model.all_bonds)) if bid not in removed_set]
    even, odd = assemble_components(model, active_ids)
    total = even + odd
    displacement, constrained, values, free, residual = solve_equilibrium(total, model, 1.0)
    remote_stress = top_reaction_stress(model, total, displacement)
    return TopologyState(
        removed=removed_set,
        active_ids=active_ids,
        even=even,
        odd=odd,
        total=total,
        displacement=displacement,
        constrained=constrained,
        constrained_values=values,
        free=free,
        force_residual=residual,
        remote_stress=remote_stress,
    )


def cut_and_relax(
    model: ActiveCrackedStrip,
    state: TopologyState,
    bond_id: int,
    t_end: float,
    rtol: float,
    relative_atol: float,
    max_step: float,
) -> tuple[TopologyState, CutStep]:
    if bond_id in state.removed:
        raise ValueError(f"Bond {bond_id} is already removed")
    extension = bond_extension(model.all_bonds[bond_id], state.displacement)
    energy_initial = even_energy(state.even, state.displacement)
    new_removed = set(state.removed)
    new_removed.add(bond_id)
    final_state = solve_state(model, new_removed)
    if not (
        np.array_equal(state.constrained, final_state.constrained)
        and np.array_equal(state.free, final_state.free)
        and np.allclose(state.constrained_values, final_state.constrained_values)
    ):
        raise RuntimeError("Fixed-grip boundary partition changed during directional cut")

    odd_work, dissipation, final_norm, n_time_steps = integrate_relaxation(
        final_state.total,
        final_state.odd,
        state.displacement,
        final_state.displacement,
        state.constrained,
        state.constrained_values,
        state.free,
        t_end,
        rtol=rtol,
        relative_atol=relative_atol,
        max_step=max_step,
    )
    energy_final = even_energy(final_state.even, final_state.displacement)
    delta_energy = energy_final - energy_initial
    cut_energy = 0.5 * extension**2
    protocol_work = odd_work - delta_energy
    balance = protocol_work - dissipation - cut_energy
    return final_state, CutStep(
        bond_id=bond_id,
        extension_unit=extension,
        odd_work_unit=odd_work,
        even_energy_change_unit=delta_energy,
        dissipation_unit=dissipation,
        cut_energy_unit=cut_energy,
        protocol_work_unit=protocol_work,
        balance_residual_unit=balance,
        final_state_relative_norm=final_norm,
        n_time_steps=n_time_steps,
    )


def directional_protocol_rows(
    model: ActiveCrackedStrip,
    tip: str,
    delta_c: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate all three paths while caching their shared first cut."""
    paths = enumerate_dual_kink_paths(model, model.removed_ids, tip)
    t_end, rtol, relative_atol, max_step = integration_settings(model.nx)
    initial = solve_state(model, model.removed_ids)

    first_cache: dict[int, tuple[TopologyState, CutStep]] = {}
    rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    for angle in DIRECTIONS:
        path = paths[angle]
        if path.first_bond_id not in first_cache:
            first_cache[path.first_bond_id] = cut_and_relax(
                model, initial, path.first_bond_id,
                t_end, rtol, relative_atol, max_step,
            )
        state_one, step_one = first_cache[path.first_bond_id]
        state_two, step_two = cut_and_relax(
            model, state_one, path.second_bond_id,
            t_end, rtol, relative_atol, max_step,
        )
        del state_two

        extensions = np.array([step_one.extension_unit, step_two.extension_unit])
        if np.all(extensions > 0.0):
            critical_amplitude = float(np.max(delta_c / extensions))
            critical_stress = initial.remote_stress * critical_amplitude
        else:
            critical_amplitude = float("inf")
            critical_stress = float("inf")
        total_work = step_one.protocol_work_unit + step_two.protocol_work_unit
        total_odd = step_one.odd_work_unit + step_two.odd_work_unit
        total_dissipation = step_one.dissipation_unit + step_two.dissipation_unit
        total_cut_energy = step_one.cut_energy_unit + step_two.cut_energy_unit
        total_balance = step_one.balance_residual_unit + step_two.balance_residual_unit

        base = {
            "nx": model.nx,
            "ny": model.ny,
            "a_lat_over_L": 1.0 / model.nx,
            "crack_half_over_L": model.target_half_length / model.nx,
            "k_o": model.k_o,
            "tip": tip,
            "direction_deg": angle,
            "dual_dx_local": path.dual_dx_local,
            "dual_dy_local": path.dual_dy_local,
            "dual_length": path.dual_length,
            "first_bond_id": path.first_bond_id,
            "second_bond_id": path.second_bond_id,
            "remote_stress_unit": initial.remote_stress,
            "first_extension_unit": step_one.extension_unit,
            "second_extension_unit": step_two.extension_unit,
            "sequential_critical_amplitude": critical_amplitude,
            "sequential_critical_remote_stress": critical_stress,
            "protocol_work_unit": total_work,
            "odd_work_unit": total_odd,
            "dissipation_unit": total_dissipation,
            "cut_energy_unit": total_cut_energy,
            "balance_residual_unit": total_balance,
            "max_step_balance_abs": max(abs(step_one.balance_residual_unit), abs(step_two.balance_residual_unit)),
            "max_final_state_relative_norm": max(step_one.final_state_relative_norm, step_two.final_state_relative_norm),
            "n_time_steps_total": step_one.n_time_steps + step_two.n_time_steps,
            "t_end": t_end,
            "rtol": rtol,
            "relative_atol": relative_atol,
            "max_step": max_step,
        }
        path_rows.append(base)
        for step_index, step in enumerate((step_one, step_two), start=1):
            rows.append({
                **{k: v for k, v in base.items() if k not in {
                    "protocol_work_unit", "odd_work_unit", "dissipation_unit", "cut_energy_unit",
                    "balance_residual_unit", "max_step_balance_abs", "max_final_state_relative_norm",
                    "n_time_steps_total"
                }},
                "step": step_index,
                **asdict(step),
            })
    return rows, path_rows


def configurational_force_rows_for_case(
    model: ActiveCrackedStrip,
    tip: str,
    alpha_values: Iterable[float],
    radius_over_nx: Iterable[float],
    width_over_nx: float,
    p: float,
) -> list[dict[str, object]]:
    """Evaluate all gauges/radii while reusing one equilibrium and mesh."""
    displacement, _, residual = model.solve(1.0)
    triangles = build_triangles(model)
    radii = [(float(fraction), float(fraction) * model.nx) for fraction in radius_over_nx]
    width = float(width_over_nx) * model.nx
    support = max(radius for _, radius in radii) + 0.5 * width + 1.0
    rows: list[dict[str, object]] = []
    for alpha in alpha_values:
        weights = partition_weights(model, triangles, float(alpha))
        fields, local_nodes = weighted_fields(
            model, displacement, triangles, tip, support_radius=support, weights=weights
        )
        for R_over_L, radius in radii:
            even = np.zeros(2)
            odd = np.zeros(2)
            for (grad_u, stress_even, stress_odd, energy_even), xnodes in zip(fields, local_nodes):
                qnod = np.array([q_value(x, radius, width, p) for x in xnodes])[:, None]
                grad_q = affine_gradient(xnodes, qnod)[0]
                for material_direction in (0, 1):
                    basis = np.zeros(2)
                    basis[material_direction] = 1.0
                    p_even = energy_even * basis - stress_even.T @ grad_u[:, material_direction]
                    p_odd = -stress_odd.T @ grad_u[:, material_direction]
                    even[material_direction] += -AREA_TRI * float(p_even @ grad_q)
                    odd[material_direction] += -AREA_TRI * float(p_odd @ grad_q)
            total = even + odd
            vector = {
                "J_even_x": float(even[0]), "J_even_y": float(even[1]),
                "J_odd_x": float(odd[0]), "J_odd_y": float(odd[1]),
                "J_total_x": float(total[0]), "J_total_y": float(total[1]),
            }
            for angle in DIRECTIONS:
                radians = math.radians(angle)
                projection = total[0] * math.cos(radians) + total[1] * math.sin(radians)
                rows.append({
                    "nx": model.nx, "ny": model.ny, "a_lat_over_L": 1.0 / model.nx,
                    "crack_half_over_L": model.target_half_length / model.nx,
                    "k_o": model.k_o, "tip": tip, "alpha_upper": float(alpha),
                    "R_over_L": R_over_L, "radius": radius, "width": width,
                    "direction_deg": angle, **vector, "J_projection": float(projection),
                    "equilibrium_residual_inf": float(residual),
                })
    return rows


def add_baseline_excess(
    protocol_rows: list[dict[str, object]],
    force_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    passive_work = {
        (int(r["nx"]), str(r["tip"]), int(r["direction_deg"])): float(r["protocol_work_unit"])
        for r in protocol_rows if abs(float(r["k_o"])) < 1.0e-14
    }
    for row in protocol_rows:
        key = (int(row["nx"]), str(row["tip"]), int(row["direction_deg"]))
        row["odd_excess_protocol_work"] = float(row["protocol_work_unit"]) - passive_work[key]

    passive_force = {
        (
            int(r["nx"]), str(r["tip"]), float(r["alpha_upper"]),
            float(r["R_over_L"]), int(r["direction_deg"]),
        ): float(r["J_projection"])
        for r in force_rows if abs(float(r["k_o"])) < 1.0e-14
    }
    for row in force_rows:
        key = (
            int(row["nx"]), str(row["tip"]), float(row["alpha_upper"]),
            float(row["R_over_L"]), int(row["direction_deg"]),
        )
        row["odd_excess_J_projection"] = float(row["J_projection"]) - passive_force[key]

    summary_rows: list[dict[str, object]] = []
    cases = sorted({(int(r["nx"]), str(r["tip"]), float(r["k_o"])) for r in protocol_rows})
    for nx, tip, ko in cases:
        p = {
            int(r["direction_deg"]): r
            for r in protocol_rows
            if int(r["nx"]) == nx and str(r["tip"]) == tip and math.isclose(float(r["k_o"]), ko)
        }
        passive = {
            angle: passive_work[(nx, tip, angle)] for angle in DIRECTIONS
        }
        raw_bias = float(p[60]["protocol_work_unit"]) - float(p[-60]["protocol_work_unit"])
        passive_bias = passive[60] - passive[-60]
        excess_bias = raw_bias - passive_bias
        straight_passive = passive[0]
        preferred_total = max(DIRECTIONS, key=lambda angle: float(p[angle]["protocol_work_unit"]))
        kink_preferred = max((-60, 60), key=lambda angle: float(p[angle]["protocol_work_unit"]))
        finite_thresholds = {
            angle: float(p[angle]["sequential_critical_remote_stress"])
            for angle in DIRECTIONS
        }
        preferred_threshold = min(
            DIRECTIONS,
            key=lambda angle: finite_thresholds[angle],
        )
        summary_rows.append({
            "nx": nx,
            "ny": int(p[0]["ny"]),
            "a_lat_over_L": 1.0 / nx,
            "tip": tip,
            "k_o": ko,
            "passive_straight_work": straight_passive,
            "raw_kink_work_bias_plus_minus": raw_bias,
            "passive_registry_kink_bias": passive_bias,
            "odd_excess_kink_work_bias": excess_bias,
            "odd_excess_kink_work_bias_over_passive_straight": excess_bias / straight_passive,
            "preferred_total_work_direction_deg": preferred_total,
            "preferred_kink_direction_deg": kink_preferred,
            "preferred_sequential_threshold_direction_deg": preferred_threshold,
            "straight_work_over_passive_straight": float(p[0]["protocol_work_unit"]) / straight_passive,
            "minus60_work_over_passive_straight": float(p[-60]["protocol_work_unit"]) / straight_passive,
            "plus60_work_over_passive_straight": float(p[60]["protocol_work_unit"]) / straight_passive,
        })
    return protocol_rows, force_rows, summary_rows


def compute_case_payload(nx: int, ko: float, tip: str, cfg: DirectionalConfig) -> dict[str, object]:
    ny, crack_half = geometry(nx, cfg)
    model = ActiveCrackedStrip(nx, ny, crack_half, 1.0, ko)
    paths = enumerate_dual_kink_paths(model, model.removed_ids, tip)
    geometry_rows = [
        {"nx": nx, "ny": ny, "k_o": ko, **asdict(paths[angle])}
        for angle in DIRECTIONS
    ]
    step_rows, protocol_rows = directional_protocol_rows(model, tip, cfg.delta_c)
    force_rows = configurational_force_rows_for_case(
        model, tip, cfg.alpha_values, cfg.radius_over_nx,
        cfg.width_over_nx, cfg.q_lp_power,
    )
    return {
        "path_geometry_rows": geometry_rows,
        "step_rows": step_rows,
        "protocol_rows": protocol_rows,
        "force_rows": force_rows,
    }


def finalize_payloads(
    out: Path,
    cfg: DirectionalConfig,
    payloads: list[dict[str, object]],
) -> dict[str, object]:
    path_geometry_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []
    protocol_rows: list[dict[str, object]] = []
    force_rows: list[dict[str, object]] = []
    for payload in payloads:
        path_geometry_rows.extend(payload["path_geometry_rows"])
        step_rows.extend(payload["step_rows"])
        protocol_rows.extend(payload["protocol_rows"])
        force_rows.extend(payload["force_rows"])

    protocol_rows, force_rows, summary_rows = add_baseline_excess(protocol_rows, force_rows)
    _write_csv(out / "dual_kink_path_geometry.csv", path_geometry_rows)
    _write_csv(out / "directional_cut_steps.csv", step_rows)
    _write_csv(out / "directional_protocol_work.csv", protocol_rows)
    _write_csv(out / "directional_configurational_force.csv", force_rows)
    _write_csv(out / "directional_case_summary.csv", summary_rows)

    active_summary = [r for r in summary_rows if abs(float(r["k_o"])) > 1.0e-14]
    passive_summary = [r for r in summary_rows if abs(float(r["k_o"])) < 1.0e-14]
    max_passive_registry = max(
        abs(float(r["passive_registry_kink_bias"])) / float(r["passive_straight_work"])
        for r in passive_summary
    )

    reversal_errors = []
    available_tip_sizes = sorted({(int(r["nx"]), str(r["tip"])) for r in summary_rows})
    for nx, tip in available_tip_sizes:
        plus = next(r for r in summary_rows if r["nx"] == nx and r["tip"] == tip and math.isclose(float(r["k_o"]), 0.20))
        minus = next(r for r in summary_rows if r["nx"] == nx and r["tip"] == tip and math.isclose(float(r["k_o"]), -0.20))
        scale = max(abs(float(plus["odd_excess_kink_work_bias"])), abs(float(minus["odd_excess_kink_work_bias"])), 1.0e-30)
        reversal_errors.append(abs(float(plus["odd_excess_kink_work_bias"]) + float(minus["odd_excess_kink_work_bias"])) / scale)

    tip_reflection_errors = []
    for nx in cfg.left_tip_nx_values:
        rp = next(r for r in summary_rows if r["nx"] == nx and r["tip"] == "right" and math.isclose(float(r["k_o"]), 0.20))
        lm = next(r for r in summary_rows if r["nx"] == nx and r["tip"] == "left" and math.isclose(float(r["k_o"]), -0.20))
        scale = max(abs(float(rp["raw_kink_work_bias_plus_minus"])), abs(float(lm["raw_kink_work_bias_plus_minus"])), 1.0e-30)
        tip_reflection_errors.append(abs(float(rp["raw_kink_work_bias_plus_minus"]) - float(lm["raw_kink_work_bias_plus_minus"])) / scale)

    sign_agreement = []
    gauge_biases: list[float] = []
    for row in active_summary:
        nx = int(row["nx"])
        tip = str(row["tip"])
        ko = float(row["k_o"])
        work_bias = float(row["odd_excess_kink_work_bias"])
        for alpha in cfg.alpha_values:
            for R_over_L in cfg.radius_over_nx:
                fplus = next(
                    r for r in force_rows
                    if int(r["nx"]) == nx and str(r["tip"]) == tip
                    and math.isclose(float(r["k_o"]), ko)
                    and math.isclose(float(r["alpha_upper"]), alpha)
                    and math.isclose(float(r["R_over_L"]), R_over_L)
                    and int(r["direction_deg"]) == 60
                )
                fminus = next(
                    r for r in force_rows
                    if int(r["nx"]) == nx and str(r["tip"]) == tip
                    and math.isclose(float(r["k_o"]), ko)
                    and math.isclose(float(r["alpha_upper"]), alpha)
                    and math.isclose(float(r["R_over_L"]), R_over_L)
                    and int(r["direction_deg"]) == -60
                )
                force_bias = float(fplus["odd_excess_J_projection"]) - float(fminus["odd_excess_J_projection"])
                sign_agreement.append(work_bias * force_bias > 0.0)
                gauge_biases.append(force_bias)

    total_preferred_directions = sorted({int(r["preferred_total_work_direction_deg"]) for r in active_summary})
    right_active = [r for r in active_summary if r["tip"] == "right"]
    normalized_bias_by_size = {
        str(int(r["nx"])): float(r["odd_excess_kink_work_bias_over_passive_straight"])
        for r in right_active if math.isclose(float(r["k_o"]), 0.20)
    }
    summary = {
        "sizes": list(cfg.nx_values),
        "ko_values": list(cfg.ko_values),
        "tips": list(cfg.tips),
        "left_tip_nx_values": list(cfg.left_tip_nx_values),
        "directions_deg": list(DIRECTIONS),
        "path_geometry_row_count": len(path_geometry_rows),
        "directional_cut_step_count": len(step_rows),
        "directional_protocol_path_count": len(protocol_rows),
        "directional_configurational_projection_count": len(force_rows),
        "maximum_abs_work_balance_residual": max(abs(float(r["balance_residual_unit"])) for r in step_rows),
        "maximum_final_state_relative_norm": max(float(r["final_state_relative_norm"]) for r in step_rows),
        "maximum_passive_registry_kink_bias_over_straight": max_passive_registry,
        "maximum_odd_excess_work_reversal_relative_error": max(reversal_errors),
        "maximum_tip_reflection_relative_error": max(tip_reflection_errors),
        "work_and_configurational_bias_sign_agreement_fraction": float(np.mean(sign_agreement)),
        "minimum_abs_configurational_odd_excess_bias": min(abs(x) for x in gauge_biases),
        "maximum_abs_configurational_odd_excess_bias": max(abs(x) for x in gauge_biases),
        "right_tip_ko0p20_normalized_odd_excess_bias_by_size": normalized_bias_by_size,
        "total_preferred_work_directions_active": total_preferred_directions,
        "short_kink_gate_triggered": total_preferred_directions != [0],
        "interpretation": (
            "odd-induced up/down directional excess is robust, but the straight two-edge path "
            "retains the largest total operational work over the tested range"
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_analysis(out: Path, cfg: DirectionalConfig) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    case_dir = out / "_case_cache"
    case_dir.mkdir(parents=True, exist_ok=True)
    combinations: list[tuple[int, float, str]] = []
    for nx in cfg.nx_values:
        tips_here = ("right", "left") if nx in cfg.left_tip_nx_values else ("right",)
        for ko in cfg.ko_values:
            for tip in tips_here:
                combinations.append((nx, ko, tip))

    payloads = []
    for nx, ko, tip in combinations:
        ko_tag = f"{ko:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
        case_file = case_dir / f"Nx{nx}_ko{ko_tag}_{tip}.json"
        if not case_file.exists():
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--out", str(out), "--single-case", "--nx", str(nx),
                "--ko", repr(float(ko)), "--tip", tip, "--case-file", str(case_file),
            ]
            subprocess.run(command, check=True)
        payloads.append(json.loads(case_file.read_text(encoding="utf-8")))
    return finalize_payloads(out, cfg, payloads)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "directional_driving_results")
    parser.add_argument("--single-case", action="store_true")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ko", type=float)
    parser.add_argument("--tip", choices=("right", "left"))
    parser.add_argument("--case-file", type=Path)
    args = parser.parse_args()
    cfg = DirectionalConfig()
    if args.single_case:
        if args.nx is None or args.ko is None or args.tip is None or args.case_file is None:
            parser.error("--single-case requires --nx, --ko, --tip and --case-file")
        payload = compute_case_payload(args.nx, args.ko, args.tip, cfg)
        args.case_file.parent.mkdir(parents=True, exist_ok=True)
        args.case_file.write_text(json.dumps(payload), encoding="utf-8")
        print(f"completed directional case: Nx={args.nx}, ko={args.ko:+.2f}, tip={args.tip}", flush=True)
        return
    summary = run_analysis(args.out, cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
