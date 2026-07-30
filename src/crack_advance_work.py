#!/usr/bin/env python3
""" virtual bond-cut / overdamped-relaxation protocol.

For a fixed remote Mode-I stress, the lattice is first relaxed with a stationary
crack. One selected crack-tip bond is then deleted instantaneously while the
boundary displacement is held fixed, and the remaining free degrees of freedom
relax according to

    eta u_dot = F_even + F_odd = -K u.

The script records the odd-force line work, the even elastic-energy change, and
the viscous dissipation. For the relaxation segment,

    W_odd - Delta U_even^relax - D_eta = 0.

Including the instantaneous loss E_cut of the deleted conservative bond gives

    W_odd - Delta U_even(total) - D_eta - E_cut = 0.

The protocol crack-driving work (per one broken lattice bond) is defined as

    A_P = W_odd - Delta U_even(total)

under fixed-displacement relaxation. It partitions into E_cut + D_eta for this
specific instantaneous-cut protocol. A passive initiation event calibrates the
protocol resistance Gc_P, allowing A_P/Gc_P to be compared with the independent
local bond-extension fracture rule.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lattice_baselines import Bond, R90
from active_tip_scan import ActiveCrackedStrip


@dataclass
class ProtocolUnitResult:
    k_o: float
    tip: str
    remote_stress_unit: float
    candidate_extension_unit: float
    cut_bond_even_energy_unit: float
    initial_even_energy_unit: float
    after_cut_even_energy_unit: float
    final_even_energy_unit: float
    odd_work_unit: float
    viscous_dissipation_unit: float
    protocol_work_unit: float
    relaxation_balance_residual_unit: float
    full_balance_residual_unit: float
    free_force_residual_initial: float
    free_force_residual_final: float
    final_state_relative_norm: float
    n_time_steps: int
    t_end: float


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def assemble_components(
    model: ActiveCrackedStrip, active_ids: list[int]
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    rows: list[int] = []
    cols: list[int] = []
    even_data: list[float] = []
    odd_data: list[float] = []
    for bond_id in active_ids:
        bond = model.all_bonds[bond_id]
        tangent = R90 @ bond.n
        even_block = np.outer(model.k * bond.n, bond.n)
        odd_block = np.outer(-model.k_o * tangent, bond.n)
        for component_i in range(2):
            for component_j in range(2):
                ii = 2 * bond.i + component_i
                ij = 2 * bond.i + component_j
                ji = 2 * bond.j + component_i
                jj = 2 * bond.j + component_j
                rows.extend((ii, ji, ii, ji))
                cols.extend((ij, jj, jj, ij))
                even_value = float(even_block[component_i, component_j])
                odd_value = float(odd_block[component_i, component_j])
                even_data.extend((even_value, even_value, -even_value, -even_value))
                odd_data.extend((odd_value, odd_value, -odd_value, -odd_value))
    shape = (model.ndof, model.ndof)
    even = sparse.coo_matrix((even_data, (rows, cols)), shape=shape).tocsr()
    odd = sparse.coo_matrix((odd_data, (rows, cols)), shape=shape).tocsr()
    return even, odd


def solve_equilibrium(
    matrix: sparse.csr_matrix,
    model: ActiveCrackedStrip,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    constrained, values = model.constrained_dofs(delta)
    free_mask = np.ones(model.ndof, dtype=bool)
    free_mask[constrained] = False
    free = np.arange(model.ndof)[free_mask]
    matrix_ff = matrix[free][:, free]
    rhs = -(matrix[free][:, constrained] @ values)
    free_values = spsolve(matrix_ff, rhs)
    if not np.all(np.isfinite(free_values)):
        raise RuntimeError("Equilibrium solve failed")
    displacement = np.zeros(model.ndof, dtype=float)
    displacement[constrained] = values
    displacement[free] = free_values
    residual = matrix @ displacement
    free_residual = float(np.max(np.abs(residual[free])))
    return displacement, constrained, values, free, free_residual


def even_energy(matrix_even: sparse.csr_matrix, displacement: np.ndarray) -> float:
    return 0.5 * float(displacement @ (matrix_even @ displacement))


def bond_extension(bond: Bond, displacement: np.ndarray) -> float:
    vector = displacement.reshape((-1, 2))
    return float((vector[bond.j] - vector[bond.i]) @ bond.n)


def integrate_relaxation(
    matrix_total: sparse.csr_matrix,
    matrix_odd: sparse.csr_matrix,
    initial: np.ndarray,
    final: np.ndarray,
    constrained: np.ndarray,
    constrained_values: np.ndarray,
    free: np.ndarray,
    t_end: float,
    rtol: float = 2.0e-8,
    relative_atol: float = 1.0e-10,
    max_step: float = np.inf,
) -> tuple[float, float, float, int]:
    matrix_ff = matrix_total[free][:, free]
    matrix_fc = matrix_total[free][:, constrained]
    odd_ff = matrix_odd[free][:, free]
    odd_fc = matrix_odd[free][:, constrained]
    initial_offset = initial[free] - final[free]
    scale = max(float(np.max(np.abs(initial_offset))), 1.0e-12)

    solution = solve_ivp(
        lambda _time, offset: -(matrix_ff @ offset),
        (0.0, t_end),
        initial_offset,
        method="BDF",
        jac=-matrix_ff,
        rtol=rtol,
        atol=relative_atol * scale,
        max_step=max_step,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    def forces(offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        free_state = final[free] + offset
        odd_force = -(odd_ff @ free_state + odd_fc @ constrained_values)
        total_force = -(matrix_ff @ free_state + matrix_fc @ constrained_values)
        return np.asarray(odd_force), np.asarray(total_force)

    odd_work = 0.0
    dissipation = 0.0
    odd_previous, total_previous = forces(solution.y[:, 0])
    for step in range(1, len(solution.t)):
        increment = solution.y[:, step] - solution.y[:, step - 1]
        odd_current, total_current = forces(solution.y[:, step])
        odd_work += 0.5 * float((odd_previous + odd_current) @ increment)
        dissipation += 0.5 * float((total_previous + total_current) @ increment)
        odd_previous = odd_current
        total_previous = total_current

    final_relative_norm = float(
        np.linalg.norm(solution.y[:, -1])
        / max(np.linalg.norm(initial_offset), 1.0e-30)
    )
    return odd_work, dissipation, final_relative_norm, len(solution.t)


def protocol_unit_result(
    nx: int,
    ny: int,
    crack_half_length: float,
    k_o: float,
    tip: str,
    t_end: float,
    rtol: float = 2.0e-8,
    relative_atol: float = 1.0e-10,
) -> ProtocolUnitResult:
    model = ActiveCrackedStrip(
        nx=nx,
        ny=ny,
        crack_half_length=crack_half_length,
        k=1.0,
        k_o=k_o,
    )
    if tip not in {"left", "right"}:
        raise ValueError("tip must be left or right")
    cut_id = model.tip_candidates()[tip]
    old_active_ids = [
        bond_id
        for bond_id in range(len(model.all_bonds))
        if bond_id not in model.removed_ids
    ]
    new_active_ids = [bond_id for bond_id in old_active_ids if bond_id != cut_id]

    even_old, odd_old = assemble_components(model, old_active_ids)
    even_new, odd_new = assemble_components(model, new_active_ids)
    total_old = even_old + odd_old
    total_new = even_new + odd_new

    initial, constrained, constrained_values, free, initial_residual = solve_equilibrium(
        total_old, model, delta=1.0
    )
    final, constrained_2, values_2, free_2, final_residual = solve_equilibrium(
        total_new, model, delta=1.0
    )
    if not (
        np.array_equal(constrained, constrained_2)
        and np.array_equal(free, free_2)
        and np.allclose(constrained_values, values_2)
    ):
        raise RuntimeError("Boundary partitions changed during virtual cut")

    odd_work, dissipation, final_relative_norm, n_steps = integrate_relaxation(
        matrix_total=total_new,
        matrix_odd=odd_new,
        initial=initial,
        final=final,
        constrained=constrained,
        constrained_values=constrained_values,
        free=free,
        t_end=t_end,
        rtol=rtol,
        relative_atol=relative_atol,
    )

    energy_initial = even_energy(even_old, initial)
    energy_after_cut = even_energy(even_new, initial)
    energy_final = even_energy(even_new, final)
    cut_extension = bond_extension(model.all_bonds[cut_id], initial)
    cut_energy = 0.5 * cut_extension**2
    relaxation_delta = energy_final - energy_after_cut
    total_delta = energy_final - energy_initial
    protocol_work = odd_work - total_delta
    relaxation_balance = odd_work - relaxation_delta - dissipation
    full_balance = odd_work - total_delta - dissipation - cut_energy

    residual_old = total_old @ initial
    top_y = np.array(
        [2 * model.node_id(i, model.ny - 1) + 1 for i in range(model.nx)],
        dtype=int,
    )
    reaction = float(np.sum(residual_old[top_y]))
    remote_stress = reaction / model.period

    return ProtocolUnitResult(
        k_o=float(k_o),
        tip=tip,
        remote_stress_unit=remote_stress,
        candidate_extension_unit=cut_extension,
        cut_bond_even_energy_unit=cut_energy,
        initial_even_energy_unit=energy_initial,
        after_cut_even_energy_unit=energy_after_cut,
        final_even_energy_unit=energy_final,
        odd_work_unit=odd_work,
        viscous_dissipation_unit=dissipation,
        protocol_work_unit=protocol_work,
        relaxation_balance_residual_unit=relaxation_balance,
        full_balance_residual_unit=full_balance,
        free_force_residual_initial=initial_residual,
        free_force_residual_final=final_residual,
        final_state_relative_norm=final_relative_norm,
        n_time_steps=n_steps,
        t_end=t_end,
    )


def scaled_row(
    unit: ProtocolUnitResult,
    target_remote_stress: float,
    delta_c: float,
    protocol_resistance: float,
) -> dict[str, object]:
    delta = target_remote_stress / unit.remote_stress_unit
    scale = delta**2
    extension = delta * unit.candidate_extension_unit
    return {
        **asdict(unit),
        "target_remote_stress": target_remote_stress,
        "applied_displacement": delta,
        "candidate_extension": extension,
        "candidate_extension_ratio": extension / delta_c,
        "cut_bond_even_energy": scale * unit.cut_bond_even_energy_unit,
        "odd_work": scale * unit.odd_work_unit,
        "viscous_dissipation": scale * unit.viscous_dissipation_unit,
        "total_even_energy_change": scale
        * (unit.final_even_energy_unit - unit.initial_even_energy_unit),
        "protocol_work": scale * unit.protocol_work_unit,
        "protocol_work_ratio": scale * unit.protocol_work_unit / protocol_resistance,
        "full_balance_residual": scale * unit.full_balance_residual_unit,
    }


def interpolate_crossing(x: np.ndarray, y: np.ndarray, level: float = 1.0) -> float:
    for index in range(len(x) - 1):
        if (y[index] - level) * (y[index + 1] - level) <= 0.0:
            if y[index + 1] == y[index]:
                return float(0.5 * (x[index] + x[index + 1]))
            fraction = (level - y[index]) / (y[index + 1] - y[index])
            return float(x[index] + fraction * (x[index + 1] - x[index]))
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("crack_advance_results"))
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=36)
    parser.add_argument("--crack-half-length", type=float, default=6.0)
    parser.add_argument("--t-end", type=float, default=70000.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    nx = args.nx
    ny = args.ny
    crack_half_length = args.crack_half_length
    delta_c = 0.02
    load_fraction = 0.90

    passive_model = ActiveCrackedStrip(
        nx=nx,
        ny=ny,
        crack_half_length=crack_half_length,
        k=1.0,
        k_o=0.0,
    )
    passive_diagnostics = passive_model.static_diagnostics(delta_c=delta_c)
    passive_initiation_stress = 0.5 * (
        passive_diagnostics["initiation_sigma_left"]
        + passive_diagnostics["initiation_sigma_right"]
    )
    passive_initiation_delta = 0.5 * (
        passive_diagnostics["initiation_delta_left"]
        + passive_diagnostics["initiation_delta_right"]
    )

    k_o_values = np.array(
        [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0,
          0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        dtype=float,
    )
    cache: dict[tuple[float, str], ProtocolUnitResult] = {}
    for k_o in k_o_values:
        cache[(float(k_o), "right")] = protocol_unit_result(
            nx=nx,
            ny=ny,
            crack_half_length=crack_half_length,
            k_o=float(k_o),
            tip="right",
            t_end=args.t_end,
        )

    passive_unit = cache[(0.0, "right")]
    protocol_resistance = (
        passive_initiation_delta**2 * passive_unit.protocol_work_unit
    )
    target_stress = load_fraction * passive_initiation_stress
    rows = [
        scaled_row(
            cache[(float(k_o), "right")],
            target_remote_stress=target_stress,
            delta_c=delta_c,
            protocol_resistance=protocol_resistance,
        )
        for k_o in k_o_values
    ]
    write_rows(args.out / "protocol_right_tip_sign_scan.csv", rows)

    # Independent mirror calculation at representative values.
    mirror_rows: list[dict[str, object]] = []
    for magnitude in (0.10, 0.20, 0.30):
        left_negative = protocol_unit_result(
            nx=nx,
            ny=ny,
            crack_half_length=crack_half_length,
            k_o=-magnitude,
            tip="left",
            t_end=args.t_end,
        )
        right_positive = cache[(magnitude, "right")]
        mirror_rows.append({
            "abs_k_o": magnitude,
            "right_positive_protocol_work_unit": right_positive.protocol_work_unit,
            "left_negative_protocol_work_unit": left_negative.protocol_work_unit,
            "protocol_work_mirror_abs_error": abs(
                right_positive.protocol_work_unit - left_negative.protocol_work_unit
            ),
            "right_positive_odd_work_unit": right_positive.odd_work_unit,
            "left_negative_odd_work_unit": left_negative.odd_work_unit,
            "odd_work_mirror_abs_error": abs(
                right_positive.odd_work_unit - left_negative.odd_work_unit
            ),
            "right_positive_extension_unit": right_positive.candidate_extension_unit,
            "left_negative_extension_unit": left_negative.candidate_extension_unit,
            "extension_mirror_abs_error": abs(
                right_positive.candidate_extension_unit
                - left_negative.candidate_extension_unit
            ),
        })
    write_rows(args.out / "protocol_chirality_mirror.csv", mirror_rows)

    ko = np.array([float(row["k_o"]) for row in rows])
    work_ratio = np.array([float(row["protocol_work_ratio"]) for row in rows])
    extension_ratio = np.array([float(row["candidate_extension_ratio"]) for row in rows])
    odd_work = np.array([float(row["odd_work"]) for row in rows])
    dissipation = np.array([float(row["viscous_dissipation"]) for row in rows])

    positive = ko >= -1.0e-14
    protocol_critical = interpolate_crossing(ko[positive], work_ratio[positive])
    bond_critical = interpolate_crossing(ko[positive], extension_ratio[positive])

    fig, axis = plt.subplots(figsize=(6.6, 4.5))
    axis.plot(ko, work_ratio, "o-", label=r"protocol work $A^{\mathcal{P}}/G_c^{\mathcal{P}}$")
    axis.plot(ko, extension_ratio, "s-", label=r"local bond criterion $\delta\ell/\delta_c$")
    axis.axhline(1.0, linestyle="--", linewidth=1.0)
    axis.set_xlabel(r"odd bond coefficient $k_o$")
    axis.set_ylabel("normalized initiation measure")
    axis.legend()
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "protocol_vs_local_initiation.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.6, 4.5))
    axis.plot(ko, odd_work, "o-", label=r"odd work $W_{\rm odd}$")
    axis.plot(ko, dissipation, "s-", label=r"viscous dissipation $D_\eta$")
    axis.set_xlabel(r"odd bond coefficient $k_o$")
    axis.set_ylabel("work per virtual bond advance")
    axis.legend()
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "protocol_work_partition.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.6, 4.5))
    axis.plot(ko, work_ratio, "o-")
    axis.axhline(1.0, linestyle="--", linewidth=1.0)
    if np.isfinite(protocol_critical):
        axis.axvline(protocol_critical, linestyle=":", linewidth=1.0)
    axis.set_xlabel(r"odd bond coefficient $k_o$")
    axis.set_ylabel(r"$A^{\mathcal{P}}/G_c^{\mathcal{P}}$ at $P/P_G=0.9$")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "protocol_threshold_at_subgriffith_load.pdf")
    plt.close(fig)

    max_relax_balance = max(
        abs(item.relaxation_balance_residual_unit) for item in cache.values()
    )
    max_full_balance = max(
        abs(item.full_balance_residual_unit) for item in cache.values()
    )
    max_final_relative = max(item.final_state_relative_norm for item in cache.values())
    maximum_mirror_work_error = max(
        float(row["protocol_work_mirror_abs_error"]) for row in mirror_rows
    )
    maximum_mirror_odd_error = max(
        float(row["odd_work_mirror_abs_error"]) for row in mirror_rows
    )

    # Small-k_o linear response of the fixed right-tip odd work.
    small = np.abs(ko) <= 0.15 + 1.0e-12
    design = np.column_stack([np.ones(np.count_nonzero(small)), ko[small]])
    coefficients, *_ = np.linalg.lstsq(design, odd_work[small], rcond=None)
    prediction = design @ coefficients
    ss_residual = float(np.sum((odd_work[small] - prediction) ** 2))
    ss_total = float(np.sum((odd_work[small] - np.mean(odd_work[small])) ** 2))
    odd_work_r2 = 1.0 - ss_residual / ss_total

    summary = {
        "grid_nx": nx,
        "grid_ny": ny,
        "crack_half_length": crack_half_length,
        "delta_c": delta_c,
        "load_fraction_of_passive_initiation_stress": load_fraction,
        "passive_initiation_stress": passive_initiation_stress,
        "passive_initiation_displacement": passive_initiation_delta,
        "protocol_resistance_per_bond": protocol_resistance,
        "protocol_critical_k_o_linear_interpolation": protocol_critical,
        "local_bond_critical_k_o_linear_interpolation": bond_critical,
        "critical_k_o_relative_difference": (
            abs(protocol_critical - bond_critical) / bond_critical
            if np.isfinite(protocol_critical) and np.isfinite(bond_critical)
            else float("nan")
        ),
        "small_k_o_odd_work_intercept": float(coefficients[0]),
        "small_k_o_odd_work_slope": float(coefficients[1]),
        "small_k_o_odd_work_fit_r2": odd_work_r2,
        "maximum_relaxation_balance_residual_unit": max_relax_balance,
        "maximum_full_balance_residual_unit": max_full_balance,
        "maximum_final_state_relative_norm": max_final_relative,
        "maximum_protocol_work_mirror_abs_error": maximum_mirror_work_error,
        "maximum_odd_work_mirror_abs_error": maximum_mirror_odd_error,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
