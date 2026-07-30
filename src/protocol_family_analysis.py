#!/usr/bin/env python3
"""Fixed-grip versus reaction-matched dead-load crack-advance protocols.

The two protocols start from the *same* pre-cut equilibrium.  The fixed-grip
protocol keeps all prescribed boundary displacements fixed.  The dead-load
protocol releases those boundary displacement constraints after the cut and
holds fixed the nodal reactions from the pre-cut state; a single reference node
is pinned at its initial position to remove rigid translation.  Thus any work
difference is caused by the post-cut control protocol rather than a different
initial crack field.

For either protocol, during overdamped relaxation

  W_ext + W_odd - Delta U_even(relax) = D_eta.

Including the instantaneous conservative energy E_cut removed with the bond,

  W_ext + W_odd - Delta U_even(total) - D_eta - E_cut = 0.

The operational work is A^P = W_ext + W_odd - Delta U_even(total).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from active_tip_scan import ActiveCrackedStrip
from crack_advance_work import (
    ProtocolUnitResult,
    assemble_components,
    bond_extension,
    even_energy,
    protocol_unit_result,
)


@dataclass
class TractionProtocolUnit:
    nx: int
    ny: int
    crack_half_length: float
    k_o: float
    tip: str
    remote_stress_unit: float
    candidate_extension_unit: float
    cut_bond_even_energy_unit: float
    initial_even_energy_unit: float
    final_even_energy_unit: float
    external_work_unit: float
    odd_work_unit: float
    viscous_dissipation_unit: float
    protocol_work_unit: float
    full_balance_residual_unit: float
    initial_state_match_residual: float
    final_force_residual: float
    final_state_relative_norm: float
    traction_top_sum_unit: float
    traction_bottom_sum_unit: float
    traction_boundary_cv: float
    n_time_steps: int
    t_end: float


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def active_ids(model: ActiveCrackedStrip) -> list[int]:
    return [
        bond_id
        for bond_id in range(len(model.all_bonds))
        if bond_id not in model.removed_ids
    ]


def fixed_grip_equilibrium(matrix, model: ActiveCrackedStrip, delta: float = 1.0):
    constrained, values = model.constrained_dofs(delta)
    mask = np.ones(model.ndof, dtype=bool)
    mask[constrained] = False
    free = np.arange(model.ndof)[mask]
    displacement = np.zeros(model.ndof, dtype=float)
    displacement[constrained] = values
    displacement[free] = spsolve(
        matrix[free][:, free], -(matrix[free][:, constrained] @ values)
    )
    residual = np.asarray(matrix @ displacement)
    return displacement, constrained, values, free, float(np.max(np.abs(residual[free])))


def integrate_dead_load_relaxation(
    total,
    odd,
    initial: np.ndarray,
    final: np.ndarray,
    pinned: np.ndarray,
    pinned_values: np.ndarray,
    free: np.ndarray,
    external_force: np.ndarray,
    t_end: float,
    rtol: float = 2.0e-8,
    relative_atol: float = 1.0e-10,
    max_step: float = np.inf,
):
    total_ff = total[free][:, free]
    total_fc = total[free][:, pinned]
    odd_ff = odd[free][:, free]
    odd_fc = odd[free][:, pinned]
    fext = external_force[free]
    offset0 = initial[free] - final[free]
    scale = max(float(np.max(np.abs(offset0))), 1.0e-12)

    solution = solve_ivp(
        lambda _time, offset: -(total_ff @ offset),
        (0.0, t_end),
        offset0,
        method="BDF",
        jac=-total_ff,
        rtol=rtol,
        atol=relative_atol * scale,
        max_step=max_step,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    def fields(offset: np.ndarray):
        uf = final[free] + offset
        odd_force = -(odd_ff @ uf + odd_fc @ pinned_values)
        net_force = fext - (total_ff @ uf + total_fc @ pinned_values)
        return np.asarray(odd_force), np.asarray(net_force)

    odd_work = 0.0
    external_work = 0.0
    dissipation = 0.0
    odd_previous, net_previous = fields(solution.y[:, 0])
    for step in range(1, len(solution.t)):
        increment = solution.y[:, step] - solution.y[:, step - 1]
        odd_current, net_current = fields(solution.y[:, step])
        odd_work += 0.5 * float((odd_previous + odd_current) @ increment)
        external_work += float(fext @ increment)
        dissipation += 0.5 * float((net_previous + net_current) @ increment)
        odd_previous, net_previous = odd_current, net_current

    relative_norm = float(
        np.linalg.norm(solution.y[:, -1])
        / max(np.linalg.norm(offset0), 1.0e-30)
    )
    return external_work, odd_work, dissipation, relative_norm, len(solution.t)


def reaction_matched_traction_unit(
    nx: int,
    ny: int,
    crack_half_length: float,
    k_o: float,
    tip: str,
    t_end: float,
) -> TractionProtocolUnit:
    model = ActiveCrackedStrip(nx, ny, crack_half_length, 1.0, k_o)
    cut_id = model.tip_candidates()[tip]
    old_ids = active_ids(model)
    new_ids = [bond_id for bond_id in old_ids if bond_id != cut_id]
    even_old, odd_old = assemble_components(model, old_ids)
    even_new, odd_new = assemble_components(model, new_ids)
    total_old = even_old + odd_old
    total_new = even_new + odd_new

    initial, _dirichlet, _values, _free_d, initial_fixed_residual = fixed_grip_equilibrium(
        total_old, model, 1.0
    )
    old_reactions = np.asarray(total_old @ initial)

    # The bottom-left node remains pinned at its initial position.  All other
    # boundary DOFs are released and loaded by their pre-cut reaction forces.
    pin_node = model.node_id(0, 0)
    pinned = np.array([2 * pin_node, 2 * pin_node + 1], dtype=int)
    pinned_values = initial[pinned].copy()
    mask = np.ones(model.ndof, dtype=bool)
    mask[pinned] = False
    free = np.arange(model.ndof)[mask]
    external_force = old_reactions.copy()

    initial_match = external_force[free] - (
        total_old[free][:, free] @ initial[free]
        + total_old[free][:, pinned] @ pinned_values
    )
    initial_match_residual = float(np.max(np.abs(initial_match)))

    final = np.zeros(model.ndof, dtype=float)
    final[pinned] = pinned_values
    rhs = external_force[free] - total_new[free][:, pinned] @ pinned_values
    final[free] = spsolve(total_new[free][:, free], rhs)
    if not np.all(np.isfinite(final)):
        raise RuntimeError("Reaction-controlled equilibrium failed")
    final_residual_vector = external_force[free] - (
        total_new[free][:, free] @ final[free]
        + total_new[free][:, pinned] @ pinned_values
    )
    final_residual = float(np.max(np.abs(final_residual_vector)))

    external_work, odd_work, dissipation, final_relative_norm, n_steps = (
        integrate_dead_load_relaxation(
            total_new,
            odd_new,
            initial,
            final,
            pinned,
            pinned_values,
            free,
            external_force,
            t_end,
        )
    )

    initial_energy = even_energy(even_old, initial)
    final_energy = even_energy(even_new, final)
    extension = bond_extension(model.all_bonds[cut_id], initial)
    cut_energy = 0.5 * extension**2
    delta_energy = final_energy - initial_energy
    protocol_work = external_work + odd_work - delta_energy
    balance = protocol_work - dissipation - cut_energy

    top_y = np.array(
        [2 * model.node_id(i, model.ny - 1) + 1 for i in range(model.nx)], dtype=int
    )
    bottom_y = np.array(
        [2 * model.node_id(i, 0) + 1 for i in range(model.nx)], dtype=int
    )
    top_sum = float(np.sum(old_reactions[top_y]))
    bottom_sum = float(np.sum(old_reactions[bottom_y]))
    remote_stress = top_sum / model.period
    active_boundary = np.concatenate([old_reactions[top_y], old_reactions[bottom_y]])
    mean_abs = float(np.mean(np.abs(active_boundary)))
    traction_cv = float(np.std(active_boundary) / max(mean_abs, 1.0e-30))

    return TractionProtocolUnit(
        nx=nx,
        ny=ny,
        crack_half_length=crack_half_length,
        k_o=float(k_o),
        tip=tip,
        remote_stress_unit=remote_stress,
        candidate_extension_unit=extension,
        cut_bond_even_energy_unit=cut_energy,
        initial_even_energy_unit=initial_energy,
        final_even_energy_unit=final_energy,
        external_work_unit=external_work,
        odd_work_unit=odd_work,
        viscous_dissipation_unit=dissipation,
        protocol_work_unit=protocol_work,
        full_balance_residual_unit=balance,
        initial_state_match_residual=max(initial_match_residual, initial_fixed_residual),
        final_force_residual=final_residual,
        final_state_relative_norm=final_relative_norm,
        traction_top_sum_unit=top_sum,
        traction_bottom_sum_unit=bottom_sum,
        traction_boundary_cv=traction_cv,
        n_time_steps=n_steps,
        t_end=t_end,
    )


def pchip_root(x: np.ndarray, y: np.ndarray, level: float) -> float:
    order = np.argsort(x)
    x = np.asarray(x)[order]
    y = np.asarray(y)[order]
    interpolator = PchipInterpolator(x, y - level)
    dense = np.linspace(float(x[0]), float(x[-1]), 2001)
    values = interpolator(dense)
    changes = np.where(values[:-1] * values[1:] <= 0.0)[0]
    if len(changes) == 0:
        return float("nan")
    i = int(changes[0])
    return float(brentq(interpolator, dense[i], dense[i + 1], xtol=1e-12))


def get_fixed_unit(
    cache: pd.DataFrame,
    nx: int,
    ny: int,
    a: float,
    k_o: float,
    t_end: float,
) -> dict[str, float]:
    crack_column = "crack_half_length" if "crack_half_length" in cache.columns else "a"
    exact = cache[
        (cache.nx == nx)
        & (cache.ny == ny)
        & (np.isclose(cache[crack_column], a))
        & (np.isclose(cache.k_o, k_o, atol=1e-12))
    ]
    if len(exact) == 1:
        row = exact.iloc[0]
        return {
            "protocol_work_unit": float(row.protocol_work_unit),
            "remote_stress_unit": float(row.remote_stress_unit),
            "candidate_extension_unit": float(row.candidate_extension_unit),
            "full_balance_residual_unit": float(row.full_balance_residual_unit),
            "final_state_relative_norm": float(row.final_state_relative_norm),
        }
    result = protocol_unit_result(nx, ny, a, k_o, "right", t_end)
    return {
        "protocol_work_unit": result.protocol_work_unit,
        "remote_stress_unit": result.remote_stress_unit,
        "candidate_extension_unit": result.candidate_extension_unit,
        "full_balance_residual_unit": result.full_balance_residual_unit,
        "final_state_relative_norm": result.final_state_relative_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PACKAGE_ROOT / "data_recomputed" / "protocol_family_results")
    parser.add_argument(
        "--fixed-cache",
        type=Path,
        default=PACKAGE_ROOT / "data" / "protocol_threshold_results" / "protocol_unit_cache.csv",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=PACKAGE_ROOT / "data" / "protocol_threshold_results" / "protocol_threshold_systematic.csv",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fixed_cache = pd.read_csv(args.fixed_cache)
    thresholds = pd.read_csv(args.thresholds)
    systems = [
        {"nx": 48, "ny": 36, "a": 6.0, "t_end": 70000.0},
        {"nx": 64, "ny": 48, "a": 8.0, "t_end": 125000.0},
    ]
    k_grid = np.array([0.0, 0.02, 0.05, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30])

    unit_rows: list[dict[str, object]] = []
    traction_cache: dict[tuple[int, float], TractionProtocolUnit] = {}
    fixed_units: dict[tuple[int, float], dict[str, float]] = {}

    for system in systems:
        nx, ny, a, t_end = system["nx"], system["ny"], system["a"], system["t_end"]
        for k_o in k_grid:
            fixed = get_fixed_unit(fixed_cache, nx, ny, a, float(k_o), t_end)
            traction = reaction_matched_traction_unit(
                nx, ny, a, float(k_o), "right", t_end
            )
            fixed_units[(nx, float(k_o))] = fixed
            traction_cache[(nx, float(k_o))] = traction
            unit_rows.append(
                {
                    "nx": nx,
                    "ny": ny,
                    "a": a,
                    "k_o": k_o,
                    "fixed_protocol_work_unit": fixed["protocol_work_unit"],
                    "traction_protocol_work_unit": traction.protocol_work_unit,
                    "traction_to_fixed_work_ratio": traction.protocol_work_unit
                    / fixed["protocol_work_unit"],
                    "fixed_remote_stress_unit": fixed["remote_stress_unit"],
                    "traction_remote_stress_unit": traction.remote_stress_unit,
                    "candidate_extension_unit": traction.candidate_extension_unit,
                    "traction_external_work_unit": traction.external_work_unit,
                    "traction_odd_work_unit": traction.odd_work_unit,
                    "traction_viscous_dissipation_unit": traction.viscous_dissipation_unit,
                    "traction_cut_energy_unit": traction.cut_bond_even_energy_unit,
                    "traction_full_balance_residual_unit": traction.full_balance_residual_unit,
                    "traction_initial_match_residual": traction.initial_state_match_residual,
                    "traction_final_force_residual": traction.final_force_residual,
                    "traction_final_state_relative_norm": traction.final_state_relative_norm,
                    "traction_boundary_cv": traction.traction_boundary_cv,
                }
            )
    write_csv(args.out / "protocol_family_unit_scan.csv", unit_rows)

    threshold_rows: list[dict[str, object]] = []
    direct_rows: list[dict[str, object]] = []
    for system in systems:
        nx, ny, a = system["nx"], system["ny"], system["a"]
        subset = thresholds[thresholds.nx == nx].sort_values("load_fraction")
        rows_sys = [row for row in unit_rows if row["nx"] == nx]
        ko = np.array([float(row["k_o"]) for row in rows_sys])
        stress = np.array([float(row["fixed_remote_stress_unit"]) for row in rows_sys])
        extension = np.array([float(row["candidate_extension_unit"]) for row in rows_sys])
        A_D = np.array([float(row["fixed_protocol_work_unit"]) for row in rows_sys])
        A_T = np.array([float(row["traction_protocol_work_unit"]) for row in rows_sys])
        i0 = int(np.argmin(np.abs(ko)))
        s0, e0, AD0, AT0 = stress[i0], extension[i0], A_D[i0], A_T[i0]
        rD = s0**2 * A_D / (stress**2 * AD0)
        rT = s0**2 * A_T / (stress**2 * AT0)
        b = s0 * extension / (stress * e0)
        interp_rD = PchipInterpolator(ko, rD)
        interp_rT = PchipInterpolator(ko, rT)
        interp_b = PchipInterpolator(ko, b)

        passive_row = subset.iloc[np.argmin(np.abs(subset.load_fraction - 0.98))]
        # Recover passive initiation displacement from G_D = AD0 * delta_G^2.
        delta_g = math.sqrt(float(passive_row.protocol_resistance) / AD0)
        GD = AD0 * delta_g**2
        GT = AT0 * delta_g**2

        for _, row in subset.iterrows():
            p = float(row.load_fraction)
            k_bond = float(row.bond_threshold_k_o)
            k_D = pchip_root(ko, rD, p ** -2)
            k_T = pchip_root(ko, rT, p ** -2)
            ratio_D_at_bond = p**2 * float(interp_rD(k_bond))
            ratio_T_at_bond = p**2 * float(interp_rT(k_bond))
            threshold_rows.append(
                {
                    "nx": nx,
                    "ny": ny,
                    "a": a,
                    "load_fraction": p,
                    "bond_threshold_k_o": k_bond,
                    "fixed_grip_threshold_k_o": k_D,
                    "matched_traction_threshold_k_o": k_T,
                    "fixed_grip_relative_k_error": abs(k_D - k_bond) / k_bond,
                    "matched_traction_relative_k_error": abs(k_T - k_bond) / k_bond,
                    "fixed_grip_work_ratio_at_bond_threshold": ratio_D_at_bond,
                    "matched_traction_work_ratio_at_bond_threshold": ratio_T_at_bond,
                    "protocol_work_ratio_difference_at_bond_threshold": ratio_T_at_bond
                    - ratio_D_at_bond,
                    "passive_fixed_grip_resistance": GD,
                    "passive_matched_traction_resistance": GT,
                    "traction_to_fixed_passive_resistance_ratio": GT / GD,
                }
            )

            # Direct evaluation at each independent bond threshold for the traction
            # protocol validates interpolation and records a closed work balance.
            key = (nx, round(k_bond, 12))
            if key not in traction_cache:
                traction_cache[key] = reaction_matched_traction_unit(
                    nx, ny, a, k_bond, "right", system["t_end"]
                )
            tr = traction_cache[key]
            target_delta = p * (float(row.passive_initiation_stress) / tr.remote_stress_unit)
            scale = target_delta**2
            direct_ratio = scale * tr.protocol_work_unit / GT
            direct_rows.append(
                {
                    "nx": nx,
                    "ny": ny,
                    "load_fraction": p,
                    "bond_threshold_k_o": k_bond,
                    "direct_matched_traction_work_ratio": direct_ratio,
                    "interpolated_matched_traction_work_ratio": ratio_T_at_bond,
                    "direct_minus_interpolated": direct_ratio - ratio_T_at_bond,
                    "scaled_external_work": scale * tr.external_work_unit,
                    "scaled_odd_work": scale * tr.odd_work_unit,
                    "scaled_even_energy_change": scale
                    * (tr.final_even_energy_unit - tr.initial_even_energy_unit),
                    "scaled_viscous_dissipation": scale * tr.viscous_dissipation_unit,
                    "scaled_cut_energy": scale * tr.cut_bond_even_energy_unit,
                    "scaled_balance_residual": scale * tr.full_balance_residual_unit,
                }
            )

    write_csv(args.out / "protocol_family_thresholds.csv", threshold_rows)
    write_csv(args.out / "protocol_family_direct_validation.csv", direct_rows)

    # A direct mirror check for the matched-traction protocol.
    mirror_rows: list[dict[str, object]] = []
    for system in systems:
        for magnitude in (0.10, 0.20):
            right = traction_cache[(system["nx"], magnitude)]
            left = reaction_matched_traction_unit(
                system["nx"], system["ny"], system["a"], -magnitude, "left", system["t_end"]
            )
            mirror_rows.append(
                {
                    "nx": system["nx"],
                    "abs_k_o": magnitude,
                    "protocol_work_abs_error": abs(
                        right.protocol_work_unit - left.protocol_work_unit
                    ),
                    "external_work_abs_error": abs(
                        right.external_work_unit - left.external_work_unit
                    ),
                    "odd_work_abs_error": abs(right.odd_work_unit - left.odd_work_unit),
                    "extension_abs_error": abs(
                        right.candidate_extension_unit - left.candidate_extension_unit
                    ),
                }
            )
    write_csv(args.out / "matched_traction_chirality_mirror.csv", mirror_rows)

    # Figure 1: protocol curves at p=0.9 for both sizes.
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    for axis, system in zip(axes, systems):
        nx = system["nx"]
        rows_sys = [row for row in unit_rows if row["nx"] == nx]
        ko = np.array([float(row["k_o"]) for row in rows_sys])
        stress = np.array([float(row["fixed_remote_stress_unit"]) for row in rows_sys])
        ext = np.array([float(row["candidate_extension_unit"]) for row in rows_sys])
        AD = np.array([float(row["fixed_protocol_work_unit"]) for row in rows_sys])
        AT = np.array([float(row["traction_protocol_work_unit"]) for row in rows_sys])
        s0, e0, AD0, AT0 = stress[0], ext[0], AD[0], AT[0]
        p = 0.9
        axis.plot(ko, p**2 * s0**2 * AD / (stress**2 * AD0), "o-", label="fixed grip")
        axis.plot(ko, p**2 * s0**2 * AT / (stress**2 * AT0), "s-", label="matched traction")
        axis.plot(ko, p * s0 * ext / (stress * e0), "^-", label="bond criterion")
        axis.axhline(1.0, linestyle="--", linewidth=1.0)
        axis.set_xlabel(r"odd coefficient $k_o/k$")
        axis.set_title(f"{system['nx']} x {system['ny']}")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("normalized initiation measure")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "protocol_family_curves_p090.pdf")
    plt.close(fig)

    df_thr = pd.DataFrame(threshold_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for nx, marker in ((48, "o"), (64, "s")):
        d = df_thr[df_thr.nx == nx].sort_values("load_fraction")
        axes[0].plot(d.load_fraction, d.fixed_grip_work_ratio_at_bond_threshold, marker + "-", label=f"fixed, Nx={nx}")
        axes[0].plot(d.load_fraction, d.matched_traction_work_ratio_at_bond_threshold, marker + "--", label=f"traction, Nx={nx}")
        axes[1].plot(d.load_fraction, 100*d.fixed_grip_relative_k_error, marker + "-", label=f"fixed, Nx={nx}")
        axes[1].plot(d.load_fraction, 100*d.matched_traction_relative_k_error, marker + "--", label=f"traction, Nx={nx}")
    axes[0].axhline(1.0, linestyle=":", linewidth=1.0)
    axes[0].set_xlabel(r"load fraction $P/P_G^{\rm lat}$")
    axes[0].set_ylabel(r"$A^{\mathcal{P}}/G_c^{\mathcal{P}}$ at bond threshold")
    axes[1].set_xlabel(r"load fraction $P/P_G^{\rm lat}$")
    axes[1].set_ylabel(r"critical-$k_o$ error (percent)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "protocol_family_threshold_comparison.pdf")
    plt.close(fig)

    # Representative work partition at p=0.9, ko=0.12, 48x36.
    representative_fixed = protocol_unit_result(48, 36, 6.0, 0.12, "right", 70000.0)
    representative_traction = traction_cache[(48, 0.12)]
    passive_stress = float(
        thresholds[(thresholds.nx == 48) & np.isclose(thresholds.load_fraction, 0.9)]
        .iloc[0]
        .passive_initiation_stress
    )
    delta = 0.9 * passive_stress / representative_fixed.remote_stress_unit
    scale = delta**2
    partition_rows = [
        {
            "protocol": "fixed_grip",
            "external_work": 0.0,
            "odd_work": scale * representative_fixed.odd_work_unit,
            "minus_even_energy_change": -scale
            * (representative_fixed.final_even_energy_unit - representative_fixed.initial_even_energy_unit),
            "viscous_dissipation": scale * representative_fixed.viscous_dissipation_unit,
            "cut_energy": scale * representative_fixed.cut_bond_even_energy_unit,
            "protocol_work": scale * representative_fixed.protocol_work_unit,
            "balance_residual": scale * representative_fixed.full_balance_residual_unit,
        },
        {
            "protocol": "matched_traction",
            "external_work": scale * representative_traction.external_work_unit,
            "odd_work": scale * representative_traction.odd_work_unit,
            "minus_even_energy_change": -scale
            * (representative_traction.final_even_energy_unit - representative_traction.initial_even_energy_unit),
            "viscous_dissipation": scale * representative_traction.viscous_dissipation_unit,
            "cut_energy": scale * representative_traction.cut_bond_even_energy_unit,
            "protocol_work": scale * representative_traction.protocol_work_unit,
            "balance_residual": scale * representative_traction.full_balance_residual_unit,
        },
    ]
    write_csv(args.out / "representative_protocol_work_partition.csv", partition_rows)

    labels = ["external", "odd", "-Delta Ue", "dissipation", "cut energy", "A^P"]
    fixed_values = [partition_rows[0][key] for key in ("external_work", "odd_work", "minus_even_energy_change", "viscous_dissipation", "cut_energy", "protocol_work")]
    traction_values = [partition_rows[1][key] for key in ("external_work", "odd_work", "minus_even_energy_change", "viscous_dissipation", "cut_energy", "protocol_work")]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.0, 4.4))
    axis.bar(x - width/2, fixed_values, width, label="fixed grip")
    axis.bar(x + width/2, traction_values, width, label="matched traction")
    axis.axhline(0.0, linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("work per virtual bond advance")
    axis.legend(frameon=False)
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "representative_protocol_work_partition.pdf")
    plt.close(fig)

    df_unit = pd.DataFrame(unit_rows)
    df_direct = pd.DataFrame(direct_rows)
    summary = {
        "maximum_traction_balance_abs": float(df_unit.traction_full_balance_residual_unit.abs().max()),
        "maximum_traction_initial_match_residual": float(df_unit.traction_initial_match_residual.max()),
        "maximum_traction_final_force_residual": float(df_unit.traction_final_force_residual.max()),
        "maximum_traction_final_relative_norm": float(df_unit.traction_final_state_relative_norm.max()),
        "traction_to_fixed_unit_work_ratio_min": float(df_unit.traction_to_fixed_work_ratio.min()),
        "traction_to_fixed_unit_work_ratio_max": float(df_unit.traction_to_fixed_work_ratio.max()),
        "maximum_direct_interpolation_abs_error": float(df_direct.direct_minus_interpolated.abs().max()),
        "maximum_scaled_direct_balance_abs": float(df_direct.scaled_balance_residual.abs().max()),
        "maximum_mirror_protocol_work_abs_error": float(max(row["protocol_work_abs_error"] for row in mirror_rows)),
        "maximum_mirror_odd_work_abs_error": float(max(row["odd_work_abs_error"] for row in mirror_rows)),
        "maximum_fixed_work_ratio_at_bond_deviation": float((df_thr.fixed_grip_work_ratio_at_bond_threshold - 1).abs().max()),
        "maximum_traction_work_ratio_at_bond_deviation": float((df_thr.matched_traction_work_ratio_at_bond_threshold - 1).abs().max()),
        "maximum_fixed_critical_k_relative_error": float(df_thr.fixed_grip_relative_k_error.max()),
        "maximum_traction_critical_k_relative_error": float(df_thr.matched_traction_relative_k_error.max()),
        "maximum_protocol_ratio_difference_at_bond_threshold": float(df_thr.protocol_work_ratio_difference_at_bond_threshold.abs().max()),
        "passive_resistance_ratio_range": [
            float(df_thr.traction_to_fixed_passive_resistance_ratio.min()),
            float(df_thr.traction_to_fixed_passive_resistance_ratio.max()),
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
