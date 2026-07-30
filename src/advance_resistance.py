#!/usr/bin/env python3
"""Stepwise crack-advance work, resistance, and arrest.

A right crack tip is advanced virtually through four successive crack-plane
bonds (two complete triangular-lattice translation periods).  For each topology
step and boundary ensemble, the code computes the operational work

    A_n^P = W_ext + W_odd - Delta U_even,

and calibrates a topology-specific passive resistance

    R_n^eff = A_{n,passive}^P(delta_c / ell_{n,passive})^2.

The stability margin is Delta_n = A_n^P - R_n^eff.  All unit-load solutions are
reused by quadratic scaling for the full load-fraction scan.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse.linalg import spsolve

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
from propagation_limit_analysis import (
    DeadLoadCascade,
    FixedGripCascade,
    active_ids,
    crossing_frontier,
    dead_load_passive_reference,
    fixed_grip_passive_reference,
    top_reaction_stress,
)
from protocol_family_analysis import integrate_dead_load_relaxation


@dataclass
class UnitStep:
    nx: int
    ny: int
    crack_fraction: float
    target_half_length: float
    effective_half_length_initial: float
    boundary_condition: str
    k_o: float
    step: int
    period: int
    bond_id: int
    bond_midpoint_x: float
    bond_nx: float
    bond_ny: float
    remote_stress_unit: float
    candidate_extension_unit: float
    external_work_unit: float
    odd_work_unit: float
    even_energy_change_unit: float
    viscous_dissipation_unit: float
    cut_energy_unit: float
    protocol_work_unit: float
    balance_residual_unit: float
    initial_force_residual: float
    final_force_residual: float
    final_state_relative_norm: float
    n_time_steps: int
    t_end: float
    rtol: float
    relative_atol: float
    max_step: float


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _dead_load_force(model: ActiveCrackedStrip, stress: float = 1.0) -> np.ndarray:
    force = np.zeros(model.ndof, dtype=float)
    nodal = stress * model.period / model.nx
    for i in range(model.nx):
        force[2 * model.node_id(i, model.ny - 1) + 1] += nodal
        force[2 * model.node_id(i, 0) + 1] -= nodal
    return force


def _dead_partition(model: ActiveCrackedStrip) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pin = model.node_id(0, 0)
    constrained = np.array([2 * pin, 2 * pin + 1], dtype=int)
    values = np.zeros(2, dtype=float)
    mask = np.ones(model.ndof, dtype=bool)
    mask[constrained] = False
    free = np.arange(model.ndof)[mask]
    return constrained, values, free


def _solve_dead(matrix, model: ActiveCrackedStrip, force: np.ndarray):
    constrained, values, free = _dead_partition(model)
    displacement = np.zeros(model.ndof, dtype=float)
    rhs = force[free] - matrix[free][:, constrained] @ values
    displacement[free] = spsolve(matrix[free][:, free], rhs)
    if not np.all(np.isfinite(displacement)):
        raise RuntimeError("Dead-load equilibrium failed")
    residual = force[free] - (
        matrix[free][:, free] @ displacement[free]
        + matrix[free][:, constrained] @ values
    )
    return displacement, constrained, values, free, float(np.max(np.abs(residual)))


def fixed_grip_sequence(
    nx: int,
    ny: int,
    crack_fraction: float,
    k_o: float,
    n_steps: int,
    t_end: float,
    rtol: float = 2.0e-8,
    relative_atol: float = 1.0e-10,
    max_step: float = np.inf,
) -> list[UnitStep]:
    target = crack_fraction * nx
    model = ActiveCrackedStrip(nx, ny, target, 1.0, k_o)
    removed = set(model.removed_ids)
    old_even, old_odd = assemble_components(model, active_ids(model, removed))
    old_total = old_even + old_odd
    initial, constrained, values, free, initial_residual = solve_equilibrium(
        old_total, model, 1.0
    )
    rows: list[UnitStep] = []
    for step in range(1, n_steps + 1):
        frontier = crossing_frontier(model, removed)
        if "right" not in frontier:
            raise RuntimeError("Right crack frontier exhausted")
        cut_id = frontier["right"]
        bond = model.all_bonds[cut_id]
        extension = bond_extension(bond, initial)
        remote_stress = top_reaction_stress(model, old_total, initial)

        new_removed = set(removed)
        new_removed.add(cut_id)
        new_even, new_odd = assemble_components(model, active_ids(model, new_removed))
        new_total = new_even + new_odd
        final, c2, v2, f2, final_residual = solve_equilibrium(new_total, model, 1.0)
        if not (
            np.array_equal(constrained, c2)
            and np.array_equal(free, f2)
            and np.allclose(values, v2)
        ):
            raise RuntimeError("Fixed-grip boundary partition changed")

        odd_work, dissipation, final_norm, n_time = integrate_relaxation(
            new_total,
            new_odd,
            initial,
            final,
            constrained,
            values,
            free,
            t_end,
            rtol=rtol,
            relative_atol=relative_atol,
            max_step=max_step,
        )
        energy_initial = even_energy(old_even, initial)
        energy_final = even_energy(new_even, final)
        delta_energy = energy_final - energy_initial
        cut_energy = 0.5 * extension**2
        protocol_work = odd_work - delta_energy
        balance = protocol_work - dissipation - cut_energy
        assert bond.midpoint_x is not None
        rows.append(
            UnitStep(
                nx, ny, crack_fraction, target, model.a_eff,
                "fixed_grip", float(k_o), step, (step + 1) // 2, cut_id,
                float(bond.midpoint_x), float(bond.n[0]), float(bond.n[1]),
                remote_stress, extension, 0.0, odd_work, delta_energy,
                dissipation, cut_energy, protocol_work, balance,
                initial_residual, final_residual, final_norm, n_time,
                t_end, rtol, relative_atol, float(max_step),
            )
        )
        removed = new_removed
        old_even, old_odd, old_total = new_even, new_odd, new_total
        initial, initial_residual = final, final_residual
    return rows


def dead_load_sequence(
    nx: int,
    ny: int,
    crack_fraction: float,
    k_o: float,
    n_steps: int,
    t_end: float,
    rtol: float = 2.0e-8,
    relative_atol: float = 1.0e-10,
    max_step: float = np.inf,
) -> list[UnitStep]:
    target = crack_fraction * nx
    model = ActiveCrackedStrip(nx, ny, target, 1.0, k_o)
    removed = set(model.removed_ids)
    force = _dead_load_force(model, 1.0)
    old_even, old_odd = assemble_components(model, active_ids(model, removed))
    old_total = old_even + old_odd
    initial, constrained, values, free, initial_residual = _solve_dead(
        old_total, model, force
    )
    rows: list[UnitStep] = []
    for step in range(1, n_steps + 1):
        frontier = crossing_frontier(model, removed)
        if "right" not in frontier:
            raise RuntimeError("Right crack frontier exhausted")
        cut_id = frontier["right"]
        bond = model.all_bonds[cut_id]
        extension = bond_extension(bond, initial)

        new_removed = set(removed)
        new_removed.add(cut_id)
        new_even, new_odd = assemble_components(model, active_ids(model, new_removed))
        new_total = new_even + new_odd
        final, c2, v2, f2, final_residual = _solve_dead(new_total, model, force)
        if not (
            np.array_equal(constrained, c2)
            and np.array_equal(free, f2)
            and np.allclose(values, v2)
        ):
            raise RuntimeError("Dead-load boundary partition changed")

        external_work, odd_work, dissipation, final_norm, n_time = (
            integrate_dead_load_relaxation(
                new_total,
                new_odd,
                initial,
                final,
                constrained,
                values,
                free,
                force,
                t_end,
                rtol=rtol,
                relative_atol=relative_atol,
                max_step=max_step,
            )
        )
        energy_initial = even_energy(old_even, initial)
        energy_final = even_energy(new_even, final)
        delta_energy = energy_final - energy_initial
        cut_energy = 0.5 * extension**2
        protocol_work = external_work + odd_work - delta_energy
        balance = protocol_work - dissipation - cut_energy
        assert bond.midpoint_x is not None
        rows.append(
            UnitStep(
                nx, ny, crack_fraction, target, model.a_eff,
                "dead_load", float(k_o), step, (step + 1) // 2, cut_id,
                float(bond.midpoint_x), float(bond.n[0]), float(bond.n[1]),
                1.0, extension, external_work, odd_work, delta_energy,
                dissipation, cut_energy, protocol_work, balance,
                initial_residual, final_residual, final_norm, n_time,
                t_end, rtol, relative_atol, float(max_step),
            )
        )
        removed = new_removed
        old_even, old_odd, old_total = new_even, new_odd, new_total
        initial, initial_residual = final, final_residual
    return rows


def scale_rows(
    unit_steps: list[UnitStep],
    delta_c: float,
    load_fractions: Iterable[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[int, int, float, str, float], list[UnitStep]] = {}
    for row in unit_steps:
        key = (row.nx, row.ny, row.crack_fraction, row.boundary_condition, row.k_o)
        grouped.setdefault(key, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: item.step)

    passive: dict[tuple[int, int, float, str], list[UnitStep]] = {}
    for key, rows in grouped.items():
        nx, ny, frac, bc, ko = key
        if abs(ko) < 1.0e-14:
            passive[(nx, ny, frac, bc)] = rows

    scaled: list[dict[str, object]] = []
    periods: list[dict[str, object]] = []
    for key, rows in grouped.items():
        nx, ny, frac, bc, ko = key
        p_rows = passive[(nx, ny, frac, bc)]
        initial_passive = p_rows[0]
        if bc == "fixed_grip":
            passive_initial_amplitude = delta_c / initial_passive.candidate_extension_unit
            passive_initial_load = (
                initial_passive.remote_stress_unit * passive_initial_amplitude
            )
            active_unit_load = rows[0].remote_stress_unit
        else:
            passive_initial_load = delta_c / initial_passive.candidate_extension_unit
            active_unit_load = 1.0

        for p in load_fractions:
            amplitude = float(p) * passive_initial_load / active_unit_load
            step_scaled: list[dict[str, object]] = []
            for row, prow in zip(rows, p_rows):
                critical_amplitude = delta_c / prow.candidate_extension_unit
                resistance = prow.protocol_work_unit * critical_amplitude**2
                factor = amplitude**2
                available = row.protocol_work_unit * factor
                margin = available - resistance
                item = {
                    **asdict(row),
                    "load_fraction": float(p),
                    "applied_amplitude": amplitude,
                    "passive_initial_load": passive_initial_load,
                    "passive_step_critical_amplitude": critical_amplitude,
                    "available_work": available,
                    "effective_resistance": resistance,
                    "stability_margin": margin,
                    "work_ratio": available / resistance,
                    "extension_ratio": row.candidate_extension_unit * amplitude / delta_c,
                    "external_work": row.external_work_unit * factor,
                    "odd_work": row.odd_work_unit * factor,
                    "even_energy_change": row.even_energy_change_unit * factor,
                    "viscous_dissipation": row.viscous_dissipation_unit * factor,
                    "cut_energy": row.cut_energy_unit * factor,
                    "balance_residual": row.balance_residual_unit * factor,
                    "resistance_increment_from_step1": resistance
                    / (
                        p_rows[0].protocol_work_unit
                        * (delta_c / p_rows[0].candidate_extension_unit) ** 2
                    )
                    - 1.0,
                }
                scaled.append(item)
                step_scaled.append(item)
            for period in (1, 2):
                subset = [r for r in step_scaled if r["period"] == period]
                periods.append(
                    {
                        "nx": nx,
                        "ny": ny,
                        "crack_fraction": frac,
                        "boundary_condition": bc,
                        "k_o": ko,
                        "load_fraction": float(p),
                        "period": period,
                        "available_work": sum(float(r["available_work"]) for r in subset),
                        "effective_resistance": sum(float(r["effective_resistance"]) for r in subset),
                        "stability_margin": sum(float(r["stability_margin"]) for r in subset),
                        "work_ratio": sum(float(r["available_work"]) for r in subset)
                        / sum(float(r["effective_resistance"]) for r in subset),
                        "odd_work": sum(float(r["odd_work"]) for r in subset),
                        "external_work": sum(float(r["external_work"]) for r in subset),
                        "minimum_extension_ratio": min(float(r["extension_ratio"]) for r in subset),
                        "maximum_extension_ratio": max(float(r["extension_ratio"]) for r in subset),
                    }
                )
    return scaled, periods


def cascade_scan(
    systems: list[tuple[int, int, float]],
    k_values: list[float],
    load_fractions: list[float],
    delta_c: float,
    max_breaks: int = 8,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for nx, ny, frac in systems:
        target = frac * nx
        passive_fixed_stress, _ = fixed_grip_passive_reference(nx, ny, target, delta_c)
        passive_dead_stress = dead_load_passive_reference(nx, ny, target, delta_c)
        for ko in k_values:
            fixed_probe = FixedGripCascade(nx, ny, target, ko)
            fixed_unit_stress = fixed_probe.initial_unit_stress()
            for p in load_fractions:
                fixed = FixedGripCascade(nx, ny, target, ko)
                delta = p * passive_fixed_stress / fixed_unit_stress
                result = fixed.run(delta, delta_c, max_breaks=max_breaks)
                rows.append({
                    **vars(result),
                    "nx": nx, "ny": ny, "crack_fraction": frac,
                    "target_half_length": target, "load_fraction": p,
                    "applied_amplitude": delta,
                })

                dead = DeadLoadCascade(nx, ny, target, ko)
                result_d = dead.run(p * passive_dead_stress, delta_c, max_breaks=max_breaks)
                rows.append({
                    **vars(result_d),
                    "nx": nx, "ny": ny, "crack_fraction": frac,
                    "target_half_length": target, "load_fraction": p,
                    "applied_amplitude": p * passive_dead_stress,
                })
    return rows


def positive_control_scan(
    systems: list[tuple[int, int, float]],
    delta_c: float,
    load_fraction: float = 1.001,
    max_breaks: int = 8,
) -> list[dict[str, object]]:
    """Passive just-above-initiation controls for both loading ensembles."""
    rows: list[dict[str, object]] = []
    for nx, ny, frac in systems:
        target = frac * nx
        passive_fixed_stress, _ = fixed_grip_passive_reference(
            nx, ny, target, delta_c
        )
        passive_dead_stress = dead_load_passive_reference(
            nx, ny, target, delta_c
        )
        fixed = FixedGripCascade(nx, ny, target, 0.0)
        delta = load_fraction * passive_fixed_stress / fixed.initial_unit_stress()
        result_f = fixed.run(delta, delta_c, max_breaks=max_breaks)
        rows.append({
            **vars(result_f), "nx": nx, "ny": ny, "crack_fraction": frac,
            "target_half_length": target, "load_fraction": load_fraction,
            "applied_amplitude": delta,
        })
        dead = DeadLoadCascade(nx, ny, target, 0.0)
        result_d = dead.run(
            load_fraction * passive_dead_stress,
            delta_c,
            max_breaks=max_breaks,
        )
        rows.append({
            **vars(result_d), "nx": nx, "ny": ny, "crack_fraction": frac,
            "target_half_length": target, "load_fraction": load_fraction,
            "applied_amplitude": load_fraction * passive_dead_stress,
        })
    return rows


def convergence_scan(out: Path, delta_c: float) -> list[dict[str, object]]:
    configs = [
        ("coarse", 30000.0, 1.0e-6, 1.0e-8, 2000.0),
        ("baseline", 30000.0, 2.0e-8, 1.0e-10, np.inf),
        ("fine", 60000.0, 1.0e-9, 1.0e-11, 250.0),
    ]
    rows: list[dict[str, object]] = []
    for label, t_end, rtol, atol, max_step in configs:
        for bc, fn in (("fixed_grip", fixed_grip_sequence), ("dead_load", dead_load_sequence)):
            seq = fn(48, 36, 0.125, 0.20, 2, t_end, rtol, atol, max_step)
            for row in seq:
                rows.append({"configuration": label, **asdict(row)})
    write_csv(out / "integration_convergence.csv", rows)
    return rows


def summarize(
    unit_rows: list[UnitStep],
    scaled: list[dict[str, object]],
    periods: list[dict[str, object]],
    cascades: list[dict[str, object]],
    convergence: list[dict[str, object]],
) -> dict[str, object]:
    nominal = [
        row for row in scaled
        if row["nx"] == 48 and abs(float(row["crack_fraction"]) - 0.125) < 1e-12
        and row["boundary_condition"] == "fixed_grip"
        and abs(float(row["load_fraction"]) - 0.90) < 1e-12
        and abs(float(row["k_o"]) - 0.20) < 1e-12
    ]
    nominal.sort(key=lambda r: int(r["step"]))
    sub = [r for r in cascades if float(r["load_fraction"]) < 1.0]
    positive = []
    for bc in ("fixed_grip", "dead_load"):
        b = [r for r in convergence if r["boundary_condition"] == bc and r["configuration"] == "baseline"]
        f = [r for r in convergence if r["boundary_condition"] == bc and r["configuration"] == "fine"]
        for rb, rf in zip(sorted(b, key=lambda x: x["step"]), sorted(f, key=lambda x: x["step"])):
            positive.append(abs(float(rb["protocol_work_unit"]) - float(rf["protocol_work_unit"])) / abs(float(rf["protocol_work_unit"])))
    p09 = [r for r in periods if abs(float(r["load_fraction"]) - 0.90) < 1e-12]
    return {
        "unit_step_count": len(unit_rows),
        "scaled_step_count": len(scaled),
        "period_count": len(periods),
        "cascade_state_count": len(cascades),
        "sizes": sorted({int(r.nx) for r in unit_rows}),
        "crack_fractions": sorted({float(r.crack_fraction) for r in unit_rows}),
        "boundary_conditions": sorted({r.boundary_condition for r in unit_rows}),
        "k_o_values": sorted({float(r.k_o) for r in unit_rows}),
        "maximum_abs_scaled_balance_residual": max(abs(float(r["balance_residual"])) for r in scaled),
        "maximum_unit_final_state_relative_norm": max(float(r.final_state_relative_norm) for r in unit_rows),
        "maximum_broken_bonds_subgriffith": max(int(r["broken_bonds"]) for r in sub),
        "number_sustained_subgriffith_states_ge_2": sum(int(r["broken_bonds"]) >= 2 for r in sub),
        "minimum_period_work_ratio_at_p0p90": min(float(r["work_ratio"]) for r in p09),
        "maximum_period_work_ratio_at_p0p90": max(float(r["work_ratio"]) for r in p09),
        "maximum_baseline_to_fine_protocol_work_relative_change": max(positive),
        "nominal_fixed_p0p90_ko0p20": [
            {
                "step": int(r["step"]),
                "work_ratio": float(r["work_ratio"]),
                "extension_ratio": float(r["extension_ratio"]),
                "odd_work": float(r["odd_work"]),
                "stability_margin": float(r["stability_margin"]),
                "resistance_increment_from_step1": float(r["resistance_increment_from_step1"]),
            }
            for r in nominal
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "advance_resistance_results")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--delta-c", type=float, default=0.02)
    parser.add_argument("--skip-convergence", action="store_true")
    parser.add_argument(
        "--bridge-sizes-only", action="store_true",
        help=("Compute only the extra fixed-grip unit steps needed by the matched "
              "flux/work bridge at Nx=80,96 and write them to a separate file. "
              "The main 224-state stepwise scan is left untouched."),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Terminal time and maximum step scale as Nx^2, matching the 32/48/64 settings.
    bridge_integration = {
        80: (75000.0, 1.0e-6, 1.0e-8, 3100.0),
        96: (108000.0, 1.0e-6, 1.0e-8, 4500.0),
    }
    if args.bridge_sizes_only:
        extra: list[UnitStep] = []
        for nx, ny in ((80, 60), (96, 72)):
            t_end, rtol, relative_atol, max_step = bridge_integration[nx]
            for ko in (0.0, 0.20):
                seq = fixed_grip_sequence(
                    nx, ny, 0.125, ko, 1, t_end,
                    rtol=rtol, relative_atol=relative_atol, max_step=max_step,
                )
                extra.extend(seq)
                print(f"completed bridge unit step: Nx={nx}, a/L=0.125, ko={ko:.2f}", flush=True)
        write_csv(args.out / "advance_unit_steps_bridge.csv", [asdict(row) for row in extra])
        print(f"wrote {args.out / 'advance_unit_steps_bridge.csv'}")
        return

    # Balanced design: all three sizes at a/L=1/8, all three crack lengths at
    # Nx=32 and 48. This independently resolves size and initial-crack effects
    # without paying for a redundant full 3x3 factorial at Nx=64.
    systems = [
        (32, 24, 0.10), (32, 24, 0.125), (32, 24, 0.15),
        (48, 36, 0.10), (48, 36, 0.125), (48, 36, 0.15),
        (64, 48, 0.125),
    ]
    k_values = [0.0, 0.12, 0.20, 0.30]
    load_fractions = [0.85, 0.90, 0.95, 0.98]
    # The broad scan uses a converged short-tail setting; the nominal case is
    # independently checked by convergence_scan() at substantially tighter
    # tolerances and twice the terminal time.
    integration = {
        32: (12000.0, 1.0e-6, 1.0e-8, 500.0),
        48: (27000.0, 1.0e-6, 1.0e-8, 1100.0),
        64: (48000.0, 1.0e-6, 1.0e-8, 2000.0),
    }

    units: list[UnitStep] = []
    for nx, ny, frac in systems:
        t_end, rtol, relative_atol, max_step = integration[nx]
        for ko in k_values:
            for fn in (fixed_grip_sequence, dead_load_sequence):
                seq = fn(
                    nx, ny, frac, ko, args.steps, t_end,
                    rtol=rtol, relative_atol=relative_atol, max_step=max_step,
                )
                units.extend(seq)
                print(f"completed {fn.__name__}: Nx={nx}, a/L={frac:.3f}, ko={ko:.2f}")

    unit_dicts = [asdict(row) for row in units]
    write_csv(args.out / "advance_unit_steps.csv", unit_dicts)
    scaled, periods = scale_rows(units, args.delta_c, load_fractions)
    write_csv(args.out / "advance_scaled_steps.csv", scaled)
    write_csv(args.out / "advance_period_summary.csv", periods)

    cascades = cascade_scan(systems, k_values, load_fractions, args.delta_c)
    write_csv(args.out / "cascade_robustness_scan.csv", cascades)
    positive_controls = positive_control_scan(systems, args.delta_c)
    write_csv(args.out / "positive_control_scan.csv", positive_controls)

    convergence = [] if args.skip_convergence else convergence_scan(args.out, args.delta_c)
    summary = summarize(units, scaled, periods, cascades, convergence) if convergence else {
        "unit_step_count": len(units), "scaled_step_count": len(scaled),
        "period_count": len(periods), "cascade_state_count": len(cascades)
    }
    if convergence:
        summary.update({
            "maximum_effective_resistance_increase_fraction": max(
                float(row["resistance_increment_from_step1"]) for row in scaled
            ),
            "odd_work_sign_alternation_all_positive_ko": all(
                (float(row["odd_work"]) > 0.0 if int(row["step"]) % 2 else float(row["odd_work"]) < 0.0)
                for row in scaled if float(row["k_o"]) > 0.0
            ),
            "positive_control_dead_load_minimum_breaks": min(
                int(row["broken_bonds"]) for row in positive_controls
                if row["boundary_condition"] == "dead_load"
            ),
            "positive_control_fixed_grip_maximum_breaks": max(
                int(row["broken_bonds"]) for row in positive_controls
                if row["boundary_condition"] == "fixed_grip"
            ),
        })
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
