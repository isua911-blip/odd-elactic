#!/usr/bin/env python3
""" multi-bond crack evolution and the full-lattice-step test.

This script tests whether the triangular odd-elastic lattice used in analyses--4
supports *sustained* sub-Griffith crack propagation under an independent tensile
bond-breaking rule. It compares fixed-grip and dead-load boundary conditions,
checks transient post-break dynamics, and resolves one coarse crack advance into
the two alternating diagonal bonds crossed by a horizontal cleavage line.

The central diagnostic is that the odd interaction enhances the first half-step
but suppresses the second half-step. Thus a chirality-selected one-bond
microadvance can occur below the passive initiation load, while a complete
lattice-cell advance does not propagate in the present minimal model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import MatrixRankWarning, spsolve

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lattice_baselines import wrap_centered
from active_tip_scan import ActiveCrackedStrip
from crack_advance_work import (
    assemble_components,
    bond_extension,
    even_energy,
    integrate_relaxation,
    solve_equilibrium,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def crossing_frontier(model: ActiveCrackedStrip, removed: set[int]) -> dict[str, int]:
    center = 0.5 * model.period
    intact: list[tuple[float, int]] = []
    for bond_id, bond in enumerate(model.all_bonds):
        if bond.crosses_crack_plane and bond_id not in removed:
            assert bond.midpoint_x is not None
            distance = wrap_centered(bond.midpoint_x - center, model.period)
            intact.append((distance, bond_id))
    left = [item for item in intact if item[0] < 0.0]
    right = [item for item in intact if item[0] > 0.0]
    out: dict[str, int] = {}
    if left:
        out["left"] = max(left, key=lambda item: item[0])[1]
    if right:
        out["right"] = min(right, key=lambda item: item[0])[1]
    return out


def active_ids(model: ActiveCrackedStrip, removed: set[int]) -> list[int]:
    return [bond_id for bond_id in range(len(model.all_bonds)) if bond_id not in removed]


def top_reaction_stress(
    model: ActiveCrackedStrip, matrix, displacement: np.ndarray
) -> float:
    residual = matrix @ displacement
    top_y = np.array(
        [2 * model.node_id(i, model.ny - 1) + 1 for i in range(model.nx)],
        dtype=int,
    )
    return float(np.sum(residual[top_y])) / model.period


@dataclass
class CascadeResult:
    boundary_condition: str
    load_fraction: float
    k_o: float
    broken_bonds: int
    status: str
    first_break_side: str
    initial_left_ratio: float
    initial_right_ratio: float
    final_left_ratio: float
    final_right_ratio: float
    maximum_free_force_residual: float


class FixedGripCascade:
    def __init__(self, nx: int, ny: int, crack_half_length: float, k_o: float):
        self.model = ActiveCrackedStrip(nx, ny, crack_half_length, 1.0, k_o)
        self.initial_removed = set(self.model.removed_ids)
        self.removed = set(self.initial_removed)

    def solve(self, delta: float):
        even, odd = assemble_components(self.model, active_ids(self.model, self.removed))
        total = even + odd
        displacement, constrained, values, free, residual = solve_equilibrium(
            total, self.model, delta
        )
        return displacement, even, odd, total, residual

    def initial_unit_stress(self) -> float:
        displacement, _even, _odd, total, _residual = self.solve(1.0)
        return top_reaction_stress(self.model, total, displacement)

    def run(self, delta: float, delta_c: float, max_breaks: int = 40) -> CascadeResult:
        first_side = "none"
        initial_ratios = {"left": float("nan"), "right": float("nan")}
        final_ratios = initial_ratios.copy()
        max_residual = 0.0
        status = "arrest"
        for step in range(max_breaks + 1):
            displacement, _even, _odd, _total, residual = self.solve(delta)
            max_residual = max(max_residual, residual)
            candidates = crossing_frontier(self.model, self.removed)
            ratios = {
                side: bond_extension(self.model.all_bonds[bond_id], displacement) / delta_c
                for side, bond_id in candidates.items()
            }
            if step == 0:
                initial_ratios = {
                    "left": ratios.get("left", float("nan")),
                    "right": ratios.get("right", float("nan")),
                }
            final_ratios = {
                "left": ratios.get("left", float("nan")),
                "right": ratios.get("right", float("nan")),
            }
            if not ratios:
                status = "complete"
                break
            side = max(ratios, key=ratios.get)
            if ratios[side] < 1.0 - 1.0e-10:
                status = "arrest"
                break
            if first_side == "none":
                first_side = side
            self.removed.add(candidates[side])
        else:
            status = "max_breaks"

        return CascadeResult(
            boundary_condition="fixed_grip",
            load_fraction=float("nan"),
            k_o=self.model.k_o,
            broken_bonds=len(self.removed) - len(self.initial_removed),
            status=status,
            first_break_side=first_side,
            initial_left_ratio=initial_ratios["left"],
            initial_right_ratio=initial_ratios["right"],
            final_left_ratio=final_ratios["left"],
            final_right_ratio=final_ratios["right"],
            maximum_free_force_residual=max_residual,
        )


class DeadLoadCascade:
    def __init__(self, nx: int, ny: int, crack_half_length: float, k_o: float):
        self.model = ActiveCrackedStrip(nx, ny, crack_half_length, 1.0, k_o)
        self.initial_removed = set(self.model.removed_ids)
        self.removed = set(self.initial_removed)
        self.constrained = np.array(
            [2 * self.model.node_id(0, 0), 2 * self.model.node_id(0, 0) + 1],
            dtype=int,
        )
        free_mask = np.ones(self.model.ndof, dtype=bool)
        free_mask[self.constrained] = False
        self.free = np.arange(self.model.ndof)[free_mask]

    def external_force(self, stress: float) -> np.ndarray:
        force = np.zeros(self.model.ndof, dtype=float)
        nodal = stress * self.model.period / self.model.nx
        for i in range(self.model.nx):
            force[2 * self.model.node_id(i, self.model.ny - 1) + 1] += nodal
            force[2 * self.model.node_id(i, 0) + 1] -= nodal
        return force

    def solve(self, stress: float):
        even, odd = assemble_components(self.model, active_ids(self.model, self.removed))
        total = even + odd
        force = self.external_force(stress)
        displacement = np.zeros(self.model.ndof, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            displacement[self.free] = spsolve(
                total[self.free][:, self.free], force[self.free]
            )
        if not np.all(np.isfinite(displacement)):
            raise RuntimeError("Dead-load equilibrium became singular")
        residual = force - total @ displacement
        return displacement, even, odd, total, float(np.max(np.abs(residual[self.free])))

    def run(self, stress: float, delta_c: float, max_breaks: int = 40) -> CascadeResult:
        first_side = "none"
        initial_ratios = {"left": float("nan"), "right": float("nan")}
        final_ratios = initial_ratios.copy()
        max_residual = 0.0
        status = "arrest"
        for step in range(max_breaks + 1):
            try:
                displacement, _even, _odd, _total, residual = self.solve(stress)
            except (RuntimeError, MatrixRankWarning):
                status = "singular"
                break
            max_residual = max(max_residual, residual)
            candidates = crossing_frontier(self.model, self.removed)
            ratios = {
                side: bond_extension(self.model.all_bonds[bond_id], displacement) / delta_c
                for side, bond_id in candidates.items()
            }
            if step == 0:
                initial_ratios = {
                    "left": ratios.get("left", float("nan")),
                    "right": ratios.get("right", float("nan")),
                }
            final_ratios = {
                "left": ratios.get("left", float("nan")),
                "right": ratios.get("right", float("nan")),
            }
            if not ratios:
                status = "complete"
                break
            side = max(ratios, key=ratios.get)
            if ratios[side] < 1.0 - 1.0e-10:
                status = "arrest"
                break
            if first_side == "none":
                first_side = side
            self.removed.add(candidates[side])
        else:
            status = "max_breaks"

        return CascadeResult(
            boundary_condition="dead_load",
            load_fraction=float("nan"),
            k_o=self.model.k_o,
            broken_bonds=len(self.removed) - len(self.initial_removed),
            status=status,
            first_break_side=first_side,
            initial_left_ratio=initial_ratios["left"],
            initial_right_ratio=initial_ratios["right"],
            final_left_ratio=final_ratios["left"],
            final_right_ratio=final_ratios["right"],
            maximum_free_force_residual=max_residual,
        )


def fixed_grip_passive_reference(
    nx: int, ny: int, crack_half_length: float, delta_c: float
) -> tuple[float, float]:
    model = ActiveCrackedStrip(nx, ny, crack_half_length, 1.0, 0.0)
    diagnostics = model.static_diagnostics(delta_c)
    stress = 0.5 * (
        diagnostics["initiation_sigma_left"]
        + diagnostics["initiation_sigma_right"]
    )
    delta = 0.5 * (
        diagnostics["initiation_delta_left"]
        + diagnostics["initiation_delta_right"]
    )
    return stress, delta


def dead_load_passive_reference(
    nx: int, ny: int, crack_half_length: float, delta_c: float
) -> float:
    cascade = DeadLoadCascade(nx, ny, crack_half_length, 0.0)
    displacement, _even, _odd, _total, _residual = cascade.solve(1.0)
    extensions = [
        bond_extension(cascade.model.all_bonds[bond_id], displacement)
        for bond_id in crossing_frontier(cascade.model, cascade.removed).values()
    ]
    return delta_c / max(extensions)


def transient_after_first_cut(
    nx: int,
    ny: int,
    crack_half_length: float,
    k_o: float,
    load_fraction: float,
    passive_stress: float,
    delta_c: float,
    t_end: float,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    model = ActiveCrackedStrip(nx, ny, crack_half_length, 1.0, k_o)
    removed = set(model.removed_ids)
    old_even, old_odd = assemble_components(model, active_ids(model, removed))
    old_total = old_even + old_odd
    unit, constrained, constrained_values, free, _ = solve_equilibrium(
        old_total, model, 1.0
    )
    unit_stress = top_reaction_stress(model, old_total, unit)
    delta = load_fraction * passive_stress / unit_stress
    initial, constrained, constrained_values, free, _ = solve_equilibrium(
        old_total, model, delta
    )
    first_candidates = crossing_frontier(model, removed)
    initial_extensions = {
        side: bond_extension(model.all_bonds[bond_id], initial)
        for side, bond_id in first_candidates.items()
    }
    first_side = max(initial_extensions, key=initial_extensions.get)
    if initial_extensions[first_side] < delta_c:
        raise RuntimeError("Transient probe selected a point below first initiation")
    removed.add(first_candidates[first_side])

    new_even, new_odd = assemble_components(model, active_ids(model, removed))
    new_total = new_even + new_odd
    final, c2, cv2, f2, _ = solve_equilibrium(new_total, model, delta)
    if not (
        np.array_equal(constrained, c2)
        and np.array_equal(free, f2)
        and np.allclose(constrained_values, cv2)
    ):
        raise RuntimeError("Boundary partition changed")

    matrix_ff = new_total[free][:, free]
    offset0 = initial[free] - final[free]
    scale = max(float(np.max(np.abs(offset0))), 1.0e-12)
    solution = solve_ivp(
        lambda _time, offset: -(matrix_ff @ offset),
        (0.0, t_end),
        offset0,
        method="BDF",
        jac=-matrix_ff,
        rtol=1.0e-7,
        atol=1.0e-11 * scale,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    second_candidates = crossing_frontier(model, removed)
    rows: list[dict[str, float]] = []
    maxima = {side: -float("inf") for side in second_candidates}
    for index, time in enumerate(solution.t):
        displacement = final.copy()
        displacement[free] = final[free] + solution.y[:, index]
        row: dict[str, float] = {"time": float(time), "k_o": float(k_o)}
        for side, bond_id in second_candidates.items():
            ratio = bond_extension(model.all_bonds[bond_id], displacement) / delta_c
            row[f"{side}_next_ratio"] = ratio
            maxima[side] = max(maxima[side], ratio)
        rows.append(row)
    # Append exact equilibrium to make the asymptote explicit.
    final_row: dict[str, float] = {"time": float(t_end), "k_o": float(k_o)}
    for side, bond_id in second_candidates.items():
        final_row[f"{side}_next_ratio"] = (
            bond_extension(model.all_bonds[bond_id], final) / delta_c
        )
    rows.append(final_row)
    summary = {
        "k_o": k_o,
        "load_fraction": load_fraction,
        "first_break_side": first_side,
        "first_extension_ratio": initial_extensions[first_side] / delta_c,
        "maximum_next_left_ratio": maxima.get("left", float("nan")),
        "maximum_next_right_ratio": maxima.get("right", float("nan")),
        "n_time_steps": len(solution.t),
    }
    return rows, summary


def protocol_step(
    model: ActiveCrackedStrip,
    removed: set[int],
    cut_id: int,
    t_end: float,
) -> tuple[dict[str, float], set[int]]:
    old_even, old_odd = assemble_components(model, active_ids(model, removed))
    old_total = old_even + old_odd
    new_removed = set(removed)
    new_removed.add(cut_id)
    new_even, new_odd = assemble_components(model, active_ids(model, new_removed))
    new_total = new_even + new_odd

    initial, constrained, values, free, initial_residual = solve_equilibrium(
        old_total, model, 1.0
    )
    final, c2, v2, f2, final_residual = solve_equilibrium(new_total, model, 1.0)
    if not (
        np.array_equal(constrained, c2)
        and np.array_equal(free, f2)
        and np.allclose(values, v2)
    ):
        raise RuntimeError("Boundary partition changed")
    odd_work, dissipation, final_relative_norm, n_steps = integrate_relaxation(
        new_total,
        new_odd,
        initial,
        final,
        constrained,
        values,
        free,
        t_end,
    )
    energy_initial = even_energy(old_even, initial)
    energy_final = even_energy(new_even, final)
    extension = bond_extension(model.all_bonds[cut_id], initial)
    cut_energy = 0.5 * extension**2
    protocol_work = odd_work - (energy_final - energy_initial)
    balance = protocol_work - dissipation - cut_energy
    return {
        "extension_unit": extension,
        "odd_work_unit": odd_work,
        "dissipation_unit": dissipation,
        "protocol_work_unit": protocol_work,
        "cut_energy_unit": cut_energy,
        "balance_unit": balance,
        "initial_force_residual": initial_residual,
        "final_force_residual": final_residual,
        "final_relative_norm": final_relative_norm,
        "n_time_steps": float(n_steps),
    }, new_removed


def two_step_protocol_scan(
    nx: int,
    ny: int,
    crack_half_length: float,
    delta_c: float,
    load_fraction: float,
    passive_stress: float,
    k_o_values: list[float],
    t_end: float,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    unit_results: dict[float, tuple[dict[str, float], dict[str, float], float]] = {}
    for k_o in k_o_values:
        model = ActiveCrackedStrip(nx, ny, crack_half_length, 1.0, k_o)
        removed = set(model.removed_ids)
        even, odd = assemble_components(model, active_ids(model, removed))
        total = even + odd
        unit, _c, _cv, _f, _res = solve_equilibrium(total, model, 1.0)
        unit_stress = top_reaction_stress(model, total, unit)
        first_id = crossing_frontier(model, removed)["right"]
        first, removed = protocol_step(model, removed, first_id, t_end)
        second_id = crossing_frontier(model, removed)["right"]
        second, removed = protocol_step(model, removed, second_id, t_end)
        unit_results[k_o] = (first, second, unit_stress)

    passive_first, passive_second, passive_unit_stress = unit_results[0.0]
    passive_delta_g = passive_stress / passive_unit_stress
    resistance_first = passive_first["protocol_work_unit"] * passive_delta_g**2
    passive_second_critical_delta = delta_c / passive_second["extension_unit"]
    resistance_second = (
        passive_second["protocol_work_unit"] * passive_second_critical_delta**2
    )
    resistance_cell = resistance_first + resistance_second

    rows: list[dict[str, float]] = []
    for k_o in k_o_values:
        first, second, unit_stress = unit_results[k_o]
        delta = load_fraction * passive_stress / unit_stress
        scale = delta**2
        row = {
            "k_o": k_o,
            "applied_delta": delta,
            "first_extension_ratio": first["extension_unit"] * delta / delta_c,
            "second_extension_ratio": second["extension_unit"] * delta / delta_c,
            "first_odd_work": first["odd_work_unit"] * scale,
            "second_odd_work": second["odd_work_unit"] * scale,
            "cell_odd_work": (
                first["odd_work_unit"] + second["odd_work_unit"]
            ) * scale,
            "first_protocol_work": first["protocol_work_unit"] * scale,
            "second_protocol_work": second["protocol_work_unit"] * scale,
            "cell_protocol_work": (
                first["protocol_work_unit"] + second["protocol_work_unit"]
            ) * scale,
            "first_protocol_ratio": (
                first["protocol_work_unit"] * scale / resistance_first
            ),
            "second_protocol_ratio": (
                second["protocol_work_unit"] * scale / resistance_second
            ),
            "cell_protocol_ratio": (
                (first["protocol_work_unit"] + second["protocol_work_unit"])
                * scale
                / resistance_cell
            ),
            "first_balance_residual": first["balance_unit"] * scale,
            "second_balance_residual": second["balance_unit"] * scale,
        }
        rows.append(row)
    summary = {
        "passive_first_protocol_resistance": resistance_first,
        "passive_second_protocol_resistance": resistance_second,
        "passive_cell_protocol_resistance": resistance_cell,
        "maximum_abs_balance_residual": max(
            max(abs(row["first_balance_residual"]), abs(row["second_balance_residual"]))
            for row in rows
        ),
    }
    return rows, summary


def phase_plot(
    path: Path,
    rows: list[dict[str, object]],
    p_values: list[float],
    k_values: list[float],
    title: str,
) -> None:
    matrix = np.zeros((len(p_values), len(k_values)), dtype=float)
    for row in rows:
        i = p_values.index(float(row["load_fraction"]))
        j = k_values.index(float(row["k_o"]))
        matrix[i, j] = float(row["broken_bonds"])
    fig, axis = plt.subplots(figsize=(6.8, 4.7))
    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=[min(k_values), max(k_values), min(p_values), max(p_values)],
        interpolation="nearest",
    )
    axis.set_xlabel(r"odd coefficient $k_o/k$")
    axis.set_ylabel(r"initial load $P/P_G$")
    axis.set_title(title)
    bar = fig.colorbar(image, ax=axis)
    bar.set_label("number of broken crack-plane bonds")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("propagation_limit_results"))
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=36)
    parser.add_argument("--crack-half-length", type=float, default=6.0)
    parser.add_argument("--transient-t-end", type=float, default=30000.0)
    parser.add_argument("--protocol-t-end", type=float, default=30000.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    nx, ny, crack_half_length = args.nx, args.ny, args.crack_half_length
    delta_c = 0.02
    p_values = [0.80, 0.85, 0.90, 0.95, 0.98]
    k_values = [0.00, 0.05, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]

    passive_fixed_stress, passive_fixed_delta = fixed_grip_passive_reference(
        nx, ny, crack_half_length, delta_c
    )
    passive_dead_stress = dead_load_passive_reference(
        nx, ny, crack_half_length, delta_c
    )

    fixed_rows: list[dict[str, object]] = []
    dead_rows: list[dict[str, object]] = []
    for p in p_values:
        for k_o in k_values:
            fixed = FixedGripCascade(nx, ny, crack_half_length, k_o)
            delta = p * passive_fixed_stress / fixed.initial_unit_stress()
            fixed_result = fixed.run(delta, delta_c)
            fixed_row = vars(fixed_result)
            fixed_row["load_fraction"] = p
            fixed_row["applied_delta"] = delta
            fixed_rows.append(fixed_row)

            dead = DeadLoadCascade(nx, ny, crack_half_length, k_o)
            dead_result = dead.run(p * passive_dead_stress, delta_c)
            dead_row = vars(dead_result)
            dead_row["load_fraction"] = p
            dead_row["applied_stress"] = p * passive_dead_stress
            dead_rows.append(dead_row)

    write_csv(args.out / "fixed_grip_phase_scan.csv", fixed_rows)
    write_csv(args.out / "dead_load_phase_scan.csv", dead_rows)
    phase_plot(
        args.out / "fixed_grip_broken_bonds_phase.pdf",
        fixed_rows,
        p_values,
        k_values,
        "Fixed grip: sub-Griffith crack-plane damage",
    )
    phase_plot(
        args.out / "dead_load_broken_bonds_phase.pdf",
        dead_rows,
        p_values,
        k_values,
        "Dead load: sub-Griffith crack-plane damage",
    )

    transient_rows: list[dict[str, float]] = []
    transient_summaries: list[dict[str, object]] = []
    for k_o in (0.12, 0.20, 0.40):
        rows, summary = transient_after_first_cut(
            nx,
            ny,
            crack_half_length,
            k_o,
            0.90,
            passive_fixed_stress,
            delta_c,
            args.transient_t_end,
        )
        transient_rows.extend(rows)
        transient_summaries.append(summary)
    write_csv(args.out / "next_bond_transient.csv", transient_rows)
    write_csv(args.out / "next_bond_transient_summary.csv", transient_summaries)

    fig, axis = plt.subplots(figsize=(6.8, 4.7))
    for k_o in (0.12, 0.20, 0.40):
        selected = [row for row in transient_rows if abs(row["k_o"] - k_o) < 1.0e-12]
        time = np.array([row["time"] for row in selected])
        favored = np.array([row["right_next_ratio"] for row in selected])
        axis.semilogx(time + 1.0e-3, favored, label=fr"$k_o={k_o:.2f}$")
    axis.axhline(1.0, linestyle="--", linewidth=1.0)
    axis.set_xlabel(r"time after first bond break $t$")
    axis.set_ylabel(r"next right-tip bond extension $\delta\ell/\delta_c$")
    axis.set_title(r"Post-break transient at $P/P_G=0.9$")
    axis.legend()
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "next_bond_transient.pdf")
    plt.close(fig)

    protocol_k = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    protocol_rows, protocol_summary = two_step_protocol_scan(
        nx,
        ny,
        crack_half_length,
        delta_c,
        0.90,
        passive_fixed_stress,
        protocol_k,
        args.protocol_t_end,
    )
    write_csv(args.out / "two_step_protocol_scan.csv", protocol_rows)

    ko = np.array([row["k_o"] for row in protocol_rows])
    ratio1 = np.array([row["first_protocol_ratio"] for row in protocol_rows])
    ratio2 = np.array([row["second_protocol_ratio"] for row in protocol_rows])
    ratio_cell = np.array([row["cell_protocol_ratio"] for row in protocol_rows])
    fig, axis = plt.subplots(figsize=(6.8, 4.7))
    axis.plot(ko, ratio1, "o-", label="first half-step")
    axis.plot(ko, ratio2, "s-", label="second half-step")
    axis.plot(ko, ratio_cell, "^-", label="complete lattice step")
    axis.axhline(1.0, linestyle="--", linewidth=1.0)
    axis.set_xlabel(r"odd coefficient $k_o/k$")
    axis.set_ylabel("protocol work / passive step resistance")
    axis.set_title(r"Two-bond crack advance at $P/P_G=0.9$")
    axis.legend()
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "two_step_protocol_ratios.pdf")
    plt.close(fig)

    odd1 = np.array([row["first_odd_work"] for row in protocol_rows])
    odd2 = np.array([row["second_odd_work"] for row in protocol_rows])
    odd_cell = np.array([row["cell_odd_work"] for row in protocol_rows])
    fig, axis = plt.subplots(figsize=(6.8, 4.7))
    axis.plot(ko, odd1, "o-", label="first half-step")
    axis.plot(ko, odd2, "s-", label="second half-step")
    axis.plot(ko, odd_cell, "^-", label="two-step sum")
    axis.axhline(0.0, linestyle="--", linewidth=1.0)
    axis.set_xlabel(r"odd coefficient $k_o/k$")
    axis.set_ylabel(r"odd-force work $W_{\rm odd}$")
    axis.set_title("Odd-work cancellation over a complete lattice step")
    axis.legend()
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "two_step_odd_work_cancellation.pdf")
    plt.close(fig)

    # Positive control: the dead-load algorithm must detect passive runaway above P_G.
    positive_control_rows: list[dict[str, object]] = []
    for p, k_o in ((1.001, 0.0), (1.01, 0.0), (1.01, 0.10), (1.01, 0.20)):
        control = DeadLoadCascade(nx, ny, crack_half_length, k_o)
        result = control.run(p * passive_dead_stress, delta_c, max_breaks=30)
        row = vars(result)
        row["load_fraction"] = p
        positive_control_rows.append(row)
    write_csv(args.out / "above_griffith_positive_control.csv", positive_control_rows)

    all_sub = fixed_rows + dead_rows
    maximum_sub_breaks = max(int(row["broken_bonds"]) for row in all_sub)
    number_sustained = sum(int(row["broken_bonds"]) >= 2 for row in all_sub)
    number_one_step = sum(int(row["broken_bonds"]) == 1 for row in all_sub)
    maximum_next_ratio = max(
        max(
            float(row["maximum_next_left_ratio"]),
            float(row["maximum_next_right_ratio"]),
        )
        for row in transient_summaries
    )
    passive_runaway_breaks = int(positive_control_rows[0]["broken_bonds"])

    summary = {
        "grid_nx": nx,
        "grid_ny": ny,
        "crack_half_length": crack_half_length,
        "delta_c": delta_c,
        "passive_fixed_grip_initiation_stress": passive_fixed_stress,
        "passive_fixed_grip_initiation_delta": passive_fixed_delta,
        "passive_dead_load_initiation_stress": passive_dead_stress,
        "subgriffith_scan_points": len(all_sub),
        "maximum_broken_bonds_in_subgriffith_scan": maximum_sub_breaks,
        "number_of_one_step_microadvance_points": number_one_step,
        "number_of_sustained_points_ge_2_bonds": number_sustained,
        "maximum_next_bond_transient_ratio": maximum_next_ratio,
        "passive_dead_load_positive_control_breaks_at_1p001_PG": passive_runaway_breaks,
        "minimum_cell_protocol_ratio": float(np.min(ratio_cell)),
        "maximum_cell_protocol_ratio": float(np.max(ratio_cell)),
        "minimum_cell_odd_work": float(np.min(odd_cell)),
        "maximum_cell_odd_work": float(np.max(odd_cell)),
        **protocol_summary,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
