#!/usr/bin/env python3
""" apparent J-integral and odd configurational-source tests.

The script coarse-grains the relaxed displacement field of the active triangular
lattice with a local quadratic polynomial.  It then evaluates

    J_app(Gamma) = integral_Gamma [W_e n_x - (sigma n).u_,x] ds

on nested rectangular keyhole contours around either crack tip and independently
computes

    Q_odd = - integral_annulus sigma_odd : partial_x(grad u) dA.

A matched passive calculation is used to quantify and subtract the residual
contour drift caused by lattice discreteness and local field reconstruction.
The main comparison is therefore

    Delta J_odd = [J(R_out)-J(R_in)]_ko - [J(R_out)-J(R_in)]_0

versus Q_odd over the same annulus.

The left-tip calculation is performed in reflected local coordinates, so the
local crack-advance direction is always +x.  Under reflection the local odd
coefficient is k_o^local = s k_o, with s=+1 for the right tip and -1 for the
left tip.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lattice_baselines import A1, A2, A3, R90, SQRT3
from active_tip_scan import ActiveCrackedStrip


@dataclass(frozen=True)
class FieldValue:
    gradient: np.ndarray
    stress_even: np.ndarray
    stress_odd: np.ndarray
    energy_even: float
    source_odd: float
    equilibrium_residual_source: float


class LocalTipField:
    """Quadratic moving-least-squares reconstruction in tip-local coordinates."""

    def __init__(
        self,
        model: ActiveCrackedStrip,
        displacement: np.ndarray,
        tip: str,
        fit_radius: float = 4.2,
        min_neighbors: int = 12,
    ) -> None:
        if tip not in {"left", "right"}:
            raise ValueError("tip must be 'left' or 'right'")
        if fit_radius <= 1.5:
            raise ValueError("fit_radius is too small for a quadratic reconstruction")
        self.model = model
        self.tip = tip
        self.direction = 1.0 if tip == "right" else -1.0
        self.tip_x_global = 0.5 * model.period + self.direction * model.a_eff
        self.crack_y = 0.5 * (
            model.positions[model.node_id(0, model.j_lower), 1]
            + model.positions[model.node_id(0, model.j_lower + 1), 1]
        )
        self.positions = model.positions.copy()
        self.positions[:, 0] = self.direction * (
            self.positions[:, 0] - self.tip_x_global
        )
        self.displacement = displacement.copy()
        self.displacement[:, 0] *= self.direction
        self.tree = cKDTree(self.positions)
        self.fit_radius = float(fit_radius)
        self.min_neighbors = int(min_neighbors)
        self.k = float(model.k)
        self.k_o_local = self.direction * float(model.k_o)
        self.area_cell = SQRT3 / 2.0
        self.directions = (A1, A2, A3)

    def _neighbor_ids(self, x: float, y: float) -> np.ndarray:
        ids = np.asarray(
            self.tree.query_ball_point([x, y], self.fit_radius), dtype=int
        )
        # Behind the tip, do not fit across the displacement discontinuity.
        if x < -1.0e-9:
            if y >= self.crack_y:
                ids = ids[self.positions[ids, 1] > self.crack_y]
            else:
                ids = ids[self.positions[ids, 1] < self.crack_y]
        if len(ids) >= self.min_neighbors:
            return ids

        distance = np.linalg.norm(self.positions - np.array([x, y]), axis=1)
        admissible = np.ones(len(distance), dtype=bool)
        if x < -1.0e-9:
            if y >= self.crack_y:
                admissible &= self.positions[:, 1] > self.crack_y
            else:
                admissible &= self.positions[:, 1] < self.crack_y
        ranked = np.argsort(np.where(admissible, distance, np.inf))
        ids = ranked[: self.min_neighbors]
        if not np.all(np.isfinite(distance[ids])):
            raise RuntimeError("Insufficient neighbors for local reconstruction")
        return ids

    def _coefficients(self, x: float, y: float) -> np.ndarray:
        ids = self._neighbor_ids(x, y)
        dx = self.positions[ids, 0] - x
        dy = self.positions[ids, 1] - y
        radius = np.hypot(dx, dy)
        design = np.column_stack(
            [
                np.ones(len(ids)),
                dx,
                dy,
                0.5 * dx * dx,
                dx * dy,
                0.5 * dy * dy,
            ]
        )
        weights = np.exp(-((radius / (0.65 * self.fit_radius)) ** 2))
        weighted_design = design * weights[:, None]
        weighted_values = self.displacement[ids] * weights[:, None]
        coefficients, _, rank, _ = np.linalg.lstsq(
            weighted_design, weighted_values, rcond=None
        )
        if rank < 6:
            raise RuntimeError("Rank-deficient local quadratic reconstruction")
        return coefficients

    def _constitutive(
        self, gradient: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        stress_even = np.zeros((2, 2), dtype=float)
        stress_odd = np.zeros((2, 2), dtype=float)
        energy_even = 0.0
        for normal in self.directions:
            tangent = R90 @ normal
            extension = float(normal @ gradient @ normal)
            stress_even += (
                self.k
                * extension
                * np.outer(normal, normal)
                / self.area_cell
            )
            stress_odd += (
                -self.k_o_local
                * extension
                * np.outer(tangent, normal)
                / self.area_cell
            )
            energy_even += (
                0.5 * self.k * extension * extension / self.area_cell
            )
        return stress_even, stress_odd, float(energy_even)

    def evaluate(self, x: float, y: float) -> FieldValue:
        c = self._coefficients(x, y)
        gradient = np.array(
            [[c[1, 0], c[2, 0]], [c[1, 1], c[2, 1]]], dtype=float
        )
        gradient_x = np.array(
            [[c[3, 0], c[4, 0]], [c[3, 1], c[4, 1]]], dtype=float
        )
        gradient_y = np.array(
            [[c[4, 0], c[5, 0]], [c[4, 1], c[5, 1]]], dtype=float
        )
        stress_even, stress_odd, energy_even = self._constitutive(gradient)
        stress_even_x, stress_odd_x, _ = self._constitutive(gradient_x)
        stress_even_y, stress_odd_y, _ = self._constitutive(gradient_y)
        divergence = (stress_even_x + stress_odd_x)[:, 0] + (
            stress_even_y + stress_odd_y
        )[:, 1]
        source_odd = -float(np.sum(stress_odd * gradient_x))
        residual_source = -float(divergence @ gradient[:, 0])
        return FieldValue(
            gradient=gradient,
            stress_even=stress_even,
            stress_odd=stress_odd,
            energy_even=energy_even,
            source_odd=source_odd,
            equilibrium_residual_source=residual_source,
        )


def integrate_segment(
    field: LocalTipField,
    point_0: tuple[float, float],
    point_1: tuple[float, float],
    outward_normal: tuple[float, float],
    target_step: float,
) -> float:
    p0 = np.asarray(point_0, dtype=float)
    p1 = np.asarray(point_1, dtype=float)
    normal = np.asarray(outward_normal, dtype=float)
    length = float(np.linalg.norm(p1 - p0))
    n_steps = max(2, int(math.ceil(length / target_step)))
    total = 0.0
    for fraction in (np.arange(n_steps) + 0.5) / n_steps:
        point = p0 + fraction * (p1 - p0)
        value = field.evaluate(float(point[0]), float(point[1]))
        total_stress = value.stress_even + value.stress_odd
        displacement_x = value.gradient[:, 0]
        integrand = (
            value.energy_even * normal[0]
            - float(displacement_x @ (total_stress @ normal))
        )
        total += integrand * length / n_steps
    return float(total)


def keyhole_j(
    field: LocalTipField,
    radius: float,
    line_step: float = 0.10,
) -> float:
    """Rectangular keyhole contour; crack-face closure is not integrated."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    x_left, x_right = -radius, radius
    y_top = field.crack_y + radius
    y_bottom = field.crack_y - radius
    y_crack = field.crack_y
    segments = (
        ((x_left, y_crack), (x_left, y_top), (-1.0, 0.0)),
        ((x_left, y_top), (x_right, y_top), (0.0, 1.0)),
        ((x_right, y_top), (x_right, y_bottom), (1.0, 0.0)),
        ((x_right, y_bottom), (x_left, y_bottom), (0.0, -1.0)),
        ((x_left, y_bottom), (x_left, y_crack), (-1.0, 0.0)),
    )
    return float(
        sum(
            integrate_segment(field, p0, p1, normal, line_step)
            for p0, p1, normal in segments
        )
    )


def annulus_sources(
    field: LocalTipField,
    inner_radius: float,
    outer_radius: float,
    area_step: float = 0.28,
) -> tuple[float, float, int]:
    if not 0 < inner_radius < outer_radius:
        raise ValueError("Require 0 < inner_radius < outer_radius")
    source_odd = 0.0
    source_residual = 0.0
    sample_count = 0
    y_values = np.arange(
        field.crack_y - outer_radius + 0.5 * area_step,
        field.crack_y + outer_radius,
        area_step,
    )
    x_values = np.arange(
        -outer_radius + 0.5 * area_step, outer_radius, area_step
    )
    cell_area = area_step * area_step
    for y in y_values:
        for x in x_values:
            radius = max(abs(float(x)), abs(float(y - field.crack_y)))
            if inner_radius <= radius < outer_radius:
                value = field.evaluate(float(x), float(y))
                source_odd += value.source_odd * cell_area
                source_residual += value.equilibrium_residual_source * cell_area
                sample_count += 1
    return float(source_odd), float(source_residual), sample_count


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fit_through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope = float(x @ y / (x @ x))
    residual = y - slope * x
    denominator = float(y @ y)
    r2 = 1.0 - float(residual @ residual) / denominator if denominator else 1.0
    return slope, r2


def solve_field(
    nx: int,
    ny: int,
    crack_half_length: float,
    k_o: float,
    tip: str,
    fit_radius: float,
) -> tuple[ActiveCrackedStrip, LocalTipField, float]:
    model = ActiveCrackedStrip(
        nx=nx,
        ny=ny,
        crack_half_length=crack_half_length,
        k=1.0,
        k_o=k_o,
    )
    displacement, _, residual = model.solve(delta=1.0)
    field = LocalTipField(
        model=model,
        displacement=displacement,
        tip=tip,
        fit_radius=fit_radius,
    )
    return model, field, residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("apparent_j_results"))
    parser.add_argument("--nx", type=int, default=80)
    parser.add_argument("--ny", type=int, default=56)
    parser.add_argument("--crack-half-length", type=float, default=10.0)
    parser.add_argument("--fit-radius", type=float, default=4.2)
    parser.add_argument("--line-step", type=float, default=0.10)
    parser.add_argument("--area-step", type=float, default=0.34)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    radii = np.arange(4.0, 8.0 + 1.0e-12, 1.0)
    inner_radius, outer_radius = float(radii[0]), float(radii[-1])
    positive_ko = (0.05, 0.10, 0.15, 0.20)

    contour_rows: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []
    mirror_rows: list[dict[str, object]] = []
    fields: dict[tuple[float, str], LocalTipField] = {}
    j_cache: dict[tuple[float, str], np.ndarray] = {}
    q_cache: dict[tuple[float, str], tuple[float, float, int]] = {}
    solve_residuals: list[float] = []

    # Passive reference at the right tip.
    _, passive_field, passive_residual = solve_field(
        args.nx,
        args.ny,
        args.crack_half_length,
        0.0,
        "right",
        args.fit_radius,
    )
    solve_residuals.append(passive_residual)
    passive_j = np.array(
        [keyhole_j(passive_field, float(r), args.line_step) for r in radii]
    )
    passive_drift = float(passive_j[-1] - passive_j[0])
    for radius, j_value in zip(radii, passive_j):
        contour_rows.append(
            {
                "tip": "right",
                "global_k_o": 0.0,
                "local_k_o": 0.0,
                "radius": radius,
                "J_app": j_value,
            }
        )

    # Favored tip for each chirality: right for +k_o and left for -k_o.
    for k_o in positive_ko:
        for global_k_o, tip in ((k_o, "right"), (-k_o, "left")):
            _, field, residual = solve_field(
                args.nx,
                args.ny,
                args.crack_half_length,
                global_k_o,
                tip,
                args.fit_radius,
            )
            fields[(global_k_o, tip)] = field
            solve_residuals.append(residual)
            j_values = np.array(
                [keyhole_j(field, float(r), args.line_step) for r in radii]
            )
            j_cache[(global_k_o, tip)] = j_values
            for radius, j_value in zip(radii, j_values):
                contour_rows.append(
                    {
                        "tip": tip,
                        "global_k_o": global_k_o,
                        "local_k_o": field.k_o_local,
                        "radius": radius,
                        "J_app": j_value,
                    }
                )
            q_odd, q_residual, count = annulus_sources(
                field,
                inner_radius,
                outer_radius,
                args.area_step,
            )
            q_cache[(global_k_o, tip)] = (q_odd, q_residual, count)
            raw_drift = float(j_values[-1] - j_values[0])
            active_excess = raw_drift - passive_drift
            balance_rows.append(
                {
                    "tip": tip,
                    "global_k_o": global_k_o,
                    "local_k_o": field.k_o_local,
                    "R_inner": inner_radius,
                    "R_outer": outer_radius,
                    "J_inner": j_values[0],
                    "J_outer": j_values[-1],
                    "raw_contour_drift": raw_drift,
                    "passive_contour_drift": passive_drift,
                    "active_excess_contour_drift": active_excess,
                    "Q_odd": q_odd,
                    "Q_equilibrium_residual": q_residual,
                    "active_excess_minus_Q_odd": active_excess - q_odd,
                    "relative_mismatch_to_Q_odd": abs(active_excess - q_odd)
                    / max(abs(q_odd), 1.0e-30),
                    "annulus_quadrature_points": count,
                }
            )

        right_field = fields[(k_o, "right")]
        left_field = fields[(-k_o, "left")]
        right_j = j_cache[(k_o, "right")]
        left_j = j_cache[(-k_o, "left")]
        q_right = q_cache[(k_o, "right")][0]
        q_left = q_cache[(-k_o, "left")][0]
        mirror_rows.append(
            {
                "abs_k_o": k_o,
                "max_abs_J_mirror_error": float(np.max(np.abs(right_j - left_j))),
                "Q_odd_right_positive": q_right,
                "Q_odd_left_negative": q_left,
                "abs_Q_odd_mirror_error": abs(q_right - q_left),
            }
        )

    write_csv(args.out / "j_contour_scan.csv", contour_rows)
    write_csv(args.out / "annulus_balance.csv", balance_rows)
    write_csv(args.out / "chirality_mirror_symmetry.csv", mirror_rows)

    # Fixed right-tip sign reversal in the small-|k_o| range.
    sign_rows: list[dict[str, object]] = []
    for k_o in (-0.10, -0.05, 0.0, 0.05, 0.10):
        if k_o == 0.0:
            field = passive_field
            j_values = passive_j
            q_odd = 0.0
        elif k_o > 0.0:
            field = fields[(k_o, "right")]
            j_values = j_cache[(k_o, "right")]
            q_odd = q_cache[(k_o, "right")][0]
        else:
            _, field, residual = solve_field(
                args.nx, args.ny, args.crack_half_length, k_o, "right", args.fit_radius
            )
            solve_residuals.append(residual)
            j_values = np.array(
                [keyhole_j(field, float(r), args.line_step) for r in radii]
            )
            q_odd = annulus_sources(
                field, inner_radius, outer_radius, args.area_step
            )[0]
        raw_drift = float(j_values[-1] - j_values[0])
        sign_rows.append(
            {
                "global_k_o": k_o,
                "raw_contour_drift": raw_drift,
                "active_excess_contour_drift": raw_drift - passive_drift,
                "Q_odd": q_odd,
            }
        )
    write_csv(args.out / "fixed_right_tip_sign_scan.csv", sign_rows)

    # Main regression uses one representative from each mirrored pair.
    main_rows = [row for row in balance_rows if row["tip"] == "right"]
    q_values = np.array([float(row["Q_odd"]) for row in main_rows])
    delta_values = np.array(
        [float(row["active_excess_contour_drift"]) for row in main_rows]
    )
    slope, r2 = fit_through_origin(q_values, delta_values)

    passive_relative_spread = float(
        (np.max(passive_j) - np.min(passive_j)) / np.mean(passive_j)
    )
    relative_mismatches = np.array(
        [float(row["relative_mismatch_to_Q_odd"]) for row in main_rows]
    )

    # Figure 1: apparent J versus contour size.
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.plot(radii, passive_j, "o-", label=r"$k_o=0$")
    for k_o in positive_ko:
        values = j_cache[(k_o, "right")]
        ax.plot(radii, values, "o-", label=rf"$k_o={k_o:.2f}$")
    ax.set_xlabel(r"keyhole contour radius $R$")
    ax.set_ylabel(r"apparent $J_{\rm app}(R)$ at unit opening")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "Japp_vs_contour_radius.pdf")
    plt.close(fig)

    # Figure 2: central validation plot.
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot(q_values, delta_values, "o", label="lattice data")
    limit = 1.08 * max(float(np.max(np.abs(q_values))), float(np.max(np.abs(delta_values))))
    line = np.linspace(-limit, limit, 200)
    ax.plot(line, line, "--", label="unit slope")
    ax.plot(line, slope * line, ":", label=rf"fit slope $={slope:.3f}$")
    ax.set_xlabel(r"odd configurational source $Q_o^{\rm conf}$")
    ax.set_ylabel(r"active excess contour drift $\Delta J_{\rm odd}$")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "active_excess_deltaJ_vs_Qodd.pdf")
    plt.close(fig)

    # Figure 3: dependence on odd coefficient.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    local_ko = np.array([float(row["local_k_o"]) for row in main_rows])
    ax.plot(local_ko, delta_values, "o-", label=r"$\Delta J_{\rm odd}$")
    ax.plot(local_ko, q_values, "s--", label=r"$Q_o^{\rm conf}$")
    ax.set_xlabel(r"local odd coefficient $k_o^{\rm local}$")
    ax.set_ylabel("annular configurational contribution")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "active_excess_and_Qodd_vs_ko.pdf")
    plt.close(fig)

    # Figure 4: sign reversal at a fixed physical tip.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    sign_ko = np.array([float(row["global_k_o"]) for row in sign_rows])
    sign_delta = np.array(
        [float(row["active_excess_contour_drift"]) for row in sign_rows]
    )
    sign_q = np.array([float(row["Q_odd"]) for row in sign_rows])
    ax.plot(sign_ko, sign_delta, "o-", label=r"$\Delta J_{\rm odd}$")
    ax.plot(sign_ko, sign_q, "s--", label=r"$Q_o^{\rm conf}$")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel(r"global odd coefficient $k_o$")
    ax.set_ylabel("right-tip annular contribution")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "fixed_right_tip_chirality_reversal.pdf")
    plt.close(fig)

    summary = {
        "grid_nx": args.nx,
        "grid_ny": args.ny,
        "crack_half_length": args.crack_half_length,
        "fit_radius": args.fit_radius,
        "line_step": args.line_step,
        "area_step": args.area_step,
        "inner_contour_radius": inner_radius,
        "outer_contour_radius": outer_radius,
        "passive_J_mean": float(np.mean(passive_j)),
        "passive_J_relative_spread": passive_relative_spread,
        "passive_contour_drift": passive_drift,
        "deltaJ_vs_Qodd_fit_slope_origin": slope,
        "deltaJ_vs_Qodd_fit_r2_origin": r2,
        "main_max_relative_mismatch": float(np.max(relative_mismatches)),
        "main_mean_relative_mismatch": float(np.mean(relative_mismatches)),
        "chirality_max_abs_J_mirror_error": float(
            max(row["max_abs_J_mirror_error"] for row in mirror_rows)
        ),
        "chirality_max_abs_Q_mirror_error": float(
            max(row["abs_Q_odd_mirror_error"] for row in mirror_rows)
        ),
        "maximum_free_force_balance_residual": float(max(solve_residuals)),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
