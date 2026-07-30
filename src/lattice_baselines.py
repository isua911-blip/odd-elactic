#!/usr/bin/env python3
""" numerical baselines for an active odd-elastic triangular lattice.

This script supplies two reproducible tests for the first-paper project.

A. Finite periodic-cell quasistatic shear cycle
   e2 = A cos(theta), e3 = A sin(theta)
   It computes the external constitutive work directly from all lattice bonds and
   verifies W_loop/area = 2*pi*K_odd*A^2, oddness in k_o, and A^2 scaling.

B. Passive pre-cracked strip (k_o = 0)
   A triangular lattice is periodic in x and displacement-controlled in y.
   A centered crack is introduced by deleting bonds crossing one horizontal
   interface. The lattice is relaxed by solving the linear force-balance problem.
   The finite-lattice initiation load is defined independently by a fixed tensile
   bond-extension threshold at the first intact crack-tip bond.

Outputs are written to an output directory given by --out.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse, stats
from scipy.sparse.linalg import spsolve

SQRT3 = math.sqrt(3.0)
A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, SQRT3 / 2.0])
A3 = A2 - A1
R90 = np.array([[0.0, -1.0], [1.0, 0.0]])


@dataclass(frozen=True)
class Bond:
    i: int
    j: int
    n: np.ndarray
    midpoint_x: float | None = None
    crosses_crack_plane: bool = False


@dataclass
class CrackSolveResult:
    target_half_length: float
    effective_half_length: float
    crack_tip_candidate_ids: list[int]
    candidate_extensions_unit: np.ndarray
    max_tensile_extension_unit: float
    reaction_unit: float
    remote_stress_unit: float
    initiation_displacement: float
    initiation_force: float
    initiation_remote_stress: float
    energy_unit: float
    clapeyron_relative_error: float
    free_residual_inf: float
    removed_bonds: int


def wrap_centered(x: float, period: float) -> float:
    """Map x to (-period/2, period/2]."""
    return (x + 0.5 * period) % period - 0.5 * period


def irreducible_H(e2: float, e3: float) -> np.ndarray:
    """Pure-shear displacement gradient for irreducible strains e2,e3."""
    return 0.5 * np.array([[e2, e3], [e3, -e2]], dtype=float)


def periodic_cycle_work_density(
    nx: int,
    ny: int,
    k: float,
    k_o: float,
    amplitude: float,
    n_steps: int = 4000,
    orientation: int = 1,
) -> tuple[float, float, float]:
    """Bond-sum work density around a quasistatic pure-shear cycle.

    Returns total, conservative, and odd contributions to external work per area.
    The finite periodic cell has nx*ny nodes and three unique bonds per node.
    """
    if orientation not in (-1, 1):
        raise ValueError("orientation must be +1 or -1")
    n_cells = nx * ny
    area = n_cells * SQRT3 / 2.0
    bond_vectors = (A1, A2, A3)
    tangents = tuple(R90 @ b for b in bond_vectors)

    theta = np.linspace(0.0, orientation * 2.0 * math.pi, n_steps + 1)
    q_prev: list[np.ndarray] = []
    fe_prev: list[np.ndarray] = []
    fo_prev: list[np.ndarray] = []

    H0 = irreducible_H(amplitude, 0.0)
    for b, t in zip(bond_vectors, tangents):
        q = H0 @ b
        qn = float(q @ b)
        q_prev.append(q)
        fe_prev.append(k * qn * b)
        fo_prev.append(-k_o * qn * t)

    w_e = 0.0
    w_o = 0.0
    for th in theta[1:]:
        H = irreducible_H(amplitude * math.cos(th), amplitude * math.sin(th))
        for d, (b, t) in enumerate(zip(bond_vectors, tangents)):
            q = H @ b
            qn = float(q @ b)
            fe = k * qn * b
            fo = -k_o * qn * t
            dq = q - q_prev[d]
            # External work needed to sustain the deformation is f_on_i . d(u_j-u_i).
            w_e += n_cells * float(0.5 * (fe_prev[d] + fe) @ dq)
            w_o += n_cells * float(0.5 * (fo_prev[d] + fo) @ dq)
            q_prev[d], fe_prev[d], fo_prev[d] = q, fe, fo
    return (w_e + w_o) / area, w_e / area, w_o / area


def fit_through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope = float(x @ y / (x @ x))
    residual = y - slope * x
    ss_res = float(residual @ residual)
    ss_tot = float(y @ y)
    r2_origin = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, r2_origin


class PassiveCrackedStrip:
    """Triangular central-force strip, periodic in x and finite in y."""

    def __init__(
        self,
        nx: int,
        ny: int,
        crack_half_length: float,
        k: float = 1.0,
    ) -> None:
        if nx < 8 or ny < 8:
            raise ValueError("nx and ny must both be at least 8")
        if crack_half_length <= 0 or crack_half_length >= nx / 2.0 - 2.0:
            raise ValueError("crack_half_length is incompatible with the periodic width")
        self.nx = int(nx)
        self.ny = int(ny)
        self.k = float(k)
        self.period = float(nx)
        self.n_nodes = nx * ny
        self.ndof = 2 * self.n_nodes
        self.j_lower = ny // 2 - 1
        self.target_half_length = float(crack_half_length)
        self.positions = self._positions()
        self.all_bonds = self._all_bonds()
        self.removed_ids, self.active_bonds, self.candidate_ids, self.a_eff = self._make_crack()
        self.K = self._assemble_stiffness()

    def node_id(self, i: int, j: int) -> int:
        return j * self.nx + (i % self.nx)

    def _positions(self) -> np.ndarray:
        pos = np.zeros((self.n_nodes, 2), dtype=float)
        for j in range(self.ny):
            for i in range(self.nx):
                pos[self.node_id(i, j)] = i * A1 + j * A2
        return pos

    def _all_bonds(self) -> list[Bond]:
        bonds: list[Bond] = []
        for j in range(self.ny):
            for i in range(self.nx):
                p = self.node_id(i, j)
                # Horizontal periodic bond.
                bonds.append(Bond(p, self.node_id(i + 1, j), A1.copy()))
                if j < self.ny - 1:
                    # Up-right bond: reference vector A2.
                    q = self.node_id(i, j + 1)
                    crosses = j == self.j_lower
                    midx = (i + 0.25 + 0.5 * j) % self.period if crosses else None
                    bonds.append(Bond(p, q, A2.copy(), midx, crosses))
                    # Up-left bond: reference vector A3 = A2-A1.
                    q = self.node_id(i - 1, j + 1)
                    midx = (i - 0.25 + 0.5 * j) % self.period if crosses else None
                    bonds.append(Bond(p, q, A3.copy(), midx, crosses))
        return bonds

    def _make_crack(self) -> tuple[set[int], list[Bond], list[int], float]:
        center = 0.5 * self.period
        crossing: list[tuple[int, float]] = []
        removed: set[int] = set()
        for bid, bond in enumerate(self.all_bonds):
            if not bond.crosses_crack_plane:
                continue
            assert bond.midpoint_x is not None
            d = wrap_centered(bond.midpoint_x - center, self.period)
            crossing.append((bid, d))
            if abs(d) <= self.target_half_length:
                removed.add(bid)
        if not removed:
            raise RuntimeError("No bonds were removed; increase crack_half_length")

        removed_d = [d for bid, d in crossing if bid in removed]
        intact = [(bid, d) for bid, d in crossing if bid not in removed]
        right_intact = [(bid, d) for bid, d in intact if d > 0]
        left_intact = [(bid, d) for bid, d in intact if d < 0]
        if not right_intact or not left_intact:
            raise RuntimeError("Crack interacts with its periodic image")

        right_bid, right_d = min(right_intact, key=lambda item: item[1])
        left_bid, left_d = max(left_intact, key=lambda item: item[1])
        right_removed = max(d for d in removed_d if d > 0)
        left_removed = min(d for d in removed_d if d < 0)
        right_tip = 0.5 * (right_removed + right_d)
        left_tip = 0.5 * (left_removed + left_d)
        a_eff = 0.5 * (right_tip - left_tip)

        # Include all intact crossing bonds at the same nearest-tip distance.
        tol = 1e-12
        candidates = [
            bid for bid, d in intact
            if abs(d - right_d) < tol or abs(d - left_d) < tol
        ]
        active = [bond for bid, bond in enumerate(self.all_bonds) if bid not in removed]
        return removed, active, candidates, float(a_eff)

    def _assemble_stiffness(self) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for bond in self.active_bonds:
            kb = self.k * np.outer(bond.n, bond.n)
            for a in range(2):
                for b in range(2):
                    ii = 2 * bond.i + a
                    ij = 2 * bond.i + b
                    ji = 2 * bond.j + a
                    jj = 2 * bond.j + b
                    val = float(kb[a, b])
                    rows.extend((ii, ji, ii, ji))
                    cols.extend((ij, jj, jj, ij))
                    data.extend((val, val, -val, -val))
        K = sparse.coo_matrix((data, (rows, cols)), shape=(self.ndof, self.ndof))
        return K.tocsr()

    def constrained_dofs(self, delta: float) -> tuple[np.ndarray, np.ndarray]:
        dofs: list[int] = []
        vals: list[float] = []
        for i in range(self.nx):
            dofs.append(2 * self.node_id(i, 0) + 1)
            vals.append(-0.5 * delta)
            dofs.append(2 * self.node_id(i, self.ny - 1) + 1)
            vals.append(0.5 * delta)
        # Remove horizontal rigid translation.
        dofs.append(2 * self.node_id(0, 0))
        vals.append(0.0)
        order = np.argsort(dofs)
        return np.asarray(dofs, dtype=int)[order], np.asarray(vals, dtype=float)[order]

    def solve(self, delta: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
        c, uc = self.constrained_dofs(delta)
        all_dofs = np.arange(self.ndof)
        free_mask = np.ones(self.ndof, dtype=bool)
        free_mask[c] = False
        f = all_dofs[free_mask]
        Kff = self.K[f][:, f]
        Kfc = self.K[f][:, c]
        rhs = -(Kfc @ uc)
        uf = spsolve(Kff, rhs)
        if not np.all(np.isfinite(uf)):
            raise RuntimeError("Linear solve failed or stiffness matrix is singular")
        u = np.zeros(self.ndof, dtype=float)
        u[c] = uc
        u[f] = uf
        residual = self.K @ u
        free_residual = float(np.max(np.abs(residual[f])))
        return u.reshape((-1, 2)), residual, free_residual

    def bond_extension(self, bond: Bond, u: np.ndarray) -> float:
        return float((u[bond.j] - u[bond.i]) @ bond.n)

    def solve_initiation(self, delta_c: float) -> CrackSolveResult:
        u, residual, free_residual = self.solve(delta=1.0)
        candidate_ext = np.array(
            [self.bond_extension(self.all_bonds[bid], u) for bid in self.candidate_ids],
            dtype=float,
        )
        max_ext = float(np.max(candidate_ext))
        if max_ext <= 0:
            raise RuntimeError("No crack-tip candidate bond is in tension")

        top_y = np.array(
            [2 * self.node_id(i, self.ny - 1) + 1 for i in range(self.nx)], dtype=int
        )
        reaction = float(np.sum(residual[top_y]))
        remote_stress = reaction / self.period
        energy = float(0.5 * np.ravel(u) @ (self.K @ np.ravel(u)))
        clapeyron = abs(energy - 0.5 * reaction) / max(abs(energy), 1e-30)
        init_delta = float(delta_c / max_ext)
        return CrackSolveResult(
            target_half_length=self.target_half_length,
            effective_half_length=self.a_eff,
            crack_tip_candidate_ids=list(self.candidate_ids),
            candidate_extensions_unit=candidate_ext,
            max_tensile_extension_unit=max_ext,
            reaction_unit=reaction,
            remote_stress_unit=remote_stress,
            initiation_displacement=init_delta,
            initiation_force=init_delta * reaction,
            initiation_remote_stress=init_delta * remote_stress,
            energy_unit=energy,
            clapeyron_relative_error=float(clapeyron),
            free_residual_inf=free_residual,
            removed_bonds=len(self.removed_ids),
        )


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_cycle_baseline(out: Path) -> dict[str, float]:
    nx, ny, k = 8, 6, 1.0
    amplitude = 0.03
    ko_values = np.array([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30])
    ko_rows: list[dict[str, object]] = []
    w_values = []
    for ko in ko_values:
        wt, we, wo = periodic_cycle_work_density(nx, ny, k, float(ko), amplitude)
        K_odd = -SQRT3 * ko / 4.0
        exact = 2.0 * math.pi * K_odd * amplitude**2
        ko_rows.append({
            "k_o": ko,
            "amplitude": amplitude,
            "work_total_density": wt,
            "work_even_density": we,
            "work_odd_density": wo,
            "analytic_density": exact,
            "absolute_error": abs(wt - exact),
        })
        w_values.append(wt)
    write_csv(
        out / "cycle_scan_ko.csv",
        ko_rows,
        ["k_o", "amplitude", "work_total_density", "work_even_density", "work_odd_density", "analytic_density", "absolute_error"],
    )
    slope_ko, r2_ko = fit_through_origin(ko_values, np.asarray(w_values))
    exact_slope_ko = -math.pi * SQRT3 * amplitude**2 / 2.0

    amplitudes = np.array([0.008, 0.012, 0.018, 0.026, 0.038, 0.052])
    ko = 0.2
    a_rows: list[dict[str, object]] = []
    wa = []
    for amp in amplitudes:
        wt, we, wo = periodic_cycle_work_density(nx, ny, k, ko, float(amp))
        K_odd = -SQRT3 * ko / 4.0
        exact = 2.0 * math.pi * K_odd * amp**2
        a_rows.append({
            "k_o": ko,
            "amplitude": amp,
            "amplitude_squared": amp**2,
            "work_total_density": wt,
            "work_even_density": we,
            "work_odd_density": wo,
            "analytic_density": exact,
            "absolute_error": abs(wt - exact),
        })
        wa.append(wt)
    write_csv(
        out / "cycle_scan_amplitude.csv",
        a_rows,
        ["k_o", "amplitude", "amplitude_squared", "work_total_density", "work_even_density", "work_odd_density", "analytic_density", "absolute_error"],
    )
    slope_a2, r2_a2 = fit_through_origin(amplitudes**2, np.asarray(wa))
    exact_slope_a2 = 2.0 * math.pi * (-SQRT3 * ko / 4.0)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(ko_values, w_values, "o", label="finite bond sum")
    xx = np.linspace(ko_values.min(), ko_values.max(), 200)
    ax.plot(xx, exact_slope_ko * xx, "--", label="analytic")
    ax.set_xlabel(r"odd bond coefficient $k_o$")
    ax.set_ylabel(r"cycle work density $W_{\rm loop}/A$")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "cycle_work_vs_ko.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(amplitudes**2, wa, "o", label="finite bond sum")
    xx = np.linspace(0.0, float(np.max(amplitudes**2)), 200)
    ax.plot(xx, exact_slope_a2 * xx, "--", label="analytic")
    ax.set_xlabel(r"squared shear amplitude $\mathcal{E}^2$")
    ax.set_ylabel(r"cycle work density $W_{\rm loop}/A$")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "cycle_work_vs_amplitude_squared.pdf")
    plt.close(fig)

    return {
        "cycle_ko_fit_slope": slope_ko,
        "cycle_ko_exact_slope": exact_slope_ko,
        "cycle_ko_fit_relative_error": abs(slope_ko - exact_slope_ko) / abs(exact_slope_ko),
        "cycle_ko_fit_r2_origin": r2_ko,
        "cycle_a2_fit_slope": slope_a2,
        "cycle_a2_exact_slope": exact_slope_a2,
        "cycle_a2_fit_relative_error": abs(slope_a2 - exact_slope_a2) / abs(exact_slope_a2),
        "cycle_a2_fit_r2_origin": r2_a2,
        "cycle_max_even_work_abs": max(abs(float(row["work_even_density"])) for row in ko_rows + a_rows),
    }


def run_passive_crack_baseline(out: Path) -> dict[str, float]:
    nx, ny = 64, 48
    delta_c = 0.02
    targets = [2.5, 3.5, 4.5, 5.5, 6.5, 8.0, 10.0, 12.0, 14.0, 16.0]
    results: list[CrackSolveResult] = []
    for a in targets:
        strip = PassiveCrackedStrip(nx=nx, ny=ny, crack_half_length=a, k=1.0)
        results.append(strip.solve_initiation(delta_c=delta_c))

    rows: list[dict[str, object]] = []
    for r in results:
        rows.append({
            "nx": nx,
            "ny": ny,
            "delta_c": delta_c,
            "target_half_length": r.target_half_length,
            "effective_half_length": r.effective_half_length,
            "removed_bonds": r.removed_bonds,
            "candidate_extensions_unit": ";".join(f"{x:.12g}" for x in r.candidate_extensions_unit),
            "max_tensile_extension_unit": r.max_tensile_extension_unit,
            "reaction_unit": r.reaction_unit,
            "remote_stress_unit": r.remote_stress_unit,
            "initiation_displacement": r.initiation_displacement,
            "initiation_force": r.initiation_force,
            "initiation_remote_stress": r.initiation_remote_stress,
            "sigma_times_sqrt_a": r.initiation_remote_stress * math.sqrt(r.effective_half_length),
            "energy_unit": r.energy_unit,
            "clapeyron_relative_error": r.clapeyron_relative_error,
            "free_residual_inf": r.free_residual_inf,
        })
    write_csv(
        out / "passive_crack_scan.csv",
        rows,
        list(rows[0].keys()),
    )

    a_eff = np.array([r.effective_half_length for r in results])
    sigma_g = np.array([r.initiation_remote_stress for r in results])
    inv_sqrt_a = 1.0 / np.sqrt(a_eff)
    # Fit sigma_G = c0 + c1/sqrt(a) to expose finite-size/lattice offset.
    X = np.column_stack([np.ones_like(inv_sqrt_a), inv_sqrt_a])
    coeff, *_ = np.linalg.lstsq(X, sigma_g, rcond=None)
    pred = X @ coeff
    ss_res = float(np.sum((sigma_g - pred) ** 2))
    ss_tot = float(np.sum((sigma_g - np.mean(sigma_g)) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    # Also fit the larger-crack subset through the origin, closer to Griffith scaling.
    subset = a_eff >= 5.0
    slope_origin, r2_origin = fit_through_origin(inv_sqrt_a[subset], sigma_g[subset])
    k_like = sigma_g * np.sqrt(math.pi * a_eff)

    def loglog_exponent(mask: np.ndarray) -> dict[str, float]:
        """Unconstrained log-log slope with a Student-t 95% interval."""
        x = np.log(a_eff[mask])
        y = np.log(sigma_g[mask])
        design = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        dof = len(x) - 2
        variance = float(residual @ residual) / dof
        covariance = variance * np.linalg.inv(design.T @ design)
        standard_error = math.sqrt(float(covariance[1, 1]))
        t_critical = float(stats.t.ppf(0.975, dof))
        return {
            "n_points": int(mask.sum()),
            "a_min": float(a_eff[mask].min()),
            "a_max": float(a_eff[mask].max()),
            "exponent": float(beta[1]),
            "ci_low": float(beta[1] - t_critical * standard_error),
            "ci_high": float(beta[1] + t_critical * standard_error),
            "k_like_relative_spread": float(
                (k_like[mask].max() - k_like[mask].min()) / k_like[mask].mean()
            ),
        }

    narrow_mask = (a_eff >= 6.4) & (a_eff <= 12.1)
    wide_mask = np.ones_like(a_eff, dtype=bool)
    narrow_fit = loglog_exponent(narrow_mask)
    wide_fit = loglog_exponent(wide_mask)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(inv_sqrt_a, sigma_g, "o", label="lattice initiation")
    xx = np.linspace(float(inv_sqrt_a.min()) * 0.95, float(inv_sqrt_a.max()) * 1.05, 200)
    ax.plot(xx, coeff[0] + coeff[1] * xx, "--", label=r"fit $c_0+c_1/\sqrt{a}$")
    ax.set_xlabel(r"$1/\sqrt{a_{\rm eff}}$")
    ax.set_ylabel(r"remote initiation stress $\sigma_G^{\rm lat}$")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "passive_initiation_griffith_scaling.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(a_eff, k_like, "o-")
    ax.set_xlabel(r"effective crack half-length $a_{\rm eff}$")
    ax.set_ylabel(r"$\sigma_G^{\rm lat}\sqrt{\pi a_{\rm eff}}$")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "passive_effective_toughness_vs_crack_length.pdf")
    plt.close(fig)

    # Visualize one representative relaxed crack field at its initiation displacement.
    representative = PassiveCrackedStrip(nx=nx, ny=ny, crack_half_length=8.0, k=1.0)
    rep_result = representative.solve_initiation(delta_c=delta_c)
    u_unit, _, _ = representative.solve(delta=1.0)
    u = rep_result.initiation_displacement * u_unit
    scale = 8.0
    deformed = representative.positions + scale * u
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    removed = representative.removed_ids
    for bid, bond in enumerate(representative.all_bonds):
        if bid in removed:
            continue
        p = deformed[bond.i]
        q = deformed[bond.j].copy()
        # Unwrap horizontal periodic bonds for plotting.
        dx = q[0] - p[0]
        if dx > representative.period / 2:
            q[0] -= representative.period
        elif dx < -representative.period / 2:
            q[0] += representative.period
        if p[0] < -1 or p[0] > representative.period + 1 or q[0] < -1 or q[0] > representative.period + 1:
            continue
        ax.plot([p[0], q[0]], [p[1], q[1]], linewidth=0.35, alpha=0.55)
    for bid in representative.candidate_ids:
        bond = representative.all_bonds[bid]
        p = deformed[bond.i]
        q = deformed[bond.j].copy()
        dx = q[0] - p[0]
        if dx > representative.period / 2:
            q[0] -= representative.period
        elif dx < -representative.period / 2:
            q[0] += representative.period
        ax.plot([p[0], q[0]], [p[1], q[1]], linewidth=2.0)
    ax.set_aspect("equal")
    ax.set_xlim(0, representative.period)
    ax.set_xlabel("x")
    ax.set_ylabel("y (displacements magnified 8x)")
    ax.set_title("Passive pre-cracked triangular strip at initiation")
    fig.tight_layout()
    fig.savefig(out / "passive_crack_field.pdf")
    plt.close(fig)

    return {
        "passive_grid_nx": nx,
        "passive_grid_ny": ny,
        "passive_delta_c": delta_c,
        "passive_fit_intercept": float(coeff[0]),
        "passive_fit_inverse_sqrt_slope": float(coeff[1]),
        "passive_fit_r2": r2,
        "passive_large_crack_origin_slope": slope_origin,
        "passive_large_crack_origin_r2": r2_origin,
        "passive_effective_toughness_mean": float(np.mean(k_like[subset])),
        "passive_effective_toughness_cv": float(np.std(k_like[subset]) / np.mean(k_like[subset])),
        "passive_initiation_exponent_narrow": narrow_fit,
        "passive_initiation_exponent_wide": wide_fit,
        "passive_max_clapeyron_relative_error": max(r.clapeyron_relative_error for r in results),
        "passive_max_free_residual_inf": max(r.free_residual_inf for r in results),
    }



def run_passive_size_convergence(out: Path) -> dict[str, float]:
    delta_c = 0.02
    a = 8.0
    sizes = [(48, 36), (56, 42), (64, 48), (72, 54), (80, 60)]
    rows: list[dict[str, object]] = []
    for nx, ny in sizes:
        result = PassiveCrackedStrip(nx=nx, ny=ny, crack_half_length=a, k=1.0).solve_initiation(delta_c)
        rows.append({
            "nx": nx,
            "ny": ny,
            "effective_half_length": result.effective_half_length,
            "initiation_remote_stress": result.initiation_remote_stress,
            "effective_toughness": result.initiation_remote_stress * math.sqrt(math.pi * result.effective_half_length),
            "free_residual_inf": result.free_residual_inf,
        })
    write_csv(
        out / "passive_size_convergence.csv",
        rows,
        list(rows[0].keys()),
    )
    inv_width = np.array([1.0 / float(row["nx"]) for row in rows])
    toughness = np.array([float(row["effective_toughness"]) for row in rows])
    X = np.column_stack([np.ones_like(inv_width), inv_width])
    coeff, *_ = np.linalg.lstsq(X, toughness, rcond=None)
    extrapolated = float(coeff[0])
    spread = float((np.max(toughness) - np.min(toughness)) / np.mean(toughness))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot([int(row["nx"]) for row in rows], toughness, "o-")
    ax.set_xlabel("periodic width nx")
    ax.set_ylabel(r"effective toughness $\sigma_G^{\rm lat}\sqrt{\pi a}$")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "passive_size_convergence.pdf")
    plt.close(fig)

    return {
        "passive_size_toughness_spread": spread,
        "passive_size_extrapolated_toughness": extrapolated,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("baseline_results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {}
    summary.update(run_cycle_baseline(args.out))
    summary.update(run_passive_crack_baseline(args.out))
    summary.update(run_passive_size_convergence(args.out))
    with (args.out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
