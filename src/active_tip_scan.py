#!/usr/bin/env python3
""" static active scan for the pre-cracked odd-elastic lattice.

The physical fracture rule remains a fixed tensile extension threshold of the
conservative longitudinal spring. This script only asks how the *equilibrated*
crack-tip bond extensions change when the odd coefficient k_o is varied.

For a centered crack there are two symmetry-related tips. Odd-elastic chirality
is expected to bias them oppositely; reversing k_o should exchange left and right.
This is a static precursor to the later protocol-work and contour-source tests,
not yet a proof of sub-Griffith dynamic propagation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.optimize import brentq, minimize_scalar

from lattice_baselines import (
    PassiveCrackedStrip,
    R90,
    wrap_centered,
)


class ActiveCrackedStrip(PassiveCrackedStrip):
    def __init__(self, nx: int, ny: int, crack_half_length: float, k: float, k_o: float) -> None:
        self.k_o = float(k_o)
        super().__init__(nx=nx, ny=ny, crack_half_length=crack_half_length, k=k)

    def _assemble_stiffness(self) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for bond in self.active_bonds:
            t = R90 @ bond.n
            # K u = -F_internal. For f_i=(k n-k_o t)n^T(u_j-u_i),
            # the bond block in K is M=(k n-k_o t) tensor n.
            mb = np.outer(self.k * bond.n - self.k_o * t, bond.n)
            for a in range(2):
                for b in range(2):
                    ii = 2 * bond.i + a
                    ij = 2 * bond.i + b
                    ji = 2 * bond.j + a
                    jj = 2 * bond.j + b
                    val = float(mb[a, b])
                    rows.extend((ii, ji, ii, ji))
                    cols.extend((ij, jj, jj, ij))
                    data.extend((val, val, -val, -val))
        return sparse.coo_matrix((data, (rows, cols)), shape=(self.ndof, self.ndof)).tocsr()

    def tip_candidates(self) -> dict[str, int]:
        center = 0.5 * self.period
        out: dict[str, int] = {}
        for bid in self.candidate_ids:
            bond = self.all_bonds[bid]
            assert bond.midpoint_x is not None
            d = wrap_centered(bond.midpoint_x - center, self.period)
            out["right" if d > 0 else "left"] = bid
        if set(out) != {"left", "right"}:
            raise RuntimeError(f"Expected one candidate at each tip, got {out}")
        return out

    def static_diagnostics(self, delta_c: float) -> dict[str, float]:
        u, residual, free_residual = self.solve(delta=1.0)
        tips = self.tip_candidates()
        ext_left = self.bond_extension(self.all_bonds[tips["left"]], u)
        ext_right = self.bond_extension(self.all_bonds[tips["right"]], u)
        if min(ext_left, ext_right) <= 0:
            raise RuntimeError("A tip candidate is not in tension")
        top_y = np.array([2 * self.node_id(i, self.ny - 1) + 1 for i in range(self.nx)])
        reaction = float(np.sum(residual[top_y]))
        sigma_unit = reaction / self.period
        return {
            "k_o": self.k_o,
            "extension_left_unit": ext_left,
            "extension_right_unit": ext_right,
            "extension_mean_unit": 0.5 * (ext_left + ext_right),
            "extension_bias_unit": 0.5 * (ext_right - ext_left),
            "initiation_delta_left": delta_c / ext_left,
            "initiation_delta_right": delta_c / ext_right,
            "initiation_sigma_left": sigma_unit * delta_c / ext_left,
            "initiation_sigma_right": sigma_unit * delta_c / ext_right,
            "remote_stress_unit": sigma_unit,
            "reaction_unit": reaction,
            "free_residual_inf": free_residual,
        }

    def min_dynamic_stability_real_part(self) -> float:
        c, _ = self.constrained_dofs(delta=0.0)
        mask = np.ones(self.ndof, dtype=bool)
        mask[c] = False
        Kff = self.K[mask][:, mask]
        # Eigenvalues of overdamped decay operator K. Positive real part means
        # decay under u_dot=-K u/eta. Shift-invert near zero captures the soft modes.
        vals = eigs(Kff, k=4, sigma=0.0, which="LM", return_eigenvectors=False)
        return float(np.min(np.real(vals)))


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("active_tip_results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    nx, ny, a, delta_c = 64, 48, 8.0, 0.02
    ko_values = np.linspace(-0.45, 0.45, 19)
    rows: list[dict[str, float]] = []
    for ko in ko_values:
        model = ActiveCrackedStrip(nx=nx, ny=ny, crack_half_length=a, k=1.0, k_o=float(ko))
        row = model.static_diagnostics(delta_c=delta_c)
        rows.append(row)
    write_rows(args.out / "active_tip_scan.csv", rows)

    ko = np.array([r["k_o"] for r in rows])
    ext_l = np.array([r["extension_left_unit"] for r in rows])
    ext_r = np.array([r["extension_right_unit"] for r in rows])
    sig_l = np.array([r["initiation_sigma_left"] for r in rows])
    sig_r = np.array([r["initiation_sigma_right"] for r in rows])
    sigma_unit = np.array([r["remote_stress_unit"] for r in rows])

    # Symmetry diagnostics: x_left(k_o)=x_right(-k_o).
    swap_error_ext = float(np.max(np.abs(ext_l - ext_r[::-1])))
    swap_error_sigma = float(np.max(np.abs(sig_l - sig_r[::-1])))
    reaction_even_error = float(np.max(np.abs(sigma_unit - sigma_unit[::-1])))

    small = np.abs(ko) <= 0.2 + 1e-12
    X = np.column_stack([np.ones(np.count_nonzero(small)), ko[small]])
    coeff_bias, *_ = np.linalg.lstsq(X, 0.5 * (ext_r[small] - ext_l[small]), rcond=None)
    pred = X @ coeff_bias
    y = 0.5 * (ext_r[small] - ext_l[small])
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_bias = 1.0 - ss_res / ss_tot

    passive_idx = int(np.argmin(np.abs(ko)))
    passive_sigma = 0.5 * (sig_l[passive_idx] + sig_r[passive_idx])
    min_tip_sigma = np.minimum(sig_l, sig_r)
    reduction = 1.0 - min_tip_sigma / passive_sigma

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(ko, ext_l, "o-", label="left tip")
    ax.plot(ko, ext_r, "s-", label="right tip")
    ax.set_xlabel(r"odd bond coefficient $k_o$")
    ax.set_ylabel("tip-bond extension at unit opening")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "active_tip_extension_vs_ko.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(ko, sig_l, "o-", label="left tip")
    ax.plot(ko, sig_r, "s-", label="right tip")
    ax.axhline(passive_sigma, linestyle="--", label="passive baseline")
    ax.set_xlabel(r"odd bond coefficient $k_o$")
    ax.set_ylabel(r"tip-specific initiation stress $\sigma_G^{\rm lat}$")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "active_tip_initiation_vs_ko.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(np.abs(ko), reduction, "o")
    ax.set_xlabel(r"$|k_o|$")
    ax.set_ylabel("reduction of first-tip initiation stress")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "active_first_tip_threshold_reduction.pdf")
    plt.close(fig)

    # Stability is more expensive; sample a smaller system at representative values.
    stability_rows = []
    for ko_sample in (-0.45, -0.30, 0.0, 0.30, 0.45):
        model = ActiveCrackedStrip(nx=32, ny=24, crack_half_length=5.0, k=1.0, k_o=ko_sample)
        stability_rows.append({
            "k_o": ko_sample,
            "min_real_eigenvalue": model.min_dynamic_stability_real_part(),
        })
    write_rows(args.out / "active_stability_sample.csv", stability_rows)

    # Convert the static tip criterion into a preliminary sub-Griffith boundary:
    # at a prescribed fraction p of the passive initiation stress, find the |k_o|
    # at which the favored tip first reaches the same bond threshold.
    cache: dict[float, float] = {}
    def first_tip_sigma(ko_value: float) -> float:
        key = round(float(ko_value), 12)
        if key not in cache:
            diag = ActiveCrackedStrip(nx=nx, ny=ny, crack_half_length=a, k=1.0, k_o=float(ko_value)).static_diagnostics(delta_c)
            cache[key] = min(diag["initiation_sigma_left"], diag["initiation_sigma_right"])
        return cache[key]

    threshold_rows: list[dict[str, float]] = []
    for load_fraction in (0.98, 0.95, 0.90, 0.85, 0.83):
        target_sigma = load_fraction * passive_sigma
        critical_ko = brentq(lambda x: first_tip_sigma(x) - target_sigma, 0.0, 0.45, xtol=1e-8)
        threshold_rows.append({
            "load_fraction_of_passive": load_fraction,
            "target_remote_stress": target_sigma,
            "critical_abs_k_o": critical_ko,
        })
    write_rows(args.out / "static_subgriffith_thresholds.csv", threshold_rows)
    optimum = minimize_scalar(first_tip_sigma, bounds=(0.0, 0.8), method="bounded", options={"xatol": 1e-6})

    summary = {
        "grid_nx": nx,
        "grid_ny": ny,
        "crack_half_length": a,
        "delta_c": delta_c,
        "tip_extension_swap_max_abs_error": swap_error_ext,
        "tip_initiation_swap_max_abs_error": swap_error_sigma,
        "remote_stress_even_max_abs_error": reaction_even_error,
        "small_ko_bias_intercept": float(coeff_bias[0]),
        "small_ko_bias_slope": float(coeff_bias[1]),
        "small_ko_bias_fit_r2": r2_bias,
        "passive_initiation_stress": float(passive_sigma),
        "max_first_tip_threshold_reduction": float(np.max(reduction)),
        "minimum_sampled_stability_real_part": float(min(r["min_real_eigenvalue"] for r in stability_rows)),
        "static_optimal_abs_k_o": float(optimum.x),
        "static_minimum_first_tip_stress": float(optimum.fun),
        "static_minimum_first_tip_stress_ratio": float(optimum.fun / passive_sigma),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
