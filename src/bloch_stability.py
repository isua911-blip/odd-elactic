#!/usr/bin/env python3
"""WP4: Bloch flutter boundary of the intact infinite odd-elastic lattice.

The manuscript's critical inertial damping is measured from the dense spectrum
of a 32x24 strip, and it explicitly disclaims being a Bloch boundary.  This
module computes the infinite-lattice result, which settles whether the flutter
is a bulk lattice property or a finite-size/boundary artefact.

Dynamical matrix.  With  f_i = (k n - k_o t)(u_j - u_i).n  and Bloch waves
u_p = U exp(i q.R_p), the six neighbours pair up into

    D(q) = sum_b 2[1 - cos(q.a_b)] (k n_b n_b^T - k_o t_b n_b^T),   b = 1,2,3

which is real but non-symmetric, so its 2x2 spectrum may be a complex pair.

Stability.  For  m s^2 + c s + kappa = 0  a root crosses the imaginary axis when
omega^2 = kappa_R/m and c omega = -kappa_I, hence (with m = 1)

    c_crit(q) = |kappa_I| / sqrt(kappa_R),      c_crit = max over q and modes.

Free check: t_b is perpendicular to n_b, so tr(sum c_b t_b n_b^T) = 0 and
tr D is independent of k_o.  Any k_o dependence in the trace is a bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import math
import sys
from pathlib import Path

import numpy as np


from lattice_baselines import A1, A2, A3, R90  # noqa: E402

BONDS = (A1, A2, A3)
TANGENTS = tuple(R90 @ n for n in BONDS)

# reciprocal basis, b_i . a_j = 2 pi delta_ij
B1 = 2.0 * math.pi * np.array([1.0, -1.0 / math.sqrt(3.0)])
B2 = 2.0 * math.pi * np.array([0.0, 2.0 / math.sqrt(3.0)])


def dynamical_matrix(q: np.ndarray, k: float, k_o: float) -> np.ndarray:
    D = np.zeros((2, 2))
    for n, t in zip(BONDS, TANGENTS):
        c = 2.0 * (1.0 - math.cos(float(q @ n)))
        D += c * (k * np.outer(n, n) - k_o * np.outer(t, n))
    return D


def c_crit_at(q: np.ndarray, k: float, k_o: float) -> tuple[float, float]:
    """Return (c_crit, min Re kappa) at one wavevector."""
    vals = np.linalg.eigvals(dynamical_matrix(q, k, k_o))
    worst = 0.0
    for kappa in vals:
        re, im = float(np.real(kappa)), float(np.imag(kappa))
        if abs(im) > 1e-14 and re > 1e-14:
            worst = max(worst, abs(im) / math.sqrt(re))
    return worst, float(np.min(np.real(vals)))


def sweep(k: float, k_o: float, n_grid: int = 241) -> dict[str, object]:
    fractions = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    best, best_q, min_re = 0.0, np.zeros(2), math.inf
    for x1 in fractions:
        for x2 in fractions:
            q = x1 * B1 + x2 * B2
            value, re = c_crit_at(q, k, k_o)
            min_re = min(min_re, re)
            if value > best:
                best, best_q = value, q
    # distance of the maximising wavevector from the zone centre, in units of
    # the zone-boundary scale 2*pi
    return {
        "k_o": k_o,
        "c_crit": best,
        "q": best_q,
        "q_norm_over_2pi": float(np.linalg.norm(best_q) / (2.0 * math.pi)),
        "min_Re_kappa": min_re,
    }


if __name__ == "__main__":
    print("free check: tr D must not depend on k_o")
    q = 0.3 * B1 + 0.17 * B2
    traces = [np.trace(dynamical_matrix(q, 1.0, ko)) for ko in (0.0, 0.2, 0.5)]
    print(f"   tr D at k_o = 0, 0.2, 0.5 : {traces[0]:.12f} {traces[1]:.12f} {traces[2]:.12f}")
    print(f"   max deviation = {max(abs(t - traces[0]) for t in traces):.3e}\n")

    print("small-q scaling: c_crit should vanish linearly in |q| (Re, Im both ~ q^2)")
    for scale in (0.20, 0.10, 0.05, 0.025):
        qs = scale * (B1 + 0.37 * B2)
        value, _ = c_crit_at(qs, 1.0, 0.20)
        print(f"   |q|/2pi = {np.linalg.norm(qs)/(2*math.pi):.4f}   c_crit(q) = {value:.5f}")

    print("\nBrillouin-zone maximum (241 x 241 grid), k = 1")
    print(f"{'k_o':>7}{'c_crit':>11}{'c_crit/k_o':>12}{'|q|/2pi':>10}{'min Re kappa':>14}   manuscript strip")
    strip = {0.05: 0.067, 0.12: 0.233, 0.20: 0.411, 0.30: 0.626, 0.50: 1.053}
    for k_o in (0.05, 0.12, 0.20, 0.30, 0.50):
        r = sweep(1.0, k_o)
        print(
            f"{k_o:>7.2f}{r['c_crit']:>11.4f}{r['c_crit']/k_o:>12.4f}"
            f"{r['q_norm_over_2pi']:>10.4f}{r['min_Re_kappa']:>14.3e}"
            f"        {strip[k_o]:.3f}",
            flush=True,
        )
