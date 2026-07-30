#!/usr/bin/env python3
"""WP6: independent continuum solver for the odd crack-tip asymptotics.

``crack_tip_asymptotics`` propagates the angular state with a closed-form matrix
polynomial that already assumes the two Williams harmonic sectors, and it
obtains the spectrum analytically.  Both the manuscript's half-integer spectrum
claim and its mode-mixing claim therefore rest on one derivation.

This module re-solves the same boundary value problem with machinery that shares
none of that: a general-purpose adaptive ODE integrator for

    dy/dtheta = A(lambda) y ,      y = (u_1, u_2, t_1, t_2) ,

plus a shooting condition for traction-free crack faces,

    det [ P(2 pi; lambda) ]_{traction rows, displacement cols} = 0 ,

whose roots are the admissible exponents.  Because the continuum takes A_o and
K_o as independent inputs, it also reaches the (A_o, K_o) plane off the lattice
ray A_o = 2 K_o, where the discrete model cannot go.
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
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


from crack_tip_asymptotics import (  # noqa: E402
    J_matrix,
    OddModuli,
    closed_form_propagator,
    state_matrix,
)

SQRT3 = math.sqrt(3.0)
# lattice ray reference: B = sqrt3/2, mu = sqrt3/4 for k = 1
B_LAT, MU_LAT = SQRT3 / 2.0, SQRT3 / 4.0


def ode_propagator(delta_theta: float, lam: float, moduli: OddModuli, rtol=1e-12, atol=1e-14):
    """Propagate the 4x4 angular system with an adaptive integrator."""
    A = state_matrix(lam, moduli)

    def rhs(_theta, y):
        return A @ y.reshape(4, 4)  # columns integrated simultaneously

    sol = solve_ivp(
        lambda th, y: (A @ y.reshape(4, 4)).ravel(),
        (0.0, delta_theta),
        np.eye(4).ravel(),
        rtol=rtol,
        atol=atol,
        method="DOP853",
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"ODE propagation failed: {sol.message}")
    return sol.y[:, -1].reshape(4, 4)


def shooting_residual(lam: float, moduli: OddModuli) -> float:
    """det of the map (face displacement) -> (opposite face traction).

    Vanishes exactly at an admissible Williams exponent.
    """
    P = ode_propagator(2.0 * math.pi, lam, moduli)
    return float(np.linalg.det(P[2:, :2]))


def find_exponents(moduli: OddModuli, lam_max: float = 3.2, n_scan: int = 1300):
    """Bracket and refine every sign change of the shooting residual."""
    grid = np.linspace(0.02, lam_max, n_scan)
    values = np.array([shooting_residual(float(x), moduli) for x in grid])
    roots = []
    for i in range(len(grid) - 1):
        a, b = values[i], values[i + 1]
        if a == 0.0:
            roots.append(float(grid[i]))
        elif a * b < 0:
            roots.append(float(brentq(shooting_residual, grid[i], grid[i + 1], args=(moduli,), xtol=1e-13)))
    return roots


if __name__ == "__main__":
    print("A) cross-check: adaptive ODE propagator vs the closed-form matrix polynomial")
    for lam in (0.5, 1.5, 2.5):
        for A_o, K_o in ((0.0, 0.0), (-0.30, -0.15), (0.25, -0.40)):
            m = OddModuli(B=B_LAT, mu=MU_LAT, A_o=A_o, K_o=K_o)
            P_ode = ode_propagator(2.0 * math.pi, lam, m)
            P_cf = closed_form_propagator(2.0 * math.pi, lam, m)
            err = float(np.max(np.abs(P_ode - P_cf)) / max(np.max(np.abs(P_cf)), 1e-30))
            print(f"   lambda={lam}  (A_o,K_o)=({A_o:+.2f},{K_o:+.2f})   max rel diff = {err:.2e}")

    print("\nB) independent spectrum by shooting, off the lattice ray A_o = 2 K_o")
    print(f"   {'A_o':>7}{'K_o':>7}{'on ray?':>9}   exponents found")
    cases = ((0.0, 0.0), (-0.30, -0.15), (-0.30, 0.0), (0.0, -0.30), (0.25, -0.40), (0.60, 0.10))
    for A_o, K_o in cases:
        m = OddModuli(B=B_LAT, mu=MU_LAT, A_o=A_o, K_o=K_o)
        roots = find_exponents(m)
        on_ray = "yes" if abs(A_o - 2 * K_o) < 1e-12 else "no"
        text = ", ".join(f"{r:.6f}" for r in roots)
        print(f"   {A_o:>7.2f}{K_o:>7.2f}{on_ray:>9}   {text}")

    print("\nC) mode mixing: off-diagonal of the flux matrix G, normalised by G_11")
    print("   manuscript claim: A_o drives first-order mixing, K_o does not")
    print(f"   {'K_o \\ A_o':>12}" + "".join(f"{a:>12.2f}" for a in (-0.30, -0.15, 0.0, 0.15, 0.30)))
    for K_o in (-0.30, -0.15, 0.0, 0.15, 0.30):
        row = []
        for A_o in (-0.30, -0.15, 0.0, 0.15, 0.30):
            m = OddModuli(B=B_LAT, mu=MU_LAT, A_o=A_o, K_o=K_o)
            G = J_matrix(m, n_theta=1601)
            row.append(G[0, 1] / G[0, 0])
        print(f"   {K_o:>12.2f}" + "".join(f"{v:>12.5f}" for v in row))
