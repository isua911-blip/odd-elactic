#!/usr/bin/env python3
"""WP3: how much of the odd work is couple work against material rotation?

Each odd bond pair carries equal and opposite transverse forces, so it exerts no
net force but a net couple  M_b = a * k_o * delta_b  on the material.  The model
closes the balance by assuming the compensating substrate torque does no work on
retained degrees of freedom (manuscript Eq. 25).

The bond's own rotation rate decomposes, for an affine gradient H, as

    tau_b / a  =  t.H.n  =  phi  +  t.eps.n ,

with phi = (H_yx - H_xy)/2 the material rotation and eps the symmetric strain.
Pairing the couple with the *bond* rotation just reproduces W_odd, which is
circular.  Pairing it with the *material* rotation phi isolates the channel a
physical rotor reservoir would actually couple to.  The reported bound is

    R_torque = sum_b delta_b phi_b / sum_b delta_b (tau_b / a).

Because the model is linear, a quasi-static load path gives W = (1/2) * (the
corresponding quadratic form), so the ratio needs no time integration.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import sys
from pathlib import Path

import numpy as np


from lattice_baselines import R90  # noqa: E402
from active_tip_scan import ActiveCrackedStrip  # noqa: E402


def nodal_rotation(model, u: np.ndarray) -> np.ndarray:
    """Material rotation phi = (H_yx - H_xy)/2 at every node.

    H is fitted per node from its bonded neighbours using the *reference* bond
    vectors, so periodic wrap-around needs no special handling.
    """
    n_nodes = len(model.positions)
    dR: list[list[np.ndarray]] = [[] for _ in range(n_nodes)]
    dU: list[list[np.ndarray]] = [[] for _ in range(n_nodes)]
    for bond in model.active_bonds:
        du = u[bond.j] - u[bond.i]
        dR[bond.i].append(bond.n)
        dU[bond.i].append(du)
        dR[bond.j].append(-bond.n)
        dU[bond.j].append(-du)

    phi = np.zeros(n_nodes)
    for p in range(n_nodes):
        if len(dR[p]) < 2:
            continue
        A = np.asarray(dR[p])           # [m, 2] reference offsets
        B = np.asarray(dU[p])           # [m, 2] displacement differences
        H, *_ = np.linalg.lstsq(A, B, rcond=None)   # solves A @ H = B, so H = grad(u)^T
        H = H.T
        phi[p] = 0.5 * (H[1, 0] - H[0, 1])
    return phi


def torque_budget(nx: int, ny: int, crack: float, k_o: float) -> dict[str, float]:
    model = ActiveCrackedStrip(nx=nx, ny=ny, crack_half_length=crack, k=1.0, k_o=k_o)
    u, _reactions, residual = model.solve(delta=1.0)
    phi = nodal_rotation(model, u)

    sum_dt = 0.0      # sum delta * tau / a   -> W_odd density
    sum_dphi = 0.0    # sum delta * phi       -> couple work against material spin
    sum_dshear = 0.0
    for bond in model.active_bonds:
        du = u[bond.j] - u[bond.i]
        t = R90 @ bond.n
        delta = float(du @ bond.n)
        tau = float(du @ t)             # a = 1 in lattice units
        phi_b = 0.5 * (phi[bond.i] + phi[bond.j])
        sum_dt += delta * tau
        sum_dphi += delta * phi_b
        sum_dshear += delta * (tau - phi_b)
    return {
        "k_o": k_o,
        "nx": nx,
        "residual": residual,
        "W_odd_form": sum_dt,
        "W_couple_form": sum_dphi,
        "W_shear_form": sum_dshear,
        "R_torque": sum_dphi / sum_dt if sum_dt != 0 else float("nan"),
    }


if __name__ == "__main__":
    print("WP3  substrate-couple work channel relative to the odd work")
    print("R_torque = (couple work against material rotation) / (total odd work)\n")
    print(f"{'nx':>5}{'k_o':>7}{'W_odd form':>14}{'couple form':>14}{'shear form':>14}{'R_torque':>11}")
    for nx, ny, crack in ((96, 72, 12.0), (128, 96, 16.0)):
        for k_o in (0.05, 0.10, 0.20, 0.40):
            d = torque_budget(nx, ny, crack, k_o)
            print(
                f"{d['nx']:>5}{d['k_o']:>7.2f}{d['W_odd_form']:>14.5e}"
                f"{d['W_couple_form']:>14.5e}{d['W_shear_form']:>14.5e}{d['R_torque']:>11.4f}",
                flush=True,
            )
