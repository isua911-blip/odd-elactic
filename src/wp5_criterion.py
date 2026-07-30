#!/usr/bin/env python3
"""WP5: is there a coarse-grained advance criterion for the odd lattice?

The lattice's own rule is microscopic: the crack advances when the tip bond
extension reaches delta_c.  A *coarse-grained* criterion "X >= X_c" can only
replace it if X, evaluated at the lattice threshold, is the same number for
every k_o.  If X drifts with k_o, no criterion of that form exists.

That turns M7 into a cheap, decisive measurement.  The model is linear, so at
fixed k_o every field scales with the grip displacement: one unit-load solve
fixes the critical load factor p_c = delta_c / e_tip, and every candidate is a
quadratic form scaled by p_c^2.

Candidates
    C1   J_h            apparent flux on a fixed contour (what the manuscript reports)
    C2   J_h - Q_o      flux corrected back to the tip by the enclosed odd source
    C3  -dU_e / da      recoverable-energy release rate
    ref  E_cut / da     tip-bond energy: constant by construction, the control

Only the on-plane first advance is tested.  The cascade leaves the cleavage
plane at step two, and asking a straight-advance criterion to predict a
branching event would make any negative result uninformative.
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


from propagation_limit_analysis import (  # noqa: E402
    active_ids,
    assemble_components,
    bond_extension,
    crossing_frontier,
    solve_equilibrium,
)
from active_tip_scan import ActiveCrackedStrip  # noqa: E402
from apparent_j_analysis import solve_field  # noqa: E402
from continuum_domain_core import MLSSampler, weight  # noqa: E402

DELTA_C = 0.02
DA = 0.5  # one crossing bond advances the crack by half a lattice spacing


def even_energy(model, removed, u) -> float:
    """Recoverable (central-spring) energy of the current bond set."""
    total = 0.0
    for bond_id in active_ids(model, removed):
        bond = model.all_bonds[bond_id]
        total += 0.5 * model.k * bond_extension(bond, u) ** 2
    return float(total)


def state_quantities(nx, ny, crack, k_o):
    """Return the critical load factor and the load-normalised energy terms."""
    model = ActiveCrackedStrip(nx, ny, crack, 1.0, k_o)
    removed = set(model.removed_ids)

    even, odd = assemble_components(model, active_ids(model, removed))
    u, *_ = solve_equilibrium(even + odd, model, 1.0)

    candidates = crossing_frontier(model, removed)
    extensions = {s: bond_extension(model.all_bonds[b], u) for s, b in candidates.items()}
    side = max(extensions, key=extensions.get)
    p_c = DELTA_C / extensions[side]

    U_before = even_energy(model, removed, u)
    cut = removed | {candidates[side]}
    even2, odd2 = assemble_components(model, active_ids(model, cut))
    u2, *_ = solve_equilibrium(even2 + odd2, model, 1.0)
    U_after = even_energy(model, cut, u2)

    return {
        "p_c": p_c,
        "side": side,
        "dUe": (U_after - U_before) * p_c**2,
        "E_cut": 0.5 * model.k * DELTA_C**2,
    }


def flux_terms(sampler, passive_grid, nx, ny, crack, k_o, fit, Ri, Ro, w=1.5):
    """Domain-form J and the enclosed odd source, both at unit load."""
    _m, field, _r = solve_field(nx, ny, crack, k_o, "right", fit)
    grid = sampler.apply(field)
    q_o, q_ox, q_oy = weight(grid, Ro, w, 4)
    q_i, q_ix, q_iy = weight(grid, Ri, w, 4)
    area = grid.step**2
    J_out = -np.sum(grid.p_x * q_ox + grid.p_y * q_oy) * area
    shell = q_o - q_i
    Q_o = float(np.sum(shell * grid.source_odd) * area)
    return float(J_out), Q_o


if __name__ == "__main__":
    nx, ny = 128, 96
    crack = nx / 4.0
    fit, step, hw = 3.0, 0.35, 26.0
    Ri, Ro = 8.0, 16.0

    _m, pf, _r = solve_field(nx, ny, crack, 0.0, "right", fit)
    sampler = MLSSampler(pf, hw, step)
    passive_grid = sampler.apply(pf)

    print("WP5  candidate coarse-grained quantities evaluated at the lattice advance threshold")
    print(f"     nx={nx}, a=nx/4, delta_c={DELTA_C}, da={DA}; each column normalised to its k_o=0 value")
    print("     a valid criterion must stay at 1.000\n")
    print(f"{'k_o':>6}{'p_c':>10}{'side':>7}   |{'C1  J_h':>11}{'C2  J_h-Q':>11}{'C3 -dUe/da':>12}{'ref E_cut':>11}")

    reference: dict[str, float] = {}
    for k_o in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40):
        s = state_quantities(nx, ny, crack, k_o)
        J1, Q = flux_terms(sampler, passive_grid, nx, ny, crack, k_o, fit, Ri, Ro)
        p2 = s["p_c"] ** 2
        vals = {
            "C1": J1 * p2,
            "C2": (J1 - Q) * p2,
            "C3": -s["dUe"] / DA,
            "ref": s["E_cut"] / DA,
        }
        if not reference:
            reference = dict(vals)
        norm = {key: vals[key] / reference[key] for key in vals}
        print(
            f"{k_o:>6.2f}{s['p_c']:>10.4f}{s['side']:>7}   |"
            f"{norm['C1']:>11.4f}{norm['C2']:>11.4f}{norm['C3']:>12.4f}{norm['ref']:>11.4f}",
            flush=True,
        )
