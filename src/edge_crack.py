#!/usr/bin/env python3
"""Single-edge-notch (SENT) odd-elastic lattice: one crack tip, no periodic image.

The centred-crack periodic strip used elsewhere in the package caps the usable
J-domain radius at half the tip separation.  Here x-periodicity is dropped
(free left and right edges) and the crack is cut inward from the left edge, so
the only obstructions are the free surfaces and the grips.

Compatibility trick: ``LocalTipField`` locates the tip as
``0.5*period + direction*a_eff``.  Rather than patch that class, ``a_eff`` is
*defined* here as ``tip_x - 0.5*period`` so the existing expression evaluates to
the true tip abscissa.  Every downstream consumer (LocalTipField, MLSSampler,
keyhole_j, annulus_sources, continuum_domain_core.pair) then works unchanged.
``a_eff`` therefore no longer means "effective half-length" for this class; the
physical crack length is ``self.crack_length``.
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


from lattice_baselines import A1, A2, A3, Bond  # noqa: E402
from active_tip_scan import ActiveCrackedStrip  # noqa: E402


class EdgeCrackedStrip(ActiveCrackedStrip):
    """Triangular odd-elastic strip, free in x, fixed-grip in y, one edge crack."""

    def __init__(self, nx: int, ny: int, crack_length: float, k: float, k_o: float) -> None:
        self.crack_length = float(crack_length)
        super().__init__(
            nx=nx, ny=ny, crack_half_length=crack_length, k=k, k_o=k_o
        )

    # -- geometry ---------------------------------------------------------
    def _all_bonds(self) -> list[Bond]:
        """Same triangular connectivity, with every wrapping bond dropped."""
        bonds: list[Bond] = []
        for j in range(self.ny):
            for i in range(self.nx):
                p = self.node_id(i, j)
                if i + 1 < self.nx:
                    bonds.append(Bond(p, self.node_id(i + 1, j), A1.copy()))
                if j < self.ny - 1:
                    crosses = j == self.j_lower
                    # up-right bond, reference vector A2
                    midx = (i + 0.25 + 0.5 * j) if crosses else None
                    bonds.append(Bond(p, self.node_id(i, j + 1), A2.copy(), midx, crosses))
                    # up-left bond, reference vector A3; absent on the left edge
                    if i - 1 >= 0:
                        midx = (i - 0.25 + 0.5 * j) if crosses else None
                        bonds.append(
                            Bond(p, self.node_id(i - 1, j + 1), A3.copy(), midx, crosses)
                        )
        return bonds

    def _make_crack(self) -> tuple[set[int], list[Bond], list[int], float]:
        crossing = [
            (bid, float(bond.midpoint_x))
            for bid, bond in enumerate(self.all_bonds)
            if bond.crosses_crack_plane
        ]
        if not crossing:
            raise RuntimeError("No bonds cross the crack plane")
        mouth = min(x for _, x in crossing)
        cut_to = mouth + self.crack_length

        removed = {bid for bid, x in crossing if x <= cut_to}
        intact = [(bid, x) for bid, x in crossing if bid not in removed]
        if not removed:
            raise RuntimeError("No bonds removed; increase crack_length")
        if not intact:
            raise RuntimeError("Crack severed the ligament; reduce crack_length")

        last_removed = max(x for bid, x in crossing if bid in removed)
        first_intact = min(x for _, x in intact)
        tip_x = 0.5 * (last_removed + first_intact)

        tol = 1.0e-12
        candidates = [bid for bid, x in intact if abs(x - first_intact) < tol]
        active = [b for bid, b in enumerate(self.all_bonds) if bid not in removed]

        self.tip_x = float(tip_x)
        self.crack_mouth_x = float(mouth)
        # see module docstring: makes LocalTipField's tip expression exact
        return removed, active, candidates, float(tip_x - 0.5 * self.period)

    # -- diagnostics ------------------------------------------------------
    def clearances(self) -> dict[str, float]:
        """Largest J-domain radius admitted by each boundary, in lattice spacings."""
        y = self.positions[:, 1]
        crack_y = 0.5 * (
            self.positions[self.node_id(0, self.j_lower), 1]
            + self.positions[self.node_id(0, self.j_lower + 1), 1]
        )
        row = self.positions[
            [self.node_id(i, self.j_lower) for i in range(self.nx)], 0
        ]
        return {
            "to_crack_mouth": self.tip_x - self.crack_mouth_x,
            "to_right_edge": float(row.max()) - self.tip_x,
            "to_top_grip": float(y.max()) - crack_y,
            "to_bottom_grip": crack_y - float(y.min()),
        }


def solve_edge_field(nx, ny, crack_length, k_o, fit_radius):
    """Return (model, LocalTipField, free-equilibrium residual) for the right-facing tip."""
    from apparent_j_analysis import LocalTipField

    model = EdgeCrackedStrip(nx=nx, ny=ny, crack_length=crack_length, k=1.0, k_o=k_o)
    displacement, _reactions, residual = model.solve(delta=1.0)
    field = LocalTipField(model, displacement, "right", fit_radius=fit_radius)
    return model, field, residual


if __name__ == "__main__":
    m, f, res = solve_edge_field(64, 48, 12.0, 0.15, 3.0)
    print(f"nodes={len(m.positions)}  bonds={len(m.active_bonds)}  removed={len(m.removed_ids)}")
    print(f"crack_length={m.crack_length}  mouth_x={m.crack_mouth_x:.3f}  tip_x={m.tip_x:.3f}")
    print(f"LocalTipField tip_x_global={f.tip_x_global:.3f}   (must match tip_x)")
    print(f"free equilibrium residual = {res:.3e}")
    print("clearances (lattice spacings):")
    for name, value in m.clearances().items():
        print(f"   {name:<18}{value:8.2f}")
