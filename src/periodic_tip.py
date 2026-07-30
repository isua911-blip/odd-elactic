#!/usr/bin/env python3
"""Periodic-aware local tip reconstruction.

``PassiveCrackedStrip`` is periodic in x: bonds wrap through ``node_id(i % nx)``,
so node ``(0, j)`` simultaneously represents the material point at
``x = 0.5*j`` and at ``x = nx + 0.5*j``.  Node *positions*, however, are stored
unwrapped as ``i*A1 + j*A2``, so the stored point cloud is a sheared
parallelogram spanning ``x in [0, nx-1 + 0.5*(ny-1)]``.

``LocalTipField`` builds its KD-tree directly on that cloud, so the moving
least-squares neighbourhood is blind to the periodicity.  For the right tip this
is harmless -- the tip sits well inside the parallelogram.  For the left tip the
sampling annulus runs off the slanted left boundary: at the crack line row the
lattice starts at x = 0.5*j_lower, and every row above it starts further right,
so points up-and-left of the tip have no neighbours at all.  The quadratic fit
then goes rank deficient (or, worse, silently one-sided).

Replicating the cloud at x offsets {-period, 0, +period} restores the true
neighbourhood.  The displacement field is genuinely periodic, so images carry
identical displacements.
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
from scipy.spatial import cKDTree


from apparent_j_analysis import LocalTipField  # noqa: E402


class PeriodicTipField(LocalTipField):
    """``LocalTipField`` whose neighbour search respects x-periodicity."""

    def __init__(self, model, displacement, tip, **kwargs) -> None:
        super().__init__(model, displacement, tip, **kwargs)
        period = float(model.period)
        base = np.asarray(model.positions, dtype=float)
        raw = np.asarray(displacement, dtype=float)

        tiles = []
        for shift in (-period, 0.0, period):
            tile = base.copy()
            tile[:, 0] += shift
            tiles.append(tile)
        positions = np.vstack(tiles)
        # identical transform to the parent: fold into tip-local coordinates
        positions[:, 0] = self.direction * (positions[:, 0] - self.tip_x_global)

        images = np.vstack([raw, raw, raw])
        images[:, 0] *= self.direction

        self.positions = positions
        self.displacement = images
        self.tree = cKDTree(positions)
        self.n_images = 3


def patch_lattice_fit() -> None:
    """Route ``crack_tip_lattice_fit.sample_lattice_stress`` through the fixed class."""
    import crack_tip_lattice_fit as ctlf

    ctlf.LocalTipField = PeriodicTipField


if __name__ == "__main__":
    from active_tip_scan import ActiveCrackedStrip

    nx, ny, a = 128, 96, 32.0
    model = ActiveCrackedStrip(nx=nx, ny=ny, crack_half_length=a, k=1.0, k_o=0.2)
    u, _, _ = model.solve(delta=1.0)
    for tip in ("right", "left"):
        old = LocalTipField(model, u, tip, fit_radius=3.0)
        new = PeriodicTipField(model, u, tip, fit_radius=3.0)
        print(f"{tip:>6} tip: nodes seen  old={len(old.positions):6d}  new={len(new.positions):6d}")
        for probe in ((-12.0, 8.0), (-12.0, -8.0), (10.0, 0.0)):
            x, y = probe[0], new.crack_y + probe[1]
            try:
                a_val = old.evaluate(x, y).energy_even
                a_txt = f"{a_val:.4e}"
            except RuntimeError as exc:
                a_txt = f"FAIL ({exc.args[0][:22]})"
            b_val = new.evaluate(x, y).energy_even
            print(f"         probe(local x={probe[0]:+6.1f}, dy={probe[1]:+5.1f})  old={a_txt:>28}  new={b_val:.4e}")
