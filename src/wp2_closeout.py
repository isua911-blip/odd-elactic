#!/usr/bin/env python3
"""WP2 closeout: lambda=5/2 basis extension and the +/-k_o tip asymmetry.

``crack_tip_lattice_fit.continuum_basis`` is hard-wired to eight columns
(lambda = 1/2 and 3/2, each with a K_I and K_II mode, plus four constant stress
components).  ``extended_basis`` generalises it to an arbitrary set of Williams
exponents so the truncation error in the amplitude calibration can be measured
rather than assumed.

All fits route through ``PeriodicTipField`` so that left-tip and large-crack
configurations are reconstructed with a complete neighbourhood.
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


from periodic_tip import patch_lattice_fit  # noqa: E402

patch_lattice_fit()

import crack_tip_lattice_fit as ctlf  # noqa: E402
from crack_tip_asymptotics import J_matrix, sample_K_field  # noqa: E402

COMPONENTS = ctlf.COMPONENTS


def extended_basis(samples, moduli, lams=(0.5, 1.5)):
    """Design tensor [n_points, 4 stress components, 2*len(lams)+4 coefficients]."""
    n_points = len(samples.radius)
    n_coeff = 2 * len(lams) + 4
    design = np.zeros((n_points, 4, n_coeff), dtype=float)
    for radius in np.unique(samples.radius):
        ids = np.flatnonzero(np.isclose(samples.radius, radius, atol=1.0e-12))
        theta = samples.theta[ids]
        fields = []
        for lam in lams:
            scale = radius ** (lam - 1.0)
            fields.append(sample_K_field(1.0, 0.0, moduli, theta, lam)["sigma"] * scale)
            fields.append(sample_K_field(0.0, 1.0, moduli, theta, lam)["sigma"] * scale)
        for local, point in enumerate(ids):
            for component, (row, column) in enumerate(COMPONENTS):
                for basis_id, basis in enumerate(fields):
                    design[point, component, basis_id] = basis[local, row, column]
                design[point, component, 2 * len(lams) + component] = 1.0
    return design


def fit_case(nx, k_o, tip="right", lams=(0.5, 1.5), fit_radius=3.0,
             window=(0.06, 0.12), n_radii=9):
    """Return K_I, K_II and the relative L2 stress residual for one configuration."""
    ny = 3 * nx // 4
    a = nx / 4.0
    r_in, r_out = window[0] * nx, window[1] * nx
    radii = np.linspace(r_in, r_out, n_radii)
    angles = np.linspace(-math.pi + 0.10, math.pi - 0.10, 73)

    model = ctlf.ActiveCrackedStrip(nx=nx, ny=ny, crack_half_length=a, k=1.0, k_o=k_o)
    displacement, _r, _res = model.solve(delta=1.0)
    field, samples = ctlf.sample_lattice_stress(
        model, displacement, tip, radii, angles, fit_radius=fit_radius
    )
    moduli = ctlf.homogenized_moduli(model.k, field.k_o_local)
    design = extended_basis(samples, moduli, lams).reshape((-1, 2 * len(lams) + 4))
    target = samples.stress.reshape(-1)
    coeff, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = float(
        np.linalg.norm(target - design @ coeff) / max(np.linalg.norm(target), 1e-30)
    )
    return float(coeff[0]), float(coeff[1]), residual


if __name__ == "__main__":
    import time

    from apparent_j_analysis import keyhole_j, solve_field

    G_passive = J_matrix(ctlf.homogenized_moduli(1.0, 0.0), n_theta=801)
    SETS = {"lam<=3/2 (paper)": (0.5, 1.5), "lam<=5/2": (0.5, 1.5, 2.5)}

    print("A)  amplitude calibration alpha_J = J_measured / K.G.K   (passive, right tip)")
    header = f"{'nx':>5}" + "".join(f"{name:>20}" for name in SETS)
    print(header)
    for nx in (48, 64, 96, 128, 160):
        t = time.time()
        _m, f, _r = solve_field(nx, 3 * nx // 4, nx / 4.0, 0.0, "right", 3.0)
        J_meas = float(np.mean([keyhole_j(f, float(R), 0.02) for R in (0.06 * nx, 0.09 * nx, 0.12 * nx)]))
        cells = []
        for lams in SETS.values():
            KI, KII, _res = fit_case(nx, 0.0, "right", lams)
            K = np.array([KI, KII])
            cells.append(J_meas / float(K @ G_passive @ K))
        print(f"{nx:>5}" + "".join(f"{v:>20.4f}" for v in cells) + f"   [{time.time()-t:.0f}s]", flush=True)

    print("\nB)  right-tip chirality asymmetry  |K_II(+k_o)| / |K_II(-k_o)|   at |k_o|=0.20")
    print(f"{'nx':>5}{'K_II/K_I (+)':>15}{'K_II/K_I (-)':>15}{'ratio':>9}{'fit resid':>11}")
    for nx in (48, 64, 96, 128, 160):
        KIp, KIIp, res = fit_case(nx, 0.20, "right", (0.5, 1.5, 2.5))
        KIm, KIIm, _ = fit_case(nx, -0.20, "right", (0.5, 1.5, 2.5))
        rp, rm = KIIp / KIp, KIIm / KIm
        print(f"{nx:>5}{rp:>15.5f}{rm:>15.5f}{abs(rm/rp):>9.4f}{res:>11.4f}", flush=True)
