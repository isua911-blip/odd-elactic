#!/usr/bin/env python3
"""Recompute the recoverable-energy representation comparison.

The archived values (measured 6.81%, microenergetic 9.35%, major-symmetric
5.08%) were obtained with alpha_J = 1.269 and a reconstruction radius scaled
with the domain.  Both settings are superseded, so the comparison is redone
here under the converged protocol: a = N_x/4, absolute fit radius, Williams
basis to lambda = 5/2, and the wide fitting annulus.

The archived text also flagged the transfer of a single passive amplitude
calibration to active increments as "an explicit, unverified assumption".  That
is now testable directly: alpha_J is evaluated separately at k_o = 0 and at
each active k_o, and their ratio measures the transfer error.
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


from apparent_j_analysis import keyhole_j, solve_field  # noqa: E402
from crack_tip_asymptotics import J_matrix  # noqa: E402
import crack_tip_lattice_fit as ctlf  # noqa: E402
from wp2_closeout import fit_case  # noqa: E402  (patches PeriodicTipField on import)

NX = 160
NY = 3 * NX // 4
A = NX / 4.0
FIT = 3.0
WIDE = (0.035, 0.22)
LAMS = (0.5, 1.5, 2.5)
RADII_FRAC = (0.06, 0.10, 0.14, 0.18)


def measured_J(k_o: float) -> tuple[float, float]:
    """Lattice flux averaged over contours inside the fitting annulus."""
    _m, field, _r = solve_field(NX, NY, A, k_o, "right", FIT)
    vals = [keyhole_j(field, float(f * NX), 0.02) for f in RADII_FRAC]
    return float(np.mean(vals)), float((max(vals) - min(vals)) / abs(np.mean(vals)))


def predicted_J(k_o: float, choice: str) -> float:
    """K^T G K with K fitted from the lattice and G from the chosen representation."""
    KI, KII, _res = fit_case(NX, k_o, "right", LAMS, fit_radius=FIT,
                             window=WIDE, n_radii=13)
    moduli = ctlf.homogenized_moduli(1.0, k_o)
    G = J_matrix(moduli, n_theta=1601, energy_choice=choice)
    K = np.array([KI, KII])
    return float(K @ G @ K)


if __name__ == "__main__":
    J0, spread0 = measured_J(0.0)
    P0 = predicted_J(0.0, "micro_hessian")
    alpha0 = J0 / P0
    print(f"protocol: N_x={NX}, a=N_x/4, fit radius {FIT} (absolute), "
          f"basis lambda<=5/2, annulus {WIDE[0]}-{WIDE[1]} L")
    print(f"passive: J_h={J0:.6e} (contour spread {spread0:.2%}), "
          f"K.G0.K={P0:.6e}, alpha_J={alpha0:.4f}\n")

    print("A) is the passive amplitude calibration transferable to active increments?")
    print(f"   {'k_o':>6}{'alpha_J(k_o)':>15}{'alpha_J(k_o)/alpha_J(0)':>26}")
    for k_o in (0.10, 0.20, 0.30):
        a_k = measured_J(k_o)[0] / predicted_J(k_o, "micro_hessian")
        print(f"   {k_o:>6.2f}{a_k:>15.4f}{a_k / alpha0:>26.4f}")

    print("\nB) representation discrimination: relative flux increment over passive")
    print(f"   {'k_o':>6}{'measured':>12}{'microenergetic':>16}{'major-symmetric':>17}"
          f"{'spread':>9}")
    for k_o in (0.10, 0.20, 0.30):
        Jk, spread = measured_J(k_o)
        meas = Jk / J0 - 1.0
        micro = predicted_J(k_o, "micro_hessian") / P0 - 1.0
        ms = predicted_J(k_o, "major_symmetric_projection") / P0 - 1.0
        print(f"   {k_o:>6.2f}{meas:>11.2%}{micro:>16.2%}{ms:>17.2%}{spread:>9.2%}")
