#!/usr/bin/env python3
"""WP1: calibration of the MLS/keyhole reconstruction bias.

A synthetic displacement field is built from an exact Williams superposition
(lambda = 1/2, 3/2 [, 5/2] plus a uniform gradient background) for the odd
continuum on the lattice modulus ray.  The same field is then processed by

  (a) the *exact* analytic derivatives, and
  (b) the identical moving-least-squares reconstruction used in
      ``apparent_j_analysis.LocalTipField``, applied to the field sampled at
      the lattice node positions,

and both are pushed through the *same* ``keyhole_j`` / ``annulus_sources``
quadrature.  Any departure of Delta J_key / Q_o from unity in route (b) that is
absent in route (a) is, by construction, pure reconstruction bias.

The two routes differ only in the source of the local Taylor coefficients:
``AnalyticCoefficientField`` overrides ``LocalTipField._coefficients`` and
inherits every downstream operation unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


from crack_tip_asymptotics import (  # noqa: E402
    OddModuli,
    _kinematic_map,
    _vec_to_matrix,
    closed_form_propagator,
    initial_state_for_K,
    sample_K_field,
    stiffness_matrix,
)
from apparent_j_analysis import LocalTipField  # noqa: E402

Array = np.ndarray
_GEN = np.array([[0.0, -1.0], [1.0, 0.0]])


@dataclass(frozen=True)
class WilliamsTerm:
    """One Williams sector, normalised exactly as in ``crack_tip_lattice_fit``."""

    lam: float
    K_I: float
    K_II: float


class AnalyticWilliamsField:
    """Exact u, H, dH/dx, dH/dy of a Williams superposition plus a uniform gradient."""

    def __init__(
        self,
        moduli: OddModuli,
        terms: tuple[WilliamsTerm, ...],
        background_gradient: Array | None = None,
    ) -> None:
        self.moduli = moduli
        self.terms = tuple(terms)
        self.H0 = (
            np.zeros((2, 2))
            if background_gradient is None
            else np.asarray(background_gradient, dtype=float).reshape(2, 2)
        )
        self._cache: dict[float, tuple[Array, Array, Array]] = {}
        for term in self.terms:
            A, F = _kinematic_map(term.lam, moduli)
            y0 = initial_state_for_K(term.K_I, term.K_II, moduli, term.lam)
            self._cache[term.lam] = (A, F, y0)

    def _term_state(self, term: WilliamsTerm, theta: float):
        A, F, y0 = self._cache[term.lam]
        y = closed_form_propagator(theta + math.pi, term.lam, self.moduli) @ y0
        h_local = _vec_to_matrix(F @ y)
        h_local_prime = _vec_to_matrix(F @ A @ y)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rot_p = rot @ _GEN
        h = rot @ h_local @ rot.T
        h_p = rot_p @ h_local @ rot.T + rot @ h_local_prime @ rot.T + rot @ h_local @ rot_p.T
        u_hat = rot @ y[:2]
        return u_hat, h, h_p, cos_t, sin_t

    def evaluate(self, x: float, y: float) -> tuple[Array, Array, Array, Array]:
        """Return (u, H, dH/dx, dH/dy) at the tip-local point (x, y)."""
        r = math.hypot(x, y)
        u = np.zeros(2)
        H = self.H0.copy()
        dH_dx = np.zeros((2, 2))
        dH_dy = np.zeros((2, 2))
        u += self.H0 @ np.array([x, y])
        if r < 1.0e-12:
            return u, H, dH_dx, dH_dy
        theta = math.atan2(y, x)
        for term in self.terms:
            u_hat, h, h_p, cos_t, sin_t = self._term_state(term, theta)
            lam = term.lam
            u += (r ** lam) * u_hat
            H += (r ** (lam - 1.0)) * h
            radial = r ** (lam - 2.0)
            dH_dx += radial * ((lam - 1.0) * cos_t * h - sin_t * h_p)
            dH_dy += radial * ((lam - 1.0) * sin_t * h + cos_t * h_p)
        return u, H, dH_dx, dH_dy

    def coefficient_block(self, x: float, y: float) -> Array:
        """Taylor coefficients in the layout used by ``LocalTipField._coefficients``.

        Rows are [u, u_,x, u_,y, u_,xx, u_,xy, u_,yy]; columns are the two
        displacement components.
        """
        u, H, dH_dx, dH_dy = self.evaluate(x, y)
        c = np.zeros((6, 2))
        c[0] = u
        c[1] = H[:, 0]
        c[2] = H[:, 1]
        c[3] = dH_dx[:, 0]
        c[4] = dH_dx[:, 1]
        c[5] = dH_dy[:, 1]
        return c


class AnalyticCoefficientField(LocalTipField):
    """``LocalTipField`` whose local derivatives come from the analytic field.

    Everything downstream of ``_coefficients`` -- the Cauchy-Born constitutive
    map, the energy, the odd source, the equilibrium residual -- is inherited
    unchanged, so a comparison against the parent class isolates the
    reconstruction step alone.
    """

    def __init__(self, model, displacement: Array, tip: str, analytic: AnalyticWilliamsField, **kwargs) -> None:
        super().__init__(model, displacement, tip, **kwargs)
        self.analytic = analytic

    def _coefficients(self, x: float, y: float) -> Array:
        return self.analytic.coefficient_block(x, y - self.crack_y)


def synthesize_nodal_displacement(
    model, tip: str, analytic: AnalyticWilliamsField
) -> Array:
    """Sample the analytic field at every lattice node, in *global* components.

    ``LocalTipField`` maps global -> tip-local by x -> d*(x - x_tip) and
    u_x -> d*u_x.  Building the synthetic data for the right tip (d=+1) keeps
    the two frames identical and avoids a second sign convention.
    """
    if tip != "right":
        raise NotImplementedError("synthetic fields are generated at the right tip")
    direction = 1.0
    tip_x = 0.5 * model.period + direction * model.a_eff
    crack_y = 0.5 * (
        model.positions[model.node_id(0, model.j_lower), 1]
        + model.positions[model.node_id(0, model.j_lower + 1), 1]
    )
    out = np.zeros_like(model.positions)
    for node in range(len(model.positions)):
        x = float(model.positions[node, 0] - tip_x)
        y = float(model.positions[node, 1] - crack_y)
        out[node] = analytic.evaluate(x, y)[0]
    return out


def _self_test() -> None:
    """Check the analytic evaluator against ``sample_K_field`` at r = 1."""
    moduli = OddModuli(B=math.sqrt(3) / 2, mu=math.sqrt(3) / 4, A_o=-math.sqrt(3) / 2 * 0.15, K_o=-math.sqrt(3) / 4 * 0.15)
    for lam in (0.5, 1.5, 2.5):
        term = WilliamsTerm(lam=lam, K_I=1.0, K_II=0.37)
        field = AnalyticWilliamsField(moduli, (term,))
        theta = np.linspace(-math.pi + 0.05, math.pi - 0.05, 41)
        reference = sample_K_field(term.K_I, term.K_II, moduli, theta, lam)
        err_u = err_h = err_dx = 0.0
        for index, angle in enumerate(theta):
            x, y = math.cos(float(angle)), math.sin(float(angle))
            u, H, dH_dx, _ = field.evaluate(x, y)
            err_u = max(err_u, float(np.max(np.abs(u - reference["u"][index]))))
            sigma = _vec_to_matrix(stiffness_matrix(moduli) @ np.array([H[0, 0], H[0, 1], H[1, 0], H[1, 1]]))
            err_h = max(err_h, float(np.max(np.abs(sigma - reference["sigma"][index]))))
            err_dx = max(err_dx, float(np.max(np.abs(dH_dx - reference["dH_dx"][index]))))
        print(f"  lambda={lam}:  max|du|={err_u:.3e}  max|dsigma|={err_h:.3e}  max|d(dH/dx)|={err_dx:.3e}")


if __name__ == "__main__":
    print("self-test: analytic evaluator vs sample_K_field at r=1")
    _self_test()
