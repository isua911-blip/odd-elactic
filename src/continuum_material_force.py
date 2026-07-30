#!/usr/bin/env python3
"""Continuum material-force identities for linear non-potential elasticity.

The primary setting is a small-strain Cauchy solid

    sigma = C^e : epsilon + C^o : epsilon,

where C^e is major symmetric and C^o is major antisymmetric.  The module also
states the same algebra for a generic strain-like variable E=L:grad(u), which is
useful for the substrate-supported odd lattice whose coarse-grained stress need
not be symmetric.

No symbolic package is required.  ``run_manufactured_verification`` evaluates
all terms of the local identity from analytic polynomial fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class BalanceTerms:
    """Terms in div(P_app)_K = material + body + nonpotential."""

    divergence_direct: float
    energetic_inhomogeneity: float
    body_force_source: float
    nonpotential_source: float

    @property
    def divergence_predicted(self) -> float:
        return (
            self.energetic_inhomogeneity
            + self.body_force_source
            + self.nonpotential_source
        )

    @property
    def residual(self) -> float:
        return self.divergence_direct - self.divergence_predicted


def symmetric_tensor_basis() -> Array:
    """Orthonormal basis for 2D symmetric tensors.

    Components are (E_xx, E_yy, sqrt(2) E_xy), so tensor contraction is the
    Euclidean dot product of component vectors.
    """

    root2 = np.sqrt(2.0)
    return np.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0 / root2], [1.0 / root2, 0.0]],
        ]
    )


def tensor_to_components(tensor: Array, basis: Array) -> Array:
    return np.einsum("aij,ij->a", basis, tensor)


def components_to_tensor(components: Array, basis: Array) -> Array:
    return np.einsum("a,aij->ij", components, basis)


def manufactured_displacement(x: float, y: float) -> tuple[Array, Array, Array]:
    """Return u, H=grad(u), and dH[K,i,j]=u_{i,jK}."""

    u = np.array(
        [
            0.10 * x - 0.04 * y + 0.030 * x * x + 0.020 * x * y - 0.010 * y * y,
            -0.02 * x - 0.05 * y - 0.020 * x * x + 0.040 * x * y + 0.015 * y * y,
        ]
    )
    H = np.array(
        [
            [0.10 + 0.060 * x + 0.020 * y, -0.04 + 0.020 * x - 0.020 * y],
            [-0.02 - 0.040 * x + 0.040 * y, -0.05 + 0.040 * x + 0.030 * y],
        ]
    )
    dH = np.zeros((2, 2, 2))
    # K=x
    dH[0] = np.array([[0.060, 0.020], [-0.040, 0.040]])
    # K=y
    dH[1] = np.array([[0.020, -0.020], [0.040, 0.030]])
    return u, H, dH


def constitutive_matrices(x: float, y: float) -> tuple[Array, Array, Array, Array]:
    """Return Ce, Co and their spatial derivatives in symmetric-tensor space."""

    ce0 = np.array(
        [
            [3.00, 0.70, 0.15],
            [0.70, 2.60, -0.10],
            [0.15, -0.10, 1.80],
        ]
    )
    co0 = np.array(
        [
            [0.00, 0.32, -0.21],
            [-0.32, 0.00, 0.17],
            [0.21, -0.17, 0.00],
        ]
    )
    se = 1.0 + 0.12 * x - 0.07 * y
    so = 0.65 - 0.05 * x + 0.09 * y
    Ce = se * ce0
    Co = so * co0
    dCe = np.stack((0.12 * ce0, -0.07 * ce0))
    dCo = np.stack((-0.05 * co0, 0.09 * co0))
    return Ce, Co, dCe, dCo


def evaluate_small_strain_balance(x: float, y: float, K: int) -> BalanceTerms:
    """Evaluate the exact local balance for the manufactured Cauchy field.

    Sign convention: physical equilibrium is div(sigma)+b=0.  For

        P_app[K,j] = W_e delta[K,j] - sigma[i,j] u[i,K],

    the identity is

        div(P_app)_K = 1/2 eps:C^e_{,K}:eps
                         + b_i u_{i,K}
                         - sigma^o_{ij} u_{i,jK}.
    """

    basis = symmetric_tensor_basis()
    _, H, dH = manufactured_displacement(x, y)
    eps = 0.5 * (H + H.T)
    deps = 0.5 * (dH + np.swapaxes(dH, 1, 2))
    e = tensor_to_components(eps, basis)
    de = np.stack([tensor_to_components(deps[a], basis) for a in range(2)])
    Ce, Co, dCe, dCo = constitutive_matrices(x, y)

    se = components_to_tensor(Ce @ e, basis)
    so = components_to_tensor(Co @ e, basis)
    sigma = se + so

    # div(sigma)_i = partial_j sigma_ij, evaluated analytically.
    div_sigma = np.zeros(2)
    for j in range(2):
        ds_components = (dCe[j] + dCo[j]) @ e + (Ce + Co) @ de[j]
        dsigma = components_to_tensor(ds_components, basis)
        div_sigma += dsigma[:, j]
    body = -div_sigma

    energetic_inhomogeneity = 0.5 * float(e @ dCe[K] @ e)
    body_force_source = float(body @ H[:, K])
    nonpotential_source = -float(np.einsum("ij,ij->", so, dH[K]))

    # Direct product-rule evaluation of div(P_app).
    dW = float((Ce @ e) @ de[K] + 0.5 * e @ dCe[K] @ e)
    divergence_direct = (
        dW
        - float(div_sigma @ H[:, K])
        - float(np.einsum("ij,ij->", sigma, dH[K]))
    )
    return BalanceTerms(
        divergence_direct=divergence_direct,
        energetic_inhomogeneity=energetic_inhomogeneity,
        body_force_source=body_force_source,
        nonpotential_source=nonpotential_source,
    )



def full_gradient_constitutive_matrices(x: float, y: float) -> tuple[Array, Array, Array, Array]:
    """Return major-symmetric/antisymmetric operators on vec(H).

    The component order is (H_xx,H_xy,H_yx,H_yy).  This manufactured law is
    deliberately nonsymmetric in the minor indices, matching the
    substrate-supported lattice continuum, while retaining the major
    symmetry required for the recoverable even quadratic form.
    """

    ce0 = np.array(
        [
            [3.2, 0.2, -0.1, 0.7],
            [0.2, 1.9, 0.35, -0.05],
            [-0.1, 0.35, 1.6, 0.15],
            [0.7, -0.05, 0.15, 2.8],
        ]
    )
    co0 = np.array(
        [
            [0.0, 0.24, -0.16, 0.11],
            [-0.24, 0.0, 0.19, -0.07],
            [0.16, -0.19, 0.0, 0.13],
            [-0.11, 0.07, -0.13, 0.0],
        ]
    )
    se = 1.0 + 0.08 * x - 0.04 * y
    so = 0.55 - 0.03 * x + 0.06 * y
    Ce = se * ce0
    Co = so * co0
    dCe = np.stack((0.08 * ce0, -0.04 * ce0))
    dCo = np.stack((-0.03 * co0, 0.06 * co0))
    return Ce, Co, dCe, dCo


def evaluate_full_gradient_balance(x: float, y: float, K: int) -> BalanceTerms:
    """Evaluate the full-displacement-gradient material-force identity.

    For W_e=1/2 H:C^e:H, sigma=(C^e+C^o):H and div(sigma)+b=0,

        div(P_app)_K = 1/2 H:C^e_{,K}:H + b_i u_{i,K}
                       - sigma^o_{ij} u_{i,jK}.
    """

    _, H, dH = manufactured_displacement(x, y)
    h = H.reshape(-1)
    dh = np.stack([dH[a].reshape(-1) for a in range(2)])
    Ce, Co, dCe, dCo = full_gradient_constitutive_matrices(x, y)
    se = (Ce @ h).reshape(2, 2)
    so = (Co @ h).reshape(2, 2)
    sigma = se + so

    div_sigma = np.zeros(2)
    for j in range(2):
        dsigma = ((dCe[j] + dCo[j]) @ h + (Ce + Co) @ dh[j]).reshape(2, 2)
        div_sigma += dsigma[:, j]
    body = -div_sigma

    energetic_inhomogeneity = 0.5 * float(h @ dCe[K] @ h)
    body_force_source = float(body @ H[:, K])
    nonpotential_source = -float(np.einsum("ij,ij->", so, dH[K]))
    dW = float((Ce @ h) @ dh[K] + 0.5 * h @ dCe[K] @ h)
    divergence_direct = (
        dW
        - float(div_sigma @ H[:, K])
        - float(np.einsum("ij,ij->", sigma, dH[K]))
    )
    return BalanceTerms(
        divergence_direct=divergence_direct,
        energetic_inhomogeneity=energetic_inhomogeneity,
        body_force_source=body_force_source,
        nonpotential_source=nonpotential_source,
    )

def run_manufactured_verification(output: Path | None = None) -> dict[str, object]:
    """Check major symmetries, passive reduction, and the generalized identity."""

    points = [(-0.43, -0.31), (-0.17, 0.28), (0.11, -0.22), (0.37, 0.19)]
    rows: list[dict[str, float | int | str]] = []
    max_residual = 0.0
    max_full_gradient_residual = 0.0
    for x, y in points:
        for K in (0, 1):
            for formulation, evaluator in (
                ("symmetric_strain", evaluate_small_strain_balance),
                ("full_gradient", evaluate_full_gradient_balance),
            ):
                terms = evaluator(x, y, K)
                max_residual = max(max_residual, abs(terms.residual))
                if formulation == "full_gradient":
                    max_full_gradient_residual = max(
                        max_full_gradient_residual, abs(terms.residual)
                    )
                rows.append(
                    {
                        "formulation": formulation,
                        "x": x,
                        "y": y,
                        "K": K,
                        "divergence_direct": terms.divergence_direct,
                        "divergence_predicted": terms.divergence_predicted,
                        "energetic_inhomogeneity": terms.energetic_inhomogeneity,
                        "body_force_source": terms.body_force_source,
                        "nonpotential_source": terms.nonpotential_source,
                        "residual": terms.residual,
                    }
                )

    Ce, Co, _, _ = constitutive_matrices(0.13, -0.21)
    eig_min = float(np.linalg.eigvalsh(Ce).min())
    result: dict[str, object] = {
        "identity": (
            "div(P_app)_K = 0.5 eps:C^e_{,K}:eps + b_i u_{i,K} "
            "- sigma^o_{ij} u_{i,jK}"
        ),
        "sign_convention": "div(sigma)+b=0",
        "max_local_balance_abs_residual": max_residual,
        "max_full_gradient_balance_abs_residual": max_full_gradient_residual,
        "full_gradient_identity": (
            "div(P_app)_K = 0.5 H:C^e_{,K}:H + b_i u_{i,K} "
            "- sigma^o_{ij} u_{i,jK}"
        ),
        "major_symmetric_even_error": float(np.max(np.abs(Ce - Ce.T))),
        "major_antisymmetric_odd_error": float(np.max(np.abs(Co + Co.T))),
        "minimum_even_modulus_eigenvalue": eig_min,
        "pass": bool(
            max_residual < 5.0e-14
            and max_full_gradient_residual < 5.0e-14
            and eig_min > 0.0
        ),
        "samples": rows,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
