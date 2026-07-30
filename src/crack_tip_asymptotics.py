#!/usr/bin/env python3
"""Crack-tip eigenfields for the 2D isotropic odd continuum used by the lattice.

The generalized isotropic law is written in the local orthonormal basis as

    sigma_0 = B e_0,
    sigma_1 = A_o e_0,
    sigma_2 = mu e_2 - K_o e_3,
    sigma_3 = K_o e_2 + mu e_3,

with e_0=H_11+H_22, e_1=H_21-H_12, e_2=H_11-H_22 and
e_3=H_12+H_21.  The dual stress convention is

    sigma_0=(sigma_11+sigma_22)/2,
    sigma_1=(sigma_21-sigma_12)/2,
    sigma_2=(sigma_11-sigma_22)/2,
    sigma_3=(sigma_12+sigma_21)/2,

so that sigma:H=sum_alpha sigma_alpha e_alpha (without an extra factor 1/2).
It is the continuum law already used in the manuscript.  The
A_o channel permits antisymmetric stress and therefore represents the
substrate-supported/generalized-distortion extension of the symmetric-stress
Cauchy theory.

For u=r^lambda U(theta), a four-dimensional polar state system is propagated
between the traction-free crack faces theta=-pi and theta=+pi.  The state matrix
has eigenvalues +/-i(lambda-1) and +/-i(lambda+1), independent of A_o and K_o
when the constitutive operator is nondegenerate.  Hence the crack spectrum is
lambda=n/2 and the leading non-rigid singular exponent is lambda=1/2.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import numpy as np
from scipy.integrate import simpson
from scipy.linalg import expm

Array = np.ndarray
PI2 = 2.0 * math.pi


@dataclass(frozen=True)
class OddModuli:
    B: float = 1.0
    mu: float = 0.5
    A_o: float = 0.0
    K_o: float = 0.0

    @property
    def pde_determinant(self) -> float:
        return self.mu * (self.B + self.mu) + self.K_o * (self.K_o + self.A_o)

    @property
    def strong_ellipticity_margin(self) -> float:
        # Minimum determinant condition for the symmetric part of the acoustic
        # matrix; K_o is skew in the longitudinal/transverse acoustic basis.
        return self.mu * (self.B + self.mu) - 0.25 * self.A_o * self.A_o


def irreducible_stress_to_tensor(s0: float, s1: float, s2: float, s3: float) -> Array:
    """Invert the manuscript's irreducible stress convention.

    The paired variables satisfy sigma:H = sum_alpha sigma_alpha e_alpha.
    Consequently sigma_1 is half the antisymmetric stress difference.
    """

    return np.array([[s0 + s2, s3 - s1], [s3 + s1, s0 - s2]])


def tensor_to_irreducible_stress(sigma: Array) -> Array:
    """Return (sigma_0,sigma_1,sigma_2,sigma_3) from a 2x2 stress tensor."""

    return 0.5 * np.array(
        [
            sigma[0, 0] + sigma[1, 1],
            sigma[1, 0] - sigma[0, 1],
            sigma[0, 0] - sigma[1, 1],
            sigma[0, 1] + sigma[1, 0],
        ]
    )


def stiffness_matrix(moduli: OddModuli) -> Array:
    """Map vec(H)=[H11,H12,H21,H22] to vec(sigma) in the same order."""

    B, mu, A_o, K_o = moduli.B, moduli.mu, moduli.A_o, moduli.K_o
    matrix = np.zeros((4, 4))
    slots = ((0, 0), (0, 1), (1, 0), (1, 1))
    for column, (i, j) in enumerate(slots):
        H = np.zeros((2, 2))
        H[i, j] = 1.0
        e0 = H[0, 0] + H[1, 1]
        e1 = H[1, 0] - H[0, 1]
        e2 = H[0, 0] - H[1, 1]
        e3 = H[0, 1] + H[1, 0]
        del e1  # objectivity removes the reciprocal rotational column
        s0 = B * e0
        s1 = A_o * e0
        s2 = mu * e2 - K_o * e3
        s3 = K_o * e2 + mu * e3
        sigma = irreducible_stress_to_tensor(s0, s1, s2, s3)
        matrix[:, column] = np.array(
            [sigma[0, 0], sigma[0, 1], sigma[1, 0], sigma[1, 1]]
        )
    return matrix


def energetic_stiffness_matrix(
    moduli: OddModuli, choice: str = "micro_hessian"
) -> Array:
    """Return the energetic modulus used to define the recoverable energy.

    ``micro_hessian`` is the Hessian of the conservative central-spring energy
    and is the default throughout the lattice calculations.  It is objective
    and contains only B and mu.  ``major_symmetric_projection`` instead takes
    the Euclidean major-symmetric part (C+C^T)/2 of the full-gradient
    constitutive matrix.  The latter is a mathematically admissible split but
    contains the non-objective cross term (A_o/2)e_0 e_1.
    """

    if choice == "micro_hessian":
        return stiffness_matrix(OddModuli(moduli.B, moduli.mu, 0.0, 0.0))
    if choice == "major_symmetric_projection":
        total = stiffness_matrix(moduli)
        return 0.5 * (total + total.T)
    raise ValueError(f"Unknown energetic split: {choice}")


def state_matrix(lam: float, moduli: OddModuli) -> Array:
    """Return y'=A y for y=(U_r,U_theta,sigma_rtheta,sigma_thetatheta)."""

    C = stiffness_matrix(moduli)
    # Hhat = Ld U' + Lu U for u=r^lambda(U_r e_r+U_theta e_theta).
    Ld = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    Lu = np.array(
        [[lam, 0.0], [0.0, -1.0], [0.0, lam], [1.0, 0.0]]
    )
    traction_rows = [1, 3]
    other_rows = [0, 2]
    D_t = C[traction_rows] @ Ld
    B_t = C[traction_rows] @ Lu
    if abs(np.linalg.det(D_t)) < 1.0e-13:
        raise ValueError("Polar traction-to-gradient block is singular")

    inv_Dt = np.linalg.inv(D_t)
    A_uu = -inv_Dt @ B_t
    A_ut = inv_Dt
    S_u = C[other_rows] @ (Ld @ A_uu + Lu)
    S_t = C[other_rows] @ (Ld @ A_ut)

    # Equilibrium for a generally nonsymmetric stress:
    # (sigma_rtheta)' = sigma_thetatheta - lambda sigma_rr
    # (sigma_thetatheta)' = -lambda sigma_thetar - sigma_rtheta.
    A_tu = np.vstack((-lam * S_u[0], -lam * S_u[1]))
    A_tt = np.zeros((2, 2))
    A_tt[0] = -lam * S_t[0]
    A_tt[0, 1] += 1.0
    A_tt[1] = -lam * S_t[1]
    A_tt[1, 0] -= 1.0
    return np.block([[A_uu, A_ut], [A_tu, A_tt]])


def closed_form_propagator(
    delta_theta: float, lam: float, moduli: OddModuli
) -> Array:
    """Exact angular propagator using the two Williams harmonic sectors.

    The state matrix satisfies

        (A^2 + (lambda-1)^2 I)(A^2 + (lambda+1)^2 I)=0.

    Therefore exp(A*delta_theta) is a finite matrix polynomial multiplying only
    the two frequencies |lambda-1| and lambda+1.
    """

    if abs(lam) < 1.0e-14:
        raise ValueError("lambda=0 is a degenerate rigid-motion sector")
    matrix = state_matrix(lam, moduli)
    identity = np.eye(4)
    omega_minus = abs(lam - 1.0)
    omega_plus = abs(lam + 1.0)
    denominator = omega_plus**2 - omega_minus**2
    projector_minus = (matrix @ matrix + omega_plus**2 * identity) / denominator
    projector_plus = -(matrix @ matrix + omega_minus**2 * identity) / denominator

    def sector(projector: Array, omega: float) -> Array:
        if omega < 1.0e-13:
            sine_factor = float(delta_theta)
            cosine = 1.0
        else:
            sine_factor = math.sin(omega * delta_theta) / omega
            cosine = math.cos(omega * delta_theta)
        return projector @ (cosine * identity + sine_factor * matrix)

    return sector(projector_minus, omega_minus) + sector(projector_plus, omega_plus)


def harmonic_state_coefficients(
    K_I: float, K_II: float, moduli: OddModuli, lam: float = 0.5
) -> dict[str, Array | float]:
    """Return exact polar-state coefficients for the two angular frequencies.

    With phi=theta+pi,

        y(phi)=c_- cos(omega_- phi)+s_- sin(omega_- phi)
              +c_+ cos(omega_+ phi)+s_+ sin(omega_+ phi).
    """

    if abs(lam) < 1.0e-14:
        raise ValueError("lambda=0 is a degenerate rigid-motion sector")
    matrix = state_matrix(lam, moduli)
    identity = np.eye(4)
    omega_minus = abs(lam - 1.0)
    omega_plus = abs(lam + 1.0)
    denominator = omega_plus**2 - omega_minus**2
    projector_minus = (matrix @ matrix + omega_plus**2 * identity) / denominator
    projector_plus = -(matrix @ matrix + omega_minus**2 * identity) / denominator
    y0 = initial_state_for_K(K_I, K_II, moduli, lam)

    def sine_coefficient(projector: Array, omega: float) -> Array:
        if omega < 1.0e-13:
            return projector @ matrix @ y0
        return projector @ matrix @ y0 / omega

    return {
        "omega_minus": omega_minus,
        "omega_plus": omega_plus,
        "cos_minus": projector_minus @ y0,
        "sin_minus": sine_coefficient(projector_minus, omega_minus),
        "cos_plus": projector_plus @ y0,
        "sin_plus": sine_coefficient(projector_plus, omega_plus),
    }


def analytic_first_order_J_derivatives(B: float, mu: float) -> tuple[Array, Array]:
    """Closed first-order derivative of the K-normalized apparent tip flux.

    Harmonic averaging gives dJ/dA_o=((B+mu)/(4 B^2 mu)) K_I K_II
    and dJ/dK_o=0.  Since J=K^T G K, the off-diagonal G coefficient is
    half the coefficient multiplying K_I K_II.
    """

    coefficient = (B + mu) / (8.0 * B * B * mu)
    return np.array([[0.0, coefficient], [coefficient, 0.0]]), np.zeros((2, 2))


def expected_state_eigenvalues(lam: float) -> Array:
    return np.array(
        [1j * (lam + 1.0), -1j * (lam + 1.0), 1j * (lam - 1.0), -1j * (lam - 1.0)]
    )


def eigenvalue_set_error(lam: float, moduli: OddModuli) -> float:
    actual = list(np.linalg.eigvals(state_matrix(lam, moduli)))
    expected = list(expected_state_eigenvalues(lam))
    error = 0.0
    for target in expected:
        index = int(np.argmin([abs(value - target) for value in actual]))
        error = max(error, abs(actual.pop(index) - target))
    return float(error)


def traction_face_map(lam: float, moduli: OddModuli) -> Array:
    """Map displacement data at -pi (zero traction) to traction at +pi."""

    transfer = closed_form_propagator(PI2, lam, moduli)
    return transfer[2:, :2]


def characteristic_measure(lam: float, moduli: OddModuli) -> tuple[float, float, float]:
    singular_values = np.linalg.svd(traction_face_map(lam, moduli), compute_uv=False)
    return (
        float(abs(np.linalg.det(traction_face_map(lam, moduli)))),
        float(singular_values.min()),
        float(singular_values.max()),
    )


def _kinematic_map(lam: float, moduli: OddModuli) -> tuple[Array, Array]:
    A = state_matrix(lam, moduli)
    Ld = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    Lu = np.array(
        [[lam, 0.0], [0.0, -1.0], [0.0, lam], [1.0, 0.0]]
    )
    select_u = np.zeros((2, 4))
    select_u[:, :2] = np.eye(2)
    F = Ld @ A[:2] + Lu @ select_u
    return A, F


def initial_state_for_K(K_I: float, K_II: float, moduli: OddModuli, lam: float = 0.5) -> Array:
    """Choose a traction-free-face mode normalized by ahead-of-tip K values."""

    A = state_matrix(lam, moduli)
    to_ahead = closed_form_propagator(math.pi, lam, moduli)
    traction_map = to_ahead[2:, :2]
    # At theta=0, state traction=(sigma_xy,sigma_yy) coefficient.
    target = np.array([K_II, K_I]) / math.sqrt(PI2)
    initial_u = np.linalg.solve(traction_map, target)
    return np.concatenate((initial_u, np.zeros(2)))


def _vec_to_matrix(vector: Array) -> Array:
    return np.array([[vector[0], vector[1]], [vector[2], vector[3]]])


def _matrix_to_vec(matrix: Array) -> Array:
    return np.array([matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]])


def sample_K_field(
    K_I: float,
    K_II: float,
    moduli: OddModuli,
    theta: Array,
    lam: float = 0.5,
    energy_choice: str = "micro_hessian",
) -> dict[str, Array]:
    """Sample displacement, stress, J-flux density, and odd source at r=1."""

    A, F = _kinematic_map(lam, moduli)
    C = stiffness_matrix(moduli)
    C_even = energetic_stiffness_matrix(moduli, energy_choice)
    C_np = C - C_even
    y0 = initial_state_for_K(K_I, K_II, moduli, lam)

    displacement_cart = np.zeros((len(theta), 2))
    stress_cart = np.zeros((len(theta), 2, 2))
    strain_grad_x = np.zeros((len(theta), 2, 2))
    source_hat = np.zeros(len(theta))
    j_density = np.zeros(len(theta))
    face_traction = np.zeros((len(theta), 2))

    generator = np.array([[0.0, -1.0], [1.0, 0.0]])
    for index, angle in enumerate(theta):
        y = closed_form_propagator(float(angle) + math.pi, lam, moduli) @ y0
        h_local = _vec_to_matrix(F @ y)
        h_local_prime = _vec_to_matrix(F @ A @ y)
        s_local = _vec_to_matrix(C @ _matrix_to_vec(h_local))

        cosine, sine = math.cos(float(angle)), math.sin(float(angle))
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        rotation_prime = rotation @ generator
        h_cart = rotation @ h_local @ rotation.T
        h_cart_prime = (
            rotation_prime @ h_local @ rotation.T
            + rotation @ h_local_prime @ rotation.T
            + rotation @ h_local @ rotation_prime.T
        )
        s_cart = rotation @ s_local @ rotation.T
        dH_dx = (lam - 1.0) * cosine * h_cart - sine * h_cart_prime
        sigma_np = _vec_to_matrix(C_np @ _matrix_to_vec(h_cart))

        h_vector = _matrix_to_vec(h_cart)
        W_even = 0.5 * float(h_vector @ (C_even @ h_vector))
        normal = np.array([cosine, sine])
        j_density[index] = W_even * cosine - h_cart[:, 0] @ (s_cart @ normal)
        source_hat[index] = -float(np.einsum("ij,ij->", sigma_np, dH_dx))

        displacement_cart[index] = rotation @ y[:2]
        stress_cart[index] = s_cart
        strain_grad_x[index] = dH_dx
        face_traction[index] = y[2:]

    return {
        "theta": theta,
        "u": displacement_cart,
        "sigma": stress_cart,
        "dH_dx": strain_grad_x,
        "source_hat": source_hat,
        "j_density": j_density,
        "polar_traction": face_traction,
    }


def J_matrix(
    moduli: OddModuli,
    n_theta: int = 1601,
    energy_choice: str = "micro_hessian",
) -> Array:
    """Return G such that J_tip=[K_I,K_II] G [K_I,K_II]^T."""

    theta = np.linspace(-math.pi, math.pi, n_theta)

    def J(KI: float, KII: float) -> float:
        field = sample_K_field(KI, KII, moduli, theta, energy_choice=energy_choice)
        return float(simpson(field["j_density"], x=theta))

    J11 = J(1.0, 0.0)
    J22 = J(0.0, 1.0)
    J12 = 0.5 * (J(1.0, 1.0) - J11 - J22)
    return np.array([[J11, J12], [J12, J22]])


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    passive = OddModuli()
    representative = OddModuli(A_o=-0.10, K_o=-0.05)

    # Parameter-independent state spectrum and lambda=1/2 face matching.
    spectrum_test_moduli = (
        passive,
        OddModuli(A_o=0.15),
        OddModuli(K_o=0.15),
        OddModuli(A_o=0.20, K_o=0.10),
        OddModuli(A_o=-0.20, K_o=-0.10),
    )
    max_state_spectrum_error = max(
        eigenvalue_set_error(lam, moduli)
        for moduli in spectrum_test_moduli
        for lam in (0.37, 0.50, 0.83, 1.20)
    )
    max_lambda_half_face_residual = max(
        characteristic_measure(0.5, moduli)[2] for moduli in spectrum_test_moduli
    )
    max_closed_form_propagator_error = max(
        float(
            np.max(
                np.abs(
                    closed_form_propagator(angle, lam, moduli)
                    - expm(state_matrix(lam, moduli) * angle)
                )
            )
        )
        for moduli in spectrum_test_moduli
        for lam in (0.37, 0.50, 0.83, 1.20)
        for angle in (-2.1, -0.4, 0.7, 2.8)
    )

    # Characteristic scan and exact half-integer candidates.
    scan_rows: list[dict[str, float]] = []
    for lam in np.linspace(0.10, 2.00, 191):
        det_abs, sv_min, sv_max = characteristic_measure(float(lam), representative)
        scan_rows.append(
            {
                "lambda": float(lam),
                "det_abs": det_abs,
                "singular_value_min": sv_min,
                "singular_value_max": sv_max,
            }
        )
    _write_csv(output_dir / "eigenvalue_scan.csv", scan_rows)

    candidates = [0.5, 1.0, 1.5, 2.0]
    candidate_rows = []
    for lam in candidates:
        det_abs, sv_min, sv_max = characteristic_measure(lam, representative)
        candidate_rows.append(
            {
                "lambda": lam,
                "stress_power": lam - 1.0,
                "det_abs": det_abs,
                "singular_value_max": sv_max,
                "state_eigenvalue_set_error": eigenvalue_set_error(lam, representative),
            }
        )
    _write_csv(output_dir / "half_integer_spectrum.csv", candidate_rows)

    # Exact half-angle/three-half-angle polar-state coefficients.
    harmonic_rows: list[dict[str, float | str]] = []
    state_names = ("U_r", "U_theta", "sigma_rtheta", "sigma_thetatheta")
    for mode_name, KI, KII in (("I", 1.0, 0.0), ("II", 0.0, 1.0)):
        coefficients = harmonic_state_coefficients(KI, KII, representative, 0.5)
        for sector_name, omega_key, cos_key, sin_key in (
            ("minus", "omega_minus", "cos_minus", "sin_minus"),
            ("plus", "omega_plus", "cos_plus", "sin_plus"),
        ):
            for component, state_name in enumerate(state_names):
                harmonic_rows.append(
                    {
                        "mode": mode_name,
                        "sector": sector_name,
                        "omega": float(coefficients[omega_key]),
                        "state_component": state_name,
                        "cos_coefficient": float(coefficients[cos_key][component]),
                        "sin_coefficient": float(coefficients[sin_key][component]),
                    }
                )
    _write_csv(output_dir / "closed_form_state_harmonics.csv", harmonic_rows)

    # J(K) on independent odd-modulus directions and on the lattice ray A_o=2K_o.
    modulus_rows: list[dict[str, float | str]] = []
    for family in ("A_only", "K_only", "lattice_ray"):
        for epsilon in (-0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10):
            if family == "A_only":
                moduli = OddModuli(A_o=epsilon, K_o=0.0)
            elif family == "K_only":
                moduli = OddModuli(A_o=0.0, K_o=epsilon)
            else:
                moduli = OddModuli(A_o=epsilon, K_o=0.5 * epsilon)
            G = J_matrix(moduli, n_theta=1001)
            modulus_rows.append(
                {
                    "family": family,
                    "epsilon": epsilon,
                    "A_o": moduli.A_o,
                    "K_o": moduli.K_o,
                    "J_KI_KI": float(G[0, 0]),
                    "J_KI_KII": float(G[0, 1]),
                    "J_KII_KII": float(G[1, 1]),
                    "pde_determinant": moduli.pde_determinant,
                    "strong_ellipticity_margin": moduli.strong_ellipticity_margin,
                }
            )
    _write_csv(output_dir / "small_odd_modulus_scan.csv", modulus_rows)

    # First derivatives at the passive point.
    h = 2.0e-4
    dG_dA = (J_matrix(OddModuli(A_o=h), 1201) - J_matrix(OddModuli(A_o=-h), 1201)) / (2.0 * h)
    dG_dK = (J_matrix(OddModuli(K_o=h), 1201) - J_matrix(OddModuli(K_o=-h), 1201)) / (2.0 * h)
    analytic_dG_dA, analytic_dG_dK = analytic_first_order_J_derivatives(
        passive.B, passive.mu
    )
    analytic_derivative_error = max(
        float(np.max(np.abs(dG_dA - analytic_dG_dA))),
        float(np.max(np.abs(dG_dK - analytic_dG_dK))),
    )

    derivative_rows: list[dict[str, float | str]] = []
    for B_test, mu_test in ((0.7, 0.3), (1.0, 0.5), (1.4, 0.6), (2.0, 0.8)):
        numerical_A = (
            J_matrix(OddModuli(B_test, mu_test, h, 0.0), 1001)
            - J_matrix(OddModuli(B_test, mu_test, -h, 0.0), 1001)
        ) / (2.0 * h)
        numerical_K = (
            J_matrix(OddModuli(B_test, mu_test, 0.0, h), 1001)
            - J_matrix(OddModuli(B_test, mu_test, 0.0, -h), 1001)
        ) / (2.0 * h)
        exact_A, exact_K = analytic_first_order_J_derivatives(B_test, mu_test)
        derivative_rows.append(
            {
                "B": B_test,
                "mu": mu_test,
                "analytic_dG12_dA": float(exact_A[0, 1]),
                "numeric_dG12_dA": float(numerical_A[0, 1]),
                "abs_error_dA": float(np.max(np.abs(numerical_A - exact_A))),
                "numeric_max_abs_dG_dK": float(np.max(np.abs(numerical_K))),
                "abs_error_dK": float(np.max(np.abs(numerical_K - exact_K))),
            }
        )
    _write_csv(output_dir / "analytic_first_order_J_certificate.csv", derivative_rows)

    # Exact finite-K_o stress invariance in the symmetric-stress A_o=0 sector.
    invariance_theta = np.linspace(-math.pi, math.pi, 601)
    max_K_only_stress_invariance_error = max(
        float(
            np.max(
                np.abs(
                    sample_K_field(KI, KII, OddModuli(K_o=K_test), invariance_theta)["sigma"]
                    - sample_K_field(KI, KII, passive, invariance_theta)["sigma"]
                )
            )
        )
        for K_test in (-0.7, -0.3, 0.1, 0.4, 0.7)
        for KI, KII in ((1.0, 0.0), (0.0, 1.0))
    )

    # First-order field sensitivity.  This separates the two odd moduli: K_o
    # changes the K-normalized displacement field while leaving the stress field
    # unchanged when A_o=0; A_o mixes the stress modes.
    sensitivity_theta = np.linspace(-math.pi, math.pi, 1001)
    field_sensitivity: dict[str, dict[str, dict[str, float]]] = {}
    correction_rows: list[dict[str, float | str]] = []
    for mode_name, KI, KII in (("I", 1.0, 0.0), ("II", 0.0, 1.0)):
        field_sensitivity[mode_name] = {}
        for parameter in ("A_o", "K_o"):
            plus = sample_K_field(
                KI, KII,
                OddModuli(A_o=h if parameter == "A_o" else 0.0, K_o=h if parameter == "K_o" else 0.0),
                sensitivity_theta,
            )
            minus = sample_K_field(
                KI, KII,
                OddModuli(A_o=-h if parameter == "A_o" else 0.0, K_o=-h if parameter == "K_o" else 0.0),
                sensitivity_theta,
            )
            du = (plus["u"] - minus["u"]) / (2.0 * h)
            ds = (plus["sigma"] - minus["sigma"]) / (2.0 * h)
            field_sensitivity[mode_name][parameter] = {
                "displacement_L2_derivative": float(
                    math.sqrt(simpson(np.sum(du * du, axis=1), x=sensitivity_theta))
                ),
                "stress_L2_derivative": float(
                    math.sqrt(simpson(np.sum(ds * ds, axis=(1, 2)), x=sensitivity_theta))
                ),
            }
            for index, angle in enumerate(sensitivity_theta):
                correction_rows.append(
                    {
                        "mode": mode_name,
                        "parameter": parameter,
                        "theta": float(angle),
                        "du_x_dparameter": float(du[index, 0]),
                        "du_y_dparameter": float(du[index, 1]),
                        "dsigma_xx_dparameter": float(ds[index, 0, 0]),
                        "dsigma_xy_dparameter": float(ds[index, 0, 1]),
                        "dsigma_yx_dparameter": float(ds[index, 1, 0]),
                        "dsigma_yy_dparameter": float(ds[index, 1, 1]),
                    }
                )
    _write_csv(output_dir / "first_order_angular_corrections.csv", correction_rows)

    # Angular fields and source structure for the representative lattice ray.
    theta = np.linspace(-math.pi, math.pi, 1441)
    angular_rows: list[dict[str, float | str]] = []
    source_summary: dict[str, dict[str, float]] = {}
    for mode_name, KI, KII in (("I", 1.0, 0.0), ("II", 0.0, 1.0)):
        field = sample_K_field(KI, KII, representative, theta)
        passive_field = sample_K_field(KI, KII, passive, theta)
        signed = float(simpson(field["source_hat"], x=theta))
        absolute = float(simpson(np.abs(field["source_hat"]), x=theta))
        source_summary[mode_name] = {
            "signed_angular_integral": signed,
            "absolute_angular_integral": absolute,
            "relative_signed_to_absolute": abs(signed) / max(absolute, 1.0e-30),
        }
        for index, angle in enumerate(theta):
            angular_rows.append(
                {
                    "mode": mode_name,
                    "theta": float(angle),
                    "u_x": float(field["u"][index, 0]),
                    "u_y": float(field["u"][index, 1]),
                    "sigma_xx": float(field["sigma"][index, 0, 0]),
                    "sigma_xy": float(field["sigma"][index, 0, 1]),
                    "sigma_yx": float(field["sigma"][index, 1, 0]),
                    "sigma_yy": float(field["sigma"][index, 1, 1]),
                    "source_hat": float(field["source_hat"][index]),
                    "delta_u_x_from_passive": float(field["u"][index, 0] - passive_field["u"][index, 0]),
                    "delta_u_y_from_passive": float(field["u"][index, 1] - passive_field["u"][index, 1]),
                }
            )
    _write_csv(output_dir / "representative_angular_fields.csv", angular_rows)

    passive_G = J_matrix(passive, 1601)
    representative_G = J_matrix(representative, 1601)
    expected_passive = 1.0 / (4.0 * passive.B * passive.mu / (passive.B + passive.mu))
    face_residual = characteristic_measure(0.5, representative)[2]
    summary: dict[str, object] = {
        "constitutive_model": "2D isotropic substrate-supported odd distortion law",
        "irreducible_pairing": "sigma:H = sum_alpha sigma_alpha e_alpha",
        "irreducible_stress_inverse": [
            "sigma_xx=sigma_0+sigma_2",
            "sigma_yy=sigma_0-sigma_2",
            "sigma_xy=sigma_3-sigma_1",
            "sigma_yx=sigma_3+sigma_1",
        ],
        "pde_reduction": [
            "(B+mu) Laplacian(div u) - K_o Laplacian(curl u) = 0",
            "(K_o+A_o) Laplacian(div u) + mu Laplacian(curl u) = 0",
        ],
        "lattice_ray_pde_determinant": "9/16*(k^2+k_o^2)>0 for k>0",
        "pde_determinant": representative.pde_determinant,
        "state_eigenvalues": "+/- i(lambda-1), +/- i(lambda+1)",
        "max_state_eigenvalue_set_error_over_parameter_tests": max_state_spectrum_error,
        "max_lambda_half_face_residual_over_parameter_tests": max_lambda_half_face_residual,
        "max_closed_form_propagator_error_vs_expm": max_closed_form_propagator_error,
        "closed_form_lambda_half_harmonics": ["1/2", "3/2"],
        "admissible_crack_spectrum": "lambda=n/2 after traction-free face matching",
        "leading_nonrigid_singular_exponent": 0.5,
        "leading_stress_power": -0.5,
        "passive_J_matrix": passive_G.tolist(),
        "passive_expected_1_over_E2D": expected_passive,
        "passive_J_matrix_max_error": float(np.max(np.abs(passive_G - expected_passive * np.eye(2)))),
        "representative_moduli": representative.__dict__,
        "representative_J_matrix": representative_G.tolist(),
        "dJ_matrix_dA_at_zero": dG_dA.tolist(),
        "dJ_matrix_dK_at_zero": dG_dK.tolist(),
        "analytic_dJ_matrix_dA_at_zero": analytic_dG_dA.tolist(),
        "analytic_dJ_matrix_dK_at_zero": analytic_dG_dK.tolist(),
        "analytic_first_order_derivative_max_error": analytic_derivative_error,
        "analytic_first_order_cross_coefficient": "(B+mu)/(8 B^2 mu)",
        "max_finite_K_only_stress_invariance_error": max_K_only_stress_invariance_error,
        "traction_face_map_determinant_factor": "sin(2*pi*lambda)^2 * 4*lambda^2*(A_o^2+B^2)*(K_o^2+mu^2)^2/Delta_PDE^2",
        "first_order_field_sensitivity": field_sensitivity,
        "source_pointwise_radial_order": "r^-2 for the pure lambda=1/2 field",
        "source_integrability": (
            "not absolutely integrable term-by-term; the complete-annulus signed "
            "coefficient cancels for a pure K eigenfield"
        ),
        "source_angular_integrals": source_summary,
        "lambda_half_face_map_max_singular_value": face_residual,
        "pass": bool(
            representative.pde_determinant > 0.0
            and representative.strong_ellipticity_margin > 0.0
            and face_residual < 1.0e-11
            and max_state_spectrum_error < 1.0e-12
            and max_lambda_half_face_residual < 1.0e-11
            and max_closed_form_propagator_error < 2.0e-12
            and analytic_derivative_error < 2.0e-7
            and max_K_only_stress_invariance_error < 2.0e-12
            and np.max(np.abs(passive_G - expected_passive * np.eye(2))) < 2.0e-10
            and max(v["relative_signed_to_absolute"] for v in source_summary.values()) < 2.0e-10
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
