#!/usr/bin/env python3
"""New figure: what the lattice cannot reach (WP6).

Panel (a)  Mode mixing G_12/G_11 over the (A_o, K_o) plane.  The discrete model
           is confined to the single ray A_o = 2 K_o; the continuum solver
           covers the plane and shows that mixing vanishes identically on the
           whole A_o = 0 axis, so K_o cannot create it, only modulate it.
Panel (b)  Independent spectrum check.  The smallest singular value of the
           face-to-face traction map, from an adaptive ODE integrator that
           shares no machinery with the closed-form propagator, collapses at
           every half-integer exponent both on and off the ray.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from continuum_independent import B_LAT, MU_LAT, ode_propagator  # noqa: E402
from crack_tip_asymptotics import J_matrix, OddModuli  # noqa: E402

plt.rcParams.update({
    "font.size": 10.5, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 8.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "lines.linewidth": 1.7, "lines.markersize": 5.5,
})

SQRT3 = math.sqrt(3.0)
GRID = 27
SPAN = 0.45


def _panel_label(ax, label, x=0.02, y=0.96):
    ax.text(x, y, label, transform=ax.transAxes, va="top", ha="left",
            fontweight="bold", fontsize=13,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))


def sigma_min(lam: float, moduli: OddModuli) -> float:
    P = ode_propagator(2.0 * math.pi, lam, moduli)
    return float(np.linalg.svd(P[2:, :2], compute_uv=False).min())


def build(cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())

    axis = np.linspace(-SPAN, SPAN, GRID)
    mixing = np.zeros((GRID, GRID))
    for i, K_o in enumerate(axis):
        for j, A_o in enumerate(axis):
            G = J_matrix(OddModuli(B=B_LAT, mu=MU_LAT, A_o=float(A_o), K_o=float(K_o)),
                         n_theta=401)
            mixing[i, j] = G[0, 1] / G[0, 0]

    lam = np.linspace(0.15, 2.85, 181)
    spectra = {}
    for name, (A_o, K_o) in {
        "on ray  $A_o=2K_o=-0.30$": (-0.30, -0.15),
        "off ray  $K_o=0$": (-0.30, 0.0),
        "off ray  $A_o=0$": (0.0, -0.30),
    }.items():
        m = OddModuli(B=B_LAT, mu=MU_LAT, A_o=A_o, K_o=K_o)
        spectra[name] = [sigma_min(float(x), m) for x in lam]

    data = {"axis": axis.tolist(), "mixing": mixing.tolist(),
            "lam": lam.tolist(), "spectra": spectra}
    cache.write_text(json.dumps(data))
    return data


def main(out: Path) -> None:
    data = build(_ROOT / "figure_data" / (out.stem + ".json"))
    axis = np.array(data["axis"])
    mixing = np.array(data["mixing"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))

    # ---- (a) the (A_o, K_o) plane ------------------------------------
    ax = axes[0]
    limit = float(np.max(np.abs(mixing)))
    levels = np.linspace(-limit, limit, 21)
    filled = ax.contourf(axis, axis, mixing, levels=levels, cmap="RdBu_r", extend="both")
    bar = fig.colorbar(filled, ax=ax, pad=0.02, ticks=[-0.15, -0.075, 0, 0.075, 0.15])
    bar.set_label(r"$G_{12}/G_{11}$", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    ax.contour(axis, axis, mixing, levels=[0.0], colors="k", linewidths=1.0)

    ray = np.linspace(-SPAN / 2, SPAN / 2, 40)
    ax.plot(2 * ray, ray, "-", color="black", lw=2.2)
    ax.plot(2 * ray, ray, "-", color="#39ff14", lw=1.3, label=r"lattice ray $A_o=2K_o$")
    for k_o in (0.10, 0.20, 0.40):
        ax.plot(-SQRT3 / 2 * k_o, -SQRT3 / 4 * k_o, "o", color="black",
                markersize=7, zorder=6)
        ax.plot(-SQRT3 / 2 * k_o, -SQRT3 / 4 * k_o, "o", color="#39ff14",
                markersize=4.2, zorder=7)
    ax.text(-0.40, -0.30, r"$k_o=0.4$", fontsize=7.5, color="black")

    ax.set_xlabel(r"$A_o$")
    ax.set_ylabel(r"$K_o$")
    ax.set_xlim(-SPAN, SPAN)
    ax.set_ylim(-SPAN, SPAN)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", frameon=True, framealpha=0.85, fontsize=7.5)
    _panel_label(ax, "(a)", x=0.02, y=0.86)

    # ---- (b) independent spectrum ------------------------------------
    ax = axes[1]
    lam = np.array(data["lam"])
    for (name, vals), colour in zip(data["spectra"].items(),
                                    ("#1f77b4", "#d62728", "#2ca02c")):
        ax.semilogy(lam, np.maximum(vals, 1e-16), "-", color=colour, label=name)
    for half in (0.5, 1.0, 1.5, 2.0, 2.5):
        ax.axvline(half, color="0.8", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel(r"Williams exponent $\lambda$")
    ax.set_ylabel(r"$\sigma_{\min}$ of face-to-face traction map")
    ax.set_ylim(1e-16, 5.0)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    _panel_label(ax, "(b)")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    zero_col = int(np.argmin(np.abs(axis)))
    print(f"  max |G12/G11| on the A_o=0 axis: {np.max(np.abs(mixing[:, zero_col])):.3e}")


if __name__ == "__main__":
    main(_ROOT / "figures" / "figNEW_modulus_plane.pdf")
