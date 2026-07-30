#!/usr/bin/env python3
"""New figure: Bloch flutter boundary of the intact odd lattice (WP4).

Panel (a)  c_crit vs k_o: the exact bulk law against the manuscript's strip values.
Panel (b)  c_crit over the Brillouin zone: flutter is confined to the zone corners.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from bloch_stability import B1, B2, c_crit_at, dynamical_matrix  # noqa: E402

plt.rcParams.update({
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "lines.linewidth": 1.7,
    "lines.markersize": 5.5,
})


def _panel_label(ax, label, x=0.02, y=0.96, color="black"):
    ax.text(x, y, label, transform=ax.transAxes, va="top", ha="left",
            fontweight="bold", fontsize=13, color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))


def zone_max(k_o, n=181):
    fr = np.linspace(0.0, 1.0, n, endpoint=False)
    best = 0.0
    for x1 in fr:
        for x2 in fr:
            v, _ = c_crit_at(x1 * B1 + x2 * B2, 1.0, k_o)
            best = max(best, v)
    return best


def main(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    # ---- (a) exact bulk law vs the finite strip -----------------------
    ax = axes[0]
    ko_fine = np.linspace(0.0, 0.55, 60)
    ax.plot(ko_fine, 3.0 / math.sqrt(2.0) * ko_fine, "-", color="0.35",
            label=r"$c_{\rm crit}=\frac{3}{\sqrt{2}}\,k_o/\sqrt{k}$")

    ko_pts = [0.05, 0.12, 0.20, 0.30, 0.50]
    bloch = [zone_max(k) for k in ko_pts]
    ax.plot(ko_pts, bloch, "o", color="#1f77b4", label="Bloch, infinite lattice")

    strip = [0.067, 0.233, 0.411, 0.626, 1.053]
    ax.plot(ko_pts, strip, "s", mfc="none", color="#d62728",
            label=r"$32\times24$ strip (manuscript)")

    ax.set_xlabel(r"$k_o/k$")
    ax.set_ylabel(r"critical inertial damping $c_{\rm crit}$")
    ax.set_xlim(0.0, 0.55)
    ax.set_ylim(0.0, 1.20)
    ax.legend(loc="upper left", frameon=False)
    _panel_label(ax, "(a)")

    inset = ax.inset_axes([0.58, 0.14, 0.38, 0.32])
    inset.plot(ko_pts, [s / b for s, b in zip(strip, bloch)], "s-", mfc="none",
               color="#d62728", markersize=4, linewidth=1.2)
    inset.axhline(1.0, color="0.35", lw=1.0)
    inset.set_ylim(0.55, 1.06)
    inset.set_xlabel(r"$k_o/k$", fontsize=8)
    inset.set_ylabel("strip / bulk", fontsize=8)
    inset.tick_params(labelsize=7.5)

    # ---- (b) where in the zone does flutter live ---------------------
    ax = axes[1]
    span = 4.6
    grid = np.linspace(-span, span, 261)
    QX, QY = np.meshgrid(grid, grid)
    field = np.zeros_like(QX)
    for a in range(QX.shape[0]):
        for b in range(QX.shape[1]):
            field[a, b] = c_crit_at(np.array([QX[a, b], QY[a, b]]), 1.0, 0.20)[0]

    mesh = ax.pcolormesh(QX / (2 * math.pi), QY / (2 * math.pi), field,
                         cmap="magma", shading="auto", rasterized=True)
    bar = fig.colorbar(mesh, ax=ax, pad=0.02)
    bar.set_label(r"$c_{\rm crit}(\mathbf{q})$ at $k_o/k=0.20$", fontsize=9)
    bar.ax.tick_params(labelsize=8)

    # first Brillouin zone: hexagon through the K points
    K = 4.0 * math.pi / 3.0
    ang = np.arange(6) * math.pi / 3.0 + math.pi / 6.0
    hexx = np.append(K * np.cos(ang), K * np.cos(ang[0])) / (2 * math.pi)
    hexy = np.append(K * np.sin(ang), K * np.sin(ang[0])) / (2 * math.pi)
    ax.plot(hexx, hexy, "-", color="white", lw=1.3)

    ax.set_xlabel(r"$q_x/2\pi$")
    ax.set_ylabel(r"$q_y/2\pi$")
    ax.set_aspect("equal")
    ax.set_xlim(-span / (2 * math.pi), span / (2 * math.pi))
    ax.set_ylim(-span / (2 * math.pi), span / (2 * math.pi))
    _panel_label(ax, "(b)")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=400)
    print(f"wrote {out}")
    print(f"  bulk c_crit/k_o at the sampled points: {[round(b/k,4) for b,k in zip(bloch,ko_pts)]}")


if __name__ == "__main__":
    main(_ROOT / "figures" / "figNEW_bloch_flutter.pdf")
