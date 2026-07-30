#!/usr/bin/env python3
"""New figure: the continuum-lattice bridge (WP2).

Panel (a)  alpha_J vs system size.  Holding the reconstruction radius at the
           lattice scale gives O(1/N_x) convergence to unity; scaling it with
           the domain, as the archived refinement does, freezes the bias.
Panel (b)  K_II/K_I vs size at several k_o, against the passive noise floor
           (which must be exactly zero, so it measures the systematic error).
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


from apparent_j_analysis import keyhole_j, solve_field  # noqa: E402
from crack_tip_asymptotics import J_matrix  # noqa: E402
import crack_tip_lattice_fit as ctlf  # noqa: E402
from wp2_closeout import fit_case  # noqa: E402  (already patches PeriodicTipField)

plt.rcParams.update({
    "font.size": 10.5, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 8.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "lines.linewidth": 1.7, "lines.markersize": 5.5,
})

SIZES = (48, 64, 96, 128, 160)
WIDE = (0.035, 0.22)   # recommended fitting window, radial span 6.3
G_PASSIVE = J_matrix(ctlf.homogenized_moduli(1.0, 0.0), n_theta=801)


def _panel_label(ax, label, x=0.02, y=0.96):
    ax.text(x, y, label, transform=ax.transAxes, va="top", ha="left",
            fontweight="bold", fontsize=13,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))


def alpha_series(lams, scaled_fit):
    out = []
    for nx in SIZES:
        fit = max(2.0, 0.05 * nx) if scaled_fit else 3.0
        _m, f, _r = solve_field(nx, 3 * nx // 4, nx / 4.0, 0.0, "right", fit)
        J = float(np.mean([keyhole_j(f, float(R), 0.02)
                           for R in (0.06 * nx, 0.09 * nx, 0.12 * nx)]))
        KI, KII, _res = fit_case(nx, 0.0, "right", lams, fit_radius=fit)
        K = np.array([KI, KII])
        out.append(J / float(K @ G_PASSIVE @ K))
    return out


def build_data(cache: Path) -> dict:
    import json
    if cache.exists():
        return json.loads(cache.read_text())
    lams3, lams5 = (0.5, 1.5), (0.5, 1.5, 2.5)
    data = {
        "alpha_scaled": alpha_series(lams3, True),
        "alpha_fixed_32": alpha_series(lams3, False),
        "alpha_fixed_52": alpha_series(lams5, False),
        "floor": [abs(fit_case(n, 0.0, "right", lams5, window=WIDE, n_radii=13)[1]
                      / fit_case(n, 0.0, "right", lams5, window=WIDE, n_radii=13)[0])
                  for n in SIZES],
        "kii": {},
    }
    for k_o in (0.10, 0.20, 0.40):
        vals = []
        for nx in SIZES:
            KIp, KIIp, _ = fit_case(nx, k_o, "right", lams5, window=WIDE, n_radii=13)
            KIm, KIIm, _ = fit_case(nx, -k_o, "right", lams5, window=WIDE, n_radii=13)
            vals.append(abs(0.5 * (KIIp / KIp - KIIm / KIm)))
        data["kii"][str(k_o)] = vals
    cache.write_text(json.dumps(data, indent=1))
    return data


def main(out: Path) -> None:
    data = build_data(_ROOT / "figure_data" / (out.stem + ".json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    inv = np.array([1.0 / n for n in SIZES])

    # ---- (a) amplitude calibration ----------------------------------
    ax = axes[0]
    series = {
        r"$\lambda\leq3/2$, fit radius $\propto L$": (data["alpha_scaled"], "s--", "#d62728"),
        r"$\lambda\leq3/2$, fit radius fixed": (data["alpha_fixed_32"], "o-", "#ff7f0e"),
        r"$\lambda\leq5/2$, fit radius fixed": (data["alpha_fixed_52"], "D-", "#1f77b4"),
    }
    for label, (vals, style, colour) in series.items():
        ax.plot(inv, vals, style, color=colour, mfc="none", label=label)
    ax.axhline(1.0, color="0.35", lw=1.0, zorder=0)
    ax.plot([1.0 / 80.0], [1.269], "*", color="0.25", markersize=11, zorder=5)
    ax.annotate("archived value,\n$N_x=80$", xy=(1.0 / 80.0, 1.269),
                xytext=(0.0145, 1.20), fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.8))
    ax.set_xlabel(r"$1/N_x$")
    ax.set_ylabel(r"amplitude calibration $\alpha_J$")
    ax.set_xlim(-0.001, 0.023)
    ax.legend(loc="lower right", frameon=False)
    _panel_label(ax, "(a)")

    # ---- (b) mode mixing --------------------------------------------
    ax = axes[1]
    lams = (0.5, 1.5, 2.5)
    floor = data["floor"]
    ax.fill_between(inv, 1e-6, floor, color="0.75", alpha=0.6,
                    label="passive noise floor (exact value 0)")

    for k_o, colour in ((0.10, "#2ca02c"), (0.20, "#1f77b4"), (0.40, "#9467bd")):
        vals = data["kii"][str(k_o)]
        ax.plot(inv, vals, "o-", color=colour, mfc="none", label=rf"$k_o/k={k_o}$")

    ax.set_yscale("log")
    ax.set_ylim(1e-4, 9e-2)
    ax.set_xlabel(r"$1/N_x$")
    ax.set_ylabel(r"$|K_{II}/K_{I}|$, odd part in $k_o$")
    ax.set_xlim(-0.001, 0.023)
    ax.legend(loc="lower right", frameon=False)
    _panel_label(ax, "(b)")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(_ROOT / "figures" / "figNEW_bridge.pdf")
