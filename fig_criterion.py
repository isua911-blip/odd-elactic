#!/usr/bin/env python3
"""New figure: no coarse-grained advance criterion survives (WP5).

Panel (a)  Candidate quantities evaluated at the lattice's own advance
           threshold, each normalised to its passive value.  A usable criterion
           "X >= X_c" requires the curve to stay at unity.
Panel (b)  The recoverable-energy release rate at three system sizes, showing
           the drift is converged physics rather than a discretisation artefact.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from apparent_j_analysis import solve_field  # noqa: E402
from continuum_domain_core import MLSSampler  # noqa: E402
from wp5_criterion import DA, flux_terms, state_quantities  # noqa: E402

plt.rcParams.update({
    "font.size": 10.5, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 8.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "lines.linewidth": 1.7, "lines.markersize": 5.5,
})

KOS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)


def _panel_label(ax, label, x=0.02, y=0.96):
    ax.text(x, y, label, transform=ax.transAxes, va="top", ha="left",
            fontweight="bold", fontsize=13,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))


def build(cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())

    nx, ny = 128, 96
    crack, fit, step, hw = nx / 4.0, 3.0, 0.35, 26.0
    _m, pf, _r = solve_field(nx, ny, crack, 0.0, "right", fit)
    sampler = MLSSampler(pf, hw, step)
    grid0 = sampler.apply(pf)

    rows = {"C1": [], "C2": [], "C3": [], "ref": []}
    for k_o in KOS:
        s = state_quantities(nx, ny, crack, k_o)
        J1, Q = flux_terms(sampler, grid0, nx, ny, crack, k_o, fit, 8.0, 16.0)
        p2 = s["p_c"] ** 2
        rows["C1"].append(J1 * p2)
        rows["C2"].append((J1 - Q) * p2)
        rows["C3"].append(-s["dUe"] / DA)
        rows["ref"].append(s["E_cut"] / DA)
    data = {k: [v / vals[0] for v in vals] for k, vals in rows.items()}

    data["sizes"] = {}
    for size in (96, 128, 160):
        vals = [-state_quantities(size, 3 * size // 4, size / 4.0, k)["dUe"] / DA for k in KOS]
        data["sizes"][str(size)] = [v / vals[0] for v in vals]

    cache.write_text(json.dumps(data, indent=1))
    return data


def main(out: Path) -> None:
    data = build(_ROOT / "figure_data" / (out.stem + ".json"))
    ko = np.array(KOS)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    ax = axes[0]
    ax.axhline(1.0, color="0.35", lw=1.0, zorder=0)
    ax.axhspan(0.98, 1.02, color="0.85", alpha=0.7, zorder=0)
    style = (
        ("ref", r"local bond energy $E_{\rm cut}$ (control)", "k:", "black"),
        ("C1", r"apparent flux $J_h$", "s-", "#d62728"),
        ("C2", r"$J_h$ corrected to tip by $Q_o$", "^--", "#ff7f0e"),
        ("C3", r"energy release rate $-\Delta U_e/\Delta a$", "o-", "#1f77b4"),
    )
    for key, label, fmt, colour in style:
        ax.plot(ko, data[key], fmt, color=colour, mfc="none", label=label)
    ax.set_xlabel(r"$k_o/k$")
    ax.set_ylabel("value at advance threshold\n(normalised to passive)")
    ax.set_ylim(0.58, 1.09)
    ax.legend(loc="lower left", frameon=False)
    _panel_label(ax, "(a)")

    ax = axes[1]
    ax.axhline(1.0, color="0.35", lw=1.0, zorder=0)
    for size, colour in ((96, "#2ca02c"), (128, "#1f77b4"), (160, "#9467bd")):
        ax.plot(ko, data["sizes"][str(size)], "o-", color=colour, mfc="none",
                label=rf"$N_x={size}$")
    ax.annotate("non-monotonic:\nno fitted $G_c(k_o)$ can work",
                xy=(0.20, data["sizes"]["160"][3]), xytext=(0.115, 0.90),
                fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))
    ax.set_xlabel(r"$k_o/k$")
    ax.set_ylabel(r"$-\Delta U_e/\Delta a$ (normalised)")
    ax.set_ylim(0.74, 1.04)
    ax.legend(loc="lower right", frameon=False)
    _panel_label(ax, "(b)")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(_ROOT / "figures" / "figNEW_criterion.pdf")
