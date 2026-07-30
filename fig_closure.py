#!/usr/bin/env python3
"""New figure: the discrete source closure needs signal-to-noise control (WP1).

Panel (a)  Closure ratio against the passive-drift gate.  The passive domain
           integral must vanish identically, so its residual measures the noise
           on the same footing as the signal; the ratio only settles once that
           gate is small.  Because the passive field carries no k_o, raising k_o
           moves a point left along the gate axis without changing the noise.
Panel (b)  The continuum-equilibrium diagnostic Q_res/Q_odd against annulus
           radius: the coarse-grained field satisfies continuum equilibrium only
           once the contour sits many lattice spacings from the tip.
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
from continuum_domain_core import MLSSampler, pair  # noqa: E402

plt.rcParams.update({
    "font.size": 10.5, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 8.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "lines.linewidth": 1.7, "lines.markersize": 5.5,
})

NX, NY, A = 160, 120, 40.0          # WP1 protocol: a = nx/4, maximal tip separation
FIT, STEP, HW = 3.0, 0.35, 36.0
KOS = (0.05, 0.15, 0.30, 0.50)
ANNULI = ((8.0, 16.0), (16.0, 30.0))
RADII = ((4, 8), (6, 12), (8, 16), (10, 20), (12, 24), (14, 28), (16, 30))


def _panel_label(ax, label, x=0.02, y=0.96):
    ax.text(x, y, label, transform=ax.transAxes, va="top", ha="left",
            fontweight="bold", fontsize=13,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))


def build(cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())

    _m, pf, _r = solve_field(NX, NY, A, 0.0, "right", FIT)
    sampler = MLSSampler(pf, HW, STEP)
    passive = sampler.apply(pf)

    grids = {}
    for k_o in KOS:
        _m, af, _r = solve_field(NX, NY, A, k_o, "right", FIT)
        grids[k_o] = sampler.apply(af)

    closure = {}
    for Ri, Ro in ANNULI:
        pts = []
        for k_o in KOS:
            d = pair(grids[k_o], passive, Ri=Ri, Ro=Ro, w=1.5, p=4)
            pts.append({
                "k_o": k_o,
                "gate": abs(d["rawp"] / d["excess"]),
                "ratio": d["excess"] / d["Qodd"],
            })
        closure[f"{Ri:.0f}-{Ro:.0f}"] = pts

    residual = []
    for Ri, Ro in RADII:
        d = pair(grids[0.50], passive, Ri=float(Ri), Ro=float(Ro), w=1.5, p=4)
        residual.append({"Ri": Ri, "qres": abs(d["Qres"] / d["Qodd"])})

    data = {"closure": closure, "residual": residual}
    cache.write_text(json.dumps(data, indent=1))
    return data


def main(out: Path) -> None:
    data = build(_ROOT / "figure_data" / (out.stem + ".json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    # ---- (a) closure vs the passive-drift gate -----------------------
    ax = axes[0]
    ax.axhline(1.0, color="0.35", lw=1.0, zorder=0)
    ax.axvspan(1e-3, 0.05, color="0.88", alpha=0.8, zorder=0)
    ax.text(0.011, 1.155, "admissible\n(gate < 5%)", fontsize=7.5, color="0.35", ha="center")

    for (key, colour, marker) in (("8-16", "#d62728", "s"), ("16-30", "#1f77b4", "o")):
        pts = data["closure"][key]
        gate = [p["gate"] for p in pts]
        ratio = [p["ratio"] for p in pts]
        ax.plot(gate, ratio, marker + "-", color=colour, mfc="none",
                label=rf"$R_i/R_o = {key.replace('-', '/')}$")
        for p in pts:
            ax.annotate(rf"${p['k_o']}$", xy=(p["gate"], p["ratio"]),
                        xytext=(2.5, -8), textcoords="offset points",
                        fontsize=7, color=colour)

    ax.set_xscale("log")
    ax.set_xlabel(r"passive-drift gate $|J_{\rm passive}|/|\Delta J|$")
    ax.set_ylabel(r"closure ratio $\Delta J/Q_o$")
    ax.set_xlim(8e-3, 12.0)
    ax.set_ylim(0.90, 1.20)
    ax.legend(loc="upper right", frameon=False)
    _panel_label(ax, "(a)")

    # ---- (b) continuum-equilibrium diagnostic ------------------------
    ax = axes[1]
    Ri = [r["Ri"] for r in data["residual"]]
    qres = [r["qres"] for r in data["residual"]]
    ax.plot(Ri, qres, "o-", color="#1f77b4", mfc="none")
    ax.axhline(0.05, color="0.5", ls="--", lw=1.0)
    ax.text(4.4, 0.058, "5% guideline", fontsize=8, color="0.4")
    ax.axvline(4.0, color="#d62728", ls=":", lw=1.4)
    ax.text(4.4, 0.55, "archived\noperating point", fontsize=7.5, color="#d62728")
    ax.set_yscale("log")
    ax.set_xlabel(r"inner radius $R_i$ (lattice spacings)")
    ax.set_ylabel(r"$|Q_{\rm res}/Q_{\rm odd}|$")
    ax.set_xlim(2.5, 17.5)
    _panel_label(ax, "(b)")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    for key, pts in data["closure"].items():
        best = min(pts, key=lambda p: p["gate"])
        print(f"  annulus {key}: lowest gate {best['gate']:.1%} -> ratio {best['ratio']:.4f}")


if __name__ == "__main__":
    main(_ROOT / "figures" / "figNEW_source_closure.pdf")
