#!/usr/bin/env python3
"""Drop the two figure panels superseded by the revision.

Run once, from the package root, before regenerating figures:

    python patch_generate_figures.py

`fig02` panel (c) plotted the source-closure ratio approaching unity under
geometric refinement; the revision retracts that reading, and
`figNEW_source_closure` reports the closure under explicit error control.
`fig05` panel (b) measured stress-fit residuals with the reconstruction radius
scaled as ``max(2, 0.05*nx)``, a setting now shown never to refine the
reconstruction; `figNEW_bridge` reports the convergence at a fixed lattice-scale
radius.

The captions in `manuscript.tex` already describe the post-patch layouts.
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "generate_figures.py"
BACKUP = TARGET.with_suffix(".py.orig")


def cut(text: str, old: str, new: str, tag: str) -> str:
    if text.count(old) != 1:
        sys.exit(f"FAIL [{tag}]: anchor matched {text.count(old)} times, expected 1")
    return text.replace(old, new)


src = TARGET.read_text()
if "REVISION-PATCHED" in src:
    sys.exit("generate_figures.py is already patched; nothing to do.")
if not BACKUP.exists():
    BACKUP.write_text(src)

# ---- fig02: three panels -> two -------------------------------------
src = cut(
    src,
    '    refine = pd.read_csv(DATA / "configurational_work_bridge_results" / "source_identity_refinement.csv")\n',
    "",
    "fig02/read",
)
src = cut(
    src,
    "    fig, axs = plt.subplots(1, 3, figsize=(14.4, 4.2))",
    "    fig, axs = plt.subplots(1, 2, figsize=(9.8, 4.2))",
    "fig02/layout",
)
src = cut(
    src,
    """    ax = axs[2]
    d = refine.sort_values("a_lat_over_L")
    ax.plot(d.a_lat_over_L, d.closure_ratio, "o-")
    ax.axhline(1.0, ls="--", lw=1)
    ax.set_xlabel(r"$a_{\\rm lat}/L$")
    ax.set_ylabel(r"$\\Delta J_{\\rm key}/Q_o^{\\rm MLS}$")
    ax.set_title("reconstructed source closure")
    ax.grid(alpha=0.25)
    _panel_label(ax, "(c)")

""",
    "",
    "fig02/panel_c",
)

# ---- fig05: four panels -> three ------------------------------------
src = cut(
    src,
    '    tip = pd.read_csv(DATA / "refinement_validation_results" / "crack_tip_refinement.csv")\n',
    "",
    "fig05/read",
)
src = cut(
    src,
    "    fig, axs = plt.subplots(2, 2, figsize=(11.2, 8.3))",
    "    fig, axs = plt.subplots(1, 3, figsize=(10.8, 3.7))",
    "fig05/layout",
)
src = cut(src, "    ax = axs[0, 0]\n    for R_over_L, marker in", "    ax = axs[0]\n    for R_over_L, marker in", "fig05/a")
src = cut(
    src,
    """    ax = axs[0, 1]
    passive = tip[np.isclose(tip.k_o, 0.0)].sort_values("a_lat_over_L")
    active = tip[np.isclose(tip.k_o, 0.15)].sort_values("a_lat_over_L")
    ax.plot(passive.a_lat_over_L, 100 * passive.matched_relative_residual, "o--", label="passive basis")
    ax.plot(active.a_lat_over_L, 100 * active.matched_relative_residual, "s-", label="matched odd basis")
    ax.plot(active.a_lat_over_L, 100 * active.passive_basis_relative_residual, "^-", label="wrong passive basis")
    ax.set_xlabel(r"$a_{\\rm lat}/L$")
    ax.set_ylabel("stress-fit residual (%)")
    ax.set_title(r"Fixed annulus $0.05\\leq r/L\\leq0.09$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _panel_label(ax, "(b)")

""",
    "",
    "fig05/panel_b",
)
# the remaining relabels are scoped to make_fig5_scaling, because the same
# anchors occur in make_fig3, make_fig6 and make_fig7
_start = src.index("def make_fig5_scaling(")
_end = src.index("def make_fig7(")
_body = src[_start:_end]
for _old, _new, _tag in (
    ("    ax = axs[1, 0]\n", "    ax = axs[1]\n", "fig05/c"),
    ('    _panel_label(ax, "(c)")', '    _panel_label(ax, "(b)")', "fig05/relabel_c"),
    ("    ax = axs[1, 1]\n", "    ax = axs[2]\n", "fig05/d"),
    ('    _panel_label(ax, "(d)")', '    _panel_label(ax, "(c)")', "fig05/relabel_d"),
):
    if _body.count(_old) != 1:
        sys.exit(f"FAIL [{_tag}]: anchor matched {_body.count(_old)} times inside make_fig5_scaling")
    _body = _body.replace(_old, _new)
src = src[:_start] + _body + src[_end:]

src = src.replace(
    '"""', '"""', 1
)
src = "# REVISION-PATCHED: fig02 panel (c) and fig05 panel (b) removed; see README.\n" + src

TARGET.write_text(src)
print("patched generate_figures.py")
print("  fig02: 1x3 -> 1x2, panel (c) source-closure refinement removed")
print("  fig05: 2x2 -> 1x3, panel (b) stress-fit residuals removed, panels relabelled")
print(f"  original saved as {BACKUP.name}")
