
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sys, math
import numpy as np
from periodic_tip import patch_lattice_fit
import crack_tip_lattice_fit as ctlf

def fit(nx,ko,tip):
    ny=3*nx//4; a=nx/4.0; ri,ro=0.06*nx,0.12*nx
    rows,_,_,_=ctlf.analyze_case(nx=nx,ny=ny,crack_half_length=a,k_o=ko,tip=tip,
        radii=np.linspace(ri,ro,9),angles=np.linspace(-math.pi+0.10,math.pi-0.10,73),
        fit_radius=3.0,annuli=((ri,ro),))
    r=next(x for x in rows if x["basis"]=="matched_odd")
    return float(r["K_I"]), float(r["K_II"])/float(r["K_I"])

for label in ("BEFORE (original LocalTipField)","AFTER (PeriodicTipField)"):
    if label.startswith("AFTER"): patch_lattice_fit()
    print(f"\n=== {label} ===")
    print(f"{'nx':>5}{'k_o':>7}{'tip':>7}{'K_I':>12}{'K_II/K_I':>11}   note")
    for nx in (96,128):
        for ko,tip,note in ((0.20,"right","reference"),
                            (-0.20,"left","mirror of reference: must MATCH"),
                            (-0.20,"right","chirality reversal: must FLIP SIGN")):
            try:
                KI,r=fit(nx,ko,tip)
                print(f"{nx:>5}{ko:>7.2f}{tip:>7}{KI:>12.4e}{r:>11.5f}   {note}")
            except RuntimeError as e:
                print(f"{nx:>5}{ko:>7.2f}{tip:>7}{'--':>12}{'FAIL':>11}   {e.args[0][:40]}")
