
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sys
import numpy as np
from wp5_criterion import state_quantities, DA
print("C3 = -dUe/da at the lattice advance threshold, normalised to k_o=0")
print("no domain integral, no MLS -- pure energy bookkeeping\n")
print(f"{'k_o':>6}" + "".join(f"{'nx='+str(n):>12}" for n in (96,128,160)))
cols={}
for nx in (96,128,160):
    ny=3*nx//4; a=nx/4.0; ref=None; col=[]
    for ko in (0.0,0.05,0.10,0.20,0.30,0.40):
        s=state_quantities(nx,ny,a,ko); v=-s["dUe"]/DA
        if ref is None: ref=v
        col.append(v/ref)
    cols[nx]=col
for i,ko in enumerate((0.0,0.05,0.10,0.20,0.30,0.40)):
    print(f"{ko:>6.2f}" + "".join(f"{cols[n][i]:>12.4f}" for n in (96,128,160)))
print("\nimplied error if a passive G_c were used to predict the odd lattice:")
for i,ko in enumerate((0.10,0.20,0.40)):
    j=(0.0,0.05,0.10,0.20,0.30,0.40).index(ko)
    print(f"   k_o={ko}:  {100*(1/cols[160][j]-1):+.1f}%")
