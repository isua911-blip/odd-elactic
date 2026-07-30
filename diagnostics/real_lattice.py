
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import math, sys
import numpy as np
from apparent_j_analysis import solve_field, keyhole_j, annulus_sources

nx,ny,a,fr = 80,56,10.0,4.2
_,pf,_ = solve_field(nx,ny,a,0.0,"right",fr)
_,af,_ = solve_field(nx,ny,a,0.15,"right",fr)

# passive-subtracted drift, Eq.(29)
dj_p = keyhole_j(pf,8.0,0.02)-keyhole_j(pf,4.0,0.02)
dj_a = keyhole_j(af,8.0,0.02)-keyhole_j(af,4.0,0.02)
dJ   = dj_a-dj_p
print(f"real lattice nx=80, k_o=0.15   dJ_odd (passive-subtracted) = {dJ:.6e}")
print(f"{'area_step':>10} {'Q_o(active)':>14} {'Q_o(passive)':>14} {'dJ/Q_o':>10} {'error':>9}")
for s in (0.34,0.28,0.14,0.07):
    qa,_,_ = annulus_sources(af,4.0,8.0,s)
    qp,_,_ = annulus_sources(pf,4.0,8.0,s)
    q = qa-qp
    print(f"{s:>10} {qa:>14.6e} {qp:>14.6e} {dJ/q:>10.5f} {100*(dJ/q-1):>8.2f}%")
