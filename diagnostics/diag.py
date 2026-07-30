
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import math, sys
import numpy as np
from apparent_j_analysis import solve_field, keyhole_j, annulus_sources
nx,ny,a=80,56,10.0
print("NOTE: paper uses fit_radius=4.2 with r_inner=4.0 -> MLS stencil straddles the tip\n")
for fr in (4.2, 2.0):
    _,pf,_=solve_field(nx,ny,a,0.0,"right",fr)
    _,af,_=solve_field(nx,ny,a,0.15,"right",fr)
    print(f"--- fit_radius = {fr} ---")
    print(f"   {'line_step':>10} {'dJ_odd':>14}")
    for ls in (0.10,0.05,0.02):
        d=(keyhole_j(af,8.0,ls)-keyhole_j(af,4.0,ls))-(keyhole_j(pf,8.0,ls)-keyhole_j(pf,4.0,ls))
        print(f"   {ls:>10} {d:>14.6e}")
    dJ=(keyhole_j(af,8.0,0.02)-keyhole_j(af,4.0,0.02))-(keyhole_j(pf,8.0,0.02)-keyhole_j(pf,4.0,0.02))
    print(f"   {'area_step':>10} {'Q_o':>14} {'dJ/Q_o':>10}")
    for s in (0.34,0.14,0.07):
        q,_,_=annulus_sources(af,4.0,8.0,s)
        print(f"   {s:>10} {q:>14.6e} {dJ/q:>10.5f}")
    print()
