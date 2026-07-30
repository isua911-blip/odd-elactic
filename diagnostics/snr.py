
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sys, time
import numpy as np
from apparent_j_analysis import solve_field
from continuum_domain_core import MLSSampler, pair

nx,ny,a,fit,step,hw = 160,120,40.0,3.0,0.35,36.0
_,pf,_ = solve_field(nx,ny,a,0.0,'right',fit)
s=MLSSampler(pf,hw,step); pg=s.apply(pf)
print(f"periodic strip nx={nx} a={a}, fit={fit}.  Passive drift is k_o-independent; signal is O(k_o).")
for Ri,Ro in ((8,16),(16,30)):
    print(f"\n--- annulus Ri={Ri} Ro={Ro} ---")
    print(f"{'k_o':>6}{'J_passive':>13}{'excess':>13}{'|Jp|/|exc|':>12}{'Qodd':>13}{'exc/Qodd':>10}{'Qres/Qodd':>11}")
    for ko in (0.05,0.15,0.30,0.50):
        _,af,_ = solve_field(nx,ny,a,ko,'right',fit)
        ag=s.apply(af); d=pair(ag,pg,Ri=Ri,Ro=Ro,w=1.5,p=4)
        print(f"{ko:>6}{d['rawp']:>13.4e}{d['excess']:>13.4e}{abs(d['rawp']/d['excess']):>11.2%}"
              f"{d['Qodd']:>13.4e}{d['excess']/d['Qodd']:>10.4f}{d['Qres']/d['Qodd']:>11.4f}",flush=True)
