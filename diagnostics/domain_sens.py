
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

nx,ny,a,ko,step = 80,56,10,0.15,0.3
print(f"domain (Q-function) route, nx={nx}, k_o={ko}, Ri=4 Ro=8 w=1.5 p=4, grid step={step}")
print(f"{'fit_radius':>11}{'excess':>13}{'Qodd(FD)':>13}{'Qodd_poly':>13}{'exc/FD':>10}{'exc/poly':>10}{'Qres':>11}")
for fit in (2.5,3.0,3.5,4.2,5.0):
    t=time.time()
    _,pf,_=solve_field(nx,ny,a,0.0,'right',fit)
    _,af,_=solve_field(nx,ny,a,ko ,'right',fit)
    s=MLSSampler(pf,9.3,step)
    pg=s.apply(pf); ag=s.apply(af)
    d=pair(ag,pg)
    print(f"{fit:>11}{d['excess']:>13.5e}{d['Qodd']:>13.5e}{d['Qodd_poly']:>13.5e}"
          f"{d['excess']/d['Qodd']:>10.4f}{d['excess']/d['Qodd_poly']:>10.4f}{d['Qres']:>11.2e}"
          f"   [{time.time()-t:.0f}s]", flush=True)
