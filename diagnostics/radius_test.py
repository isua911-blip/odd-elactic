
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sys
import numpy as np
from apparent_j_analysis import solve_field
from continuum_domain_core import MLSSampler, pair

nx,ny,a,ko,step,fit = 80,56,10,0.15,0.3,3.0
_,pf,_=solve_field(nx,ny,a,0.0,'right',fit)
_,af,_=solve_field(nx,ny,a,ko ,'right',fit)
s=MLSSampler(pf,18.0,step)
pg=s.apply(pf); ag=s.apply(af)
print(f"closure vs annulus radius at FIXED nx={nx}, fit_radius={fit}  (R in lattice spacings)")
print(f"{'Ri':>5}{'Ro':>5}{'excess':>13}{'Qodd':>13}{'exc/Qodd':>10}{'Qres/Qodd':>11}{'Qclos/Qodd':>12}")
for Ri,Ro in ((2,4),(3,6),(4,8),(5,10),(6,12),(7,14),(8,16)):
    d=pair(ag,pg,Ri=Ri,Ro=Ro,w=1.5,p=4)
    print(f"{Ri:>5}{Ro:>5}{d['excess']:>13.5e}{d['Qodd']:>13.5e}"
          f"{d['excess']/d['Qodd']:>10.4f}{d['Qres']/d['Qodd']:>11.4f}{d['Qclosure']/d['Qodd']:>12.4f}", flush=True)
