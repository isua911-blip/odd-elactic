
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
nx,ny,a,ko,step,fit = 128,96,16,0.15,0.3,3.0
_,pf,_=solve_field(nx,ny,a,0.0,'right',fit)
_,af,_=solve_field(nx,ny,a,ko ,'right',fit)
s=MLSSampler(pf,22.0,step); pg=s.apply(pf); ag=s.apply(af)
print(f"nx={nx} (crack half-length {a}, other tip {2*a} away)  fit_radius={fit}")
print(f"{'Ri':>4}{'Ro':>4}{'excess':>13}{'Qodd':>13}{'exc/Qodd':>10}{'Qres/Qodd':>11}{'Qclos/Qodd':>12}")
for Ri,Ro in ((4,8),(6,12),(8,16),(10,20)):
    d=pair(ag,pg,Ri=Ri,Ro=Ro,w=1.5,p=4)
    print(f"{Ri:>4}{Ro:>4}{d['excess']:>13.5e}{d['Qodd']:>13.5e}"
          f"{d['excess']/d['Qodd']:>10.4f}{d['Qres']/d['Qodd']:>11.4f}{d['Qclosure']/d['Qodd']:>12.4f}",flush=True)
