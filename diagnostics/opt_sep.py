
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

# tips at 0.5*nx +/- a  -> separation 2a one way, nx-2a the other. max-min at a=nx/4.
nx,ny,a,ko,fit,step,hw = 160,120,40.0,0.15,3.0,0.35,36.0
t=time.time()
_,pf,rp = solve_field(nx,ny,a,0.0,'right',fit)
_,af,ra = solve_field(nx,ny,a,ko ,'right',fit)
print(f"periodic strip nx={nx} ny={ny} a={a} -> tip separation {2*a:.0f} and {nx-2*a:.0f}")
print(f"residual p/a = {rp:.1e}/{ra:.1e}")
s=MLSSampler(pf,hw,step); pg=s.apply(pf); ag=s.apply(af)
print(f"[sampler {time.time()-t:.0f}s]\n")
print(f"{'Ri':>4}{'Ro':>5}{'J_passive':>13}{'excess':>13}{'|Jp|/|exc|':>12}{'Qodd':>13}{'exc/Qodd':>10}{'Qres/Qodd':>11}")
for Ri,Ro in ((4,8),(6,12),(8,16),(10,20),(12,24),(14,28),(16,30)):
    d=pair(ag,pg,Ri=Ri,Ro=Ro,w=1.5,p=4)
    print(f"{Ri:>4}{Ro:>5}{d['rawp']:>13.4e}{d['excess']:>13.4e}{abs(d['rawp']/d['excess']):>11.2%}"
          f"{d['Qodd']:>13.4e}{d['excess']/d['Qodd']:>10.4f}{d['Qres']/d['Qodd']:>11.4f}",flush=True)
