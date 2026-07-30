
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sys, time
import numpy as np
from continuum_domain_core import MLSSampler, pair
from edge_crack import solve_edge_field

nx,ny,cl,ko,fit,step,hw = 128,96,24.0,0.15,3.0,0.3,24.0
t=time.time()
mp,pf,rp = solve_edge_field(nx,ny,cl,0.0,fit)
ma,af,ra = solve_edge_field(nx,ny,cl,ko ,fit)
print(f"SENT nx={nx} ny={ny} crack={cl}  tip_x={ma.tip_x:.2f}  residual p/a = {rp:.1e}/{ra:.1e}")
print("clearances:", {k:round(v,1) for k,v in ma.clearances().items()})
s=MLSSampler(pf,hw,step); pg=s.apply(pf); ag=s.apply(af)
print(f"[sampler {time.time()-t:.0f}s]\n")
print(f"{'Ri':>4}{'Ro':>5}{'J_passive':>13}{'excess':>13}{'|Jp|/|exc|':>12}{'Qodd':>13}{'exc/Qodd':>10}{'Qres/Qodd':>11}")
for Ri,Ro in ((4,8),(6,12),(8,16),(10,18),(12,20),(14,22)):
    d=pair(ag,pg,Ri=Ri,Ro=Ro,w=1.5,p=4)
    gate=abs(d['rawp'])/abs(d['excess'])
    print(f"{Ri:>4}{Ro:>5}{d['rawp']:>13.4e}{d['excess']:>13.4e}{gate:>11.2%}"
          f"{d['Qodd']:>13.4e}{d['excess']/d['Qodd']:>10.4f}{d['Qres']/d['Qodd']:>11.4f}",flush=True)
