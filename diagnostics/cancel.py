
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sys
from apparent_j_analysis import solve_field
from continuum_domain_core import MLSSampler, pair
for nx,ny,a,hw in ((80,56,10,14.0),(128,96,16,22.0)):
    _,pf,_=solve_field(nx,ny,a,0.0,'right',3.0)
    _,af,_=solve_field(nx,ny,a,0.15,'right',3.0)
    s=MLSSampler(pf,hw,0.3); pg=s.apply(pf); ag=s.apply(af)
    print(f"\nnx={nx}  passive-subtraction conditioning")
    print(f"{'Ri':>4}{'Ro':>4}{'J_active':>13}{'J_passive':>13}{'excess':>13}{'|exc|/|J_p|':>13}{'exc/Qodd':>10}")
    for Ri,Ro in ((4,8),(6,12),(8,16)):
        d=pair(ag,pg,Ri=Ri,Ro=Ro,w=1.5,p=4)
        can=abs(d['excess'])/max(abs(d['rawp']),1e-30)
        print(f"{Ri:>4}{Ro:>4}{d['rawa']:>13.5e}{d['rawp']:>13.5e}{d['excess']:>13.5e}"
              f"{can:>12.3%}{d['excess']/d['Qodd']:>10.4f}",flush=True)
