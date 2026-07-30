
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sys
import numpy as np
from active_tip_scan import ActiveCrackedStrip
from apparent_j_analysis import LocalTipField, keyhole_j
from periodic_tip import PeriodicTipField
# apparent_j_analysis.main settings: nx=80 ny=56 a=10 fit_radius=4.2, left tip used for -k_o
nx,ny,a,fit=80,56,10.0,4.2
for ko,tip in ((-0.15,"left"),(0.15,"right")):
    m=ActiveCrackedStrip(nx=nx,ny=ny,crack_half_length=a,k=1.0,k_o=ko)
    u,_,_=m.solve(delta=1.0)
    old=LocalTipField(m,u,tip,fit_radius=fit); new=PeriodicTipField(m,u,tip,fit_radius=fit)
    print(f"\nk_o={ko:+.2f}  {tip} tip   (paper's apparent_j_analysis.main configuration)")
    print(f"   {'R':>5}{'J_old':>13}{'J_periodic':>13}{'shift':>9}")
    for R in (4.0,6.0,8.0):
        jo=keyhole_j(old,R,0.02); jn=keyhole_j(new,R,0.02)
        print(f"   {R:>5.1f}{jo:>13.5e}{jn:>13.5e}{(jn-jo)/abs(jo):>8.2%}")
