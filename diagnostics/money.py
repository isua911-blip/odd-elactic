
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import math, sys
import numpy as np
from active_tip_scan import ActiveCrackedStrip
from apparent_j_analysis import annulus_sources, keyhole_j
from crack_tip_asymptotics import OddModuli
from reconstruction_bias import AnalyticWilliamsField, AnalyticCoefficientField, WilliamsTerm, synthesize_nodal_displacement

def darea(crack_y,ri,ro,s):
    ys=np.arange(crack_y-ro+0.5*s, crack_y+ro, s); xs=np.arange(-ro+0.5*s, ro, s)
    X,Y=np.meshgrid(xs,ys); R=np.maximum(np.abs(X),np.abs(Y-crack_y))
    return int(np.count_nonzero((R>=ri)&(R<ro)))*s*s

print("A) geometric domain error at the settings actually used in apparent_j_analysis.main")
for s in (0.34,0.28):
    for ri,ro in ((4.0,8.0),):
        ex=(2*ro)**2-(2*ri)**2; a=darea(0.5,ri,ro,s)
        print(f"   area_step={s}  annulus[{ri},{ro}]  area {a:.4f} vs {ex:.1f}  -> {100*(a-ex)/ex:+.3f}%")

ko=0.15
mod=OddModuli(B=math.sqrt(3)/2, mu=math.sqrt(3)/4, A_o=-math.sqrt(3)/2*ko, K_o=-math.sqrt(3)/4*ko)
m=ActiveCrackedStrip(nx=80, ny=56, crack_half_length=10.0, k=1.0, k_o=ko)
an=AnalyticWilliamsField(mod,(WilliamsTerm(0.5,1.0,0.0),WilliamsTerm(1.5,0.30,0.10)))
u=synthesize_nodal_displacement(m,"right",an)

exact=AnalyticCoefficientField(m,u,"right",an,fit_radius=4.2)   # exact derivatives
mls  =LocalTipField=None
from apparent_j_analysis import LocalTipField
mls  =LocalTipField(m,u,"right",fit_radius=4.2)                 # MLS derivatives, same field

print("\nB) closure ratio dJ/Q_o  (annulus [4,8], nx=80)")
print(f"   {'route':<22}{'line/area':>14}{'dJ':>14}{'Q_o':>14}{'dJ/Q_o':>10}{'error':>10}")
for name,f in (("exact derivatives",exact),("MLS reconstruction",mls)):
    for ls,as_ in ((0.10,0.34),(0.02,0.07)):
        dj=keyhole_j(f,8.0,ls)-keyhole_j(f,4.0,ls)
        q,_,_=annulus_sources(f,4.0,8.0,as_)
        print(f"   {name:<22}{str(ls)+'/'+str(as_):>14}{dj:>14.6e}{q:>14.6e}{dj/q:>10.5f}{100*(dj/q-1):>9.2f}%")
