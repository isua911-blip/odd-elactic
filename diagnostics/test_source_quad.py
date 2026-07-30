
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import math, sys
import numpy as np
from active_tip_scan import ActiveCrackedStrip
from apparent_j_analysis import annulus_sources
from crack_tip_asymptotics import OddModuli
from reconstruction_bias import AnalyticWilliamsField, AnalyticCoefficientField, WilliamsTerm, synthesize_nodal_displacement

ko=0.15
mod=OddModuli(B=math.sqrt(3)/2, mu=math.sqrt(3)/4, A_o=-math.sqrt(3)/2*ko, K_o=-math.sqrt(3)/4*ko)
m=ActiveCrackedStrip(nx=64, ny=48, crack_half_length=8.0, k=1.0, k_o=ko)

def q_of(terms, steps):
    an=AnalyticWilliamsField(mod,terms)
    u=synthesize_nodal_displacement(m,"right",an)
    f=AnalyticCoefficientField(m,u,"right",an,fit_radius=4.2)
    return [annulus_sources(f,4.0,8.0,s)[0] for s in steps]

steps=(0.28,0.14,0.07,0.035)
pure   = q_of((WilliamsTerm(0.5,1.0,0.0),), steps)                       # self-source only: exact Q_o = 0
cross  = q_of((WilliamsTerm(0.5,1.0,0.0),WilliamsTerm(1.5,0.30,0.10)), steps)
onlyhi = q_of((WilliamsTerm(1.5,0.30,0.10),), steps)

print(f"{'area_step':>10} {'Q[lam=1/2 only]':>18} {'Q[lam=3/2 only]':>18} {'Q[both]':>14} {'spurious/total':>15}")
for i,s in enumerate(steps):
    print(f"{s:>10} {pure[i]:>18.6e} {onlyhi[i]:>18.6e} {cross[i]:>14.6e} {abs(pure[i]/cross[i])*100:>14.2f}%")
