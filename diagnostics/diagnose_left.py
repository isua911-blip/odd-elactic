
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sys, math
import numpy as np
from active_tip_scan import ActiveCrackedStrip
nx,ny,a=128,96,32.0
m=ActiveCrackedStrip(nx=nx,ny=ny,crack_half_length=a,k=1.0,k_o=0.2)
p=m.positions
print(f"period={m.period}  but node x spans [{p[:,0].min():.1f}, {p[:,0].max():.1f}]  <-- not wrapped")
crack_y=0.5*(p[m.node_id(0,m.j_lower),1]+p[m.node_id(0,m.j_lower+1),1])
tipL=0.5*m.period-m.a_eff; tipR=0.5*m.period+m.a_eff
R=0.12*nx
print(f"crack_y={crack_y:.2f}  left tip x={tipL:.1f}  right tip x={tipR:.1f}  sampling R={R:.1f}\n")
print(f"{'row j':>7}{'row y':>8}{'row x_min':>11}{'row x_max':>11}{'L-tip needs x>=':>17}{'ok?':>6}")
for j in (30,40,50,60,65):
    ys=p[m.node_id(0,j),1]
    xs=[p[m.node_id(i,j),0] for i in range(nx)]
    need=tipL-R
    print(f"{j:>7}{ys:>8.1f}{min(xs):>11.1f}{max(xs):>11.1f}{need:>17.1f}{'OK' if min(xs)<=need else 'MISSING':>8}")
