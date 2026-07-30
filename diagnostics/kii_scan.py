
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sys, math, time
import numpy as np
from crack_tip_lattice_fit import analyze_case

def fit_K(nx, ko):
    ny=3*nx//4; a=nx/4.0; ri,ro=0.06*nx,0.12*nx
    radii=np.linspace(ri,ro,9); angles=np.linspace(-math.pi+0.10,math.pi-0.10,73)
    rows,_,_,_=analyze_case(nx=nx,ny=ny,crack_half_length=a,k_o=ko,tip="right",
                            radii=radii,angles=angles,fit_radius=3.0,annuli=((ri,ro),))
    r=next(x for x in rows if x["basis"]=="matched_odd")
    return float(r["K_I"]), float(r["K_II"]), float(r["relative_L2_residual"])

print("K_II/K_I from the odd-matched stress fit, fit_radius=3.0 (absolute), a=nx/4")
print("passive (k_o=0) column is the systematic noise floor: exact answer is 0\n")
print(f"{'nx':>5}" + "".join(f"{'k_o='+str(k):>14}" for k in (0.0,0.10,0.20,0.40)))
for nx in (48,64,96,128,160):
    t=time.time(); out=[]
    for ko in (0.0,0.10,0.20,0.40):
        KI,KII,res=fit_K(nx,ko); out.append(KII/KI)
    print(f"{nx:>5}" + "".join(f"{v:>14.5f}" for v in out) + f"   [{time.time()-t:.0f}s]", flush=True)
