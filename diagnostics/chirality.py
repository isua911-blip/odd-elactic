
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sys, math
import numpy as np
from crack_tip_lattice_fit import analyze_case
def fit(nx,ko,tip):
    ny=3*nx//4; a=nx/4.0; ri,ro=0.06*nx,0.12*nx
    rows,_,_,_=analyze_case(nx=nx,ny=ny,crack_half_length=a,k_o=ko,tip=tip,
        radii=np.linspace(ri,ro,9),angles=np.linspace(-math.pi+0.10,math.pi-0.10,73),
        fit_radius=3.0,annuli=((ri,ro),))
    r=next(x for x in rows if x["basis"]=="matched_odd")
    return float(r["K_II"])/float(r["K_I"])
nx=128
print(f"chirality / mirror validation at nx={nx}, |k_o|=0.20")
for ko,tip,note in ((0.20,"right","reference"),(-0.20,"right","k_o -> -k_o, same tip: expect sign flip"),
                    (-0.20,"left","mirror image of reference: expect match")):
    print(f"  k_o={ko:+.2f} tip={tip:<6} K_II/K_I = {fit(nx,ko,tip):+.5f}   ({note})")
