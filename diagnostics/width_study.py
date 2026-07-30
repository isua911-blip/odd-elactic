
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sys, math
import numpy as np
from periodic_tip import patch_lattice_fit; patch_lattice_fit()
import crack_tip_lattice_fit as ctlf
from crack_tip_asymptotics import J_matrix
from apparent_j_analysis import keyhole_j, solve_field
from wp2_closeout import extended_basis

def fit(nx,ko,fin,fout,lams):
    ny=3*nx//4; a=nx/4.0; ri,ro=fin*nx,fout*nx
    radii=np.linspace(ri,ro,13); angles=np.linspace(-math.pi+0.10,math.pi-0.10,73)
    m=ctlf.ActiveCrackedStrip(nx=nx,ny=ny,crack_half_length=a,k=1.0,k_o=ko)
    u,_,_=m.solve(delta=1.0)
    f,s=ctlf.sample_lattice_stress(m,u,"right",radii,angles,fit_radius=3.0)
    mod=ctlf.homogenized_moduli(m.k,f.k_o_local)
    D=extended_basis(s,mod,lams).reshape((-1,2*len(lams)+4))
    c,*_=np.linalg.lstsq(D,s.stress.reshape(-1),rcond=None)
    return float(c[0]),float(c[1]),np.linalg.cond(D)

G=J_matrix(ctlf.homogenized_moduli(1.0,0.0),n_theta=801)
nx=160
print(f"nx={nx}, lambda<=5/2 basis.  Widening the annulus to break collinearity.")
print(f"{'R/L window':>14}{'r_out/r_in':>11}{'cond(D)':>11}{'alpha_J':>10}{'floor':>10}{'odd part':>11}{'odd/floor':>11}")
for fin,fout in ((0.06,0.12),(0.05,0.15),(0.04,0.18),(0.035,0.22)):
    _m,fl,_ = solve_field(nx,3*nx//4,nx/4.0,0.0,"right",3.0)
    Jm=float(np.mean([keyhole_j(fl,float(R),0.02) for R in (fin*nx,0.5*(fin+fout)*nx,fout*nx)]))
    KI0,KII0,cd = fit(nx,0.0,fin,fout,(0.5,1.5,2.5)); floor=KII0/KI0
    K=np.array([KI0,KII0]); alpha=Jm/float(K@G@K)
    KIp,KIIp,_ = fit(nx, 0.20,fin,fout,(0.5,1.5,2.5))
    KIm,KIIm,_ = fit(nx,-0.20,fin,fout,(0.5,1.5,2.5))
    odd=0.5*(KIIp/KIp - KIIm/KIm)
    print(f"{f'{fin}-{fout}':>14}{fout/fin:>11.2f}{cd:>11.1f}{alpha:>10.4f}{floor:>10.5f}"
          f"{odd:>11.5f}{abs(odd/floor):>11.1f}",flush=True)
