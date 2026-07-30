
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sys, math, time
import numpy as np
from crack_tip_lattice_fit import analyze_case, homogenized_moduli
from crack_tip_asymptotics import J_matrix
from apparent_j_analysis import solve_field, keyhole_j

def alpha_at(nx, fit_mode):
    ny  = 3*nx//4
    a   = nx/4.0                       # WP1: maximal tip separation
    ri, ro = 0.06*nx, 0.12*nx          # annulus scales with L
    radii  = np.linspace(ri, ro, 9)
    angles = np.linspace(-math.pi+0.10, math.pi-0.10, 73)
    fit = 3.0 if fit_mode=="absolute" else max(2.0, 0.05*nx)
    rows,_,_,_ = analyze_case(nx=nx, ny=ny, crack_half_length=a, k_o=0.0, tip="right",
                              radii=radii, angles=angles, fit_radius=fit,
                              annuli=((ri,ro),))
    r = next(x for x in rows if x["basis"]=="passive")
    K = np.array([float(r["K_I"]), float(r["K_II"])])
    G = J_matrix(homogenized_moduli(1.0,0.0), n_theta=801)
    J_pred = float(K @ G @ K)
    # measured J: passive is path independent, so average over well separated contours
    _,f,_ = solve_field(nx,ny,a,0.0,"right",fit)
    Js = [keyhole_j(f,float(R),0.02) for R in (0.06*nx,0.09*nx,0.12*nx)]
    J_meas = float(np.mean(Js)); spread = (max(Js)-min(Js))/abs(J_meas)
    return dict(nx=nx, fit=fit, K_I=K[0], K_II=K[1], J_pred=J_pred, J_meas=J_meas,
                alpha=J_meas/J_pred, spread=spread, resid=float(r["relative_L2_residual"]))

for mode in ("absolute","scaled_as_in_paper"):
    print(f"\n=== fit_radius {mode} ===")
    print(f"{'nx':>5}{'fit_r':>7}{'K_I':>11}{'K_II/K_I':>10}{'J_pred':>12}{'J_meas':>12}"
          f"{'alpha_J':>9}{'path spread':>12}{'fit resid':>10}")
    for nx in (48,64,96,128,160):
        t=time.time(); d=alpha_at(nx,mode)
        print(f"{d['nx']:>5}{d['fit']:>7.2f}{d['K_I']:>11.4e}{d['K_II']/d['K_I']:>10.4f}"
              f"{d['J_pred']:>12.4e}{d['J_meas']:>12.4e}{d['alpha']:>9.4f}"
              f"{d['spread']:>11.2%}{d['resid']:>10.4f}   [{time.time()-t:.0f}s]", flush=True)
