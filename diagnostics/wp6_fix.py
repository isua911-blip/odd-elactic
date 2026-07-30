
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import math, sys
import numpy as np
from continuum_independent import ode_propagator, B_LAT, MU_LAT
from crack_tip_asymptotics import OddModuli, J_matrix, analytic_first_order_J_derivatives

def smin(lam,m):
    P=ode_propagator(2*math.pi,lam,m)
    return float(np.linalg.svd(P[2:,:2],compute_uv=False).min())

print("B') spectrum by minimum singular value (det touches zero, so no sign change)")
print("    the traction block collapses to rank 0 at admissible exponents\n")
print(f"    {'A_o':>6}{'K_o':>7}{'ray?':>6}   sigma_min at lambda = 0.5, 1.0, 1.5, 2.0, 2.5")
for A_o,K_o in ((0.0,0.0),(-0.30,-0.15),(-0.30,0.0),(0.0,-0.30),(0.25,-0.40),(0.60,0.10)):
    m=OddModuli(B=B_LAT,mu=MU_LAT,A_o=A_o,K_o=K_o)
    vals=[smin(l,m) for l in (0.5,1.0,1.5,2.0,2.5)]
    ray="yes" if abs(A_o-2*K_o)<1e-12 else "no"
    print(f"    {A_o:>6.2f}{K_o:>7.2f}{ray:>6}   "+"  ".join(f"{v:.2e}" for v in vals))

print("\n    off-half-integer control (must be O(1), not ~0):")
m=OddModuli(B=B_LAT,mu=MU_LAT,A_o=0.25,K_o=-0.40)
print("    "+"  ".join(f"lam={l}: {smin(l,m):.3f}" for l in (0.7,1.2,1.8)))

print("\nC') first-order mixing coefficient d(G12/G11)/dA_o at the origin")
G0=J_matrix(OddModuli(B=B_LAT,mu=MU_LAT,A_o=0.0,K_o=0.0),n_theta=1601)
for h in (0.02,0.01,0.005):
    Gp=J_matrix(OddModuli(B=B_LAT,mu=MU_LAT,A_o=h,K_o=0.0),n_theta=1601)
    Gm=J_matrix(OddModuli(B=B_LAT,mu=MU_LAT,A_o=-h,K_o=0.0),n_theta=1601)
    print(f"    h={h}:  d(G12)/dA_o = {(Gp[0,1]-Gm[0,1])/(2*h)/G0[0,0]:.6f}")
dA,dK = analytic_first_order_J_derivatives(B_LAT,MU_LAT)
print(f"    analytic dG/dA_o [0,1] / G11 = {dA[0,1]/G0[0,0]:.6f}")
print(f"    analytic dG/dK_o [0,1] / G11 = {dK[0,1]/G0[0,0]:.6f}   (must be 0)")
