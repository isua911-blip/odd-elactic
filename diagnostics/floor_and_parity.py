
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sys
import numpy as np
from wp2_closeout import fit_case
L=(0.5,1.5,2.5)
print("passive noise floor and parity decomposition, lambda<=5/2 basis, |k_o|=0.20")
print(f"{'nx':>5}{'floor (k_o=0)':>15}{'odd part':>12}{'even part':>12}{'even/|odd|':>12}{'odd/floor':>11}")
for nx in (48,64,96,128,160):
    KI0,KII0,_ = fit_case(nx,0.0,"right",L); floor=KII0/KI0
    KIp,KIIp,_ = fit_case(nx, 0.20,"right",L); rp=KIIp/KIp
    KIm,KIIm,_ = fit_case(nx,-0.20,"right",L); rm=KIIm/KIm
    odd=0.5*(rp-rm); even=0.5*(rp+rm)
    print(f"{nx:>5}{floor:>15.5f}{odd:>12.5f}{even:>12.5f}{abs(even/odd):>12.3f}{abs(odd/floor):>11.1f}",flush=True)
