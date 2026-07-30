
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import sys
import numpy as np
from lattice_baselines import A1,A2,A3,R90
rng=np.random.default_rng(0)
print("affine limit: is  sum_b delta_b*tau_b  ==  sum_b delta_b*phi  exactly?")
print(f"{'trial':>6}{'sum d*tau':>14}{'sum d*phi':>14}{'ratio':>12}")
for k in range(5):
    H=rng.normal(size=(2,2))
    phi=0.5*(H[1,0]-H[0,1])
    st=sp=0.0
    for n in (A1,A2,A3):
        t=R90@n
        d=float(n@H@n); tau=float(t@H@n)
        st+=d*tau; sp+=d*phi
    print(f"{k:>6}{st:>14.6e}{sp:>14.6e}{st/sp:>12.8f}")
print("\n-> for any uniform gradient the shear part contracts to zero,")
print("   because sum_b t_b (x) n_b = (3/2)*epsilon is purely antisymmetric.")
