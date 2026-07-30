import numpy as np
def discrete_area(crack_y,r_in,r_out,step):
    ys=np.arange(crack_y-r_out+0.5*step, crack_y+r_out, step)
    xs=np.arange(-r_out+0.5*step, r_out, step)
    X,Y=np.meshgrid(xs,ys,indexing='xy')
    R=np.maximum(np.abs(X),np.abs(Y-crack_y))
    return int(np.count_nonzero((R>=r_in)&(R<r_out)))*step*step
exact=(2*8.0)**2-(2*4.0)**2
print("geometric domain error of the annulus_sources sampling rule (exact area = %.1f)"%exact)
print(f"{'area_step':>10} {'discrete area':>15} {'rel. error':>12}")
for s in (0.28,0.14,0.07,0.035,0.0175,0.00875):
    a=discrete_area(0.5,4.0,8.0,s)
    print(f"{s:>10} {a:>15.5f} {100*(a-exact)/exact:>11.3f}%")
