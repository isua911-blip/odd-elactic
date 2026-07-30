#!/usr/bin/env python3
from __future__ import annotations
import math, sys, json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from apparent_j_analysis import solve_field
from lattice_baselines import A1,A2,A3,R90,SQRT3

@dataclass
class GridFields:
    x: np.ndarray; y: np.ndarray; step: float
    p_x: np.ndarray; p_y: np.ndarray
    source_odd: np.ndarray; source_odd_poly: np.ndarray; source_residual: np.ndarray; source_divp: np.ndarray
    closure: np.ndarray

class MLSSampler:
    def __init__(self, reference_field, half_width, step):
        self.field=reference_field; self.step=step
        vals=np.arange(-half_width+0.5*step,half_width,step)
        self.vals=vals; self.n=len(vals)
        xx,yy=np.meshgrid(vals,vals,indexing='xy')
        self.x=xx; self.y=yy
        self.ops=[]
        for xr,yr in zip(xx.ravel(),yy.ravel()):
            yabs=reference_field.crack_y+yr
            ids=reference_field._neighbor_ids(float(xr),float(yabs))
            dx=reference_field.positions[ids,0]-xr
            dy=reference_field.positions[ids,1]-yabs
            radius=np.hypot(dx,dy)
            design=np.column_stack([np.ones(len(ids)),dx,dy,0.5*dx*dx,dx*dy,0.5*dy*dy])
            weights=np.exp(-((radius/(0.65*reference_field.fit_radius))**2))
            A=design*weights[:,None]
            op=np.linalg.pinv(A) * weights[None,:]
            self.ops.append((ids,op))
    def apply(self, field):
        disp=field.displacement
        coeff=np.empty((len(self.ops),6,2))
        for m,(ids,op) in enumerate(self.ops): coeff[m]=op@disp[ids]
        G=np.empty((len(self.ops),2,2))
        G[:,0,0]=coeff[:,1,0]; G[:,0,1]=coeff[:,2,0]
        G[:,1,0]=coeff[:,1,1]; G[:,1,1]=coeff[:,2,1]
        se=np.zeros_like(G); so=np.zeros_like(G); W=np.zeros(len(self.ops))
        area=SQRT3/2
        for n in (A1,A2,A3):
            t=R90@n
            ext=np.einsum('i,nij,j->n',n,G,n)
            se += field.k*ext[:,None,None]*np.outer(n,n)[None,:,:]/area
            so += -field.k_o_local*ext[:,None,None]*np.outer(t,n)[None,:,:]/area
            W += .5*field.k*ext**2/area
        st=se+so; ux=G[:,:,0]
        px=W-np.einsum('ni,ni->n',ux,st[:,:,0])
        py=-np.einsum('ni,ni->n',ux,st[:,:,1])
        N=self.n; h=self.step
        G=G.reshape(N,N,2,2); st=st.reshape(N,N,2,2); so=so.reshape(N,N,2,2)
        px=px.reshape(N,N); py=py.reshape(N,N)
        dGdx=np.gradient(G,h,axis=1,edge_order=2)
        dstdx=np.gradient(st,h,axis=1,edge_order=2)
        dstdy=np.gradient(st,h,axis=0,edge_order=2)
        div= dstdx[:,:,:,0]+dstdy[:,:,:,1]
        source_odd=-np.sum(so*dGdx,axis=(2,3))
        dGdx_poly=np.empty_like(G)
        c=coeff.reshape(N,N,6,2)
        dGdx_poly[:,:,0,0]=c[:,:,3,0]; dGdx_poly[:,:,0,1]=c[:,:,4,0]
        dGdx_poly[:,:,1,0]=c[:,:,3,1]; dGdx_poly[:,:,1,1]=c[:,:,4,1]
        source_odd_poly=-np.sum(so*dGdx_poly,axis=(2,3))
        source_res=-np.sum(div*G[:,:,:,0],axis=2)
        source_divp=np.gradient(px,h,axis=1,edge_order=2)+np.gradient(py,h,axis=0,edge_order=2)
        closure=source_divp-source_odd-source_res
        return GridFields(self.x,self.y,h,px,py,source_odd,source_odd_poly,source_res,source_divp,closure)

def lp(x,y,p):
    ax=np.abs(x); ay=np.abs(y); r=(ax**p+ay**p)**(1/p)
    gx=np.zeros_like(r); gy=np.zeros_like(r); m=r>1e-14; den=r[m]**(p-1)
    gx[m]=np.sign(x[m])*ax[m]**(p-1)/den; gy[m]=np.sign(y[m])*ay[m]**(p-1)/den
    return r,gx,gy

def weight(g,R,w,p=4,shift=(0,0)):
    r,rx,ry=lp(g.x-shift[0],g.y-shift[1],p); a=R-w/2; b=R+w/2
    q=np.ones_like(r); dq=np.zeros_like(r); q[r>=b]=0; m=(r>a)&(r<b); t=(r[m]-a)/w
    q[m]=.5*(1+np.cos(np.pi*t)); dq[m]=-.5*np.pi/w*np.sin(np.pi*t)
    return q,dq*rx,dq*ry

def pair(a,pas,Ri=4,Ro=8,w=1.5,p=4,shift=(0,0)):
    qi,qix,qiy=weight(a,Ri,w,p,shift); qo,qox,qoy=weight(a,Ro,w,p,shift)
    area=a.step*a.step
    jd=lambda g,qx,qy: -np.sum(g.p_x*qx+g.p_y*qy)*area
    da=(jd(a,qox,qoy)-jd(a,qix,qiy)); dp=(jd(pas,qox,qoy)-jd(pas,qix,qiy)); ex=da-dp
    shell=qo-qi
    ints=lambda arr: np.sum(shell*arr)*area
    return dict(excess=ex,Qodd=ints(a.source_odd),Qodd_poly=ints(a.source_odd_poly),Qres=ints(a.source_residual)-ints(pas.source_residual),Qdiv=ints(a.source_divp)-ints(pas.source_divp),Qclosure=ints(a.closure)-ints(pas.closure),rawa=da,rawp=dp)

if __name__=='__main__':
    nx,ny,a,fit,step=80,56,10,4.2,.3
    _,pf,rp=solve_field(nx,ny,a,0,'right',fit)
    sampler=MLSSampler(pf,9.3,step)
    pg=sampler.apply(pf)
    rows=[]
    for ko in (.05,.1,.15,.2):
        _,af,ra=solve_field(nx,ny,a,ko,'right',fit)
        ag=sampler.apply(af); d=pair(ag,pg); d['ko']=ko; rows.append(d)
    print(json.dumps(rows,indent=2))
