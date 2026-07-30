#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from discrete_configurational_analysis import (
    build_triangles, affine_gradient, localize_coords, q_value, AREA_TRI, tip_geometry
)
from active_tip_scan import ActiveCrackedStrip
from lattice_baselines import R90

OUT = PACKAGE_ROOT / 'data' / 'localization_gauge_results'
OUT.mkdir(parents=True, exist_ok=True)

def partition_weights(model, triangles, alpha: float, axis: str='y'):
    adj=defaultdict(list)
    for ti,t in enumerate(triangles):
        for bid in t.bond_ids: adj[bid].append(ti)
    weights={}
    for bid,tis in adj.items():
        if len(tis)==1:
            weights[(tis[0],bid)]=1.0
        elif len(tis)==2:
            vals=[]
            for ti in tis:
                c=np.mean(triangles[ti].coords,axis=0)
                vals.append(c[1] if axis=='y' else c[0])
            # periodic x makes x ordering ambiguous; y is the primary audit.
            hi=0 if vals[0]>=vals[1] else 1
            lo=1-hi
            weights[(tis[hi],bid)]=alpha
            weights[(tis[lo],bid)]=1-alpha
        else:
            raise RuntimeError((bid,len(tis)))
    return weights

def fields_weighted(model,u,triangles,tip,support,weights):
    active=set(range(len(model.all_bonds)))-set(model.removed_ids)
    _,_,direction=tip_geometry(model,tip); S=np.diag([direction,1.0])
    out=[]; xnodes=[]
    for ti,tri in enumerate(triangles):
        xloc=localize_coords(tri.coords,model,tip); c=np.mean(xloc,axis=0)
        if np.max(np.abs(c))>support+2: continue
        uloc=(S@u[np.asarray(tri.nodes)].T).T
        H=affine_gradient(xloc,uloc)
        se=np.zeros((2,2)); so=np.zeros((2,2)); W=0.0
        for bid in tri.bond_ids:
            if bid not in active: continue
            w=weights[(ti,bid)]
            b=model.all_bonds[bid]; n=S@b.n; t=R90@n
            du=S@(u[b.j]-u[b.i]); ext=float(du@n)
            fac=w/AREA_TRI
            se += fac*model.k*ext*np.outer(n,n)
            so += fac*(-direction*model.k_o)*ext*np.outer(t,n)
            W += fac*0.5*model.k*ext*ext
        out.append((H,se,so,W)); xnodes.append(xloc)
    return out,xnodes

def J(fields,xnodes,R,width=1.5,p=4.0):
    te=to=0.0
    for (H,se,so,W),xn in zip(fields,xnodes):
        qnod=np.array([q_value(x,R,width,p) for x in xn])[:,None]
        gq=affine_gradient(xn,qnod)[0]; ux=H[:,0]
        Pe=np.array([W,0.0])-se.T@ux; Po=-so.T@ux
        te += -AREA_TRI*float(Pe@gq); to += -AREA_TRI*float(Po@gq)
    return te,to,te+to

def force_repro(model,u,triangles,weights):
    active=set(range(len(model.all_bonds)))-set(model.removed_ids)
    rr=np.zeros_like(u)
    for ti,tri in enumerate(triangles):
        se=np.zeros((2,2)); so=np.zeros((2,2))
        for bid in tri.bond_ids:
            if bid not in active:continue
            w=weights[(ti,bid)]; b=model.all_bonds[bid]; n=b.n;t=R90@n
            ext=float((u[b.j]-u[b.i])@n); fac=w/AREA_TRI
            se+=fac*model.k*ext*np.outer(n,n);so+=fac*(-model.k_o)*ext*np.outer(t,n)
        B=np.column_stack([np.ones(3),tri.coords[:,0],tri.coords[:,1]])
        C=np.linalg.inv(B)
        for a,node in enumerate(tri.nodes): rr[node]+=AREA_TRI*(se+so)@C[1:,a]
    modelres=(model.K@u.ravel()).reshape(-1,2)
    mask=np.ones(model.n_nodes,bool)
    for i in range(model.nx):mask[model.node_id(i,0)]=False;mask[model.node_id(i,model.ny-1)]=False
    return float(np.max(np.abs((rr-modelres)[mask])))

def solve_case(ko):
    m=ActiveCrackedStrip(80,56,10,1.0,ko);u,_,_=m.solve(1.0);tr=build_triangles(m)
    return m,u,tr

pm,pu,pt=solve_case(0.0); am,au,at=solve_case(0.15)
assert len(pt)==len(at)
rows=[]
for alpha in [0.0,0.1,0.25,0.5,0.75,0.9,1.0]:
    pw=partition_weights(pm,pt,alpha,'y'); aw=partition_weights(am,at,alpha,'y')
    pf,px=fields_weighted(pm,pu,pt,'right',9.5,pw); af,ax=fields_weighted(am,au,at,'right',9.5,aw)
    p4=J(pf,px,4);p8=J(pf,px,8);a4=J(af,ax,4);a8=J(af,ax,8)
    ex=[(a8[i]-a4[i])-(p8[i]-p4[i]) for i in range(3)]
    rows.append({'alpha_upper':alpha,'passive_J4':p4[2],'passive_J8':p8[2],
                 'passive_shell':p8[2]-p4[2],
                 'active_excess_even':ex[0], 'active_excess_odd':(a8[1]-a4[1]),
                 'active_excess_total':ex[2],
                 'odd_fraction':(a8[1]-a4[1])/ex[2],
                 'force_repro_passive':force_repro(pm,pu,pt,pw),
                 'force_repro_active':force_repro(am,au,at,aw)})
with (OUT/'triangle_partition_gauge.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
summary={
 'total_excess_range':[min(r['active_excess_total'] for r in rows),max(r['active_excess_total'] for r in rows)],
 'total_excess_relative_span':(max(r['active_excess_total'] for r in rows)-min(r['active_excess_total'] for r in rows))/abs(np.mean([r['active_excess_total'] for r in rows])),
 'odd_fraction_range':[min(r['odd_fraction'] for r in rows),max(r['odd_fraction'] for r in rows)],
 'passive_shell_range':[min(r['passive_shell'] for r in rows),max(r['passive_shell'] for r in rows)],
 'max_force_repro_error':max(max(r['force_repro_passive'],r['force_repro_active']) for r in rows)
}
(OUT/'gauge_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

# exact elementwise volume-to-edge identity for selected gauges
edge_rows=[]
for alpha in [0.0,0.5,1.0]:
    aw=partition_weights(am,at,alpha,'y')
    af,ax=fields_weighted(am,au,at,'right',9.5,aw)
    for R in [4.0,8.0]:
        vol=bdry=0.0
        for (H,se,so,W),xn in zip(af,ax):
            qn=np.array([q_value(x,R,1.5,4.0) for x in xn])
            gq=affine_gradient(xn,qn[:,None])[0]
            ux=H[:,0]; P=np.array([W,0.0])-(se+so).T@ux
            vol += -AREA_TRI*float(P@gq)
            # triangle nodes are counter-clockwise
            for e in range(3):
                a=e; b=(e+1)%3
                d=xn[b]-xn[a]; L=float(np.linalg.norm(d))
                nout=np.array([d[1],-d[0]])/L
                bdry += -L*0.5*(qn[a]+qn[b])*float(P@nout)
        edge_rows.append({'alpha_upper':alpha,'R':R,'volume_J':vol,'element_boundary_J':bdry,'abs_error':abs(vol-bdry)})
with (OUT/'edge_jump_identity.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=edge_rows[0].keys());w.writeheader();w.writerows(edge_rows)
print('edge max error',max(r['abs_error'] for r in edge_rows))
