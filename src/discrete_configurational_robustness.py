import sys,csv,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from discrete_configurational_analysis import prepare_model_fields,discrete_J,fit_origin,write_csv,build_triangles,AREA_TRI
from active_tip_scan import ActiveCrackedStrip
from lattice_baselines import R90
OUT = PACKAGE_ROOT / 'data' / 'discrete_configurational_results'
OUT.mkdir(parents=True, exist_ok=True)
# chirality
radii=(4.,5.,6.,7.,8.); rows=[]
for ko in (.05,.1,.15,.2):
 _,_,rf,rx,_=prepare_model_fields(80,56,10,ko,'right',9.5)
 _,_,lf,lx,_=prepare_model_fields(80,56,10,-ko,'left',9.5)
 for R in radii:
  jr=discrete_J(rf,rx,R);jl=discrete_J(lf,lx,R)
  rows.append({'abs_k_o':ko,'R':R,'right_total':jr['J_total'],'left_total':jl['J_total'],'right_odd':jr['J_odd'],'left_odd':jl['J_odd'],'abs_error_total':abs(jr['J_total']-jl['J_total']),'abs_error_odd':abs(jr['J_odd']-jl['J_odd'])})
write_csv(OUT/'discrete_chirality_mirror.csv',rows)
# size/scale scan
sizes=[(64,48,8.,4.,8.,9.5),(80,56,10.,4.,8.,9.5),(96,72,12.,6.,12.,13.5),(128,96,16.,8.,16.,17.5)]
srows=[]
for nx,ny,a,ri,ro,sup in sizes:
 _,_,pf,px,_=prepare_model_fields(nx,ny,a,0,'right',sup)
 _,_,af,ax,_=prepare_model_fields(nx,ny,a,.15,'right',sup)
 pi=discrete_J(pf,px,ri);po=discrete_J(pf,px,ro)
 ai=discrete_J(af,ax,ri);ao=discrete_J(af,ax,ro)
 et=(ao['J_total']-ai['J_total'])-(po['J_total']-pi['J_total'])
 ee=(ao['J_even']-ai['J_even'])-(po['J_even']-pi['J_even'])
 oo=ao['J_odd']-ai['J_odd']
 srows.append({'nx':nx,'ny':ny,'a':a,'R_inner':ri,'R_outer':ro,'excess_total':et,'even_redistribution':ee,'direct_odd':oo,'odd_fraction':oo/et,'even_fraction':ee/et,'closure':et-ee-oo})
write_csv(OUT/'discrete_scale_scan.csv',srows)
# exact nodal force reproduction check
m=ActiveCrackedStrip(40,30,6,1,.15);u,_,_=m.solve(1.0);tris=build_triangles(m);active=set(range(len(m.all_bonds)))-set(m.removed_ids);rr=np.zeros_like(u)
for tri in tris:
 se=np.zeros((2,2));so=np.zeros((2,2))
 for bid in tri.bond_ids:
  if bid not in active:continue
  b=m.all_bonds[bid];n=b.n;t=R90@n;ext=(u[b.j]-u[b.i])@n;fac=.5/AREA_TRI
  se+=fac*m.k*ext*np.outer(n,n);so+=fac*(-m.k_o)*ext*np.outer(t,n)
 B=np.column_stack([np.ones(3),tri.coords[:,0],tri.coords[:,1]]);C=np.linalg.inv(B)
 for a,node in enumerate(tri.nodes):rr[node]+=AREA_TRI*(se+so)@C[1:,a]
model=(m.K@u.ravel()).reshape(-1,2);mask=np.ones(m.n_nodes,bool)
for i in range(m.nx):mask[m.node_id(i,0)]=False;mask[m.node_id(i,m.ny-1)]=False
force_err=float(np.max(np.abs((rr-model)[mask])))
# figures from existing
b=pd.read_csv(OUT/'discrete_configurational_balance.csv');rad=pd.read_csv(OUT/'discrete_domain_radius_scan.csv');rob=pd.read_csv(OUT/'discrete_weight_robustness.csv')
fig,ax=plt.subplots(figsize=(6.4,4.6))
for ko in (0,.05,.1,.15,.2):
 d=rad[np.isclose(rad.k_o,ko)];ax.plot(d.R,d.J_total,'o-',label=fr'$k_o={ko:.2f}$')
ax.set_xlabel('discrete domain radius $R$');ax.set_ylabel('$J_h[q_R]$');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(OUT/'discrete_J_vs_radius.pdf');plt.close(fig)
odd=b.direct_odd_term.to_numpy();tot=b.excess_total.to_numpy();ev=b.active_change_even.to_numpy();so,r2o=fit_origin(odd,tot);ss,r2s=fit_origin(odd+ev,tot)
fig,ax=plt.subplots(figsize=(6.4,4.8));ax.plot(odd,tot,'o',label='direct odd bond term');ax.plot(odd+ev,tot,'s',label='odd + even-field redistribution');lim=1.08*max(abs(tot).max(),abs((odd+ev)).max());xx=np.linspace(-lim,lim,200);ax.plot(xx,xx,'--',label='unit slope');ax.set_xlabel('discrete configurational contribution');ax.set_ylabel('active excess domain drift');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(OUT/'discrete_source_decomposition.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(6.4,4.6));ax.plot([r['nx'] for r in srows],[100*abs(r['even_fraction']) for r in srows],'o-');ax.set_xlabel('lattice width $N_x$');ax.set_ylabel('magnitude of even-field redistribution (%)');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/'discrete_scale_remainder.pdf');plt.close(fig)
passive=rad[np.isclose(rad.k_o,0)].J_total.to_numpy();summary={'passive_relative_spread':float((passive.max()-passive.min())/abs(passive.mean())),'odd_only_fit':{'slope':so,'r2_origin':r2o},'exact_decomposition_fit':{'slope':ss,'r2_origin':r2s},'max_algebraic_closure_abs':float(abs(b.algebraic_closure).max()),'odd_fraction_range':[float(b.odd_fraction.min()),float(b.odd_fraction.max())],'even_redistribution_fraction_range':[float(b.even_redistribution_fraction.min()),float(b.even_redistribution_fraction.max())],'robustness_odd_fraction_range':[float(rob.odd_fraction.min()),float(rob.odd_fraction.max())],'robustness_max_closure_abs':float(abs(rob.closure).max()),'chirality_max_total_abs_error':max(r['abs_error_total'] for r in rows),'chirality_max_odd_abs_error':max(r['abs_error_odd'] for r in rows),'interior_nodal_force_reproduction_max_abs_error':force_err,'scale_scan':srows}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
