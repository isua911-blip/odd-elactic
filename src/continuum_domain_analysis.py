#!/usr/bin/env python3
import sys,csv,json,math,time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from apparent_j_analysis import solve_field,fit_through_origin,keyhole_j
from continuum_domain_core import MLSSampler,pair,weight
OUT = PACKAGE_ROOT / 'data' / 'continuum_domain_results'
OUT.mkdir(parents=True, exist_ok=True)

def write(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def jd(g,qx,qy): return float(-np.sum(g.p_x*qx+g.p_y*qy)*g.step*g.step)

nx,ny,a,fit,step=80,56,10.0,4.2,0.1
half=9.5; kos=(.025,.05,.075,.1,.125,.15,.175,.2)
_,pf,rp=solve_field(nx,ny,a,0,'right',fit); sampler=MLSSampler(pf,half,step); pg=sampler.apply(pf)
active={}; fields={}
for ko in kos:
 _,af,ra=solve_field(nx,ny,a,ko,'right',fit); fields[ko]=af; active[ko]=sampler.apply(af)
main=[]
for ko in kos:
 d=pair(active[ko],pg,4,8,1.5,4)
 row={'k_o':ko,**{k:float(v) for k,v in d.items()}}
 row['coarse_graining_remainder']=row['Qdiv']-row['Qodd']
 row['odd_fraction_of_total_source']=row['Qodd']/row['Qdiv']
 row['relative_Qodd_mismatch']=abs(row['excess']-row['Qodd'])/abs(row['Qodd'])
 row['relative_total_source_mismatch']=abs(row['excess']-row['Qdiv'])/abs(row['Qdiv'])
 main.append(row)
write('domain_main_balance.csv',main)
# radius curves and adjacent shells
radius_rows=[]; shell_rows=[]
radii=(4.,5.,6.,7.,8.)
for ko in (0.,.05,.1,.15,.2):
 g=pg if ko==0 else active[ko]
 for R in radii:
  q,qx,qy=weight(g,R,1.5,4)
  radius_rows.append({'k_o':ko,'R':R,'J_domain':jd(g,qx,qy)})
 if ko>0:
  for ri,ro in zip(radii[:-1],radii[1:]):
   d=pair(g,pg,ri,ro,1.5,4)
   shell_rows.append({'k_o':ko,'R_inner':ri,'R_outer':ro,'R_center':.5*(ri+ro),**{k:float(v) for k,v in d.items()}})
write('domain_radius_scan.csv',radius_rows);write('domain_adjacent_shells.csv',shell_rows)
# kernel shape/width and center shift at ko=.15
rob=[]; g=active[.15]
for p in (2.,4.,8.,16.):
 for w in (.75,1.,1.5,2.,2.5):
  d=pair(g,pg,4,8,w,p)
  rob.append({'scan':'shape_width','kernel_p':p,'width':w,'shift_x':0.,'shift_y':0.,**{k:float(v) for k,v in d.items()}})
for sx in (-.3,-.15,0.,.15,.3):
 for sy in (-.3,-.15,0.,.15,.3):
  d=pair(g,pg,4,8,1.5,4,(sx,sy))
  rob.append({'scan':'center_shift','kernel_p':4.,'width':1.5,'shift_x':sx,'shift_y':sy,**{k:float(v) for k,v in d.items()}})
write('domain_kernel_robustness.csv',rob)
# Keyhole comparison, using positive ko 0.05..0.2
pj4=keyhole_j(pf,4,.1); pj8=keyhole_j(pf,8,.1); pd=pj8-pj4
kh=[]
for ko in (.05,.1,.15,.2):
 af=fields[ko]; kd=(keyhole_j(af,8,.1)-keyhole_j(af,4,.1))-pd
 row=next(r for r in main if r['k_o']==ko)
 kh.append({'k_o':ko,'keyhole_excess':kd,'domain_excess':row['excess'],'domain_Qodd':row['Qodd'],'domain_Qdiv':row['Qdiv']})
write('domain_vs_keyhole.csv',kh)
# regressions
for source in ('Qodd','Qdiv'):
 x=np.array([r[source] for r in main]);y=np.array([r['excess'] for r in main]);print(source,fit_through_origin(x,y))
# figures
x=np.array([r['Qodd'] for r in main]);y=np.array([r['excess'] for r in main]);z=np.array([r['Qdiv'] for r in main])
so,r2o=fit_through_origin(x,y);st,r2t=fit_through_origin(z,y)
fig,ax=plt.subplots(figsize=(6.2,4.8));ax.plot(x,y,'o',label='odd source only');ax.plot(z,y,'s',label='total reconstructed source');lim=1.08*max(abs(y).max(),abs(z).max());l=np.linspace(-lim,0,100);ax.plot(l,l,'--',label='unit slope');ax.set_xlabel('weighted configurational source');ax.set_ylabel('active excess smooth-domain drift');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/'domain_source_closure.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(6.5,4.6));
for ko in (0.,.05,.1,.15,.2):
 rr=[r for r in radius_rows if r['k_o']==ko];ax.plot([r['R'] for r in rr],[r['J_domain'] for r in rr],'o-',label=f'$k_o={ko:.2f}$')
ax.set_xlabel('smooth-domain radius R');ax.set_ylabel('$J_D[q_R]$');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(OUT/'domain_J_vs_radius.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(6.5,4.6));k=np.array([r['k_o'] for r in kh]);ax.plot(k,[r['keyhole_excess'] for r in kh],'o-',label='keyhole');ax.plot(k,[r['domain_excess'] for r in kh],'s-',label='smooth domain');ax.plot(k,[r['domain_Qodd'] for r in kh],'x--',label='odd source');ax.plot(k,[r['domain_Qdiv'] for r in kh],'+--',label='total source');ax.set_xlabel('$k_o$');ax.set_ylabel('active broad-shell contribution');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(OUT/'domain_vs_keyhole.pdf');plt.close(fig)
summary={'nominal':{'nx':nx,'ny':ny,'a':a,'fit_radius':fit,'step':step,'R_inner':4,'R_outer':8,'width':1.5,'p':4},'Qodd_fit':{'slope':so,'r2':r2o},'Qtotal_fit':{'slope':st,'r2':r2t},'passive_domain_relative_spread':(max(r['J_domain'] for r in radius_rows if r['k_o']==0)-min(r['J_domain'] for r in radius_rows if r['k_o']==0))/np.mean([r['J_domain'] for r in radius_rows if r['k_o']==0]),'kernel_ratio_range':{},'center_shift_ratio_range':{}}
for scan in ('shape_width','center_shift'):
 rr=[r for r in rob if r['scan']==scan];rat=np.array([r['excess']/r['Qodd'] for r in rr]);mis=np.array([abs(r['excess']-r['Qdiv'])/abs(r['Qdiv']) for r in rr]);summary[scan]={'odd_ratio_min':float(rat.min()),'odd_ratio_max':float(rat.max()),'odd_ratio_mean':float(rat.mean()),'total_source_mismatch_max':float(mis.max()),'count':len(rr)}
(OUT/'nominal_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
