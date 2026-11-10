#!/usr/bin/env python3
"""Quick ice-edge campaign for local execution."""
import json, hashlib, time
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorlbm.d2q9 import C, W, OPPOSITE

def run(nx, ny, u0, re, ice_frac, ls, nsteps, label=''):
    device = torch.device('cpu')
    nu = u0*(ny-2)/re; tau = 3*nu+0.5; omega = 1/tau
    xe = int(nx*ice_frac); beta = ls/(1+ls)
    c = C.to(device).float(); w = W.to(device).float(); opp = OPPOSITE.to(device)
    my = [0,1,4,3,2,8,7,6,5]
    rho = torch.ones(ny,nx,device=device)
    ux = torch.full((ny,nx),u0,device=device)
    uy = torch.zeros(ny,nx,device=device)
    usq = ux**2+uy**2
    cu = c[:,0].view(9,1,1)*ux.unsqueeze(0)+c[:,1].view(9,1,1)*uy.unsqueeze(0)
    f = w.view(9,1,1)*rho.unsqueeze(0)*(1+3*cu+4.5*cu**2-1.5*usq.unsqueeze(0))
    us = torch.zeros_like(ux); na = 0
    for s in range(1,nsteps+1):
        rho=f.sum(0); jx=(f*c[:,0].view(9,1,1)).sum(0); jy=(f*c[:,1].view(9,1,1)).sum(0)
        ux=jx/rho; uy=jy/rho
        usq=ux**2+uy**2; cu=c[:,0].view(9,1,1)*ux.unsqueeze(0)+c[:,1].view(9,1,1)*uy.unsqueeze(0)
        feq=w.view(9,1,1)*rho.unsqueeze(0)*(1+3*cu+4.5*cu**2-1.5*usq.unsqueeze(0))
        f=f-omega*(f-feq)
        fn=torch.empty_like(f)
        for q in range(9):
            cx,cy=int(C[q,0]),int(C[q,1]); src=f[q]
            if cy>0:
                t=torch.zeros_like(src);t[cy:,:]=src[:ny-cy,:];src=t
            elif cy<0:
                t=torch.zeros_like(src);t[:ny+cy,:]=src[-cy:,:];src=t
            if cx!=0: src=torch.roll(src,shifts=cx,dims=1)
            fn[q]=src
        f=fn
        for q in range(9):
            oq=int(opp[q])
            if q!=oq: f[q,0,:]=f[oq,0,:]
        for q in range(9):
            oq=int(opp[q]); mq=my[q]
            if C[q,1]<0:
                f[q,ny-1,:xe]=(1-beta)*f[oq,ny-1,:xe]+beta*f[mq,ny-1,:xe]
                f[q,ny-1,xe:]=f[mq,ny-1,xe:]
        if s>nsteps//2 and s%2==0:
            rho=f.sum(0);jx=(f*c[:,0].view(9,1,1)).sum(0);us+=jx/rho;na+=1
    um=us/na if na>0 else jx/rho
    tb=nu*(um[1,:]-um[0,:]); tt=nu*(um[-1,:]-um[-2,:])
    fb=tb.abs().sum().item();fi=tt[:xe].abs().sum().item();fo=tt[xe:].abs().sum().item()
    cd=(fb+fi+fo)/(0.5*u0**2*(ny-2))
    print(f'  [{label}] C_D={cd:.4f}')
    return {'u':um.numpy(),'tb':tb.numpy(),'tt':tt.numpy(),'C_D':cd,'C_Di':fi/(0.5*u0**2*(ny-2))}

out=Path('runs/ice_edge_zone/quick'); out.mkdir(parents=True,exist_ok=True)
fd=out/'figures'; fd.mkdir(exist_ok=True)

print('=== ICE-EDGE QUICK CAMPAIGN ===')
b=run(150,30,0.05,100,0.5,2.0,1000,'base')
m=[]
for nx,ny,l in [(80,16,'C'),(150,30,'M'),(300,60,'F')]:
    r=run(nx,ny,0.05,100,0.5,2.0,800,l); m.append({'l':l,'C_D':r['C_D']})
rs=[]
for re in [50,100,200]:
    r=run(120,24,0.05,re,0.5,2.0,800,f'Re{re}'); rs.append({'Re':re,'C_D':r['C_D']})
ss=[]
for ls in [0.0,1.0,2.0,5.0]:
    r=run(120,24,0.05,100,0.5,ls,800,f'Ls{ls}'); ss.append({'Ls':ls,'C_D':r['C_D']})

# Figures
xe=75
fig,ax=plt.subplots(figsize=(10,3))
ax.imshow(b['u'],origin='lower',cmap='viridis',aspect='auto')
ax.axvline(x=xe,color='r',ls='--',lw=2)
ax.set_title('Velocity field'); plt.colorbar(ax.images[0],label='u')
fig.savefig(fd/'fig1.png',dpi=120,bbox_inches='tight'); plt.close(fig)

fig,ax=plt.subplots(figsize=(10,3))
ax.plot(b['tb'],'b-'); ax.axvline(x=xe,color='r',ls='--')
ax.set_title('Bottom wall shear'); ax.grid(True,alpha=0.3)
fig.savefig(fd/'fig2.png',dpi=120,bbox_inches='tight'); plt.close(fig)

fig,ax=plt.subplots(figsize=(6,4))
ax.bar([x['l'] for x in m],[x['C_D'] for x in m])
ax.set_ylabel('C_D'); ax.set_title('Mesh independence')
fig.savefig(fd/'fig3.png',dpi=120,bbox_inches='tight'); plt.close(fig)

fig,ax=plt.subplots(figsize=(6,4))
ax.loglog([x['Re'] for x in rs],[x['C_D'] for x in rs],'o-')
ax.set_xlabel('Re'); ax.set_ylabel('C_D'); ax.set_title('C_D vs Re'); ax.grid(True,alpha=0.3)
fig.savefig(fd/'fig4.png',dpi=120,bbox_inches='tight'); plt.close(fig)

fig,ax=plt.subplots(figsize=(6,4))
ax.semilogx([x['Ls'] for x in ss[1:]],[x['C_D'] for x in ss[1:]],'^-')
ax.plot([ss[0]['Ls']],[ss[0]['C_D']],'k^')
ax.set_xlabel('Ls'); ax.set_ylabel('C_D'); ax.set_title('C_D vs slip length'); ax.grid(True,alpha=0.3)
fig.savefig(fd/'fig5.png',dpi=120,bbox_inches='tight'); plt.close(fig)

h=hashlib.sha256()
for p in fd.glob('*.png'): h.update(p.read_bytes())
sha=h.hexdigest()

rpt={'baseline_C_D':b['C_D'],'mesh':m,'re':rs,'slip':ss,'sha256':sha}
with open(out/'report.json','w') as f: json.dump(rpt,f,indent=2)

print(f'\nBASELINE C_D={b["C_D"]:.4f}')
print(f'SHA256={sha[:16]}...')
print('DONE')
