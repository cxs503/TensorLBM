#!/usr/bin/env python
"""XVAL: TensorLBM Shan-Chen vs CFD-GO/TCLB shan-chen-edm-lycett-luo.

Pilot: N=48, 2 radii, 1500 steps. Fixed streaming shifts.
"""
from __future__ import annotations
import json, math, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_C=torch.tensor([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]],dtype=torch.int64)
_W=torch.tensor([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36],dtype=torch.float64)
_CS2=1/3

def eq(rho,ux,uy):
    d=rho.device;dt=rho.dtype;c=_C.to(device=d,dtype=dt);w=_W.to(device=d,dtype=dt).view(9,1,1)
    cx=c[:,0].view(9,1,1);cy=c[:,1].view(9,1,1);cu=cx*ux[None]+cy*uy[None];us=ux**2+uy**2
    return w*rho[None]*(1+3*cu+4.5*cu**2-1.5*us[None])

def stream(f):
    """Periodic streaming: f_i(x,t+1) = f_i(x-c_i, t).
    For velocity (cx,cy), shift by (cy, cx) in (row, col)."""
    o=torch.zeros_like(f)
    # D2Q9: (0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)
    shifts=[(0,0),(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,-1),(-1,1)]
    for i,(dy,dx) in enumerate(shifts):
        o[i]=torch.roll(f[i],shifts=(dy,dx),dims=(0,1))
    return o

def psi_exp(rho,rho0=1): return rho0*(1-torch.exp(-rho/rho0))
def psi_cs(rho,G=5):
    cs2=1/3;a=1;b=0.5;p=rho*cs2+a*rho**2-b*rho**3
    return torch.sqrt(torch.clamp(2*(p-rho*cs2)/(G*cs2),min=1e-12))

def gmls(f):
    d=f.device;dt=f.dtype;n,m=f.shape
    gx=torch.zeros(n,m,device=d,dtype=dt);gy=torch.zeros_like(gx)
    gx+=(1/9)*torch.roll(f,shifts=(0,-1),dims=(0,1));gy+=(1/9)*torch.roll(f,shifts=(-1,0),dims=(0,1))
    gx-=(1/9)*torch.roll(f,shifts=(0,1),dims=(0,1));gy-=(1/9)*torch.roll(f,shifts=(1,0),dims=(0,1))
    gx+=(1/36)*torch.roll(f,shifts=(-1,-1),dims=(0,1));gy+=(1/36)*torch.roll(f,shifts=(-1,-1),dims=(0,1))
    gx-=(1/36)*torch.roll(f,shifts=(-1,1),dims=(0,1));gy+=(1/36)*torch.roll(f,shifts=(-1,1),dims=(0,1))
    gx-=(1/36)*torch.roll(f,shifts=(1,1),dims=(0,1));gy-=(1/36)*torch.roll(f,shifts=(1,1),dims=(0,1))
    gx+=(1/36)*torch.roll(f,shifts=(1,-1),dims=(0,1));gy-=(1/36)*torch.roll(f,shifts=(1,-1),dims=(0,1))
    return gx/_CS2,gy/_CS2

def gtclb(f):
    d=f.device;dt=f.dtype;n,m=f.shape
    gx=torch.zeros(n,m,device=d,dtype=dt);gy=torch.zeros_like(gx)
    gx+=torch.roll(f,shifts=(0,-1),dims=(0,1));gy+=torch.roll(f,shifts=(-1,0),dims=(0,1))
    gx-=torch.roll(f,shifts=(0,1),dims=(0,1));gy-=torch.roll(f,shifts=(1,0),dims=(0,1))
    gx+=.25*torch.roll(f,shifts=(-1,-1),dims=(0,1));gy+=.25*torch.roll(f,shifts=(-1,-1),dims=(0,1))
    gx-=.25*torch.roll(f,shifts=(-1,1),dims=(0,1));gy+=.25*torch.roll(f,shifts=(-1,1),dims=(0,1))
    gx-=.25*torch.roll(f,shifts=(1,1),dims=(0,1));gy-=.25*torch.roll(f,shifts=(1,1),dims=(0,1))
    gx+=.25*torch.roll(f,shifts=(1,-1),dims=(0,1));gy-=.25*torch.roll(f,shifts=(1,-1),dims=(0,1))
    return gx/3,gy/3

def cvs(f,Fx,Fy,tau):
    d=f.device;dt=f.dtype;c=_C.to(device=d,dtype=dt);w=_W.to(device=d,dtype=dt).view(9,1,1)
    cx=c[:,0].view(9,1,1);cy=c[:,1].view(9,1,1);rho=f.sum(0);rs=torch.clamp(rho,min=1e-12)
    ux=(f*cx).sum(0)/rs;uy=(f*cy).sum(0)/rs;xe=ux+tau*Fx/rs;ye=uy+tau*Fy/rs
    cu=cx*xe[None]+cy*ye[None];us=xe**2+ye**2;feq=w*rho[None]*(1+3*cu+4.5*cu**2-1.5*us[None])
    return f-(f-feq)/tau

def cedm(f,Fx,Fy,tau):
    d=f.device;dt=f.dtype;c=_C.to(device=d,dtype=dt);w=_W.to(device=d,dtype=dt).view(9,1,1)
    cx=c[:,0].view(9,1,1);cy=c[:,1].view(9,1,1);rho=f.sum(0);rs=torch.clamp(rho,min=1e-12)
    ux=(f*cx).sum(0)/rs;uy=(f*cy).sum(0)/rs;cu=cx*ux[None]+cy*uy[None];us=ux**2+uy**2
    feq=w*rho[None]*(1+3*cu+4.5*cu**2-1.5*us[None])
    dx=Fx/rs;dy=Fy/rs;A=cx*dx[None]+cy*dy[None];ud=ux*dx+uy*dy;dus=dx**2+dy**2
    delta=w*rho[None]*(3*A+4.5*(2*A*cu+A**2)-1.5*(2*ud+dus)[None])
    return f-(f-feq)/tau+delta

def cguo(f,Fx,Fy,tau):
    d=f.device;dt=f.dtype;c=_C.to(device=d,dtype=dt);w=_W.to(device=d,dtype=dt).view(9,1,1)
    cx=c[:,0].view(9,1,1);cy=c[:,1].view(9,1,1);rho=f.sum(0);rs=torch.clamp(rho,min=1e-12)
    ux=(f*cx).sum(0)/rs;uy=(f*cy).sum(0)/rs;xp=ux+.5*Fx/rs;yp=uy+.5*Fy/rs
    cu=cx*xp[None]+cy*yp[None];us=xp**2+yp**2;feq=w*rho[None]*(1+3*cu+4.5*cu**2-1.5*us[None])
    cf=cx*Fx[None]+cy*Fy[None];uf=(xp*Fx+yp*Fy)[None]
    src=w*(1-.5/tau)*((cf-uf)/_CS2+cu*cf/_CS2**2)
    return f-(f-feq)/tau+src

def ftl(rho,G,pf=psi_exp):
    p=pf(rho);gx,gy=gmls(p); return G*_CS2*p*gx,G*_CS2*p*gy
def ftclb(rho,G,pf=psi_cs):
    p=pf(rho,G);gx,gy=gtclb(p); return G*_CS2*p*gx,G*_CS2*p*gy

def s_tl_vs(f,G,tau): r=f.sum(0);Fx,Fy=ftl(r,G); return stream(cvs(f,Fx,Fy,tau))
def s_tl_edm(f,G,tau): r=f.sum(0);Fx,Fy=ftl(r,G); return stream(cedm(f,Fx,Fy,tau))
def s_tl_guo(f,G,tau): r=f.sum(0);Fx,Fy=ftl(r,G); return stream(cguo(f,Fx,Fy,tau))
def s_tclb_edm(f,G,tau): r=f.sum(0);Fx,Fy=ftclb(r,G); return stream(cedm(f,Fx,Fy,tau))
def s_tclb_guo(f,G,tau): r=f.sum(0);Fx,Fy=ftclb(r,G); return stream(cguo(f,Fx,Fy,tau))

@dataclass
class R:
    label:str;Ri:float;G:float;tau:float;forcing:str;pt:str;ns:int
    rl:float;rv:float;dp:float;Req:float;sig:float;mu:float;me:float;t:float

def run(lab,Ri,G,tau,forcing,pt,ns,sf,pf,dev="cpu"):
    dt=torch.float64;N=48;rl=1.96;rv=0.16
    y=torch.arange(N,device=dev,dtype=dt);x=torch.arange(N,device=dev,dtype=dt)
    Y,X=torch.meshgrid(y,x,indexing='ij');cx,cy=N/2,N/2
    dx=X-cx;dy=Y-cy;dx-=N*torch.round(dx/N);dy-=N*torch.round(dy/N)
    r=torch.sqrt(dx**2+dy**2)
    rho=rv+(rl-rv)*.5*(1+torch.tanh((Ri-r)/3))
    ux=torch.zeros(N,N,device=dev,dtype=dt);uy=torch.zeros_like(ux)
    f=eq(rho,ux,uy);m0=float(f.sum());t0=time.time()
    for _ in range(ns): f=sf(f,G,tau)
    el=time.time()-t0
    rf=f.sum(0);c=_C.to(device=dev,dtype=dt);cx_=c[:,0].view(9,1,1);cy_=c[:,1].view(9,1,1)
    rs=torch.clamp(rf,min=1e-12);uxf=(f*cx_).sum(0)/rs;uyf=(f*cy_).sum(0)/rs
    rc=float(rf.max());rv2=float(rf.min())
    if math.isnan(rc) or math.isnan(rv2):
        return R(label=lab,Ri=Ri,G=G,tau=tau,forcing=forcing,pt=pt,ns=ns,
                 rl=float('nan'),rv=float('nan'),dp=0,Req=0,sig=0,mu=float('nan'),me=0,t=el)
    pc=pf(torch.tensor(rc));pv=pf(torch.tensor(rv2))
    dp=rc*_CS2+G*_CS2/2*pc**2-(rv2*_CS2+G*_CS2/2*pv**2)
    mask=rf>rf.mean();area=float(mask.sum());Req=math.sqrt(area/math.pi) if area>0 else 0
    sig=dp*Req if Req>0 else 0;mu=float(torch.sqrt(uxf**2+uyf**2).max());me=abs(float(f.sum())-m0)/m0
    return R(label=lab,Ri=Ri,G=G,tau=tau,forcing=forcing,pt=pt,ns=ns,rl=rc,rv=rv2,dp=dp,Req=Req,sig=sig,mu=mu,me=me,t=el)

def main():
    dev="cpu";radii=[12,18];G=5;tau=1;ns=1500;results=[]
    cases=[("TL_vs",s_tl_vs,psi_exp,"vs","exp"),("TL_edm",s_tl_edm,psi_exp,"edm","exp"),
           ("TL_guo",s_tl_guo,psi_exp,"guo","exp"),("TCLB_edm",s_tclb_edm,psi_cs,"edm","cs"),
           ("TCLB_guo",s_tclb_guo,psi_cs,"guo","cs")]
    print("="*70);print("XVAL: TensorLBM SC vs TCLB (pilot, N=48, 1500 steps)");print("="*70)
    for Ri in radii:
        for pre,sf,pf,forcing,pt in cases:
            lab=f"{pre}_R{Ri}";print(f"\n--- {lab} ---")
            r=run(lab,Ri,G,tau,forcing,pt,ns,sf,pf,dev)
            rl_s=f"{r.rl:.4f}" if not math.isnan(r.rl) else "nan"
            print(f"  rl={rl_s} rv={r.rv:.4f} dp={r.dp:.6f} sig={r.sig:.6f} u={r.mu:.2e} t={r.t:.1f}s")
            results.append(r)

    print("\n"+"="*70);print("TABLE");print("="*70)
    print(f"{'Label':<16} {'rl':>7} {'rv':>7} {'dp':>9} {'sig':>9} {'u_max':>9}")
    print("-"*60)
    for r in results:
        rl_s=f"{r.rl:.4f}" if not math.isnan(r.rl) else "  nan"
        rv_s=f"{r.rv:.4f}" if not math.isnan(r.rv) else "  nan"
        mu_s=f"{r.mu:.2e}" if not math.isnan(r.mu) else "  nan"
        print(f"{r.label:<16} {rl_s:>7} {rv_s:>7} {r.dp:>9.6f} {r.sig:>9.6f} {mu_s:>9}")

    def fit(cases):
        valid=[c for c in cases if c.Req>0 and not math.isnan(c.dp)]
        if len(valid)<2: return None,None
        iR=[1/c.Req for c in valid];dp=[c.dp for c in valid]
        A=np.array(iR).reshape(-1,1);b=np.array(dp);s=float(np.linalg.lstsq(A,b,rcond=None)[0][0])
        p=s*np.array(iR);ssr=np.sum((b-p)**2);sst=np.sum((b-np.mean(b))**2)
        return s,1-ssr/sst if sst>0 else 0

    groups={"TL_vs":[r for r in results if "TL_vs" in r.label],"TL_edm":[r for r in results if "TL_edm" in r.label],
            "TL_guo":[r for r in results if "TL_guo" in r.label],"TCLB_edm":[r for r in results if "TCLB_edm" in r.label],
            "TCLB_guo":[r for r in results if "TCLB_guo" in r.label]}
    print("\n--- Laplace Fit ---");fits={}
    for n,c in groups.items():
        s,r2=fit(c);fits[n]={"sigma":s,"r2":r2}
        if s is not None: print(f"  {n}: σ={s:.6f} R²={r2:.4f}")
        else: print(f"  {n}: insufficient data")

    ref={r.Ri:r for r in results if "TCLB_edm" in r.label and not math.isnan(r.rl)}
    print("\n--- Error vs TCLB_EDM ---")
    for r in results:
        if "TCLB" in r.label: continue
        rr=ref.get(r.Ri)
        if rr and not math.isnan(r.rl):
            es=abs(r.sig-rr.sig)/max(abs(rr.sig),1e-12)*100
            el=abs(r.rl-rr.rl)/max(rr.rl,1e-12)*100
            print(f"  {r.label}: σ err={es:.1f}% [{'PASS' if es<15 else 'FAIL'}] rl err={el:.1f}%")

    ad=Path(__file__).parent/"artifacts";ad.mkdir(parents=True,exist_ok=True)
    cols={"TL_vs":"blue","TL_edm":"green","TL_guo":"cyan","TCLB_edm":"red","TCLB_guo":"magenta"}
    mks={"TL_vs":"o","TL_edm":"s","TL_guo":"D","TCLB_edm":"^","TCLB_guo":"v"}
    fig,ax=plt.subplots(figsize=(7,5))
    for r in results:
        if math.isnan(r.Req) or r.Req==0: continue
        pre=r.label.rsplit("_R",1)[0]
        ax.scatter(1/r.Req,r.dp,color=cols.get(pre,"gray"),marker=mks.get(pre,"o"),s=60)
    for n,f in fits.items():
        if f["sigma"]:
            x=np.linspace(.03,.12,100);ax.plot(x,f["sigma"]*x,"--",color=cols.get(n,"gray"),label=f"{n}: σ={f['sigma']:.4f}")
    ax.set_xlabel("1/R");ax.set_ylabel("Δp");ax.set_title("Laplace Law");ax.legend(fontsize=7);ax.grid(True,alpha=.3)
    fig.savefig(ad/"fig1_laplace.png",dpi=150,bbox_inches="tight");plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,5))
    for r in results:
        if math.isnan(r.mu): continue
        pre=r.label.rsplit("_R",1)[0]
        ax.semilogy(r.Ri,r.mu,marker=mks.get(pre,"o"),color=cols.get(pre,"gray"))
    ax.set_xlabel("R");ax.set_ylabel("|u|_max");ax.set_title("Spurious Current");ax.grid(True,alpha=.3)
    fig.savefig(ad/"fig2_spurious.png",dpi=150,bbox_inches="tight");plt.close(fig)

    rd=[asdict(r) for r in results];rd.append({"fits":fits})
    with open(ad/"xval_results.json","w") as fp: json.dump(rd,fp,indent=2)
    print(f"\nResults saved to {ad}/xval_results.json")

    print("\n"+"="*70);print("VERDICT");print("="*70)
    ok=True
    for r in results:
        if "TCLB" in r.label: continue
        rr=ref.get(r.Ri)
        if rr and not math.isnan(r.rl):
            es=abs(r.sig-rr.sig)/max(abs(rr.sig),1e-12)*100
            if es>15: ok=False
    print(f"  {'PASS' if ok else 'FAIL'}: TensorLBM {'within' if ok else 'exceeds'} 15% of TCLB-EDM reference")
    print("  Note: Different EOS (exp vs CS) yield different coexistence curves;")
    print("  the forcing scheme comparison (vs/edm/guo) is the key discriminant.")

if __name__=="__main__":
    main()
