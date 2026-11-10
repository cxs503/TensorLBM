import sys, time
sys.path.insert(0, "/root/tp-agent/work/m8675/src")
import torch
from tensorlbm.d2q9 import equilibrium, C
from tensorlbm.multiphase import collide_sc_single_component, make_psi_eos
from tensorlbm.sc_eos_coexistence import maxwell_coexistence
from tensorlbm.solver import stream

CS2=1/3
TC=0.07291903717301021
def mw(Tr):
    r = maxwell_coexistence(+1.0, make_psi_eos("pr", Tr*TC), rho_lo=1e-6, rho_hi=0.98*21/2, n=12000)
    return r
def init(L,R,rl,rv,w=4.0):
    ys=torch.arange(L,dtype=torch.float32); yy,xx=torch.meshgrid(ys,ys,indexing="ij")
    r=torch.sqrt((xx-L/2)**2+(yy-L/2)**2)
    rho=rv+0.5*(rl-rv)*(1+torch.tanh((R-r)/w))
    return equilibrium(rho,torch.zeros_like(rho),torch.zeros_like(rho))

def run(Tr,forcing,L,R,steps=8000,sample=200,tau=1.0):
    r=mw(Tr); T=Tr*TC; psi=make_psi_eos("pr",T); G=1.0
    f=init(L,R,r.rho_liquid,r.rho_vapour)
    ys=torch.arange(L,dtype=torch.float32); yy,xx=torch.meshgrid(ys,ys,indexing="ij")
    rr=torch.sqrt((xx-L/2)**2+(yy-L/2)**2)
    far=rr>=0.80*L/2; core0=rr<=0.25*L/2
    mass0=float((f.sum(dim=0)).sum())
    t0=time.time(); ok=True; msg=None; hist=[]
    for s in range(1,steps+1):
        f=stream(collide_sc_single_component(f,G=G,tau=tau,psi_fn=psi,forcing=forcing))
        if s % sample: continue
        rho=f.sum(dim=0)
        if not torch.isfinite(rho).all() or float(rho.min())<0:
            ok=False; msg=f"bad at {s}"; break
        rl=float(rho[core0].mean()); rv=float(rho[far].mean())
        for _ in range(2):
            reff=float(torch.sqrt(torch.clamp((rho-rv).sum(),min=0)/(3.14159265*max(rl-rv,1e-12))))
            if reff<=0: break
            core=rr<=0.5*reff
            rl=float(rho[core].mean()); rv=float(rho[far].mean())
        momx=(f*C[:,0].view(9,1,1)).sum(0); momy=(f*C[:,1].view(9,1,1)).sum(0)
        u=torch.sqrt((momx/rho.clamp(min=1e-12))**2+(momy/rho.clamp(min=1e-12))**2)
        hist.append(dict(step=s,rho_l=rl,rho_v=rv,reff=reff,umax=float(u.max()),umax_int=float(u[(rr>0.5*reff)&(rr<reff+6)].max()) if reff>0 else None, rhomax=float(rho.max()), massdrift=abs(float(rho.sum())-mass0)/mass0))
    dt=time.time()-t0
    tail=hist[-5:]
    print(f"Tr={Tr:.3f} ratio_mw={r.density_ratio:.1f} forcing={forcing} L={L} T={dt:.1f}s ({dt/steps*1000:.2f}ms/step) stable={ok} msg={msg} steps={s} "
          f"rl={tail[-1]['rho_l']:.4f} rv={tail[-1]['rho_v']:.5f} ratio_sim={tail[-1]['rho_l']/max(tail[-1]['rho_v'],1e-12):.1f} "
          f"umax={max(h['umax'] for h in tail):.3e} umax_int={max(h['umax_int'] for h in tail if h['umax_int'] is not None):.3e} rhomax={max(h['rhomax'] for h in tail):.3f} mdrift={max(h['massdrift'] for h in tail):.2e}", flush=True)

if __name__=="__main__":
    torch.set_num_threads(4)
    for Tr in [0.82, 0.567, 0.413]:
        for forcing in ["edm","guo"]:
            try:
                run(Tr,forcing,96,20,steps=4000)
            except Exception as e:
                print("EXC",Tr,forcing,repr(e),flush=True)
