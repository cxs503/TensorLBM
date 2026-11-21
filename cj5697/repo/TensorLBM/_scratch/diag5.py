import sys,time; sys.path.insert(0,"src"); sys.path.insert(0,"_scratch")
import torch; torch.set_default_dtype(torch.float64)
import tensorlbm.multiphase as mp
from tensorlbm.d2q9 import macroscopic, stream
from sc_proto import droplet, sc_force_fixed
import itertools, json

def run(G=5.0,R=16,n=64,width=6.0,tau=1.0,steps=12000,ramp=4000,rep=1000,forcing="guo",Gstart=0.0):
    f=droplet(R,n,width); t0=time.time()
    for s in range(1,steps+1):
        g = Gstart + (G-Gstart)*min(1.0,s/ramp)
        rho=f.sum(0); fx,fy=sc_force_fixed(rho,g)
        fpost = mp.collide_guo(f,fx,fy,tau) if forcing=="guo" else (mp.collide_edm(f,fx,fy,tau) if forcing=="edm" else mp.collide_velocity_shift(f,fx,fy,tau))
        f=stream(fpost)
        if not torch.isfinite(f).all():
            print(f"n={n} R={R} w={width} tau={tau} {forcing} ramp={ramp} DIV at {s}",flush=True); return
        if s%rep==0:
            rho,ux,uy=macroscopic(f); u=torch.sqrt(ux**2+uy**2)
            print(f"  s={s:6d} g={g:.2f} rho[{float(rho.min()):.4f},{float(rho.max()):.4f}] umax={float(u.max()):.4f} t={time.time()-t0:.1f}s",flush=True)

print("A: guo ramp 4000, w=6, n=64"); run()
print("B: edm  ramp 4000, w=6, n=64"); run(forcing="edm")
