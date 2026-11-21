import sys; sys.path.insert(0,"src"); sys.path.insert(0,"_scratch")
import torch; torch.set_default_dtype(torch.float64)
import tensorlbm.multiphase as mp
from tensorlbm.d2q9 import macroscopic, stream
from sc_proto import droplet, sc_force_fixed
n=64
f = droplet(16,n,3.0)
print("initial u:", float(macroscopic(f)[1].abs().max()), " mass", float(f.sum()))
for s in range(1,401):
    rho=f.sum(0); fx,fy=sc_force_fixed(rho,0.2)
    f=stream(mp.collide_guo(f,fx,fy,1.0))
    if s%50==0:
        rho,ux,uy=macroscopic(f); u=torch.sqrt(ux**2+uy**2)
        i=torch.argmax(u); iy,ix=int(i//n),int(i%n)
        print(f"s={s} umax={float(u.max()):.5f} at ({ix},{iy}) rho_there={float(rho[iy,ix]):.4f} mass={float(f.sum()):.8f} Fmax={float(torch.sqrt(fx**2+fy**2).max()):.4f}")
# also: pure diffusion G=0 with sharp droplet, check umax
f = droplet(16,n,3.0)
for s in range(1,401):
    f=stream(mp.collide_bgk(f,1.0))
    if s%100==0:
        rho,ux,uy=macroscopic(f); print(f"[G=0] s={s} umax={float(torch.sqrt(ux**2+uy**2).max()):.3e}")
