import sys; sys.path.insert(0,"src"); sys.path.insert(0,"_scratch")
import torch; torch.set_default_dtype(torch.float64)
import tensorlbm.multiphase as mp
from tensorlbm.d2q9 import equilibrium, macroscopic, stream
from sc_proto import droplet, sc_force_fixed

# 1) single phase stability with G=0 through library sc_force
f = droplet(16,64,3.0)
for s in range(1,201):
    fx,fy = mp.sc_force(f.sum(0), G=0.0)
    f = stream(mp.collide_guo(f,fx,fy,1.0))
print("G=0 fine, rho range", float(f.sum(0).min()), float(f.sum(0).max()))

# 2) fixed force, watch first steps
f = droplet(16,64,3.0)
for s in range(1,30):
    rho = f.sum(0)
    fx,fy = sc_force_fixed(rho, G=5.0)
    F = torch.sqrt(fx**2+fy**2)
    _,ux,uy = macroscopic(f)
    print(f"s={s:3d} rho[{float(rho.min()):.4f},{float(rho.max()):.4f}] Fmax={float(F.max()):.3f} umax={float(torch.sqrt(ux**2+uy**2).max()):.4f} fmin={float(f.min()):.3e} fmax={float(f.max()):.3e}")
    f = stream(mp.collide_guo(f,fx,fy,1.0))
    if not torch.isfinite(f).all():
        print("diverged", s); break
