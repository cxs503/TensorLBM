"""Compare library collide+stream against the reference recipe on the reference's own
initial condition (chapter9/shanchen.cpp: rhol=2.1, rhog=0.15, R=15, g=-4.7, tau=1).

Arms:
  A) tensorlbm.d2q9.stream as shipped (rolls by -c)
  B) same but rolling by +c (standard lattice advection)
"""
import sys, time; sys.path.insert(0, "src"); sys.path.insert(0, "_scratch")
import torch; torch.set_default_dtype(torch.float64)
import tensorlbm.multiphase as mp
from tensorlbm.d2q9 import C, equilibrium, macroscopic, stream
from tensorlbm.multiphase.laplace import laplace_pressure_calibration
from sc_proto import sc_force_fixed

N = 64; R = 15.0; RHOL = 2.1; RHOV = 0.15; G = 4.7; TAU = 1.0


def stream_fixed(f):
    out = torch.empty_like(f)
    for i in range(9):
        cx, cy = int(C[i, 0]), int(C[i, 1])
        out[i] = f[i] if (cx == 0 and cy == 0) else torch.roll(f[i], shifts=(cy, cx), dims=(0, 1))
    return out


def init():
    yy, xx = torch.meshgrid(torch.arange(N, dtype=torch.float64),
                            torch.arange(N, dtype=torch.float64), indexing="ij")
    ins = ((yy - N / 2.) ** 2 + (xx - N / 2.) ** 2) <= R * R
    rho = torch.where(ins, torch.tensor(RHOL, dtype=torch.float64),
                      torch.tensor(RHOV, dtype=torch.float64))
    return equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))


def arm(name, streamer, steps=4000, rep=500):
    f = init(); t0 = time.time()
    for s in range(1, steps + 1):
        rho = f.sum(0)
        fx, fy = sc_force_fixed(rho, G)
        f = streamer(mp.collide_guo(f, fx, fy, TAU))
        if not torch.isfinite(f).all():
            print(f"  [{name}] DIVERGED at step {s}", flush=True); return
        if s % rep == 0:
            r = f.sum(0); _, ux, uy = macroscopic(f)
            u = float(torch.sqrt(ux ** 2 + uy ** 2).max())
            try:
                c = laplace_pressure_calibration(r, G)
                print(f"  [{name}] s={s:5d} rg={c['rho_v']:.4f} rl={c['rho_l']:.4f} "
                      f"dp={c['dp']:+.5e} r={c['radius']:.3f} gamma={c['gamma']:+.5e} "
                      f"umax={u:.4f} ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"  [{name}] s={s:5d} rho[{float(r.min()):.4f},{float(r.max()):.4f}] "
                      f"umax={u:.4f} cal={type(e).__name__} ({time.time()-t0:.0f}s)", flush=True)


print("A) library stream (ship)", flush=True); arm("A", stream)
print("B) stream +c (standard)", flush=True); arm("B", stream_fixed)