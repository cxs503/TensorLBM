import sys, time
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import (
    power_law, newtonian, channel_reference, l2_relative_error, run_channel,
)
from tensorlbm.nonnewtonian_frontier.reference import power_law_velocity

torch.set_num_threads(8)
h, U, nu_wall = 12, 0.02, 0.1
H = h / 2
y = torch.arange(h + 2, dtype=torch.float64) - 0.5 - H

# --- Newtonian baseline: validates force + wall + driver ---
nu = 0.1
cl = newtonian(nu)
a = 2 * nu * U / H**2
out = run_channel(h, a, cl, nx=4, n_steps=6000, tol=1e-12, check_every=6000)
rows = out["fluid_rows"]
uref = channel_reference(cl, a, H, y=y)["u"][rows]
print(f"NEWTONIAN h={h} a={a:.4g} umax={out['u_max']:.6f}/{float(uref.max()):.6f} "
      f"L2={l2_relative_error(out['u'][rows].double(), uref)*100:.3f}% steps={out['steps']}", flush=True)
print("  u_num:", [round(float(v), 6) for v in out["u"][rows]])
print("  u_ref:", [round(float(v), 6) for v in uref])

# --- power law ---
n = 0.7
gw = U * (n + 1) / (n * H)
K = nu_wall * gw ** (1 - n)
a = K * gw ** n / H
cl = power_law(K, n)
ref = channel_reference(cl, a, H, y=y)
uinit = ref["u"].clone(); uinit[0] = 0; uinit[-1] = 0
out = run_channel(h, a, cl, nx=4, n_steps=6000, tol=1e-12, check_every=6000, u_init=uinit)
rows = out["fluid_rows"]
uref = power_law_velocity(y[rows], H, a, K, n)
g_ref = ref["gamma_dot"][rows]
# symmetrised measured gamma
from tensorlbm.nonnewtonian_frontier import symmetrise
r, g_sym = symmetrise(y, out["gamma_dot"][:, 0])
r2, gref_sym = symmetrise(y, ref["gamma_dot"])
r3, u_sym = symmetrise(y, out["u"])
print(f"\nPOWERLAW n={n} K={K:.5g} a={a:.4g} L2={l2_relative_error(out['u'][rows].double(), uref)*100:.3f}%")
print("  r      u_num      u_ref      g_num      g_ref")
for i in range(0, len(r), 1):
    print(f"  {float(r[i]):5.1f} {float(u_sym[i]):10.6f} "
          f"{float(channel_reference(cl, a, H, y=torch.tensor([float(r[i])]))['u'][0]):10.6f} "
          f"{float(g_sym[i]):10.3e} {float(gref_sym[i]):10.3e}")