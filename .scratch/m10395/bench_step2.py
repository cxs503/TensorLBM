import math, time, sys, os
sys.path.insert(0, "src")
import torch
nt = int(os.environ.get("NT", "1"))
torch.set_num_threads(nt)
from tensorlbm.sloshing_resonance.tank2d import (
    SloshingTank, LatticeScaling, build_tank_field, run_sway,
)
from tensorlbm.sloshing_resonance.theory import abramson_frequency

tank = SloshingTank(length=1.0, height=0.4, g=9.81, rho_liquid=1000.0)
cph = int(os.environ.get("CPH", "40"))
u_lat = float(os.environ.get("ULAT", "0.005"))
sc = LatticeScaling.from_resolution(tank, cph, u_lat=u_lat, u_char=0.074)
st = build_tank_field(tank, sc, depth=0.2, mode=1, amplitude=0.001, headroom=0.2)
f1 = abramson_frequency(1, 0.2, 1.0, 9.81)
# warm-up
run_sway(st, 0.02, sway_amplitude=0.005, omega=2*math.pi*f1, record_every=1000, slip_walls=True)
print("dx %.4f dt %.3e nx,ny %d,%d 3gh_lat=%.4f" % (sc.dx, sc.dt, st.nx, st.ny, 3*sc.g_lattice*st.depth_cells))
t0 = time.time()
res = run_sway(st, 0.2, sway_amplitude=0.005, omega=2*math.pi*f1, record_every=1000, slip_walls=True)
el = time.time() - t0
n = res["n_steps"]
print("steps", n, "wall %.2fs -> %.3f ms/step; steps/period=%.0f" % (el, 1000*el/n, 1/f1/sc.dt))
print("threads", nt, "umax", float(res["umax"].max()))