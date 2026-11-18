import math, time, sys, os
sys.path.insert(0, "src")
import torch
torch.set_num_threads(2)
from tensorlbm.sloshing_resonance.tank2d import (
    SloshingTank, LatticeScaling, build_tank_field, run_sway, run_free_decay,
    surface_elevation,
)
from tensorlbm.sloshing_resonance.theory import abramson_frequency, abramson_period

tank = SloshingTank(length=1.0, height=0.4, g=9.81, rho_liquid=1000.0)
for cph in (40,):
    sc = LatticeScaling.from_resolution(tank, cph)
    print("dx", sc.dx, "dt", sc.dt, "g_lat", sc.g_lattice)
    st = build_tank_field(tank, sc, depth=0.2, mode=1, amplitude=0.001, headroom=0.2)
    print("nx,ny", st.nx, st.ny, "h_cells", st.depth_cells)
    f1 = abramson_frequency(1, 0.2, 1.0, 9.81)
    print("f1", f1)
    n_steps = int(round(2.0 * (1/f1) / sc.dt))
    print("steps for 2 periods", n_steps)
    t0 = time.time()
    res = run_sway(st, 2.0, sway_amplitude=0.005, omega=2*math.pi*f1, record_every=50, slip_walls=True)
    el = time.time() - t0
    print("elapsed %.2f s -> %.3f ms/step" % (el, 1000*el/max(res["n_steps"],1)))
    print("mass drift", res["mass_drift"], "umax", float(res["umax"].max()))
    print("eta shape", res["eta"].shape, "pressure", res["pressure"].shape)