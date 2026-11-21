import math, sys, os, cProfile, pstats, io
sys.path.insert(0, "src")
import torch
torch.set_num_threads(1)
from tensorlbm.sloshing_resonance.tank2d import SloshingTank, LatticeScaling, build_tank_field, run_sway
from tensorlbm.sloshing_resonance.theory import abramson_frequency

tank = SloshingTank(length=1.0, height=0.4, g=9.81, rho_liquid=1000.0)
sc = LatticeScaling.from_resolution(tank, 40, u_lat=0.005, u_char=0.074)
st = build_tank_field(tank, sc, depth=0.2, mode=1, amplitude=0.001, headroom=0.2)
f1 = abramson_frequency(1, 0.2, 1.0, 9.81)
pr = cProfile.Profile()
pr.enable()
run_sway(st, 0.05, sway_amplitude=0.005, omega=2*math.pi*f1, record_every=1000, slip_walls=True)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(18)
print(s.getvalue()[:3500])