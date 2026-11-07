import sys, time, os
sys.path.insert(0, "/root/tp-agent/work/m5469/src")
import torch
torch.set_num_threads(8)
from tensorlbm.wind_load_lng import LatticeUnits, WindLoadSolver, cube_case
t=time.time()
geom = cube_case(side=1.0, cells_per_side=6, domain=(10.0,5.0,5.0))
units = LatticeUnits.from_reynolds(dx=geom.dx, re_target=1e4, length_m=1.0, u_lat=0.1)
s = WindLoadSolver(geom, units)
print("setup", round(time.time()-t,2), geom.solid.shape, flush=True)
t=time.time(); s.run(20, average_from=0); dt=time.time()-t
print("20 steps", round(dt,2), "per step ms", round(dt/20*1e3,1), flush=True)
