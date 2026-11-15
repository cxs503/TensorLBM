import sys, time, os
sys.path.insert(0, "/root/tp-agent/work/m5469/src")
import torch
nt = int(sys.argv[1]); torch.set_num_threads(nt)
from tensorlbm.wind_load_lng import LatticeUnits, WindLoadSolver, cube_case
geom = cube_case(side=1.0, cells_per_side=6, domain=(10.0,5.0,5.0))
units = LatticeUnits.from_reynolds(dx=geom.dx, re_target=1e4, length_m=1.0, u_lat=0.1)
s = WindLoadSolver(geom, units)
s.run(3, average_from=0)
t=time.time(); s.run(20, average_from=0); dt=time.time()-t
print(f"threads={nt} cells={geom.solid.size} per_step_ms={dt/20*1e3:.1f}", flush=True)
