import sys, time
sys.path.insert(0, "/root/tp-agent/work/m5469/src")
import torch
torch.set_num_threads(1)
from tensorlbm.wind_load_lng import LatticeUnits, WindLoadSolver, cube_case
geom = cube_case(side=1.0, cells_per_side=6, domain=(10.0,5.0,5.0))
units = LatticeUnits.from_reynolds(dx=geom.dx, re_target=1e4, length_m=1.0, u_lat=0.1)
s = WindLoadSolver(geom, units)
s.run(2)
import cProfile, pstats
pr = cProfile.Profile(); pr.enable(); s.run(10); pr.disable()
st = pstats.Stats(pr); st.sort_stats("cumulative")
st.print_stats(14)
