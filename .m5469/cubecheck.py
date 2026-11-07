import sys, time
sys.path.insert(0,"/root/tp-agent/work/m5469/src")
import torch; torch.set_num_threads(1)
from tensorlbm.wind_load_lng import LatticeUnits, WindLoadSolver, cube_case
geom = cube_case(side=1.0, cells_per_side=6, domain=(10.0,5.0,5.0))
units = LatticeUnits.from_reynolds(dx=geom.dx, re_target=20000.0, length_m=1.0, u_lat=0.15)
s = WindLoadSolver(geom, units)
print("tau0", units.tau0, "nu_lat", units.nu_lat, "A_F cells", geom.area_frontal/geom.dx**2, flush=True)
t=time.time(); r=s.run(300, average_from=150)
q=0.5*units.u_lat**2; A=geom.area_frontal/geom.dx**2
print("C_D", r["C_X"], "C_D_altrho", r["C_X_std"], "wall frac", r["C_X_wall"]/max(r["C_X"],1e-9), "t", round(time.time()-t,1))
print("world mean fx", r["world_frame_mean"])
