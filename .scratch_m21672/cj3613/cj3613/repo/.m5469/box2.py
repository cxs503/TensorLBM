import sys, time
sys.path.insert(0, "/root/tp-agent/work/m5469/src")
import torch
torch.set_num_threads(1)
from tensorlbm.wind_load_lng import LatticeUnits, WindLoadSolver, voxelise, Box

dx = float(sys.argv[1]); cs = float(sys.argv[2]); wm = int(sys.argv[3])
steps = int(sys.argv[4])
box = Box(0.0, 18.0, -18.0, 18.0, 0.0, 21.0, "block")
kw = dict(margin_upstream=60.0, margin_downstream=120.0, margin_side=100.0,
          margin_top=100.0)
g0 = voxelise([box], dx, 0.0, **kw)
units = LatticeUnits.from_velocity(dx=dx, u_phys=15.0, u_lat=0.15)
s = WindLoadSolver(g0, units, ref_areas=(g0.area_frontal, g0.area_lateral),
                   ref_breadth_height=(g0.breadth, g0.height),
                   cs_smag=cs, wall_model=bool(wm), wall_feedback=bool(wm))
print(f"dx={dx} cs={cs} wall={wm} grid={g0.solid.shape} cells={g0.solid.size} "
      f"AF={g0.area_frontal} solid={int(g0.solid.sum())} tau0={units.tau0:.6f}", flush=True)
t0 = time.time()
for k in range(steps // 100):
    r = s.run(100, average_from=s.step_count - 100)
    print(f"  step {s.step_count:5d} C_D={r['C_X']:6.3f} CXw={r['C_X_wall']:6.3f} "
          f"mass={s.mass_budget(from_index=s.step_count-100)['mean_rel_error']:.4f} "
          f"t={time.time()-t0:5.1f}s", flush=True)