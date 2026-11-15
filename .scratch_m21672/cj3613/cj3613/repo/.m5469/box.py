import sys, time, math
sys.path.insert(0, "/root/tp-agent/work/m5469/src")
import torch
torch.set_num_threads(1)
from tensorlbm.wind_load_lng import LatticeUnits, WindLoadSolver, voxelise, Box

# single box normal to the flow: 18 m (flow) x 36 m (cross) x 21 m (tall)
dx = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
crossw = float(sys.argv[2]) if len(sys.argv) > 2 else 36.0
depth = float(sys.argv[3]) if len(sys.argv) > 3 else 18.0
tall = float(sys.argv[4]) if len(sys.argv) > 4 else 21.0
box = Box(0.0, depth, -0.5 * crossw, 0.5 * crossw, 0.0, tall, "block")
kw = dict(margin_upstream=3 * tall, margin_downstream=6 * tall,
          margin_side=5 * tall, margin_top=5 * tall)
g0 = voxelise([box], dx, 0.0, **kw)
units = LatticeUnits.from_velocity(dx=dx, u_phys=15.0, u_lat=0.15)
s = WindLoadSolver(g0, units, ref_areas=(g0.area_frontal, g0.area_lateral),
                   ref_breadth_height=(g0.breadth, g0.height))
print(f"box {depth}x{crossw}x{tall} dx={dx} grid={g0.solid.shape} cells={g0.solid.size} "
      f"AF={g0.area_frontal} solid_cells={int(g0.solid.sum())}", flush=True)
t0 = time.time()
for k in range(6):
    r = s.run(100, average_from=s.step_count - 100)
    w = r["world_frame_mean"]
    print(f"  step {s.step_count:5d} fxw={w['fx']:8.3f} C_D={r['C_X']:6.3f} "
          f"CY={r['C_Y']:7.3f} mass={s.mass_budget(from_index=s.step_count-100)['mean_rel_error']:.3f} "
          f"t={time.time()-t0:5.1f}s", flush=True)