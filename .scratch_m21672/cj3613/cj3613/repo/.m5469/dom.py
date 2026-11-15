import sys, time
sys.path.insert(0, "/root/tp-agent/work/m5469/src")
import torch
torch.set_num_threads(1)
from tensorlbm.ship_wind_aero import catalogue
from tensorlbm.wind_load_lng import LatticeUnits, WindLoadSolver, voxelise

shipname = sys.argv[1]
dx = float(sys.argv[2])
angle = float(sys.argv[3])
mu = [float(v) for v in sys.argv[4:8]]
steps = int(sys.argv[8]) if len(sys.argv) > 8 else 600
ship = catalogue()[shipname]
boxes = ship.to_lng_boxes()
kw = dict(margin_upstream=mu[0], margin_downstream=mu[1], margin_side=mu[2],
          margin_top=mu[3])
g0 = voxelise(boxes, dx, 0.0, **kw)
geom = voxelise(boxes, dx, angle, **kw)
units = LatticeUnits.from_velocity(dx=dx, u_phys=15.0, u_lat=0.15)
s = WindLoadSolver(geom, units, ref_areas=(g0.area_frontal, g0.area_lateral),
                   ref_breadth_height=(g0.breadth, g0.height))
print(f"{shipname} dx={dx} ang={angle} grid={geom.solid.shape} cells={geom.solid.size} "
      f"AF={g0.area_frontal} AL={g0.area_lateral}", flush=True)
t0 = time.time()
for k in range(steps // 100):
    r = s.run(100, average_from=s.step_count - 100)
    w = r["world_frame_mean"]
    print(f"  step {s.step_count:5d} fxw={w['fx']:8.3f} fyw={w['fy']:8.3f} "
          f"CX={r['C_X']:7.3f} CY={r['C_Y']:7.3f} CXstd={r['C_X_std']:5.3f} "
          f"mass={s.mass_budget(from_index=s.step_count-100)['mean_rel_error']:.3f} "
          f"t={time.time()-t0:5.1f}s", flush=True)