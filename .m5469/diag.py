import sys, time, json
sys.path.insert(0, "/root/tp-agent/work/m5469/src")
import torch
torch.set_num_threads(1)
from tensorlbm.ship_wind_aero import catalogue
from tensorlbm.wind_load_lng import LatticeUnits, WindLoadSolver, voxelise

ship = catalogue()["container"]
dx = float(sys.argv[1]) if len(sys.argv) > 1 else 3.5
wall = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True
boxes = ship.to_lng_boxes()
g0 = voxelise(boxes, dx, 0.0, margin_upstream=40, margin_downstream=40,
              margin_side=18, margin_top=20)
geom = voxelise(boxes, dx, 90.0, margin_upstream=40, margin_downstream=40,
                margin_side=18, margin_top=20)
units = LatticeUnits.from_velocity(dx=dx, u_phys=15.0, u_lat=0.15)
s = WindLoadSolver(geom, units, ref_areas=(g0.area_frontal, g0.area_lateral),
                   ref_breadth_height=(g0.breadth, g0.height),
                   wall_model=wall, wall_feedback=wall)
print("grid", geom.solid.shape, "solid", int(geom.solid.sum()), "AF", g0.area_frontal,
      "AL", g0.area_lateral, flush=True)
t0 = time.time()
for k in range(10):
    r = s.run(100, average_from=s.step_count - 100)
    w = r["world_frame_mean"]
    print(f"step {s.step_count:5d} fxw={w['fx']:8.3f} fyw={w['fy']:8.3f} "
          f"CX={r['C_X']:6.3f} CY={r['C_Y']:7.3f} CXwall={r['C_X_wall']:6.3f} "
          f"masserr={s.mass_budget(from_index=s.step_count-100)['mean_rel_error']:.3f} "
          f"t={time.time()-t0:5.1f}s", flush=True)