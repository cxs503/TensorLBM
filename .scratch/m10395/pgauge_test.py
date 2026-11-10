import sys, math
sys.path.insert(0, "src")
import torch; torch.set_num_threads(1)
import numpy as np
from tensorlbm.sloshing_forced import TankSpec, build_case
from tensorlbm.sloshing_resonance.tank2d import wall_pressure_profile

spec = TankSpec(cells_per_height=32)
tank, sc, st = build_case(spec, mode=1, amplitude=0.01)   # tilted standing wave, t=0, no motion
p = wall_pressure_profile(st)
rows = spec.gauge_rows(st.depth_cells)
eta_wall = 0.01 * math.cos(0.0)  # A cos(k x) at x=0
print("eta_wall (imposed) = %.5f m -> rho g eta = %.2f Pa" % (eta_wall, 1000*9.81*eta_wall))
print("depth_cells", st.depth_cells, "dx", sc.dx, "rows", rows)
for r in rows:
    print("  row %3d  y %+0.4f m  p_dyn %8.3f Pa  ratio %.3f" % (
        r, (r-0.5-st.depth_cells)*sc.dx, float(p[r]), float(p[r])/(1000*9.81*eta_wall)))
# column-2 pressure (before extrapolation) for reference
from tensorlbm.sloshing_resonance.tank2d import CS2
rho = st.rho
dp = CS2*(rho - st.rho_ref)*sc.pressure_scale(tank)
print("col1/col2 at bottom rows:", [(float(dp[r,1]), float(dp[r,2])) for r in (1,2,5)])
print("rho col1 rows1-3:", [float(rho[r,1]) for r in (1,2,3)], "rho_ref:", [float(st.rho_ref[r,1]) for r in (1,2,3)])
print("flags col1 rows1-3:", [int(st.flags[r,1]) for r in (1,2,3)])