import sys, math, time
sys.path.insert(0, "src")
import os
import torch
torch.set_num_threads(1)
import numpy as np
from tensorlbm.sloshing_forced import TankSpec, run_decay_case, run_forced_case, pressure_metrics
from tensorlbm.sloshing_resonance.identify import identify_mode

spec = TankSpec(cells_per_height=int(os.environ.get("CPH", "20")),
                u_lat=float(os.environ.get("ULAT", "0.0035")))
print("depth", spec.depth, "freeboard", spec.freeboard)
t0 = time.time()
case = run_decay_case(spec, mode=1, amplitude=1e-3, n_periods=2.5)
print("decay steps", case["n_steps"], "wall %.1fs" % (time.time() - t0),
      "dt %.3e nx? eta" % case["dt"], case["eta"].shape, "p", case["pressure"].shape)
f1 = case["f1_theory"]
fit = identify_mode(case["t"], case["a"][1], f1)
print("f1 theory %.5f  f_d %.5f f_n %.5f zeta %.5f  err_fd %+.2f%% err_fn %+.2f%%" % (
    f1, fit.f_damped, fit.f_natural, fit.zeta, 100*(fit.f_damped/f1-1), 100*(fit.f_natural/f1-1)))
print("mass drift", case["mass_drift"], "gauge rows", case["gauge_rows"], "depth_cells", case["depth_cells"])

t0 = time.time()
fc = run_forced_case(spec, ratio=1.0, sway_frac=0.005, n_periods=3.0)
print("forced steps", fc["n_steps"], "wall %.1fs" % (time.time() - t0))
pm = pressure_metrics(fc, spec)
print("runup amp %.4f m peak %.4f  v_imp %.4f" % (pm["runup_m"]["amplitude"], pm["runup_m"]["peak"], pm["impact_velocity_m_s"]))
for gv in pm["gauges"]:
    print("  gauge row", gv["row"], "y %.4f" % gv["y_from_swl_m"], "amp %.1f Pa peak %.1f Pa" % (gv["amplitude_Pa"], gv["peak_Pa"]))
print("bounds", pm["bounds"], "p_peak/(rho g h)", pm["p_peak_over_rho_g_h"])
print("a1 amp from modal:", np.max(np.abs(fc["a"][1][-50:])), "a2:", np.max(np.abs(fc["a"][2][-50:])))