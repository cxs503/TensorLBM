import sys, math
sys.path.insert(0, "src")
import torch; torch.set_num_threads(1)
import numpy as np
from tensorlbm.sloshing_forced import TankSpec, run_decay_case, run_forced_case, abramson_frequency
from tensorlbm.sloshing_resonance.identify import identify_mode

spec = TankSpec(cells_per_height=16, u_lat=0.0035)
d = run_decay_case(spec, mode=1, amplitude=1e-3, n_periods=2.0)
f1 = abramson_frequency(1, spec.depth, spec.length, spec.g)
fit = identify_mode(d["t"], d["a"][1], f1)
print("decay: f1 %.4f f_d %.4f err %.2f%% mass_drift %.4f ncr %d" % (
    f1, fit.f_damped, 100*(fit.f_damped/f1-1), d["max_mass_drift"], fit.n_crossings))
print("a1 max", float(np.max(np.abs(d["a"][1]))), "a2 max", float(np.max(np.abs(d["a"][2]))))
c = run_forced_case(spec, ratio=1.0, sway_frac=0.002, n_periods=1.5)
print("forced: a1max %.5f  gauges %s pressureshape %s" % (
    float(np.max(np.abs(c["a"][1]))), c["gauge_rows"], c["pressure"].shape))
from tensorlbm.sloshing_forced import pressure_metrics
pm = pressure_metrics(c, spec)
print("p_amp", pm["p_amp_Pa"], "runup amp", pm["runup"]["amplitude"], "vimp", pm["impact_velocity_m_s"])
c2 = run_forced_case(spec, ratio=abramson_frequency(2,0.2,1.0,9.81)/f1, sway_frac=0.002, n_periods=1.5)
print("a1 first %.5f  a1 second %.5f" % (float(np.max(np.abs(c["a"][1]))), float(np.max(np.abs(c2["a"][1])))))