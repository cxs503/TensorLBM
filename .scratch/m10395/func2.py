import sys, json
sys.path.insert(0, "src")
import torch; torch.set_num_threads(1)
import numpy as np
from tensorlbm.sloshing_forced.campaign import analyse_decay, analyse_forced
from tensorlbm.sloshing_forced.sweep import TankSpec
spec = TankSpec(cells_per_height=32, u_lat=0.0035)
dc = {"id":"decay_m1","kind":"decay","mode":1,"amplitude":1e-3,"periods":3.0}
s1,a1 = analyse_decay(dc, spec)
print("DECAY", json.dumps({k:s1[k] for k in ("f_theory_hz","f_damped_hz","f_natural_hz","f_fft_hz","zeta","error_natural_pct","mass_drift","n_steps")}, indent=1))
for ratio in (1.00, 1.75):
    fc = {"id":f"sweep_r{ratio:.2f}","kind":"forced","ratio":ratio,"sway_frac":0.002,"periods":2.0,"branch":"linear"}
    s2,a2 = analyse_forced(fc, spec, s1["zeta"])
    print("FORCED r", ratio, "a1", round(s2["a1"]["amplitude"],6), "ref_peak", round(s2["reference"]["a_peak_half"],6),
          "ratio", round(s2["reference_ratio_measured_over_theory"],3), "parity", round(s2["parity_ratio_a2_over_a1"],4))
    print("  runup amp", round(s2["pressure"]["runup"]["amplitude"],5), "rawmax", round(s2["pressure"]["runup"]["raw_max_m"],5),
          "v_imp", round(s2["pressure"]["impact_velocity_m_s"],4))
    for g in s2["pressure"]["gauges"]:
        print("   gauge", g["row"], "y %.3f" % g["y_from_swl_m"], "amp %.1f" % g["p_measured_amp_Pa"],
              "lin %.1f" % g["p_linear_RAO_amp_Pa"], "RAO %.1f" % g["p_forced_RAO_amp_Pa"],
              "err_lin %+.1f%%" % g["error_vs_linear_pct"], "err_RAO %+.1f%%" % g["error_vs_forced_RAO_pct"])
    print("  p_amp", round(s2["pressure"]["p_amp_Pa"],1), "rawmax", round(s2["pressure"]["p_raw_max_Pa"],1),
          "bounds", {k: round(v,1) for k,v in s2["pressure"]["bounds"].items()})
