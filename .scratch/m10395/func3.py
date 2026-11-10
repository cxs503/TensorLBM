import sys, json
sys.path.insert(0, "src")
import torch; torch.set_num_threads(1)
from tensorlbm.sloshing_forced.campaign import analyse_forced
from tensorlbm.sloshing_forced.sweep import TankSpec
spec = TankSpec(cells_per_height=16, u_lat=0.0035)
fc = {"id": "sweep_r1.00", "kind": "forced", "ratio": 1.0, "sway_frac": 0.002, "periods": 1.5, "branch": "linear"}
s, a = analyse_forced(fc, spec, 0.0)
print("keys ok")
print("a1", round(s["a1"]["amplitude"], 6), "ref", round(s["reference"]["a_peak_half"], 6), "ratio", round(s["reference_ratio_measured_over_theory"], 3))
print("parity", round(s["parity_ratio_a2_over_a1"], 4))
for g in s["pressure"]["gauges"]:
    print("  gauge", g["row"], "y %.3f" % g["y_from_swl_m"], "amp %.1f" % g["p_measured_amp_Pa"],
          "lin %.1f" % g["p_linear_RAO_amp_Pa"], "err %+.1f%%" % g["error_vs_linear_pct"])
print("runup", round(s["pressure"]["runup"]["amplitude"], 5), "rawmax", round(s["pressure"]["runup"]["raw_max_m"], 5))
print("p_amp", round(s["pressure"]["p_amp_Pa"], 1), "bounds", {k: round(v, 1) for k, v in s["pressure"]["bounds"].items()})