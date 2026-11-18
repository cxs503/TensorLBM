import sys, os, json, math
sys.path.insert(0, "src")
import torch
torch.set_num_threads(1)
import numpy as np
from tensorlbm.sloshing_forced.campaign import analyse_decay, analyse_forced, spec_for
from tensorlbm.sloshing_forced.sweep import TankSpec

out = __import__("pathlib").Path(".scratch/m10395/func")
out.mkdir(parents=True, exist_ok=True)
spec = TankSpec(cells_per_height=24, u_lat=0.0035)
dc = {"id": "decay_m1", "kind": "decay", "mode": 1, "amplitude": 1e-3, "periods": 1.5}
summ, arr = analyse_decay(dc, spec)
print("DECAY", json.dumps({k: summ[k] for k in ("f_theory_hz","f_damped_hz","f_natural_hz","f_fft_hz","zeta","error_natural_pct","mass_drift","n_steps")}, indent=1))
zeta = summ["zeta"]
fc = {"id": "sweep_r1.00", "kind": "forced", "ratio": 1.0, "sway_frac": 0.002, "periods": 1.5, "branch": "linear"}
summ2, arr2 = analyse_forced(fc, spec, zeta)
print("FORCED a1", summ2["a1"])
print("ref", summ2["reference"])
print("ratio meas/theory", summ2["reference_ratio_measured_over_theory"], "parity", summ2["parity_ratio_a2_over_a1"])
print("gauges")
for g in summ2["pressure"]["gauges"]:
    print("  ", {k: round(v, 3) if isinstance(v, float) else v for k, v in g.items()})
print("p_peak", summ2["pressure"]["p_peak_Pa"], "bounds", {k: round(v, 1) for k, v in summ2["pressure"]["bounds"].items()})
print("runup", {k: round(v, 5) for k, v in summ2["pressure"]["runup"].items() if isinstance(v, float)})