import sys, math, time, os
sys.path.insert(0, "src")
import torch
torch.set_num_threads(1)
import numpy as np
from tensorlbm.sloshing_forced import TankSpec, run_decay_case
from tensorlbm.sloshing_resonance.identify import identify_mode, fft_peak

spec = TankSpec(cells_per_height=int(os.environ.get("CPH", "32")),
                u_lat=float(os.environ.get("ULAT", "0.0035")))
case = run_decay_case(spec, mode=1, amplitude=1e-3, n_periods=float(os.environ.get("PER", "3")))
f1 = case["f1_theory"]
t, a1 = case["t"], case["a"][1]
print("steps", case["n_steps"], "dt", case["dt"], "rec", t.size, "depth_cells", case["depth_cells"],
      "mass_drift", case["mass_drift"])
print("a1 first 8:", np.round(a1[:8], 5))
print("a1 last 8:", np.round(a1[-8:], 6))
print("a1 max/min", a1.max(), a1.min())
# high-frequency content: count sign changes of the difference between consecutive samples
d = np.diff(a1)
print("sign changes of diff:", int(np.sum(np.sign(d[1:]) != np.sign(d[:-1]))), "of", d.size)
for name, sig in [("raw", a1), ("ma5", np.convolve(a1, np.ones(5)/5, mode="same"))]:
    fit = identify_mode(t, sig, f1)
    print(name, "f_d %.5f err %+.2f%% f_n %.5f zeta %.5f ncr %d" % (
        fit.f_damped, 100*(fit.f_damped/f1-1), fit.f_natural, fit.zeta, fit.n_crossings))
fp, amp = fft_peak(t, a1)
print("fft peak", fp, "err %+.2f%%" % (100*(fp/f1-1)))
# zero crossing times of raw
from tensorlbm.sloshing_resonance.identify import zero_crossings
zc = zero_crossings(t, a1)
print("crossings", len(zc), "first few periods", np.round(np.diff(zc[:10]), 4))