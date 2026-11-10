import sys
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import (
    herschel_bulkley, channel_reference, plug_report,
)

H = 24.0
h = int(2 * H)
y = torch.arange(h + 2, dtype=torch.float64) - 0.5 - H
rfrac = 0.35
for n in (0.6, 0.8):
    # design u_c = 0.02:  A (H-r_c)^((n+1)/n) = 0.02 with A = (G/K)^(1/n) n/(n+1)
    p = (n + 1) / n
    Hrc = H * (1 - rfrac)
    A = 0.02 / Hrc ** p
    K = 0.02
    G = K * (A * (n + 1) / n) ** n
    tau_y = rfrac * G * H
    for m in (1e4, 3e4, 1e5, 3e5, 1e6, 3e6):
        cl = herschel_bulkley(tau_y, K, n, regularization=m)
        ref = channel_reference(cl, G, H, y=y)
        rep = plug_report(cl, G, y, ref["u"], ref["gamma_dot"], y_ref=y, u_ref=ref["u"], gamma_ref=ref["gamma_dot"])
        print(f"n={n} m={m:6.0f} G={G:.3e} tau_y={tau_y:.3e} nu_solid={tau_y*m:6.2f} u_max={ref['u_max']:.4f} "
              f"ideal={rep['r_c_ideal']:.3f} | ref vel={rep['rel_ref_velocity']:+.1%} stress={rep['rel_ref_stress']:+.1%} fit={rep['rel_ref_fit']:+.1%}",
              flush=True)