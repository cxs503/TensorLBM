import sys
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import herschel_bulkley, channel_reference, plug_half_width_velocity, plug_half_width_stress, symmetrise

H = 24.0
h = int(2 * H)
y = torch.arange(h + 2, dtype=torch.float64) - 0.5 - H
rows = slice(1, h + 1)
for m in (1e3, 3e3, 1e4, 3e4):
    for n in (0.6, 1.0):
        # design: plug fraction 0.35, u_c = 0.02, K chosen so nu_wall ~ 0.3
        rfrac = 0.35
        GoverK = 2.1e-3
        K = 0.02
        G = GoverK * K
        tau_y = rfrac * G * H
        cl = herschel_bulkley(tau_y, K, n, regularization=m)
        ref = channel_reference(cl, G, H, y=y)
        u = ref["u"]; g = ref["gamma_dot"]
        r, usym = symmetrise(y, u)
        r2, gsym = symmetrise(y, g)
        rcv = plug_half_width_velocity(r, usym, delta=0.01)
        rcs = plug_half_width_stress(r2, gsym, cl)
        ideal = tau_y / G
        print(f"m={m:6.0f} n={n} tau_y={tau_y:.3e} nu_solid={tau_y*m:.2f} u_max={ref['u_max']:.5f} "
              f"r_c: ideal={ideal:.3f} vel={rcv:.3f}({rcv/ideal-1:+.1%}) stress={rcs:.3f}({rcs/ideal-1:+.1%}) "
              f"nu_wall={float(cl.nu_of_gamma(torch.tensor([ref['gamma_dot_wall']], dtype=torch.float64))[0]):.3f}")