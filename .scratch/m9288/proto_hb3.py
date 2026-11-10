import sys
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import herschel_bulkley, channel_reference, power_law_velocity
from tensorlbm.nonnewtonian_frontier.plug import symmetrise, plug_half_width_fit

H, n, K = 24.0, 0.6, 0.02
rfrac = 0.35
p = (n + 1) / n
A = 0.02 / (H * (1 - rfrac)) ** p
G = K * (A * (n + 1) / n) ** n
tau_y = rfrac * G * H
print(f"G={G:.6e} tau_y={tau_y:.6e} r_c_ideal={tau_y/G:.4f} A={A:.5e}")

def ideal(r):
    r = torch.as_tensor(r, dtype=torch.float64) if not torch.is_tensor(r) else r
    rc = tau_y / G
    return A * ((H - rc) ** p - torch.clamp(r - rc, min=0.0) ** p)

for m in (1e4, 1e6, 3e8):
    cl = herschel_bulkley(tau_y, K, n, regularization=m)
    rr = torch.linspace(0, H, 2001, dtype=torch.float64)
    ref = channel_reference(cl, G, H, y=rr)["u"]
    u_ideal = ideal(rr)
    rel = ((ref - u_ideal).abs() / float(u_ideal[0]))
    print(f"m={m:.0e} maxrel|reg-ideal|={float(rel.max()):.3e} at r={float(rr[int(rel.argmax())]):.2f} "
          f"u_c_reg={float(ref[0]):.6f} u_c_ideal={float(u_ideal[0]):.6f}", flush=True)
    # fit on the 'ideal' profile itself (sanity of the fitter)
    print("   fit(ideal) =", round(plug_half_width_fit(rr, u_ideal, G, cl), 4) if m > 1e7 else "skip",
          " fit(reg) =", round(plug_half_width_fit(rr, ref, G, cl), 4))
    if m > 1e7:
        break