"""DE-X-RELX3-RRIFACE: count RR kernel invocations and measure the
pre-ABB collision-output difference inside the real probe loop."""

import torch

import benchmarks.water_entry.single_drop.free_surface_probe as P
import tensorlbm.collision_d3q19_advanced as ADV
import tensorlbm.free_surface_lbm as F

torch.set_num_threads(7)
torch.manual_seed(0)

calls = {"n": 0, "max_diff": 0.0}
orig_kernel = ADV.collide_regularized_stress_d3q19


def spy_kernel(f_total, feq, tau):
    out = orig_kernel(f_total, feq, tau)
    calls["n"] += 1
    f_bgk = f_total - (f_total - feq) / tau
    calls["max_diff"] = max(calls["max_diff"], (out - f_bgk).abs().max().item())
    return out


# The solver imports the kernel inside the branch (from .collision_d3q19_advanced
# import collide_regularized_stress_d3q19), so patch the module attribute.
ADV.collide_regularized_stress_d3q19 = spy_kernel

orig = P.free_surface_step
P.free_surface_step = orig  # no probe-level spy needed
try:
    P.probe(
        8,
        0.06,
        20,
        0.18,
        capacity=True,
        nearest=False,
        pilot=False,
        span=4.0,
        pool_depth=3.0,
        simple=True,
        release=1.53,
        planes_path=None,
        silhouette_path=None,
        ledger=False,
        diag_path=None,
        u_limit=0.15,
        cav_clos_alpha=0.0,
        rr_tau=0.0,
        rr_iface=True,
    )
finally:
    ADV.collide_regularized_stress_d3q19 = orig_kernel

print("RR kernel calls:", calls["n"])
print("max|f_rr - f_bgk| over all kernel calls:", calls["max_diff"])
