"""Round-4 feasibility probe: can lbmpy build D3Q27 SRT/TRT/MRT/cumulant
pointwise references, and do the repo kernels agree with them?

Throwaway probe (not part of the gate); kept as the provenance of the
round-4 arm selection, i.e. which operators were found to be algebraically
comparable at all before any threshold was fixed.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
XV = HERE.parent
REPO = XV.parents[1]
for p in (str(REPO / "src"), str(XV), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import lbmpy_pointwise_ref as lpr  # noqa: E402
import refcore27 as core  # noqa: E402
from tensorlbm.cumulant import (  # noqa: E402
    collide_cumulant_d3q27,
    collide_cumulant_geier_d3q27,
)
from tensorlbm.d3q27 import (  # noqa: E402
    _M_D3Q27_DATA,
    collide_bgk27,
    collide_mrt27,
    collide_trt27,
)

tau = 0.7
rng = np.random.default_rng(0)
shape = (4, 4, 4)
rho = 1.0 + 0.01 * rng.standard_normal(shape)
u = 0.05 * rng.standard_normal((3,) + shape)
f = core.discrete_equilibrium(rho, u) + 1e-3 * rng.standard_normal((27,) + shape)
ft = torch.tensor(f)


def cmp(name, got, ref):
    print("%-32s eps_inf=%.3e" % (name, core.compare(got, ref)["eps_scaled_inf"]), flush=True)


t0 = time.time()
ref = lpr.PointwiseReference("D3Q27", "srt", tau=tau, continuous_equilibrium=False)
print("build srt %.1fs" % (time.time() - t0), flush=True)
cmp("bgk27 vs srt", collide_bgk27(ft, tau).numpy(), ref.apply(f))

t0 = time.time()
rt = lpr.PointwiseReference("D3Q27", "trt", tau_plus=tau)
print("build trt %.1fs" % (time.time() - t0), flush=True)
cmp("trt27 vs trt", collide_trt27(ft, tau).numpy(), rt.apply(f))

M = np.asarray(_M_D3Q27_DATA, dtype=float)
s_e, s_q = 1.19, 1.2
rates = [0.0, 0.0, 0.0, 0.0, s_e] + [1.0 / tau] * 5 + [s_q] * 9 + [s_e] + [s_e] * 7
t0 = time.time()
rm = lpr.PointwiseReference("D3Q27", "mrt_from_matrix", moment_matrix=M, relaxation_rates=rates)
print("build mrt %.1fs" % (time.time() - t0), flush=True)
print("basis gap", lpr.moment_basis_check(M, "D3Q27"), flush=True)
cmp("mrt27 vs mrt_from_matrix", collide_mrt27(ft, tau, s_e=s_e, s_q=s_q).numpy(), rm.apply(f))

for model in ("cumulant", "cumulant_discrete_eq"):
    t0 = time.time()
    try:
        rc = lpr.PointwiseReference("D3Q27", model, tau=tau)
        print("build %s %.1fs" % (model, time.time() - t0), flush=True)
        fref = rc.apply(f)
        cmp("geier vs " + model, collide_cumulant_geier_d3q27(ft, tau).numpy(), fref)
        cmp("legacy vs " + model, collide_cumulant_d3q27(ft, tau).numpy(), fref)
    except Exception as exc:  # noqa: BLE001
        print("  %s FAILED: %r" % (model, exc), flush=True)