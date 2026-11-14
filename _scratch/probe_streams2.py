"""Probe 2: streaming implementations with their *native* layout contracts."""
import sys

import numpy as np
import torch

sys.path.insert(0, 'src')
sys.path.insert(0, 'xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/stepwise')
import stepwise_ref as S

rng = np.random.default_rng(0)
ny, nx = 7, 11
f2 = 0.2 + rng.standard_normal((9, ny, nx))
fdef2 = S.definition_stream(f2)

# 1) lattice.stream : (ny, nx, 9)
from tensorlbm import lattice

got = lattice.stream(torch.from_numpy(np.moveaxis(f2, 0, -1).copy())).numpy()
got = np.moveaxis(got.astype(np.float64), -1, 0)
print('lattice.stream (ny,nx,9)  eps=%.3e' % S.compare(got, fdef2)['eps_scaled_inf'])

# 2) les_bptt.stream : (B, 9, ny, nx)
from tensorlbm import les_bptt

got = les_bptt.stream(torch.from_numpy(f2[None].copy())).numpy()[0].astype(np.float64)
print('les_bptt.stream (B,9,ny,nx) eps=%.3e' % S.compare(got, fdef2)['eps_scaled_inf'])

# 3) diff_boundary_grad.stream : (9, ny, nx)
from tensorlbm import diff_boundary_grad

got = diff_boundary_grad.stream(torch.from_numpy(f2.copy())).numpy().astype(np.float64)
print('diff_boundary_grad.stream (9,ny,nx) eps=%.3e' % S.compare(got, fdef2)['eps_scaled_inf'])

# 4) solver.stream : (9, ny, nx)
from tensorlbm import solver

got = solver.stream(torch.from_numpy(f2.copy())).numpy().astype(np.float64)
print('solver.stream (9,ny,nx) eps=%.3e' % S.compare(got, fdef2)['eps_scaled_inf'])

# 5) h_theorem.stream(f, spec): locate LatticeSpec
import tensorlbm.h_theorem as hth

spec_cls = None
for name in dir(hth):
    if 'Spec' in name:
        spec_cls = getattr(hth, name)
        print('h_theorem spec-like:', name, getattr(spec_cls, '__module__', ''))
print('h_theorem.stream sig annotation:', hth.stream.__annotations__)

# 6) channel_kernel_numpy.stream : D3Q19, find layout by trial
from tensorlbm import channel_kernel_numpy as ckn

print('ckn module C shape:', np.asarray(ckn.C).shape, 'OPP len', len(ckn.OPP) if hasattr(ckn, 'OPP') else None)
for shape in [(19, 5, 6, 7), (5, 6, 7, 19)]:
    try:
        f3 = 0.2 + rng.standard_normal(shape)
        out = ckn.stream(f3)
        print('  shape', shape, '-> ok', np.asarray(out).shape)
    except Exception as e:  # noqa: BLE001
        print('  shape', shape, '-> ', type(e).__name__, str(e)[:120])