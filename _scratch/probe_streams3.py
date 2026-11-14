"""Probe 3: h_theorem.stream(spec) and channel_kernel_numpy D3Q19 stream axes."""
import inspect
import sys

import numpy as np
import torch

sys.path.insert(0, 'src')
sys.path.insert(0, 'xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/stepwise')
import stepwise_ref as S
import lbmpy_pointwise_ref as ref

import tensorlbm.h_theorem as hth

print('LatticeSpec sig:', inspect.signature(hth.LatticeSpec))
print('LatticeSpec fields:', [f for f in dir(hth.LatticeSpec) if not f.startswith('_')][:20])

rng = np.random.default_rng(1)

# --- h_theorem.stream
from tensorlbm.d2q9 import C as C2, W_EXACT
try:
    spec = hth.LatticeSpec(c=C2.numpy(), w=np.array(W_EXACT))
    print('spec ok', spec)
except Exception as e:  # noqa: BLE001
    print('spec build failed:', type(e).__name__, str(e)[:200])
    spec = None
if spec is not None:
    f = 0.2 + rng.standard_normal((9, 7, 11))
    got = hth.stream(torch.from_numpy(f.copy()), spec).numpy().astype(np.float64)
    print('h_theorem.stream eps=%.3e' % S.compare(got, S.definition_stream(f))['eps_scaled_inf'])

# --- channel_kernel_numpy D3Q19: which axes?
from tensorlbm import channel_kernel_numpy as ckn
C19 = np.asarray(ckn.C)
print('ckn.C[:5]', C19[:5].tolist())
print('ref D3Q19 vel[:5]', ref.tensorlbm_velocities('D3Q19')[:5].tolist())
print('same order?', bool(np.allclose(C19, ref.tensorlbm_velocities('D3Q19'))))


def definition_stream3d(f, vel, axes=((0, 1, 2),)):
    out = np.empty_like(f)
    for q in range(f.shape[0]):
        sh = tuple(int(v) for v in vel[q])
        out[q] = np.roll(f[q], shift=sh, axis=(0, 1, 2))
    return out


nz, ny, nx = 5, 6, 7
f3 = 0.2 + rng.standard_normal((19, nz, ny, nx))
got = np.asarray(ckn.stream(f3), dtype=np.float64)
# try velocity orderings / axis permutations
best = None
for name, vel in (('ckn.C as (cz,cy,cx)', C19),
                  ('ckn.C as (cz,cx,cy)', C19[:, [0, 2, 1]]),
                  ('ref D3Q19', ref.tensorlbm_velocities('D3Q19'))):
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 1, 0), (1, 2, 0), (2, 0, 1)]:
        v = vel[:, perm]
        cand = definition_stream3d(f3, v)
        eps = S.compare(got, cand)['eps_scaled_inf']
        if best is None or eps < best[0]:
            best = (eps, name, perm)
print('ckn best match: eps=%.3e ordering=%s axes-perm=%s' % best)