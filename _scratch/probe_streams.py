"""Probe: do all in-repo streaming implementations agree with the definition?"""
import inspect
import sys

import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/stepwise')
import stepwise_ref as S

import torch

cands = [
    ('tensorlbm.lattice', 'stream'),
    ('tensorlbm.les_bptt', 'stream'),
    ('tensorlbm.diff_boundary_grad', 'stream'),
    ('tensorlbm.h_theorem', 'stream'),
    ('tensorlbm.channel_kernel_numpy', 'stream'),
    ('tensorlbm.solver', 'stream'),
]

rng = np.random.default_rng(0)
f = 0.2 + rng.standard_normal((9, 7, 11))
fdef = S.definition_stream(f)

for mod_name, fn_name in cands:
    try:
        __import__(mod_name)
        mod = sys.modules[mod_name]
        fn = getattr(mod, fn_name)
        print('---', mod_name, fn_name, inspect.signature(fn))
        if mod_name.endswith('channel_kernel_numpy'):
            got = np.asarray(fn(f), dtype=np.float64)
        elif mod_name.endswith('h_theorem'):
            from tensorlbm.lattice import LatticeSpec
            # try to find a spec constructor
            print('   LatticeSpec sig', inspect.signature(LatticeSpec))
            try:
                spec = LatticeSpec.d2q9() if hasattr(LatticeSpec, 'd2q9') else LatticeSpec()
            except Exception as e:  # noqa: BLE001
                print('   spec build failed', e)
                continue
            got = fn(torch.from_numpy(f), spec).numpy().astype(np.float64)
        else:
            got = fn(torch.from_numpy(f)).numpy().astype(np.float64)
        c = S.compare(got, fdef)
        print('   eps=%.3e' % c['eps_scaled_inf'])
    except Exception as e:  # noqa: BLE001
        print('   FAILED:', type(e).__name__, str(e)[:200])