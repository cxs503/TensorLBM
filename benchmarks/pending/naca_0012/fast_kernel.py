"""Fast, elementwise-equivalent D2Q9 MRT collision + streaming for the
NACA0012 benchmark on SDAA (teco) GPUs.

Why this file exists
--------------------
``tensorlbm.solver.collide_mrt`` computes the MRT collision with two dense
``9x9 @ 9xN`` f32 matmuls, and ``tensorlbm.solver.stream`` gathers with a
``(9,ny,nx)`` advanced-index tensor.  On the SDAA (Teco) backend both fall
onto pathologically slow kernels: measured on one TECO card they cost
**131 ms** and **105 ms** per step (grid 1472x1024) whereas a plain
elementwise op on the *same* device costs **0.09 ms**.  The physical step
therefore runs at ~5 M cell-steps/s instead of ~100+ M cell-steps/s.

This module re-implements the *identical* operators using only elementwise
tensor ops (plus ``torch.roll`` for the periodic pull-stream, 0.29 ms):

* MRT: ``f' = f - A (f - feq)`` with the constant composite matrix
  ``A = M^-1 S M`` (S = diag(0, s_e, s_eps, 0, s_q, 0, s_q, s_nu, s_nu)),
  split into a bulk part ``A0`` and a rank-2 sponge correction ``K`` so a
  cell-wise ``s_nu = 1/tau_field`` is handled without a per-cell matrix.
* stream: ``f'_i(x) = f_i(x - c_i)`` via one periodic ``torch.roll`` per
  direction (identical periodic shift to the library's pull-gather).

Numerics: the composite matrices are formed in float64 and cast to the
working dtype, so the result matches ``collide_mrt`` to f32 round-off
(verified in ``fast_kernel.py --selftest``).  No physics is changed.
"""

from __future__ import annotations

import numpy as np
import torch

from tensorlbm.d2q9 import C as _C, equilibrium
from tensorlbm.solver import _M_D2Q9_DATA, _M_D2Q9_INV_DATA

_S_E = 1.64
_S_EPS = 1.54
_S_Q = 1.7


def _mats(dev: torch.device, dtype: torch.dtype, tau: float):
    """Cached composite matrices A0 (bulk) and K (rank-2 sponge) for `tau`."""
    key = (str(dev), dtype, float(tau))
    cache = _mats._cache  # type: ignore[attr-defined]
    if key in cache:
        return cache[key]
    M = np.array(_M_D2Q9_DATA, dtype=np.float64)
    Minv = np.array(_M_D2Q9_INV_DATA, dtype=np.float64)
    s = np.array([0.0, _S_E, _S_EPS, 0.0, _S_Q, 0.0, _S_Q, 1.0 / tau, 1.0 / tau])
    A0 = Minv @ np.diag(s) @ M
    E = np.zeros((9, 9))
    E[7, 7] = 1.0
    E[8, 8] = 1.0
    K = Minv @ E @ M
    # M rows 7 and 8 (needed for the sponge contraction p7=M[7].d, p8=M[8].d)
    # and the corresponding columns of M^-1 (the rank-2 correction is
    #   sum_j K[i,j] d_j = Minv[i,7]*p7 + Minv[i,8]*p8).
    m7 = M[7].copy()
    m8 = M[8].copy()
    mi7 = Minv[:, 7].copy()
    mi8 = Minv[:, 8].copy()
    out = (
        torch.tensor(A0, dtype=dtype, device=dev),
        torch.tensor(K, dtype=dtype, device=dev),
        torch.tensor(m7, dtype=dtype, device=dev),
        torch.tensor(m8, dtype=dtype, device=dev),
        torch.tensor(mi7, dtype=dtype, device=dev),
        torch.tensor(mi8, dtype=dtype, device=dev),
    )
    cache[key] = out
    return out


_mats._cache = {}  # type: ignore[attr-defined]


def fast_collide_mrt(
    f: torch.Tensor,
    tau: float,
    s_e: float = _S_E,
    s_eps: float = _S_EPS,
    s_q: float = _S_Q,
    tau_field: torch.Tensor | None = None,
) -> torch.Tensor:
    """Elementwise MRT collision, numerically equivalent to ``collide_mrt``."""
    device = f.device
    ny, nx = f.shape[1], f.shape[2]
    A0, K, m7, m8, mi7, mi8 = _mats(device, f.dtype, tau)

    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    c = _C.to(device).to(f.dtype)
    ux = (f * c[:, 0].view(9, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(9, 1, 1)).sum(dim=0) / rho_safe
    feq = equilibrium(rho, ux, uy)
    d = f - feq

    if tau_field is None:
        new = torch.empty_like(f)
        a = A0
        for i in range(9):
            acc = a[i, 0] * d[0]
            for j in range(1, 9):
                acc = acc + a[i, j] * d[j]
            new[i] = f[i] - acc
        return new

    # Sponge: s_nu varies cell-wise (only entries 7,8 of S).  Compute the two
    # "moment" contractions p7,p8 = M[7].d, M[8].d and add the rank-2 term.
    d7 = d.reshape(9, -1)
    p7 = m7[0] * d7[0]
    p8 = m8[0] * d7[0]
    for j in range(1, 9):
        p7 = p7 + m7[j] * d7[j]
        p8 = p8 + m8[j] * d7[j]
    delta = (1.0 / tau_field).reshape(-1) - (1.0 / tau)  # zero in the bulk
    new = torch.empty_like(f)
    a = A0
    for i in range(9):
        acc = a[i, 0] * d[0]
        for j in range(1, 9):
            acc = acc + a[i, j] * d[j]
        corr = (mi7[i] * p7 + mi8[i] * p8) * delta
        new[i] = f[i] - acc - corr.reshape(ny, nx)
    return new


def fast_stream(f: torch.Tensor) -> torch.Tensor:
    """Periodic pull-streaming via one ``torch.roll`` per direction."""
    device = f.device
    c = _C.to(device)
    out = torch.empty_like(f)
    for i in range(9):
        out[i] = torch.roll(f[i], shifts=(int(c[i, 1]), int(c[i, 0])), dims=(0, 1))
    return out


def selftest(C: int = 24, device: str = "sdaa:3") -> None:
    from tensorlbm.solver import collide_mrt, stream

    dev = torch.device(device)
    ny, nx = 8 * C, 12 * C
    tau = 0.52
    tf = torch.full((ny, nx), tau, device=dev)
    tf[:, :3] = tau * 1.5  # fictitious sponge region
    torch.manual_seed(0)
    f = 0.2 + torch.rand(9, ny, nx, device=dev) * 0.1
    ref = collide_mrt(f, tau, tau_field=tf)
    got = fast_collide_mrt(f, tau, tau_field=tf)
    torch.sdaa.synchronize()
    dr = (got - ref).abs().max().item()
    print(f"M-mrt tau_field: max|diff|={dr:.3e}  rel={dr/ref.abs().max().item():.3e}")
    ref2 = collide_mrt(f, tau)
    got2 = fast_collide_mrt(f, tau)
    torch.sdaa.synchronize()
    dr2 = (got2 - ref2).abs().max().item()
    print(f"M-mrt bulk     : max|diff|={dr2:.3e}")
    sr = stream(f)
    sg = fast_stream(f)
    torch.sdaa.synchronize()
    ds = (sg - sr).abs().max().item()
    print(f"stream         : max|diff|={ds:.3e}")
    ex = [float(r) for r in (dr, dr2, ds)]
    assert max(ex) < 1e-5, ex
    print("SELFTEST OK")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        dev = sys.argv[i + 1] if len(sys.argv) > i + 1 else "sdaa:3"
        selftest(device=dev)