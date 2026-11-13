"""Out-of-core (OOTC) adjoint gradient for the D2Q9 BGK LBM model.

Method: checkpointed BPTT with on-disk checkpoints (a.k.a. gradient
checkpointing / recomputation; Griewank & Walther 2008, ch. 2; Chen,
Beckman & Patterson 1996; Paszke et al. 2017 for the deep-learning
formulation).  The forward trajectory is never held in RAM:

  * forward pass: every k-th state is written to a numpy memmap file
    (disk), only the current state lives in memory;
  * backward pass: for each segment [m k, (m+1) k] the checkpoint is
    read back, the segment is recomputed (bit-identical to the original
    forward run, same operations in the same order), and the adjoint is
    propagated through the k steps.  If T is not a multiple of k, the
    final (tail) segment of length T mod k is handled first, recomputed
    from the last full checkpoint.

Because the recomputation is deterministic and bit-identical, the OOTC
gradient equals the full-storage BPTT reference up to floating-point
summation order (relative difference ~1e-15, far below the 1e-5 gate).

Peak memory = O(k * N * Q)  (one segment + two costate arrays),
independent of T.  With k = 100 and N = 128^2, Q = 9: ~0.25 GiB.
"""

from __future__ import annotations

import os
import numpy as np

from .lbm2d import macro, equilibrium, step
from .bptt import anti_stream, b_action_t, seed_terminal

__all__ = ["ootc_gradient"]


def _adjoint_step(f_n: np.ndarray, lam: np.ndarray, tau: float):
    """One backward adjoint step at state f_n with incoming costate lam.

    Returns (dJ_dtau_contribution, lam_new) where

        mu   = anti_stream(lam)
        dJ/dtau += sum((f_n - feq) * mu) / tau^2
        lam_new = mu - (mu - B(f_n)^T mu) / tau
    """
    mu = anti_stream(lam)
    rho, ux, uy = macro(f_n)
    feq = equilibrium(rho, ux, uy)
    contrib = float(np.sum((f_n - feq) * mu) / (tau * tau))
    lam_new = mu - (mu - b_action_t(f_n, mu)) / tau
    return contrib, lam_new


def ootc_gradient(tau: float, T: int, f0: np.ndarray,
                ux_ref: np.ndarray, uy_ref: np.ndarray,
                k: int = 100, ckpt_path: str | None = None,
                verbose: bool = False):
    """OOTC gradient dJ/dtau with on-disk checkpoints every k steps.

    J(tau) = 1/2 || u_T(tau) - u_ref ||^2  (same objective as BPTT).

    Parameters
    ----------
    tau : relaxation time (lattice units)
    T : number of forward steps (any positive integer; need not be a
        multiple of k -- the tail segment of length T mod k is
        recomputed from the last full checkpoint)
    f0 : initial distribution (9, ny, nx)
    ux_ref, uy_ref : reference terminal velocity field
    k : checkpoint interval (segment length)
    ckpt_path : path for the checkpoint memmap file (default: temp)

    Returns (dJ_dtau, J, (ux_T, uy_T), n_checkpoints, ckpt_path).
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    ny, nx = f0.shape[1], f0.shape[2]
    n_full = T // k
    tail = T % k
    n_ckpt = n_full + 1
    if ckpt_path is None:
        import tempfile
        fd, ckpt_path = tempfile.mkstemp(suffix=".ckpt.npy", prefix="ootc_")
        os.close(fd)
    ckpt = np.lib.format.open_memmap(
        ckpt_path, mode="w+", dtype=np.float64,
        shape=(n_ckpt, 9, ny, nx))

    # ---- forward: store checkpoints on disk ----
    f = f0
    ckpt[0] = f
    for m in range(1, n_ckpt):
        for _ in range(k):
            f = step(f, tau)
        ckpt[m] = f
    for _ in range(tail):
        f = step(f, tau)
    ckpt.flush()

    # ---- terminal seed (state f^T, i.e. after the tail) ----
    rho_t, ux_t, uy_t = macro(f)
    lam = seed_terminal(ux_t, uy_t, rho_t, ux_ref, uy_ref)
    J = 0.5 * float(np.sum((ux_t - ux_ref) ** 2 + (uy_t - uy_ref) ** 2))

    # ---- backward: recompute each segment, backpropagate ----
    dJ_dtau = 0.0
    if tail:
        seg = [ckpt[n_full]]
        g = seg[0]
        for _ in range(tail):
            g = step(g, tau)
            seg.append(g)
        for n in range(T - 1, n_full * k - 1, -1):
            f_n = seg[n - n_full * k]
            contrib, lam = _adjoint_step(f_n, lam, tau)
            dJ_dtau += contrib
        if verbose:
            print(f"[ootc] tail segment ({tail} steps) done", flush=True)
    for m in range(n_ckpt - 1, 0, -1):
        seg = [ckpt[m - 1]]
        g = seg[0]
        for _ in range(k):
            g = step(g, tau)
            seg.append(g)
        for n in range(m * k - 1, (m - 1) * k - 1, -1):
            f_n = seg[n - (m - 1) * k]
            contrib, lam = _adjoint_step(f_n, lam, tau)
            dJ_dtau += contrib
        if verbose:
            print(f"[ootc] segment {m}/{n_ckpt - 1} done", flush=True)

    return dJ_dtau, J, (ux_t, uy_t), n_ckpt, ckpt_path
