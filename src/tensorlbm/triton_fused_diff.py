"""Differentiable Triton-fused D3Q19 step: forward kernel + adjoint kernel pair.

Motivation
----------
:mod:`tensorlbm.triton_fused` gives a fast *forward* periodic D3Q19 step
(stream-then-collide fused in one kernel launch) but the result is detached:
there is no path from ``loss`` back to the relaxation time ``tau`` (or to the
initial distribution) through the fused kernel.  :mod:`tensorlbm.autograd_path`
does keep the eager torch operators in the graph, at the price of
materialising every intermediate in DRAM.

This module closes the gap: the *same* audited forward kernel
(:func:`tensorlbm.triton_fused.triton_fused`) is wrapped in a
``torch.autograd.Function`` whose backward is a **second Triton kernel** —
the hand-derived discrete adjoint of the fused step.  So a rollout of N
fused steps costs N kernel launches forward and N kernel launches backward,
instead of the eager graph's ~19 gathers + ~15 elementwise ops per step.

Discrete forward map (per cell ``i``, periodic wrap, pull form)
--------------------------------------------------------------
Streaming and collision are fused as *stream-then-collide* (the operator
order of the audited kernel; equivalently "pull-gather, then relax"):

.. math::

    f^{str}_q(i) &= f_q(i - c_q), \\
    \\rho(i) &= \\sum_q f^{str}_q(i), \\quad
    \\rho u(i) = \\sum_q c_q f^{str}_q(i), \\
    f^{eq}_q(i) &= w_q \\rho \\left[1 + 3 c_q\\!\\cdot\\!u
        + \\tfrac{9}{2}(c_q\\!\\cdot\\!u)^2 - \\tfrac{3}{2}u^2\\right], \\
    g_q(i) &= f^{str}_q(i) - \\frac{1}{\\tau}
        \\left(f^{str}_q(i) - f^{eq}_q(i)\\right).

Adjoint (derived by hand, implemented as one kernel)
----------------------------------------------------
With .. math:: \\bar g = \\partial L/\\partial g and ``k = 1/\\tau``:

* **collision adjoint.**  :math:`\\partial f^{eq}_q/\\partial f^{str}_p
  = w_q A_q + w_q (3 + 9 c_q\\!\\cdot\\!u)\\,(c_p\\!\\cdot\\!u - c_q\\!\\cdot\\!u)`
  with :math:`A_q = 1 + 3c_q\\!\\cdot\\!u + \\tfrac{9}{2}(c_q\\!\\cdot\\!u)^2
  - \\tfrac{3}{2}u^2` (the ``1/\\rho`` from :math:`\\partial u/\\partial f_p`
  cancels the ``\\rho`` prefactor), so with
  :math:`S_0 = \\sum_q \\bar g_q w_q A_q`,
  :math:`S_1 = \\sum_q \\bar g_q w_q (3+9c_q\\!\\cdot\\!u)`,
  :math:`S_2 = \\sum_q \\bar g_q w_q (3+9c_q\\!\\cdot\\!u)(c_q\\!\\cdot\\!u)`:

  .. math:: \\bar f^{str}_p = (1-k)\\,\\bar g_p + k\\,(S_0 - S_2 + (c_p\\!\\cdot\\!u) S_1).

* **streaming adjoint.**  Streaming is a pure permutation, so the adjoint is
  the *scatter* :math:`\\bar f_p(i + c_p) \\mathrel{+}= \\bar f^{str}_p(i)`.
  For fixed ``p`` the map ``i -> i + c_p`` is a bijection, therefore no two
  lanes of one program (and no two programs) touch the same address: the
  scatter needs **no atomics**.
* **tau adjoint.**  :math:`\\partial g_q/\\partial k = -(f^{str}_q - f^{eq}_q)`
  and :math:`\\mathrm{d}k/\\mathrm{d}\\tau = -1/\\tau^2`, hence

  .. math:: \\frac{\\partial L}{\\partial \\tau}
      = \\frac{1}{\\tau^2}\\sum_i \\sum_q \\bar g_q(i)\\,
        \\bigl(f^{str}_q(i) - f^{eq}_q(i)\\bigr).

  Per-program partial sums are written to a scratch buffer (one slot per
  program) and reduced with ``torch.sum`` — no atomic float accumulation, so
  ``dL/dtau`` is **bitwise reproducible** for a fixed grid/block shape.

Conventions and scope
---------------------
* D3Q19, fp32, periodic box, BGK collision (single relaxation time).  No
  walls / inlet / outlet / obstacle: those live in
  :func:`tensorlbm.autograd_path.differentiable_step`, which remains the
  eager reference for boundary-condition cases.
* The eager reference for the *same* discrete map is
  :func:`eager_stream_collide_step` (``collide_bgk3d(stream3d(f), tau)``) —
  i.e. the audited eager primitives, applied in the kernel's operator order.
  ``autograd_path.differentiable_step`` applies collision *before* streaming
  (``C∘S`` rather than ``S∘C``); the two orders are different maps, so a
  gradient comparison must be made against the same map.  Both orders are
  legitimate LBM schemes; this module documents which one it is.
* ``TRITON_INTERPRET=1`` executes the very same kernel source through
  Triton's NumPy interpreter on CPU.  That is what makes the gradient check
  runnable on a CPU-only host; it is *not* a throughput measurement (the
  interpreter is orders of magnitude slower than compiled GPU code).

Public API
----------
:func:`triton_fused_step`, :func:`triton_fused_rollout`,
:func:`eager_stream_collide_step`, :func:`macroscopic`, :func:`is_available`.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .triton_fused import _Q_PAD, make_lattice_tensors, triton_fused

__all__ = [
    "triton_fused_step",
    "triton_fused_rollout",
    "eager_stream_collide_step",
    "macroscopic",
    "is_available",
    "interpreter_mode",
    "DEFAULT_BLOCK",
]


# Tile shape used for both kernels.  The kernels are tile-per-program; a
# square tile keeps the pull offsets a single broadcast, like the forward
# kernel in ``triton_fused``.
DEFAULT_BLOCK = 8


def interpreter_mode() -> bool:
    """True when Triton runs kernels through its NumPy interpreter (CPU)."""
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def is_available() -> bool:
    """True if this module can execute its kernels on the current host.

    Either a CUDA device is visible (compiled kernels) or
    ``TRITON_INTERPRET=1`` is set (interpreter, CPU only).
    """
    return bool(torch.cuda.is_available()) or interpreter_mode()


# ---------------------------------------------------------------------------
# Backward kernel: discrete adjoint of the fused stream-then-collide step
# ---------------------------------------------------------------------------


@triton.jit
def _step_adjoint_kernel(
    f_ptr,
    gout_ptr,
    df_ptr,
    dtau_part_ptr,
    cxi_ptr,
    cyi_ptr,
    czi_ptr,
    cxf_ptr,
    cyf_ptr,
    czf_ptr,
    w_ptr,
    gram_ptr,
    tau_inv,
    nz,
    ny,
    nx,
    stride_q,
    stride_z,
    stride_y,
    stride_x,
    npy,
    npx,
    Q_PAD: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """Reverse-mode adjoint of ``_fused_collide_stream_anydtype``.

    One program owns the same (z-slice, y-tile, x-tile) block as the forward
    kernel, so the pull offsets are identical and the scatter target
    ``i + c_q`` lands inside the periodic box.
    """
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_z = tl.program_id(2)
    pid = pid_z * npy * npx + pid_y * npx + pid_x

    offs_y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offs_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    offs_q = tl.arange(0, Q_PAD)

    mask_q = offs_q < 19
    spatial_mask = (offs_y < ny)[:, None] & (offs_x < nx)[None, :]
    rw_mask = mask_q[:, None, None] & spatial_mask[None, :, :]

    cx_i = tl.load(cxi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]
    cy_i = tl.load(cyi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]
    cz_i = tl.load(czi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]

    # --- 1. replay the forward pull-stream (same offsets as the forward kernel)
    src_z = pid_z - cz_i
    src_z = tl.where(src_z < 0, src_z + nz, tl.where(src_z >= nz, src_z - nz, src_z))
    src_y = offs_y[None, :, None] - cy_i
    src_y = tl.where(src_y < 0, src_y + ny, tl.where(src_y >= ny, src_y - ny, src_y))
    src_x = offs_x[None, None, :] - cx_i
    src_x = tl.where(src_x < 0, src_x + nx, tl.where(src_x >= nx, src_x - nx, src_x))

    src_offs = (
        offs_q.to(tl.int64)[:, None, None] * stride_q
        + src_z.to(tl.int64) * stride_z
        + src_y.to(tl.int64) * stride_y
        + src_x.to(tl.int64) * stride_x
    )
    f_str = tl.load(f_ptr + src_offs, mask=rw_mask, other=0.0)

    # --- 2. replay the post-stream macroscopic state (needed by the adjoint)
    cx_b = tl.load(cxf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    cy_b = tl.load(cyf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    cz_b = tl.load(czf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    w_b = tl.load(w_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]

    rho = tl.sum(f_str, axis=0)
    rho_safe = tl.where(rho > 1e-12, rho, 1e-12)
    ux = tl.sum(cx_b * f_str, axis=0) / rho_safe
    uy = tl.sum(cy_b * f_str, axis=0) / rho_safe
    uz = tl.sum(cz_b * f_str, axis=0) / rho_safe
    usq = ux * ux + uy * uy + uz * uz
    cu = cx_b * ux[None, :, :] + cy_b * uy[None, :, :] + cz_b * uz[None, :, :]

    a_q = 1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None, :, :]
    feq = rho_safe[None, :, :] * w_b * a_q

    # --- 3. collision adjoint at the destination cell (adjoint has no shift)
    g = tl.load(
        gout_ptr
        + offs_q.to(tl.int64)[:, None, None] * stride_q
        + pid_z.to(tl.int64) * stride_z
        + offs_y.to(tl.int64)[None, :, None] * stride_y
        + offs_x.to(tl.int64)[None, None, :] * stride_x,
        mask=rw_mask,
        other=0.0,
    )

    # d feq_q / d f_str_p = w_q A_q + w_q (3+9cu_q)(c_q.c_p - cu_q)
    #                       - 3 w_q (cu_p - u^2)
    # The c_q.c_p factor is the lattice GRAM matrix (0/1/2 entries) -- it is
    # NOT cu_p = c_p.u (that substitution is wrong and was caught by the
    # gradient check).  Contracting the Gram matrix with
    # h_q = gbar_q w_q (3+9cu_q) is a 19x19 matmul per cell block, done with
    # tl.dot in ieee precision (tf32 would cost ~1e-3 and break the 1e-5 gate).
    w_a = w_b * a_q
    h = g * w_b * (3.0 + 9.0 * cu)
    s0 = tl.sum(g * w_a, axis=0)
    s2 = tl.sum(h * cu, axis=0)
    s_t = tl.sum(g * w_b, axis=0)  # sum_q gbar_q w_q
    gram = tl.load(
        gram_ptr + offs_q[:, None] * Q_PAD + offs_q[None, :]
    )
    h2 = tl.reshape(h, (Q_PAD, BLOCK_Y * BLOCK_X))
    gram_h = tl.reshape(
        tl.dot(gram, h2, input_precision="ieee"), (Q_PAD, BLOCK_Y, BLOCK_X)
    )
    dfeq = (
        s0[None, :, :]
        - s2[None, :, :]
        + gram_h
        - 3.0 * cu * s_t[None, :, :]
        + 3.0 * usq[None, :, :] * s_t[None, :, :]
    )
    df_str = (1.0 - tau_inv) * g + tau_inv * dfeq

    # --- 4. streaming adjoint: scatter to i - c_q (bijective per lane, no atomics)
    # ``f_str_q(i) = f_q(i - c_q)`` is a permutation, so its adjoint is a
    # scatter *back along the same edge*: ``df_q(i - c_q) += df_str_q(i)``.
    # The target offsets are therefore the pull offsets ``src_*`` (wrapped, so
    # always inside the box); for fixed q the map i -> i - c_q is a bijection,
    # hence no two lanes/programs write the same (cell, q) slot.
    tl.store(df_ptr + src_offs, df_str, mask=rw_mask)

    # --- 5. tau adjoint partial sum (deterministic: one slot per program)
    part = tl.sum(tl.sum(g * (f_str - feq), axis=0))
    tl.store(dtau_part_ptr + pid, part)


# ---------------------------------------------------------------------------
# Kernel launchers
# ---------------------------------------------------------------------------


_GRAM_CACHE: dict[str, torch.Tensor] = {}


def _gram_matrix(device) -> torch.Tensor:
    """Padded lattice Gram matrix ``G[p, q] = c_p . c_q`` (``Q_PAD x Q_PAD``).

    Exact integer entries (0/1/2) stored in fp32; zero-padded lanes decouple,
    so ``tl.dot(G, h)`` returns the 19 real contractions unchanged.
    """
    import torch as _t

    from .d3q19 import C

    key = str(device)
    if key not in _GRAM_CACHE:
        c = C.to(_t.float32)
        g = _t.zeros((_Q_PAD, _Q_PAD), dtype=_t.float32, device=device)
        g[:19, :19] = c @ c.T
        _GRAM_CACHE[key] = g
    return _GRAM_CACHE[key]


def _grid(ny: int, nx: int, nz: int, block: int):
    return (triton.cdiv(ny, block), triton.cdiv(nx, block), nz)


def _fused_step_fwd(f: torch.Tensor, tau: float, block: int) -> torch.Tensor:
    return triton_fused(f, tau, block_x=block, block_y=block)


def _fused_step_bwd(
    f: torch.Tensor, gout: torch.Tensor, tau: float, block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the adjoint kernel; return ``(dL/df, dL/dtau)``."""
    Q, nz, ny, nx = f.shape
    if Q != 19:
        raise ValueError(f"expected 19 populations, got {Q}")
    lat = make_lattice_tensors(str(f.device))
    gram = _gram_matrix(f.device)
    df = torch.zeros_like(f)
    grid = _grid(ny, nx, nz, block)
    nprog = grid[0] * grid[1] * grid[2]
    dtau_part = torch.zeros(nprog, dtype=f.dtype, device=f.device)
    _step_adjoint_kernel[grid](
        f,
        gout,
        df,
        dtau_part,
        lat["cxi"],
        lat["cyi"],
        lat["czi"],
        lat["cxf"],
        lat["cyf"],
        lat["czf"],
        lat["w"],
        gram,
        1.0 / tau,
        nz,
        ny,
        nx,
        f.stride(0),
        f.stride(1),
        f.stride(2),
        f.stride(3),
        grid[0],
        grid[1],
        Q_PAD=_Q_PAD,
        BLOCK_X=block,
        BLOCK_Y=block,
    )
    dtau = dtau_part.sum() / (tau * tau)
    return df, dtau


# ---------------------------------------------------------------------------
# Autograd wiring
# ---------------------------------------------------------------------------


class _FusedStep(torch.autograd.Function):
    """One fused periodic D3Q19 step with an adjoint-kernel backward."""

    @staticmethod
    def forward(ctx, f: torch.Tensor, tau: torch.Tensor, block: int):
        tau_value = float(tau)
        if not (tau_value > 0.5):
            raise ValueError(
                f"tau must be > 0.5 for the BGK kernel, got {tau_value}"
            )
        if f.dtype != torch.float32:
            raise TypeError(
                "the fused Triton step is fp32-only; use "
                "eager_stream_collide_step for fp64 reference runs"
            )
        out = _fused_step_fwd(f.detach().contiguous(), tau_value, block)
        ctx.save_for_backward(f.detach())
        ctx.tau = tau_value
        ctx.block = block
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (f,) = ctx.saved_tensors
        df, dtau = _fused_step_bwd(
            f, grad_out.detach().contiguous(), ctx.tau, ctx.block
        )
        return df, torch.as_tensor(dtau, dtype=f.dtype, device=f.device), None


def triton_fused_step(
    f: torch.Tensor, tau: torch.Tensor | float, *, block: int = DEFAULT_BLOCK
) -> torch.Tensor:
    """One differentiable fused step: ``f -> collide(stream(f))``.

    Args:
        f: ``(19, nz, ny, nx)`` fp32 distribution (periodic box).
        tau: Relaxation time; a 0-dim tensor keeps ``dL/dtau`` in the graph.

    Returns:
        ``(19, nz, ny, nx)`` post-step distribution, differentiable w.r.t.
        both ``f`` and ``tau``.
    """
    tau_t = torch.as_tensor(tau, dtype=f.dtype, device=f.device)
    if not tau_t.requires_grad and torch.is_grad_enabled():
        # keep the node: the adjoint kernel computes dL/dtau anyway
        tau_t = tau_t.detach().requires_grad_(True)
    return _FusedStep.apply(f, tau_t, block)


def triton_fused_rollout(
    f0: torch.Tensor,
    tau: torch.Tensor | float,
    steps: int,
    *,
    block: int = DEFAULT_BLOCK,
) -> torch.Tensor:
    """``steps`` chained :func:`triton_fused_step` calls (one kernel each)."""
    f = f0
    for _ in range(int(steps)):
        f = triton_fused_step(f, tau, block=block)
    return f


# ---------------------------------------------------------------------------
# Eager reference (same discrete map, torch autograd) and helpers
# ---------------------------------------------------------------------------


def eager_stream_collide_step(f: torch.Tensor, tau) -> torch.Tensor:
    """Eager reference for the *same* map: ``collide_bgk3d(stream3d(f), tau)``.

    Built from the audited eager primitives in
    :mod:`tensorlbm.solver3d`, so it is differentiable for any float dtype
    (fp64 is used for the finite-difference cross-check).
    """
    from .solver3d import collide_bgk3d, stream3d

    tau_t = torch.as_tensor(tau, dtype=f.dtype, device=f.device)
    return collide_bgk3d(stream3d(f), tau_t)


def macroscopic(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rho, u)`` with ``u`` of shape ``(3, nz, ny, nx)``; differentiable."""
    from .d3q19 import C

    c = C.to(dtype=f.dtype, device=f.device)  # (3, 19)
    rho = f.sum(dim=0)
    u = torch.einsum("dq,dzyx->qzyx", c, f) / rho.unsqueeze(0)
    return rho, u