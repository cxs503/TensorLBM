"""Triton-fused SUBOFF step backend.

Lives in the ``tensorlbm.backends`` layer alongside
:mod:`tensorlbm.backends.torch_backend`, :mod:`paddle_backend` and
:mod:`mindspore_backend`.  Unlike the framework backends, Triton is a
*kernel compiler* — it does not replace the underlying tensor library;
the active backend must still be PyTorch (CUDA) for the Triton kernels
to run.  This module provides the **step implementation** for the
SUBOFF production runner when ``use_triton_step=True``.

The step fuses (one Triton kernel launch + small post-ops):

*  pull-stream (x-as-streamwise axis convention matching production SUBOFF)
*  wet-node bounce-back (Ladd 1994, half-way)
*  BGK collide + LES Smagorinsky SGS tau-eff
*  per-cell force reduction (Ladd momentum-exchange)
*  6-face far-field Dirichlet BC writes (PyTorch, O(Q·area))
*  mass correction every ``mass_period`` steps (PyTorch, GPU↔CPU sync)

The implementation reuses the lattice tables and Triton kernels from
``tensorlbm_triton_fused_obstacle`` (the canonical kernel module).
"""
from __future__ import annotations

import torch

try:
    # Pre-deployment path: kernels live at the repo root with
    # underscore-separated names.
    from tensorlbm_triton_fused_obstacle import (
        _Q_PAD,
        _compute_uniform_equilibrium_vec,
        apply_far_field_bc_6face,
        apply_mass_correction,
        triton_fused_obstacle_xfar_les,
    )
except ImportError:
    from tensorlbm.triton_fused_obstacle import (  # type: ignore
        _Q_PAD,
        _compute_uniform_equilibrium_vec,
        apply_far_field_bc_6face,
        apply_mass_correction,
        triton_fused_obstacle_xfar_les,
    )

# Swap table for "swap-at-solid" boundary condition.  Matches PyTorch's
# ``bounce_back_cells_3d(f, solid)`` which is called inside
# ``far_field_bc_3d``.  Forces interior solid cells to be symmetric
# (f[1]=f[2], f[3]=f[4], ...) so that the *next* collide gives u=0 at
# solid cells, enforcing no-slip at the fluid-solid interface.
from tensorlbm.d3q19 import OPPOSITE as _OPPOSITE_Q19

# Production Q for D3Q19 (matches production SUBOFF runner).
_PROD_Q = 19


__all__ = [
    "TritonStepState",
    "triton_suboff_step",
    "is_available",
]


def is_available() -> bool:
    """Return True iff the Triton kernel backend can run on this host.

    Requires CUDA.  The Triton kernels compile on first launch (slow)
    and cache for subsequent calls.
    """
    return torch.cuda.is_available() and _Q_PAD > 0


class TritonStepState:
    """Persistent buffers for the Triton-fused SUBOFF step.

    Holds the ping-pong output tensor (Q=19, matching production) and a
    persistent ``feq_pad_buf`` scratch for :func:`apply_far_field_bc_6face`
    (avoids a per-step allocation).  Persistent force-reduction scalars
    keep allocations out of the per-step hot loop.

    The V2 kernel (``_fused_v2_kernel_xfar_les``) accepts Q=19 buffers
    directly; the old Q=32 padded-input scratch is gone.

    Use :func:`triton_suboff_step` (which auto-manages state) unless you
    want to drive the kernels directly.
    """

    __slots__ = (
        "f_buf", "f_buf_alt", "feq_pad_buf", "feq_vec_buf", "feq_vec_u",
        "fx_buf", "fy_buf", "fz_buf", "tau_eff_buf",
    )

    def __init__(
        self,
        f: torch.Tensor,
        device: torch.device,
    ) -> None:
        # Internal buffers are Q=19 (production D3Q19).  V2 kernel masks
        # the internal Q=32 arange to 19 internally, so the public buffer
        # can match the production layout.
        spatial = f.shape[1:]
        self.f_buf = torch.empty(
            (_PROD_Q, *spatial), dtype=f.dtype, device=device,
        )
        # Second half of the ping-pong pair.  The kernel streams from
        # neighbour cells, so its input and output MUST be distinct
        # buffers: with a single buffer the neighbour loads race against
        # stores from other programs and the flow field silently
        # corrupts (drag came out ~2x too high).  Callers feed the
        # returned tensor straight back in as ``f``, so
        # :func:`triton_suboff_step` picks whichever of the two is not
        # currently aliased to ``f``.
        self.f_buf_alt = torch.empty(
            (_PROD_Q, *spatial), dtype=f.dtype, device=device,
        )
        # Persistent 1-D scratch for ``apply_far_field_bc_6face`` when
        # the cache (``feq_vec_buf``) is shorter than f's Q-channel count
        # (currently always 1-D shape ``(Q,)`` since ``feq_vec_buf`` is
        # also 1-D).  Allocated once; reused per step.
        self.feq_pad_buf = torch.zeros(
            (_PROD_Q,), dtype=f.dtype, device=device,
        )
        # Persistent cached equilibrium ``(Q,)`` vector for the BC.  The
        # actual equilibrium for the runner is computed lazily on the
        # first ``triton_suboff_step`` call (u_in is not known at
        # construction time).  When ``feq_vec_buf is None``, BC takes
        # the slow full-grid-equilibrium path.
        self.feq_vec_buf: torch.Tensor | None = None
        self.feq_vec_u: tuple[float, float, float] | None = None
        self.fx_buf = torch.zeros((), dtype=torch.float32, device=device)
        self.fy_buf = torch.zeros((), dtype=torch.float32, device=device)
        self.fz_buf = torch.zeros((), dtype=torch.float32, device=device)
        # Persistent per-cell tau_eff buffer for WALE / Vreman SGS coupling
        # (Phase 3 — external tau_eff for CM/CUMULANT/BGK).  Shape
        # ``(nz, ny, nx)``, fp32, allocated once per runner.
        self.tau_eff_buf = torch.zeros(
            spatial, dtype=torch.float32, device=device,
        )


def triton_suboff_step(
    f: torch.Tensor,
    obstacle_int8: torch.Tensor,
    u_in: float,
    step: int,
    mass_period: int,
    initial_mass: float,
    nu_lb: float,
    Cs: float = 0.1,
    delta: float = 1.0,
    *,
    collision: str = "BGK",
    state: TritonStepState | None = None,
    do_mass_correction: bool = True,
    tau_eff: torch.Tensor | None = None,
    compute_force: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One fused SUBOFF step on a CUDA device.

    Args:
        f: Distribution tensor ``(Q=19, nz, ny, nx)``, fp32, on CUDA.
            Treated as the *post-collision* (pre-stream) state.  V2 kernel
            accepts Q=19 directly (uses ``tl.arange(0, 32)`` internally
            with ``mask_q = offs_q < 19``).
        obstacle_int8: ``int8[NZ, NY, NX]`` wall mask (1 = wall).
        u_in: Free-stream x-velocity (lattice units).
        step: Current step number (1-indexed).
        mass_period: Apply mass correction every ``mass_period`` steps.
        initial_mass: Target mass (sum of distributions at t=0).
        nu_lb: Molecular kinematic viscosity.
        Cs: Smagorinsky constant.
        delta: LES filter width.
        collision: Collision family — ``"BGK"``, ``"CM"``, or
            ``"CUMULANT"``.  Selects the collide sub-branch in the
            Triton kernel via ``COLLISION: tl.constexpr``.  KBC is not
            supported here (use the PyTorch 5-op chain).
        state: Optional persistent state.  Allocated on first call if
            None; reused on subsequent calls.
        do_mass_correction: If False, skip mass correction entirely
            (caller responsibility).  When True, mass correction runs
            every ``mass_period`` steps starting at step ``mass_period``.
        tau_eff: Optional ``[NZ, NY, NX]`` per-cell effective relaxation
            time (lattice units).  When supplied, BGK skips its
            internal |S|-based Smagorinsky and CM/CUMULANT use
            ``omega_eff = 1/tau_eff`` element-wise.  Pass the
            pre-allocated ``state.tau_eff_buf`` after writing the
            SGS-computed values into it (typically via
            :func:`_compute_sgs_tau_eff(f, config, tau_base, out=buf)`).
        compute_force: When True (default), the kernel computes the Ladd
            wet-node momentum-exchange force and returns ``(fx, fy, fz)``
            in lattice units.  Set False to skip the per-cell force
            accumulation in the kernel — saves ~8% of kernel time at
            n=128 (the per-program wall-cell mask load + per-cell sum +
            atomic-add).  When False, the returned ``(fx, fy, fz)`` are
            zero scalar tensors.

    Returns:
        ``(f_new, fx, fy, fz)`` — post-step distribution (shape matches
        the *input* ``f``) and three scalar force tensors (streamwise,
        lateral, vertical).
    """
    if not f.is_cuda:
        raise RuntimeError(
            "triton_suboff_step requires a CUDA tensor; got "
            f"device={f.device!r}"
        )
    Q_in = f.shape[0]
    if Q_in != _PROD_Q:
        raise ValueError(
            f"Input Q={Q_in} does not match production D3Q19 (Q={_PROD_Q})"
        )
    if obstacle_int8.shape != f.shape[1:]:
        raise ValueError(
            f"obstacle_int8 shape {tuple(obstacle_int8.shape)} does not "
            f"match f's spatial shape {tuple(f.shape[1:])}"
        )
    if tau_eff is not None and tau_eff.shape != f.shape[1:]:
        raise ValueError(
            f"tau_eff shape {tuple(tau_eff.shape)} does not match "
            f"f's spatial shape {tuple(f.shape[1:])}"
        )

    if state is None:
        state = TritonStepState(f, f.device)
        # First-call Q mismatch (e.g. user passed Q=19 after default state
        # was built for a different Q): re-allocate to be safe.
        if state.f_buf.shape[1:] != f.shape[1:]:
            state = TritonStepState(f, f.device)

    # 0. The kernel now accepts Q=19 directly (V2 rewrite) — no
    #    Q=19→Q=32 padding needed.  The wrapper used to require _Q_PAD=32
    #    for the inner tl.arange constraint, but V2 uses 32 internally
    #    with mask=offs_q<19, so the public buffer can be Q=19.
    f_in = f

    # Ping-pong: never let the kernel stream out of the buffer it is
    # writing into.  Callers loop with ``f = triton_suboff_step(f, ...)``,
    # so ``f`` alternates between the two state buffers.
    out = (
        state.f_buf_alt
        if f.data_ptr() == state.f_buf.data_ptr()
        else state.f_buf
    )

    # 1. Fused collide + pull-stream + wet-node bounce-back + LES Smag +
    #    Ladd (1994) wet-node momentum-exchange force reduction.
    #    The kernel is periodic on all three axes; the BC writes below
    #    overwrite the boundary planes to match production's
    #    ``far_field_bc_3d``.
    #
    #    Force reduction is now *fused* into the same launch: each program
    #    accumulates 2 · Σ_q c_q · f_post[q] over its OWN wall cells into a
    #    per-program scalar then ``tl.atomic_add`` into ``state.fx_buf``
    #    etc.  This eliminates the previous ~13 ms separate reduction
    #    kernel call (which scanned the entire grid for ~0.1% wall cells).
    #
    #    The buffers are zeroed here (kernel uses ``atomic_add``, not
    #    assignment, so they MUST be zero before each call).
    if compute_force:
        state.fx_buf.zero_()
        state.fy_buf.zero_()
        state.fz_buf.zero_()
        triton_fused_obstacle_xfar_les(
            f_in, nu_lb, obstacle_int8, Cs, delta,
            collision=collision, out=out, tau_eff=tau_eff,
            fx_buf=state.fx_buf, fy_buf=state.fy_buf, fz_buf=state.fz_buf,
        )
        fx = state.fx_buf
        fy = state.fy_buf
        fz = state.fz_buf
    else:
        # Force-compute skipped: the wrapper falls back to ``out`` as
        # the fp32 placeholder, kernel prunes the force block entirely.
        # Return zero scalars so the call signature is consistent.
        triton_fused_obstacle_xfar_les(
            f_in, nu_lb, obstacle_int8, Cs, delta,
            collision=collision, out=out, tau_eff=tau_eff,
        )
        fx = torch.zeros((), dtype=torch.float32, device=f.device)
        fy = torch.zeros((), dtype=torch.float32, device=f.device)
        fz = torch.zeros((), dtype=torch.float32, device=f.device)

    # 2. 6-face far-field Dirichlet BC writes (matches production).
    #    Inlet at x=0 free-stream eq; outlet at x=nx-1 zero-gradient;
    #    lateral y± and z± Dirichlet.
    #
    #    Equilibrium is constant for the whole runner (u=(u_in, 0, 0)),
    #    so we cache the ``(Q,)`` vector at first call and reuse it on
    #    every subsequent call.  Without the cache, the BC function
    #    allocates 4 full-grid tensors (~200 MB / step at n=128, ~3 ms)
    #    to compute a value that's the same everywhere.
    if state.feq_vec_buf is None or state.feq_vec_u != (u_in, 0.0, 0.0):
        state.feq_vec_buf = _compute_uniform_equilibrium_vec(
            u_in, 0.0, 0.0, _PROD_Q, f.device, f.dtype,
        )
        state.feq_vec_u = (u_in, 0.0, 0.0)
    apply_far_field_bc_6face(
        out, u_in,
        feq_pad_buf=state.feq_pad_buf,
        feq_vec=state.feq_vec_buf,
    )

    # 3. Mass correction every mass_period steps (one GPU↔CPU sync).
    if do_mass_correction and (step % mass_period == 0):
        apply_mass_correction(out, initial_mass)

    # 4. Output buffer shape matches input (Q=19 in V2).
    return out, fx, fy, fz
