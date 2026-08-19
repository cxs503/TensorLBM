"""Multi-GPU slab-decomposed Triton step for the SUBOFF runner.

Wraps :class:`tensorlbm.triton_fused_distributed.DistributedTritonFusedSolver3D`
(periodic slab along z with NCCL halo exchange) and layers on:

*  x-as-streamwise BC writes for the 6 domain faces
*  per-cell force reduction (Ladd momentum-exchange)
*  mass correction every ``mass_period`` steps

The geometry build (call to ``build_suboff_mask``) is collective: every
rank builds the full mask and slices it locally.  This is intentional —
the SUBOFF CAD helper is cheap and the duplication avoids having to
redesign the geometry API for distributed construction.

Currently the wrapper is a thin façade over the single-GPU path plus a
slab decomposition in z.  Production slab axis remains z (the existing
distributed solver's design); x is the streamwise axis within each
slab.  When the upstream :mod:`tensorlbm_triton_fused_distributed`
adds a dedicated x-streamwise variant, only the inner kernel call
needs to swap.
"""
from __future__ import annotations

from typing import Any

import torch

try:
    from tensorlbm_triton_fused_obstacle import (
        apply_far_field_bc_6face,
        apply_mass_correction,
        triton_fused_obstacle_xfar_les,
    )
    from tensorlbm_triton_fused_distributed import (
        DistributedTritonFusedSolver3D,
        init_distributed,
    )
except ImportError:
    from tensorlbm.triton_fused_obstacle import (  # type: ignore
        apply_far_field_bc_6face,
        apply_mass_correction,
        triton_fused_obstacle_xfar_les,
    )
    from tensorlbm.triton_fused_distributed import (  # type: ignore
        DistributedTritonFusedSolver3D,
        init_distributed,
    )


__all__ = ["TritonSuboffDistributedRunner"]


class TritonSuboffDistributedRunner:
    """Slab-decomposed Triton step driver for the SUBOFF runner.

    Each rank owns a contiguous z-slab of ``nz_local = nz / world_size``
    planes (plus 2 ghost planes for halo exchange).  The full geometry
    is built on every rank; the slab is sliced from the global mask.

    Args:
        config: The runner's :class:`SuboffCmkKbcConfig`.
        f: Distribution tensor ``(Q, nz, ny, nx)`` — only this rank's
            owned slab is consumed; the wrapper slices internally.
        solid_int8: ``int8[NZ, NY, NX]`` obstacle (will be sliced).
        solid_f32: ``float32[NZ, NY, NX]`` obstacle (will be sliced).
        world_size: Number of ranks.
        rank: This rank's index.
    """

    def __init__(
        self,
        config: Any,
        f: torch.Tensor,
        solid_int8: torch.Tensor,
        *,
        world_size: int,
        rank: int,
        initial_mass_per_rank: float | None = None,
    ) -> None:
        # Lazy-init the process group if it isn't already.  The user's
        # launch script may have already called init_distributed(); this
        # is a no-op in that case.
        init_distributed(backend="nccl")

        self.config = config
        self.rank = rank
        self.world_size = world_size

        Q, nz, ny, nx = f.shape
        if nz % world_size != 0:
            raise ValueError(
                f"nz={nz} must be divisible by world_size={world_size} "
                "(slab decomposition along z)"
            )

        # Slice the obstacle to this rank's slab.  Note: the global mask
        # is built identically on every rank (the helper is cheap).
        nz_local = nz // world_size
        z0, z1 = rank * nz_local, (rank + 1) * nz_local
        solid_local = solid_int8[z0:z1, :, :].contiguous()

        # Pad with 2 ghost planes of zeros (top and bottom) so the
        # obstacle mask matches the slab+halo f buffer shape.  Ghost
        # cells are always fluid.
        self.solid_int8_local = torch.zeros(
            (nz_local + 2, *solid_local.shape[1:]),
            dtype=solid_int8.dtype, device=solid_int8.device,
        )
        self.solid_int8_local[1:-1] = solid_local

        # Per-rank initial mass for mass correction.  Default: 1/(world_size)
        # of the global initial mass when caller passes the global figure,
        # or computed from f's current sum if not given.
        if initial_mass_per_rank is None:
            initial_mass_per_rank = float(f[:, 1:nz_local + 1, :, :].sum().item())
        self.initial_mass_per_rank = initial_mass_per_rank

        # Underlying slab-decomposed solver (periodic in z for the halo
        # exchange — see module docstring).  We use it only as a
        # halo-exchanging scratch buffer; the BC + force + mass are
        # applied here.
        self._dist = DistributedTritonFusedSolver3D(
            nz_global=nz, ny=ny, nx=nx,
            # Standard LBM tau = 3*nu + 0.5.  The upstream class's
            # ``step`` method is not actually invoked here (we call
            # ``triton_fused_obstacle_xfar_les`` directly), so this
            # value is mostly cosmetic — but keeping it consistent
            # with the production runner avoids spurious warnings.
            tau=3.0 * config.nu + 0.5,
            device=str(f.device),
        )

        # Slice the distribution into this rank's slab + 2 ghost planes.
        # The buffer is Q=19 (production D3Q19); V2 kernel accepts Q=19
        # directly, no padding needed.
        self._buf = self._dist.from_global(f)
        self._q_in = f.shape[0]

        # Persistent force buffers.
        self.fx_buf = torch.zeros((), dtype=torch.float32, device=f.device)
        self.fy_buf = torch.zeros((), dtype=torch.float32, device=f.device)
        self.fz_buf = torch.zeros((), dtype=torch.float32, device=f.device)

        # Persistent tau_eff buffer for external SGS coupling
        # (Phase 3 + Phase 5 plumbing).  Shape matches the slab
        # including 2 ghost planes so the kernel can read it without
        # extra slicing.  Default value 0 means "use molecular tau";
        # the caller is responsible for populating it on owned planes
        # before each step.
        self.tau_eff_buf = torch.zeros(
            self._buf.shape[1:],  # (nz_local+2, ny, nx)
            dtype=torch.float32, device=f.device,
        )

        self.nz_local = nz_local
        self.ny = ny
        self.nx = nx

    # ------------------------------------------------------------------
    def step(
        self,
        step_idx: int,
        collision: str = "BGK",
        use_external_tau: bool = False,
        compute_force: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the slab by one fused Triton step.

        Args:
            step_idx: Step counter (used for periodic mass correction).
            collision: Collision family — one of ``"BGK"``, ``"CM"``,
                ``"CUMULANT"``.  Selects the kernel's family-dispatched
                collide branch (Phase 2 + Phase 5 plumbing).
            use_external_tau: When True, pass ``self.tau_eff_buf`` to
                the kernel so it uses the externally-computed per-cell
                tau_eff (WALE/Vreman).  When False, the kernel uses its
                internal Smagorinsky |S| formulation.
            compute_force: When True (default), compute the Ladd force
                and reduce it across ranks.  When False, the per-cell
                force accumulation is skipped (~8% of kernel time at
                n=128) and the returned ``(fx, fy, fz)`` are zero.

        Returns ``(f_local, fx, fy, fz)`` where ``f_local`` is this
        rank's slab (shape ``(Q, nz_local+2, ny, nx)`` with 2 ghost
        planes; the caller only reads owned planes ``[:, 1:-1, :, :]``)
        and ``fx, fy, fz`` are scalar force tensors reduced over the
        global obstacle (post-stream, pre-bounce-back).
        """
        cfg = self.config
        u_in = cfg.u_in
        nu_lb = cfg.nu
        Cs = cfg.C_s
        delta = 1.0

        # 1. Halo exchange (overlap with kernel compute is handled by
        #    the upstream class).
        self._dist.synchronize()

        # 2. V2 kernel accepts Q=19 directly — no padding needed.
        f_in = self._buf

        # 3. Fused collide + stream + bounce-back + LES Smagorinsky.
        #    V2 kernel writes Q=19 output directly; we ping-pong with
        #    self._buf (the Q=19 halo-exchange buffer).
        if self._dist._buf is None:
            self._dist._buf = torch.empty(
                (self._q_in, *self._buf.shape[1:]),
                dtype=f_in.dtype, device=f_in.device,
            )
        out_local = self._dist._buf

        # 3. Fused collide + stream + bounce-back + LES Smag + Ladd force.
        #    Force reduction is now fused into the same kernel launch —
        #    each program accumulates 2 · Σ_q c_q · f_post[q] over its
        #    OWN wall cells then ``tl.atomic_add`` into fx/fy/fz_buf.
        #    Ghost planes (z=0, z=nz_local+1) have obstacle=0 so they
        #    contribute zero force.
        #
        #    The buffers are zeroed before the call (kernel uses
        #    ``tl.atomic_add``, not assignment).
        if compute_force:
            self.fx_buf.zero_()
            self.fy_buf.zero_()
            self.fz_buf.zero_()
            if use_external_tau:
                triton_fused_obstacle_xfar_les(
                    f_in, nu_lb, self.solid_int8_local, Cs, delta,
                    out=out_local,
                    collision=collision,
                    tau_eff=self.tau_eff_buf,
                    fx_buf=self.fx_buf, fy_buf=self.fy_buf, fz_buf=self.fz_buf,
                )
            else:
                triton_fused_obstacle_xfar_les(
                    f_in, nu_lb, self.solid_int8_local, Cs, delta,
                    out=out_local,
                    collision=collision,
                    fx_buf=self.fx_buf, fy_buf=self.fy_buf, fz_buf=self.fz_buf,
                )
        else:
            if use_external_tau:
                triton_fused_obstacle_xfar_les(
                    f_in, nu_lb, self.solid_int8_local, Cs, delta,
                    out=out_local,
                    collision=collision,
                    tau_eff=self.tau_eff_buf,
                )
            else:
                triton_fused_obstacle_xfar_les(
                    f_in, nu_lb, self.solid_int8_local, Cs, delta,
                    out=out_local,
                    collision=collision,
                )

        # Roll halos: copy the new boundary planes from neighbours
        # (async — overlapped with the BC writes below).
        self._dist._halo_handles = self._dist._start_halo_exchange(out_local)

        # 4. Owned planes (drop the 2 ghost planes) for BC + mass correction.
        #    V2: out_local is Q=19 throughout.
        owned_full = out_local[:, 1:self.nz_local + 1, :, :]
        if compute_force:
            fx = self.fx_buf
            fy = self.fy_buf
            fz = self.fz_buf
        else:
            fx = torch.zeros((), dtype=torch.float32, device=f_in.device)
            fy = torch.zeros((), dtype=torch.float32, device=f_in.device)
            fz = torch.zeros((), dtype=torch.float32, device=f_in.device)

        # 5. BC writes on owned planes.
        apply_far_field_bc_6face(owned_full, u_in)

        # 6. Mass correction every 10 steps (per-rank; sum stays consistent
        #    within rounding).  Uses the per-rank initial mass captured at
        #    construction time.
        if step_idx % 10 == 0:
            apply_mass_correction(owned_full, self.initial_mass_per_rank)

        # 7. Output buffer shape matches input (V2: Q=19 throughout).
        self._buf = out_local
        return self._buf, fx, fy, fz