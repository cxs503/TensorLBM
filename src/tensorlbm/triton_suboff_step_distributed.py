"""Multi-GPU slab-decomposed Triton step for the SUBOFF runner.

Wraps :class:`tensorlbm.triton_fused_distributed.DistributedTritonFusedSolver3D`
(periodic slab along z with NCCL halo exchange) and layers on:

*  x-as-streamwise BC writes for the 6 domain faces
*  per-cell force reduction (Ladd momentum-exchange)
*  mass correction every ``mass_period`` steps

The geometry build is slab-local: every rank builds only the z-planes it
owns via :func:`build_suboff_solid_slab` (the full-grid CPU build was
61-82 s per rank at n=1024 and dominated end-to-end setup).  The
constructor still accepts a full-grid ``solid_int8`` and slices it, so
legacy callers that already hold the global mask keep working.

Currently the wrapper is a thin façade over the single-GPU path plus a
slab decomposition in z.  Production slab axis remains z (the existing
distributed solver's design); x is the streamwise axis within each
slab.  When the upstream :mod:`tensorlbm_triton_fused_distributed`
adds a dedicated x-streamwise variant, only the inner kernel call
needs to swap.

Multi-rank correctness (4 fixes, validated bitwise vs the single-GPU
Triton path at 4 ranks — final ``f`` identical bit-for-bit, fx rel at
scale 3.8e-7; see ``triton_bench_20260819/dist_revalidate``):

1. **BC writes only touch physical faces.**  The stock step called
   :func:`apply_far_field_bc_6face` on every rank's owned slab, so
   each *interior* slab interface (z = k*nz_local for k = 1..w-1) was
   reset to free-stream equilibrium once per step.  Now the z faces
   are written only by the rank that owns the *global* z boundary
   (rank 0 writes z-, rank world-1 writes z+); the x inlet/outlet and
   y faces are interior to every slab and stay per-rank.  At
   world_size == 1 this reduces exactly to the stock 6-face BC (the
   single rank owns both global z faces and the ``nz > 4`` guard
   applies to the same tensor).
2. **First-step halo exchange at construction.**  ``from_global``
   pre-fills the ghost planes with the nearest *owned* plane, so the
   first kernel launch read wrong values wherever a slab interface
   crosses a solid/fluid boundary (step-1 error concentrated exactly
   on the interface planes).  ``__init__`` now exchanges + lands the
   halos once before returning.
3. **Halo exchange moved after BC + mass correction.**  The exchange
   used to be posted right after the kernel, so the ghost planes
   carried pre-BC boundary values and the BC/mass fixes could never
   reach the neighbours (error resurfacing at the interfaces from
   step ~4).  The kernel is periodic in z, and the single-GPU
   reference reads the previous step's post-BC planes through that
   wrap; posting the exchange after BC + mass reproduces exactly
   those values in the ghosts.
4. **Global (mass-bitwise) mass correction.**  The per-rank
   ``initial_mass_per_rank`` rescale does not reproduce the
   single-GPU reduction: the scale factor differed per rank and
   leaked an O(1e-9) step at every interface.  Each mass step now
   all-gathers the owned slabs, concatenates them in global z order
   and sums once — bitwise the same reduction order as the
   single-GPU ``f.sum()`` — then rescales every rank by the same
   ``initial_mass_global / current`` factor.  When the gathered
   global tensor cannot fit in GPU memory (production n >= 512
   cubes), the sum degrades to an all_reduce of the per-rank sums,
   which differs only in the last ulp of the scale factor.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

try:
    from tensorlbm_triton_fused_distributed import (
        DistributedTritonFusedSolver3D,
        init_distributed,
    )
    from tensorlbm_triton_fused_obstacle import (
        triton_fused_obstacle_xfar_les,
    )
except ImportError:
    from tensorlbm.triton_fused_distributed import (  # type: ignore
        DistributedTritonFusedSolver3D,
        init_distributed,
    )
    from tensorlbm.triton_fused_obstacle import (  # type: ignore
        triton_fused_obstacle_xfar_les,
    )


__all__ = ["TritonSuboffDistributedRunner", "build_suboff_solid_slab"]

# Mass correction period (steps), matching the single-GPU production
# runner and the revalidation gates.
_MASS_EVERY = 10


def build_suboff_solid_slab(
    hull_type: Any = "bare_hull",
    nx: int = 200,
    ny: int = 80,
    nz: int = 80,
    *,
    world_size: int,
    rank: int,
    cx: float | None = None,
    cy: float | None = None,
    cz: float | None = None,
    length: float | None = None,
    radius: float | None = None,
    config: Any = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """Build only this rank's z-slab of the SUBOFF obstacle mask.

    Returns ``(mask_slab, stats)`` where ``mask_slab`` has shape
    ``(nz_local, ny, nx)`` with ``nz_local = nz // world_size``, bitwise
    identical to slicing the global :func:`build_suboff_mask` result to
    ``[rank * nz_local : (rank + 1) * nz_local]``.

    Every ``suboff_cad`` predicate references the z axis only through
    ``(zz - cz)`` (the hull radius profile depends on x alone), so
    rebuilding with ``nz -> nz_local`` planes and ``cz -> cz - z0``
    evaluates exactly the same arithmetic on the slab's global
    coordinates.  That is ``1 / world_size`` of the full-grid geometry
    work per rank — the full-grid CPU build costs 61-82 s at n=1024
    (vs ~11 s for 300 simulation steps) and dominates end-to-end setup.

    ``stats`` mirrors ``build_suboff_mask``'s but is computed on the
    slab (``nz`` / ``solid_cells`` are slab-local; form coefficients
    are grid-independent).  ``world_size == 1`` reproduces the global
    build exactly.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} out of range for world_size={world_size}")
    if nz % world_size != 0:
        raise ValueError(
            f"nz={nz} must be divisible by world_size={world_size} (slab decomposition along z)"
        )
    try:
        from tensorlbm.suboff_cad import build_suboff_mask
    except ImportError:  # pragma: no cover - top-level module layout
        from suboff_cad import build_suboff_mask  # type: ignore

    nz_local = nz // world_size
    z0 = rank * nz_local
    # Re-base the global default cz = nz / 2 onto the slab BEFORE
    # shrinking nz — otherwise build_suboff_mask would re-derive it as
    # nz_local / 2 and recentre the geometry inside the slab.
    cz_slab = (nz / 2.0 if cz is None else float(cz)) - z0
    return build_suboff_mask(
        hull_type=hull_type,
        nx=nx,
        ny=ny,
        nz=nz_local,
        cx=cx,
        cy=cy,
        cz=cz_slab,
        length=length,
        radius=radius,
        config=config,
        device=device,
    )


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
        initial_mass_per_rank: Legacy per-rank mass target.  Kept for
            signature compatibility; the mass correction now uses the
            global initial mass (:attr:`initial_mass_global`, fix 4).

    The constructor performs the first halo exchange (fix 2) and
    captures the global initial mass (fix 4), so the instance is ready
    to step immediately.
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
                f"nz={nz} must be divisible by world_size={world_size} (slab decomposition along z)"
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
            dtype=solid_int8.dtype,
            device=solid_int8.device,
        )
        self.solid_int8_local[1:-1] = solid_local

        # Per-rank initial mass (legacy attribute; the step-time mass
        # correction uses the global figure below).  Default: the sum of
        # THIS rank's owned slab (the old default sliced global planes
        # 1..nz_local on every rank, which was rank-0's slab — not this
        # rank's).
        if initial_mass_per_rank is None:
            initial_mass_per_rank = float(f[:, z0:z1, :, :].sum().item())
        self.initial_mass_per_rank = initial_mass_per_rank

        # Underlying slab-decomposed solver (periodic in z for the halo
        # exchange — see module docstring).  We use it only as a
        # halo-exchanging scratch buffer; the BC + force + mass are
        # applied here.
        self._dist = DistributedTritonFusedSolver3D(
            nz_global=nz,
            ny=ny,
            nx=nx,
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

        # Fix 2 — first-step halo exchange.  ``from_global`` pre-fills
        # the ghost planes with the nearest OWNED plane, so the first
        # kernel launch would read wrong values wherever a slab
        # interface crosses a solid/fluid boundary (measured: step-1
        # error concentrated exactly on the interface planes at
        # n=(128,64,64)/w=4).  Exchange + land once now.  At
        # world_size == 1 this repeats the nearest-owned-plane copy
        # ``from_global`` already did — a no-op.
        self._dist._halo_handles = self._dist._start_halo_exchange(self._buf)
        self._dist.synchronize()

        # Fix 4 — global initial mass, reduced in the single-GPU order
        # (gather slabs -> concatenate along z -> one sum).  This is
        # bitwise equal to ``f_global.sum()`` on one GPU, i.e. the mass
        # target the single-GPU path rescales back to.  When the
        # gathered global tensor does not fit comfortably in free GPU
        # memory (production n >= 512 cubes), fall back to an all_reduce
        # of the per-rank sums — the scale factor then differs by at
        # most ~1 ulp, which the A/B revalidation measured as
        # numerically indistinguishable (fx rel 3.390e-5 vs 3.391e-5).
        self._mass_reduce_gather = self._gather_fits_memory(f.shape[0], nz, ny, nx, f.device)
        self.initial_mass_global = float(
            self._current_global_mass(self._buf[:, 1 : nz_local + 1, :, :]).item()
        )

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
            dtype=torch.float32,
            device=f.device,
        )

        self.nz_local = nz_local
        self.nz = nz
        self.ny = ny
        self.nx = nx

    # ------------------------------------------------------------------
    def _gather_fits_memory(
        self,
        q: int,
        nz: int,
        ny: int,
        nx: int,
        device: torch.device,
    ) -> bool:
        """Whether the gather-based (bitwise) global mass fits in memory.

        The gather materializes the full global ``(Q, nz, ny, nx)``
        tensor twice transiently (the all_gather output list plus the
        concatenated copy), which is only affordable on validation-size
        grids — at n=512/w=4 that is 2 x 20 GiB against a 31 GiB card.

        The decision must be identical on every rank (a rank-local
        mismatch would desynchronise the collective sequence and
        deadlock), so rank 0 decides and the verdict is broadcast.
        """
        if self.world_size == 1:
            return True  # no gather involved; direct owned sum
        if self.rank == 0:
            if device.type != "cuda":
                fits = True  # CPU tensors: host RAM is not the constraint
            else:
                global_bytes = q * nz * ny * nx * 4
                try:
                    free_bytes, _total = torch.cuda.mem_get_info(device)
                except (RuntimeError, ValueError):  # pragma: no cover
                    free_bytes = 0
                # 2x for the gather list + concatenated copy, 2 GiB headroom.
                fits = 2 * global_bytes <= max(free_bytes - (2 << 30), 0)
        else:
            fits = True  # placeholder; overwritten by the broadcast
        if not dist.is_available() or not dist.is_initialized():
            return fits
        flag = torch.tensor(
            [1 if fits else 0], dtype=torch.int64, device=device if device.type == "cuda" else "cpu"
        )
        dist.broadcast(flag, src=0)
        return bool(flag.item())

    def _current_global_mass(self, owned_full: torch.Tensor) -> torch.Tensor:
        """Current total mass over all ranks' owned planes.

        Gather mode (validation-size grids, ``_mass_reduce_gather``):
        all_gather the owned slabs and sum the globally-ordered
        concatenation in one reduction — bitwise the same order as the
        single-GPU ``f.sum()`` the mass correction is calibrated
        against.  All-reduce mode (large grids): sum per rank then
        ``all_reduce``; differs from the gather tree only in the last
        ulp of the resulting scale factor.

        Args:
            owned_full: the caller's owned-plane view of the CURRENT
                step's output buffer (``out_local[:, 1:-1]``), not the
                ping-pong input — the mass correction must see the
                post-kernel, post-BC state.

        Returns a 0-d fp32 tensor on this rank's device (identical on
        every rank).  Collective when ``world_size > 1``.
        """
        # The owned-plane view of the halo-padded buffer is strided.  The
        # contiguous copy is needed ONLY in the gather branch: NCCL's
        # all_gather requires a dense operand, and the globally ordered
        # concatenation must reproduce the single-GPU tensor layout for
        # the bitwise sum.  The world==1 branch keeps the stock
        # contiguous+sum order bit-for-bit.  In the all-reduce fallback
        # branch the strided view is summed directly — torch.sum handles
        # non-contiguous inputs and all_reduce only ever sees the 0-d
        # scalar result, so no NCCL layout requirement applies.
        # Materialising the copy there allocated a full owned-slab
        # transient on EVERY mass step (9.5 GiB at n=1024/w=8), which
        # OOMed the stock wrapper at the first mass-correction step on
        # 31.4 GiB cards (triton_bench_20260819/ac_n1024_fullstack/
        # perf_oom_repro.json: 20.23 GiB allocated + 9.5 GiB transient).
        if self.world_size == 1:
            owned = owned_full.contiguous()
            return owned.sum()
        if self._mass_reduce_gather:
            owned = owned_full.contiguous()
            parts = [torch.empty_like(owned) for _ in range(self.world_size)]
            dist.all_gather(parts, owned)
            cur = torch.cat(parts, dim=1).sum()
            del parts
            return cur
        cur = owned_full.sum()
        dist.all_reduce(cur, op=dist.ReduceOp.SUM)
        return cur

    def _apply_far_field_bc_owned(self, owned_full: torch.Tensor, u_in: float) -> None:
        """Far-field BC on this rank's owned planes — physical faces only.

        Fix 1: the x inlet/outlet and the y faces are interior to every
        slab and are written on every rank exactly as
        :func:`apply_far_field_bc_6face` does, but the z faces are
        written only by the rank that owns the *global* z boundary
        (rank 0 writes z-, rank world-1 writes z+).  Writing both z
        faces on every rank would overwrite every interior slab
        interface with free-stream values once per step.

        The ``self.nz > 4`` guard mirrors production's
        ``boundaries3d.far_field_bc_3d`` / ``apply_far_field_bc_6face``
        (2D-extruded mode keeps the z axis periodic).  At world 1 the
        single rank owns both global z faces, so this reduces exactly
        to the stock 6-face BC.
        """
        try:
            from tensorlbm.d3q19 import equilibrium3d
        except ImportError:  # pragma: no cover - flat module layout
            from d3q19 import equilibrium3d  # type: ignore

        rho1 = torch.ones((1, 1, 1), dtype=owned_full.dtype, device=owned_full.device)
        feq_vec = equilibrium3d(
            rho1, torch.full_like(rho1, u_in), torch.zeros_like(rho1), torch.zeros_like(rho1)
        )[:, 0, 0, 0]
        _q, _nzl, ny, nx = owned_full.shape
        # x inlet (x=0): free-stream equilibrium.
        owned_full[:, :, :, 0] = feq_vec[:, None, None]
        # x outlet (x=nx-1): zero-gradient copy from x=nx-2.
        if nx >= 2:
            owned_full[:, :, :, -1] = owned_full[:, :, :, -2]
        # Lateral y± Dirichlet.
        owned_full[:, :, 0, :] = feq_vec[:, None, None]
        if ny >= 2:
            owned_full[:, :, -1, :] = feq_vec[:, None, None]
        # Global z faces only (see docstring).  Guard on the GLOBAL nz
        # — the stock helper guarded on the tensor it was handed, which
        # at world > 1 was the local slab height.
        if self.nz > 4:
            if self.rank == 0:
                owned_full[:, 0, :, :] = feq_vec[:, None, None]
            if self.rank == self.world_size - 1:
                owned_full[:, -1, :, :] = feq_vec[:, None, None]

    # ------------------------------------------------------------------
    def synchronize(self) -> None:
        """Land any in-flight halo exchange into ``self._buf``'s ghosts.

        The exchange posted at the end of :meth:`step` is asynchronous;
        until it lands, ``self._buf``'s ghost planes still hold the
        kernel's (unused, possibly non-finite) output for those rows.
        :meth:`step` itself lands it before the next kernel launch, so
        plain step loops need no extra call — but callers that READ
        ``self._buf`` between steps (e.g. the faithful-tau SGS coupling,
        which streams the current state to recompute ``tau_eff``) must
        call this first.
        """
        self._dist.synchronize()

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

        Step ordering (fixes 1/3/4, see module docstring): kernel ->
        BC on physical faces -> global mass correction -> halo
        exchange.  The opening ``synchronize`` lands the exchange that
        the PREVIOUS step posted after its BC + mass writes, so this
        step's kernel reads the neighbours' post-BC boundary planes —
        the same values the single-GPU kernel reads through its
        periodic wrap.

        The exchange posted at the end of this step is asynchronous:
        the ghost planes of the returned ``f_local`` hold stale (kernel
        garbage) values until the next call to :meth:`step` or
        :meth:`synchronize`.
        """
        cfg = self.config
        u_in = cfg.u_in
        nu_lb = cfg.nu
        Cs = cfg.C_s
        delta = 1.0

        # 1. Land the halo exchange posted at the END of the previous
        #    step (fix 3) so the kernel below reads fresh, post-BC
        #    ghost planes.  No-op on the very first step — __init__
        #    already exchanged and landed (fix 2).
        self._dist.synchronize()

        # 2. V2 kernel accepts Q=19 directly — no padding needed.
        f_in = self._buf

        # 3. Fused collide + stream + bounce-back + LES Smagorinsky.
        #    V2 kernel writes Q=19 output directly; we ping-pong with
        #    self._buf (the Q=19 halo-exchange buffer).
        if self._dist._buf is None:
            self._dist._buf = torch.empty(
                (self._q_in, *self._buf.shape[1:]),
                dtype=f_in.dtype,
                device=f_in.device,
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
                    f_in,
                    nu_lb,
                    self.solid_int8_local,
                    Cs,
                    delta,
                    out=out_local,
                    collision=collision,
                    tau_eff=self.tau_eff_buf,
                    fx_buf=self.fx_buf,
                    fy_buf=self.fy_buf,
                    fz_buf=self.fz_buf,
                )
            else:
                triton_fused_obstacle_xfar_les(
                    f_in,
                    nu_lb,
                    self.solid_int8_local,
                    Cs,
                    delta,
                    out=out_local,
                    collision=collision,
                    fx_buf=self.fx_buf,
                    fy_buf=self.fy_buf,
                    fz_buf=self.fz_buf,
                )
        else:
            if use_external_tau:
                triton_fused_obstacle_xfar_les(
                    f_in,
                    nu_lb,
                    self.solid_int8_local,
                    Cs,
                    delta,
                    out=out_local,
                    collision=collision,
                    tau_eff=self.tau_eff_buf,
                )
            else:
                triton_fused_obstacle_xfar_les(
                    f_in,
                    nu_lb,
                    self.solid_int8_local,
                    Cs,
                    delta,
                    out=out_local,
                    collision=collision,
                )

        # Roll halos — FIX 3: the exchange is now posted AFTER the BC and
        # mass writes (see step 5/6 below), so the ghost planes carry the
        # post-BC values of the neighbours' boundary planes.  The kernel is
        # periodic in z; the single-GPU reference reads the previous step's
        # post-BC planes through that periodic wrap, which is exactly what
        # the delayed exchange reproduces here.  (The stock order posted
        # the exchange right after the kernel, so the BC/mass fixes could
        # never reach the neighbours — measured as fresh 3.7e-9 error
        # appearing on the z interfaces from step ~4.)

        # 4. Owned planes (drop the 2 ghost planes) for BC + mass correction.
        #    V2: out_local is Q=19 throughout.
        owned_full = out_local[:, 1 : self.nz_local + 1, :, :]
        if compute_force:
            fx = self.fx_buf
            fy = self.fy_buf
            fz = self.fz_buf
        else:
            fx = torch.zeros((), dtype=torch.float32, device=f_in.device)
            fy = torch.zeros((), dtype=torch.float32, device=f_in.device)
            fz = torch.zeros((), dtype=torch.float32, device=f_in.device)

        # 5. BC writes on owned planes — physical faces only (fix 1).
        self._apply_far_field_bc_owned(owned_full, u_in)

        # 6. Mass correction (fix 4): rescale every rank by the same
        #    global factor so the current GLOBAL mass returns to
        #    ``initial_mass_global``.  The stock per-rank rescale let
        #    each rank's scale factor differ in the last ulp and leaked
        #    a persistent interface error.
        if step_idx % _MASS_EVERY == 0:
            cur = self._current_global_mass(owned_full)
            if abs(float(cur.item())) >= 1e-30:
                owned_full.mul_(self.initial_mass_global / cur)

        # 5b/6b. Halo exchange AFTER BC + mass (fix 3).
        self._dist.synchronize()
        self._dist._halo_handles = self._dist._start_halo_exchange(out_local)

        # 7. Output buffer shape matches input (V2: Q=19 throughout).
        self._buf = out_local
        # Repoint the scratch to the buffer this step just consumed so
        # the next step writes there.  Without this, f_in and out_local
        # are the same tensor from the second call onward and the kernel
        # runs in place — V2 is not in-place safe (n=256, 50 steps:
        # force trajectory silently off by 29% relative vs a true
        # double-buffered run; values stay finite so nothing flags it).
        self._dist._buf = f_in
        return self._buf, fx, fy, fz
