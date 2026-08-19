"""z-slab multi-GPU SUBOFF step on the production PyTorch operators.

Distributed companion to the production single-GPU runner
:mod:`tensorlbm.suboff_cmk_kbc_runner` (default PyTorch path).  Every rank
owns a contiguous z-slab of the global ``(19, nz, ny, nx)`` domain and
executes the *same* per-step operator chain as the production loop —
SGS-coupled collision (``_collide_with_sgs`` → ``collide_advanced_3d``),
streaming (``stream3d`` / ``stream3d_roll``), momentum-exchange force
(``compute_obstacle_forces_3d``), far-field BC incl. bounce-back
(``far_field_bc_3d``) and the every-10-step mass correction
(``correct_mass3d`` semantics) — on its slab, with a one-plane NCCL halo
exchange in z and NCCL all-reduces for the obstacle force and the global
mass.

Design (halo protocol follows :mod:`tensorlbm.triton_fused_distributed`;
structural reference: :mod:`tensorlbm.triton_suboff_step_distributed`):

* Buffer per rank: ``(19, nz_local + 2, ny, nx)`` — owned planes at local
  z in ``[1, nz_local + 1)``; ghost plane 0 holds the last owned plane of
  rank ``r-1``, ghost plane ``nz_local + 1`` the first owned plane of
  rank ``r+1`` (neighbours wrap periodically).
* NCCL halo exchange uses *dense staging buffers*: a ghost slice
  ``buf[:, 0:1]`` of a ``(Q, nz, ny, nx)`` tensor is **not** contiguous
  (its Q-dim stride is ``nz*ny*nx``, not ``ny*nx``), so
  ``buf[:, 0:1].contiguous()`` materialises a *copy* — an ``irecv``
  posted on such a copy fills the copy and leaves the ghost plane
  untouched (a silent stale-ghost bug; the reference
  ``triton_fused_distributed`` posts its recvs exactly this way).  Here
  sends/recv run on pre-allocated dense ``(19, 1, ny, nx)`` staging
  tensors and the received planes are copied into the ghost planes after
  the exchange is awaited.
* Collision is pointwise, so it runs on the full padded buffer.  Ghost
  planes collide the neighbour's post-BC values — exactly the value the
  pull-scheme streaming needs when the slab's first/last owned plane
  reads across the slab boundary.
* Streaming runs on the padded buffer: D3Q19 shifts are ±1 in z, so
  every *owned* output plane reads only owned or ghost planes.  The
  periodic wrap inside ``stream3d``/``stream3d_roll`` only ever touches
  the ghost *outputs*, which the next halo exchange overwrites.
  Streaming is a pure permutation, so owned-plane results are bitwise
  identical to the global single-GPU stream regardless of which of the
  two production streaming kernels is used.  ``stream_impl`` selects
  ``"gather"`` (:func:`tensorlbm.solver3d.stream3d`, the runner default)
  or ``"roll"`` (:func:`tensorlbm.solver3d.stream3d_roll`); the gather
  materialises ~8× f bytes of int64 broadcast indices, so ``"auto"``
  falls back to the (bitwise-equal) roll when that footprint would not
  fit.
* Far-field BC: applied to the owned slab with a rank-dependent
  ``bc_config`` — the global z-faces are written only by the ranks that
  own them (rank 0 writes ``z-``, rank ``world_size-1`` writes ``z+``);
  interior ranks leave their slab boundaries periodic.  x inlet/outlet
  and the y-faces are identical on every rank.  All face writes hit the
  same global planes as the single-GPU call.
* Force: the momentum-exchange reduction runs per rank on its slab and
  is summed with a NCCL all-reduce.  The summation order differs from
  the single-GPU reduction, so forces match to ~1e-7 relative, not
  bitwise.
* Mass correction: per-rank sum → NCCL all-reduce → global rescale, i.e.
  ``correct_mass3d`` semantics.  Its ``|mass| < 1e-30`` early-out is not
  replicated (it never triggers for this initialisation and would force
  a host sync).

torch.compile (opt-in via ``compile_mode``): the whole per-step compute
chain — collide+SGS, stream, local force, BC — is wrapped in a single
``torch.compile`` call, following the validated production recipe:
whole-step compilation, the step counter and the every-``mass_every``
mass-correction branch stay outside the compiled code object (no
per-step recompiles), and cudagraph-class modes are rejected outright
because the LBM step feeds its own output back, which cudagraph replay
overwrites.  The NCCL collectives (force/mass all-reduce, halo exchange)
stay eager around the graph.  Only CM and CUMULANT collisions are
supported — KBC's entropy bisection calls ``.item()`` per step.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist

from .advanced_collision_contract import collide_advanced_3d  # noqa: F401  (re-export convenience)
from .boundaries3d import far_field_bc_3d
from .d3q19 import equilibrium3d
from .obstacles import compute_obstacle_forces_3d
from .solver3d import stream3d, stream3d_roll
from .suboff_cmk_kbc_runner import SuboffCmkKbcConfig, _collide_with_sgs

try:  # reuse the slab-local geometry build (bitwise equal to slicing the
    # global build, 5-6x faster than building the full mask per rank)
    from .triton_suboff_step_distributed import build_suboff_solid_slab
except Exception:  # pragma: no cover - triton-free fallback
    build_suboff_solid_slab = None

try:  # reuse the production process-group bootstrap
    from .triton_fused_distributed import init_distributed
except Exception:  # pragma: no cover - triton-free fallback
    init_distributed = None

__all__ = ["SuboffTorchDistributedRunner"]

# torch.compile modes proven on this step chain (PR #174 / the production
# ``compile_mode`` knob).  Anything with cudagraphs is rejected: the step
# consumes its own output buffer.
_ALLOWED_COMPILE_MODES = ("default", "max-autotune-no-cudagraphs")


def _fallback_init_distributed(backend: str | None = None) -> tuple[int, int]:
    """Minimal env-based process-group init (mirrors ``init_distributed``)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
    return rank, world


class SuboffTorchDistributedRunner:
    """Z-slab multi-GPU driver for the production SUBOFF PyTorch step.

    Args:
        config: A :class:`~tensorlbm.suboff_cmk_kbc_runner.SuboffCmkKbcConfig`.
            Its ``device`` field is ignored in favour of ``device``; only
            CM/CUMULANT collisions are accepted (KBC is rejected).
        world_size: Number of ranks (defaults to ``WORLD_SIZE`` env / the
            initialised process group).  ``nz`` must be divisible by it.
        rank: This rank's index (defaults to ``RANK`` env).
        device: This rank's CUDA device (defaults to ``cuda:LOCAL_RANK``).
        compile_mode: ``None`` (eager) or one of ``"default"`` /
            ``"max-autotune-no-cudagraphs"`` — wraps the whole per-step
            compute chain in ``torch.compile``.  Cudagraph-class modes and
            unknown modes raise ``ValueError``.
        mass_every: Mass-correction period in steps (production: 10).
        stream_impl: ``"auto"``, ``"gather"`` (``stream3d``) or ``"roll"``
            (``stream3d_roll``) — bitwise-equal on owned planes; see the
            module docstring for the memory trade-off.
        check_every: isfinite watchdog period for :meth:`run`.
    """

    def __init__(
        self,
        config: SuboffCmkKbcConfig,
        *,
        world_size: int | None = None,
        rank: int | None = None,
        device: str | torch.device | None = None,
        compile_mode: str | None = None,
        mass_every: int = 10,
        stream_impl: str = "auto",
        check_every: int = 10,
    ) -> None:
        if config.collision.upper() not in {"CM", "CUMULANT"}:
            raise ValueError(
                f"suboff_torch_distributed supports CM and CUMULANT only; "
                f"got {config.collision!r} (KBC's entropy bisection calls "
                f".item() per step and cannot run distributed/compiled)"
            )
        if config.use_triton_step:
            raise ValueError(
                "use_triton_step=True is the single-GPU fused Triton path; "
                "pass a plain config to SuboffTorchDistributedRunner"
            )
        if compile_mode is not None:
            if compile_mode not in _ALLOWED_COMPILE_MODES:
                raise ValueError(
                    f"compile_mode must be None or one of "
                    f"{_ALLOWED_COMPILE_MODES}; got {compile_mode!r} "
                    f"(cudagraph-class modes overwrite the step's own "
                    f"input buffer)"
                )
        if mass_every < 1:
            raise ValueError("mass_every must be >= 1")
        if stream_impl not in ("auto", "gather", "roll"):
            raise ValueError(f"stream_impl must be auto/gather/roll, got {stream_impl!r}")

        # --- process group -------------------------------------------------
        init = init_distributed or _fallback_init_distributed
        if not (dist.is_available() and dist.is_initialized()) and world_size is not None and world_size > 1:
            init("nccl")
        env_rank = int(os.environ.get("RANK", "0"))
        env_world = int(os.environ.get("WORLD_SIZE", "1"))
        if dist.is_available() and dist.is_initialized():
            env_rank, env_world = dist.get_rank(), dist.get_world_size()
        self.rank = env_rank if rank is None else int(rank)
        self.world_size = env_world if world_size is None else int(world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} out of range for world_size {self.world_size}")

        if device is None:
            local = int(os.environ.get("LOCAL_RANK", str(self.rank)))
            device = f"cuda:{local}"
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError(f"a CUDA device is required; got {self.device}")
        torch.cuda.set_device(self.device)
        if self.world_size > 1 and not (dist.is_available() and dist.is_initialized()):
            # world_size passed explicitly without torchrun env: caller
            # must have set MASTER_ADDR/MASTER_PORT.
            dist.init_process_group(
                backend="nccl", rank=self.rank, world_size=self.world_size
            )

        self.config = config
        self.compile_mode = compile_mode
        self.mass_every = int(mass_every)
        self.check_every = int(check_every)

        # --- slab layout ----------------------------------------------------
        nz, ny, nx = config.nz, config.ny, config.nx
        if nz % self.world_size != 0:
            raise ValueError(
                f"nz={nz} must be divisible by world_size={self.world_size} "
                f"(z-slab decomposition)"
            )
        self.nz, self.ny, self.nx = nz, ny, nx
        self.nz_local = nz // self.world_size
        if self.nz_local < 2:
            raise ValueError(f"nz_local={self.nz_local} must be >= 2")
        self.z_start = self.rank * self.nz_local
        self.left = (self.rank - 1) % self.world_size
        self.right = (self.rank + 1) % self.world_size

        # --- streaming implementation ----------------------------------------
        slab_elems = 19 * (self.nz_local + 2) * ny * nx
        slab_bytes = slab_elems * 4
        gather_index_bytes = 8 * slab_bytes  # 4 int64 broadcast index tensors
        if stream_impl == "auto":
            total = torch.cuda.get_device_properties(self.device).total_memory
            budget = 0.45 * total
            if compile_mode is not None:
                # Inductor lowers the gather without materialising the
                # broadcast indices, but the roll keeps codegen predictable;
                # prefer the gather only when it would also fit eagerly.
                stream_impl = "gather" if gather_index_bytes <= budget else "roll"
            else:
                stream_impl = "gather" if gather_index_bytes <= budget else "roll"
        self.stream_impl = stream_impl
        self._stream = stream3d if stream_impl == "gather" else stream3d_roll

        # --- geometry (slab-local, bitwise equal to the global slice) --------
        if build_suboff_solid_slab is not None:
            solid, _stats = build_suboff_solid_slab(
                hull_type="bare_hull",
                nx=nx, ny=ny, nz=nz,
                world_size=self.world_size, rank=self.rank,
                cx=nx * 0.35, cy=ny / 2.0, cz=nz / 2.0,
                length=config.hull_length,
                device=str(self.device),
            )
        else:  # pragma: no cover - fallback when triton modules are absent
            from .suboff_cad import SuboffHullType, build_suboff_mask

            solid_full, _ = build_suboff_mask(
                hull_type=SuboffHullType.BARE_HULL,
                nx=nx, ny=ny, nz=nz,
                cx=nx * 0.35, cy=ny / 2.0, cz=nz / 2.0,
                length=config.hull_length,
                device="cpu",
            )
            solid = solid_full[self.z_start : self.z_start + self.nz_local].to(self.device)
        self.solid = solid.to(self.device)

        # --- rank-dependent far-field BC config ------------------------------
        # ``far_field_bc_3d``'s face labels vs the ``(Q, nz, ny, nx)`` axes:
        # its "y-" branch writes ``f[:, 0, :, :]`` — dim 1, the *z*-axis
        # face — and its "z-" branch writes ``f[:, :, 0, :]`` — dim 2, the
        # *y*-axis face (the labels are swapped relative to the axes; the
        # legacy bc_config=None call writes all four lateral faces so the
        # naming never mattered there).  Same global face writes as the
        # single-GPU call therefore means: the dim-2 (y-axis) faces on
        # every rank ("z±" in the function's naming), and the dim-1
        # (z-axis, slab-decomposed) faces only on the ranks that own the
        # global z-boundary planes ("y±" in the function's naming).
        ff = ["z-", "z+"]
        periodic = []
        if self.rank == 0 or self.world_size == 1:
            ff.append("y-")
        else:
            periodic.append("y-")
        if self.rank == self.world_size - 1 or self.world_size == 1:
            ff.append("y+")
        else:
            periodic.append("y+")
        self._bc_config = {"far_field_faces": ff, "periodic_faces": periodic}

        # --- buffers + initial populations ------------------------------------
        self._bufs = [
            torch.empty((19, self.nz_local + 2, ny, nx), dtype=torch.float32, device=self.device)
            for _ in range(2)
        ]
        self._idx = 0
        buf = self._bufs[0]
        rho0 = torch.ones((self.nz_local, ny, nx), device=self.device)
        ux0 = torch.full_like(rho0, config.u_in)
        uy0 = torch.zeros_like(rho0)
        uz0 = torch.zeros_like(rho0)
        ux0[self.solid] = 0.0
        f_owned = equilibrium3d(rho0, ux0, uy0, uz0)
        mass = f_owned.sum()
        if self.world_size > 1:
            dist.all_reduce(mass)
        self.initial_mass = float(mass.item())
        buf[:, 1 : self.nz_local + 1] = f_owned
        self._tau_base = config.tau

        # --- step core (eager or compiled) ------------------------------------
        cfg, solid, tau = self.config, self.solid, self._tau_base
        u_in = self.config.u_in
        bc_config = self._bc_config
        stream = self._stream
        nz_local = self.nz_local

        def _step_core(fbuf: torch.Tensor):
            # 1. collision + SGS on the padded buffer (pointwise; ghost planes
            #    collide the neighbour's post-BC values, which is exactly what
            #    the ghost reads below need).
            fc = _collide_with_sgs(fbuf, cfg, tau)
            # 2. streaming on the padded buffer: owned planes read owned/ghost
            #    planes only; the in-buffer periodic wrap only affects ghost
            #    outputs, overwritten by the next halo exchange.
            fs = stream(fc)
            # 3. momentum-exchange force on owned planes (post-stream,
            #    pre-bounce-back — production order).  Local slab sum; the
            #    caller all-reduces across ranks.
            owned = fs[:, 1 : nz_local + 1]
            fx, fy, fz = compute_obstacle_forces_3d(owned, solid)
            # 4. far-field BC + bounce-back on the owned slab.
            fb = far_field_bc_3d(owned, u_in, obstacle_mask=solid, bc_config=bc_config)
            return fb, fx, fy, fz

        self._step_core = _step_core
        self._core = _step_core if compile_mode is None else torch.compile(_step_core, mode=compile_mode)

        # Dense staging planes for the NCCL halo exchange (see the module
        # docstring: ghost slices of the padded buffer are NOT contiguous,
        # so irecv must not be posted on ``buf[:, 0:1].contiguous()``).
        self._send_l = torch.empty((19, 1, ny, nx), dtype=torch.float32, device=self.device)
        self._send_r = torch.empty((19, 1, ny, nx), dtype=torch.float32, device=self.device)
        self._recv_l = torch.empty((19, 1, ny, nx), dtype=torch.float32, device=self.device)
        self._recv_r = torch.empty((19, 1, ny, nx), dtype=torch.float32, device=self.device)
        # Buffer the last exchange was posted on (for the ghost copy-back).
        self._halo_buf: torch.Tensor | None = None

        self._halo_handles: list | None = None
        self._post_initial_halo()

        self.fx = torch.zeros((), device=self.device)
        self.fy = torch.zeros((), device=self.device)
        self.fz = torch.zeros((), device=self.device)

    # ------------------------------------------------------------------
    # Halo exchange (protocol of tensorlbm.triton_fused_distributed,
    # with dense staging so the recvs actually land in the ghosts)
    # ------------------------------------------------------------------
    def _start_halo(self, buf: torch.Tensor) -> list | None:
        """Post one async NCCL send/recv batch filling ``buf``'s ghost planes.

        Owned plane 1 -> left neighbour's right ghost, owned plane
        ``nz_local`` -> right neighbour's left ghost.  Returns work handles
        to await before the next step reads the ghosts (single rank: plain
        nearest-plane copy, keeping the code path identical).
        """
        if self.world_size == 1:
            buf[:, 0:1].copy_(buf[:, 1:2])
            buf[:, -1:].copy_(buf[:, -2:-1])
            return None
        self._send_l.copy_(buf[:, 1:2])
        self._send_r.copy_(buf[:, self.nz_local : self.nz_local + 1])
        ops = [
            dist.P2POp(dist.isend, self._send_l, self.left),
            dist.P2POp(dist.isend, self._send_r, self.right),
            dist.P2POp(dist.irecv, self._recv_r, self.right),
            dist.P2POp(dist.irecv, self._recv_l, self.left),
        ]
        self._halo_buf = buf
        return dist.batch_isend_irecv(ops)

    def _wait_halo(self) -> None:
        if self._halo_handles:
            for h in self._halo_handles:
                h.wait()
            # Copy the received dense planes into the ghost planes of the
            # buffer the exchange was posted on.
            buf = self._halo_buf
            buf[:, 0:1].copy_(self._recv_l)
            buf[:, -1:].copy_(self._recv_r)
        self._halo_handles = None
        self._halo_buf = None

    def _post_initial_halo(self) -> None:
        # Pre-fill ghosts with the nearest owned plane so a step before the
        # first exchange never reads uninitialised memory, then post the real
        # exchange on the initialised owned planes.
        buf = self._bufs[self._idx]
        buf[:, 0:1].copy_(buf[:, 1:2])
        buf[:, -1:].copy_(buf[:, -2:-1])
        self._halo_handles = self._start_halo(buf)

    def synchronize(self) -> None:
        """Wait for any in-flight halo exchange."""
        self._wait_halo()

    @property
    def buf(self) -> torch.Tensor:
        """Current padded slab buffer; owned planes at ``[:, 1:nz_local+1]``."""
        return self._bufs[self._idx]

    def owned(self) -> torch.Tensor:
        """View of the owned planes of the current buffer."""
        return self.buf[:, 1 : self.nz_local + 1]

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, step_idx: int, compute_force: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the slab by one production step.

        Runs the full production chain (collide+SGS → stream → force →
        far-field BC) on the padded slab — eager or as one compiled graph —
        then all-reduces the force, applies the every-``mass_every`` global
        mass correction, writes the owned planes into the ping-pong buffer
        and posts the halo exchange for the next step.

        Args:
            step_idx: 1-based step counter (mass correction when
                ``step_idx % mass_every == 0``; the counter never enters the
                compiled code object).
            compute_force: Skip the force reduction when False (matching the
                production ``force_every`` thinning).

        Returns:
            ``(fx, fy, fz)`` — globally reduced force scalars (undefined
            values when ``compute_force=False``).
        """
        self._wait_halo()
        fb, fx, fy, fz = self._core(self.buf)

        if self.world_size > 1:
            if compute_force:
                dist.all_reduce(fx)
                dist.all_reduce(fy)
                dist.all_reduce(fz)
            if step_idx % self.mass_every == 0:
                s = fb.sum()
                dist.all_reduce(s)
                # correct_mass3d semantics with a global (all-reduced) mass;
                # its |mass|<1e-30 early-out never triggers here and would
                # force a host sync, so it is not replicated.
                fb = fb * (self.initial_mass / s)
        elif step_idx % self.mass_every == 0:
            fb = fb * (self.initial_mass / fb.sum())

        out = self._bufs[1 - self._idx]
        out[:, 1 : self.nz_local + 1].copy_(fb)
        self._halo_handles = self._start_halo(out)
        self._idx = 1 - self._idx

        self.fx, self.fy, self.fz = fx, fy, fz
        return fx, fy, fz

    # ------------------------------------------------------------------
    # Production-loop driver
    # ------------------------------------------------------------------
    def run(self, n_steps: int, force_every: int = 1) -> dict[str, Any]:
        """Run ``n_steps`` steps with the production loop semantics.

        Records the (all-reduced) force every ``force_every`` steps and runs
        the isfinite watchdog every ``check_every`` steps (plus the final
        step), all-ranked.  Returns a dict with the force time series and
        finite/steps_completed status.
        """
        forces: list[dict[str, float]] = []
        all_finite = True
        completed = 0
        for step in range(1, n_steps + 1):
            fx, fy, fz = self.step(step, compute_force=(step % force_every == 0))
            if step % force_every == 0:
                forces.append(
                    {"step": step, "fx": float(fx.item()), "fy": float(fy.item()), "fz": float(fz.item())}
                )
            completed = step
            if step % self.check_every == 0 or step == n_steps:
                ok = bool(torch.isfinite(self.owned()).all().item())
                if self.world_size > 1:
                    flag = torch.tensor([1.0 if ok else 0.0], device=self.device)
                    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                    ok = bool(flag.item() > 0.0)
                all_finite = all_finite and ok
                if not ok:
                    break
        return {
            "forces": forces,
            "finite": all_finite,
            "steps_completed": completed,
            "world_size": self.world_size,
            "stream_impl": self.stream_impl,
            "compile_mode": self.compile_mode,
        }
