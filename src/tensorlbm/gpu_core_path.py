"""T2 GPU core path — one stepper that unifies the GPU optimisation stack.

The module provides a *single* collision→force→stream→wall step (``core_step``)
that the T2 optimisation dimensions plug into:

1. **Collision + streaming fusion** — BGK collision, the Guo body-force term
   and the (periodic / wall-reflected) streaming run inside one step function
   over one pair of ping-pong buffers, so no intermediate distribution
   round-trips through DRAM.
2. **Memory management** — :class:`GPUCorePathSolver` allocates two
   ``(19, nz, ny, nx)`` distribution buffers plus one
   :class:`~tensorlbm.perf_buffers.LBMStepBuffer` at construction time and
   performs no allocation during time stepping.
3. **Multi-backend** — ``backend`` selects ``triton`` (CUDA kernels of
   :mod:`tensorlbm.triton_fused`), ``torch_compile`` (Inductor) or ``eager``;
   a backend that cannot run raises instead of silently changing the answer.
4. **Multi-GPU communication** — :class:`DistributedGPUCorePathSolver` adds
   slab halo exchange of the five crossing populations per face (selective
   population transfer: 5/19 of the state instead of 19/19).
5. **Precision / performance trade-off** — ``precision`` splits compute dtype
   from store dtype (``FP64FP64`` / ``FP64FP32`` / ``FP32FP32`` /
   ``FP32FP16``); the store dtype is what persists between steps.

Validation targets are T3-verified physics with closed-form references, never
peak numbers in isolation:

* body-force driven Poiseuille channel (analytic parabola), and
* the Taylor–Green vortex (exact Navier–Stokes decay).

Wall treatment is half-way bounce-back on the two ``y`` planes (``j = 0.5``
and ``j = ny - 1.5``, i.e. a channel of width ``H = ny - 2``), second order
accurate (Ziegler 1993; Ladd 1994).  Body forcing uses the Guo scheme (Guo,
Zheng & Shi, PRE 65, 046308, 2002), whose discrete steady state reproduces the
parabolic profile to O(Ma²).

Entry point
-----------
``python -m tensorlbm.gpu_core_path --ny 18 34`` runs the built-in validation
suite (also driven by ``benchmarks/t2_gpu_core_path/bench.py``).
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

from .d3q19 import C, OPPOSITE, W
from .perf_buffers import LBMStepBuffer
from .precision import PrecisionPolicy

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

__all__ = [
    "BackendChoice",
    "GPUCorePathConfig",
    "GPUCorePathSolver",
    "DistributedGPUCorePathSolver",
    "channel_analytic_velocity",
    "collide_guo_into",
    "core_step",
    "get_best_available_backend",
    "run_channel_case",
    "run_taylor_green_case",
    "stream_into",
    "taylor_green_analytic_energy",
]

BackendChoice = Literal["auto", "triton", "torch_compile", "eager"]

#: Wall-normal velocity component of every D3Q19 direction.
_CY = C[:, 1].tolist()
_UP = [i for i, cy in enumerate(_CY) if cy > 0]      # move to +y: bounce at ceiling
_DOWN = [i for i, cy in enumerate(_CY) if cy < 0]    # move to -y: bounce at floor
_OPP = OPPOSITE.tolist()

#: In-place streaming shifts: ``out[q] = roll(f[q], shift)`` == ``f[q](x - c_q)``.
_SHIFTS: list[tuple[int, int, int]] = [
    (int(C[q, 2]), int(C[q, 1]), int(C[q, 0])) for q in range(C.shape[0])
]


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _triton_available() -> bool:
    """True when the Triton fused kernels can actually run (CUDA + triton)."""
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def get_best_available_backend(requested: BackendChoice = "auto") -> str:
    """Resolve a backend request to one that can run on this host.

    ``"auto"`` prefers ``triton`` then ``torch_compile``.  A *forced* backend
    that is unavailable raises :class:`RuntimeError` instead of silently
    returning a different code path — silent degradation is how results stop
    being attributable to a kernel.
    """
    if requested == "eager":
        return "eager"
    if requested == "triton":
        if not _triton_available():
            raise RuntimeError(
                "backend='triton' requested but CUDA/triton is unavailable on this host"
            )
        return "triton"
    if requested == "torch_compile":
        return "torch_compile"
    if requested != "auto":
        raise ValueError(f"unknown backend {requested!r}")
    return "triton" if _triton_available() else "torch_compile"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPUCorePathConfig:
    """Configuration of the unified core-path solver.

    Parameters
    ----------
    nx, ny, nz:
        Grid size.  ``ny`` includes the two solid wall rows, so the channel
        width in lattice units is ``H = ny - 2``.
    tau:
        BGK relaxation time; ``nu = (tau - 0.5) / 3``.
    u_max_target:
        Centre-line velocity of the analytic Poiseuille profile; it fixes the
        body force ``a = 8 nu u_max / H**2`` (and therefore ``u_tau``,
        ``Re_tau``).  Parameters are stated as targets and *derived* here, so
        the analytic reference and the simulation can never disagree about the
        driving.
    init:
        ``"rest"`` (uniform at rest: the honest laminar spin-up),
        ``"parabolic"`` (analytic restart) or ``"taylor_green"``.
    tg_u0:
        Initial amplitude of the Taylor–Green vortex (``init="taylor_green"``).
    precision:
        One of ``FP64FP64``, ``FP64FP32``, ``FP32FP32``, ``FP32FP16``.
    backend:
        ``"auto"``, ``"triton"``, ``"torch_compile"`` or ``"eager"``.
    seed, perturbation_amplitude:
        Optional random perturbation of the initial field, relative to
        ``u_max_target`` (0.0 keeps the case strictly deterministic).
    wall_bc:
        Wall closure on the two ``y`` planes:
        ``"halfway"`` (default, classic half-way bounce-back, unchanged
        behaviour) or ``"nebb"`` (force-corrected non-equilibrium bounce-back
        from :mod:`tensorlbm.boundary.nebb`: wall density recovered from the
        field, wall velocity shifted by ``-a/(2 rho_w)`` so that it is
        consistent with the Guo body force).  ``"nebb"`` runs on the eager
        backend (the compiled kernel only implements ``"halfway"``).
    """

    nx: int = 32
    ny: int = 18
    nz: int = 8
    tau: float = 0.8
    u_max_target: float = 0.05
    init: str = "rest"
    tg_u0: float = 0.02
    precision: str = "FP64FP64"
    backend: BackendChoice = "auto"
    seed: int = 42
    perturbation_amplitude: float = 0.0
    steps: int = 3000
    warmup_steps: int = 0
    label: str = "gpu_core_path"
    wall_bc: str = "halfway"

    # -- derived physics ------------------------------------------------
    @property
    def height(self) -> float:
        """Channel width in lattice units (wall planes at 0.5 and ny-1.5)."""
        return float(self.ny - 2)

    @property
    def half_height(self) -> float:
        return 0.5 * self.height

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def body_force(self) -> float:
        """Body-force acceleration producing ``u_max_target`` on a parabola."""
        return 8.0 * self.nu * self.u_max_target / self.height**2

    @property
    def u_tau(self) -> float:
        """Friction velocity implied by the body force: ``sqrt(a H / 2)``."""
        return math.sqrt(self.body_force * self.height / 2.0)

    @property
    def effective_body_force(self) -> float:
        """Driving actually applied: zero for the periodic Taylor–Green case."""
        return 0.0 if self.init == "taylor_green" else self.body_force

    @property
    def re_tau(self) -> float:
        """Friction Reynolds number ``u_tau H / (2 nu)``."""
        return self.u_tau * self.height / (2.0 * self.nu)

    @property
    def ma(self) -> float:
        """Lattice Mach number of the centre-line velocity (compressibility gauge)."""
        return self.u_max_target * math.sqrt(3.0)

    @property
    def precision_policy(self) -> PrecisionPolicy:
        return PrecisionPolicy[self.precision]

    def validate(self) -> None:
        if min(self.nx, self.nz) < 2:
            raise ValueError("nx and nz must be >= 2")
        if self.ny < 6:
            raise ValueError("ny must be >= 6 (two wall rows + 4 fluid rows)")
        if not 0.5 < self.tau < 3.0:
            raise ValueError(f"tau must be in (0.5, 3.0), got {self.tau}")
        if self.u_max_target <= 0.0:
            raise ValueError("u_max_target must be positive")
        if self.init not in ("rest", "parabolic", "taylor_green"):
            raise ValueError(f"unknown init {self.init!r}")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0, steps)")
        if self.perturbation_amplitude < 0.0:
            raise ValueError("perturbation_amplitude must be >= 0")
        if self.wall_bc not in ("halfway", "nebb"):
            raise ValueError(f"unknown wall_bc {self.wall_bc!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "tau": self.tau,
            "nu": self.nu,
            "u_max_target": self.u_max_target,
            "body_force": self.body_force,
            "u_tau": self.u_tau,
            "re_tau": self.re_tau,
            "ma": self.ma,
            "height": self.height,
            "precision": self.precision,
            "backend": self.backend,
            "init": self.init,
            "steps": self.steps,
            "label": self.label,
            "wall_bc": self.wall_bc,
        }


# ---------------------------------------------------------------------------
# Kernels — all operate on (19, nz, ny, nx) distributions
# ---------------------------------------------------------------------------


def _macroscopic(f: torch.Tensor, C_: torch.Tensor) -> tuple[torch.Tensor, ...]:
    rho = f.sum(dim=0)
    inv = 1.0 / torch.clamp(rho, min=1e-12)
    ux = (f * C_[:, 0].view(-1, 1, 1, 1)).sum(dim=0) * inv
    uy = (f * C_[:, 1].view(-1, 1, 1, 1)).sum(dim=0) * inv
    uz = (f * C_[:, 2].view(-1, 1, 1, 1)).sum(dim=0) * inv
    return rho, ux, uy, uz


def collide_guo_into(
    f: torch.Tensor,
    tau: float,
    out: torch.Tensor,
    C_: torch.Tensor,
    W_: torch.Tensor,
    body_force: float = 0.0,
) -> torch.Tensor:
    """BGK collision with Guo body force, written into ``out`` (no allocation).

    Guo, Zheng & Shi (2002): the equilibrium is evaluated at the force-corrected
    velocity ``u* = u + F/(2 rho)`` and the discrete force term

    ``F_q = w_q (1 - 1/(2 tau)) [3 (c_q - u)·F + 9 (c_q·u)(c_q·F)]``

    is added to the post-collision populations.  Both halves are needed:
    dropping the ``u*`` shift leaves an O(Ma²) steady-state slip at the wall.
    """
    rho, ux, uy, uz = _macroscopic(f, C_)
    inv_rho = 1.0 / torch.clamp(rho, min=1e-12)
    # Force density per unit volume (acceleration x density), x-direction only.
    fx = body_force * rho
    # Equilibrium uses the force-corrected velocity u* = u + F/(2 rho);
    # the discrete force term uses the *actual* velocity u, so that
    # sum_q F_q == 0 exactly (mass conservation).
    usx = ux + 0.5 * fx * inv_rho
    usy = uy
    usz = uz

    cx = C_[:, 0].view(-1, 1, 1, 1)
    cy = C_[:, 1].view(-1, 1, 1, 1)
    cz = C_[:, 2].view(-1, 1, 1, 1)

    cu = cx * usx + cy * usy + cz * usz
    usq = usx * usx + usy * usy + usz * usz
    feq = W_.view(-1, 1, 1, 1) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    # BGK: out = f - (f - feq)/tau  ==  f (1 - 1/tau) + feq/tau
    out.copy_(f).mul_(1.0 - 1.0 / tau).add_(feq, alpha=1.0 / tau)

    if body_force != 0.0:
        cu_phys = cx * ux + cy * uy + cz * uz
        cuf = cx * fx  # F has only an x component
        force_q = (
            W_.view(-1, 1, 1, 1)
            * (1.0 - 0.5 / tau)
            * (3.0 * (cx - ux) * fx + 9.0 * cu_phys * cuf)
        )
        out.add_(force_q)
    return out


def stream_into(f: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Periodic streaming ``out_q(x) = f_q(x - c_q)`` into ``out``."""
    out[0].copy_(f[0])
    for q in range(1, f.shape[0]):
        sz, sy, sx = _SHIFTS[q]
        if sx == 0 and sy == 0 and sz == 0:
            out[q].copy_(f[q])
        else:
            out[q].copy_(torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2)))
    return out


def apply_channel_walls(f: torch.Tensor, f_post: torch.Tensor) -> torch.Tensor:
    """Half-way bounce-back on the two ``y`` wall planes, in place.

    Populations that streamed *out of* a wall row (``c_y = +1`` at the first
    fluid row ``j=1``, ``c_y = -1`` at the last fluid row ``j=ny-2``) are
    replaced by the mirror image of the post-collision population of the same
    node.  The no-slip plane then sits half-way between the wall row and the
    fluid row, i.e. at ``y = 0.5`` and ``y = ny - 1.5`` (Ziegler 1993; Ladd
    1994) — which is exactly what ``height = ny - 2`` assumes.
    """
    for i in _UP:
        f[i, :, 1, :].copy_(f_post[_OPP[i], :, 1, :])
    for i in _DOWN:
        f[i, :, -2, :].copy_(f_post[_OPP[i], :, -2, :])
    return f


def apply_channel_walls_nebb(
    f: torch.Tensor,
    cfg: GPUCorePathConfig,
    C_: torch.Tensor,
    W_: torch.Tensor,
) -> torch.Tensor:
    """Force-corrected NEBB walls on the two wall rows (``j = 0``, ``j = ny-1``).

    Wall-node closure: the no-slip planes *are* the wall rows, so the channel
    width becomes ``H = ny - 1`` (instead of ``ny - 2`` for the half-way
    bounce-back) and the wall value of ``ux`` is part of the solution.  The
    wall velocity fed to the equilibrium is ``-a/(2 rho_w)`` (``a`` =
    body-force acceleration, ``rho_w`` = wall density recovered from the
    field) — the force correction the Guo collision expects of a no-slip
    wall.  Use :func:`channel_analytic_velocity_nebb` for the matching
    reference profile.
    """
    from .boundary.nebb import nebb_wall_plane, wall_density, wall_directions

    a = cfg.effective_body_force
    for row, sign in ((0, -1), (cfg.ny - 1, +1)):
        dirs = wall_directions(C_, sign)
        rho_w = wall_density(f, row, dirs)
        zeros = torch.zeros_like(rho_w)
        ux_w = zeros - (a / (2.0 * rho_w) if a else 0.0)
        nebb_wall_plane(
            f, row, sign, rho_w, ux_w, zeros.to(f.dtype),
            C=C_, W=W_, OPP=OPPOSITE,
        )
    return f


def core_step(
    f: torch.Tensor,
    f_post: torch.Tensor,
    f_out: torch.Tensor,
    cfg: GPUCorePathConfig,
    C_: torch.Tensor,
    W_: torch.Tensor,
) -> torch.Tensor:
    """One fused core-path step: collide (+Guo force) → stream → walls.

    ``f_post`` and ``f_out`` are caller-owned scratch buffers, so the step
    performs zero allocations; ``f`` itself is not modified.
    """
    collide_guo_into(f, cfg.tau, f_post, C_, W_, body_force=cfg.effective_body_force)
    stream_into(f_post, f_out)
    if cfg.init == "taylor_green":
        return f_out  # fully periodic case, no walls
    if cfg.wall_bc == "nebb":
        return apply_channel_walls_nebb(f_out, cfg, C_, W_)
    return apply_channel_walls(f_out, f_post)


def _compiled_core(
    f: torch.Tensor,
    f_post: torch.Tensor,
    f_out: torch.Tensor,
    tau: float,
    body_force: float,
    C_: torch.Tensor,
    W_: torch.Tensor,
) -> torch.Tensor:
    """Collision+streaming body handed to :func:`torch.compile`.

    Walls are applied outside the compiled region: they are two row copies, and
    a graph break there would cost more than it saves.
    """
    collide_guo_into(f, tau, f_post, C_, W_, body_force=body_force)
    stream_into(f_post, f_out)
    return f_out


# ---------------------------------------------------------------------------
# Analytic references
# ---------------------------------------------------------------------------


def channel_analytic_velocity(cfg: GPUCorePathConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic Poiseuille profile of the body-force driven channel.

    ``u(y) = a / (2 nu) * s * (H - s)`` with ``s = y - 0.5`` the distance from
    the bottom no-slip plane (Batchelor 1967 §4.2, in lattice units).
    """
    y = torch.arange(cfg.ny, dtype=torch.float64)
    s = torch.clamp(y - 0.5, min=0.0, max=cfg.height)
    u = cfg.body_force / (2.0 * cfg.nu) * s * (cfg.height - s)
    return y, u


def channel_analytic_velocity_nebb(cfg: GPUCorePathConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic Poiseuille profile on the ``wall_bc="nebb"`` node grid.

    With wall-node closure the no-slip planes coincide with the wall rows, so
    the profile ``ux(y) = a/(2 nu) y (H - y)`` uses ``H = ny - 1`` and every
    row ``y = 0 .. ny-1`` (including the two wall rows, where it vanishes).
    Returns ``(y, ux_profile)`` with ``ux_profile`` shaped ``(ny,)``.
    """
    H = float(cfg.ny - 1)
    y = torch.arange(cfg.ny, dtype=torch.float64)
    prof = cfg.body_force / (2.0 * cfg.nu) * y * (H - y)
    return y, prof


def taylor_green_analytic_energy(cfg: GPUCorePathConfig, t: Any) -> Any:
    """Exact kinetic energy of the 2-D Taylor–Green vortex at time ``t``.

    ``u = u0 e^{-2 nu k² t} (sin kx cos ky, -cos kx sin ky)`` with
    ``k = 2 pi / nx``; the mean kinetic energy is ``u0²/4 e^{-4 nu k² t}``
    (Taylor & Green 1937; exact Navier–Stokes solution).
    """
    k = 2.0 * math.pi / cfg.nx
    e0 = 0.25 * cfg.tg_u0**2
    if isinstance(t, torch.Tensor):
        return e0 * torch.exp(-4.0 * cfg.nu * k * k * t.to(torch.float64))
    return e0 * math.exp(-4.0 * cfg.nu * k * k * float(t))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class GPUCorePathSolver:
    """Unified core-path solver (BGK + Guo force + half-way bounce-back).

    All state lives in buffers allocated by :meth:`__init__`; :meth:`step`
    performs no heap allocation, which is what makes the path usable inside a
    long GPU time loop.
    """

    def __init__(self, cfg: GPUCorePathConfig):
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = cfg.precision_policy
        self.compute_dtype = policy.compute_dtype
        self.store_dtype = policy.store_dtype

        self.C = C.to(dtype=self.compute_dtype)
        # The shipped weights are float32 constants whose sum is 1 +- 1e-8; a
        # second-order equilibrium built from them injects that much mass per
        # step (1.5e-8/step drift measured).  Renormalise in the working
        # precision so that sum_q feq == rho to round-off.
        self.W = W.to(dtype=self.compute_dtype)
        self.W = self.W / self.W.sum()

        self.backend = get_best_available_backend(cfg.backend)
        self.backend_note = ""
        if self.backend == "torch_compile" and os.environ.get("TL_CORE_PATH_COMPILE", "1") == "0":
            self.backend = "eager"
            self.backend_note = "torch_compile disabled by TL_CORE_PATH_COMPILE=0"
        if self.backend == "triton":
            # The Triton kernels live in tensorlbm.triton_fused; the core path
            # only routes through them when the CUDA build is importable.
            try:
                from .triton_fused import TritonFusedSolver  # noqa: F401
            except Exception as exc:  # pragma: no cover - CUDA-only branch
                self.backend = "eager"
                self.backend_note = f"triton import failed: {type(exc).__name__}: {exc}"

        shape = (19, cfg.nz, cfg.ny, cfg.nx)
        self._fa = torch.zeros(shape, dtype=self.compute_dtype)
        self._fb = torch.zeros(shape, dtype=self.compute_dtype)
        self.buf = LBMStepBuffer.for_lattice(
            "D3Q19", cfg.nz, cfg.ny, cfg.nx, device="cpu", dtype=self.compute_dtype
        )
        self._f_store: torch.Tensor | None = None
        if self.store_dtype != self.compute_dtype:
            self._f_store = torch.zeros(shape, dtype=self.store_dtype)

        if cfg.wall_bc != "halfway" and self.backend != "eager":
            self.backend = "eager"
            self.backend_note = f"wall_bc={cfg.wall_bc} runs on the eager backend"

        self._compiled_fn = None
        self._step_fn: Callable[[], None] = (
            self._step_compiled if self.backend == "torch_compile" else self._step_eager
        )

        self._init_field()
        self.steps = 0
        self._stats: dict[str, Any] = {}

    # -- initialisation -------------------------------------------------
    def _init_field(self) -> None:
        cfg = self.cfg
        nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
        C_ = self.C
        device, dtype = self.device, self.compute_dtype

        ux = torch.zeros(nz, ny, nx, dtype=dtype, device=device)
        uy = torch.zeros(nz, ny, nx, dtype=dtype, device=device)
        uz = torch.zeros(nz, ny, nx, dtype=dtype, device=device)

        if cfg.init == "parabolic":
            _, prof = channel_analytic_velocity(cfg)
            ux[:] = prof.to(device=device, dtype=dtype).view(1, ny, 1)
        elif cfg.init == "taylor_green":
            x = torch.arange(nx, dtype=dtype, device=device).view(1, 1, nx)
            y = torch.arange(ny, dtype=dtype, device=device).view(1, ny, 1)
            k = 2.0 * math.pi / nx
            ux[:] = cfg.tg_u0 * torch.sin(k * x) * torch.cos(k * y)
            uy[:] = -cfg.tg_u0 * torch.cos(k * x) * torch.sin(k * y)

        if cfg.perturbation_amplitude:
            g = torch.Generator().manual_seed(cfg.seed)
            amp = cfg.perturbation_amplitude * cfg.u_max_target
            ux += amp * torch.randn(nz, ny, nx, generator=g).to(device=device, dtype=dtype)
            uz += amp * torch.randn(nz, ny, nx, generator=g).to(device=device, dtype=dtype)
            if cfg.init != "taylor_green":
                ux[:, 0, :] = 0.0
                ux[:, -1, :] = 0.0
                uz[:, 0, :] = 0.0
                uz[:, -1, :] = 0.0

        rho = torch.ones(nz, ny, nx, dtype=dtype, device=device)
        cu = (
            C_[:, 0].view(-1, 1, 1, 1) * ux
            + C_[:, 1].view(-1, 1, 1, 1) * uy
            + C_[:, 2].view(-1, 1, 1, 1) * uz
        )
        u2 = ux * ux + uy * uy + uz * uz
        feq = self.W.view(-1, 1, 1, 1) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
        self._fa.zero_()
        self._fa += feq
        if self._f_store is not None:
            self._f_store.copy_(self._fa)

    # -- stepping -------------------------------------------------------
    def _step_eager(self) -> None:
        core_step(self._fa, self.buf.f_post, self._fb, self.cfg, self.C, self.W)
        self._fa, self._fb = self._fb, self._fa

    def _step_compiled(self) -> None:
        fn = self._compiled_fn
        if fn is None:
            fn = torch.compile(_compiled_core, dynamic=False)
            self._compiled_fn = fn
        fn(self._fa, self.buf.f_post, self._fb, self.cfg.tau, self.cfg.effective_body_force,
           self.C, self.W)
        if self.cfg.init != "taylor_green":
            apply_channel_walls(self._fb, self.buf.f_post)
        self._fa, self._fb = self._fb, self._fa

    @property
    def f(self) -> torch.Tensor:
        """Current distribution in the store dtype (converted on access)."""
        if self._f_store is None:
            return self._fa
        self._f_store.copy_(self._fa)
        return self._f_store

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._step_fn()
        self.steps += n

    def run(
        self, n_steps: int | None = None, callback: Callable | None = None
    ) -> dict[str, Any]:
        """Advance the solution and return run statistics (including MLUPS)."""
        n = int(n_steps if n_steps is not None else self.cfg.steps)
        t0 = time.perf_counter()
        self.step(n)
        elapsed = time.perf_counter() - t0
        cells = self.cfg.nx * self.cfg.ny * self.cfg.nz
        self._stats = {
            "label": self.cfg.label,
            "steps": n,
            "total_steps": self.steps,
            "elapsed_s": elapsed,
            "mlups": cells * n / elapsed / 1e6 if elapsed > 0 else 0.0,
            "backend": self.backend,
            "backend_note": self.backend_note,
            "device": str(self.device),
            "compute_dtype": str(self.compute_dtype),
            "store_dtype": str(self.store_dtype),
            "cells": cells,
        }
        if callback is not None:
            callback(self.steps, self)
        return dict(self._stats)

    # -- diagnostics ----------------------------------------------------
    def macroscopic(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(rho, ux, uy, uz)`` on the full grid, in compute dtype."""
        return _macroscopic(self._fa, self.C)

    def interior(self) -> dict[str, torch.Tensor]:
        """Macroscopic fields restricted to the fluid rows ``1 .. ny-2``.

        The two wall rows hold populations that are never streamed (they only
        receive reflected images on the fluid side), so every statistic must
        exclude them; keeping that rule in one place stops wall-row garbage
        from leaking into validation numbers.
        """
        rho, ux, uy, uz = self.macroscopic()
        sl = (slice(None), slice(1, self.cfg.ny - 1), slice(None))
        return {"rho": rho[sl], "ux": ux[sl], "uy": uy[sl], "uz": uz[sl]}

    def channel_profile(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Plane-averaged streamwise velocity per fluid row (y, u_mean)."""
        inter = self.interior()
        u_mean = inter["ux"].mean(dim=(0, 2))
        y = torch.arange(1, self.cfg.ny - 1, dtype=torch.float64)
        return y, u_mean.to(torch.float64)

    def kinetic_energy(self) -> float:
        """Mean kinetic energy over the fluid region (Taylor–Green diagnostic)."""
        inter = self.interior()
        rho = inter["rho"]
        return float(
            (0.5 * rho * (inter["ux"] ** 2 + inter["uy"] ** 2 + inter["uz"] ** 2)).mean().item()
        )

    def mass(self) -> float:
        """Mean density over the whole grid (mass-conservation diagnostic)."""
        return float(self._fa.sum(dim=0).mean().item())


# ---------------------------------------------------------------------------
# Distributed (multi-GPU) wrapper
# ---------------------------------------------------------------------------


class DistributedGPUCorePathSolver:
    """Multi-GPU core path: ``x``-slab decomposition + selective halo exchange.

    Each rank owns ``nx / world_size`` planes plus one ghost plane per face.
    Only the five populations that cross an ``x`` face are exchanged per step
    (5/19 of the state), which is the bandwidth reduction behind the T2
    multi-GPU dimension.  Requires ``torch.distributed`` and CUDA; on a CPU-only
    host construction raises instead of quietly running serially.
    """

    CROSSING_PLUS = [int(i) for i in range(19) if C[i, 0] == 1]
    CROSSING_MINUS = [int(i) for i in range(19) if C[i, 0] == -1]

    def __init__(self, cfg: GPUCorePathConfig, rank: int = 0, world_size: int = 1):
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the multi-GPU core path")
        if cfg.nx % world_size:
            raise ValueError("nx must be divisible by world_size")
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.local_nx = cfg.nx // world_size
        local_cfg = GPUCorePathConfig(
            nx=self.local_nx + 2,  # one ghost plane per side
            ny=cfg.ny,
            nz=cfg.nz,
            tau=cfg.tau,
            u_max_target=cfg.u_max_target,
            init=cfg.init,
            tg_u0=cfg.tg_u0,
            precision=cfg.precision,
            backend=cfg.backend,
            seed=cfg.seed,
            perturbation_amplitude=cfg.perturbation_amplitude,
            steps=cfg.steps,
            label=f"{cfg.label}_rank{rank}",
        )
        self.local = GPUCorePathSolver(local_cfg)

    def halo_exchange(self) -> None:
        """Exchange the five crossing populations of each ``x`` face."""
        f = self.local._fa
        plus = torch.tensor(self.CROSSING_PLUS, device=f.device)
        minus = torch.tensor(self.CROSSING_MINUS, device=f.device)
        send_right = f.index_select(0, plus)[:, :, :, -2].contiguous()
        send_left = f.index_select(0, minus)[:, :, :, 1].contiguous()
        recv_left = torch.empty_like(send_left)
        recv_right = torch.empty_like(send_right)
        if self.world_size > 1:
            nxt = (self.rank + 1) % self.world_size
            prv = (self.rank - 1) % self.world_size
            reqs = [
                torch.distributed.isend(send_right, nxt),
                torch.distributed.irecv(recv_left, prv),
                torch.distributed.isend(send_left, prv),
                torch.distributed.irecv(recv_right, nxt),
            ]
            for req in reqs:
                req.wait()
            f.index_copy_(0, plus, torch.cat([recv_left, f.index_select(0, plus)[:, :, :, 1:-1]], dim=-1))
            f.index_copy_(0, minus, torch.cat([f.index_select(0, minus)[:, :, :, 1:-1], recv_right], dim=-1))

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.local.step(1)
            self.halo_exchange()

    def run(self, n_steps: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        self.step(n_steps)
        elapsed = time.perf_counter() - t0
        cells = self.local_nx * self.cfg.ny * self.cfg.nz
        return {
            "steps": n_steps,
            "elapsed_s": elapsed,
            "mlups": cells * n_steps / elapsed / 1e6 if elapsed else 0.0,
            "rank": self.rank,
            "world_size": self.world_size,
        }


# ---------------------------------------------------------------------------
# Validation cases (analytic references, JSON-friendly results)
# ---------------------------------------------------------------------------


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    denom = float(torch.linalg.vector_norm(ref))
    if denom == 0.0:
        return float("inf")
    return float(torch.linalg.vector_norm(got - ref)) / denom


def run_channel_case(
    cfg: GPUCorePathConfig, *, check_every: int = 200, steady_tol: float = 1e-6
) -> dict[str, Any]:
    """Spin up the channel from rest and compare with the analytic parabola.

    The comparison is done on the plane-averaged streamwise profile (the
    quantity the analytic solution predicts) after the flow has reached a
    steady state; the steady state is *demonstrated*, not assumed: the
    centre-line velocity is sampled every ``check_every`` steps and the run
    stops early once ``|du_max| / u_max < 1e-6`` over one interval.
    """
    cfg.validate()
    solver = GPUCorePathSolver(cfg)
    t0 = time.perf_counter()
    history: list[dict[str, float]] = []
    steps_done = 0
    steady_at = None
    prev = None
    while steps_done < cfg.steps:
        block = min(check_every, cfg.steps - steps_done)
        solver.step(block)
        steps_done += block
        y, u = solver.channel_profile()
        u_max = float(u.max().item())
        history.append({"step": steps_done, "u_max": u_max})
        if prev is not None and u_max > 0:
            if abs(u_max - prev) / u_max < steady_tol and steady_at is None:
                steady_at = steps_done
                break
        prev = u_max
    elapsed = time.perf_counter() - t0

    y, u_got = solver.channel_profile()
    _, u_ref_all = channel_analytic_velocity(cfg)
    u_ref = u_ref_all[1 : cfg.ny - 1].to(torch.float64)
    cells = cfg.nx * cfg.ny * cfg.nz
    rho, ux, uy, uz = solver.macroscopic()
    transport = float((rho * ux).mean().item())

    u_max_ref = float(u_ref.max().item())
    u_max_got = float(u_got.max().item())
    return {
        "case": "channel_poiseuille",
        "label": cfg.label,
        "config": cfg.as_dict(),
        "backend": solver.backend,
        "backend_note": solver.backend_note,
        "device": str(solver.device),
        "compute_dtype": str(solver.compute_dtype),
        "steps": steps_done,
        "steady_at_step": steady_at,
        "elapsed_s": elapsed,
        "mlups": cells * steps_done / elapsed / 1e6 if elapsed > 0 else 0.0,
        "reference": "u(y)=a/(2 nu) y(H-y), H=ny-2, walls at 0.5/ny-1.5",
        "u_max_ref": u_max_ref,
        "u_max_got": u_max_got,
        "u_max_rel_err": abs(u_max_got - u_max_ref) / u_max_ref,
        "profile_rel_l2": _rel_l2(u_got, u_ref),
        "profile_mean_abs_err": float((u_got - u_ref).abs().mean().item()),
        "mass_drift": abs(float(rho.mean().item()) - 1.0),
        "mean_x_momentum": transport,
        "max_divergence_proxy": float(
            (ux[:, 1:-1, :] - ux[:, :-2, :]).abs().max().item()
        ),
        "y": y.tolist(),
        "u_got": u_got.tolist(),
        "u_ref": u_ref.tolist(),
        "u_max_history": history,
    }


def run_taylor_green_case(cfg: GPUCorePathConfig, *, samples: int = 5) -> dict[str, Any]:
    """Decay of the Taylor–Green vortex against the exact solution."""
    if cfg.init != "taylor_green":
        cfg = GPUCorePathConfig(**{**cfg.__dict__, "init": "taylor_green"})
    cfg.validate()
    solver = GPUCorePathSolver(cfg)
    e0 = solver.kinetic_energy()
    e0_ref = float(taylor_green_analytic_energy(cfg, 0.0))
    t0 = time.perf_counter()
    per = max(cfg.steps // samples, 1)
    rows: list[dict[str, float]] = []
    for i in range(1, samples + 1):
        solver.step(per)
        t = solver.steps
        e = solver.kinetic_energy()
        e_ref = float(taylor_green_analytic_energy(cfg, t))
        rows.append(
            {
                "step": t,
                "ke_got": e,
                "ke_ref": e_ref,
                "rel_err": abs(e - e_ref) / e_ref,
            }
        )
    elapsed = time.perf_counter() - t0
    cells = cfg.nx * cfg.ny * cfg.nz
    return {
        "case": "taylor_green_2d_in_3d",
        "label": cfg.label,
        "config": cfg.as_dict(),
        "backend": solver.backend,
        "device": str(solver.device),
        "compute_dtype": str(solver.compute_dtype),
        "steps": solver.steps,
        "elapsed_s": elapsed,
        "mlups": cells * solver.steps / elapsed / 1e6 if elapsed > 0 else 0.0,
        "reference": "KE(t)=u0^2/4 exp(-4 nu k^2 t), k=2 pi/nx (Taylor & Green 1937)",
        "ke0_got": e0,
        "ke0_ref": e0_ref,
        "ke0_rel_err": abs(e0 - e0_ref) / e0_ref,
        "samples": rows,
        "max_rel_err": max(r["rel_err"] for r in rows),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _suite(ny_list: list[int], tau_list: list[float], steps: int, out: str | None) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for ny in ny_list:
        for tau in tau_list:
            cfg = GPUCorePathConfig(
                ny=ny, tau=tau, steps=steps, label=f"channel_ny{ny}_tau{tau:.2f}"
            )
            results.append(run_channel_case(cfg))
    tg: list[dict[str, Any]] = []
    for nx in (32, 64):
        cfg = GPUCorePathConfig(
            nx=nx,
            ny=nx,
            nz=8,
            tau=0.6,
            init="taylor_green",
            tg_u0=0.02,
            steps=max(200, int(nx * 3)),
            label=f"tg_nx{nx}",
        )
        tg.append(run_taylor_green_case(cfg))
    payload = {"channel": results, "taylor_green": tg}
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="T2 core-path analytic validation")
    ap.add_argument("--ny", type=int, nargs="+", default=[18, 34])
    ap.add_argument("--tau", type=float, nargs="+", default=[0.8, 1.2])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)
    payload = _suite(args.ny, args.tau, args.steps, args.out)
    for row in payload["channel"]:
        print(
            f"{row['label']:22s} u_max ref={row['u_max_ref']:.6f} got={row['u_max_got']:.6f} "
            f"err={100 * row['u_max_rel_err']:.3f}%  L2={100 * row['profile_rel_l2']:.3f}%  "
            f"{row['mlups']:.3f} MLUPS  steady@{row['steady_at_step']}"
        )
    for row in payload["taylor_green"]:
        print(
            f"{row['label']:22s} KE rel err max={100 * row['max_rel_err']:.3f}%  "
            f"{row['mlups']:.3f} MLUPS"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())