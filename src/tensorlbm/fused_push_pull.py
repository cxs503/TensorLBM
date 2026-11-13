"""Fused push-pull (single-kernel stream + collide) D2Q9 stepper.

Anchor
------
``XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN``.  The reference implementation is the
upstream chapter 13 GPU code of ``lbm-principles-practice/code`` (MIT, Krüger et
al., *The LBM: Principles and Practice*, Springer 2016), kernel
``gpu_stream_collide_save`` in ``chapter13/gpu/LBM.cu``.  The source was fetched
through the GitHub contents API and is snapshotted (with sha256) under
``xval/fused_push_pull_aa_pattern/ref/`` -- no line of the upstream method below
was transcribed from prose.

Upstream method (read off the source, not paraphrased)
------------------------------------------------------
``chapter13/gpu/LBM.cu:23-35`` -- storage split and layout::

    gpu_field0_index(x, y)        = NX*y + x                  # rest direction, own array
    gpu_fieldn_index(x, y, d)     = NX*(NY*(d-1) + y) + x     # d = 1..8, direction-major

``chapter13/gpu/LBM.cu:172-252`` -- ``gpu_stream_collide_save(f0,f1,f2,r,u,v,save)``
does, in one kernel launch per timestep and in registers only:

1. *pull* the eight shifted neighbours of the **post-collision** field ``f1``
   (``LBM.cu:181-201``), wrapping periodically with ``%NX`` / ``%NY``::

       ft1 = f1[fieldn(xm1, y,   1)]   ...   ft8 = f1[fieldn(xm1, yp1, 8)]

2. moments from those nine numbers only (``rho``, ``ux``, ``uy``);
   ``r``/``u``/``v`` are written **only when the save flag is set**
   (``LBM.cu:203-216``);

3. relax to equilibrium and write the **post-collision** populations of
   directions 1..8 into the *other* buffer ``f2`` (``LBM.cu:218-252``) and the
   rest direction back into ``f0`` in place.  ``main.cu`` then swaps ``f1`` and
   ``f2`` pointers.

So the fusion removes the intermediate *post-streaming / pre-collision* field:
there is no array that ever holds "streamed but not yet collided" populations.
The price is one extra 8-direction buffer (``f1``/``f2`` ping-pong, 17 ``N``
doubles per node including ``f0``) instead of the 18 ``N`` of the textbook AB
pattern; the "A-A" wording of the anchor refers precisely to that fused step
(read and write happen in the same kernel, no post-stream copy), not to
minimal-buffer storage.

What this module adds / does not add
------------------------------------
* Same *contract* as upstream: direction-major layout, rest direction in its own
  array, periodic wrap by index arithmetic, double-buffered 8-direction fields,
  save-flag gated macroscopic output, BGK with ``1/tau`` folded into the
  equilibrium weights (``tauinv = 2/(6 nu + 1)``, i.e. ``tau = 3 nu + 1/2``).
* Extension (not upstream): ``single_buffer=True`` runs the identical scheme
  with one 8-direction array (17 ``N`` -> 9 ``N`` doubles).  It is bit-identical
  **provided** every pull completes before any write -- which is what a
  vectorised gather guarantees.  It is *not* safe under a per-node sequential
  loop or a concurrent kernel; that hazard is the subject of
  :mod:`tensorlbm.aa_pattern`.
* Not upstream: no walls, no forcing, no multiphase.  Upstream chapter 13 is
  periodic, single-phase, unforced; this module keeps exactly that scope
  (``periodic`` arguments do not exist on purpose) and delegates everything else
  to the general :mod:`tensorlbm.solver` path.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

from .d2q9 import C


def _relaxation(tau: float) -> tuple[float, float]:
    tau = float(tau)
    if not tau > 0.5:
        raise ValueError(f"tau must be > 0.5, got {tau}")
    tauinv = 1.0 / tau
    return tauinv, 1.0 - tauinv


def tau_from_nu(nu: float) -> float:
    """Upstream ``tau = 3*nu + 1/2`` (``chapter13/gpu/LBM.h``: ``tau = 3.0*nu+0.5``)."""
    return 3.0 * float(nu) + 0.5


def nu_from_tau(tau: float) -> float:
    return (float(tau) - 0.5) / 3.0


#: D2Q9 weights in the upstream order (w0, ws x4, wd x4).
W0 = 4.0 / 9.0
WS = 1.0 / 9.0
WD = 1.0 / 36.0


@functools.cache
def _pull_index(ny: int, nx: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather index for the eight moving directions: ``dst[y,x] <- src[y-cy, x-cx]``.

    Periodic in both directions, exactly the upstream ``%NX`` / ``%NY`` wrap.
    """
    yy = torch.arange(ny, device=device).view(ny, 1).expand(ny, nx)
    xx = torch.arange(nx, device=device).view(1, nx).expand(ny, nx)
    c = C.to(device)
    sy = torch.empty(8, ny, nx, dtype=torch.int64, device=device)
    sx = torch.empty(8, ny, nx, dtype=torch.int64, device=device)
    for d in range(1, 9):
        sy[d - 1] = (yy - int(c[d, 1])) % ny
        sx[d - 1] = (xx - int(c[d, 0])) % nx
    return sy, sx


@dataclass
class FusedStepStats:
    """Per-step scalar bookkeeping (all in lattice units)."""

    mass: float
    momentum_x: float
    momentum_y: float


class FusedPushPullStepper:
    """D2Q9 BGK stepper using the upstream fused push-pull kernel.

    Parameters
    ----------
    f:
        Initial populations ``(9, ny, nx)``; direction order of
        :data:`tensorlbm.d2q9.C` (identical to the upstream numbering
        ``6 2 5 / 3 0 1 / 7 4 8``).
    tau:
        BGK relaxation time, ``> 0.5``.
    single_buffer:
        ``False`` (default) mirrors upstream: two 8-direction buffers swapped
        every step.  ``True`` keeps one (see the module docstring for the
        precondition under which it is safe).
    keep_momenta:
        Default ``True`` writes ``rho/ux/uy`` into attribute fields every step
        (the upstream ``save=True`` branch).  ``False`` reproduces the upstream
        ``save=False`` bandwidth saving and leaves the macroscopic attributes
        ``None``.

    Attributes
    ----------
    f0, f_pre, f_post:
        Rest-direction field ``(ny, nx)`` and the 8-direction buffers
        ``(8, ny, nx)``.  ``f_pre`` holds the populations at the current time
        level (post-collision of the previous step = pre-streaming of the next).
    rho, ux, uy:
        Macroscopic fields of the last step (``None`` if ``keep_momenta=False``).
    """

    n_persistent_fields_upstream: int = 17  # f0 + 2 x 8, per node (documented, structural)

    def __init__(
        self,
        f: torch.Tensor,
        tau: float,
        *,
        single_buffer: bool = False,
        keep_momenta: bool = True,
    ) -> None:
        if f.dim() != 3 or f.shape[0] != 9:
            raise ValueError(f"f must have shape (9, ny, nx), got {tuple(f.shape)}")
        self.tauinv, self.omtauinv = _relaxation(tau)
        self.tau = float(tau)
        self.nu = nu_from_tau(tau)
        f = f.detach().to(memory_format=torch.contiguous_format).clone()
        self.f0 = f[0].contiguous()
        self.f_pre = f[1:].contiguous()
        self.f_post = torch.empty_like(self.f_pre) if not single_buffer else self.f_pre
        self.single_buffer = bool(single_buffer)
        self.keep_momenta = bool(keep_momenta)
        self.time = 0
        self.rho: torch.Tensor | None = None
        self.ux: torch.Tensor | None = None
        self.uy: torch.Tensor | None = None
        self._sy, self._sx = _pull_index(
            int(f.shape[1]), int(f.shape[2]), f.device
        )

    # --------------------------------------------------------------- storage
    def persistent_doubles_per_node(self) -> int:
        """Structural storage claim in doubles per node (not a measurement)."""
        return 1 + (1 if self.single_buffer else 2) * 8

    # ------------------------------------------------------------------ step
    def _pull(self, fm: torch.Tensor) -> torch.Tensor:
        """Pull the eight shifted neighbours -> ``(8, ny, nx)`` in direction order."""
        q = torch.arange(8, device=fm.device).view(8, 1, 1).expand_as(fm)
        return fm[q, self._sy, self._sx]

    def step(self) -> FusedStepStats:
        """One fused stream+collide step (upstream ``gpu_stream_collide_save``)."""
        p = self._pull(self.f_pre)  # 0..7 <-> directions 1..8
        p1, p2, p3, p4, p5, p6, p7, p8 = p.unbind(0)

        rho = self.f0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8
        rhoinv = 1.0 / rho
        ux = rhoinv * (p1 + p5 + p8 - (p3 + p6 + p7))
        uy = rhoinv * (p2 + p5 + p6 - (p4 + p7 + p8))

        tw0r = self.tauinv * W0 * rho
        twsr = self.tauinv * WS * rho
        twdr = self.tauinv * WD * rho
        omusq = 1.0 - 1.5 * (ux * ux + uy * uy)
        tux = 3.0 * ux
        tuy = 3.0 * uy

        c_dot = (
            tux,
            tuy,
            -tux,
            -tuy,
            tux + tuy,
            tuy - tux,
            -(tux + tuy),
            tux - tuy,
        )
        tw = (twsr, twsr, twsr, twsr, twdr, twdr, twdr, twdr)
        new0 = self.omtauinv * self.f0 + tw0r * omusq
        # single_buffer: write back into the array just read (safe because ``p``
        # is already materialised); otherwise into the other buffer.
        out = p.new_empty(p.shape) if not self.single_buffer else self.f_pre
        for i, (pi, cdi, twi) in enumerate(zip(p.unbind(0), c_dot, tw, strict=True)):
            out[i] = self.omtauinv * pi + twi * (omusq + cdi * (1.0 + 0.5 * cdi))

        # write: rest direction in place (upstream), moving directions to the other buffer
        self.f0 = new0
        self.f_post = out
        if not self.single_buffer:
            self.f_pre, self.f_post = out, self.f_pre
        self.time += 1

        if self.keep_momenta:
            self.rho, self.ux, self.uy = rho, ux, uy
        return FusedStepStats(
            mass=float(rho.sum()),
            momentum_x=float((rho * ux).sum()),
            momentum_y=float((rho * uy).sum()),
        )

    def run(self, nsteps: int) -> list[FusedStepStats]:
        return [self.step() for _ in range(int(nsteps))]

    # ------------------------------------------------------------- read-out
    def fields(self) -> torch.Tensor:
        """Assemble the current populations as ``(9, ny, nx)``."""
        return torch.cat([self.f0.unsqueeze(0), self.f_pre], dim=0)

    def macroscopic(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.fields()
        rho = f.sum(dim=0)
        c = C.to(f.device)
        ux = (f * c[:, 0].view(9, 1, 1)).sum(dim=0) / rho
        uy = (f * c[:, 1].view(9, 1, 1)).sum(dim=0) / rho
        return rho, ux, uy


def taylor_green_fields(
    ny: int,
    nx: int,
    *,
    u_max: float,
    nu: float,
    rho0: float = 1.0,
    t: int = 0,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upstream 2-D Taylor-Green state (``chapter13/gpu/LBM.cu:65-84``).

    Cell-centred ``X = x + 1/2``, ``Y = y + 1/2``; ``td = 1/(nu (kx^2 + ky^2))``;
    velocities decay as ``exp(-t/td)``, pressure as ``exp(-2t/td)``, and the
    density carries the pressure through ``rho = rho0 + 3 P``.
    """
    import math

    kx = 2.0 * math.pi / nx
    ky = 2.0 * math.pi / ny
    td = 1.0 / (nu * (kx * kx + ky * ky))
    X = torch.arange(nx, dtype=dtype).view(1, nx) + 0.5
    Y = torch.arange(ny, dtype=dtype).view(ny, 1) + 0.5
    dec = math.exp(-1.0 * t / td)
    ux = -u_max * math.sqrt(ky / kx) * torch.cos(kx * X) * torch.sin(ky * Y) * dec
    uy = u_max * math.sqrt(kx / ky) * torch.sin(kx * X) * torch.cos(ky * Y) * dec
    P = (
        -0.25
        * rho0
        * u_max
        * u_max
        * ((ky / kx) * torch.cos(2.0 * kx * X) + (kx / ky) * torch.cos(2.0 * ky * Y))
        * math.exp(-2.0 * t / td)
    )
    rho = rho0 + 3.0 * P
    return rho, ux, uy


def equilibrium_upstream(
    rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor
) -> torch.Tensor:
    """Upstream ``gpu_init_equilibrium`` (``LBM.cu:120-159``), ``(9, ny, nx)``."""
    omusq = 1.0 - 1.5 * (ux * ux + uy * uy)
    tux = 3.0 * ux
    tuy = 3.0 * uy
    w0r = W0 * rho
    wsr = WS * rho
    wdr = WD * rho
    c_dot = (
        tux,
        tuy,
        -tux,
        -tuy,
        tux + tuy,
        tuy - tux,
        -(tux + tuy),
        tux - tuy,
    )
    out = [w0r * omusq]
    for i, cdi in enumerate(c_dot):
        wr = wdr if i >= 4 else wsr
        out.append(wr * (omusq + cdi * (1.0 + 0.5 * cdi)))
    return torch.stack(out, dim=0)


def tg_l2_errors(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    *,
    nx: int,
    ny: int,
    u_max: float,
    nu: float,
    t: int,
    rho0: float = 1.0,
) -> dict[str, float]:
    """Upstream flow-property diagnostics (``LBM.cu:231-262``).

    Relative L2 norms ``sqrt(sum e^2 / sum a^2)``; the density norm uses the
    *fluctuation* ``rhoa - rho0`` as upstream does.  Units are lattice
    (``*_lu``) and SI-free; absolute ``*_si`` keys are not produced here.
    """
    ra, uxa, uya = taylor_green_fields(
        ny, nx, u_max=u_max, nu=nu, rho0=rho0, t=t, dtype=rho.dtype
    )
    e2r = float(((rho - ra) ** 2).sum())
    a2r = float(((ra - rho0) ** 2).sum())
    e2x = float(((ux - uxa) ** 2).sum())
    a2x = float((uxa**2).sum())
    e2y = float(((uy - uya) ** 2).sum())
    a2y = float((uya**2).sum())
    out = {
        "rho_lu2": (e2r / a2r) ** 0.5 if a2r > 0 else float("nan"),
        "ux_lu2": (e2x / a2x) ** 0.5 if a2x > 0 else float("nan"),
        "uy_lu2": (e2y / a2y) ** 0.5 if a2y > 0 else float("nan"),
        "energy": float((rho * (ux * ux + uy * uy)).sum()),
    }
    return out