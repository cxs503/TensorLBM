"""Wall-modelled LES (WMLES) of a pressure-driven plane channel, D3Q19.

Purpose
-------
Provide the *high-Reynolds-number* tier that the wall-resolved solvers in
:mod:`tensorlbm.turbulence.turbulent_channel3d` cannot reach: the near-wall
region below the first off-wall node is replaced by a wall model
(:mod:`.models`), so the first resolved node sits at ``y+ = O(10-40)`` instead
of ``y+ < 1``.  That is a large reduction of the wall-normal resolution at the
same ``Re_tau`` -- the whole point of wall modelling.

Numerics
--------
* D3Q19, BGK collision with a Smagorinsky eddy viscosity
  (:func:`tensorlbm.turbulence.turbulence.collide_smagorinsky_bgk3d`);
* Guo (2002) body force for the driving pressure gradient
  (:func:`tensorlbm.boundary.wall_model.guo_body_force_d3q19`);
* periodic in x and z, halfway bounce-back in y at ``y = 0.5`` and
  ``y = ny - 0.5``;
* the wall model is evaluated at the *exchange* location (second node above the
  wall) and the modelled velocity is assimilated at the wall-adjacent node by
  shifting only the equilibrium part of the populations, so the
  non-equilibrium part (the stress) survives.

Two wall-model closures are available (``wall_model="equilibrium"`` /
``"nonequilibrium"``); in a statistically one-dimensional channel the
non-equilibrium source collapses to the driving acceleration, so the two
models are expected to agree -- that agreement is itself checked and reported.

Self-consistency gates recorded by :func:`run_channel_wmles`
-----------------------------------------------------------
``tau_w_model``    plane- and time-averaged wall-model shear stress;
``tau_w_balance``  ``rho * a * h`` (exact force balance of a stationary channel);
``tau_w_injected`` momentum injected by the wall treatment (bounce-back plus
                   assimilation), measured from the populations.

They must agree for a converged run; disagreement is reported, never hidden.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ..boundary.wall_model import guo_body_force_d3q19
from ..cumulant import collide_cumulant_d3q19
from ..d3q19 import OPPOSITE, C, equilibrium3d, macroscopic3d
from ..population_positivity import limit_nonequilibrium_for_positivity
from ..solver3d import stream3d
from .models import EquilibriumWallModel, NonEquilibriumWallModel

__all__ = [
    "ChannelWMLESConfig",
    "run_channel_wmles",
    "laminar_poiseuille_profile",
    "solenoidal_perturbation",
]

_CS2 = 1.0 / 3.0


@dataclass(frozen=True)
class ChannelWMLESConfig:
    """Configuration of a wall-modelled channel run (all lengths in lattice units)."""

    nx: int = 48
    nz: int = 48
    ny: int = 49
    re_tau: float = 590.0
    u_tau: float = 0.02
    smagorinsky_cs: float = 0.1
    steps: int = 20000
    averaging_start: int = 8000
    stat_interval: int = 10
    wall_model: str = "equilibrium"
    wall_law: str = "spalding"
    wall_coupling: str = "assimilate"
    wall_relaxation: float = 1.0
    wall_origin_offset: float = 1.0
    exchange_offset: int = 1
    diagnostics: bool = False
    exchange_averaging: str = "plane"
    perturbation_amplitude: float = 0.05
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
    log_interval: int = 2000
    positivity_limiter: bool = True
    run_name: str = "channel_wmles"

    def __post_init__(self) -> None:
        if not (0.5 <= self.wall_origin_offset <= 1.5):
            msg = "wall_origin_offset must lie in [0.5, 1.5] lattice units"
            raise ValueError(msg)
        if self.ny < 7:
            msg = "ny must be >= 7 (wall nodes plus an exchange location)"
            raise ValueError(msg)
        if self.wall_model not in {"equilibrium", "nonequilibrium"}:
            msg = f"unsupported wall_model {self.wall_model!r}"
            raise ValueError(msg)
        if not (0.0 < self.wall_relaxation <= 1.0):
            msg = "wall_relaxation must lie in (0, 1] (1 = no under-relaxation)"
            raise ValueError(msg)
        if self.wall_coupling not in {"assimilate", "stress", "none"}:
            msg = f"unsupported wall_coupling {self.wall_coupling!r}"
            raise ValueError(msg)
        if self.exchange_offset < 1:
            msg = "exchange_offset must be >= 1 (row counted from the wall node)"
            raise ValueError(msg)
        if self.exchange_averaging not in {"plane", "local"}:
            msg = f"unsupported exchange_averaging {self.exchange_averaging!r}"
            raise ValueError(msg)
        if self.averaging_start >= self.steps:
            msg = "averaging_start must be < steps"
            raise ValueError(msg)

    @property
    def H(self) -> float:
        """Channel height between the two bounce-back walls (lattice units)."""
        return self.ny - 1.0

    @property
    def h(self) -> float:
        return self.H / 2.0

    @property
    def nu(self) -> float:
        """Molecular viscosity implied by ``Re_tau = u_tau h / nu``."""
        return self.u_tau * self.h / self.re_tau

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def accel(self) -> float:
        """Driving acceleration giving ``tau_w = rho a h``."""
        return 2.0 * self.u_tau**2 / self.H

    @property
    def delta_y_plus(self) -> float:
        return self.u_tau / self.nu

    @property
    def j_wall_node(self) -> int:
        """First row that carries a *physical* velocity (a fluid node).

        With ``wall_origin_offset = 1.0`` the reflect plane of the
        post-streaming bounce-back sits one cell inside row 0 (between rows 0
        and 1), so row 0 *is* the wall row and row 1 is the first fluid node at
        ``y = 0.5``.  The historical default (``0.5``) treated row 0 as a fluid
        node half a cell from the wall, which made the wall model sample the
        velocity one cell too far out and collapse ``u_tau`` to 0.60 of the
        force-balance value (see ``docs/wall_model_les_m7797_report.md``).
        """
        return int(math.ceil(self.wall_origin_offset - 0.5))

    @property
    def j_exchange(self) -> int:
        """Row where the wall model samples the resolved velocity.

        ``exchange_offset = 1`` (default) samples the first resolved node above
        the wall node; larger values push the sampling point outwards, which is
        only meaningful when ``ny`` leaves room for it.  It exists because the
        exchange location is one of the knobs that control the coupling
        stability of a WMLES (see ``benchmarks/wall_model_les/re590_stability.py``).
        """
        return self.j_wall_node + self.exchange_offset

    def y_of_node(self, j: int) -> float:
        """Distance of row ``j`` from the lower wall (measured wall origin)."""
        return j + 0.5 - self.wall_origin_offset

    @property
    def H_effective(self) -> float:
        """Channel height between the two measured wall origins."""
        return self.H - 2.0 * self.wall_origin_offset


# ---------------------------------------------------------------------------
# initialisation helpers
# ---------------------------------------------------------------------------
def laminar_poiseuille_profile(config: ChannelWMLESConfig) -> torch.Tensor:
    """Parabolic mean profile ``u(y) = a/(2 nu) y (H - y)`` on the fluid rows."""
    y = torch.arange(config.ny, dtype=torch.float64) + 0.5 - config.wall_origin_offset
    return config.accel / (2.0 * config.nu) * y * (config.H - y)


def initial_mean_profile(config: ChannelWMLESConfig) -> torch.Tensor:
    """Anisotropic (law-of-the-wall) mean profile used to initialise the channel.

    Initialising from the *laminar* Poiseuille parabola would be catastrophic at
    high ``Re_tau``: the same driving force and molecular viscosity give
    ``U_max ~ 6`` (lattice units) instead of ``~ 0.4``, i.e. a lattice Mach
    number of order ten, and the run diverges within a few hundred steps.  The
    Reichardt (1951) composite law at the target ``u_tau`` gives a
    turbulent-looking profile with ``U_max ~ 20 u_tau``; it is mirrored about
    the centreline because both walls are modelled.
    """
    y = torch.arange(config.ny, dtype=torch.float64) + 0.5 - config.wall_origin_offset
    y_plus = (y * config.u_tau / config.nu).clamp_min(1.0e-6)
    kappa = 0.41
    u_plus = torch.log1p(kappa * y_plus) / kappa + 7.8 * (
        1.0 - torch.exp(-y_plus / 11.0) - (y_plus / 11.0) * torch.exp(-y_plus / 3.0)
    )
    mirrored = 0.5 * (u_plus + u_plus.flip(0))
    return mirrored * config.u_tau


def solenoidal_perturbation(
    config: ChannelWMLESConfig,
    *,
    modes: int = 8,
    seed: int = 0,
) -> torch.Tensor:
    """Random solenoidal perturbation ``u = curl A``, shape ``(3, nz, ny, nx)``.

    ``A`` is a sum of separable Fourier modes in x and z windowed by
    ``sin(pi y / H)`` so the perturbation neither violates periodicity nor
    leaks through the walls.  Derivatives are spectral in x/z and central in y,
    hence ``div u`` vanishes to discretisation accuracy.
    """
    gen = torch.Generator().manual_seed(int(seed))
    shape = (config.nz, config.ny, config.nx)
    ax = torch.zeros(shape, dtype=torch.float64)
    ay = torch.zeros(shape, dtype=torch.float64)
    az = torch.zeros(shape, dtype=torch.float64)
    y = torch.arange(config.ny, dtype=torch.float64) + 0.5 - config.wall_origin_offset
    win = torch.sin(math.pi * y / config.H).view(1, -1, 1)
    zz = torch.arange(config.nz, dtype=torch.float64).view(-1, 1, 1) * (2 * math.pi / config.nz)
    xx = torch.arange(config.nx, dtype=torch.float64).view(1, 1, -1) * (2 * math.pi / config.nx)
    for _ in range(modes):
        mx = int(torch.randint(1, 4, (1,), generator=gen).item())
        mz = int(torch.randint(1, 4, (1,), generator=gen).item())
        phase = float(torch.rand(1, generator=gen).item()) * 2.0 * math.pi
        amp = 1.0 + float(torch.rand(1, generator=gen).item())
        field = amp * torch.cos(mx * xx + mz * zz + phase) * win
        ax = ax + field
        az = az + 0.5 * field
        ay = ay + 0.5 * field
    kx = torch.fft.fftfreq(config.nx, d=1.0) * config.nx * 1j
    kz = torch.fft.fftfreq(config.nz, d=1.0) * config.nz * 1j
    kx = kx.view(1, 1, -1)
    kz = kz.view(-1, 1, 1)

    def ddx(f: torch.Tensor) -> torch.Tensor:
        return torch.real(torch.fft.ifftn(torch.fft.fftn(f, dim=(0, 2)) * kx, dim=(0, 2)))

    def ddz(f: torch.Tensor) -> torch.Tensor:
        return torch.real(torch.fft.ifftn(torch.fft.fftn(f, dim=(0, 2)) * kz, dim=(0, 2)))

    def ddy(f: torch.Tensor) -> torch.Tensor:
        return torch.gradient(f, dim=1, spacing=1.0)[0]

    ux = ddy(az) - ddz(ay)
    uy = ddz(ax) - ddx(az)
    uz = ddx(ay) - ddy(ax)
    u = torch.stack([ux, uy, uz], dim=0)
    peak = float(u.abs().max())
    if peak > 0.0:
        u = u / peak
    return u


# ---------------------------------------------------------------------------
# boundary treatment
# ---------------------------------------------------------------------------
def _bounce_back_y(f: torch.Tensor, fpost: torch.Tensor) -> torch.Tensor:
    """Halfway bounce-back at the two wall rows (periodic wrap removed)."""
    c = C.to(f.device)
    opp = torch.as_tensor(OPPOSITE, device=f.device, dtype=torch.long)
    lower = c[:, 1] < 0  # populations heading into the lower wall
    upper = c[:, 1] > 0  # populations heading into the upper wall
    out = f
    out[lower, :, 0, :] = fpost[opp[lower], :, 0, :]
    out[upper, :, -1, :] = fpost[opp[upper], :, -1, :]
    return out


def _velocity_gradients(
    ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Nine velocity gradients with periodic x/z and one-sided y differences."""
    def dd(a: torch.Tensor, axis: int) -> torch.Tensor:
        if axis == 2:  # x, periodic
            return 0.5 * (torch.roll(a, -1, dims=2) - torch.roll(a, 1, dims=2))
        if axis == 0:  # z, periodic
            return 0.5 * (torch.roll(a, -1, dims=0) - torch.roll(a, 1, dims=0))
        d = torch.empty_like(a)
        d[:, 1:-1, :] = 0.5 * (a[:, 2:, :] - a[:, :-2, :])
        d[:, 0, :] = a[:, 1, :] - a[:, 0, :]
        d[:, -1, :] = a[:, -1, :] - a[:, -2, :]
        return d

    dux_dx, dux_dy, dux_dz = dd(ux, 2), dd(ux, 1), dd(ux, 0)
    duy_dx, duy_dy, duy_dz = dd(uy, 2), dd(uy, 1), dd(uy, 0)
    duz_dx, duz_dy, duz_dz = dd(uz, 2), dd(uz, 1), dd(uz, 0)
    return (
        dux_dx, dux_dy, dux_dz,
        duy_dx, duy_dy, duy_dz,
        duz_dx, duz_dy, duz_dz,
    )  # fmt: skip


def _strain_sq(grads: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """``S_ij S_ij`` for the symmetric strain-rate tensor."""
    dux_dx, dux_dy, dux_dz, duy_dx, duy_dy, duy_dz, duz_dx, duz_dy, duz_dz = grads
    s11 = dux_dx
    s22 = duy_dy
    s33 = duz_dz
    s12 = 0.5 * (dux_dy + duy_dx)
    s13 = 0.5 * (dux_dz + duz_dx)
    s23 = 0.5 * (duy_dz + duz_dy)
    return s11 * s11 + s22 * s22 + s33 * s33 + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23)


def _wall_model_velocity(
    *,
    config: ChannelWMLESConfig,
    wm,
    ux_ex: torch.Tensor,
    uz_ex: torch.Tensor,
    wall: str,
    source: float = 0.0,
):
    """Evaluate the wall model on one wall and return target node velocity."""
    y_ex = config.y_of_node(config.j_exchange)
    y_wn = config.y_of_node(config.j_wall_node)
    if config.exchange_averaging == "plane":
        mx = float(ux_ex.mean())
        mz = float(uz_ex.mean())
        mag = math.hypot(mx, mz)
        if mag < 1.0e-14:
            return 0.0, 0.0, 0.0, None
        sign = 1.0 if mx + mz >= 0.0 else -1.0
        direction = (mx / mag * sign, mz / mag * sign)
        res = wm.skin_friction(
            float(mag),
            y_ex,
            config.nu,
            source=source,
            y_wall_node=y_wn,
        )
        node = float(res.u_wall_node.reshape(-1)[0])
        u_tau = float(res.u_tau.reshape(-1)[0])
        return node * direction[0], node * direction[1], u_tau, res
    res = wm.skin_friction(
        torch.stack([ux_ex.reshape(-1), uz_ex.reshape(-1)], dim=1).norm(dim=1),
        y_ex,
        config.nu,
        source=source,
        y_wall_node=y_wn,
    )
    node = res.u_wall_node.reshape(-1)
    mag = torch.stack([ux_ex.reshape(-1), uz_ex.reshape(-1)], dim=1).norm(dim=1)
    safe = mag.clamp_min(1.0e-14)
    return (node * ux_ex.reshape(-1) / safe).reshape(ux_ex.shape), (
        node * uz_ex.reshape(-1) / safe
    ).reshape(uz_ex.shape), res.u_tau, res


def _blend_target(wx, wz, target_ux, target_uz, beta: float):
    """Under-relax a wall-model target towards the current wall-node velocity."""
    if not torch.is_tensor(target_ux):
        target_ux = torch.as_tensor(target_ux, dtype=wx.dtype, device=wx.device).expand_as(wx)
    if not torch.is_tensor(target_uz):
        target_uz = torch.as_tensor(target_uz, dtype=wz.dtype, device=wz.device).expand_as(wz)
    return (1.0 - beta) * wx + beta * target_ux, (1.0 - beta) * wz + beta * target_uz


def _assimilate(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    j: int,
    target_ux,
    target_uz,
) -> None:
    """Shift the equilibrium part of the populations at row ``j`` (in place).

    Only ``f^eq`` is replaced, so the non-equilibrium (stress-carrying) part of
    the distribution -- and with it the resolved shear at the wall node -- is
    preserved instead of being overwritten by a fresh equilibrium.
    """
    sl = slice(j, j + 1) if j >= 0 else slice(j, None)
    rho_row = rho[:, sl, :]
    feq_old = equilibrium3d(rho_row, ux[:, sl, :], uy[:, sl, :], uz[:, sl, :])
    zero = torch.zeros_like(rho_row)
    t_ux = torch.as_tensor(target_ux, dtype=rho.dtype, device=rho.device)
    t_uz = torch.as_tensor(target_uz, dtype=rho.dtype, device=rho.device)
    if t_ux.ndim == 0:
        t_ux = t_ux.expand_as(rho_row)
    else:
        t_ux = t_ux.reshape(rho_row.shape)
    if t_uz.ndim == 0:
        t_uz = t_uz.expand_as(rho_row)
    else:
        t_uz = t_uz.reshape(rho_row.shape)
    feq_new = equilibrium3d(rho_row, t_ux, zero, t_uz)
    f[:, :, sl, :] += feq_new - feq_old



def _stress_coupling(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    j: int,
    u_tau: torch.Tensor,
    nu: float,
    *,
    y_wall_node: float = 0.5,
    cell_height: float = 1.0,
) -> dict:
    """Impose the modelled wall shear stress as a momentum flux at row ``j``.

    Why this exists
    ---------------
    A halfway bounce-back wall is no-slip at ``y = 0``, so the numerical wall
    flux is ``tau_num = rho nu |u_j| / y_wall_node``.  Overwriting the
    wall-adjacent velocity with the wall-model value (the *assimilation*
    coupling) therefore acts as an *additional* momentum sink, and the stress
    that actually drags the flow is the assimilation flux rather than the
    modelled one: in the ``Re_tau = 180`` probe the wall model reported
    ``tau_w = 1.56e-4`` while the force balance demanded ``4.0e-4``
    (``u_tau_model / u_tau_balance = 0.62``), and the mismatch fed back into
    the exchange-node velocity.

    Instead, the *difference* between the modelled stress and the numerical
    one is injected as a per-step velocity increment

        ``du = (tau_num - rho u_tau^2) / (rho * cell_height)``

    (``dt = 1`` in lattice units).  The steady state of this feedback loop is
    ``tau_num = rho u_tau^2``, i.e. the modelled stress *is* the wall stress,
    and the channel force balance ``tau_w = a h`` then pins ``u_tau`` instead
    of leaving it floating.  The increment is applied as a *relative* scaling
    of the instantaneous row velocity by the plane-averaged magnitude, so the
    spanwise structure of ``u`` is preserved and the update stays regular when
    the local velocity passes through zero.
    """
    sl = slice(j, j + 1) if j >= 0 else slice(j, None)
    u_row = ux[:, sl, :]
    w_row = uz[:, sl, :]
    mag = torch.sqrt(u_row * u_row + w_row * w_row)
    mag_mean = float(mag.mean())
    tau_num = nu * mag_mean / y_wall_node
    tau_w = float(torch.as_tensor(u_tau, dtype=torch.float64).mean()) ** 2
    # The no-slip wall removes ``tau_num`` of streamwise momentum per step; the
    # model says it must remove ``tau_w``.  The difference is handed back to (or
    # taken from) the wall-adjacent row, so the net wall flux equals ``tau_w``.
    du = (tau_num - tau_w) / cell_height
    if mag_mean > 1.0e-12:
        scale = 1.0 + du / mag_mean
    else:
        scale = 1.0
    # never overshoot through zero: cap the relative change per step
    scale = min(max(scale, 0.0), 1.5)
    rho_row = rho[:, sl, :]
    uy_row = uy[:, sl, :]
    feq_old = equilibrium3d(rho_row, u_row, uy_row, w_row)
    feq_new = equilibrium3d(rho_row, u_row * scale, uy_row, w_row * scale)
    f[:, :, sl, :] += feq_new - feq_old
    return {"tau_w_model": tau_w, "tau_num": tau_num, "du": du}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_channel_wmles(config: ChannelWMLESConfig, *, output_dir: str | Path | None = None) -> dict:
    """Run a wall-modelled channel and return the statistics dictionary."""
    device = torch.device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    torch.manual_seed(int(config.seed))
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    if config.wall_model == "equilibrium":
        wm = EquilibriumWallModel(law=config.wall_law)
    else:
        wm = NonEquilibriumWallModel()

    # --- initial condition: laminar mean + solenoidal perturbation ---------
    u_mean = initial_mean_profile(config).to(dtype)
    ux0 = u_mean.view(1, -1, 1).expand(config.nz, config.ny, config.nx).clone()
    uy0 = torch.zeros_like(ux0)
    uz0 = torch.zeros_like(ux0)
    if config.perturbation_amplitude > 0.0:
        pert = solenoidal_perturbation(config, seed=config.seed).to(dtype)
        amp = config.perturbation_amplitude * float(u_mean.max())
        ux0 = ux0 + amp * pert[0]
        uy0 = uy0 + amp * pert[1]
        uz0 = uz0 + amp * pert[2]
    rho0 = torch.ones_like(ux0)
    f = equilibrium3d(rho0, ux0, uy0, uz0)

    # --- statistic accumulators -------------------------------------------
    n_y = config.ny
    acc = {
        "rho": torch.zeros(n_y, dtype=torch.float64),
        "ux": torch.zeros(n_y, dtype=torch.float64),
        "uy": torch.zeros(n_y, dtype=torch.float64),
        "uz": torch.zeros(n_y, dtype=torch.float64),
        "uxux": torch.zeros(n_y, dtype=torch.float64),
        "uyuy": torch.zeros(n_y, dtype=torch.float64),
        "uzuz": torch.zeros(n_y, dtype=torch.float64),
        "uxuy": torch.zeros(n_y, dtype=torch.float64),
        "uxuz": torch.zeros(n_y, dtype=torch.float64),
        "uyuz": torch.zeros(n_y, dtype=torch.float64),
        "uyuyuy": torch.zeros(n_y, dtype=torch.float64),
        "uxuyuy": torch.zeros(n_y, dtype=torch.float64),
        "pux": torch.zeros(n_y, dtype=torch.float64),
        "puy": torch.zeros(n_y, dtype=torch.float64),
        "diss_res": torch.zeros(n_y, dtype=torch.float64),
        "diss_sgs": torch.zeros(n_y, dtype=torch.float64),
        "prod_res": torch.zeros(n_y, dtype=torch.float64),
    }
    n_samples = 0
    u_tau_lower: list[float] = []
    u_tau_upper: list[float] = []
    history: list[dict] = []

    nu = config.nu
    tau = config.tau
    accel_field_shape = (config.nz, config.ny, config.nx)
    accel_x = torch.full(accel_field_shape, config.accel, dtype=dtype, device=device)
    zero_field = torch.zeros(accel_field_shape, dtype=dtype, device=device)
    # Non-equilibrium source: dp/dx = -rho a, plus zero advection (developed flow)
    source = -config.accel

    t0 = time.time()
    for step in range(config.steps):
        # Cumulant collision + Smagorinsky: stable at tau -> 1/2, which is
        # exactly the regime a high-Re WMLES reaches (molecular viscosity is
        # negligible next to the SGS viscosity).  BGK with tau-1/2 ~ 1e-3 was
        # tried first and blew up in the channel core within 60 steps.
        fpost = collide_cumulant_d3q19(f, tau, C_s=config.smagorinsky_cs)
        if config.positivity_limiter:
            fpost, _diag = limit_nonequilibrium_for_positivity(fpost)
        rho, ux, uy, uz = macroscopic3d(fpost)
        fpost = guo_body_force_d3q19(fpost, accel_x, zero_field, zero_field, ux, uy, uz)
        f_new = stream3d(fpost)

        f_new = _bounce_back_y(f_new, fpost)
        rho2, ux2, uy2, uz2 = macroscopic3d(f_new)
        # --- wall model on both walls -------------------------------------
        ux_lo, uz_lo, ut_lo, _ = _wall_model_velocity(
            config=config,
            wm=wm,
            ux_ex=ux2[:, config.j_exchange, :],
            uz_ex=uz2[:, config.j_exchange, :],
            wall="lower",
            source=source,
        )
        ux_hi, uz_hi, ut_hi, _ = _wall_model_velocity(
            config=config,
            wm=wm,
            ux_ex=ux2[:, -1 - config.j_exchange, :],
            uz_ex=uz2[:, -1 - config.j_exchange, :],
            wall="upper",
            source=source,
        )
        if config.wall_coupling == "none":
            pass  # pure halfway bounce-back: the no-wall-model control case
        elif config.wall_coupling == "stress":
            _stress_coupling(f_new, rho2, ux2, uy2, uz2, 0, ut_lo, nu)
            _stress_coupling(f_new, rho2, ux2, uy2, uz2, -1, ut_hi, nu)
        else:
            # Optional under-relaxation of the wall-model target: the modelled
            # velocity is blended with the current wall-node velocity instead of
            # being imposed in one step.  A stiff exchange-point/wall-model loop
            # at high Re_tau overshoots and reverses the mean flow (see
            # benchmarks/wall_model_les/re590_stability.py); blending sets the
            # loop gain below the oscillation threshold without changing the
            # fixed point of the coupled system.
            beta = config.wall_relaxation
            if beta < 1.0:
                ux_lo, uz_lo = _blend_target(ux2[:, 0, :], uz2[:, 0, :], ux_lo, uz_lo, beta)
                ux_hi, uz_hi = _blend_target(ux2[:, -1, :], uz2[:, -1, :], ux_hi, uz_hi, beta)
            _assimilate(f_new, rho2, ux2, uy2, uz2, 0, ux_lo, uz_lo)
            _assimilate(f_new, rho2, ux2, uy2, uz2, -1, ux_hi, uz_hi)
        f = f_new

        if step >= config.averaging_start and (step - config.averaging_start) % config.stat_interval == 0:
            _accumulate(
                acc,
                f=f,
                nu=nu,
                cs=config.smagorinsky_cs,
                config=config,
            )
            n_samples += 1
            u_tau_lower.append(float(ut_lo))
            u_tau_upper.append(float(ut_hi))

        if config.log_interval and (step + 1) % config.log_interval == 0:
            rho_s, uxs, uys, uzs = macroscopic3d(f)
            ub = float(uxs[:, 1:-1, :].mean())
            entry = {
                "step": step + 1,
                "u_bulk": ub,
                "re_tau_measured": float(ut_lo) * config.h / nu,
                "u_tau_lower": float(ut_lo),
                "u_tau_upper": float(ut_hi),
                "elapsed_s": round(time.time() - t0, 2),
            }
            if config.diagnostics:
                entry.update(stability_diagnostics(f, nu=nu, cs=config.smagorinsky_cs))
            history.append(entry)
            print(
                f"[{config.run_name}] step={step + 1} U_b={ub:.5f} "
                f"Re_tau={entry['re_tau_measured']:.1f} t={entry['elapsed_s']:.1f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    stats = _finalise(acc, n_samples, config)
    stats.update(
        {
            "config": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in config.__dict__.items()
            },
            "wall_model": config.wall_model,
            "wall_law": config.wall_law,
            "nu": nu,
            "tau": tau,
            "accel": config.accel,
            "H": config.H,
            "h": config.h,
            "delta_y_plus": config.delta_y_plus,
            "wall_origin_offset": config.wall_origin_offset,
            "H_effective": config.H_effective,
            "y_plus_first_node": config.y_of_node(config.j_wall_node) * config.u_tau / nu,
            "y_plus_exchange": config.y_of_node(config.j_exchange) * config.u_tau / nu,
            "j_wall_node": config.j_wall_node,
            "j_exchange": config.j_exchange,
            "tau_w_balance_effective": config.accel * config.H_effective / 2.0,
            "u_tau_model_mean": float(torch.tensor(u_tau_lower).mean()) if u_tau_lower else float("nan"),
            "u_tau_model_upper_mean": float(torch.tensor(u_tau_upper).mean()) if u_tau_upper else float("nan"),
            "tau_w_model": float(torch.tensor(u_tau_lower).mean() ** 2) if u_tau_lower else float("nan"),
            "tau_w_model_upper": float(torch.tensor(u_tau_upper).mean() ** 2) if u_tau_upper else float("nan"),
            "tau_w_balance": config.accel * config.h,
            "wall_coupling": config.wall_coupling,
            "u_tau_model_over_balance": (
                float(torch.tensor(u_tau_lower).mean()) / config.u_tau if u_tau_lower else float("nan")
            ),
            "n_samples": n_samples,
            "elapsed_s": round(elapsed, 2),
            "history": history,
        }
    )
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {k: v for k, v in stats.items() if isinstance(v, torch.Tensor)},
            out / "profiles.pt",
        )
        (out / "summary.json").write_text(
            json.dumps(
                {k: v for k, v in stats.items() if k != "profiles"},
                indent=2,
                default=str,
            )
        )
    return stats


def stability_diagnostics(
    f: torch.Tensor, *, nu: float, cs: float
) -> dict:
    """Cell-level stability margins of a WMLES state (used by log entries).

    The cumulant collision is driven by ``tau - 1/2 = 3 (nu + nu_t)``; a
    vanishing margin is the classical blow-up route of a high-``Re`` lattice
    run.  ``nu_t`` is the Smagorinsky viscosity with ``Delta = 1`` and the same
    strain invariant as the accumulator (:func:`_strain_sq`), so the numbers
    reported here are the numbers the solver actually sees.

    Returns
    -------
    dict with ``max_speed`` (lattice Mach proxy), ``rho_min``/``rho_max``,
    ``nu_sgs_min``/``mean``/``max``, ``tau_minus_half_min`` (smallest total
    relaxation margin in the box) and ``sgs_ratio_mean = nu_t.mean()/nu``.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    grads = _velocity_gradients(ux, uy, uz)
    s_sq = _strain_sq(grads)
    nu_sgs = cs * cs * torch.sqrt(2.0 * s_sq.clamp_min(0.0))
    speed = torch.sqrt(ux * ux + uy * uy + uz * uz)
    nu_tot = nu + nu_sgs
    return {
        "max_speed": float(speed.max()),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "nu_sgs_min": float(nu_sgs.min()),
        "nu_sgs_mean": float(nu_sgs.mean()),
        "nu_sgs_max": float(nu_sgs.max()),
        "tau_minus_half_min": 3.0 * float(nu_tot.min()),
        "tau_minus_half_mean": 3.0 * float(nu_tot.mean()),
        "sgs_ratio_mean": float(nu_sgs.mean()) / nu if nu > 0.0 else float("nan"),
    }


def _accumulate(acc: dict, *, f: torch.Tensor, nu: float, cs: float, config: ChannelWMLESConfig) -> None:
    """Accumulate plane means and dissipation at one sample."""
    rho, ux, uy, uz = macroscopic3d(f)
    p = rho * _CS2
    # fluctuations about the *instantaneous* plane mean (Reynolds decomposition)
    ux_f = ux - ux.mean(dim=(0, 2), keepdim=True)
    uy_f = uy - uy.mean(dim=(0, 2), keepdim=True)
    uz_f = uz - uz.mean(dim=(0, 2), keepdim=True)
    p_f = p - p.mean(dim=(0, 2), keepdim=True)

    def plane(a: torch.Tensor) -> torch.Tensor:
        return a.mean(dim=(0, 2)).double().cpu()

    acc["rho"] += plane(rho)
    acc["ux"] += plane(ux)
    acc["uy"] += plane(uy)
    acc["uz"] += plane(uz)
    acc["uxux"] += plane(ux_f * ux_f)
    acc["uyuy"] += plane(uy_f * uy_f)
    acc["uzuz"] += plane(uz_f * uz_f)
    acc["uxuy"] += plane(ux_f * uy_f)
    acc["uxuz"] += plane(ux_f * uz_f)
    acc["uyuz"] += plane(uy_f * uz_f)
    acc["uyuyuy"] += plane(uy_f * uy_f * uy_f)
    acc["uxuyuy"] += plane(ux_f * uy_f * uy_f)
    acc["pux"] += plane(p_f * ux_f)
    acc["puy"] += plane(p_f * uy_f)

    grads = _velocity_gradients(ux, uy, uz)
    s_sq = _strain_sq(grads)
    nu_sgs = cs * cs * torch.sqrt(2.0 * s_sq.clamp_min(0.0))
    acc["diss_res"] += plane(2.0 * nu * s_sq)
    acc["diss_sgs"] += plane(2.0 * nu_sgs * s_sq)
    acc["prod_res"] += plane(-ux_f * uy_f * grads[1])


def _finalise(acc: dict, n_samples: int, config: ChannelWMLESConfig) -> dict:
    """Convert accumulators to time-averaged profiles in wall units."""
    n = max(n_samples, 1)
    mean = {k: (v / n) for k, v in acc.items()}
    u_tau = config.u_tau
    nu = config.nu
    y = torch.arange(config.ny, dtype=torch.float64) + 0.5 - config.wall_origin_offset
    profiles = {
        "delta_y_plus": config.delta_y_plus,
        "y": y,
        "y_plus": y * u_tau / nu,
        "u_mean": mean["ux"],
        "u_plus": mean["ux"] / u_tau,
        "rho_mean": mean["rho"],
        "uu": mean["uxux"],
        "vv": mean["uyuy"],
        "ww": mean["uzuz"],
        "uv": mean["uxuy"],
        "uw": mean["uxuz"],
        "vw": mean["uyuz"],
        "vvv": mean["uyuyuy"],
        "uvv": mean["uxuyuy"],
        "pu": mean["pux"],
        "pv": mean["puy"],
        "diss_res": mean["diss_res"],
        "diss_sgs": mean["diss_sgs"],
        "prod_res": mean["prod_res"],
    }
    u_bulk = float(profiles["u_mean"][1:-1].mean())
    tau_w = u_tau**2
    profiles["u_bulk"] = u_bulk
    profiles["re_bulk"] = u_bulk * config.H / nu
    profiles["c_f"] = 2.0 * tau_w / u_bulk**2 if u_bulk > 0 else float("nan")
    profiles["tau_w_from_bulk"] = tau_w
    # Scalars needed to re-derive wall units downstream (peak diagnostics):
    # the profile arrays are raw lattice units, so the normalisers travel with
    # them instead of being re-guessed from the run configuration.
    profiles["nu"] = float(nu)
    profiles["u_tau"] = float(u_tau)
    profiles["re_tau_nominal"] = float(config.re_tau)
    return {"profiles": profiles}