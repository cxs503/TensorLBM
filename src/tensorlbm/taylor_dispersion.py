"""Taylor–Aris shear dispersion: analytic references + lattice-Boltzmann verification.

Background
----------
A passive scalar advected by a *shear* flow spreads along the flow direction much
faster than molecular diffusion alone.  For long waves the cross-section averaged
concentration obeys a one-dimensional advection–diffusion equation whose
*effective* diffusivity ``D_eff`` is (Taylor 1953, Aris 1956, Brenner 1980)::

    D_eff = D + <psi^2> / D ,      psi(y) = int^y (u - u_mean) dy' - <psi>

where ``<>`` is the cross-section average and ``psi`` is the primitive of the
velocity defect with zero mean.  Three closed forms are used here as external
references:

======  ==================================  =======================================
flow    profile                             D_eff
======  ==================================  =======================================
pipe    u = 2u_m (1 - r^2/a^2)              D + a^2 u_m^2 / (48 D)
channel u = 3/2 u_m (1 - y^2/h^2)           D + 2/105 * u_m^2 h^2 / D
periodic u = U sin(2 pi y / H) (u_m = 0)    D + U^2 H^2 / (8 pi^2 D)
======  ==================================  =======================================

The pipe/channel forms need a no-flux wall condition.  The periodic-shear form
needs none — the transverse cell is periodic — so it isolates the *shipped*
D2Q5 advection–diffusion kernel of :mod:`tensorlbm.passive_scalar` without any
boundary-condition confound.  That is the case measured numerically here.

Verification protocol (modal decay)
-----------------------------------
Initialise ``phi = 1 + eps cos(k x)`` (uniform across the shear direction).  The
transverse modes decay at ``O(D (2 pi / H)^2)``, the axial Taylor mode at
``k^2 D_eff``; for ``k`` small enough the two scales separate and the amplitude
of the cross-section averaged axial mode decays as ``exp(-k^2 D_eff t)``.
Fitting ``log amp`` over a late window yields ``D_eff`` without any fitting of
transverse structure, and the cross-section average projects out the transverse
modes exactly (they have zero mean), so the measurement is self-cleaning.

References
----------
Taylor, G. I. (1953). Dispersion of soluble matter in solvent flowing slowly
    through a tube. *Proc. R. Soc. Lond. A* 219, 186–203.
Aris, R. (1956). On the dispersion of a solute in a fluid flowing through a
    tube. *Proc. R. Soc. Lond. A* 235, 67–77.
Brenner, H. (1980). Dispersion resulting from flow through spatially periodic
    porous media. *Phil. Trans. R. Soc. Lond. A* 297, 81–133.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = [
    "ShearDispersionConfig",
    "channel_taylor_diffusivity",
    "pipe_taylor_diffusivity",
    "periodic_shear_taylor_diffusivity",
    "taylor_diffusivity_from_profile",
    "run_periodic_shear_dispersion",
    "run_uniform_flow_control",
]


# ---------------------------------------------------------------------------
# Analytic references
# ---------------------------------------------------------------------------


def channel_taylor_diffusivity(D: float, u_mean: float, half_gap: float) -> float:
    """Plane-Poiseuille Taylor–Aris diffusivity ``D + 2/105 * u_mean^2 h^2 / D``."""
    _check_positive(D=D, half_gap=half_gap)
    return D + (2.0 / 105.0) * u_mean**2 * half_gap**2 / D


def pipe_taylor_diffusivity(D: float, u_mean: float, radius: float) -> float:
    """Circular-pipe Taylor–Aris diffusivity ``D + a^2 u_mean^2 / (48 D)``."""
    _check_positive(D=D, radius=radius)
    return D + radius**2 * u_mean**2 / (48.0 * D)


def periodic_shear_taylor_diffusivity(D: float, u_amp: float, period: float) -> float:
    """Transversely periodic shear ``u = U sin(2 pi y / H)``: ``D + U^2 H^2/(8 pi^2 D)``."""
    _check_positive(D=D, period=period)
    return D + u_amp**2 * period**2 / (8.0 * math.pi**2 * D)


def _check_positive(**kwargs: float) -> None:
    for name, value in kwargs.items():
        if not math.isfinite(value) or value <= 0.0:
            msg = f"{name} must be positive and finite, got {value!r}"
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# General (numerical) Aris integral
# ---------------------------------------------------------------------------


def taylor_diffusivity_from_profile(
    D: float,
    u: torch.Tensor | list[float],
    *,
    dy: float = 1.0,
    periodic: bool = False,
) -> float:
    """``D_eff`` from the general Taylor–Aris integral for an arbitrary profile.

    ``D_eff = D + <psi^2> / D`` with ``psi`` the zero-mean primitive of the
    velocity defect.  The primitive is integrated with the trapezoidal rule on
    the *cell* grid: for a periodic cell the defect integrates to zero over the
    cell (checked), for a wall-bounded section it does not and the zero-mean
    shift of ``psi`` is essential.

    Args:
        D: molecular diffusivity (> 0).
        u: velocity samples on a uniform cross-section grid (length >= 3).
        dy: grid spacing in lattice units.
        periodic: if True the cell is periodic and ``sum(u - u_mean)`` must
            vanish to integration accuracy; if False the primitive is
            normalised with ``psi = integral from the first sample``.

    Returns:
        Effective diffusivity in lattice units (``D`` has units of lu^2/ts).
    """
    _check_positive(D=D, dy=dy)
    uu = torch.as_tensor(u, dtype=torch.float64).flatten()
    if uu.numel() < 3:
        msg = f"need at least 3 cross-section samples, got {uu.numel()}"
        raise ValueError(msg)
    u_mean = float(uu.mean())
    defect = uu - u_mean
    if periodic:
        # periodic cell: the defect must be a pure derivative; report the
        # quadrature residual so callers can see whether the grid is adequate.
        resid = float(defect.sum()) / max(float(uu.abs().sum()), 1e-300)
        if abs(resid) > 1e-3:
            msg = f"periodic profile defect integrates to {resid:.3e} (not a periodic cell)"
            raise ValueError(msg)
        # trapezoid on a periodic grid: half-weight the endpoints' wrap-around
        d = defect.clone()
        psi = torch.zeros_like(uu)
        # psi_j = integral from 0 to j ; endpoints wrap for the periodic case
        steps = 0.5 * (d[:-1] + d[1:]) * dy
        psi[1:] = torch.cumsum(steps, dim=0)
        psi = psi - psi.mean()
    else:
        psi = torch.zeros_like(uu)
        psi[1:] = torch.cumsum(0.5 * (defect[:-1] + defect[1:]) * dy, dim=0)
    # <psi^2> by trapezoid (uniform weights; endpoints half-weighted per unit length)
    sq = psi**2
    if not periodic:
        w = torch.ones_like(sq)
        w[0] = w[-1] = 0.5
        return D + float((w * sq).sum() / w.sum()) / D
    return D + float(sq.mean()) / D


# ---------------------------------------------------------------------------
# LBM measurement of the dispersion coefficient
# ---------------------------------------------------------------------------


@dataclass
class ShearDispersionConfig:
    """Configuration for the periodic-shear Taylor dispersion measurement."""

    ny: int = 16  # transverse period H (lattice units)
    nx: int = 128  # axial period L
    D: float = 0.1  # molecular diffusivity (lattice units)
    pe: float = 8.0  # transverse Péclet number Pe = U H / D
    amplitude: float = 1e-3  # initial axial mode amplitude (linear response)
    n_steps: int = 6000
    measure_every: int = 10
    fit_start_fraction: float = 0.5  # fit the second half of the record
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.ny < 4:
            msg = "ny must be >= 4 to resolve the shear period"
            raise ValueError(msg)
        if self.nx < 8:
            msg = "nx must be >= 8 to resolve the axial mode"
            raise ValueError(msg)
        _check_positive(D=self.D, pe=self.pe)
        if self.ny >= self.nx:
            msg = "transverse period must be shorter than the axial period (scale separation)"
            raise ValueError(msg)

    @property
    def u_amp(self) -> float:
        """Shear amplitude U = Pe * D / H."""
        return self.pe * self.D / self.ny

    @property
    def tau_d(self) -> float:
        """D2Q5 relaxation time, ``D = (tau - 1/2)/3``."""
        return 3.0 * self.D + 0.5

    @property
    def k(self) -> float:
        return 2.0 * math.pi / self.nx

    @property
    def d_eff_theory(self) -> float:
        return periodic_shear_taylor_diffusivity(self.D, self.u_amp, self.ny)


@dataclass
class ShearDispersionResult:
    """Outcome of one dispersion measurement."""

    pe: float
    k: float
    rate_fit: float
    d_eff_numeric: float
    d_eff_theory: float
    rel_error: float
    transverse_rate: float
    axial_rate_theory: float
    times: list[float] = field(default_factory=list)
    amplitudes: list[float] = field(default_factory=list)
    fit_start_time: float = 0.0

    @property
    def ok(self) -> bool:
        return abs(self.d_eff_numeric / self.d_eff_theory - 1.0) < 1e-12


def _amplitude(phi: torch.Tensor, k: float) -> float:
    """Amplitude of the cos(k x) axial mode of the cross-section average."""
    nx = phi.shape[1]
    x = torch.arange(nx, dtype=phi.dtype, device=phi.device)
    mean_x = phi.mean(dim=0)
    cos_c = (mean_x * torch.cos(k * x)).sum() * (2.0 / nx)
    sin_c = (mean_x * torch.sin(k * x)).sum() * (2.0 / nx)
    return float(torch.hypot(cos_c, sin_c))


def run_periodic_shear_dispersion(
    cfg: ShearDispersionConfig | None = None,
    **kwargs: object,
) -> ShearDispersionResult:
    """Measure ``D_eff`` of the shipped D2Q5 scalar kernel in periodic shear.

    The velocity field ``u_x(y) = U sin(2 pi y / H)`` is imposed externally (the
    scalar is passive, so the momentum solver is irrelevant and is not run).
    Only fully periodic shipped operators are used — no boundary condition.
    """
    from .passive_scalar import (  # noqa: PLC0415
        collide_scalar_bgk,
        equilibrium_scalar,
        macroscopic_scalar,
        stream_scalar,
    )

    if cfg is None:
        fields = set(ShearDispersionConfig.__dataclass_fields__)
        cfg = ShearDispersionConfig(**{k: v for k, v in kwargs.items() if k in fields})
    dtype = getattr(torch, cfg.dtype)
    ny, nx = cfg.ny, cfg.nx
    k = cfg.k
    y = torch.arange(ny, dtype=dtype)
    u_profile = cfg.u_amp * torch.sin(2.0 * math.pi * y / ny)
    ux = u_profile.view(ny, 1).repeat(1, nx).contiguous()
    uy = torch.zeros_like(ux)

    x = torch.arange(nx, dtype=dtype)
    phi0 = 1.0 + cfg.amplitude * torch.cos(k * x).view(1, nx).repeat(ny, 1)
    g = equilibrium_scalar(phi0, ux, uy)

    times: list[float] = []
    amps: list[float] = []
    for step in range(cfg.n_steps):
        if step % cfg.measure_every == 0:
            times.append(float(step))
            amps.append(_amplitude(macroscopic_scalar(g), k))
        phi = macroscopic_scalar(g)
        g = collide_scalar_bgk(g, phi, ux, uy, cfg.tau_d)
        g = stream_scalar(g)

    t = torch.tensor(times, dtype=torch.float64)
    a = torch.tensor(amps, dtype=torch.float64)
    start = t[0] + cfg.fit_start_fraction * (t[-1] - t[0])
    mask = t >= start
    tt, ll = t[mask], torch.log(a[mask])
    # least-squares slope of log(amplitude) vs time
    tm, lm = tt.mean(), ll.mean()
    slope = float(((tt - tm) * (ll - lm)).sum() / ((tt - tm) ** 2).sum())
    rate = -slope
    d_num = rate / k**2
    d_th = cfg.d_eff_theory
    return ShearDispersionResult(
        pe=cfg.pe,
        k=k,
        rate_fit=rate,
        d_eff_numeric=d_num,
        d_eff_theory=d_th,
        rel_error=d_num / d_th - 1.0,
        transverse_rate=cfg.D * (2.0 * math.pi / cfg.ny) ** 2,
        axial_rate_theory=k**2 * d_th,
        times=times,
        amplitudes=amps,
        fit_start_time=float(start),
    )


def run_uniform_flow_control(
    cfg: ShearDispersionConfig | None = None,
    **kwargs: object,
) -> ShearDispersionResult:
    """Negative control: *uniform* advection has no shear, so ``D_eff = D``.

    Any shear-induced enhancement leaking into the uniform case (e.g. from a
    profile sign error or a mis-indexed transverse average) would show up here.
    """
    from .passive_scalar import (  # noqa: PLC0415
        collide_scalar_bgk,
        equilibrium_scalar,
        macroscopic_scalar,
        stream_scalar,
    )

    if cfg is None:
        fields = set(ShearDispersionConfig.__dataclass_fields__)
        cfg = ShearDispersionConfig(**{k: v for k, v in kwargs.items() if k in fields})
    dtype = getattr(torch, cfg.dtype)
    ny, nx = cfg.ny, cfg.nx
    k = cfg.k
    ux = torch.full((ny, nx), cfg.u_amp, dtype=dtype)
    uy = torch.zeros(ny, nx, dtype=dtype)
    x = torch.arange(nx, dtype=dtype)
    phi0 = 1.0 + cfg.amplitude * torch.cos(k * x).view(1, nx).repeat(ny, 1)
    g = equilibrium_scalar(phi0, ux, uy)
    times, amps = [], []
    for step in range(cfg.n_steps):
        if step % cfg.measure_every == 0:
            times.append(float(step))
            amps.append(_amplitude(macroscopic_scalar(g), k))
        phi = macroscopic_scalar(g)
        g = collide_scalar_bgk(g, phi, ux, uy, cfg.tau_d)
        g = stream_scalar(g)
    t = torch.tensor(times, dtype=torch.float64)
    a = torch.tensor(amps, dtype=torch.float64)
    start = t[0] + cfg.fit_start_fraction * (t[-1] - t[0])
    mask = t >= start
    tt, ll = t[mask], torch.log(a[mask])
    tm, lm = tt.mean(), ll.mean()
    slope = float(((tt - tm) * (ll - lm)).sum() / ((tt - tm) ** 2).sum())
    d_num = -slope / k**2
    return ShearDispersionResult(
        pe=cfg.pe,
        k=k,
        rate_fit=-slope,
        d_eff_numeric=d_num,
        d_eff_theory=cfg.D,
        rel_error=d_num / cfg.D - 1.0,
        transverse_rate=cfg.D * (2.0 * math.pi / cfg.ny) ** 2,
        axial_rate_theory=k**2 * cfg.D,
        times=times,
        amplitudes=amps,
        fit_start_time=float(start),
    )


def main() -> None:  # pragma: no cover - CLI convenience
    import json

    cfg = ShearDispersionConfig()
    res = run_periodic_shear_dispersion(cfg)
    ctl = run_uniform_flow_control(cfg)
    print(
        json.dumps(
            {
                "pe": res.pe,
                "d_eff_numeric": res.d_eff_numeric,
                "d_eff_theory": res.d_eff_theory,
                "rel_error": res.rel_error,
                "uniform_control_rel_error": ctl.rel_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()