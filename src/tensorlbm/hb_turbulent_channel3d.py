"""Herschel-Bulkley (yield-stress) turbulent channel flow, 3D LBM.

Campaign node **P295W1R2-N04** (successor of P295W1-N04, which only delivered the
analytic framework).  This module supplies the *numerical* half of the node:

* a wall-bounded 3D lattice-Boltzmann channel driven by a constant body force,
  with a **spatially varying relaxation time** obtained from the
  Papanastasiou-regularised Herschel-Bulkley (HB) constitutive law

      tau      = tau_y + K * gd^n                       (|tau| > tau_y)
      gd       = 0                                      (|tau| <= tau_y)

      nu_eff(gd) = K * gd^(n-1) + tau_y * (1 - exp(-m*gd)) / (rho * gd)

  (Papanastasiou 1987, J. Rheol. 31, 385) plus an optional Smagorinsky
  sub-grid eddy viscosity,

      nu_t = (Cs * Delta)^2 * |S|,   |S| = sqrt(2 S_ij S_ij),  Delta = 1 lu.

* mean-flow / Reynolds-stress / TKE statistics reduced to wall-normal profiles,
* the three dimensionless groups the campaign is judged on:

      Metzner-Reed generalised Reynolds number
          Re_MR = rho * U_b^(2-n) * D_h^n / (K * 8^(n-1) * ((3n+1)/(4n))^n)
      yield number
          Y = tau_y / (rho * u_tau^2)
      Darcy friction factor (hydraulic diameter of a plane channel D_h = 4h)
          f = 8 * u_tau^2 / U_b^2

Physics of the driving mode
---------------------------
The flow is driven by a constant body-force acceleration ``G``; global momentum
balance then fixes the wall shear stress *by construction*,

      tau_w = rho * u_tau^2 = G * h  (per unit width, h = half channel height)

independently of the rheology.  Consequently ``u_tau`` (hence ``Re_tau``) is a
*chosen* scale and ``Y`` is a well-posed independent parameter, while the bulk
velocity ``U_b`` (hence ``Re_MR`` and ``f``) responds to ``n`` and ``Y``.

A corollary that matters for the campaign's acceptance criteria: because the
imposed stress decreases linearly to zero at the centre line, the material can
never flow when ``tau_y >= tau_w``, i.e. ``Y -> 1`` is the *hard* shut-off of
this driving mode.  Turbulent drag/yield physics therefore lives in
``0 < Y < 1`` and any "critical yield number" above unity is unphysical for a
pressure-driven channel (see ``reports`` in the campaign benchmark).

References
----------
* Papanastasiou, T.C. (1987) J. Rheol. 31, 385.
* Metzner, A.B. & Reed, J.C. (1955) AIChE J. 1, 434.
* Dodge, D.W. & Metzner, A.B. (1959) Ind. Eng. Chem. 51, 1161.
* Smagorinsky, J. (1963) Mon. Weather Rev. 91, 99.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .d3q19 import equilibrium3d, macroscopic3d_low_memory
from .solver3d import stream3d

__all__ = [
    "HBTurbulentChannelConfig",
    "HBTurbulentChannel",
    "papanastasiou_viscosity",
    "strain_rate_magnitude",
    "metzner_reed_reynolds",
    "darcy_friction_factor",
    "dodge_metzner_friction",
    "consistency_for_target_remr",
    "run_hb_turbulent_channel",
]


# ---------------------------------------------------------------------------
# constitutive / dimensionless helpers
# ---------------------------------------------------------------------------
def papanastasiou_viscosity(
    gamma_dot: torch.Tensor,
    *,
    tau_y: float,
    K: float,
    n: float,
    rho: float = 1.0,
    m_pap: float = 500.0,
    nu_min: float = 1.0e-4,
    nu_max: float = 0.25,
) -> torch.Tensor:
    """Papanastasiou-regularised HB kinematic viscosity ``nu_eff(gd)``.

    ``nu_eff = K gd^(n-1) + tau_y (1 - exp(-m gd)) / (rho gd)``.

    The regularisation makes the viscosity finite at ``gd = 0`` (where it
    approaches ``K*0 + tau_y*m/rho``) so that the relaxation time stays bounded.
    The result is clipped to ``[nu_min, nu_max]`` for lattice stability.
    """
    if tau_y < 0.0:
        raise ValueError("tau_y must be >= 0")
    if K <= 0.0:
        raise ValueError("K must be > 0")
    if n <= 0.0:
        raise ValueError("n must be > 0")
    gd = torch.clamp(gamma_dot, min=1.0e-10)
    shear_thinning = K * torch.pow(gd, n - 1.0) if n != 1.0 else K * torch.ones_like(gd)
    if tau_y > 0.0:
        plug = tau_y * (1.0 - torch.exp(-m_pap * gd)) / (rho * gd)
        nu = shear_thinning + plug
    else:
        nu = shear_thinning
    return torch.clamp(nu, min=nu_min, max=nu_max)


def _grad(a: torch.Tensor, dim: int, periodic: bool) -> torch.Tensor:
    """Central difference along ``dim`` of a ``(nz, ny, nx)`` field.

    ``periodic`` selects wrap-around differences; otherwise one-sided
    differences are used on the two boundary planes of that axis.
    """
    if periodic:
        return 0.5 * (torch.roll(a, -1, dims=dim) - torch.roll(a, 1, dims=dim))
    g = torch.empty_like(a)
    lo = a.narrow(dim, 2, a.shape[dim] - 2) - a.narrow(dim, 0, a.shape[dim] - 2)
    g.narrow(dim, 1, a.shape[dim] - 2).copy_(0.5 * lo)
    g.narrow(dim, 0, 1).copy_(a.narrow(dim, 1, 1) - a.narrow(dim, 0, 1))
    g.narrow(dim, a.shape[dim] - 1, 1).copy_(
        a.narrow(dim, a.shape[dim] - 1, 1) - a.narrow(dim, a.shape[dim] - 2, 1)
    )
    return g


def strain_rate_magnitude(ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor) -> torch.Tensor:
    """Shear-rate magnitude ``gd = sqrt(2 S_ij S_ij)`` from central differences.

    ``ux, uy, uz`` have shape ``(nz, ny, nx)``; ``x`` (dim 2) and ``z`` (dim 0)
    are periodic, ``y`` (dim 1) is wall bounded.
    """
    Sxx = _grad(ux, 2, True)
    Syy = _grad(uy, 1, False)
    Szz = _grad(uz, 0, True)
    Sxy = 0.5 * (_grad(ux, 1, False) + _grad(uy, 2, True))
    Sxz = 0.5 * (_grad(ux, 0, True) + _grad(uz, 2, True))
    Syz = 0.5 * (_grad(uy, 0, True) + _grad(uz, 1, False))
    return torch.sqrt(
        torch.clamp(
            2.0 * (Sxx * Sxx + Syy * Syy + Szz * Szz) + 4.0 * (Sxy * Sxy + Sxz * Sxz + Syz * Syz),
            min=0.0,
        )
    )


def metzner_reed_reynolds(
    u_bulk: float,
    D_h: float,
    K: float,
    n: float,
    rho: float = 1.0,
) -> float:
    """Metzner-Reed generalised Reynolds number (power-law group)."""
    if u_bulk <= 0.0 or K <= 0.0:
        return 0.0
    return (
        rho
        * u_bulk ** (2.0 - n)
        * D_h**n
        / (K * 8.0 ** (n - 1.0) * ((3.0 * n + 1.0) / (4.0 * n)) ** n)
    )


def consistency_for_target_remr(
    re_mr_target: float,
    u_bulk: float,
    D_h: float,
    n: float,
    rho: float = 1.0,
) -> float:
    """Consistency index ``K`` that puts a power-law fluid at ``Re_MR_target``.

    Inverts :func:`metzner_reed_reynolds`.  Used to *match* ``Re_MR`` across
    flow indices so that the friction-factor collapse can be tested at equal
    generalised Reynolds number instead of at equal ``K``.
    """
    if re_mr_target <= 0.0 or u_bulk <= 0.0:
        raise ValueError("re_mr_target and u_bulk must be positive")
    return (
        rho
        * u_bulk ** (2.0 - n)
        * D_h**n
        / (re_mr_target * 8.0 ** (n - 1.0) * ((3.0 * n + 1.0) / (4.0 * n)) ** n)
    )


def darcy_friction_factor(u_tau: float, u_bulk: float) -> float:
    """Darcy friction factor on the hydraulic diameter of a plane channel."""
    if u_bulk <= 0.0:
        return float("inf")
    return 8.0 * u_tau * u_tau / (u_bulk * u_bulk)


def dodge_metzner_friction(re_mr: float, n: float) -> float:
    """Dodge-Metzner (1959) turbulent friction-factor correlation (slit form).

    ``1/sqrt(f) = (4/n^0.75) log10(Re_MR f^(1-n/2)) - 0.4/n^1.2``

    solved by fixed-point iteration; returns ``nan`` if it does not converge
    (outside the correlation's validity).
    """
    if re_mr <= 0.0:
        return float("nan")
    inv_sqrt_f = 8.0 / math.sqrt(re_mr)  # Blasius-like initial guess
    for _ in range(200):
        f = 1.0 / (inv_sqrt_f * inv_sqrt_f)
        rhs = (4.0 / n**0.75) * math.log10(re_mr * f ** (1.0 - 0.5 * n)) - 0.4 / n**1.2
        if not math.isfinite(rhs) or rhs <= 0.0:
            return float("nan")
        new = 0.5 * (inv_sqrt_f + rhs)
        if abs(new - inv_sqrt_f) < 1e-12:
            inv_sqrt_f = new
            break
        inv_sqrt_f = new
    f = 1.0 / (inv_sqrt_f * inv_sqrt_f)
    return f if math.isfinite(f) and f > 0.0 else float("nan")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class HBTurbulentChannelConfig:
    """Configuration of a body-force-driven HB channel."""

    # grid (x streamwise, y wall normal, z spanwise)
    nx: int = 64
    ny: int = 48
    nz: int = 32

    # driving
    re_tau: float = 180.0
    u_tau: float = 0.006

    # rheology
    tau_y: float = 0.0
    K: float = 0.02
    n: float = 1.0
    rho: float = 1.0
    #: Papanastasiou regularisation parameter.  ``0`` selects the automatic
    #: value ``m_factor * nu_ref / tau_y`` (see :attr:`m_effective`), which is
    #: the choice that makes the unyielded plug actually rigid: the plug
    #: viscosity of the regularised law is ``tau_y * m / rho``, so it must be
    #: many times ``nu_ref`` to approximate the discontinuous HB law.
    m_pap: float = 0.0
    m_factor: float = 50.0

    # sub-grid model
    smagorinsky_cs: float = 0.1

    # numerics
    lattice: str = "d3q19"
    #: ``trt`` (two-relaxation-time, recommended) or ``bgk``.  BGK with a
    #: position-dependent relaxation time is unstable once tau - 1/2 ~ 1e-3,
    #: which is exactly the wall-resolved channel regime at lattice viscosities
    #: of ~1e-3 (documented trap of this campaign).
    collision: str = "trt"
    #: TRT magic parameter Lambda = (tau-1/2)(tau_- - 1/2).
    trt_lambda: float = 3.0 / 16.0
    #: source of the shear-rate magnitude: ``neq`` (non-equilibrium momentum
    #: flux, local and noise-free) or ``grad`` (central velocity differences).
    shear_rate_source: str = "neq"
    steps: int = 30000
    warmup_steps: int = 8000
    sample_interval: int = 10
    report_interval: int = 2000
    #: Recompute the HB viscosity field every N steps.  The turbulent time
    #: scale is O(100-1000) steps, so freezing nu over a few steps is a
    #: controlled approximation (verified by the interval-refinement check in
    #: the campaign's convergence table).
    nu_update_interval: int = 4
    nu_min: float = 1.0e-4
    nu_max: float = 0.2
    tau_min: float = 0.502
    seed: int = 20261110
    dtype: str = "float32"

    name: str = "case"
    output: Path = field(default=Path("results/hb_turb_channel/result.json"))

    def __post_init__(self) -> None:
        self.output = Path(self.output)
        self.validate()

    # -- derived geometry -------------------------------------------------
    @property
    def height(self) -> int:
        """Number of fluid rows (walls are the first and last node rows)."""
        return self.ny - 2

    @property
    def half_height(self) -> float:
        """Half channel height in lattice units (wall at half-way BB)."""
        return 0.5 * self.height

    @property
    def hydraulic_diameter(self) -> float:
        return 4.0 * self.half_height

    @property
    def body_force(self) -> float:
        """Acceleration giving the prescribed ``u_tau``.

        Global momentum balance of the plane channel (half height ``h``) is
        ``rho * u_tau^2 = rho * G * h``, hence ``G = u_tau^2 / h``.
        """
        return self.u_tau**2 / self.half_height

    @property
    def tau_wall(self) -> float:
        return self.rho * self.u_tau**2

    @property
    def yield_number(self) -> float:
        return self.tau_y / self.tau_wall

    @property
    def nu_newtonian(self) -> float:
        return self.u_tau * self.half_height / self.re_tau

    @property
    def nu_max_effective(self) -> float:
        """Viscosity ceiling that still lets the plug be rigid.

        The regularised plug viscosity is ``tau_y m / rho``; clipping it at a
        fixed ``nu_max`` (default 0.2) silently *mobilises* the plug and makes
        the yield stress invisible (measured: laminar Y=0.3 profile collapses
        back onto the Y=0 parabola).  The ceiling therefore always sits a
        factor 4 above the plug viscosity.
        """
        ceiling = float(self.nu_max)
        if self.tau_y > 0.0:
            ceiling = max(ceiling, 4.0 * self.tau_y * self.m_effective / self.rho)
        return ceiling

    @property
    def m_effective(self) -> float:
        """Papanastasiou parameter actually used.

        The regularised plug viscosity is ``tau_y * m / rho``; for the plug to
        behave as a rigid body it must exceed the reference viscosity
        ``nu_ref = u_tau h / Re_tau`` by the factor ``m_factor``.  A fixed
        ``m = 500`` (the value quoted in most LBM papers for O(1) viscosities)
        is *not* sufficient at lattice-scale viscosities of ~1e-3 and leaves
        the plug free to creep - a documented trap of this campaign.
        """
        if self.m_pap > 0.0:
            return float(self.m_pap)
        if self.tau_y <= 0.0:
            return 500.0
        # the viscosity *scale actually present in the run*: the Newtonian
        # reference is only a proxy - for a power-law fluid the consistency
        # index K sets the viscosity at unit shear rate and is usually larger.
        nu_ref = max(self.nu_newtonian, self.K, 1.0e-8)
        return min(1.0e6, max(200.0, self.m_factor * nu_ref / self.tau_y))

    def validate(self) -> None:
        if min(self.nx, self.nz) < 8 or self.ny < 10:
            raise ValueError("grid too small for a wall-resolved channel")
        if self.re_tau <= 0 or self.u_tau <= 0:
            raise ValueError("re_tau and u_tau must be positive")
        if self.tau_y < 0:
            raise ValueError("tau_y must be >= 0")
        if self.K <= 0:
            raise ValueError("K must be > 0")
        if not 0 < self.n <= 2:
            raise ValueError("flow index n must lie in (0, 2]")
        if self.m_pap < 0:
            raise ValueError("Papanastasiou m must be >= 0 (0 = automatic)")
        if self.lattice not in {"d3q19", "d3q27"}:
            raise ValueError("lattice must be d3q19 or d3q27")
        if self.collision not in {"bgk", "trt"}:
            raise ValueError("collision must be bgk or trt")
        if self.shear_rate_source not in {"neq", "grad"}:
            raise ValueError("shear_rate_source must be neq or grad")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("need 0 <= warmup_steps < steps")
        if self.nu_update_interval < 1:
            raise ValueError("nu_update_interval must be >= 1")
        if self.sample_interval < 1 or self.report_interval < 1:
            raise ValueError("sample/report intervals must be >= 1")


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
class HBTurbulentChannel:
    """D3Q19/D3Q27 BGK channel with position-dependent HB viscosity."""

    def __init__(self, cfg: HBTurbulentChannelConfig):
        self.cfg = cfg
        self.dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
        self.device = torch.device("cpu")

        if cfg.lattice == "d3q19":
            from .d3q19 import OPPOSITE as OQ
            from .d3q19 import C as Cq
            from .d3q19 import W as Wq

            self._equilibrium = equilibrium3d
            self._macroscopic = macroscopic3d_low_memory
        else:  # pragma: no cover - exercised by the D3Q27 comparison case
            from .d3q27 import OPPOSITE as OQ
            from .d3q27 import C as Cq
            from .d3q27 import W as Wq
            from .d3q27 import equilibrium27 as _eq27
            from .d3q27 import macroscopic27 as _mac27

            self._equilibrium = _eq27
            self._macroscopic = _mac27
        self.C = Cq.to(dtype=self.dtype)
        self.W = Wq.to(dtype=self.dtype)
        self.Q = int(self.C.shape[0])
        self.opp_flat = OQ.to(device=self.device, dtype=torch.long)
        self.cx = self.C[:, 0].view(self.Q, 1, 1, 1)
        self.cy = self.C[:, 1].view(self.Q, 1, 1, 1)
        self.cz = self.C[:, 2].view(self.Q, 1, 1, 1)
        self.w = self.W.view(self.Q, 1, 1, 1)

        self._coef = {
            "xx": (self.cx * self.cx),
            "yy": (self.cy * self.cy),
            "zz": (self.cz * self.cz),
            "xy": (self.cx * self.cy),
            "xz": (self.cx * self.cz),
            "yz": (self.cy * self.cz),
        }
        self.nx, self.ny, self.nz = cfg.nx, cfg.ny, cfg.nz
        self.solid = torch.zeros(self.nz, self.ny, self.nx, dtype=torch.bool)
        self.solid[:, 0, :] = True
        self.solid[:, -1, :] = True
        self.fluid = ~self.solid

        self._tau = None
        self._step_count = 0
        self._reset_statistics()
        self._initialise()

    # -- setup ------------------------------------------------------------
    def _reset_statistics(self) -> None:
        n = self.cfg.ny
        self._samples = 0
        self._u_sum = torch.zeros(n, dtype=self.dtype)
        self._uu_sum = torch.zeros(n, dtype=self.dtype)
        self._vv_sum = torch.zeros(n, dtype=self.dtype)
        self._ww_sum = torch.zeros(n, dtype=self.dtype)
        self._uv_sum = torch.zeros(n, dtype=self.dtype)
        self._ubulk_series: list[float] = []
        self._tke_series: list[float] = []

    def _initialise(self) -> None:
        """Spalding-law mean profile plus a finite-amplitude perturbation.

        A laminar parabola is a *bad* turbulent initial guess: at
        ``Re_tau = 180`` it gives ``u_max = u_tau Re_tau / 2 ~ 4.5 u_bulk``,
        so the flow would have to decelerate by a factor ~4 while the
        relaxation time sits next to 1/2 (the historically observed blow-up).
        The wall law starts within a few percent of the turbulent solution.
        """
        from .spalding_wall_model import spalding_u_plus_from_y_plus

        cfg = self.cfg
        h = cfg.half_height
        ny, nz, nx = cfg.ny, self.nz, self.nx
        y = torch.arange(ny, dtype=self.dtype).view(1, ny, 1)
        z = torch.arange(nz, dtype=self.dtype).view(nz, 1, 1)
        x = torch.arange(nx, dtype=self.dtype).view(1, 1, nx)

        distance = torch.clamp(torch.minimum(y - 0.5, (ny - 1.5) - y), min=0.0)
        nu_ref = max(cfg.nu_newtonian, 1e-6)
        y_plus = distance * cfg.u_tau / nu_ref
        base = spalding_u_plus_from_y_plus(y_plus) * cfg.u_tau
        ux = base.expand(nz, ny, nx).clone()

        amp = 0.10 * cfg.u_tau
        phase_x = 2.0 * math.pi * x / nx
        phase_y = math.pi * torch.clamp(y - 0.5, min=0.0) / (2.0 * h)
        phase_z = 2.0 * math.pi * z / nz
        ux = ux + amp * torch.sin(phase_y) * torch.cos(phase_z)
        uy = amp * torch.sin(phase_x) * torch.sin(phase_y) ** 2 * torch.cos(phase_z)
        uz = (
            -amp
            * (nx / (2.0 * h))
            * torch.sin(phase_x)
            * torch.sin(2.0 * phase_y)
            * torch.sin(phase_z)
        )

        gen = torch.Generator().manual_seed(cfg.seed)
        noise = 0.02 * cfg.u_tau * torch.randn((nz, ny, nx), generator=gen, dtype=self.dtype)
        ux = (ux + noise) * self.fluid
        uy = uy * self.fluid
        uz = (uz + 0.5 * noise) * self.fluid
        rho = torch.ones_like(ux)
        self.f = self._equilibrium(rho, ux, uy, uz).to(self.dtype)

    # -- kernels ----------------------------------------------------------
    def _micro(self):
        return self._macroscopic(self.f)

    def _stream(self, f: torch.Tensor) -> torch.Tensor:
        if self.Q == 19:
            return stream3d(f)
        out = torch.empty_like(f)  # pragma: no cover
        for q in range(self.Q):
            out[q] = torch.roll(
                f[q],
                shifts=(
                    int(self.C[q, 2]),
                    int(self.C[q, 1]),
                    int(self.C[q, 0]),
                ),
                dims=(0, 1, 2),
            )
        return out

    def _bounce_back(self, f: torch.Tensor) -> torch.Tensor:
        """Half-way bounce-back on the two wall rows (in place).

        Only the two wall planes are touched, which keeps the reflection a
        narrow-slice copy instead of a full-field gather.
        """
        bottom = f[:, :, 0, :].index_select(0, self.opp_flat)
        top = f[:, :, -1, :].index_select(0, self.opp_flat)
        f[:, :, 0, :] = bottom
        f[:, :, -1, :] = top
        return f

    def _guo_force(self, f: torch.Tensor, rho, ux, uy, uz) -> torch.Tensor:
        cfg = self.cfg
        F = cfg.rho * cfg.body_force * self.fluid
        cu = self.cx * ux + self.cy * uy + self.cz * uz
        src = self.w * (3.0 * self.cx * F + 9.0 * self.cx * cu * F)
        return f + src

    def _shear_rate_from_stress(self, f, feq, rho, tau) -> torch.Tensor:
        """``gd`` from the non-equilibrium momentum flux (standard non-Newtonian
        LBM estimator ``S_ab = -3 Pi^neq_ab / (2 rho tau)``).

        Local and free of the grid-scale noise that central differences of the
        velocity field amplify (which otherwise feeds back into the viscosity
        field and destabilises the run).
        """
        d = f - feq
        co = self._coef
        scale = -1.5 / (rho * tau)
        Sxx = (d * co["xx"]).sum(0) * scale
        Syy = (d * co["yy"]).sum(0) * scale
        Szz = (d * co["zz"]).sum(0) * scale
        Sxy = (d * co["xy"]).sum(0) * scale
        Sxz = (d * co["xz"]).sum(0) * scale
        Syz = (d * co["yz"]).sum(0) * scale
        return torch.sqrt(
            torch.clamp(
                2.0 * (Sxx * Sxx + Syy * Syy + Szz * Szz)
                + 4.0 * (Sxy * Sxy + Sxz * Sxz + Syz * Syz),
                min=0.0,
            )
        )

    def _viscosity_field(self, ux, uy, uz, gd=None) -> torch.Tensor:
        cfg = self.cfg
        if gd is None:
            gd = strain_rate_magnitude(ux, uy, uz)
        nu = papanastasiou_viscosity(
            gd,
            tau_y=cfg.tau_y,
            K=cfg.K,
            n=cfg.n,
            rho=cfg.rho,
            m_pap=cfg.m_effective,
            nu_min=cfg.nu_min,
            nu_max=cfg.nu_max_effective,
        )
        if cfg.smagorinsky_cs > 0.0:
            nu = torch.clamp(
                nu + cfg.smagorinsky_cs**2 * gd, min=cfg.nu_min, max=cfg.nu_max_effective
            )
        return nu

    def _collide(self, f, feq, tau) -> torch.Tensor:
        """Collision operator (BGK or two-relaxation-time)."""
        if self.cfg.collision == "bgk":
            return f - (f - feq) / tau.unsqueeze(0)
        # TRT: split into symmetric/antisymmetric parts of every lattice pair
        f_opp = f.index_select(0, self.opp_flat)
        eq_opp = feq.index_select(0, self.opp_flat)
        fp = 0.5 * (f + f_opp)
        fm = 0.5 * (f - f_opp)
        ep = 0.5 * (feq + eq_opp)
        em = 0.5 * (feq - eq_opp)
        s_p = 1.0 / tau
        tau_minus = 0.5 + self.cfg.trt_lambda / (tau - 0.5)
        s_m = 1.0 / tau_minus
        return f - s_p.unsqueeze(0) * (fp - ep) - s_m.unsqueeze(0) * (fm - em)

    def step(self) -> None:
        cfg = self.cfg
        rho, ux, uy, uz = self._micro()
        feq = self._equilibrium(rho, ux, uy, uz)

        if self._step_count % cfg.nu_update_interval == 0 or self._tau is None:
            if cfg.shear_rate_source == "neq":
                gd = self._shear_rate_from_stress(
                    self.f,
                    feq,
                    rho,
                    self._tau if self._tau is not None else torch.full_like(rho, 0.6),
                )
            else:
                gd = strain_rate_magnitude(ux, uy, uz)
            nu = self._viscosity_field(ux, uy, uz, gd=gd)
            self._tau = torch.clamp(0.5 + 3.0 * nu, min=cfg.tau_min)
        self._step_count += 1

        self.f = self._collide(self.f, feq, self._tau)
        f_new = self._stream(self.f)
        f_new = self._guo_force(f_new, rho, ux, uy, uz)
        self.f = self._bounce_back(f_new)

    # -- statistics --------------------------------------------------------
    def accumulate(self) -> None:
        _, ux, uy, uz = self._micro()
        fl = self.fluid.to(self.dtype)
        w = fl.sum(dim=(0, 2)).clamp(min=1.0)
        self._u_sum += (ux * fl).sum(dim=(0, 2)) / w
        self._uu_sum += (ux * ux * fl).sum(dim=(0, 2)) / w
        self._vv_sum += (uy * uy * fl).sum(dim=(0, 2)) / w
        self._ww_sum += (uz * uz * fl).sum(dim=(0, 2)) / w
        self._uv_sum += (ux * uy * fl).sum(dim=(0, 2)) / w
        self._samples += 1
        ub = float((ux * fl).sum() / fl.sum())
        self._ubulk_series.append(ub)
        tke = (
            0.5
            * (
                (ux * ux * fl).sum()
                - (ux * fl).sum() ** 2 / fl.sum()
                + (uy * uy * fl).sum()
                + (uz * uz * fl).sum()
            )
            / fl.sum()
        )
        self._tke_series.append(float(tke))

    def statistics(self) -> dict:
        n = max(self._samples, 1)
        u = (self._u_sum / n).numpy()
        uu = (self._uu_sum / n).numpy()
        vv = (self._vv_sum / n).numpy()
        ww = (self._ww_sum / n).numpy()
        uv = (self._uv_sum / n).numpy()
        up2 = uu - u * u
        tke = 0.5 * (up2 + vv + ww)

        series = list(self._ubulk_series)
        blocks = self._block_stats(series, 8)
        ub = float(sum(series) / len(series)) if series else float("nan")
        sem = float(blocks["sem"]) if series else float("nan")
        return {
            "u_mean": u,
            "u_prime2": up2,
            "v_prime2": vv,
            "w_prime2": ww,
            "uv_prime": uv,
            "tke": tke,
            "samples": self._samples,
            "u_bulk": ub,
            "u_bulk_sem": sem,
            "u_bulk_blocks": blocks["blocks"],
            "tke_volume_mean": float(sum(self._tke_series) / len(self._tke_series))
            if self._tke_series
            else float("nan"),
            "u_bulk_series_decimated": series[:: max(1, len(series) // 400)][:400],
        }

    @staticmethod
    def _block_stats(values: list[float], n_blocks: int) -> dict:
        if len(values) < 2 * n_blocks:
            return {"mean": float("nan"), "sem": float("nan"), "blocks": []}
        size = len(values) // n_blocks
        means = [sum(values[i * size : (i + 1) * size]) / size for i in range(n_blocks)]
        mean = sum(means) / n_blocks
        var = sum((m - mean) ** 2 for m in means) / (n_blocks - 1)
        return {"mean": mean, "sem": math.sqrt(var / n_blocks), "blocks": means}

    def wall_shear_from_gradient(self) -> float:
        """Wall shear stress inferred from the wall-adjacent velocity gradient.

        Diagnostic only: ``tau_w,grad = rho * nu_eff * du/dy`` evaluated one
        lattice unit from each wall, averaged over both walls and the
        homogeneous (x, z) planes.  This is what an experiment would measure;
        global momentum balance gives ``tau_w = rho*u_tau^2`` exactly in the
        statistically steady state.
        """
        cfg = self.cfg
        _, ux, uy, uz = self._micro()
        gd = strain_rate_magnitude(ux, uy, uz)
        nu = papanastasiou_viscosity(
            gd,
            tau_y=cfg.tau_y,
            K=cfg.K,
            n=cfg.n,
            rho=cfg.rho,
            m_pap=cfg.m_effective,
            nu_min=cfg.nu_min,
            nu_max=cfg.nu_max_effective,
        )
        if cfg.smagorinsky_cs > 0.0:
            nu = torch.clamp(
                nu + cfg.smagorinsky_cs**2 * gd, min=cfg.nu_min, max=cfg.nu_max_effective
            )
        dudy_bottom = float(ux[:, 1, :].mean()) / 1.0  # wall plane at y = 0.5
        dudy_top = float(ux[:, -2, :].mean()) / 1.0
        nu_b = float(nu[:, 1, :].mean())
        nu_t = float(nu[:, -2, :].mean())
        return cfg.rho * 0.5 * (nu_b * dudy_bottom + nu_t * dudy_top)

    def run(self, log=print) -> dict:
        cfg = self.cfg
        t0 = time.time()
        for step in range(1, cfg.steps + 1):
            self.step()
            if step > cfg.warmup_steps and step % cfg.sample_interval == 0:
                self.accumulate()
            if log and step % cfg.report_interval == 0:
                fl = self.fluid.to(self.dtype)
                ub = float((self._micro()[1] * fl).sum() / fl.sum())
                log(f"  {cfg.name}: step {step}/{cfg.steps} U_b={ub:.5f}")
        return {"steps": cfg.steps, "elapsed_s": time.time() - t0}


def wall_units(half_height: float, u_tau: float, re_tau: float):
    """Wall-unit sampling coordinates ``(y, y+)`` for the node rows."""
    ny = int(round(2.0 * half_height)) + 2
    y = torch.arange(ny, dtype=torch.float64) - 0.5
    y = torch.clamp(y, min=0.0)
    nu = u_tau * half_height / re_tau
    return y.numpy(), (y * u_tau / nu).numpy()


def run_hb_turbulent_channel(cfg: HBTurbulentChannelConfig, log=print) -> dict:
    """Run one HB channel case and return a JSON-ready result dict."""
    solver = HBTurbulentChannel(cfg)
    info = solver.run(log=log)
    stats = solver.statistics()
    u_tau = cfg.u_tau  # fixed by the momentum balance of the forcing
    u_b = float(stats["u_bulk"])
    u_mean = stats["u_mean"]
    core = u_mean[cfg.ny // 2]
    if not math.isfinite(u_b) or u_b <= 1e-9:
        u_b = max(float(core), 0.0)
    re_mr = metzner_reed_reynolds(u_b, cfg.hydraulic_diameter, cfg.K, cfg.n, cfg.rho)
    f_darcy = darcy_friction_factor(u_tau, u_b)
    tke_max = float(stats["tke"].max())
    tke_vol = float(stats["tke_volume_mean"])
    return {
        "name": cfg.name,
        "config": {
            "nx": cfg.nx,
            "ny": cfg.ny,
            "nz": cfg.nz,
            "re_tau": cfg.re_tau,
            "u_tau": cfg.u_tau,
            "tau_y": cfg.tau_y,
            "K": cfg.K,
            "n": cfg.n,
            "m_pap": cfg.m_effective,
            "smagorinsky_cs": cfg.smagorinsky_cs,
            "lattice": cfg.lattice,
            "steps": cfg.steps,
            "warmup_steps": cfg.warmup_steps,
            "sample_interval": cfg.sample_interval,
            "dtype": cfg.dtype,
            "seed": cfg.seed,
            "half_height": cfg.half_height,
            "hydraulic_diameter": cfg.hydraulic_diameter,
            "body_force": cfg.body_force,
            "nu_newtonian": cfg.nu_newtonian,
        },
        "derived": {
            "u_tau": u_tau,
            "u_bulk": u_b,
            "u_bulk_core": float(core),
            "u_bulk_sem": stats["u_bulk_sem"],
            "yield_number": cfg.yield_number,
            "re_mr": re_mr,
            "f_darcy": f_darcy,
            "f_re_mr": f_darcy * re_mr,
            "tke_max": tke_max,
            "tke_volume_mean": tke_vol,
            "f_dodge_metzner": dodge_metzner_friction(re_mr, cfg.n),
            "tau_w_nominal": cfg.tau_wall,
            "tau_w_gradient": solver.wall_shear_from_gradient(),
        },
        "profiles": {
            "u_mean": [float(v) for v in stats["u_mean"]],
            "u_prime2": [float(v) for v in stats["u_prime2"]],
            "v_prime2": [float(v) for v in stats["v_prime2"]],
            "w_prime2": [float(v) for v in stats["w_prime2"]],
            "uv_prime": [float(v) for v in stats["uv_prime"]],
            "tke": [float(v) for v in stats["tke"]],
        },
        "run_info": info,
        "stationarity": {
            "u_bulk_blocks": stats["u_bulk_blocks"],
            "u_bulk_series_decimated": stats["u_bulk_series_decimated"],
            "samples": stats["samples"],
        },
    }
