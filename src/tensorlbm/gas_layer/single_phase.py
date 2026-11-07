"""Single-phase plane-channel LBM: wall-shear and profile calibration gate.

Why this module
---------------
The drag-reduction claim of the gas-layer campaign is a *ratio* of wall shear
stresses, ``DR = 1 - tau_w,2ph / tau_w,1ph``.  A ratio is only meaningful if the
denominator is calibrated: the same wall-shear estimator that is used on the
two-phase run must reproduce the closed-form laminar channel shear to within the
3 % gate.  This module provides

* a D2Q9 BGK channel with half-way bounce-back walls, Guo (2002) forcing and
  exact momentum-exchange wall shear (:class:`SinglePhaseChannel2D`);
* the analytic cross-checks from :mod:`tensorlbm.gas_layer.reference`;
* :func:`run_single_phase_gate`, which runs the calibration at a prescribed
  ``Re_tau`` (default 180) and returns the numbers the acceptance gate needs.

The wall-shear estimator (momentum exchange, Ladd 1994)
-------------------------------------------------------
For every population that streams from a fluid cell into a wall node, the wall
receives ``2 c_i f_i``.  Summing over the wall and dividing by the wall length
gives the wall stress in lattice units.  No finite difference, no fitted
velocity gradient: the estimator is exact for half-way bounce-back and is the
same estimator used on the two-phase run.

What the ``Re_tau = 180`` reference is -- and is not
----------------------------------------------------
At ``Re_tau = 180`` a *turbulent* channel has the universal log layer.  A 2-D
lattice channel at that ``Re_tau`` is laminar and cannot reproduce it, so the
log law is **not** used as a gate here (it is plotted as context only, and the
report labels a point-by-point log-law comparison WITHHELD for lack of a
turbulent 3-D reference run).  What *is* exact and therefore gates the solver at
``Re_tau = 180`` is the laminar law ``u+ = y+ (1 - y+/(2 Re_tau))`` and
``Cf = 12/Re_h``; both are closed-form and independent of any simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..d2q9 import OPPOSITE, C, W, equilibrium
from . import reference

CS2 = 1.0 / 3.0

#: Directions whose ``c_y`` is negative (stream towards the bottom wall).
_DOWN = [i for i in range(9) if int(C[i, 1]) < 0]
_UP = [i for i in range(9) if int(C[i, 1]) > 0]


def tau_from_nu(nu: float) -> float:
    """Relaxation time from kinematic viscosity: ``nu = cs^2 (tau - 1/2)``."""
    return 0.5 + nu / CS2


def nu_from_tau(tau: float) -> float:
    """Kinematic viscosity from relaxation time."""
    return CS2 * (tau - 0.5)


@dataclass
class SinglePhaseChannel2D:
    """D2Q9 BGK plane channel, half-way bounce-back walls, Guo body force.

    Attributes
    ----------
    ny, nx:   lattice size; rows ``0`` and ``ny-1`` are solid wall nodes, the
              fluid occupies rows ``1 .. ny-2`` and is periodic along ``x``.
    nu:       kinematic viscosity (lattice units).
    g:        body-force acceleration along ``x`` (lattice units).
    rho0:     initial density (lattice units, ~1).
    """

    ny: int = 66
    nx: int = 8
    nu: float = 0.05
    g: float = 1.0e-4
    rho0: float = 1.0
    dtype: torch.dtype = torch.float64
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    f: torch.Tensor = field(init=False, repr=False)
    shear_accum: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.ny < 8 or self.nx < 2:
            raise ValueError("need ny >= 8 and nx >= 2 fluid columns")
        if self.nu <= 0.0:
            raise ValueError("nu must be positive")
        self.tau = tau_from_nu(self.nu)
        if self.tau <= 0.55:
            raise ValueError("nu too small: tau would be below 0.55")
        shape = (9, self.ny, self.nx)
        self.f = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.shear_accum = torch.zeros(2, dtype=self.dtype, device=self.device)
        self.reset()

    # -- geometry -----------------------------------------------------------
    @property
    def height(self) -> float:
        """Channel height between the two half-way walls."""
        return float(self.ny - 1)

    @property
    def fluid_rows(self) -> tuple[int, int]:
        """Inclusive row range of fluid nodes."""
        return (1, self.ny - 2)

    def _y_coords(self) -> torch.Tensor:
        """Wall distance of the fluid-node centres (``y = 0.5`` at the wall)."""
        return torch.arange(1, self.ny - 1, dtype=self.dtype, device=self.device).view(-1, 1)

    # -- initialisation -----------------------------------------------------
    def reset(self) -> None:
        """Fill the domain with still fluid at ``rho0``."""
        rho = torch.full((self.ny, self.nx), self.rho0, dtype=self.dtype, device=self.device)
        zero = torch.zeros_like(rho)
        self.f = equilibrium(rho, zero, zero)
        self._mask_walls()

    def initialize_with_exact_profile(self) -> None:
        """Start from the exact laminar solution (steady-state verification)."""
        y = self._y_coords()
        u = reference.laminar_profile(self.g, self.nu, self.height, y)
        rho = torch.full((self.ny, self.nx), self.rho0, dtype=self.dtype, device=self.device)
        ux = torch.zeros_like(rho)
        ux[1 : self.ny - 1, :] = u.expand(self.ny - 2, self.nx)
        zero = torch.zeros_like(rho)
        self.f = equilibrium(rho, ux, zero)
        self._mask_walls()

    def _mask_walls(self) -> None:
        self.f[:, 0, :] = 0.0
        self.f[:, self.ny - 1, :] = 0.0

    # -- operators ----------------------------------------------------------
    def macroscopic(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(rho, ux, uy)`` including the Guo half-force correction."""
        rho = self.f.sum(dim=0)
        rho_s = torch.clamp(rho, min=1e-12)  # wall rows carry no populations
        ux = (self.f * C[:, 0].view(9, 1, 1)).sum(dim=0) / rho_s
        uy = (self.f * C[:, 1].view(9, 1, 1)).sum(dim=0) / rho_s
        ux = ux + 0.5 * self.g  # F/(2 rho) with F = rho g, rho ~ const
        return rho, ux, uy

    def _collide(self) -> torch.Tensor:
        """BGK collision with Guo forcing; returns post-collision populations."""
        rho, ux, uy = self.macroscopic()
        # Guo (2002): the equilibrium is built on the *force-corrected* velocity
        # u_phys = sum(c f)/rho + F/(2 rho) -- which is what macroscopic()
        # returns -- and the source term is built on the same velocity.  Using
        # the bare moment here would deliver only (1 - 1/(2 tau)) F per step.
        feq = equilibrium(rho, ux, uy)
        f_out = self.f - (self.f - feq) / self.tau
        c = C.to(dtype=self.dtype)
        w = W.to(dtype=self.dtype)
        fx = rho * self.g
        cf = c[:, 0].view(9, 1, 1) * fx
        cu = c[:, 0].view(9, 1, 1) * ux
        uf = ux * fx
        src = w.view(9, 1, 1) * (1.0 - 0.5 / self.tau) * ((cf - uf) / CS2 + cu * cf / CS2**2)
        return f_out + src

    def _stream(self, f_post: torch.Tensor) -> torch.Tensor:
        """Half-way bounce-back streaming (periodic ``x``, solid rows 0/ny-1)."""
        ny = self.ny
        f_new = torch.zeros_like(f_post)
        f_new[0] = f_post[0]
        for i in range(1, 9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            src = f_post[i]
            if cx:
                src = torch.roll(src, shifts=-cx, dims=1)
            j = int(OPPOSITE[i])
            if cy == 1:
                f_new[i][1:, :] += src[:-1, :]
                f_new[j][ny - 2, :] += src[ny - 2, :]
            elif cy == -1:
                f_new[i][:-1, :] += src[1:, :]
                f_new[j][1, :] += src[1, :]
            else:
                f_new[i] += src
        f_new[:, 0, :] = 0.0
        f_new[:, ny - 1, :] = 0.0
        return f_new

    def step(self, n: int = 1) -> None:
        """Advance ``n`` lattice steps, accumulating momentum-exchange shear."""
        for _ in range(n):
            f_post = self._collide()
            # momentum exchange into the bottom/top walls before streaming
            fx_bottom = torch.zeros(self.nx, dtype=self.dtype, device=self.device)
            fx_top = torch.zeros(self.nx, dtype=self.dtype, device=self.device)
            for i in _DOWN:
                fx_bottom += 2.0 * float(C[i, 0]) * f_post[i, 1, :]
            for i in _UP:
                fx_top += 2.0 * float(C[i, 0]) * f_post[i, self.ny - 2, :]
            self.shear_accum[0] += fx_bottom.sum() / self.nx
            self.shear_accum[1] += fx_top.sum() / self.nx
            self.f = self._stream(f_post)

    # -- diagnostics --------------------------------------------------------
    def wall_shear(self, reset: bool = False) -> tuple[float, float]:
        """Time-averaged bottom/top wall shear stress (momentum exchange)."""
        n = getattr(self, "_n_shear", 0)
        if n == 0:
            raise RuntimeError("no steps accumulated")
        bottom = float(self.shear_accum[0]) / n
        top = float(self.shear_accum[1]) / n
        if reset:
            self.shear_accum.zero_()
            self._n_shear = 0
        return bottom, top

    def bulk_velocity(self) -> float:
        """Cross-section averaged streamwise velocity over the fluid rows."""
        _, ux, _ = self.macroscopic()
        return float(ux[1:-1, :].mean())

    def profile(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(y, ux)`` for the fluid rows (column-averaged)."""
        _, ux, _ = self.macroscopic()
        y = torch.arange(1, self.ny - 1, dtype=self.dtype, device=self.device)
        # wall distance: fluid node y sits 0.5 above the wall node at y=0
        return y - 0.5, ux[1:-1, :].mean(dim=1)

    def shear_steps(self, n: int) -> None:
        """Run ``n`` steps with shear accounting (bookkeeping for the average)."""
        self.step(n)
        self._n_shear = getattr(self, "_n_shear", 0) + n


def run_single_phase_gate(
    re_tau_target: float = 180.0,
    ny: int = 130,
    nx: int = 4,
    n_steps: int = 4000,
    n_warmup: int = 1000,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> dict[str, object]:
    """Calibrate the wall-shear estimator against the exact laminar channel.

    The channel is started from the exact laminar profile so that the run is a
    *steady-state* verification of the discrete scheme, wall-shear estimator and
    buoyancy/forcing balance -- not a transient.  ``Re_tau`` sets ``g`` through
    ``tau_w = rho g h/2 = rho u_tau^2``.

    Returns a JSON-serialisable dict with the numbers the acceptance gate needs
    (``cf_rel_error``, ``profile_rel_error``, ``tau_w_rel_error``,
    ``bulk_velocity``, ``re_tau_measured``) plus the profile for plotting.
    """
    dev = torch.device(device)
    h = float(ny - 1)
    half = 0.5 * h
    # pick nu so that the channel is well resolved and tau ~ 0.8
    tau = 0.8
    nu = nu_from_tau(tau)
    u_tau = re_tau_target * nu / half
    g = 2.0 * u_tau * u_tau / h
    ch = SinglePhaseChannel2D(ny=ny, nx=nx, nu=nu, g=g, dtype=dtype, device=dev)
    ch.initialize_with_exact_profile()
    ch.step(n_warmup)
    # reset the shear accumulator after the warm-up
    ch.shear_accum.zero_()
    ch._n_shear = 0
    ch.shear_steps(n_steps)

    tau_w_bottom, tau_w_top = ch.wall_shear()
    u_bulk = ch.bulk_velocity()
    tau_w_exact = reference.laminar_wall_shear(g, 1.0, h)
    U_exact = reference.laminar_mean_velocity(g, nu, h)
    cf_exact = reference.friction_coefficient(U_exact, h, nu)
    cf_meas = reference.friction_coefficient_from_shear(tau_w_bottom, u_bulk)

    y, ux = ch.profile()
    y_np = y.cpu().numpy()
    u_np = ux.cpu().numpy()
    u_exact = reference.laminar_profile(g, nu, h, y_np)
    y_plus = y_np * u_tau / nu
    re_tau_meas = reference.re_tau(reference.utau(tau_w_bottom, 1.0), half, nu)
    u_plus = u_np / reference.utau(tau_w_bottom, 1.0)
    u_plus_exact = reference.laminar_u_plus(y_plus, re_tau_meas)
    # Profile accuracy: relative L2 norm over the whole profile (the metric the
    # gate uses) plus the worst pointwise error.  The pointwise maximum is
    # reported separately because the wall-adjacent node carries the largest
    # discrete error while contributing almost none of the momentum flux.
    profile_err = float(((u_np - u_exact) ** 2).sum() ** 0.5) / float(
        (u_exact**2).sum() ** 0.5
    )
    profile_max_err = float(abs(u_np - u_exact).max() / float(abs(u_exact).max()))

    return {
        "case": "single_phase_channel_gate",
        "re_tau_target": re_tau_target,
        "re_tau_measured": re_tau_meas,
        "ny": ny,
        "nx": nx,
        "nu": nu,
        "tau": tau,
        "g": g,
        "n_steps": n_steps,
        "n_warmup": n_warmup,
        "u_bulk_measured": u_bulk,
        "u_bulk_exact": U_exact,
        "u_bulk_rel_error": abs(u_bulk - U_exact) / U_exact,
        "tau_w_measured": tau_w_bottom,
        "tau_w_top": tau_w_top,
        "tau_w_exact": tau_w_exact,
        "tau_w_rel_error": abs(tau_w_bottom - tau_w_exact) / tau_w_exact,
        "cf_measured": cf_meas,
        "cf_exact": cf_exact,
        "cf_rel_error": abs(cf_meas - cf_exact) / cf_exact,
        "profile_rel_error": profile_err,
        "profile_max_rel_error": profile_max_err,
        "y_over_h": (y_np / h).tolist(),
        "u_plus": u_plus.tolist(),
        "u_plus_exact": u_plus_exact.tolist(),
        "y_plus": y_plus.tolist(),
        "log_law_context_only": [
            reference.log_law_u_plus(float(yp)) for yp in y_plus.tolist()
        ],
        "gate": {
            "cf_rel_error_le_3pct": bool(abs(cf_meas - cf_exact) / cf_exact <= 0.03),
            "profile_rel_error_le_3pct": bool(profile_err <= 0.03),
        },
    }