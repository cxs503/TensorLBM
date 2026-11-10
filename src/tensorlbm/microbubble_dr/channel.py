"""Shan-Chen microbubble drag-reduction channel (campaign node IN67-N01).

Physical set-up (lattice units; ``x`` streamwise, ``y`` wall-normal)
-------------------------------------------------------------------
* plane channel, walls at rows ``0`` and ``ny-1`` (half-way bounce-back), fluid
  rows ``1 .. ny-2``, streamwise periodic, driven by a uniform body force
  ``g_x`` that an integral controller adjusts so that the **mass flux**
  ``J = <rho u_x>`` matches between the single-phase reference and every
  two-phase run.  Matching ``J`` (not the forcing) is what makes the wall-shear
  ratio a drag-reduction statement: at matched forcing the momentum balance
  ``tau_w = g_x * (mass per unit area of the half channel)`` fixes ``tau_w`` and
  no drag change can exist by construction.
* the carrier is a Shan-Chen single-component fluid with the exponential
  pseudopotential; gas is seeded as circular bubbles of the *lattice* gas
  density and the void fraction is the resolved field
  ``alpha = (rho_l - rho)/(rho_l - rho_g)``.  ``rho_l``/``rho_g`` come from the
  Maxwell construction on this model's own pressure (:mod:`.eos`).
* a **phase-dependent relaxation time** ``tau(alpha) = tau_l + alpha (tau_g -
  tau_l)`` is used, so the dispersed gas really does change the mixture
  viscosity -- that is the model's viscosity-contrast channel and the reason the
  viscosity drag term is non-zero and measurable rather than assumed.

Diagnostics
-----------
* wall shear from the half-way bounce-back **momentum exchange** (Ladd), one
  value per wall per step, averaged over the second half of the run;
* Reynolds statistics accumulated as ``x``-averages of products, so that
  ``<u'v'> = <uv> - <u><v>`` is available on the half channel;
* three independent friction-velocity estimates (wall gradient, Reynolds-stress
  identity, total shear from the momentum balance) as a consistency check.

Scope / honesty
---------------
The resolved friction Reynolds number is ``Re_tau ~ 20-40`` here (a CPU-budget
Shan-Chen channel), i.e. the carrier is a *laminar* plane Poiseuille flow and the
fluctuations are bubble-induced (pseudo-turbulence), not wall turbulence.  The
``d+`` scan therefore probes bubble size *relative to the channel*; absolute
wall-unit scaling of a real turbulent boundary layer needs a 3D DNS at
``Re_tau >= 180`` that this node does not claim (see the report's WITHHELD item).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..gas_layer import kernels

# Small lattice arrays: intra-op threading costs far more than it saves (measured
# 20x slowdown at the default 128 threads), so pin this module to one thread.
torch.set_num_threads(1)
from ..multiphase import psi_exp
from .decomposition import HalfChannelProfiles, wall_shear_budget
from .eos import maxwell_coexistence, kinematic_viscosity


_COEX_CACHE: dict[tuple[float, ...], tuple[float, float]] = {}


def calibrate_coexistence(G: float = 5.0, rho0: float = 1.0, n: int = 24,
                          steps: int = 1500, tau: float = 0.7,
                          seed: int = 7) -> tuple[float, float]:
    """Measure the model's own ``(rho_gas, rho_liquid)`` from a phase-separating box.

    The Maxwell construction on the lattice EOS (:func:`.eos.maxwell_coexistence`)
    is the leading-order answer, but a lattice Shan-Chen fluid coexists at
    slightly different densities.  Since the void fraction
    ``alpha = (rho_l - rho)/(rho_l - rho_g)`` is a *linear* functional of the
    density, a percent-level error in ``rho_g`` shows up as a percent-level drift
    of ``int alpha`` -- which would then be mis-read as a mass-conservation
    failure.  The densities are therefore measured from an unbiased periodic box
    run of this very collision/streaming pair and cached per ``(G, rho0)``.
    """
    key = (float(G), float(rho0), int(n), int(steps), float(tau))
    if key in _COEX_CACHE:
        return _COEX_CACHE[key]
    rg0, rl0 = maxwell_coexistence(G, rho0)
    rng = np.random.default_rng(seed)
    rho_np = np.full((n, n), 0.5 * (rl0 + rg0)) + 0.02 * rng.standard_normal((n, n))
    rho_t = torch.as_tensor(rho_np, dtype=torch.float32)
    f = kernels.equilibrium(rho_t, torch.zeros_like(rho_t), torch.zeros_like(rho_t))
    psi = lambda r: psi_exp(r, rho0)
    for _ in range(steps):
        rho, ux, uy = kernels.macroscopic(f)
        fx, fy = kernels.sc_force(rho, psi, G=G, gx=0.0, gy=0.0, solid=None,
                                  wall_psi=None, periodic_x=True)
        rho_s = torch.clamp(rho, min=1e-12)
        feq_u = kernels.equilibrium(rho, ux, uy)
        feq_f = kernels.equilibrium(rho, ux + fx / rho_s, uy + fy / rho_s)
        f = f - (f - feq_u) / tau + (feq_f - feq_u)
        f = kernels.stream(f, solid=None, periodic_x=True, periodic_y=True)["f"]
    rho_end = kernels.macroscopic(f)[0].numpy()
    mid = 0.5 * (rl0 + rg0)
    gas = rho_end[rho_end < mid]
    liq = rho_end[rho_end >= mid]
    if gas.size < 8 or liq.size < 8:
        rg, rl = rg0, rl0            # no clean separation -> keep the EOS values
    else:
        rg = float(np.median(gas))
        rl = float(np.median(liq))
        if not (rl > 2.0 * rg > 0.0):
            rg, rl = rg0, rl0
    _COEX_CACHE[key] = (rg, rl)
    return rg, rl


@dataclass
class MicrobubbleConfig:
    """One channel run (lattice units)."""

    nx: int = 128
    ny: int = 129
    #: liquid relaxation time -> nu_l = (tau - 1/2)/3
    tau_liquid: float = 0.56
    #: gas relaxation time (viscosity contrast channel); floored at 0.52 by the
    #: solver -- a sharply varying relaxation time on top of the Shan-Chen force
    #: is the classic instability of variable-viscosity multiphase LBM
    tau_gas: float = 0.55
    #: Shan-Chen coupling (library sign convention: > 0 attracts)
    G: float = 5.0
    rho0: float = 1.0
    #: target mass flux <rho u_x> in lattice units
    J_target: float = 0.18
    #: target bubble diameter in wall units (d+), used with the *reference*
    #: friction velocity to size the bubbles
    d_plus: float = 10.0
    #: number of bubbles; ``None`` -> derived from ``void_fraction``
    n_bubbles: int | None = None
    #: target gas volume fraction (only used when ``n_bubbles is None``)
    void_fraction: float = 0.02
    #: seeding band half thickness, in bubble diameters, measured from the wall
    band_diameters: float = 1.5
    n_steps: int = 16000
    sample_every: int = 20
    control_every: int = 100
    control_gain: float = 0.03
    warmup_fraction: float = 0.5
    seed: int = 20261109
    #: float64 is the default: in float32 two of the nine production cases
    #: diverged late in the run (the Shan-Chen force with a phase-dependent
    #: relaxation time is only marginally stable in single precision)
    dtype: torch.dtype = torch.float64
    #: interface smoothing width for the initial bubbles (lattice cells)
    interface_width: float = 0.9
    wall_psi_ratio: float = 1.0
    #: if False the relaxation time is uniform (viscosity contrast switched off)
    viscosity_contrast: bool = True
    #: measure the coexistence densities from a probe run instead of using the
    #: Maxwell construction alone (see :func:`calibrate_coexistence`)
    calibrate_coexistence: bool = True
    coex_calibration_steps: int = 1500

    def __post_init__(self) -> None:
        if self.ny % 2 == 0:
            raise ValueError("ny must be odd so that a centreline row exists")
        if self.tau_liquid <= 0.5 or self.tau_gas <= 0.5:
            raise ValueError("relaxation times must exceed 1/2")

    @property
    def half_height(self) -> float:
        return (self.ny - 1) / 2.0

    @property
    def nu_liquid(self) -> float:
        return kinematic_viscosity(self.tau_liquid)

    @property
    def nu_gas(self) -> float:
        return kinematic_viscosity(self.tau_gas)


class MicrobubbleChannel2D:
    """Two-phase plane channel with seeded micro-bubbles."""

    def __init__(self, config: MicrobubbleConfig | None = None):
        self.cfg = cfg = config or MicrobubbleConfig()
        dtype, device = cfg.dtype, torch.device("cpu")
        self.device = device
        self.solid = torch.zeros((cfg.ny, cfg.nx), dtype=torch.bool, device=device)
        self.solid[0, :] = True
        self.solid[cfg.ny - 1, :] = True
        self.fluid = ~self.solid
        if cfg.calibrate_coexistence:
            self.rho_gas, self.rho_liquid = calibrate_coexistence(
                cfg.G, cfg.rho0, steps=cfg.coex_calibration_steps)
            self._rho_from_maxwell = maxwell_coexistence(cfg.G, cfg.rho0)
        else:
            self.rho_gas, self.rho_liquid = maxwell_coexistence(cfg.G, cfg.rho0)
            self._rho_from_maxwell = (self.rho_gas, self.rho_liquid)
        self.psi = lambda r: psi_exp(r, cfg.rho0)
        self.wall_psi = cfg.wall_psi_ratio * float(self.psi(torch.tensor(self.rho_liquid)))
        self.t = 0
        self.gx0 = self._initial_gx()
        self.gx = self.gx0
        self.mass_initial = 0.0
        self.alpha_initial = 0.0
        self._history: dict[str, list[float]] = {
            "tau_w": [], "J": [], "gx": [], "alpha": [],
        }
        self._acc: dict[str, torch.Tensor] = {}
        self._n_samples = 0
        self.bubble_centers = np.zeros((0, 2))
        self.bubble_radius = 0.0
        self.f = self._initial_field()

    # -- set-up --------------------------------------------------------------
    def _initial_gx(self) -> float:
        """Laminar estimate ``g_x = 3 nu U_b / h^2`` with ``U_b = J/rho_l``."""
        cfg = self.cfg
        u_bulk = cfg.J_target / self.rho_liquid
        return 3.0 * cfg.nu_liquid * u_bulk / (cfg.half_height ** 2)

    def friction_velocity_estimate(self) -> float:
        """Laminar ``u_tau = sqrt(g_x h)`` (used only to size the bubbles)."""
        return math.sqrt(max(self.gx * self.cfg.half_height, 1e-14))

    def bubble_radius_for(self, d_plus: float) -> float:
        """Radius ``R = d+ nu_l / (2 u_tau)`` from the reference friction velocity."""
        u_tau = self.friction_velocity_estimate()
        return 0.5 * d_plus * self.cfg.nu_liquid / u_tau

    def _place_bubbles(self, radius: float, n_bubbles: int) -> np.ndarray:
        """Seeded, non-overlapping placement in near-wall bands (both walls)."""
        cfg = self.cfg
        if n_bubbles <= 0 or radius <= 0.0:
            return np.zeros((0, 2))
        rng = np.random.default_rng(cfg.seed)
        h = cfg.half_height
        band = max(cfg.band_diameters * 2.0 * radius, 2.0)
        y_lo = radius + 1.5
        y_hi = min(h - radius - 0.5, y_lo + band)
        if y_hi <= y_lo:
            y_lo, y_hi = max(1.0 + radius, 1.0), max(1.0 + radius, h - radius)
        centers: list[tuple[float, float]] = []
        for k in range(n_bubbles):
            for _ in range(4000):
                x = float(rng.uniform(0.0, cfg.nx))
                y_half = float(rng.uniform(y_lo, max(y_lo, y_hi)))
                y = y_half if k % 2 == 0 else (cfg.ny - 1) - y_half
                ok = True
                for (xc, yc) in centers:
                    dx = abs(x - xc)
                    dx = min(dx, cfg.nx - dx)          # periodic in x
                    if math.hypot(dx, y - yc) < 2.25 * radius:
                        ok = False
                        break
                if ok:
                    centers.append((x, y))
                    break
        return np.array(centers, dtype=float)

    def _initial_field(self) -> torch.Tensor:
        cfg = self.cfg
        ny, nx = cfg.ny, cfg.nx
        radius = self.bubble_radius_for(cfg.d_plus)
        fluid_area = float(self.fluid.sum())
        if cfg.n_bubbles is None:
            area = cfg.void_fraction * fluid_area
            n_bubbles = (max(1, int(round(area / (math.pi * radius ** 2))))
                         if radius > 0 and cfg.void_fraction > 0 else 0)
        else:
            n_bubbles = int(cfg.n_bubbles)
        centers = self._place_bubbles(radius, n_bubbles)
        self.bubble_centers = centers
        self.bubble_radius = radius

        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        rho = np.full((ny, nx), self.rho_liquid, dtype=np.float64)
        w = max(cfg.interface_width, 0.3)
        for (xc, yc) in centers:
            dx = np.abs(xx - xc)
            dx = np.minimum(dx, nx - dx)
            r = np.hypot(dx, yy - yc)
            # smooth tanh bubble profile (sharp jumps destabilise the SC force)
            s = 0.5 * (1.0 - np.tanh((r - radius) / w))
            rho = rho * (1.0 - s) + self.rho_gas * s
        rho[self.solid.cpu().numpy()] = self.rho_liquid
        self.f = self._equilibrium_from(rho)
        self.f = self._superpose_laminar(self.f, rho)
        self.mass_initial = float(rho[self.fluid.cpu().numpy()].sum())
        self.alpha_initial = self.mean_void_fraction()
        self.alpha_linear_initial = self.linear_void_fraction()
        self._reset_accumulators()
        return self.f

    def laminar_profile(self) -> np.ndarray:
        """Analytic plane-Poiseuille ``u(y) = g_x y (H - y) / (2 nu)`` (cells)."""
        cfg = self.cfg
        H = float(cfg.ny - 1)
        y = np.arange(cfg.ny, dtype=float) - 0.5
        u = self.gx * np.clip(y, 0.0, H) * (H - np.clip(y, 0.0, H)) / (2.0 * cfg.nu_liquid)
        u[0] = 0.0
        u[cfg.ny - 1] = 0.0
        return u

    def _superpose_laminar(self, f: torch.Tensor, rho: np.ndarray) -> torch.Tensor:
        """Start from the analytic Poiseuille field plus a tiny perturbation.

        The viscous spin-up time of this channel is ``h^2/nu ~ 3e4`` steps, far
        longer than the run, so starting from rest would leave the controller
        chasing a transient for the whole run.  Starting on the analytic solution
        means the run only has to reach the *two-phase* steady state.  A
        deterministic 1e-3 ``u_max`` perturbation breaks the exact symmetry so
        bubble wakes are not artificially suppressed.
        """
        cfg = self.cfg
        rho_t = torch.as_tensor(rho, dtype=cfg.dtype, device=self.device)
        u = self.laminar_profile()
        u_max = float(np.abs(u).max()) or 1.0
        rng = np.random.default_rng(cfg.seed + 1)
        u = u + 1e-3 * u_max * rng.standard_normal(u.shape)
        u[0] = 0.0
        u[cfg.ny - 1] = 0.0
        u2 = np.broadcast_to(u[:, None], rho.shape).copy()
        v2 = np.zeros_like(u2)
        ux_t = torch.as_tensor(u2, dtype=cfg.dtype, device=self.device)
        uy2_t = torch.as_tensor(v2, dtype=cfg.dtype, device=self.device)
        f = kernels.equilibrium(rho_t, ux_t, uy2_t).clone()
        f[:, self.solid] = 0.0
        return f

    def _equilibrium_from(self, rho: np.ndarray) -> torch.Tensor:
        cfg = self.cfg
        rho_t = torch.as_tensor(rho, dtype=cfg.dtype, device=self.device)
        zero = torch.zeros_like(rho_t)
        f = kernels.equilibrium(rho_t, zero, zero).clone()
        f[:, self.solid] = 0.0
        return f

    # -- fields --------------------------------------------------------------
    def macroscopic(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return kernels.macroscopic(self.f)

    def void_fraction(self, rho: torch.Tensor) -> torch.Tensor:
        a = (self.rho_liquid - rho) / (self.rho_liquid - self.rho_gas)
        return torch.clamp(a, 0.0, 1.0)

    def mean_void_fraction(self) -> float:
        rho, _, _ = self.macroscopic()
        return float(self.void_fraction(rho)[self.fluid].mean())

    def linear_void_fraction(self) -> float:
        """Unclamped ``<(rho_l - rho)/(rho_l - rho_g)>``.

        This functional is *linear* in ``rho``, so it is conserved to machine
        precision whenever the mass is (``sum(rho_l - rho)`` is a mass
        functional).  The clamped field is the physically meaningful one but its
        integral can drift a few percent when the gas relaxes to a density other
        than ``rho_g``; both are reported.
        """
        rho, _, _ = self.macroscopic()
        lin = (self.rho_liquid - rho) / (self.rho_liquid - self.rho_gas)
        return float(lin[self.fluid].mean())

    def mass_flux(self) -> float:
        rho, ux, _ = self.macroscopic()
        return float((rho * ux)[self.fluid].mean())

    def _tau_field(self, alpha: torch.Tensor) -> torch.Tensor:
        """Phase-dependent relaxation time ``tau(alpha)``, smoothed and floored.

        Two numerical safety measures, both measured rather than guessed: the
        void fraction is smoothed with a 3x3 box filter before it enters ``tau``
        (a cell-sharp ``tau`` jump across the interface destabilises the
        Shan-Chen force -- the run diverged within ~13 steps at 32x33 without
        this), and ``tau`` is floored well above 1/2.
        """
        cfg = self.cfg
        if not cfg.viscosity_contrast:
            return torch.full_like(alpha, cfg.tau_liquid)
        a = alpha
        for _ in range(2):
            a = (a + torch.roll(a, 1, 0) + torch.roll(a, -1, 0)) / 3.0
            a = (a + torch.roll(a, 1, 1) + torch.roll(a, -1, 1)) / 3.0
        a = torch.clamp(a, 0.0, 1.0)
        tau = cfg.tau_liquid + a * (cfg.tau_gas - cfg.tau_liquid)
        return torch.clamp(tau, min=0.52)

    # -- one step ------------------------------------------------------------
    def _collide(self) -> torch.Tensor:
        cfg = self.cfg
        rho, ux, uy = self.macroscopic()
        # the pseudopotential is evaluated on a clamped density only: a vanishing
        # density cell (gas cavitation) otherwise feeds back into psi and the
        # whole field diverges.  f itself is never clamped, so mass is exact.
        rho = torch.clamp(rho, min=1e-4, max=8.0)
        fx, fy = kernels.sc_force(
            rho,
            self.psi,
            G=cfg.G,
            gx=self.gx,
            gy=0.0,
            solid=self.solid,
            wall_psi=self.wall_psi,
            periodic_x=True,
        )
        # solid rows must not contribute a void fraction: with rho = 0 in the
        # solid, the indicator would read alpha = 1 there and the smoothing below
        # would then hand a gas relaxation time to the wall-adjacent fluid row
        alpha = torch.where(self.solid, torch.zeros_like(rho), self.void_fraction(rho))
        tau = self._tau_field(alpha).unsqueeze(0)
        rho_s = torch.clamp(rho, min=1e-12)
        feq_u = kernels.equilibrium(rho, ux, uy)
        feq_f = kernels.equilibrium(rho, ux + fx / rho_s, uy + fy / rho_s)
        f_out = self.f - (self.f - feq_u) / tau + (feq_f - feq_u)
        f_out = torch.where(self.solid.unsqueeze(0), torch.zeros_like(f_out), f_out)
        return f_out

    def step(self, n: int = 1) -> None:
        cfg = self.cfg
        for _ in range(n):
            f_post = self._collide()
            out = kernels.stream(f_post, solid=self.solid, periodic_x=True)
            self.f = out["f"]  # type: ignore[assignment]
            self.t += 1
            tau_w = float(out["wall_force_x"].sum()) / (2.0 * cfg.nx)
            self._history["tau_w"].append(tau_w)
            if self.t % cfg.control_every == 0:
                self._flow_control()
            if self.t % cfg.sample_every == 0:
                self._accumulate()
            if self.t % 100 == 0:
                self._check_finite()

    def _check_finite(self) -> None:
        # solid rows carry zero populations, so only the fluid rows can diverge
        if not torch.isfinite(self.f[:, self.fluid]).all():
            raise FloatingPointError(f"two-phase channel diverged at step {self.t}")

    def _flow_control(self) -> None:
        """Multiplicative integral control on the mass flux ``J = <rho u_x>``."""
        cfg = self.cfg
        J = self.mass_flux()
        if not math.isfinite(J) or J <= 0.0:
            return
        ratio = cfg.J_target / J
        # the drive is bounded relative to the laminar value for the same mass
        # flux: without the band the controller keeps raising g_x when a large
        # bubble chokes the channel, and the run reaches u ~ 0.5 (Mach 1) and
        # diverges -- this is what killed the d+ = 20 case at 96x97 in float32
        self.gx = float(np.clip(self.gx * (1.0 + cfg.control_gain * (ratio - 1.0)),
                                0.2 * self.gx0, 4.0 * self.gx0))
        self._history["J"].append(J)
        self._history["gx"].append(self.gx)
        self._history["alpha"].append(self.mean_void_fraction())

    # -- statistics ----------------------------------------------------------
    def _reset_accumulators(self) -> None:
        cfg = self.cfg
        self._acc = {k: torch.zeros(cfg.ny, dtype=torch.float64, device=self.device)
                     for k in ("rho", "u", "v", "rhouv", "rhov", "mu", "alpha", "uu")}
        self._n_samples = 0

    def _accumulate(self) -> None:
        cfg = self.cfg
        rho, ux, uy = self.macroscopic()
        alpha = torch.where(self.solid, torch.zeros_like(rho), self.void_fraction(rho))
        mu = rho * (self._tau_field(alpha) - 0.5) / 3.0
        avg = lambda t: t.mean(dim=1).double()
        self._acc["rho"] += avg(rho)
        self._acc["u"] += avg(ux)
        self._acc["v"] += avg(uy)
        self._acc["uu"] += avg(ux * ux)
        self._acc["rhouv"] += avg(rho * ux * uy)
        self._acc["rhov"] += avg(rho * uy)
        self._acc["mu"] += avg(mu)
        self._acc["alpha"] += avg(alpha)
        self._n_samples += 1

    def profiles(self) -> tuple[HalfChannelProfiles, dict[str, float]]:
        """Time/x-averaged half-channel profiles (lower and upper halves merged).

        The two halves are folded together (the upper half mirrored) because the
        seeded bubble layout is only *statistically* symmetric: using one half
        alone puts the wall-shear of one wall against the half-channel momentum
        balance of the other, which shows up as a spurious few-percent spread
        between the friction-velocity estimators.  ``<u'v'>`` is antisymmetric
        under the reflection, so its symmetric part is ``(lo - hi)/2``.
        """
        cfg = self.cfg
        if self._n_samples == 0:
            raise RuntimeError("no samples accumulated")
        n = float(self._n_samples)
        mean = {k: (v / n).cpu().numpy() for k, v in self._acc.items()}
        h = cfg.half_height
        nh = int(h)
        lo = np.arange(1, nh + 1)
        hi = np.arange(cfg.ny - 1 - nh, cfg.ny - 1)[::-1]
        y = lo - 0.5
        sym = lambda key, sign=1.0: 0.5 * (mean[key][lo] + sign * mean[key][hi])
        u = sym("u")
        rho = sym("rho")
        mu = sym("mu")
        alpha = sym("alpha")
        rho_uv = 0.5 * (mean["rhouv"][lo] - mean["rhouv"][hi])
        prof = HalfChannelProfiles(y=y, rho=rho, u=u, mu=mu, rho_uv=rho_uv,
                                   alpha=alpha, h=h)
        mu_liquid = self.rho_liquid * cfg.nu_liquid
        budget = wall_shear_budget(prof, mu_liquid)
        uu = 0.5 * (mean["uu"][lo] - u * u + mean["uu"][hi] - mean["u"][hi] ** 2)
        stats = {
            **budget,
            "u_bulk": float(np.trapezoid(u, y) / h),
            "mu_mean": float(np.trapezoid(mu, y) / h),
            "rho_mean": float(np.trapezoid(rho, y) / h),
            "alpha_mean_half": float(np.trapezoid(alpha, y) / h),
            "alpha_wall": float(alpha[0]),
            "uu_mean": float(np.trapezoid(uu, y) / h),
        }
        return prof, stats

    # -- run -----------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        self.step(cfg.n_steps)
        prof, stats = self.profiles()
        hist = np.asarray(self._history["tau_w"], dtype=float)
        n0 = int(cfg.warmup_fraction * len(hist))
        tau_w_exchange = float(hist[n0:].mean())
        tau_w_std = float(hist[n0:].std())
        gx_hist = np.asarray(self._history["gx"], dtype=float)
        gx_mean = float(gx_hist[len(gx_hist) // 2:].mean()) if gx_hist.size else self.gx
        # Headline wall shear = momentum balance of the half channel,
        # tau_w = g_x * int_0^h rho dy (exact in a statistically steady channel
        # and independent of the bounce-back bookkeeping).  The half-way
        # bounce-back momentum-exchange accumulator is kept as a cross-check: it
        # omits the static Shan-Chen wall-potential force and reads ~2-4 % high
        # (measured, see the report's trap list).
        nh = int(cfg.half_height)
        lo = np.arange(1, nh + 1)
        hi = np.arange(cfg.ny - 1 - nh, cfg.ny - 1)
        rho_half = 0.5 * float(self._acc["rho"][lo].mean() + self._acc["rho"][hi].mean()) / self._n_samples
        tau_w = gx_mean * rho_half * cfg.half_height
        J = float(np.mean(self._history["J"][len(self._history["J"]) // 2:]))
        alpha = self.mean_void_fraction()
        rho, _, _ = self.macroscopic()
        mass = float(rho[self.fluid].sum())
        u_tau = math.sqrt(max(tau_w / self.rho_liquid, 1e-18))
        h = cfg.half_height
        # momentum balance: tau_w must equal g_x * (mass per unit area of half channel)
        # momentum balance: tau_w = g_x * int_0^h rho dy  (per unit wall length)
        mass_half = float(rho[1:int(h) + 1, :].mean()) * h
        tau_balance = self.gx * mass_half
        out = {
            "config": {
                "nx": cfg.nx, "ny": cfg.ny, "tau_liquid": cfg.tau_liquid,
                "tau_gas": cfg.tau_gas, "G": cfg.G, "J_target": cfg.J_target,
                "d_plus_target": cfg.d_plus, "n_bubbles": int(len(self.bubble_centers)),
                "n_steps": cfg.n_steps, "viscosity_contrast": cfg.viscosity_contrast,
                "rho_liquid": self.rho_liquid, "rho_gas": self.rho_gas,
                "nu_liquid": cfg.nu_liquid, "nu_gas": cfg.nu_gas,
            },
            "tau_w": tau_w,
            "tau_w_exchange": tau_w_exchange,
            "tau_w_std": tau_w_std,
            "tau_w_balance": tau_balance,
            "u_tau": u_tau,
            "u_tau_exchange": math.sqrt(max(tau_w_exchange / self.rho_liquid, 1e-18)),
            "u_tau_gradient": math.sqrt(max(stats["tau_w_gradient"] / self.rho_liquid, 1e-18)),
            "u_tau_identity": math.sqrt(max(stats["tau_w_identity"] / self.rho_liquid, 1e-18)),
            "u_tau_balance": math.sqrt(max(tau_balance / self.rho_liquid, 1e-18)),
            "J": J,
            "gx": self.gx,
            "gx0": self.gx0,
            "gx_over_gx0": self.gx / self.gx0,
            "gx_mean": gx_mean,
            "alpha_mean": alpha,
            "alpha_integral_drift": (0.0 if self.alpha_initial < 1e-6
                                     else abs(alpha - self.alpha_initial) / self.alpha_initial),
            "alpha_linear_drift": (0.0 if self.alpha_linear_initial < 1e-6
                                   else abs(self.linear_void_fraction()
                                            - self.alpha_linear_initial) / self.alpha_linear_initial),
            "mass_drift": abs(mass - self.mass_initial) / self.mass_initial,
            "bubble_radius": self.bubble_radius,
            "bubble_diameter_lattice": 2.0 * self.bubble_radius,
            "d_plus_achieved": 2.0 * self.bubble_radius * u_tau / cfg.nu_liquid,
            "Re_tau": u_tau * h / cfg.nu_liquid,
            "Re_bulk": 2.0 * h * (J / self.rho_liquid) / cfg.nu_liquid,
            "n_bubbles_placed": int(len(self.bubble_centers)),
            "bubble_centers": self.bubble_centers.tolist(),
            "profile": {k: np.asarray(v, dtype=float).tolist() for k, v in {
                "y": prof.y, "u": prof.u, "rho": prof.rho, "mu": prof.mu,
                "rho_uv": prof.rho_uv, "alpha": prof.alpha,
            }.items()},
            "tau_w_history": hist[:: max(1, len(hist) // 400)].tolist(),
            "alpha_field": self.void_fraction(rho)[::2, ::2].cpu().numpy().tolist(),
            "u_field": torch.where(self.fluid, self.macroscopic()[1],
                                   torch.zeros_like(rho))[::2, ::2].cpu().numpy().tolist(),
        }
        out.update({k: float(v) for k, v in stats.items()})
        return out


def single_phase_reference(cfg: MicrobubbleConfig) -> dict[str, Any]:
    """Reference run: same channel, no gas, uniform liquid relaxation time."""
    ref_cfg = MicrobubbleConfig(**{
        **{f: getattr(cfg, f) for f in cfg.__dataclass_fields__},
        "d_plus": 0.0, "n_bubbles": 0, "void_fraction": 0.0,
        "viscosity_contrast": False,
    })
    return MicrobubbleChannel2D(ref_cfg).run()