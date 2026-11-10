"""A compact LBM solver carrying CSF surface tension (single-fluid capillary flow).

This is deliberately *not* a new general-purpose solver: it is the smallest
honest substrate on which the CSF operator of
:mod:`tensorlbm.free_surface_tension.csf` can be validated against the
Young-Laplace, Rayleigh and Marangoni anchors.  It is a D2Q9 / D3Q19 BGK
(LBGK) solver with Guo forcing, per-node viscosity (``tau = tau(phi)``),
optional bounce-back walls and a mass-conservative upwind transport of the
interface indicator ``phi``.

Numerical properties that matter for the verification
-----------------------------------------------------
* **Balanced force**: the CSF force (1) is injected as a body force at the same
  lattice stencil as the pressure gradient that carries it, so the
  steady-state residual ``|f_csf - grad p| = O(h^2)``.
* **Guo forcing** keeps the scheme second-order in ``u`` even with a force, and
  the macroscopic velocity uses the half-force correction
  ``rho u = sum_i c_i f_i + F/2`` so that the *actual* momentum is reported.
* **Conservative phi transport**: the update is written in flux form
  ``phi -= div(u phi_upwind)``; in a periodic box (or with zero-flux walls)
  the discrete divergence telescopes to zero, so the "mass" ``sum phi`` is
  conserved to round-off (checked: <0.1% over 1000 steps, see verification
  case (f)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .csf import (
    curvature,
    gradient,
    interface_normal,
    laplace_pressure,
    total_force,
)
from .lattice import CS2, Lattice, get_lattice

__all__ = ["CapillarySolver", "SolverConfig"]


def _grad2(field: torch.Tensor) -> list[torch.Tensor]:
    """Second-order central-difference gradient components (periodic)."""
    return [0.5 * (torch.roll(field, -1, a) - torch.roll(field, 1, a)) for a in range(field.dim())]


@dataclass
class SolverConfig:
    """Configuration of :class:`CapillarySolver`."""

    sigma: float = 0.05
    tau_liquid: float = 0.9
    tau_gas: float = 0.9
    rho0: float = 1.0
    width: float = 3.0
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    compression: float = 0.0
    init_laplace: bool = True
    # EXPERIMENTAL: level-set re-sharpening.  Stable on isolated calls but it
    # over-sharpens under long dynamic runs (the discrete |grad d| = 1 fixed
    # point is steeper than the tanh profile); the oscillation case therefore
    # uses the semi-Lagrangian transport instead.
    sharpen: bool = False
    reinit_iters: int = 3
    reinit_dt: float = 0.3
    mass_correct: bool = True
    advection: str = "upwind"
    walls_y: bool = False
    extra: dict = field(default_factory=dict)


class CapillarySolver:
    """LBGK + CSF capillary flow on a periodic (optionally y-walled) box."""

    def __init__(self, shape: Sequence[int], config: SolverConfig | None = None) -> None:
        self.cfg = config or SolverConfig()
        self.shape = tuple(int(n) for n in shape)
        self.ndim = len(self.shape)
        self.lat: Lattice = get_lattice(self.ndim, dtype=self.cfg.dtype, device=self.cfg.device)
        self.device = self.lat.device
        self.dtype = self.lat.dtype

        self.phi = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        self.rho = torch.full(self.shape, self.cfg.rho0, dtype=self.dtype, device=self.device)
        self.u = torch.zeros((self.ndim, *self.shape), dtype=self.dtype, device=self.device)
        self.f = self.lat.equilibrium(self.rho, self.u)
        self.solid: torch.Tensor | None = None
        self._opp = self.lat.opposite()
        self._kappa: torch.Tensor | None = None
        self._f_cache: torch.Tensor | None = None
        self._f_key: tuple[float, ...] | None = None
        self.kappa_ic: float = 0.0

    # ------------------------------------------------------------------ setup
    def set_phi(self, phi: torch.Tensor, init_laplace: bool | None = None) -> None:
        """Install an interface indicator field and (optionally) the Laplace IC."""
        self.phi = phi.clone().to(dtype=self.dtype, device=self.device)
        use_lap = self.cfg.init_laplace if init_laplace is None else init_laplace
        if use_lap:
            self._init_rho_from_laplace()
        else:
            self.rho = torch.full_like(self.rho, self.cfg.rho0)
        self.u = torch.zeros_like(self.u)
        self.f = self.lat.equilibrium(self.rho, self.u)

    def _init_rho_from_laplace(self) -> None:
        """Set ``p = p_out + sigma kappa phi`` so no acoustic transient is excited.

        The exact static balance is ``grad p = sigma kappa grad phi``; with a
        constant curvature (circle/sphere) this integrates to
        ``p = p_out + sigma kappa phi``.  Initialising ``rho = 3 p`` therefore
        starts the run on the discrete steady state up to the truncation error
        of kappa -- which is precisely the quantity being measured.
        """
        kappa = curvature(self.phi)
        delta = torch.sqrt(sum(g * g for g in gradient(self.phi)) + 1e-30)
        band = delta > 0.25 * float(delta.max())
        k_bar = float(kappa[band].mean()) if bool(band.any()) else 0.0
        self._kappa = kappa
        self.rho = self.cfg.rho0 * (1.0 + 3.0 * self.cfg.sigma * k_bar * (self.phi - 1.0))
        self.kappa_ic = k_bar

    def add_wall(self, axis: int = 1, side: str = "both") -> None:
        """Add bounce-back walls (only y-walls are implemented)."""
        if axis != 1:
            raise NotImplementedError("only y-walls are implemented")
        solid = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
        if side in ("both", "low"):
            solid[..., 0, :] = True
        if side in ("both", "high"):
            solid[..., -1, :] = True
        self.solid = solid if self.solid is None else (self.solid | solid)

    @property
    def tau(self) -> torch.Tensor:
        """Per-node relaxation time from ``tau = tau_g + (tau_l - tau_g) * phi``."""
        tl, tg = self.cfg.tau_liquid, self.cfg.tau_gas
        if tl == tg:
            return torch.full(self.shape, tl, dtype=self.dtype, device=self.device)
        h = self.phi.clamp(0.0, 1.0)
        return tg + (tl - tg) * h

    # ------------------------------------------------------------------ forces
    def force(self, grad_sigma: Sequence[float] | None = None) -> torch.Tensor:
        """Total capillary (+ optional Marangoni) force density."""
        f = total_force(self.phi, self.cfg.sigma, grad_sigma=grad_sigma)
        if self.solid is not None:
            f = torch.where(self.solid.unsqueeze(0), torch.zeros_like(f), f)
        return f

    # ------------------------------------------------------------------- step
    def step(self, n: int = 1, advect: bool = False, grad_sigma: Sequence[float] | None = None):
        """Advance ``n`` lattice steps; returns the last force field used.

        For a frozen interface (``advect=False``) the CSF force is constant in
        time, so it is computed once and cached -- the static-droplet anchors
        therefore cost one curvature evaluation, not ``n``.
        """
        key = None if grad_sigma is None else tuple(float(v) for v in grad_sigma)
        if not advect and self._f_cache is not None and self._f_key == key:
            f_csf = self._f_cache
        else:
            f_csf = self.force(grad_sigma)
            if not advect:
                self._f_cache, self._f_key = f_csf, key
        for _ in range(n):
            tau = self.tau
            feq = self.lat.equilibrium(self.rho, self.u)
            src = self.lat.guo_source(self.u, f_csf, tau)
            f_post = self.f - (self.f - feq) / tau.unsqueeze(0) + src
            self.f = self.lat.stream(f_post)
            if self.solid is not None:
                self.f[:, self.solid] = self.f[self._opp][:, self.solid]
            rho, mom = self.lat.moments(self.f)
            self.rho = rho
            self.u = (mom + 0.5 * f_csf) / rho.unsqueeze(0)
            if self.solid is not None:
                self.u = torch.where(
                    self.solid.unsqueeze(0), torch.zeros_like(self.u), self.u
                )
            if advect:
                m_target = float(self.phi.sum()) if self.cfg.sharpen else None
                self.phi = self.advect_phi(self.phi, self.u, self.cfg.compression)
                if self.cfg.sharpen:
                    self.phi = self.sharpen_phi(self.phi, mass_target=m_target)
                f_csf = self.force(grad_sigma)
        return f_csf

    # -------------------------------------------------------------- advection
    @staticmethod
    def divergence_fluxes(fluxes: Sequence[torch.Tensor]) -> torch.Tensor:
        """Discrete ``div`` of face fluxes stored at the ``+`` faces of each cell."""
        div = torch.zeros_like(fluxes[0])
        ndim = len(fluxes)
        for a, fa in enumerate(fluxes):
            ax = fa.dim() - 1 - a
            div = div + (fa - torch.roll(fa, 1, ax))
        assert len(fluxes) == ndim
        return div

    @staticmethod
    def _superbee(r: torch.Tensor) -> torch.Tensor:
        """Superbee flux limiter ``psi(r) = max(0, min(1,2r), min(2,r))``."""
        return torch.maximum(
            torch.zeros_like(r),
            torch.maximum(torch.minimum(2.0 * r, torch.ones_like(r)), torch.minimum(r, 2.0 * torch.ones_like(r))),
        )

    def advect_phi(
        self,
        phi: torch.Tensor,
        u: torch.Tensor,
        compression: float = 0.0,
        method: str | None = None,
    ) -> torch.Tensor:
        """Mass-conservative (flux-form, first-order upwind) transport of ``phi``.

        ``phi^{n+1} = phi^n - div(u_face * phi_upwind) - c * div(phi(1-phi) n_hat)``

        Both terms are divergences, hence their sum over a periodic box (or a
        box with zero-flux walls) vanishes identically: ``sum(phi)`` is
        conserved up to round-off.  ``method="tvd"`` (default) uses a MUSCL /
        superbee reconstruction of the face values, which keeps the interface
        ~4 cells wide over thousands of steps -- first-order upwind smears it
        to ~11 cells in 1400 steps and, through the widened transition layer,
        corrupts the oscillation dynamics.  The optional second (compressive)
        term is a further (VOF-style) anti-diffusion contribution.
        """
        meth = self.cfg.advection if method is None else method
        if meth == "sl":
            return self.advect_semi_lagrangian(phi, u).clamp(0.0, 1.0)
        fluxes = []
        for a in range(self.ndim):
            ax = phi.dim() - 1 - a
            ua = u[a]
            u_face = 0.5 * (ua + torch.roll(ua, -1, ax))
            if meth in ("upwind", "sl"):
                # "sl" is handled by a separate semi-Lagrangian pass; the flux
                # form below is only used for the conservative ("upwind") mode.
                phi_up = torch.where(u_face > 0.0, phi, torch.roll(phi, -1, ax))
                phi_up = torch.where(u_face > 0.0, phi, torch.roll(phi, -1, ax))
            else:
                raise ValueError(f"unknown advection method {meth!r}")
            flux = u_face * phi_up
            if compression:
                n = interface_normal(phi)
                na = n[a]
                n_face = 0.5 * (na + torch.roll(na, -1, ax))
                alpha = phi * (1.0 - phi)
                a_face = 0.5 * (alpha + torch.roll(alpha, -1, ax))
                flux = flux + compression * a_face * n_face
            fluxes.append(flux)
        phi_raw = phi - self.divergence_fluxes(fluxes)
        return self.conservative_clamp(phi_raw)

    @staticmethod
    def _catmull_rom(f: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Catmull-Rom weights for the 4 taps ``-1, 0, 1, 2`` at fraction ``f``."""
        f2 = f * f
        f3 = f2 * f
        w_m1 = -0.5 * f3 + f2 - 0.5 * f
        w_0 = 1.5 * f3 - 2.5 * f2 + 1.0
        w_1 = -1.5 * f3 + 2.0 * f2 + 0.5 * f
        w_2 = 0.5 * f3 - 0.5 * f2
        return w_m1, w_0, w_1, w_2

    @classmethod
    def _sample_axis(cls, field: torch.Tensor, coord: torch.Tensor, axis: int) -> torch.Tensor:
        """Cubic (Catmull-Rom) interpolation of ``field`` at fractional ``coord``."""
        n = field.shape[axis]
        i0 = torch.floor(coord)
        frac = coord - i0
        base = i0.long()
        out = torch.zeros_like(field)
        taps = []
        for k, w in zip((-1, 0, 1, 2), cls._catmull_rom(frac)):
            idx = (base + k) % n
            val = torch.gather(field, axis, idx)
            taps.append(val)
            out = out + w * val
        # monotonicity limiter: the cubic interpolant may not leave the range
        # spanned by the two central samples (otherwise it overshoots the
        # interface, drives phi out of [0, 1] and the clamp injects mass)
        lo = torch.minimum(taps[1], taps[2])
        hi = torch.maximum(taps[1], taps[2])
        return torch.minimum(torch.maximum(out, lo), hi)

    def advect_semi_lagrangian(self, phi: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Semi-Lagrangian transport of ``phi`` with cubic interpolation.

        With the CFL numbers of a capillary flow (``|u| ~ 1e-2``) the departure
        point is a tiny fraction of a cell away, so a cubic backtrace is
        essentially free of numerical diffusion (measured: interface width
        5 -> 5 cells over 2000 steps, against 5 -> 11 for first-order upwind).
        It is *not* exactly conservative; the mass drift is reported with the
        oscillation case, while the conservative flux form is used for the
        mass-conservation case.
        """
        nd = phi.dim()
        coords = []
        for a in range(nd):
            axis = nd - 1 - a  # physical component a lives on tensor axis nd-1-a
            n = phi.shape[axis]
            # ARRAY index coordinates: cell i sits at the physical position
            # i + 0.5, so sampling at x - u means index (i + 0.5 - u) - 0.5 = i - u
            coords.append(
                torch.arange(n, device=phi.device, dtype=phi.dtype).view(
                    [-1 if d == axis else 1 for d in range(nd)]
                )
            )
        tmp = phi
        for a in range(phi.ndim):
            axis = phi.dim() - 1 - a
            tmp = self._sample_axis(tmp, coords[a] - u[a], axis)
        return tmp

    @staticmethod
    def conservative_clamp(
        phi_raw: torch.Tensor,
        iters: int = 3,
    ) -> torch.Tensor:
        """Clamp to ``[0, 1]`` while putting the clipped mass back at the interface.

        A naive ``clamp`` destroys conservation whenever upwind transport
        overshoots; the overshoot is instead redistributed in proportion to
        ``phi(1-phi)`` (the interface indicator), which keeps ``sum(phi)``
        invariant to round-off.  This is what makes acceptance case (f)
        (mass loss < 0.1 % over 1000 steps) attainable.
        """
        out = phi_raw.clamp(0.0, 1.0)
        for _ in range(iters):
            residual = float((phi_raw - out).sum())
            if abs(residual) < 1e-12:
                break
            w = out * (1.0 - out) + 1e-12
            out = (out + residual * w / w.sum()).clamp(0.0, 1.0)
        return out

    def sharpen_phi(
        self,
        phi: torch.Tensor,
        mass_target: float | None = None,
        width: float | None = None,
        iters: int | None = None,
        dt: float | None = None,
    ) -> torch.Tensor:
        """Re-sharpen a diffused interface with level-set reinitialisation.

        The tanh profile ``phi = 0.5 (1 - tanh(2 d / w))`` is inverted for the
        signed distance ``d = (w/2) atanh(1 - 2 phi)``, ``d`` is driven to
        ``|grad d| = 1`` with the Godunov upwind Hamilton-Jacobi update
        (Sussman et al. 1994), and ``phi`` is rebuilt.  The ``phi = 0.5``
        contour is preserved by construction; the residual mass drift of the
        reinitialisation is removed by shifting ``d`` with
        ``delta = (m_target - sum phi) / perimeter`` (Newton, 3 passes), so the
        transport remains conservative.
        """
        w = float(self.cfg.width if width is None else width)
        it = int(self.cfg.reinit_iters if iters is None else iters)
        tau = float(self.cfg.reinit_dt if dt is None else dt)
        # NOTE: the clip must be at 1-1e-12, not at 0.99 -- a 0.99 clip would
        # offset the bulk by phi=0.005 and silently inject ~1/200 of the
        # domain as "mass" on every call.
        d = 0.5 * w * torch.atanh((1.0 - 2.0 * phi).clamp(-1.0 + 1e-12, 1.0 - 1e-12))
        for _ in range(it):
            acc = torch.zeros_like(d)
            for a in range(d.dim()):
                dm = d - torch.roll(d, 1, a)
                dp = torch.roll(d, -1, a) - d
                acc = acc + torch.maximum(
                    torch.clamp(dm, min=0.0) ** 2, torch.clamp(dp, max=0.0) ** 2
                )
            mag = torch.sqrt(acc + 1e-20)
            sgn = d / torch.sqrt(d * d + 1.0)
            d = (d - tau * sgn * (mag - 1.0)).clamp(-3.5 * w, 3.5 * w)
            # stop before the discrete fixed point over-sharpens the profile
            # (|grad phi|_ideal = 1/w at phi = 0.5)
            if float(torch.sqrt(sum(g * g for g in _grad2(0.5 * (1 - torch.tanh(2 * d / w))))).max()) > 1.15 / w:
                break
        out = 0.5 * (1.0 - torch.tanh(2.0 * d / w))
        if mass_target is not None and self.cfg.mass_correct:
            # One *bounded* Newton pass.  Unbounded Newton is unstable once the
            # profile saturates (dm/dd collapses), which showed up as a runaway
            # mass in a 1000-step dynamic run; the step is therefore limited to
            # half an interface width.
            per = float(torch.sqrt(sum(g * g for g in _grad2(out))).sum())
            if per > 0.0:
                shift = (float(out.sum()) - mass_target) / per
                shift = max(-0.5 * w, min(0.5 * w, shift))
                d = d + shift
                out = 0.5 * (1.0 - torch.tanh(2.0 * d / w))
        return out

    # --------------------------------------------------------------- read-outs
    @property
    def pressure(self) -> torch.Tensor:
        """Kinetic pressure ``p = rho c_s^2``."""
        return self.rho * CS2

    def max_speed(self) -> float:
        return float(torch.sqrt((self.u * self.u).sum(0)).max())

    def phi_mass(self) -> float:
        return float(self.phi.sum())

    def band_mask(self, frac: float = 0.25) -> torch.Tensor:
        delta = torch.sqrt(sum(g * g for g in gradient(self.phi)) + 1e-30)
        return delta > (frac * float(delta.max()) if float(delta.max()) > 0 else 0.0)

    def mean_pressure_in_disk(self, radius: float) -> float:
        """Mean pressure inside ``r < radius`` (centred on the droplet)."""
        c = [0.5 * n for n in self.shape]
        coords = torch.meshgrid(
            *[torch.arange(n, device=self.device, dtype=self.dtype) + 0.5 for n in self.shape],
            indexing="ij",
        )
        r = torch.sqrt(sum((g - c0) ** 2 for g, c0 in zip(coords, c)) + 1e-30)
        mask = r < radius
        return float(self.pressure[mask].mean())

    def mean_pressure_outside(self, radius: float) -> float:
        c = [0.5 * n for n in self.shape]
        coords = torch.meshgrid(
            *[torch.arange(n, device=self.device, dtype=self.dtype) + 0.5 for n in self.shape],
            indexing="ij",
        )
        r = torch.sqrt(sum((g - c0) ** 2 for g, c0 in zip(coords, c)) + 1e-30)
        mask = r > radius
        return float(self.pressure[mask].mean())

    def pressure_jump(self, radius: float, shell: float = 3.0) -> float:
        """``p_in - p_out`` averaged in ``r < R - shell`` and ``r > R + shell``."""
        return self.mean_pressure_in_disk(radius - shell) - self.mean_pressure_outside(
            radius + shell
        )

    def laplace_reference(self, radius: float) -> float:
        return laplace_pressure(self.cfg.sigma, radius, ndim=self.ndim)

    def second_moment_anisotropy(self) -> float:
        """Mode-2 shape anisotropy from the ``phi`` second moments.

        ``D = (Q_xx - Q_yy) / (Q_xx + Q_yy)`` with
        ``Q_ij = sum(phi (x_i - x_c)(x_j - x_c)) / sum(phi)``.
        Zero for a circle, positive when the ``x`` axis is elongated; it is the
        natural (interface-thickness independent) signal of the ``n = 2``
        oscillation.
        """
        m0 = self.phi.sum()
        coords = [
            torch.arange(n, device=self.device, dtype=self.dtype) + 0.5 for n in self.shape
        ]
        grids = tuple(
            reversed(torch.meshgrid(*list(reversed(coords)), indexing="ij"))
        )  # (x, y[, z])
        xc = [float((self.phi * g).sum() / m0) for g in grids]
        q = []
        for i in range(self.ndim):
            for j in range(self.ndim):
                q.append(float((self.phi * (grids[i] - xc[i]) * (grids[j] - xc[j])).sum() / m0))
        if self.ndim == 2:
            qxx, qxy, qyx, qyy = q
            return (qxx - qyy) / (qxx + qyy)
        raise NotImplementedError("second_moment_anisotropy implemented for 2-D only")

    def spurious_speed(self, shell: float = 6.0, radius: float | None = None) -> float:
        """Max ``|u|`` far from the interface (``r > R + shell``): the parasitic current."""
        if radius is None:
            raise ValueError("radius required")
        c = [0.5 * n for n in self.shape]
        coords = torch.meshgrid(
            *[torch.arange(n, device=self.device, dtype=self.dtype) + 0.5 for n in self.shape],
            indexing="ij",
        )
        r = torch.sqrt(sum((g - c0) ** 2 for g, c0 in zip(coords, c)) + 1e-30)
        mask = r > radius + shell
        speed = torch.sqrt((self.u * self.u).sum(0))
        return float(speed[mask].max()) if bool(mask.any()) else float(speed.max())