"""Pressure-evolution velocity LBM coupled to a conservative Allen-Cahn phase field.

Why this module exists
---------------------
TensorLBM carries Shan-Chen multiphase (:mod:`tensorlbm.multiphase`) and several
Allen-Cahn phase-field kernels that run with a *prescribed* velocity
(:mod:`tensorlbm.cac_lbm2d`, :mod:`tensorlbm.phasefield.cac_h_population`).  It
carries no **density-independent pressure** velocity solver, i.e. no
``p``-evolution formulation.  The mature open-source solver CFD-GO/``TCLB`` has
one (``models/multiphase/d2q9_pf_pressureEvolution``, file header: "Model
Developed: A. Fakhari, M. Geier, T. Lee"), and the XVAL mission *m18194* asks
whether that formulation can be adopted here.

This module is the clean-room TensorLBM counterpart of that formulation, so that
the two can be cross-validated.  **No GPL-3.0 source text is transcribed**: the
discrete scheme below is the published pressure-evolution / conservative
Allen-Cahn recipe (Fakhari & Lee; Fakhari, Geier & Lee; Chiu & Lin), and the
XVAL driver ``scripts/xval_m18194_pf_pressure_evolution.py`` *parses* the TCLB
side out of the source text at run time and compares it against this
implementation on random smooth fields (structural residual; see the report
``docs/XVAL-LBM-PRESSURE-EVOLUTION-ALLENCAHN-PHASEFIELD.md``).

The discrete scheme
-------------------
Phase field (h-population, anti-diffusion inside the equilibrium)::

    phi        = sum_i h_i
    n          = grad(phi) / |grad(phi)|
    theta      = 3 M (1 - 4 (phi - phi_avg)^2) / W
    h_i^eq     = w_i phi [1 + 3 c_i.u + 4.5 (c_i.u)^2 - 1.5 |u|^2] + theta w_i (c_i.n)
    omega_phi  = 1 / (3 M + 1/2)
    h_i       <- h_i - omega_phi (h_i - h_i^eq)

``sum_i theta w_i (c_i.n) = 0`` by lattice symmetry, so the total phase mass is
conserved *exactly* (to round-off) -- unlike an explicit counter-gradient source,
which needs the ``(1-omega/2)`` split.

Momentum (f-population carries the pressure, ``gbar``-equilibrium)::

    rho        = rho_l + (rho_h - rho_l) (phi - phi_l)/(phi_h - phi_l)
    mu         = 4 (12 sigma/W)(phi-phi_l)(phi-phi_h)(phi-phi_avg) - 1.5 sigma W lap(phi)
    u          = (3/rho) [ sum_i f_i c_i + (1/6)(mu grad(phi) + F_b) ]
    p          = sum_i f_i + (rho_h - rho_l)(grad(phi).u)/6
    Gamma_i    = w_i [1 + 3 c_i.u + 4.5 (c_i.u)^2 - 1.5 |u|^2]
    gbar_i^eq  = Gamma_i rho/3 + w_i (p - rho/3)
    I_i        = ((Gamma_i - w_i)(rho_h-rho_l)/3 + mu Gamma_i) (D_i phi - u.grad(phi))
    R_i        = f_i - gbar_i^eq + I_i/2 + F_i/2
    f_i       <- f_i - [MRT relaxation of R] + I_i + F_i

with ``F_i = Gamma_i (c_i - u).F_b``, ``MRT`` = the Lallemand-Luo moments with
**only** the two viscous moments relaxed: ``lambda = 1/(tau + 1/2)`` at the
viscous moments, ``tau = 1/omega``, ``omega = 1/(3 nu)`` interpolated linearly in
``phi``.  ``D_i phi`` is the *centred* directional difference in the
pre-collision half and the *mixed (upwind)* 4-point difference in the
post-collision half -- the reference source notes that the mixed stencil is what
recovers the momentum equation at the interface.  The h-collision uses the
velocity of the *already collided* f-population (the reference's stage order).

The pressure evolution equation behind this is ``d_t p + rho cs^2 div(u) = 0``,
so ``p`` is *not* tied to ``rho`` by an equation of state: that is what makes the
pressure field density-ratio independent, the property cross-validated in the
XVAL report.

Boundaries: periodic (pull streaming).  The Solid/Wall/EVelocity node types of the
reference model are not ported in round 1 (see report section 6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .cac_h_population import (
    Stencil,
    _neighbours,
    _shift,
    get_stencil,
    isotropic_gradient,
    isotropic_laplacian,
)

__all__ = [
    "D2Q9_MOMENTS",
    "VISCOUS_MOMENTS",
    "PFPressureEvolution",
    "chem_potential",
    "directional_centred",
    "directional_mixed",
    "gbar_equilibrium",
    "interface_term",
    "laplace_pressure_2d",
    "maxwellian",
    "mrt_relax",
    "pressure_from_populations",
    "taylor_green_amplitude",
    "velocity",
    "viscous_lambda",
]

# ---------------------------------------------------------------------------
# lattice / moment constants
# ---------------------------------------------------------------------------

#: Lallemand-Luo D2Q9 moment rows, ordered exactly as the reference model lists
#: them (density, energy, energy^2, x-momentum, x-flux, y-momentum, y-flux,
#: xy-stress, ghost).  Only rows 7 and 8 carry a viscosity (``p_xx``, ``p_xy``).
D2Q9_MOMENTS: tuple[tuple[int, ...], ...] = (
    (1, 1, 1, 1, 1, 1, 1, 1, 1),
    (-4, -1, -1, -1, -1, 2, 2, 2, 2),
    (4, -2, -2, -2, -2, 1, 1, 1, 1),
    (0, 1, 0, -1, 0, 1, -1, -1, 1),
    (0, -2, 0, 2, 0, 1, -1, -1, 1),
    (0, 0, 1, 0, -1, 1, 1, -1, -1),
    (0, 0, -2, 0, 2, 1, 1, -1, -1),
    (0, 1, -1, 1, -1, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, -1, 1, -1),
)

#: 0-based indices of the moments that carry the kinematic viscosity (``p_xx``,
#: ``p_xy``); the reference relaxes exactly ``S[7]``/``S[8]`` of its 0-based moment
#: vector, i.e. the same two rows.
VISCOUS_MOMENTS: tuple[int, int] = (7, 8)


# ---------------------------------------------------------------------------
# extracted-formula kernels (module level so they can be tested in isolation)
# ---------------------------------------------------------------------------


def _wview(st: Stencil, ndim: int, *, dtype, device) -> torch.Tensor:
    return st.W.to(device=device, dtype=dtype).view(st.q, *([1] * ndim))


def _cdot(st: Stencil, vec: tuple[torch.Tensor, ...], ndim: int, *, dtype, device):
    """``c_i . vec`` as a ``(Q, *shape)`` tensor (one einsum, no temporaries)."""
    c = st.C.to(device=device, dtype=dtype)
    stack = torch.stack(vec, dim=0)
    return torch.einsum("qa,a...->q...", c, stack)


def _opposite_index(st: Stencil) -> list[int]:
    key = {tuple(int(v) for v in row): i for i, row in enumerate(st.C.tolist())}
    return [key[tuple(-v for v in row)] for row in st.C.tolist()]


def maxwellian(
    st: Stencil, u: tuple[torch.Tensor, ...], ndim: int, *, dtype, device
) -> torch.Tensor:
    """``Gamma_i = w_i [1 + 3 c_i.u + 4.5 (c_i.u)^2 - 1.5 |u|^2]``."""
    w = _wview(st, ndim, dtype=dtype, device=device)
    cdotu = _cdot(st, u, ndim, dtype=dtype, device=device)
    u2 = torch.zeros_like(u[0])
    for a in range(st.dim):
        u2 = u2 + u[a] * u[a]
    return w * (1.0 + 3.0 * cdotu + 4.5 * cdotu * cdotu - 1.5 * u2.unsqueeze(0))


def gbar_equilibrium(
    gamma: torch.Tensor,
    p: torch.Tensor,
    rho: torch.Tensor,
    st: Stencil,
) -> torch.Tensor:
    """``gbar_i^eq = Gamma_i rho/3 + w_i (p - rho/3)``."""
    ndim = rho.dim()
    w = st.W.to(device=rho.device, dtype=rho.dtype).view(st.q, *([1] * ndim))
    return gamma * rho.unsqueeze(0) / 3.0 + w * (p - rho / 3.0).unsqueeze(0)


def directional_centred(
    phi: torch.Tensor, st: Stencil, nb: torch.Tensor | None = None
) -> torch.Tensor:
    """``Dc_i phi = [phi(x+c_i) - phi(x-c_i)]/2`` (``0`` for ``c_i = 0``)."""
    nb = _neighbours(phi, st) if nb is None else nb
    opp = _opposite_index(st)
    out = torch.zeros_like(nb)
    for q in range(st.q):
        out[q] = 0.5 * (nb[q] - nb[opp[q]])
    return out


def directional_mixed(
    phi: torch.Tensor, st: Stencil, nb: torch.Tensor | None = None
) -> torch.Tensor:
    """``Dm_i phi = [-phi(x+2c_i) + 5 phi(x+c_i) - 3 phi - phi(x-c_i)]/4``.

    The mixed (upwind-biased) directional derivative of the reference model: it is
    second-order accurate along ``c_i``, and the reference uses it in the
    post-collision half of the interface force.
    """
    nb = _neighbours(phi, st) if nb is None else nb
    opp = _opposite_index(st)
    out = torch.zeros_like(nb)
    for q in range(st.q):
        if not bool(st.C[q].any()):
            continue
        c = tuple(int(v) for v in st.C[q].tolist())
        p2 = _shift(phi, tuple(2 * v for v in c))
        out[q] = 0.25 * (-p2 + 5.0 * nb[q] - 3.0 * phi - nb[opp[q]])
    return out


def interface_term(
    gamma: torch.Tensor,
    dphi: torch.Tensor,
    mu: torch.Tensor,
    u: tuple[torch.Tensor, ...],
    grad: tuple[torch.Tensor, ...],
    rho_h: float,
    rho_l: float,
    st: Stencil,
) -> torch.Tensor:
    """``((Gamma_i - w_i)(rho_h-rho_l)/3 + mu Gamma_i)(D_i phi - u.grad(phi))``."""
    ndim = mu.dim()
    w = _wview(st, ndim, dtype=mu.dtype, device=mu.device)
    u_dot_grad = torch.zeros_like(mu)
    for a in range(st.dim):
        u_dot_grad = u_dot_grad + u[a] * grad[a]
    coef = (gamma - w) * ((rho_h - rho_l) / 3.0) + mu.unsqueeze(0) * gamma
    return coef * (dphi - u_dot_grad.unsqueeze(0))


def body_term(
    gamma: torch.Tensor,
    u: tuple[torch.Tensor, ...],
    body: tuple[torch.Tensor, ...],
    st: Stencil,
) -> torch.Tensor:
    """``Gamma_i (c_i - u).F_b``."""
    ndim = u[0].dim()
    cdotf = _cdot(st, body, ndim, dtype=u[0].dtype, device=u[0].device)
    udotf = torch.zeros_like(u[0])
    for a in range(len(u)):
        udotf = udotf + u[a] * body[a]
    return gamma * (cdotf - udotf.unsqueeze(0))


def mrt_relax(
    r: torch.Tensor,
    lambda_: torch.Tensor,
    moments: torch.Tensor,
    moments_inv: torch.Tensor,
) -> torch.Tensor:
    """``M^{-1} diag(lambda) M r`` for a ``(Q, *shape)`` population residual.

    ``lambda_`` is either ``(Q,)`` (uniform relaxation) or ``(Q, *shape)`` (a
    relax rate per node, as required by a phase-dependent viscosity).
    """
    s = torch.tensordot(moments, r, dims=([1], [0]))  # (Q, *shape)
    if lambda_.dim() == 1:
        s = s * lambda_.view(-1, *([1] * (r.dim() - 1)))
    else:
        s = s * lambda_
    return torch.tensordot(moments_inv, s, dims=([1], [0]))


def viscous_lambda(
    relax: torch.Tensor, st: Stencil, *, dtype=None, device=None
) -> torch.Tensor:
    """``lambda`` with ``1`` everywhere except the two viscous moments."""
    lam = torch.ones(
        (st.q,) + tuple(relax.shape),
        dtype=relax.dtype if dtype is None else dtype,
        device=relax.device if device is None else device,
    )
    for idx in VISCOUS_MOMENTS:
        lam[idx] = relax
    return lam


def pressure_from_populations(
    fsum: torch.Tensor,
    grad: tuple[torch.Tensor, ...],
    u: tuple[torch.Tensor, ...],
    rho_h: float,
    rho_l: float,
) -> torch.Tensor:
    """``p = sum_i f_i + (rho_h-rho_l)(grad(phi).u)/6``."""
    return fsum + (rho_h - rho_l) * (grad[0] * u[0] + grad[1] * u[1]) / 6.0


def velocity(
    mom: tuple[torch.Tensor, ...],
    rho: torch.Tensor,
    mu: torch.Tensor,
    grad: tuple[torch.Tensor, ...],
    body: tuple[torch.Tensor, ...] | None,
) -> tuple[torch.Tensor, ...]:
    """``u = (3/rho)[sum_i f_i c_i + (1/6)(mu grad(phi) + F_b)]``."""
    out = []
    for a in range(len(mom)):
        flux = mu * grad[a]
        if body is not None:
            flux = flux + body[a]
        out.append(3.0 * (mom[a] + flux / 6.0) / rho)
    return tuple(out)


def chem_potential(
    phi: torch.Tensor,
    st: Stencil,
    *,
    sigma: float,
    width: float,
    phi_h: float = 1.0,
    phi_l: float = 0.0,
    nb: torch.Tensor | None = None,
) -> torch.Tensor:
    """``mu = 4(12 sigma/W)(phi-phi_l)(phi-phi_h)(phi-phi_avg) - 1.5 sigma W lap(phi)``.

    The coefficients follow from the double-well free energy density
    ``beta phi^2 (1-phi)^2`` with ``beta = 12 sigma/W`` and ``kappa = 1.5 sigma W``:
    ``sigma = sqrt(2 kappa beta)/6`` reproduces the input ``sigma`` exactly, so a
    static diffuse drop must satisfy the 2D Young-Laplace law ``dp = sigma/R``.
    """
    lap = isotropic_laplacian(phi, st, nb)
    avg = 0.5 * (phi_h + phi_l)
    return (
        4.0 * (12.0 * sigma / width) * (phi - phi_l) * (phi - phi_h) * (phi - avg)
        - 1.5 * sigma * width * lap
    )


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


@dataclass
class PFPressureEvolution:
    """Pressure-evolution f-population + conservative Allen-Cahn h-population.

    Parameters
    ----------
    shape:
        ``(ny, nx)`` grid.
    rho_h, rho_l:
        Heavy / light density.  ``rho`` is an *input* of the phase field, not the
        result of an equation of state -- the point of the formulation.
    sigma, width, mobility:
        Surface tension, anti-diffusion coefficient ``W``, mobility ``M``.
    viscosity_h, viscosity_l:
        Kinematic viscosity per phase; ``omega = 1/(3 nu)``, interpolated in
        ``phi`` like the density (the reference's choice).
    """

    shape: tuple[int, int]
    rho_h: float = 1.0
    rho_l: float = 1.0
    sigma: float = 0.05
    width: float = 4.0
    mobility: float = 0.05
    viscosity_h: float = 0.1
    viscosity_l: float = 0.1
    phi_h: float = 1.0
    phi_l: float = 0.0
    dtype: torch.dtype = torch.float64
    device: torch.device | str = "cpu"
    min_gradient: float = 0.0
    stencil_name: str = "d2q9"
    st: Stencil = field(init=False, repr=False)
    f: torch.Tensor = field(init=False, repr=False)
    h: torch.Tensor = field(init=False, repr=False)
    omega_phi: float = field(init=False)
    _moments: torch.Tensor = field(init=False, repr=False)
    _moments_inv: torch.Tensor = field(init=False, repr=False)
    _opp: list = field(init=False, repr=False)
    step_count: int = field(init=False, default=0)
    mass0: float = field(init=False, default=float("nan"))

    def __post_init__(self) -> None:
        self.st = get_stencil(self.stencil_name)
        if self.st.dim != 2:
            raise ValueError("PFPressureEvolution is the 2D (d2q9) port")
        self.omega_phi = 1.0 / (3.0 * self.mobility + 0.5)
        self._opp = _opposite_index(self.st)
        m = torch.tensor(D2Q9_MOMENTS, dtype=self.dtype, device=self.device)
        self._moments = m
        self._moments_inv = torch.linalg.inv(m)
        self.f = torch.zeros((self.st.q,) + tuple(self.shape), dtype=self.dtype,
                             device=self.device)
        self.h = torch.zeros_like(self.f)

    # -- derived fields -------------------------------------------------
    @property
    def phi(self) -> torch.Tensor:
        return self.h.sum(dim=0)

    @property
    def delta_rho(self) -> float:
        return float(self.rho_h - self.rho_l)

    @property
    def _c(self) -> torch.Tensor:
        return self.st.C.to(dtype=self.dtype, device=self.device)

    def _phase_fraction(self, phi: torch.Tensor) -> torch.Tensor:
        den = self.phi_h - self.phi_l
        if abs(den) < 1e-14:
            return torch.zeros_like(phi)
        return (phi - self.phi_l) / den

    def density(self, phi: torch.Tensor | None = None) -> torch.Tensor:
        phi = self.phi if phi is None else phi
        return self.rho_l + self.delta_rho * self._phase_fraction(phi)

    def omega(self, phi: torch.Tensor | None = None) -> torch.Tensor:
        """``omega = 1/(3 nu)``, linear in ``phi`` between the two phases."""
        phi = self.phi if phi is None else phi
        w_l = 1.0 / (3.0 * self.viscosity_l)
        w_h = 1.0 / (3.0 * self.viscosity_h)
        return w_l + (w_h - w_l) * self._phase_fraction(phi)

    def kinetics(self, phi: torch.Tensor | None = None):
        """``phi``-derived fields the collision needs: neighbours, grad, mu."""
        phi = self.phi if phi is None else phi
        nb = _neighbours(phi, self.st)
        grad = isotropic_gradient(phi, self.st, nb)
        mu = chem_potential(
            phi, self.st, sigma=self.sigma, width=self.width,
            phi_h=self.phi_h, phi_l=self.phi_l, nb=nb,
        )
        return phi, nb, grad, mu

    def momentum(self) -> tuple[torch.Tensor, ...]:
        c = self._c
        return tuple(
            (self.f * c[:, a].view(-1, *([1] * (self.f.dim() - 1)))).sum(0) for a in range(2)
        )

    def velocity(
        self, phi: torch.Tensor | None = None, body: tuple[torch.Tensor, ...] | None = None,
        *, grad: tuple[torch.Tensor, ...] | None = None, mu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        phi = self.phi if phi is None else phi
        if grad is None or mu is None:
            _, _, grad, mu = self.kinetics(phi)
        return velocity(self.momentum(), self.density(phi), mu, grad, body)

    def pressure(
        self, phi: torch.Tensor | None = None, body: tuple[torch.Tensor, ...] | None = None
    ) -> torch.Tensor:
        phi = self.phi if phi is None else phi
        _, _, grad, mu = self.kinetics(phi)
        u = self.velocity(phi, body, grad=grad, mu=mu)
        return pressure_from_populations(self.f.sum(0), grad, u, self.rho_h, self.rho_l)

    # -- initialization -------------------------------------------------
    def initialize(
        self, phi0: torch.Tensor, u0: tuple[torch.Tensor, ...] | None = None,
        body: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        """Reference initialisation: ``h = h^eq``, ``f = -I/2 - F/2`` (``p = 0``)."""
        phi0 = phi0.to(dtype=self.dtype, device=self.device)
        if tuple(phi0.shape) != tuple(self.shape):
            raise ValueError(f"phi0 shape {tuple(phi0.shape)} != {self.shape}")
        self.h = torch.zeros((self.st.q,) + tuple(phi0.shape), dtype=self.dtype,
                             device=self.device)
        self.f = torch.zeros_like(self.h)
        self.step_count = 0
        self.mass0 = float(phi0.sum().item())

        ndim = phi0.dim()
        nb = _neighbours(phi0, self.st)
        grad = isotropic_gradient(phi0, self.st, nb)
        mu = chem_potential(
            phi0, self.st, sigma=self.sigma, width=self.width,
            phi_h=self.phi_h, phi_l=self.phi_l, nb=nb,
        )
        if u0 is None:
            u0 = tuple(torch.zeros_like(phi0) for _ in range(2))
        gamma = maxwellian(self.st, u0, ndim, dtype=self.dtype, device=self.device)
        mag = torch.zeros_like(phi0)
        for g in grad:
            mag = mag + g * g
        mag = mag.sqrt()
        safe = torch.where(mag > 0.0, mag, torch.ones_like(mag))
        n = tuple(g / safe for g in grad)
        cdotn = _cdot(self.st, n, ndim, dtype=self.dtype, device=self.device)
        w = _wview(self.st, ndim, dtype=self.dtype, device=self.device)
        avg = 0.5 * (self.phi_h + self.phi_l)
        theta = 3.0 * self.mobility * (1.0 - 4.0 * (phi0 - avg) ** 2) / self.width
        self.h = gamma * phi0.unsqueeze(0) + theta.unsqueeze(0) * w * cdotn

        dc = directional_centred(phi0, self.st, nb)
        int_pre = interface_term(gamma, dc, mu, u0, grad, self.rho_h, self.rho_l, self.st)
        if body is not None:
            self.f = -0.5 * int_pre - 0.5 * body_term(gamma, u0, body, self.st)
        else:
            self.f = -0.5 * int_pre

    @staticmethod
    def tanh_drop(
        shape: tuple[int, int], radius: float, width: float,
        center: tuple[float, float] | None = None, inside: bool = True,
    ) -> torch.Tensor:
        """``phi = 1/2 [1 -/+ tanh(2 (r - R)/W)]`` (the reference model's init)."""
        ny, nx = shape
        cx, cy = center if center is not None else (0.5 * (nx - 1), 0.5 * (ny - 1))
        y, x = torch.meshgrid(
            torch.arange(ny, dtype=torch.float64),
            torch.arange(nx, dtype=torch.float64),
            indexing="ij",
        )
        r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        s = torch.tanh(2.0 * (r - radius) / width)
        return 0.5 * (1.0 + s) if not inside else 0.5 * (1.0 - s)

    # -- one LBE step ---------------------------------------------------
    def collision(self, body: tuple[torch.Tensor, ...] | None = None) -> None:
        """Fluid collision first, then the phase collision with the new velocity.

        This mirrors the reference stage order (``f`` is overwritten in the fluid
        collision before the h-equilibrium is evaluated).
        """
        phi, nb, grad, mu = self.kinetics()
        rho = self.density(phi)
        self._fluid_collision(phi, nb, grad, mu, rho, body)
        self._phase_collision(phi, grad)

    def _fluid_collision(self, phi, nb, grad, mu, rho, body):
        ndim = phi.dim()
        u = velocity(self.momentum(), rho, mu, grad, body)
        p = pressure_from_populations(self.f.sum(0), grad, u, self.rho_h, self.rho_l)

        gamma = maxwellian(self.st, u, ndim, dtype=self.dtype, device=self.device)
        gbar_eq = gbar_equilibrium(gamma, p, rho, self.st)

        dc = directional_centred(phi, self.st, nb)
        dm = directional_mixed(phi, self.st, nb)
        int_pre = interface_term(gamma, dc, mu, u, grad, self.rho_h, self.rho_l, self.st)
        int_post = interface_term(gamma, dm, mu, u, grad, self.rho_h, self.rho_l, self.st)
        if body is not None:
            fb = body_term(gamma, u, body, self.st)
            r = self.f - gbar_eq + 0.5 * int_pre + 0.5 * fb
        else:
            fb = 0.0
            r = self.f - gbar_eq + 0.5 * int_pre

        tau = 1.0 / self.omega(phi)
        lam = viscous_lambda(1.0 / (tau + 0.5), self.st)
        r_relaxed = mrt_relax(r, lam, self._moments, self._moments_inv)
        self.f = self.f - r_relaxed + int_post + fb

    def _phase_collision(self, phi, grad):
        ndim = phi.dim()
        mag = torch.zeros_like(phi)
        for g in grad:
            mag = mag + g * g
        mag = mag.sqrt()
        safe = torch.where(mag > max(self.min_gradient, 0.0), mag, torch.ones_like(mag))
        n = tuple(g / safe for g in grad)
        u = self.velocity(phi, grad=grad)
        gamma = maxwellian(self.st, u, ndim, dtype=self.dtype, device=self.device)
        cdotn = _cdot(self.st, n, ndim, dtype=self.dtype, device=self.device)
        w = _wview(self.st, ndim, dtype=self.dtype, device=self.device)
        avg = 0.5 * (self.phi_h + self.phi_l)
        theta = 3.0 * self.mobility * (1.0 - 4.0 * (phi - avg) ** 2) / self.width
        h_eq = gamma * phi.unsqueeze(0) + theta.unsqueeze(0) * w * cdotn
        self.h = self.h - self.omega_phi * (self.h - h_eq)

    def stream(self) -> None:
        """Periodic pull streaming for both populations."""
        out_f = torch.empty_like(self.f)
        out_h = torch.empty_like(self.h)
        for q in range(self.st.q):
            off = tuple(-int(v) for v in self.st.C[q].tolist())
            out_f[q] = _shift(self.f[q], off)
            out_h[q] = _shift(self.h[q], off)
        self.f, self.h = out_f, out_h

    def step(self, body: tuple[torch.Tensor, ...] | None = None) -> None:
        self.collision(body)
        self.stream()
        self.step_count += 1

    def run(self, n_steps: int, body: tuple[torch.Tensor, ...] | None = None) -> None:
        for _ in range(int(n_steps)):
            self.step(body)

    # -- diagnostics ----------------------------------------------------
    def mass(self) -> float:
        return float(self.phi.sum().item())

    def mass_error(self) -> float:
        m = self.mass()
        return abs(m - self.mass0) / abs(self.mass0) if self.mass0 else float("nan")

    def divergence(self, u: tuple[torch.Tensor, ...] | None = None) -> torch.Tensor:
        u = self.velocity() if u is None else u
        d = torch.zeros_like(u[0])
        for a in range(2):
            d = d + isotropic_gradient(u[a], self.st)[a]
        return d


# ---------------------------------------------------------------------------
# analytic references
# ---------------------------------------------------------------------------


def laplace_pressure_2d(sigma: float, radius: float) -> float:
    """2D Young-Laplace: ``dp = sigma/R``."""
    return float(sigma) / float(radius)


def taylor_green_amplitude(u0: float, nu: float, k: float, t: float) -> float:
    """Exact viscous decay of ``u_x = U sin(k y)``: ``U exp(-nu k^2 t)``."""
    return float(u0) * math.exp(-float(nu) * float(k) ** 2 * float(t))