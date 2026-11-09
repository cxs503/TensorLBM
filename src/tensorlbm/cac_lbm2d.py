"""Two-dimensional (D2Q9) conservative Allen-Cahn phase-field lattice Boltzmann.

This is the D2Q9 sibling of :mod:`tensorlbm.cac_lbm` (the D3Q19 CAC/HCH scheme,
W9-R2, equations from Liang, Xu, Chen, Wang, Chai & Shi, PRE 97, 033309 (2018)).
The *equations* are identical; only the velocity set, the isotropic stencils and
the array rank change:

* h-LBE (Allen-Cahn / phase field) ``f_i`` with ``sum_i f_i = phi``.
* g-LBE (Hua-Hu-Chun incompressible hydrodynamics) ``g_i``.
* mixing laws ``rho = phi (rho_l - rho_g) + rho_g``, harmonic or linear viscosity.
* chemical potential ``mu_phi = 4 beta phi (phi-1)(phi-0.5) - k lap(phi)`` with
  ``beta = 12 sigma / W`` and ``k = 1.5 sigma W`` (so that the flat-interface
  solution is ``phi = 0.5 (1 + tanh(2 (y0-y)/W))`` and the sharp-interface
  tension is ``sigma``).
* body force ``F = mu_phi grad(phi) + rho g`` (``gravity_mode="rho"``, the
  physical buoyancy form) or ``+ g`` (``gravity_mode="uniform"``).

Wetting (gas-liquid-solid) walls
--------------------------------
Solid nodes carry a ghost phase-field value.  Three policies are provided:

``"freeze"``    keep whatever bounce-back leaves there (default, no wetting).
``"neumann"``   zero normal gradient, i.e. the neutral 90 deg contact angle.
``"contact"``   Young wall condition for a prescribed contact angle.  With the
                linear wall energy ``f_w = -h phi`` the natural boundary
                condition is ``k dphi/dn = h = sigma cos(theta)`` (n = solid ->
                fluid), i.e. ``dphi/dn = 2 cos(theta) / (3 W)``.  The ghost
                value is set to ``phi_fluid - dphi/dn`` (unit node spacing,
                bounce-back wall half way between the nodes).

The ``"contact"`` policy is verified in ``tensorlbm.wetting`` (IN32-L01
ladder); it is opt-in and bit-identical to the D3Q19 scheme at
``contact_angle_deg = 90`` (the Neumann limit).

This module deliberately owns no validation claims: the acceptance numbers
live in ``benchmarks/in32_l01_wetting_ladder`` and ``docs/IN32_L01_WETTING_LADDER.md``.
"""

from __future__ import annotations

import math

import torch

from .d2q9 import C, OPPOSITE, W

CS2 = 1.0 / 3.0

WALL_POLICIES = ("freeze", "neumann", "contact")


def _w(dtype, device) -> torch.Tensor:
    return W.to(device=device, dtype=dtype)


def _c(dtype, device) -> torch.Tensor:
    return C.to(device=device, dtype=dtype)


def wall_contact_flux_from_angle(contact_angle_deg: float, width: float) -> float:
    """Continuum Young wall flux ``dphi/dn = h/k = 2 cos(theta) / (3 W)``.

    ``sigma cos(theta) = h`` (linear wall energy) and ``k = 1.5 sigma W``.
    ``theta = 90`` deg returns 0 (neutral/Neumann).  This constant is the
    *sharp-interface* limit; the discrete ghost condition used by
    :class:`CAC2D` multiplies it by the local ``4 phi (1-phi)`` profile factor
    (see :func:`wall_contact_ghost`) so that the wall exchanges no phase field
    deep inside either bulk phase.
    """
    return -(2.0 / 3.0) * math.cos(math.radians(float(contact_angle_deg))) / float(width)


def wall_contact_ghost(phi_fluid: torch.Tensor, cos_theta: float, width: float) -> torch.Tensor:
    """Ghost phase-field value realising a Young contact angle on a wall.

    Diffuse-interface geometry: at the wall the interface normal ``n_i`` (liquid
    -> gas) makes the contact angle ``theta`` with the wall normal ``n_w``
    (solid -> fluid), and the equilibrium profile satisfies
    ``|grad phi| = 4 phi (1 - phi) / W``.  Hence

        dphi/dn_w = -cos(theta) |grad phi| = -cos(theta) 4 phi (1-phi) / W,

    and the ghost node one spacing behind the half-way wall carries
    ``phi_solid = phi_fluid - cos(theta) 4 phi (1-phi) / W`` (sign fixed by
    the measured calibration: the symmetric wetting case must bring the
g    apparent angle below 90 deg -- see the ladder report).  The
    ``4 phi (1-phi)`` factor makes the wall flux vanish in both bulk phases
    (constant-flux ghosts act as an infinite phase-field reservoir and pump
    mass -- measured, see docs/IN32_L01_WETTING_LADDER.md).  ``theta = 90`` deg
    reduces to the neutral Neumann condition.
    """
    f = phi_fluid.clamp(0.0, 1.0)
    return phi_fluid + cos_theta * 4.0 * f * (1.0 - f) / width


# ---------------------------------------------------------------------------
# Stencils (second-order isotropic, D2Q9)
# ---------------------------------------------------------------------------


def _shift(f: torch.Tensor, dy: int, dx: int, axes: str = "pp") -> torch.Tensor:
    """Return g with ``g[y, x] = f[y + dy, x + dx]``.

    ``axes`` is a 2-char policy for (y, x): ``'p'`` periodic roll, ``'w'``
    non-wrapping zero fill (the zero-filled halo strands only on wall nodes,
    which are overwritten by the bounce-back swap).
    """
    if dy == 0 and dx == 0:
        return f
    if axes[0] == "p":
        f = torch.roll(f, shifts=(-dy, -dx), dims=(0, 1))
        return f
    ny, nx = f.shape
    out = torch.zeros_like(f)
    ys = slice(max(dy, 0), ny + min(dy, 0))
    yd = slice(max(-dy, 0), ny + min(-dy, 0))
    xs = slice(max(dx, 0), nx + min(dx, 0))
    xd = slice(max(-dx, 0), nx + min(-dx, 0))
    if ys.stop - ys.start <= 0 or xs.stop - xs.start <= 0:
        return out
    out[yd, xd] = f[ys, xs]
    return out


def shifted_neighbors(phi: torch.Tensor, axes: str = "pp") -> torch.Tensor:
    """Stack (8, ny, nx) of ``phi(x + c_i)``, i = 1..8."""
    c = _c(phi.dtype, phi.device)
    out = torch.empty((8, *phi.shape), dtype=phi.dtype, device=phi.device)
    for q in range(1, 9):
        dx, dy = int(c[q, 0].item()), int(c[q, 1].item())
        out[q - 1] = _shift(phi, dy, dx, axes)
    return out


def iso_gradient_2d(
    phi: torch.Tensor, axes: str = "pp", nb: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``grad(phi)`` = 3 sum_{i!=0} w_i c_i phi(x + c_i)  (second order, isotropic).

    ``nb`` (optional) is :func:`shifted_neighbors` output, shared with the
    Laplacian to halve the stencil cost.
    """
    if nb is None:
        nb = shifted_neighbors(phi, axes=axes)
    c = _c(phi.dtype, phi.device)
    w = _w(phi.dtype, phi.device)
    gx = torch.zeros_like(phi)
    gy = torch.zeros_like(phi)
    for q in range(1, 9):
        dx, dy = int(c[q, 0].item()), int(c[q, 1].item())
        term = nb[q - 1] * (3.0 * float(w[q]))
        gx = gx + term * dx
        gy = gy + term * dy
    return gx, gy


def iso_laplacian_2d(phi: torch.Tensor, axes: str = "pp", nb: torch.Tensor | None = None) -> torch.Tensor:
    """``lap(phi)`` = 6 sum_i w_i [phi(x + c_i) - phi(x)]  (second order, isotropic)."""
    if nb is None:
        nb = shifted_neighbors(phi, axes=axes)
    w = _w(phi.dtype, phi.device)
    acc = torch.zeros_like(phi)
    for q in range(1, 9):
        acc = acc + float(w[q]) * nb[q - 1]
    return 6.0 * (acc - (1.0 - float(w[0])) * phi)


# ---------------------------------------------------------------------------
# Free energy, chemical potential, mixing laws
# ---------------------------------------------------------------------------


def rho_of_phi(phi: torch.Tensor, rho_l: float, rho_g: float) -> torch.Tensor:
    return phi * (rho_l - rho_g) + rho_g


def free_energy_params(sigma: float, width: float) -> tuple[float, float]:
    return 12.0 * sigma / width, 1.5 * sigma * width


def viscosity_of_phi(phi, rho_l, rho_g, mu_l, mu_g, law="harmonic"):
    nu_l, nu_g = mu_l / rho_l, mu_g / rho_g
    if law == "linear":
        nu = phi * (nu_l - nu_g) + nu_g
    elif law == "harmonic":
        nu = 1.0 / (phi * (1.0 / nu_l - 1.0 / nu_g) + 1.0 / nu_g)
    else:
        raise ValueError(f"unknown mixing law {law!r}")
    return nu * rho_of_phi(phi, rho_l, rho_g)


def chemical_potential(phi, lap, beta, k):
    return 4.0 * beta * phi * (phi - 1.0) * (phi - 0.5) - k * lap


def free_energy_density(phi, grad_x, grad_y, beta, k):
    """``f = beta phi^2 (phi-1)^2 + (k/2)|grad phi|^2`` (used by the tension integral)."""
    return beta * phi * phi * (phi - 1.0) ** 2 + 0.5 * k * (grad_x**2 + grad_y**2)


# ---------------------------------------------------------------------------
# LBE pieces
# ---------------------------------------------------------------------------


def _views(dtype, device):
    c = _c(dtype, device)
    w = _w(dtype, device)
    return w.view(9, 1, 1), [c[:, 0].view(9, 1, 1), c[:, 1].view(9, 1, 1)]


def f_equilibrium(phi, u, wv, cf):
    cu = cf[0] * u[0] + cf[1] * u[1]
    return wv * phi * (1.0 + 3.0 * cu)


def f_source(phi_u_dt, lam, n_hat, tau_f, wv, cf):
    coef = 1.0 - 0.5 / tau_f
    b0 = phi_u_dt[0] + (lam / 3.0) * n_hat[0]
    b1 = phi_u_dt[1] + (lam / 3.0) * n_hat[1]
    return coef * 3.0 * wv * (cf[0] * b0 + cf[1] * b1)


def lambda_phi(phi, width):
    return 4.0 * phi * (1.0 - phi) / width


def s_hermite(u, wv, cf):
    cu = cf[0] * u[0] + cf[1] * u[1]
    u2 = u[0] * u[0] + u[1] * u[1]
    return wv * (3.0 * cu + 4.5 * cu * cu - 1.5 * u2)


def g_equilibrium(p, rho, u, wv, cf, w0):
    s = s_hermite(u, wv, cf)
    geq = (p / CS2) * wv + rho * s
    geq = torch.cat((((p / CS2) * (w0 - 1.0) + rho * s[0]).unsqueeze(0), geq[1:]), dim=0)
    return geq


def g_force(F, u, grad_phi, drho, inv_tau_g, wv, cf):
    coef = 1.0 - 0.5 * inv_tau_g
    cu = cf[0] * u[0] + cf[1] * u[1]
    cgr = cf[0] * grad_phi[0] + cf[1] * grad_phi[1]
    cF = cf[0] * F[0] + cf[1] * F[1]
    return coef * wv * (3.0 * cF + 3.0 * cu * cgr * drho)


def macroscopic_g(g, rho, grad_phi, drho, F, w0):
    """``rho u = sum c g + 0.5 F``; ``p = cs2/(1-w0)[sum_{i!=0} g + u.grad(rho) + rho s_0]``."""
    _, cf = _views(g.dtype, g.device)
    mom = [(cf[a] * g).sum(dim=0) for a in range(2)]
    rho_safe = torch.where(rho != 0, rho, torch.ones_like(rho))
    u_star = [mom[a] / rho_safe for a in range(2)]
    u = [u_star[a] + 0.5 * F[a] / rho_safe for a in range(2)]
    ugr = u[0] * grad_phi[0] + u[1] * grad_phi[1]
    u2 = u[0] ** 2 + u[1] ** 2
    p = (CS2 / (1.0 - w0)) * (g[1:].sum(dim=0) + drho * ugr + rho * w0 * (-1.5 * u2))
    return u, p, u_star


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class CAC2D:
    """D2Q9 CAC + HCH phase-field solver with optional wetting walls."""

    def __init__(
        self,
        shape: tuple[int, int],
        *,
        rho_l: float,
        rho_g: float,
        mu_l: float,
        mu_g: float,
        sigma: float,
        width: float,
        mobility: float = 0.05,
        mu_law: str = "harmonic",
        p0: float = 0.01,
        gravity: tuple[float, float] = (0.0, 0.0),
        gravity_mode: str = "uniform",
        axes: str = "pp",
        wall_mask: torch.Tensor | None = None,
        wall_phi_policy: str = "freeze",
        contact_angle_deg: float | None = None,
        wall_contact_zero_net: bool = True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ):
        if wall_phi_policy not in WALL_POLICIES:
            raise ValueError(f"unknown wall_phi_policy {wall_phi_policy!r}")
        if wall_phi_policy == "contact" and contact_angle_deg is None:
            raise ValueError("wall_phi_policy='contact' needs contact_angle_deg")
        if contact_angle_deg is not None and wall_phi_policy != "contact":
            raise ValueError("contact_angle_deg is only used by wall_phi_policy='contact'")
        if gravity_mode not in ("uniform", "rho"):
            raise ValueError(f"unknown gravity_mode {gravity_mode!r}")
        self.shape = shape
        self.device, self.dtype = device, dtype
        self.rho_l, self.rho_g = rho_l, rho_g
        self.mu_l, self.mu_g, self.mu_law = mu_l, mu_g, mu_law
        self.sigma, self.width = sigma, width
        self.beta, self.k = free_energy_params(sigma, width)
        self.mobility = mobility
        self.tau_f = 0.5 + 3.0 * mobility
        self.p0 = p0
        self.gravity = gravity
        self.gravity_mode = gravity_mode
        self.axes = axes
        self.wall_phi_policy = wall_phi_policy
        self.wall_contact_zero_net = bool(wall_contact_zero_net)
        self.contact_angle_deg = None if contact_angle_deg is None else float(contact_angle_deg)
        self.wall_contact_flux = (
            0.0 if contact_angle_deg is None else wall_contact_flux_from_angle(contact_angle_deg, width)
        )
        self.wall_contact_cos = (
            0.0 if contact_angle_deg is None else math.cos(math.radians(float(contact_angle_deg)))
        )
        self.wall_mask = wall_mask
        self.fluid_mask = (
            torch.ones(shape, dtype=torch.bool, device=device) if wall_mask is None else ~wall_mask
        )
        self.drho = rho_l - rho_g
        self.wv, self.cf = _views(dtype, device)
        self.w0 = float(W[0].item())
        fm = self.fluid_mask
        self.fm_f = fm.to(dtype).unsqueeze(0)
        self.inv_tau_f = torch.where(
            fm,
            torch.full(shape, 1.0 / self.tau_f, dtype=dtype, device=device),
            torch.zeros(shape, dtype=dtype, device=device),
        ).unsqueeze(0)
        self.step_count = 0
        self.f = None
        self.g = None

    # -- initialisation ---------------------------------------------------
    def initialize(self, phi0: torch.Tensor) -> None:
        phi0 = phi0.to(dtype=self.dtype, device=self.device)
        rho0 = rho_of_phi(phi0, self.rho_l, self.rho_g)
        zero = (torch.zeros(self.shape, dtype=self.dtype, device=self.device),) * 2
        self.f = f_equilibrium(phi0, zero, self.wv, self.cf)
        p0f = torch.full(self.shape, self.p0, dtype=self.dtype, device=self.device)
        self.g = g_equilibrium(p0f, rho0, zero, self.wv, self.cf, self.w0)
        self.phi_u_prev = tuple(phi0 * zero[a] for a in range(2))
        self.phi0 = phi0.clone()

    def set_wall_mask(self, mask: torch.Tensor | None) -> None:
        self.wall_mask = mask
        self.fluid_mask = (
            torch.ones(self.shape, dtype=torch.bool, device=self.device) if mask is None else ~mask
        )
        fm = self.fluid_mask
        self.fm_f = fm.to(self.dtype).unsqueeze(0)
        self.inv_tau_f = torch.where(
            fm,
            torch.full(self.shape, 1.0 / self.tau_f, dtype=self.dtype, device=self.device),
            torch.zeros(self.shape, dtype=self.dtype, device=self.device),
        ).unsqueeze(0)

    # -- one step ---------------------------------------------------------
    def refresh_solid_ghosts(self) -> None:
        """Ghost phase field at solid nodes (``neumann`` / ``contact`` policies)."""
        if self.wall_phi_policy == "freeze" or self.wall_mask is None:
            return
        target = self.wall_mask
        if not bool(target.any()):
            return
        phi = self.f.sum(dim=0)
        zero = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        pf = torch.where(self.fluid_mask, phi, zero)
        cnt = shifted_neighbors(self.fluid_mask.to(self.dtype), axes=self.axes).sum(dim=0)
        has = target & (cnt > 0)
        if not bool(has.any()):
            return
        ghost = shifted_neighbors(pf, axes=self.axes).sum(dim=0) / cnt.clamp(min=1.0)
        if self.wall_contact_cos != 0.0:
            ghost = wall_contact_ghost(ghost, self.wall_contact_cos, self.width)
            if self.wall_contact_zero_net:
                # Global phase conservation: the conservative Allen-Cahn field is
                # a conserved quantity, so a wall flux may *redistribute* phase
                # but must not create it.  The mean wetting shift over the active
                # wall is therefore removed (zero-net-flux constraint), which
                # keeps int(phi) drift at the round-off level.
                shift = (ghost - shifted_neighbors(pf, axes=self.axes).sum(dim=0) / cnt.clamp(min=1.0))
                ghost = ghost - shift[has].mean()
        phi_new = torch.where(has, ghost, phi)
        feq = f_equilibrium(phi_new, zero, self.wv, self.cf)
        self.f = torch.where(has.unsqueeze(0), feq, self.f)

    def collide(self) -> None:
        fm = self.fluid_mask
        phi = self.f.sum(dim=0)
        rho = rho_of_phi(phi, self.rho_l, self.rho_g)
        nb = shifted_neighbors(phi, axes=self.axes)  # (8, ny, nx): phi(x + c_i)
        grad = iso_gradient_2d(phi, axes=self.axes, nb=nb)
        lap = iso_laplacian_2d(phi, axes=self.axes, nb=nb)
        mu_phi = chemical_potential(phi, lap, self.beta, self.k)
        mu = viscosity_of_phi(phi, self.rho_l, self.rho_g, self.mu_l, self.mu_g, self.mu_law)
        rho_safe = torch.where(fm, rho, torch.ones_like(rho))
        tau_g = 0.5 + 3.0 * mu / rho_safe
        inv_tau_g = torch.where(fm, 1.0 / tau_g, torch.zeros_like(tau_g)).unsqueeze(0)
        gnorm = (grad[0] ** 2 + grad[1] ** 2).sqrt().clamp(min=1e-12)
        n_hat = (grad[0] / gnorm, grad[1] / gnorm)
        lam = lambda_phi(phi, self.width)
        if self.gravity_mode == "rho":
            rho_fm = rho * fm.to(rho.dtype)
            gx, gy = self.gravity[0] * rho_fm, self.gravity[1] * rho_fm
        else:
            gx, gy = self.gravity[0] * fm, self.gravity[1] * fm
        F = (mu_phi * grad[0] + gx, mu_phi * grad[1] + gy)
        u, p, _ = macroscopic_g(self.g, rho, grad, self.drho, F, self.w0)
        phi_u_now = (phi * u[0], phi * u[1])
        phi_u_dt = (phi_u_now[0] - self.phi_u_prev[0], phi_u_now[1] - self.phi_u_prev[1])
        self.phi_u_prev = phi_u_now

        feq = f_equilibrium(phi, u, self.wv, self.cf)
        fsrc = f_source(phi_u_dt, lam, n_hat, self.tau_f, self.wv, self.cf)
        self.f = self.f + self.inv_tau_f * (feq - self.f) + self.fm_f * fsrc

        geq = g_equilibrium(p, rho, u, self.wv, self.cf, self.w0)
        gsrc = g_force(F, u, grad, self.drho, inv_tau_g, self.wv, self.cf)
        self.g = self.g + inv_tau_g * (geq - self.g) + self.fm_f * gsrc

    def stream(self) -> None:
        c = _c(self.dtype, self.device)
        f = torch.empty_like(self.f)
        g = torch.empty_like(self.g)
        f[0] = self.f[0]
        g[0] = self.g[0]
        for q in range(1, 9):
            dx, dy = int(c[q, 0].item()), int(c[q, 1].item())
            # pull streaming: population arriving at x came from x - c_i
            f[q] = _shift(self.f[q], -dy, -dx, self.axes)
            g[q] = _shift(self.g[q], -dy, -dx, self.axes)
        self.f, self.g = f, g

    def apply_boundaries(self) -> None:
        if self.wall_mask is None:
            return
        opp = OPPOSITE.to(self.f.device)
        m = self.wall_mask.unsqueeze(0)
        self.f = torch.where(m, self.f[opp], self.f)
        self.g = torch.where(m, self.g[opp], self.g)

    def step(self) -> None:
        self.refresh_solid_ghosts()
        self.collide()
        self.stream()
        self.apply_boundaries()
        self.step_count += 1


# ---------------------------------------------------------------------------
# Initial / boundary conditions
# ---------------------------------------------------------------------------


def init_phi_droplet(
    shape: tuple[int, int],
    radius: float,
    width: float,
    center: tuple[float, float] | None = None,
    device=torch.device("cpu"),
    dtype=torch.float32,
) -> torch.Tensor:
    ny, nx = shape
    cy, cx = center if center is not None else (ny / 2.0, nx / 2.0)
    y = torch.arange(ny, device=device, dtype=dtype).view(-1, 1)
    x = torch.arange(nx, device=device, dtype=dtype).view(1, -1)
    d = radius - ((y - cy) ** 2 + (x - cx) ** 2).sqrt()
    return 0.5 + 0.5 * torch.tanh(2.0 * d / width)


def init_phi_ellipse(
    shape: tuple[int, int],
    rx: float,
    ry: float,
    width: float,
    center: tuple[float, float] | None = None,
    device=torch.device("cpu"),
    dtype=torch.float32,
) -> torch.Tensor:
    """Slightly elliptical drop: seeds the m=2 capillary mode for the Lamb anchor."""
    ny, nx = shape
    cy, cx = center if center is not None else (ny / 2.0, nx / 2.0)
    y = torch.arange(ny, device=device, dtype=dtype).view(-1, 1)
    x = torch.arange(nx, device=device, dtype=dtype).view(1, -1)
    r = (((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2).sqrt()
    d = (1.0 - r) * min(rx, ry)
    return 0.5 + 0.5 * torch.tanh(2.0 * d / width)


def init_phi_layers(
    shape: tuple[int, int],
    width: float,
    y0: float,
    device=torch.device("cpu"),
    dtype=torch.float32,
) -> torch.Tensor:
    ny, _ = shape
    y = torch.arange(ny, device=device, dtype=dtype).view(-1, 1)
    return (0.5 + 0.5 * torch.tanh(2.0 * (y0 - y) / width)).expand(*shape).contiguous()


def init_phi_slab(
    shape: tuple[int, int],
    width: float,
    y_lo: float,
    y_hi: float,
    device=torch.device("cpu"),
    dtype=torch.float32,
) -> torch.Tensor:
    """Liquid slab ``y_lo < y < y_hi`` in a periodic (or walled) y domain.

    Two flat interfaces: ``phi = 0.5 (tanh(2 (y-y_lo)/W) - tanh(2 (y-y_hi)/W))``
    is not used; instead the product form keeps both interfaces at exactly the
    ``0.5 (1 +- tanh)`` profile: ``phi = L(y) * (1 - U(y))`` with
    ``L = 0.5 (1 + tanh(2 (y - y_lo)/W))``, ``U = 0.5 (1 + tanh(2 (y - y_hi)/W))``.
    """
    ny, _ = shape
    y = torch.arange(ny, device=device, dtype=dtype).view(-1, 1)
    lo = 0.5 * (1.0 + torch.tanh(2.0 * (y - y_lo) / width))
    hi = 0.5 * (1.0 + torch.tanh(2.0 * (y - y_hi) / width))
    return (lo * (1.0 - hi)).expand(*shape).contiguous()


def solid_plate(shape: tuple[int, int], thickness: int, side: str = "bottom", device=torch.device("cpu")):
    """Flat solid layer of ``thickness`` rows at ``bottom`` or ``top``."""
    ny, nx = shape
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    if side == "bottom":
        mask[:thickness, :] = True
    elif side == "top":
        mask[ny - thickness :, :] = True
    else:
        raise ValueError("side must be 'bottom' or 'top'")
    return mask


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def macro(sim: CAC2D, dtype=torch.float64) -> dict:
    """Recompute the macroscopic chain in ``dtype`` from ``f, g``."""
    phi = sim.f.to(dtype).sum(dim=0)
    rho = rho_of_phi(phi, sim.rho_l, sim.rho_g)
    grad = iso_gradient_2d(phi, axes=sim.axes)
    lap = iso_laplacian_2d(phi, axes=sim.axes)
    mu_phi = chemical_potential(phi, lap, sim.beta, sim.k)
    fm = sim.fluid_mask
    if sim.gravity_mode == "rho":
        rho_fm = rho * fm.to(dtype)
        F = (mu_phi * grad[0] + sim.gravity[0] * rho_fm, mu_phi * grad[1] + sim.gravity[1] * rho_fm)
    else:
        F = (mu_phi * grad[0] + sim.gravity[0] * fm.to(dtype), mu_phi * grad[1] + sim.gravity[1] * fm.to(dtype))
    u, p, _ = macroscopic_g(sim.g.to(dtype), rho, grad, sim.drho, F, sim.w0)
    return {"phi": phi, "rho": rho, "grad": grad, "mu_phi": mu_phi, "F": F, "u": u, "p": p}


def diagnostics(sim: CAC2D) -> dict:
    m = macro(sim)
    fm = sim.fluid_mask
    phi = m["phi"][fm]
    speed = (m["u"][0] ** 2 + m["u"][1] ** 2).sqrt()
    gnorm = (m["grad"][0] ** 2 + m["grad"][1] ** 2).sqrt()
    return {
        "step": sim.step_count,
        "phi_integral_fluid": float(phi.sum().item()),
        "phi_min_fluid": float(phi.min().item()),
        "phi_max_fluid": float(phi.max().item()),
        "rho_min_fluid": float(m["rho"][fm].min().item()),
        "rho_max_fluid": float(m["rho"][fm].max().item()),
        "u_max_fluid": float(speed[fm].max().item()),
        "grad_phi_max": float(gnorm[fm].max().item()),
        "nan_f": int(torch.isnan(sim.f).sum().item()),
        "nan_g": int(torch.isnan(sim.g).sum().item()),
    }


def run(sim: CAC2D, n_steps: int, sample_every: int = 100, on_sample=None, progress_every: int = 0):
    import time as _time

    diags: list[dict] = []
    t0 = _time.time()
    for it in range(n_steps):
        sim.step()
        n = it + 1
        if sample_every and (n % sample_every == 0 or n == 1):
            d = diagnostics(sim)
            d["wall_time_s"] = _time.time() - t0
            diags.append(d)
            if on_sample is not None:
                on_sample(sim, d)
        if progress_every and n % progress_every == 0:
            print(f"[cac2d] step {n}/{n_steps}  {(time.time() - t0) / n * 1e3:.2f} ms/step", flush=True)
    return diags