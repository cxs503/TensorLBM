"""Three-dimensional (D3Q19) conservative Allen-Cahn phase-field lattice Boltzmann.

D3Q19 CAC + HCH double-distribution scheme following Liang et al.,
Phys. Rev. E 97, 033309 (2018).

* h-LBE (Allen-Cahn / phase field) ``f_i`` with ``sum_i f_i = phi``.
* g-LBE (Hua-Hu-Chen incompressible hydrodynamics) ``g_i``.
* mixing laws ``rho = phi (rho_l - rho_g) + rho_g``, harmonic or linear viscosity.
* chemical potential ``mu_phi = 4 beta phi (phi-1)(phi-0.5) - k lap(phi)`` with
  ``beta = 12 sigma / W`` and ``k = 1.5 sigma W``.
* body force ``F = mu_phi grad(phi) + rho g`` (``gravity_mode="rho"``) or ``+ g``
  (``gravity_mode="uniform"``).

Wetting (gas-liquid-solid) walls
--------------------------------
Solid nodes carry a ghost phase-field value.  Policies:

``"freeze"``    keep whatever bounce-back leaves there (default, no wetting).
``"neumann"``   zero normal gradient, i.e. the neutral 90 deg contact angle.
``"contact"``   Young wall condition for a prescribed contact angle.

The ``"neumann"`` policy is the default for hydrostatic/mass-conservation
verification (no wetting bias).

References
----------
Liang, Xu, Chen, Wang, Chai & Shi (2018) Phys. Rev. E 97, 033309.
"""

from __future__ import annotations

import math

import torch

from .d3q19 import C, OPPOSITE, W

CS2 = 1.0 / 3.0

WALL_POLICIES = ("freeze", "neumann", "contact")


def _w(dtype, device) -> torch.Tensor:
    return W.to(device=device, dtype=dtype)


def _c(dtype, device) -> torch.Tensor:
    return C.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Stencils (second-order isotropic, D3Q19)
# ---------------------------------------------------------------------------


def _shift(f: torch.Tensor, dz: int, dy: int, dx: int, axes: str = "ppp") -> torch.Tensor:
    """Return g with ``g[z, y, x] = f[z + dz, y + dy, x + dx]``.

    ``axes`` is a 3-char policy for (z, y, x): ``'p'`` periodic roll, ``'w'``
    non-wrapping zero fill.
    """
    if dz == 0 and dy == 0 and dx == 0:
        return f
    nz, ny, nx = f.shape
    if dz != 0:
        if axes[0] == "p":
            f = torch.roll(f, shifts=-dz, dims=0)
        else:
            out = torch.zeros_like(f)
            zs = slice(max(dz, 0), nz + min(dz, 0))
            zd = slice(max(-dz, 0), nz + min(-dz, 0))
            if zs.stop - zs.start > 0:
                out[zd, :, :] = f[zs, :, :]
            f = out
    if dy != 0:
        if len(axes) > 1 and axes[1] == "p":
            f = torch.roll(f, shifts=-dy, dims=1)
        else:
            out = torch.zeros_like(f)
            ys = slice(max(dy, 0), ny + min(dy, 0))
            yd = slice(max(-dy, 0), ny + min(-dy, 0))
            if ys.stop - ys.start > 0:
                out[:, yd, :] = f[:, ys, :]
            f = out
    if dx != 0:
        if len(axes) > 2 and axes[2] == "p":
            f = torch.roll(f, shifts=-dx, dims=2)
        else:
            out = torch.zeros_like(f)
            xs = slice(max(dx, 0), nx + min(dx, 0))
            xd = slice(max(-dx, 0), nx + min(-dx, 0))
            if xs.stop - xs.start > 0:
                out[:, :, xd] = f[:, :, xs]
            f = out
    return f


def shifted_neighbors(phi: torch.Tensor, axes: str = "ppp") -> torch.Tensor:
    """Stack (18, nz, ny, nx) of ``phi(x + c_i)``, i = 1..18."""
    c = _c(phi.dtype, phi.device)
    out = torch.empty((18, *phi.shape), dtype=phi.dtype, device=phi.device)
    for q in range(1, 19):
        dx, dy, dz = int(c[q, 0].item()), int(c[q, 1].item()), int(c[q, 2].item())
        out[q - 1] = _shift(phi, dz, dy, dx, axes)
    return out


def iso_gradient_3d(
    phi: torch.Tensor, axes: str = "ppp", nb: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``grad(phi)`` = 3 sum_{i!=0} w_i c_i phi(x + c_i)  (second order, isotropic)."""
    if nb is None:
        nb = shifted_neighbors(phi, axes=axes)
    c = _c(phi.dtype, phi.device)
    w = _w(phi.dtype, phi.device)
    gx = torch.zeros_like(phi)
    gy = torch.zeros_like(phi)
    gz = torch.zeros_like(phi)
    for q in range(1, 19):
        dx, dy, dz = int(c[q, 0].item()), int(c[q, 1].item()), int(c[q, 2].item())
        term = nb[q - 1] * (3.0 * float(w[q]))
        gx = gx + term * dx
        gy = gy + term * dy
        gz = gz + term * dz
    return gx, gy, gz


def iso_laplacian_3d(phi: torch.Tensor, axes: str = "ppp", nb: torch.Tensor | None = None) -> torch.Tensor:
    """``lap(phi)`` = 6 sum_i w_i [phi(x + c_i) - phi(x)]  (second order, isotropic)."""
    if nb is None:
        nb = shifted_neighbors(phi, axes=axes)
    w = _w(phi.dtype, phi.device)
    acc = torch.zeros_like(phi)
    for q in range(1, 19):
        acc = acc + float(w[q]) * nb[q - 1]
    return 6.0 * (acc - (1.0 - float(w[0])) * phi)


# ---------------------------------------------------------------------------
# Free energy, chemical potential, mixing laws
# ---------------------------------------------------------------------------


def rho_of_phi(phi: torch.Tensor, rho_l: float, rho_g: float) -> torch.Tensor:
    return phi * (rho_l - rho_g) + rho_g


def free_energy_params(sigma: float, width: float) -> tuple[float, float]:
    """Return (beta, kappa) for the double-well free energy.

    beta = 12 sigma / W, kappa = 1.5 sigma W
    so that the flat-interface solution is phi = 0.5 (1 + tanh(2 (y0-y)/W))
    and the sharp-interface tension is sigma.
    """
    beta = 12.0 * sigma / width
    kappa = 1.5 * sigma * width
    return beta, kappa


def chemical_potential(phi: torch.Tensor, lap_phi: torch.Tensor, beta: float, kappa: float) -> torch.Tensor:
    """mu = 4 beta phi (phi-1)(phi-0.5) - kappa lap(phi)."""
    return 4.0 * beta * phi * (phi - 1.0) * (phi - 0.5) - kappa * lap_phi


def lambda_phi(phi: torch.Tensor, width: float) -> torch.Tensor:
    """Interface speed coefficient: lambda = 4 phi (1-phi) / W."""
    return 4.0 * phi * (1.0 - phi) / width


def viscosity_of_phi(
    phi: torch.Tensor, rho_l: float, rho_g: float, mu_l: float, mu_g: float, law: str = "harmonic"
) -> torch.Tensor:
    if law == "harmonic":
        nu_l = mu_l / rho_l if rho_l > 0 else 1e12
        nu_g = mu_g / rho_g if rho_g > 0 else 1e12
        nu_inv = phi / nu_l + (1.0 - phi) / nu_g
        nu = 1.0 / nu_inv.clamp(min=1e-12)
        rho = rho_of_phi(phi, rho_l, rho_g)
        return nu * rho
    elif law == "linear":
        return phi * mu_l + (1.0 - phi) * mu_g
    else:
        raise ValueError(f"unknown viscosity law: {law}")


# ---------------------------------------------------------------------------
# Equilibrium distributions
# ---------------------------------------------------------------------------


def _views(dtype, device):
    """Return (w_vec, c_mat) for D3Q19."""
    w = _w(dtype, device)  # (19,)
    c = _c(dtype, device)  # (19, 3)
    return w, c


def f_equilibrium(
    phi: torch.Tensor,
    u: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    wv: torch.Tensor,
    cf: torch.Tensor,
) -> torch.Tensor:
    """Phase-field equilibrium [Liang 2018 Eq.9]: feq_i = w_i phi [1 + 3 (c_i·u)].

    First-order form only (no quadratic velocity terms) — the CAC h-LBE
    equilibrium is linear in u, unlike the standard hydrodynamic equilibrium.
    """
    ux, uy, uz = u
    # Reshape velocity for broadcasting: (1, nz, ny, nx)
    ux = ux.unsqueeze(0)
    uy = uy.unsqueeze(0)
    uz = uz.unsqueeze(0)
    phi = phi.unsqueeze(0)
    # cf is (19, 3), reshape to (19, 1, 1, 1) for each component
    cu = cf[:, 0].view(19, 1, 1, 1) * ux + cf[:, 1].view(19, 1, 1, 1) * uy + cf[:, 2].view(19, 1, 1, 1) * uz
    feq = wv.view(19, 1, 1, 1) * phi * (1.0 + 3.0 * cu)
    return feq


def g_equilibrium(
    p: torch.Tensor,
    rho: torch.Tensor,
    u: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    wv: torch.Tensor,
    cf: torch.Tensor,
    w0: float,
) -> torch.Tensor:
    """HCH incompressible equilibrium.

    geq_0 = w0 (p/c_s^2 - rho) + rho
    geq_i = w_i (p/c_s^2 + rho [3 (c_i·u) + 4.5 (c_i·u)^2 - 1.5 u^2])  for i > 0
    """
    ux, uy, uz = u
    u2 = ux ** 2 + uy ** 2 + uz ** 2
    # Reshape for broadcasting: (1, nz, ny, nx)
    ux = ux.unsqueeze(0)
    uy = uy.unsqueeze(0)
    uz = uz.unsqueeze(0)
    u2 = u2.unsqueeze(0)
    rho = rho.unsqueeze(0)
    p = p.unsqueeze(0)
    # cf is (19, 3), reshape to (19, 1, 1, 1) for each component
    cu = cf[:, 0].view(19, 1, 1, 1) * ux + cf[:, 1].view(19, 1, 1, 1) * uy + cf[:, 2].view(19, 1, 1, 1) * uz
    p_cs2 = p / CS2

    geq = torch.empty((19, *p.shape[1:]), dtype=p.dtype, device=p.device)
    geq[0] = w0 * (p_cs2[0] - rho[0]) + rho[0]
    for i in range(1, 19):
        geq[i] = float(wv[i]) * (p_cs2[0] + rho[0] * (3.0 * cu[i] + 4.5 * cu[i] ** 2 - 1.5 * u2[0]))
    return geq


# ---------------------------------------------------------------------------
# Source terms
# ---------------------------------------------------------------------------


def f_source(
    phi_u_dt: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    lam: torch.Tensor,
    n_hat: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tau_f: float,
    wv: torch.Tensor,
    cf: torch.Tensor,
) -> torch.Tensor:
    """Source term for the phase-field LBE [Liang 2018 Eq.11].

    F_i = (1 - 1/(2τ_f)) (w_i/c_s²) c_i · [d(φu)/dt + c_s² λ n]

    With c_s² = 1/3, this becomes:
    F_i = (1 - 1/(2τ_f)) * 3 * w_i * c_i · [d(φu)/dt + (1/3) λ n]
    """
    coef = 1.0 - 0.5 / tau_f  # (1 - 1/(2τ_f))
    dphix, dphiy, dphiz = phi_u_dt
    nx, ny, nz = n_hat
    # Combined flux: d(φu)/dt + c_s² λ n = d(φu)/dt + (1/3) λ n
    fx = dphix + (lam / 3.0) * nx
    fy = dphiy + (lam / 3.0) * ny
    fz = dphiz + (lam / 3.0) * nz

    src = torch.zeros((19, *fx.shape), dtype=fx.dtype, device=fx.device)
    for i in range(19):
        src[i] = coef * 3.0 * float(wv[i]) * (
            float(cf[i, 0]) * fx + float(cf[i, 1]) * fy + float(cf[i, 2]) * fz
        )
    return src


def g_force(
    F: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    u: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    drho: float,
    inv_tau_g: torch.Tensor,
    wv: torch.Tensor,
    cf: torch.Tensor,
) -> torch.Tensor:
    """Force source term for the HCH LBE [Liang 2018 Eq.18→20].

    G_i = (1 - 1/(2τ_g)) w_i [3 (c·F) + 3 (c·u)(c·∇ρ)]

    where ∇ρ = drho * ∇φ.
    """
    fx, fy, fz = F
    ux, uy, uz = u
    gx, gy, gz = grad

    src = torch.zeros((19, *fx.shape), dtype=fx.dtype, device=fx.device)
    for i in range(19):
        ci_x, ci_y, ci_z = float(cf[i, 0]), float(cf[i, 1]), float(cf[i, 2])
        cF = ci_x * fx + ci_y * fy + ci_z * fz
        cu = ci_x * ux + ci_y * uy + ci_z * uz
        cgr = ci_x * gx + ci_y * gy + ci_z * gz
        # Eq.20: 3(c·F) + 3(c·u)(c·∇ρ) = 3(c·F) + 3(c·u)(c·∇φ)*drho
        src[i] = float(wv[i]) * (1.0 - 0.5 * inv_tau_g[0]) * (
            3.0 * cF + 3.0 * cu * cgr * drho
        )
    return src


# ---------------------------------------------------------------------------
# Macroscopic variables
# ---------------------------------------------------------------------------


def macroscopic_g(
    g: torch.Tensor,
    rho: torch.Tensor,
    grad: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    drho: float,
    F: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    w0: float,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Recover velocity and pressure from g distribution.

    u* = sum_i c_i g_i / (rho c_s^2)
    u = u* + 0.5 F / rho
    p = c_s^2 / (1 - w0) * (sum_{i>0} g_i + drho (u·grad_phi) + rho w0 (-1.5 u^2))
    """
    c = _c(g.dtype, g.device)
    u_star = [torch.zeros_like(rho) for _ in range(3)]
    for i in range(1, 19):
        for a in range(3):
            u_star[a] = u_star[a] + float(c[i, a]) * g[i]
    rho_safe = torch.clamp(rho, min=1e-12)
    # [Liang 2018 Eq.21]: rho u = sum c_i g_i + 0.5 F
    # => u* = sum c_i g_i / rho, u = u* + 0.5 F / rho
    u_star = [u / rho_safe for u in u_star]
    u = [u_star[a] + 0.5 * F[a] / rho_safe for a in range(3)]
    ugr = u[0] * grad[0] + u[1] * grad[1] + u[2] * grad[2]
    u2 = u[0] ** 2 + u[1] ** 2 + u[2] ** 2
    p = (CS2 / (1.0 - w0)) * (g[1:].sum(dim=0) + drho * ugr + rho * w0 * (-1.5 * u2))
    return u, p, u_star


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class CAC3D:
    """D3Q19 CAC + HCH phase-field solver with optional wetting walls."""

    def __init__(
        self,
        shape: tuple[int, int, int],
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
        gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gravity_mode: str = "uniform",
        axes: str = "ppp",
        wall_mask: torch.Tensor | None = None,
        wall_phi_policy: str = "freeze",
        contact_angle_deg: float | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        if wall_phi_policy not in WALL_POLICIES:
            raise ValueError(f"unknown wall_phi_policy {wall_phi_policy!r}")
        if wall_phi_policy == "contact" and contact_angle_deg is None:
            raise ValueError("wall_phi_policy='contact' needs contact_angle_deg")
        if gravity_mode not in ("uniform", "rho"):
            raise ValueError(f"unknown gravity_mode {gravity_mode!r}")

        dev = torch.device(device)
        self.shape = shape
        self.device, self.dtype = dev, dtype
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
        self.contact_angle_deg = None if contact_angle_deg is None else float(contact_angle_deg)
        self.wall_mask = wall_mask
        self.fluid_mask = (
            torch.ones(shape, dtype=torch.bool, device=dev) if wall_mask is None else ~wall_mask
        )
        self.drho = rho_l - rho_g
        self.wv, self.cf = _views(dtype, dev)
        self.w0 = float(W[0].item())
        fm = self.fluid_mask
        self.fm_f = fm.to(dtype).unsqueeze(0)
        self.inv_tau_f = torch.where(
            fm,
            torch.full(shape, 1.0 / self.tau_f, dtype=dtype, device=dev),
            torch.zeros(shape, dtype=dtype, device=dev),
        ).unsqueeze(0)
        self.step_count = 0
        self.f = None
        self.g = None

    # -- initialisation ---------------------------------------------------
    def initialize(self, phi0: torch.Tensor) -> None:
        phi0 = phi0.to(dtype=self.dtype, device=self.device)
        rho0 = rho_of_phi(phi0, self.rho_l, self.rho_g)
        zero = (torch.zeros(self.shape, dtype=self.dtype, device=self.device),) * 3
        self.f = f_equilibrium(phi0, zero, self.wv, self.cf)
        p0f = torch.full(self.shape, self.p0, dtype=self.dtype, device=self.device)
        self.g = g_equilibrium(p0f, rho0, zero, self.wv, self.cf, self.w0)
        self.phi_u_prev = tuple(phi0 * zero[a] for a in range(3))

    # -- wall ghost -------------------------------------------------------
    def _wall_ghost_neumann(self, phi: torch.Tensor) -> torch.Tensor:
        """Neumann: zero normal gradient -> ghost = phi at fluid node."""
        return phi.clone()

    def _wall_ghost_contact(self, phi: torch.Tensor) -> torch.Tensor:
        """Young contact angle ghost."""
        cos_theta = math.cos(math.radians(self.contact_angle_deg))
        # dphi/dn = -cos(theta) * 4 * phi * (1-phi) / W
        f = phi.clamp(0.0, 1.0)
        return phi + cos_theta * 4.0 * f * (1.0 - f) / self.width

    def refresh_solid_ghosts(self) -> None:
        if self.wall_mask is None:
            return
        phi = self.f.sum(dim=0)
        if self.wall_phi_policy == "neumann":
            ghost = self._wall_ghost_neumann(phi)
        elif self.wall_phi_policy == "contact":
            ghost = self._wall_ghost_contact(phi)
        else:
            return  # freeze: do nothing

        # Apply ghost only at wall nodes
        phi_new = torch.where(self.wall_mask, ghost, phi)
        zero = (torch.zeros(self.shape, dtype=self.dtype, device=self.device),) * 3
        feq = f_equilibrium(phi_new, zero, self.wv, self.cf)
        self.f = torch.where(self.wall_mask.unsqueeze(0), feq, self.f)

    # -- collide ----------------------------------------------------------
    def collide(self) -> None:
        fm = self.fluid_mask
        phi = self.f.sum(dim=0)
        rho = rho_of_phi(phi, self.rho_l, self.rho_g)

        nb = shifted_neighbors(phi, axes=self.axes)
        grad = iso_gradient_3d(phi, axes=self.axes, nb=nb)
        lap = iso_laplacian_3d(phi, axes=self.axes, nb=nb)
        mu_phi = chemical_potential(phi, lap, self.beta, self.k)
        mu = viscosity_of_phi(phi, self.rho_l, self.rho_g, self.mu_l, self.mu_g, self.mu_law)
        rho_safe = torch.where(fm, rho, torch.ones_like(rho))
        tau_g = 0.5 + 3.0 * mu / rho_safe
        inv_tau_g = torch.where(fm, 1.0 / tau_g, torch.zeros_like(tau_g)).unsqueeze(0)

        gnorm = (grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2).sqrt().clamp(min=1e-12)
        n_hat = (grad[0] / gnorm, grad[1] / gnorm, grad[2] / gnorm)
        lam = lambda_phi(phi, self.width)

        if self.gravity_mode == "rho":
            rho_fm = rho * fm.to(rho.dtype)
            gx = self.gravity[0] * rho_fm
            gy = self.gravity[1] * rho_fm
            gz = self.gravity[2] * rho_fm
        else:
            gx = self.gravity[0] * fm.to(rho.dtype)
            gy = self.gravity[1] * fm.to(rho.dtype)
            gz = self.gravity[2] * fm.to(rho.dtype)

        F = (mu_phi * grad[0] + gx, mu_phi * grad[1] + gy, mu_phi * grad[2] + gz)
        u, p, _ = macroscopic_g(self.g, rho, grad, self.drho, F, self.w0)

        phi_u_now = (phi * u[0], phi * u[1], phi * u[2])
        phi_u_dt = (phi_u_now[0] - self.phi_u_prev[0], phi_u_now[1] - self.phi_u_prev[1], phi_u_now[2] - self.phi_u_prev[2])
        self.phi_u_prev = phi_u_now

        feq = f_equilibrium(phi, u, self.wv, self.cf)
        fsrc = f_source(phi_u_dt, lam, n_hat, self.tau_f, self.wv, self.cf)
        self.f = self.f + self.inv_tau_f * (feq - self.f) + self.fm_f * fsrc

        geq = g_equilibrium(p, rho, u, self.wv, self.cf, self.w0)
        gsrc = g_force(F, u, grad, self.drho, inv_tau_g, self.wv, self.cf)
        self.g = self.g + inv_tau_g * (geq - self.g) + self.fm_f * gsrc

    # -- stream -----------------------------------------------------------
    def stream(self) -> None:
        c = _c(self.dtype, self.device)
        f = torch.empty_like(self.f)
        g = torch.empty_like(self.g)
        f[0] = self.f[0]
        g[0] = self.g[0]
        for q in range(1, 19):
            dx, dy, dz = int(c[q, 0].item()), int(c[q, 1].item()), int(c[q, 2].item())
            f[q] = _shift(self.f[q], -dz, -dy, -dx, self.axes)
            g[q] = _shift(self.g[q], -dz, -dy, -dx, self.axes)
        self.f, self.g = f, g

    # -- boundaries -------------------------------------------------------
    def apply_boundaries(self) -> None:
        if self.wall_mask is None:
            return
        opp = OPPOSITE.to(self.f.device)
        m = self.wall_mask.unsqueeze(0)
        self.f = torch.where(m, self.f[opp], self.f)
        self.g = torch.where(m, self.g[opp], self.g)

    # -- step -------------------------------------------------------------
    def step(self) -> None:
        self.refresh_solid_ghosts()
        self.collide()
        self.stream()
        self.apply_boundaries()
        self.step_count += 1


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def init_phi_slab_3d(
    shape: tuple[int, int, int],
    width: float,
    z_lo: float,
    z_hi: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Liquid slab ``z_lo < z < z_hi`` in a periodic (or walled) z domain.

    Two flat interfaces: product form keeps both interfaces at exactly the
    ``0.5 (1 +- tanh)`` profile.
    """
    nz, ny, nx = shape
    dev = torch.device(device)
    z = torch.arange(nz, device=dev, dtype=dtype).view(-1, 1, 1)
    lo = 0.5 * (1.0 + torch.tanh(2.0 * (z - z_lo) / width))
    hi = 0.5 * (1.0 + torch.tanh(2.0 * (z - z_hi) / width))
    return (lo * (1.0 - hi)).expand(*shape).contiguous()


def init_phi_droplet_3d(
    shape: tuple[int, int, int],
    radius: float,
    width: float,
    center: tuple[float, float, float] | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    nz, ny, nx = shape
    dev = torch.device(device)
    cz, cy, cx = center if center is not None else (nz / 2.0, ny / 2.0, nx / 2.0)
    z = torch.arange(nz, device=dev, dtype=dtype).view(-1, 1, 1)
    y = torch.arange(ny, device=dev, dtype=dtype).view(1, -1, 1)
    x = torch.arange(nx, device=dev, dtype=dtype).view(1, 1, -1)
    d = radius - ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2).sqrt()
    return 0.5 + 0.5 * torch.tanh(2.0 * d / width)


def solid_plate_3d(
    shape: tuple[int, int, int],
    thickness: int,
    side: str = "bottom",
    device: str = "cpu",
) -> torch.Tensor:
    """Flat solid layer of ``thickness`` rows at ``bottom`` or ``top`` (z direction)."""
    nz, ny, nx = shape
    dev = torch.device(device)
    mask = torch.zeros(shape, dtype=torch.bool, device=dev)
    if side == "bottom":
        mask[:thickness, :, :] = True
    elif side == "top":
        mask[nz - thickness :, :, :] = True
    else:
        raise ValueError("side must be 'bottom' or 'top'")
    return mask


def solid_channel_3d(
    shape: tuple[int, int, int],
    thickness: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Solid walls at top and bottom (z direction), periodic in x, y."""
    nz, ny, nx = shape
    dev = torch.device(device)
    mask = torch.zeros(shape, dtype=torch.bool, device=dev)
    mask[:thickness, :, :] = True
    mask[nz - thickness :, :, :] = True
    return mask


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def macro(sim: CAC3D, dtype=torch.float64) -> dict:
    """Recompute the macroscopic chain in ``dtype`` from ``f, g``."""
    phi = sim.f.to(dtype).sum(dim=0)
    rho = rho_of_phi(phi, sim.rho_l, sim.rho_g)
    grad = iso_gradient_3d(phi, axes=sim.axes)
    lap = iso_laplacian_3d(phi, axes=sim.axes)
    mu_phi = chemical_potential(phi, lap, sim.beta, sim.k)
    fm = sim.fluid_mask
    if sim.gravity_mode == "rho":
        rho_fm = rho * fm.to(dtype)
        F = (
            mu_phi * grad[0] + sim.gravity[0] * rho_fm,
            mu_phi * grad[1] + sim.gravity[1] * rho_fm,
            mu_phi * grad[2] + sim.gravity[2] * rho_fm,
        )
    else:
        F = (
            mu_phi * grad[0] + sim.gravity[0] * fm.to(dtype),
            mu_phi * grad[1] + sim.gravity[1] * fm.to(dtype),
            mu_phi * grad[2] + sim.gravity[2] * fm.to(dtype),
        )
    u, p, _ = macroscopic_g(sim.g.to(dtype), rho, grad, sim.drho, F, sim.w0)
    return {"phi": phi, "rho": rho, "grad": grad, "mu_phi": mu_phi, "F": F, "u": u, "p": p}


def diagnostics(sim: CAC3D) -> dict:
    m = macro(sim)
    fm = sim.fluid_mask
    phi = m["phi"][fm]
    speed = (m["u"][0] ** 2 + m["u"][1] ** 2 + m["u"][2] ** 2).sqrt()
    gnorm = (m["grad"][0] ** 2 + m["grad"][1] ** 2 + m["grad"][2] ** 2).sqrt()
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


def run(sim: CAC3D, n_steps: int, sample_every: int = 100, on_sample=None, progress_every: int = 0):
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
            print(f"[cac3d] step {n}/{n_steps}  {(_time.time() - t0) / n * 1e3:.2f} ms/step", flush=True)
    return diags
