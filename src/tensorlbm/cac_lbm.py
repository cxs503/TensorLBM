"""Conservative Allen-Cahn phase-field two-phase LBM — W9-R2 (2026-09-29).

Scheme locked in prereg.md (phasefield_dev/):
  [L1] Liang, Xu, Chen, Wang, Chai, Shi, arXiv:1710.09541v1 (PRE 97,033309):
       CAC Eq.(1-3,7); h-LBE Eq.(8-12); chemical potential Eq.(5-6);
       HCH g-LBE Eq.(15-23); mixing Eq.(13,24-26); stencils Eq.(27-28);
       layered-Poiseuille protocol Eq.(31-33).
  [L2] Wang, He, Wang, arXiv:2306.11320v1: same-family cross-check of the
       parameter-contrast hydro equations (their Eq.21-27).
D3Q19 dual LBE, dx=dt=1, cs^2=1/3.  fp32 compute + fp64 diagnostics.

Written from the published equations.  The Fakhari/Bolster PRE 96,053301
companion code (GPL-3.0) was used ONLY as an equation-level oracle
(controller license line 2026-09-29): no line- or structure-level
transcription.

Library reuse (worktree /nfs/wangxi/worktrees/bm_w9/src, via sys.path):
  tensorlbm.d3q19            C, W_EXACT64, OPPOSITE
  tensorlbm.boundaries3d     bounce_back_cells_3d  (full-way post-side BB)
  tensorlbm.phasefield.stream_boundary_contract
                             stream_d3q19_adapter (periodic pull-stream)
New physics (library has no equivalent; NOTES.md section 6a):
  isotropic Eq.(27/28) stencils (library operators are 7-point / plain
  central-difference — coefficient-checked different), CAC h-LBE, HCH
  g-LBE, chemical potential, mixing laws, Guo-type force, mixed-axis
  non-wrapping pull-stream (plumbing for walled axes; physics stays the
  library bounce call).
"""

from __future__ import annotations

import torch

from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import W_EXACT64, C
from tensorlbm.phasefield.stream_boundary_contract import stream_d3q19_adapter

CS2 = 1.0 / 3.0


def _weights(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return W_EXACT64.to(device=device, dtype=dtype)


def _c_float(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return C.to(device=device, dtype=dtype)


def _shift_field(
    f: torch.Tensor,
    dim: int,
    s: int,
    pol: str,
) -> torch.Tensor:
    """Return g with g[pos] = f[pos + s] along ``dim``.

    pol='p': periodic (torch.roll, shifts=-s gives g[i]=f[i+s]).
    pol='w': non-wrapping; the |s| edge entries that would read outside are
             ZERO-filled.  They land only on wall nodes in directions that,
             after the full-way bounce swap, no fluid node ever reads (1-D round-trip
             proof, NOTES 6a).  Zero fill (not edge-block copy) so no duplicated ghost
             populations are created.
    """
    if s == 0:
        return f
    if pol == "p":
        return torch.roll(f, shifts=-s, dims=dim)
    n = f.shape[dim]
    a = abs(s)
    pre = tuple(slice(None) if d != dim else slice(n - a, n) for d in range(f.dim()))
    post = tuple(slice(None) if d != dim else slice(0, a) for d in range(f.dim()))
    body = tuple(
        slice(None) if d != dim else (slice(a, n) if s > 0 else slice(0, n - a))
        for d in range(f.dim())
    )
    if s > 0:
        # g[i]=f[i+s] for i<n-s ; g[n-s:]=0 (out-of-domain read)
        return torch.cat((f[body], torch.zeros_like(f[pre])), dim=dim)
    # s<0: g[i]=f[i+s] for i>=a ; g[:a]=0 (out-of-domain read)
    return torch.cat((torch.zeros_like(f[post]), f[body]), dim=dim)


def shifted_neighbors(
    phi: torch.Tensor,
    *,
    axes: str = "ppp",
) -> torch.Tensor:
    """Stack (18, nz, ny, nx) of phi(x + c_i), i = 1..18 (skip rest).

    ``axes`` is a 3-char policy for dims (z, y, x): 'p' periodic, 'w' walled
    (non-wrapping, junk confined to wall layers).
    """
    nz, ny, nx = phi.shape
    out = torch.empty((18, nz, ny, nx), dtype=phi.dtype, device=phi.device)
    for j, (cx, cy, cz) in enumerate(C[1:].tolist()):
        f = phi
        for dim, s, pol in ((0, cz, axes[0]), (1, cy, axes[1]), (2, cx, axes[2])):
            f = _shift_field(f, dim, s, pol)
        out[j] = f
    return out


def iso_gradient_3d(nb: torch.Tensor, device: torch.device):
    """(gx, gy, gz) = Eq.(27) = 3 * sum_{i!=0} w_i c_ia phi(x+c_i).

    ``nb`` = shifted_neighbors(phi).  D2Q9 collapse cross-checked against
    the [O1] oracle at equation level (NOTES section 5).
    """
    w = _weights(nb.dtype, device)[1:].view(18, 1, 1, 1)
    c = _c_float(nb.dtype, device)[1:]
    return tuple(3.0 * (w * c[:, a].view(18, 1, 1, 1) * nb).sum(dim=0) for a in range(3))


def iso_laplacian_3d(phi: torch.Tensor, nb: torch.Tensor) -> torch.Tensor:
    """Eq.(28): 6 * sum_{i!=0} w_i [phi(x+c_i) - phi(x)] =
    6 * ( sum_{i!=0} w_i phi(x+c_i) - (2/3) phi )."""
    w = _weights(nb.dtype, phi.device)[1:].view(18, 1, 1, 1)
    return 6.0 * ((w * nb).sum(dim=0) - (2.0 / 3.0) * phi)


def normal_from_gradient(gx, gy, gz, eps: float = 1e-32):
    """n = grad(phi)/|grad(phi)|, |grad| floored (engineering floor)."""
    gnorm = (gx * gx + gy * gy + gz * gz).sqrt().clamp_min(eps)
    return gx / gnorm, gy / gnorm, gz / gnorm


def stream_mixed(
    f: torch.Tensor,
    *,
    axes: str = "ppp",
) -> torch.Tensor:
    """Pull-stream new[q][x] = old[q][x - c_q] with per-axis policy.

    Periodic axes wrap (same convention as the library
    ``stream_d3q19_adapter(..., boundary='periodic')``, which is used
    directly for the all-periodic case); walled axes do not wrap — the
    out-of-domain pulls land only on wall nodes, whose content is replaced
    by the subsequent full-way bounce anyway (1-D round-trip verified:
    junk never enters the fluid, NOTES section 6a).
    """
    nz, ny, nx = f.shape[1:]
    out = torch.empty_like(f)
    out[0] = f[0]
    for q, (cx, cy, cz) in enumerate(C.tolist()):
        if q == 0:
            continue
        g = f[q]
        for dim, s, pol in ((0, -cz, axes[0]), (1, -cy, axes[1]), (2, -cx, axes[2])):
            g = _shift_field(g, dim, s, pol)
        out[q] = g
    return out


# ---------------------------------------------------------------------------
# Mixing laws [L1] Eq.(13),(24),(25),(26) and chemical potential Eq.(5-7)
# ---------------------------------------------------------------------------


def rho_of_phi(phi: torch.Tensor, rho_l: float, rho_g: float) -> torch.Tensor:
    """Eq.(13): rho = phi (rho_l - rho_g) + rho_g."""
    return phi * (rho_l - rho_g) + rho_g


def viscosity_of_phi(
    phi: torch.Tensor,
    rho_l: float,
    rho_g: float,
    mu_l: float,
    mu_g: float,
    law: str = "harmonic",
) -> torch.Tensor:
    """Dynamic viscosity field.  law='harmonic' Eq.(25) (W10-M1 default,
    selected by the pre-registered P0 FD smear rule, out/p0_smear_*.json),
    'linear' Eq.(24), 'step' Eq.(26) (kept for A/B diagnostics; the frozen
    W9 verdicts used 'step')."""
    if law == "step":
        return torch.where(phi >= 0.5, torch.full_like(phi, mu_l), torch.full_like(phi, mu_g))
    nu_l, nu_g = mu_l / rho_l, mu_g / rho_g
    if law == "linear":
        nu = phi * (nu_l - nu_g) + nu_g
    elif law == "harmonic":
        nu = 1.0 / (phi * (1.0 / nu_l - 1.0 / nu_g) + 1.0 / nu_g)
    else:
        raise ValueError(f"unknown mixing law {law!r}")
    return nu * rho_of_phi(phi, rho_l, rho_g)


def chemical_potential(phi: torch.Tensor, nb: torch.Tensor, beta: float, k: float):
    """Eq.(6): mu_phi = 4 beta phi(phi-1)(phi-0.5) - k lap(phi)."""
    return 4.0 * beta * phi * (phi - 1.0) * (phi - 0.5) - k * iso_laplacian_3d(phi, nb)


def free_energy_params(sigma: float, width: float) -> tuple[float, float]:
    """Eq.(7): beta = 12 sigma / W, k = 1.5 sigma W."""
    return 12.0 * sigma / width, 1.5 * sigma * width


# ---------------------------------------------------------------------------
# h-LBE pieces, [L1] Eq.(8-12)   (phi in [0,1], interface at 0.5)
# ---------------------------------------------------------------------------


def f_equilibrium(phi, u, wv, cf):
    """Eq.(9): feq_i = w_i phi (1 + c_i.u / cs^2)."""
    cu = cf[0] * u[0] + cf[1] * u[1] + cf[2] * u[2]
    return wv * phi * (1.0 + 3.0 * cu)


def f_source(phi_u_dt, lam, n_hat, tau_f, wv, cf):
    """Eq.(11): F_i = (1 - 1/(2 tau_f)) (w_i/cs^2) c_i . [d(phi u)/dt + cs^2 lam n]."""
    coef = 1.0 - 0.5 / tau_f
    brack0 = phi_u_dt[0] + (lam / 3.0) * n_hat[0]
    brack1 = phi_u_dt[1] + (lam / 3.0) * n_hat[1]
    brack2 = phi_u_dt[2] + (lam / 3.0) * n_hat[2]
    return coef * 3.0 * wv * (cf[0] * brack0 + cf[1] * brack1 + cf[2] * brack2)


def lambda_phi(phi: torch.Tensor, width: float) -> torch.Tensor:
    """Eq.(3): lambda(phi) = 4 phi (1 - phi) / W."""
    return 4.0 * phi * (1.0 - phi) / width


# ---------------------------------------------------------------------------
# g-LBE (HCH incompressible) pieces, [L1] Eq.(15-23)
# ---------------------------------------------------------------------------


def s_hermite(u, wv, cf):
    """Eq.(17): s_i(u) = w_i [ c.u/cs2 + (c.u)^2/(2 cs4) - u^2/(2 cs2) ]."""
    cu = cf[0] * u[0] + cf[1] * u[1] + cf[2] * u[2]
    u2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2]
    return wv * (3.0 * cu + 4.5 * cu * cu - 1.5 * u2)


def g_equilibrium(p, rho, u, wv, cf, w0: float):
    """Eq.(16): geq_0 = (p/cs2)(w0-1) + rho s_0 ; geq_i = (p/cs2) w_i + rho s_i."""
    s = s_hermite(u, wv, cf)
    geq = (p / CS2) * wv + rho * s
    geq = torch.cat((((p / CS2) * (w0 - 1.0) + rho * s[0]).unsqueeze(0), geq[1:]), dim=0)
    return geq


def g_force(F, u, grad_phi, drho, inv_tau_g, wv, cf):
    """[L1] Eq.(18) -> Eq.(20) (cross-checked against [L2] Eq.(24)):
    G_i = (1 - 1/(2 tau_g)) w_i [ (c.F)/cs2
                                   + u.grad(rho) + u grad(rho):(cc - cs2 I)/cs2 ]
    with grad(rho) = drho * grad(phi).  The cs2-subtracted tensor term plus the
    standalone u.grad(rho) telescope, giving the paper's own simplified form
    Eq.(20):  3 (c.F) + 3 (c.u)(c.grad(rho)).
    Erratum 2026-09-29: prereg section 2 originally transcribed the tensor
    denominator as cs4 (coefficients 9 / -2); that reading contradicts
    [L1] Eq.(20) and [L2] Eq.(24), both of which give coefficient 3 with no
    residual u.grad(rho) term.  Corrected here; see prereg erratum + NOTES.
    """
    coef = 1.0 - 0.5 * inv_tau_g
    cu = cf[0] * u[0] + cf[1] * u[1] + cf[2] * u[2]
    cgr = cf[0] * grad_phi[0] + cf[1] * grad_phi[1] + cf[2] * grad_phi[2]
    cF = cf[0] * F[0] + cf[1] * F[1] + cf[2] * F[2]
    return coef * wv * (3.0 * cF + 3.0 * cu * cgr * drho)


def macroscopic_g(g, rho, grad_phi, drho, F, w0: float):
    """Eq.(21): rho u = sum c g + 0.5 F ; p = cs2/(1-w0) [ sum_{i!=0} g
    + u.grad(rho) + rho s_0(u) ].  Correction terms use the physical u
    of Eq.(21a) (velocity-convention erratum 2026-09-29, prereg section 8:
    equilibria and forces use Eq.(21a)'s u, not u*)."""
    device = g.device
    c = _c_float(g.dtype, device)
    cf = [c[:, a].view(19, 1, 1, 1) for a in range(3)]
    mom = [(cf[a] * g).sum(dim=0) for a in range(3)]
    rho_safe = torch.where(rho != 0, rho, torch.ones_like(rho))
    u_star = [mom[a] / rho_safe for a in range(3)]
    u = [u_star[a] + 0.5 * F[a] / rho_safe for a in range(3)]
    ugr = u[0] * grad_phi[0] + u[1] * grad_phi[1] + u[2] * grad_phi[2]
    u2 = u[0] ** 2 + u[1] ** 2 + u[2] ** 2
    p = (CS2 / (1.0 - w0)) * (g[1:].sum(dim=0) + drho * ugr + rho * w0 * (-1.5 * u2))
    return u, p, u_star


# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------


class CACSim:
    """Dual-LBE CAC phase-field simulation state + step (prereg section 2).

    Wall handling: solid nodes carry no collision (inv-tau zeroed there,
    vectorised NoDynamics) and are full-way bounced post-stream via the
    library ``bounce_back_cells_3d`` (h_eff convention: prereg amendment 2).
    Out-of-domain pulls land only on wall nodes in directions that, after
    the bounce swap, no fluid node ever reads (1-D round-trip proof in
    NOTES 6a).
    """

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
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.rho_l, self.rho_g = rho_l, rho_g
        self.mu_l, self.mu_g = mu_l, mu_g
        self.mu_law = mu_law
        self.sigma, self.width = sigma, width
        self.beta, self.k = free_energy_params(sigma, width)
        self.mobility = mobility
        self.tau_f = 0.5 + 3.0 * mobility  # Eq.(14)
        self.lam_coef = 4.0 / width
        self.p0 = p0
        self.axes = axes
        self.wall_mask = wall_mask
        self.fluid_mask = (
            torch.ones(shape, dtype=torch.bool, device=device) if wall_mask is None else ~wall_mask
        )
        self.gravity = gravity
        if gravity_mode not in ("uniform", "rho"):
            raise ValueError(f"unknown gravity_mode {gravity_mode!r}")
        self.gravity_mode = gravity_mode
        nz, ny, nx = shape
        self.wv, self.c3, self.cf = _unit_views(nz, ny, nx, dtype, device)
        self.cf18 = [self.c3[1:, a].view(18, 1, 1, 1) for a in range(3)]
        self.w0 = float(_weights(dtype, device)[0].item())
        self.drho = rho_l - rho_g
        fm = self.fluid_mask
        self.fm_f = fm.to(dtype).unsqueeze(0)  # (1,nz,ny,nx) multiplier
        self.inv_tau_f = torch.where(
            fm,
            torch.full(shape, 1.0 / self.tau_f, dtype=dtype, device=device),
            torch.zeros(shape, dtype=dtype, device=device),
        ).unsqueeze(0)
        self.step_count = 0

    # -- initialisation ----------------------------------------------------

    def initialize(self, phi0: torch.Tensor) -> None:
        """Set f = feq(phi0, u=0), g = geq(p0, rho(phi0), u=0)."""
        phi0 = phi0.to(dtype=self.dtype, device=self.device)
        rho0 = rho_of_phi(phi0, self.rho_l, self.rho_g)
        zero = (torch.zeros(self.shape, dtype=self.dtype, device=self.device),) * 3
        self.f = f_equilibrium(phi0, zero, self.wv, self.cf)
        p0f = torch.full(self.shape, self.p0, dtype=self.dtype, device=self.device)
        self.g = g_equilibrium(p0f, rho0, zero, self.wv, self.cf, self.w0)
        self.phi_u_prev = tuple(phi0 * zero[a] for a in range(3))
        self.phi0 = phi0.clone()

    # -- one step ----------------------------------------------------------

    def step(self) -> None:
        fm = self.fluid_mask
        phi = self.f.sum(dim=0)  # Eq.(12)
        rho = rho_of_phi(phi, self.rho_l, self.rho_g)  # Eq.(13)
        nb = shifted_neighbors(phi, axes=self.axes)
        grad = iso_gradient_3d(nb, self.device)  # Eq.(27)
        mu_phi = chemical_potential(phi, nb, self.beta, self.k)  # Eq.(6)
        mu = viscosity_of_phi(phi, self.rho_l, self.rho_g, self.mu_l, self.mu_g, self.mu_law)
        rho_safe = torch.where(fm, rho, torch.ones_like(rho))  # wall guard
        tau_g = 0.5 + 3.0 * mu / rho_safe  # Eq.(23)
        inv_tau_g = torch.where(fm, 1.0 / tau_g, torch.zeros_like(tau_g)).unsqueeze(0)
        n_hat = normal_from_gradient(*grad)
        lam = lambda_phi(phi, self.width)
        if self.gravity_mode == "rho":
            rho_fm = rho * fm.to(rho.dtype)
            gz = self.gravity[2] * rho_fm
            gy = self.gravity[1] * rho_fm
            gx2 = self.gravity[0] * rho_fm
        else:
            gz, gy, gx2 = (self.gravity[2] * fm, self.gravity[1] * fm, self.gravity[0] * fm)
        F = (
            mu_phi * grad[0] + gx2,
            mu_phi * grad[1] + gy,
            mu_phi * grad[2] + gz,
        )  # Eq.(5)+body
        u, p, u_star = macroscopic_g(self.g, rho, grad, self.drho, F, self.w0)
        phi_u_now = tuple(phi * u[a] for a in range(3))
        phi_u_dt = tuple(phi_u_now[a] - self.phi_u_prev[a] for a in range(3))
        self.phi_u_prev = phi_u_now

        feq = f_equilibrium(phi, u, self.wv, self.cf)
        fsrc = f_source(phi_u_dt, lam, n_hat, self.tau_f, self.wv, self.cf)
        self.f = self.f + self.inv_tau_f * (feq - self.f) + self.fm_f * fsrc

        geq = g_equilibrium(p, rho, u, self.wv, self.cf, self.w0)
        gsrc = g_force(F, u, grad, self.drho, inv_tau_g, self.wv, self.cf)
        self.g = self.g + inv_tau_g * (geq - self.g) + self.fm_f * gsrc

        if self.axes == "ppp":
            self.f = stream_d3q19_adapter(self.f, boundary="periodic")
            self.g = stream_d3q19_adapter(self.g, boundary="periodic")
        else:
            self.f = stream_mixed(self.f, axes=self.axes)
            self.g = stream_mixed(self.g, axes=self.axes)
        if self.wall_mask is not None:
            self.f = bounce_back_cells_3d(self.f, self.wall_mask)
            self.g = bounce_back_cells_3d(self.g, self.wall_mask)
        self.step_count += 1


# ---------------------------------------------------------------------------
# Initial conditions, [L1] Eq.(30) droplet / Eq.(33) layered
# ---------------------------------------------------------------------------


def _unit_views(nz: int, ny: int, nx: int, dtype, device):
    c = _c_float(dtype, device)
    w = _weights(dtype, device).view(19, 1, 1, 1)
    cf = [c[:, a].view(19, 1, 1, 1) for a in range(3)]
    return w, c, cf


def init_phi_droplet(
    shape: tuple[int, int, int],
    radius: float,
    width: float,
    center: tuple[float, float, float] | None = None,
    device=torch.device("cpu"),
    dtype=torch.float32,
) -> torch.Tensor:
    """Eq.(30): phi = 0.5 + 0.5 tanh(2 (R - r) / W)  (phi=1 inside)."""
    nz, ny, nx = shape
    cz, cy, cx = center if center is not None else (nz / 2.0, ny / 2.0, nx / 2.0)
    z = torch.arange(nz, device=device, dtype=dtype).view(-1, 1, 1)
    y = torch.arange(ny, device=device, dtype=dtype).view(1, -1, 1)
    x = torch.arange(nx, device=device, dtype=dtype).view(1, 1, -1)
    r2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
    d = radius - r2.sqrt()
    return 0.5 + 0.5 * torch.tanh(2.0 * d / width)


def init_phi_layers(
    shape: tuple[int, int, int],
    width: float,
    y0: float,
    device=torch.device("cpu"),
    dtype=torch.float32,
) -> torch.Tensor:
    """Eq.(33): phi = 0.5 + 0.5 tanh(2 (y0 - y) / W)  (phi=1 liquid, y <= y0)."""
    nz, ny, nx = shape
    y = torch.arange(ny, device=device, dtype=dtype).view(1, -1, 1)
    return 0.5 + 0.5 * torch.tanh(2.0 * (y0 - y) / width).expand(nz, ny, nx).contiguous()


# ---------------------------------------------------------------------------
# fp64 diagnostics: recompute the full macro chain in float64 from f, g
# ---------------------------------------------------------------------------


def _macro_fp64(sim: "CACSim"):
    """fp64 recompute of the full macro chain from f, g.  W10-M1: the body
    force now honours sim.gravity_mode ("rho" -> rho * g * fluid_mask),
    i.e. the same functional form as the stepping force in step()
    (prereg 3.3; the W9 version always used the uniform form, biasing B5
    U readouts by 0.5*G_ACC*(1-rho_i)/rho_i, quantified per-arm there)."""
    f64 = torch.float64
    dev = sim.device
    phi = sim.f.to(f64).sum(dim=0)
    rho = rho_of_phi(phi, sim.rho_l, sim.rho_g)
    nb = shifted_neighbors(phi, axes=sim.axes)
    grad = iso_gradient_3d(nb, dev)
    mu_phi = chemical_potential(phi, nb, sim.beta, sim.k)
    if sim.gravity_mode == "rho":
        rho_fm = rho * sim.fluid_mask.to(f64)
        F = (
            mu_phi * grad[0] + sim.gravity[0] * rho_fm,
            mu_phi * grad[1] + sim.gravity[1] * rho_fm,
            mu_phi * grad[2] + sim.gravity[2] * rho_fm,
        )
    else:
        F = (
            mu_phi * grad[0] + sim.gravity[0] * sim.fluid_mask.to(f64),
            mu_phi * grad[1] + sim.gravity[1] * sim.fluid_mask.to(f64),
            mu_phi * grad[2] + sim.gravity[2] * sim.fluid_mask.to(f64),
        )
    u, p, u_star = macroscopic_g(sim.g.to(f64), rho, grad, sim.drho, F, sim.w0)
    return phi, rho, grad, mu_phi, F, u, p, u_star


def diagnostics(sim: "CACSim") -> dict:
    """Generic fp64 health/mass diagnostics (prereg out/*.json core block)."""
    phi, rho, grad, mu_phi, F, u, p, u_star = _macro_fp64(sim)
    fm = sim.fluid_mask
    phi_f = phi[fm]
    gnorm = (grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2).sqrt()
    umax = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt()[fm].max().item()
    return {
        "step": sim.step_count,
        "phi_integral_fluid": phi_f.sum().item(),
        "phi_integral_total": phi.sum().item(),
        "phi_min_fluid": phi_f.min().item(),
        "phi_max_fluid": phi_f.max().item(),
        "rho_min_fluid": rho[fm].min().item(),
        "rho_max_fluid": rho[fm].max().item(),
        "u_max_fluid": umax,
        "grad_phi_max": gnorm[fm].max().item(),
        "nan_f": int(torch.isnan(sim.f).sum().item()),
        "nan_g": int(torch.isnan(sim.g).sum().item()),
    }


def run(
    sim: "CACSim",
    n_steps: int,
    sample_every: int = 100,
    on_sample=None,
    progress_every: int = 5000,
) -> list[dict]:
    """Drive ``sim`` for ``n_steps``; collect :func:`diagnostics` samples."""
    import time as _time

    diags: list[dict] = []
    t0 = _time.time()
    for it in range(n_steps):
        sim.step()
        n = it + 1
        if n % sample_every == 0 or n == 1:
            d = diagnostics(sim)
            d["wall_time_s"] = _time.time() - t0
            diags.append(d)
            if on_sample is not None:
                on_sample(sim, d)
        if progress_every and n % progress_every == 0:
            dt = (_time.time() - t0) / n
            print(f"[cac] step {n}/{n_steps}  {dt * 1e3:.1f} ms/step", flush=True)
    return diags
