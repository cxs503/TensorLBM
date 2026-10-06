"""XH-1 hybrid Color-Gradient x CAC multiphase engine (task #234, Phase 1).

Architecture (approved 2026-10-06T11:05Z, prereg md5 80513afc...):

* Momentum skeleton: the two-relaxation-time D2Q9 Color-Gradient step of
  ``tensorlbm.color_gradient2d.color_gradient_two_tau_step`` (Latva-Kokko
  recolor with segregation ``beta`` pinned at 0.9, per-color ``tau``, exact
  per-color mass ledger), with the anisotropic A-perturbation REMOVED
  (A = 0 fixed: the perturbation term is absent from the code, which is
  arithmetically identical to the library branch at A == 0.0, as receipted
  bitwise in probe P1 over 5000 steps and re-receipted for this module by
  the Phase-1 t0 fixture).
* Surface tension: CAC chemical-potential force ``F = mu_phi grad(phi01)``
  with ``mu_phi = 4 beta_fe phi01 (phi01-1)(phi01-0.5) - k lap(phi01)``,
  ``beta_fe = 12 sigma / W``, ``k = 1.5 sigma W`` (the library
  ``cac_lbm`` free-energy mapping), applied through the Guo channel as
  per-mass acceleration fields.
* Pressure-form upgrade (T3/T6, the R5 probe target): the momentum
  populations' equilibrium carries a mass-free isotropic pressure
  correction Psi so the mechanical pressure becomes
  ``p = cs^2 rho + Psi = p0 + G(phi) - (k/2)|grad phi|^2 + cs^2 (rho - rho_t(phi))``
  with ``G(phi) = beta_fe phi^2 (1-phi)^2`` and the linear target density
  ``rho_t(phi) = rho_b_bulk + phi (rho_r_bulk - rho_b_bulk)`` (the CAC
  ``rho_of_phi`` mixing).  Bulk regions sit at equal pressure ``p0`` at
  their natural densities (kills the cs^2 drho collapse driver, probe P3),
  density deviations are restored acoustically, and the flat-interface
  discrete force balance is exact through the Korteweg term.  The
  correction is ported from the structure of ``cac_lbm.g_equilibrium``
  (geq_0 = (p/cs2)(w0-1) + rho s_0;  geq_i = (p/cs2) w_i + rho s_i):
  here corr_i = w_i Psi/cs2 (i != 0), corr_0 = (w0-1) Psi/cs2, which has
  zero zeroth moment (per-color mass ledger untouched to round-off) and
  second moment ``Psi * I``.
* 3D: D3Q19 twins of the same step using the library ``cac_lbm``
  isotropic stencils (``shifted_neighbors`` / ``iso_gradient_3d`` /
  ``iso_laplacian_3d``, Liang et al. Eq.(27)/(28), natively 3D), the
  library D3Q19 constants (``C`` / ``W_EXACT64``), and streaming /
  bounce-back left to the caller (library ``stream_d3q19_adapter`` +
  ``bounce_back_cells_3d``, the validated CAC wall pattern).

Conventions
-----------
* 2D tensors are ``(9, ny, nx)`` (library d2q9 convention); 3D tensors are
  ``(19, nz, ny, nx)`` (library d3q19 convention, x = axis 2).
* ``gxf``/``gyf``/``gzf`` are per-mass ACCELERATION fields (force density
  divided by total density); ``None`` means exact zeros (bitwise-identical
  arithmetic to the library's scalar-zero path).
* phi01 = (1 + (rho_r - rho_b)/rho)/2 in [0,1]; phi01 = 1 is pure red.
"""

from __future__ import annotations

import torch

from .cac_lbm import (
    free_energy_params,
    iso_gradient_3d,
    iso_laplacian_3d,
    shifted_neighbors,
)
from .d2q9 import C as C_2D
from .d3q19 import C as C_3D
from .d3q19 import W_EXACT64

CS2 = 1.0 / 3.0

#: Recolor/segregation parameter pinned by prereg (library frozen-run value).
BETA_PINNED = 0.9

__all__ = [
    "CS2",
    "BETA_PINNED",
    "iso_grad2d",
    "iso_lap2d",
    "chem_force2d",
    "chem_force3d",
    "phi01_of",
    "mechanical_pressure_2d",
    "mechanical_pressure_3d",
    "pressure_correction_2d",
    "pressure_correction_3d",
    "hybrid_two_tau_step",
    "hybrid_two_tau_step_3d",
    "init_droplet_fields_2d",
    "init_droplet_fields_3d",
    "init_layers_fields_2d",
    "droplet_sigma_series_2d",
    "droplet_diagnostics_3d",
]


# ---------------------------------------------------------------------------
# D2Q9 isotropic stencils (2-D collapse of cac_lbm's Eq.(27)/(28); the
# construction is identical to the Phase-0 probe xh_common, which was
# self-tested against analytic periodic waves and bitwise-compared with the
# library functional forms; re-receipted by the Phase-1 t0 fixture).
# ---------------------------------------------------------------------------

_DIRS2D = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
           (1, 1), (1, -1), (-1, 1), (-1, -1)]
_W2D = [4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4
_SUM_W_NONREST_2D = 5.0 / 9.0


def _shift2d(field: torch.Tensor, cx: int, cy: int) -> torch.Tensor:
    """phi(x + c) via periodic roll (library operator convention)."""
    return torch.roll(field, shifts=(-cy, -cx), dims=(0, 1))


def iso_grad2d(phi: torch.Tensor):
    """grad = 3 sum_{i!=0} w_i c_i phi(x + c_i)  (Liang Eq.(27), D2Q9)."""
    gx = torch.zeros_like(phi)
    gy = torch.zeros_like(phi)
    for (cx, cy), w in zip(_DIRS2D[1:], _W2D[1:]):
        nb = _shift2d(phi, cx, cy)
        gx = gx + w * cx * nb
        gy = gy + w * cy * nb
    return 3.0 * gx, 3.0 * gy


def iso_lap2d(phi: torch.Tensor) -> torch.Tensor:
    """lap = 6 (sum_{i!=0} w_i phi(x+c_i) - (5/9) phi)  (Liang Eq.(28), D2Q9)."""
    acc = torch.zeros_like(phi)
    for (cx, cy), w in zip(_DIRS2D[1:], _W2D[1:]):
        acc = acc + w * _shift2d(phi, cx, cy)
    return 6.0 * (acc - _SUM_W_NONREST_2D * phi)


def stencil_selftest_2d(n: int = 64, period: float = 16.0) -> dict:
    """Assert the 2-D iso stencils against analytic periodic waves."""
    y, x = torch.meshgrid(
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        indexing="ij",
    )
    k = 2.0 * torch.pi / period
    errs = {}
    for tag, (fld, gex, gey, lapex) in {
        "x-wave": (
            torch.cos(k * x),
            -k * torch.sin(k * x),
            torch.zeros_like(x),
            -(k ** 2) * torch.cos(k * x),
        ),
        "y-wave": (
            torch.cos(k * y),
            torch.zeros_like(y),
            -k * torch.sin(k * y),
            -(k ** 2) * torch.cos(k * y),
        ),
    }.items():
        gx, gy = iso_grad2d(fld)
        lap = iso_lap2d(fld)
        errs[tag] = {
            "grad_x": float((gx - gex).abs().max()),
            "grad_y": float((gy - gey).abs().max()),
            "lap": float((lap - lapex).abs().max()),
        }
        assert errs[tag]["grad_x"] < 2e-2, (tag, errs[tag])
        assert errs[tag]["grad_y"] < 2e-2, (tag, errs[tag])
        assert errs[tag]["lap"] < 1e-2, (tag, errs[tag])
    return errs


# ---------------------------------------------------------------------------
# Chemical-potential surface force (identical functional form to
# cac_lbm.chemical_potential + Eq.(5) force, on the [0,1] order parameter)
# ---------------------------------------------------------------------------


def phi01_of(f_r: torch.Tensor, f_b: torch.Tensor) -> torch.Tensor:
    rho_r = f_r.sum(dim=0)
    rho_b = f_b.sum(dim=0)
    rho = (rho_r + rho_b).clamp(min=1e-12)
    return 0.5 * (1.0 + (rho_r - rho_b) / rho)


def chem_force2d(phi01: torch.Tensor, sigma: float, width: float):
    """(Fx, Fy) = mu_phi grad(phi01) per unit volume (2-D, periodic)."""
    beta_fe, k = free_energy_params(sigma, width)
    lap = iso_lap2d(phi01)
    mu = 4.0 * beta_fe * phi01 * (phi01 - 1.0) * (phi01 - 0.5) - k * lap
    gx, gy = iso_grad2d(phi01)
    return mu * gx, mu * gy


def chem_force3d(phi01: torch.Tensor, sigma: float, width: float, axes: str = "ppp"):
    """(Fx, Fy, Fz) = mu_phi grad(phi01) per unit volume (3-D).

    Uses the library ``cac_lbm`` D3Q19 isotropic stencils natively.
    """
    beta_fe, k = free_energy_params(sigma, width)
    nb = shifted_neighbors(phi01, axes=axes)
    mu = 4.0 * beta_fe * phi01 * (phi01 - 1.0) * (phi01 - 0.5) \
        - k * iso_laplacian_3d(phi01, nb)
    grad = iso_gradient_3d(nb, phi01.device)
    return tuple(mu * g for g in grad)


# ---------------------------------------------------------------------------
# Pressure-form upgrade pieces (R5 / T3 / T6)
# ---------------------------------------------------------------------------


def _gphi(phi01: torch.Tensor, beta_fe: float) -> torch.Tensor:
    """Bulk free-energy pressure G(phi) = beta_fe phi^2 (1-phi)^2 (G(0)=G(1)=0)."""
    return beta_fe * phi01 * phi01 * (1.0 - phi01) * (1.0 - phi01)


def _rho_target(phi01: torch.Tensor, rho_r_bulk: float, rho_b_bulk: float) -> torch.Tensor:
    """Linear target total density (cac_lbm.rho_of_phi mixing law)."""
    return rho_b_bulk + phi01 * (rho_r_bulk - rho_b_bulk)


def _resolve_p0(p0, rho_r_bulk: float, rho_b_bulk: float) -> float:
    """Background pressure; ``None`` -> cs^2 * max(bulk) (SL-3 fix attempt 2).

    Rationale (r5diag_2d evidence): the Latva-Kokko recolor demand
    beta*(rho_r*rho_b/rho)*(c.n)w_i peaks at the interface midpoint at
    0.9*(rho_max/4)*sqrt(2) per direction, INDEPENDENT of how the
    pressure-form sets the isotropic population level p/cs^2.  With a low
    p0 the supply p/cs^2 ~ p0/cs^2 under-fills the populations and the
    recolor drives them negative (equal-rho 1.06x, rho2 1.59x, rho10 5.8x
    overdrive -- the d0/d1/rho10-NaN signature).  p0 = cs^2*rho_max makes
    the worst per-direction demand/supply ratio 0.318 for ANY density
    pair -- exactly the margin the plain mode runs with (T1 receipt).
    """
    if p0 is None:
        return CS2 * max(rho_r_bulk, rho_b_bulk)
    return float(p0)


def pressure_correction_2d(
    phi01: torch.Tensor,
    sigma: float,
    width: float,
    rho_r_bulk: float,
    rho_b_bulk: float,
    p0: float | None = None,
    korteweg: bool = True,
) -> torch.Tensor:
    """Psi such that p = cs^2 rho + Psi (see module docstring)."""
    p0 = _resolve_p0(p0, rho_r_bulk, rho_b_bulk)
    beta_fe, k = free_energy_params(sigma, width)
    psi = p0 + _gphi(phi01, beta_fe) - CS2 * _rho_target(phi01, rho_r_bulk, rho_b_bulk)
    if korteweg:
        gx, gy = iso_grad2d(phi01)
        psi = psi - 0.5 * k * (gx * gx + gy * gy)
    return psi


def pressure_correction_3d(
    phi01: torch.Tensor,
    sigma: float,
    width: float,
    rho_r_bulk: float,
    rho_b_bulk: float,
    p0: float | None = None,
    korteweg: bool = True,
    axes: str = "ppp",
) -> torch.Tensor:
    p0 = _resolve_p0(p0, rho_r_bulk, rho_b_bulk)
    beta_fe, k = free_energy_params(sigma, width)
    psi = p0 + _gphi(phi01, beta_fe) - CS2 * _rho_target(phi01, rho_r_bulk, rho_b_bulk)
    if korteweg:
        nb = shifted_neighbors(phi01, axes=axes)
        gx, gy, gz = iso_gradient_3d(nb, phi01.device)
        psi = psi - 0.5 * k * (gx * gx + gy * gy + gz * gz)
    return psi


def mechanical_pressure_2d(
    f_r: torch.Tensor,
    f_b: torch.Tensor,
    pform: dict | None,
) -> torch.Tensor:
    """Mechanical pressure field p = cs^2 rho (+ Psi in pressure-form mode).

    Plain mode returns cs^2 rho (the M1/T1 Laplace convention); pressure
    mode returns cs^2 rho + Psi with Psi rebuilt from the CURRENT phi01
    (the same functional form the equilibrium used).
    """
    rho = (f_r + f_b).sum(dim=0)
    p = CS2 * rho
    if pform is not None:
        psi = pressure_correction_2d(
            phi01_of(f_r, f_b),
            pform["sigma"], pform["width"],
            pform["rho_r_bulk"], pform["rho_b_bulk"],
            pform.get("p0"), pform.get("korteweg", True),
        )
        p = p + psi
    return p


def mechanical_pressure_3d(
    f_r: torch.Tensor,
    f_b: torch.Tensor,
    pform: dict | None,
) -> torch.Tensor:
    rho = (f_r + f_b).sum(dim=0)
    p = CS2 * rho
    if pform is not None:
        psi = pressure_correction_3d(
            phi01_of(f_r, f_b),
            pform["sigma"], pform["width"],
            pform["rho_r_bulk"], pform["rho_b_bulk"],
            pform.get("p0"), pform.get("korteweg", True),
            pform.get("axes", "ppp"),
        )
        p = p + psi
    return p


def _pressure_corr_populations(psi: torch.Tensor, w_v: torch.Tensor) -> torch.Tensor:
    """corr_i = w_i Psi/cs2 (i != 0); corr_0 = (w0 - 1) Psi/cs2.

    Zero zeroth moment (mass-free) and second moment ``Psi * I``: the
    ``cac_lbm.g_equilibrium`` structure applied as an additive correction
    on top of the mass-carrying CG equilibrium.
    """
    corr = (w_v * (psi / CS2)).clone()
    corr[0] = corr[0] - psi / CS2
    return corr


# ---------------------------------------------------------------------------
# 2-D stepper
# ---------------------------------------------------------------------------


def _cg_phase_gradient_2d(rho_r, rho_b):
    n = rho_r + rho_b
    n_safe = torch.clamp(n, min=1e-12)
    phi = (rho_r - rho_b) / n_safe
    dphi_dx = 0.5 * (torch.roll(phi, -1, dims=-1) - torch.roll(phi, 1, dims=-1))
    dphi_dy = 0.5 * (torch.roll(phi, -1, dims=-2) - torch.roll(phi, 1, dims=-2))
    mag = torch.sqrt(dphi_dx ** 2 + dphi_dy ** 2)
    mag_safe = torch.clamp(mag, min=1e-12)
    return phi, dphi_dx / mag_safe, dphi_dy / mag_safe, mag


def hybrid_two_tau_step(
    f_r: torch.Tensor,
    f_b: torch.Tensor,
    tau_r: float = 1.0,
    tau_b: float = 1.0,
    beta: float = BETA_PINNED,
    gxf: torch.Tensor | None = None,
    gyf: torch.Tensor | None = None,
    solid_mask: torch.Tensor | None = None,
    visc_mode: str = "mixture",
    pform: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One XH-1 collision step (2-D; streaming is external).

    Port of ``color_gradient_two_tau_step`` with (i) the A-perturbation
    removed (A = 0 fixed), (ii) scalar gx/gy generalized to per-mass
    acceleration FIELDS, (iii) optional pressure-form equilibrium
    correction.  With ``pform=None`` and constant fields the arithmetic is
    bitwise-identical to the library step at A=0 (probe P1 receipt,
    re-receipted by t0).

    ``pform`` dict fields: sigma, width, rho_r_bulk, rho_b_bulk,
    p0 (default 0), korteweg (default True).
    """
    if f_r.shape != f_b.shape:
        raise ValueError(f"shape mismatch: {tuple(f_r.shape)} vs {tuple(f_b.shape)}")
    if tau_r <= 0.5 or tau_b <= 0.5:
        raise ValueError(f"tau must exceed 0.5 (tau_r={tau_r}, tau_b={tau_b})")
    if visc_mode not in ("mixture", "per_color"):
        raise ValueError(f"visc_mode must be 'mixture' or 'per_color', got {visc_mode!r}")

    device = f_r.device
    c = C_2D.to(device=device)
    cx = c[:, 0].to(f_r.dtype).view(9, 1, 1)
    cy = c[:, 1].to(f_r.dtype).view(9, 1, 1)
    w_v = torch.tensor(_W2D, dtype=f_r.dtype, device=device).view(9, 1, 1)
    zero = torch.zeros_like(f_r.sum(dim=0))
    gxf = zero if gxf is None else gxf
    gyf = zero if gyf is None else gyf

    rho_r = f_r.sum(dim=0)
    rho_b = f_b.sum(dim=0)
    rho = rho_r + rho_b
    rho_s = rho.clamp(min=1e-12)
    jx = ((f_r + f_b) * cx).sum(dim=0)
    jy = ((f_r + f_b) * cy).sum(dim=0)
    ux = jx / rho_s
    uy = jy / rho_s

    ux_eq = ux + 0.5 * gxf
    uy_eq = uy + 0.5 * gyf

    cu = cx * ux_eq.unsqueeze(0) + cy * uy_eq.unsqueeze(0)
    u_sq = (ux_eq ** 2 + uy_eq ** 2).unsqueeze(0)

    corr = None
    if pform is not None:
        psi = pressure_correction_2d(
            phi01_of(f_r, f_b),
            pform["sigma"], pform["width"],
            pform["rho_r_bulk"], pform["rho_b_bulk"],
            pform.get("p0"), pform.get("korteweg", True),
        )
        corr = _pressure_corr_populations(psi, w_v)

    def _feq(rho_field: torch.Tensor) -> torch.Tensor:
        feq = w_v * rho_field.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq)
        return feq if corr is None else feq + corr

    def _guo(rho_field: torch.Tensor, tau_field) -> torch.Tensor:
        fx = rho_field * gxf
        fy = rho_field * gyf
        ci_f = cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0)
        u_f = (ux_eq * fx + uy_eq * fy).unsqueeze(0)
        src = w_v * (3.0 * ci_f + 9.0 * cu * ci_f - 3.0 * u_f)
        return (1.0 - 0.5 / tau_field) * src

    if visc_mode == "mixture":
        tau_eff = (rho_r * tau_r + rho_b * tau_b) / rho_s
        f_post = (f_r + f_b) - ((f_r + f_b) - _feq(rho)) / tau_eff + _guo(rho, tau_eff)
    else:
        def _collide(f_sig, rho_sig, tau, corr_sig):
            feq = w_v * rho_sig.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq)
            if corr_sig is not None:
                feq = feq + corr_sig
            return f_sig - (f_sig - feq) / tau + _guo(rho_sig, tau)

        if corr is None:
            f_post = _collide(f_r, rho_r, tau_r, None) + _collide(f_b, rho_b, tau_b, None)
        else:
            frac_r = (rho_r / rho_s).unsqueeze(0)
            frac_b = (rho_b / rho_s).unsqueeze(0)
            f_post = (
                _collide(f_r, rho_r, tau_r, corr * frac_r)
                + _collide(f_b, rho_b, tau_b, corr * frac_b)
            )

    # Masked densities avoid spurious color gradients at solid boundaries
    # (stock convention).
    rho_r_safe = rho_r if solid_mask is None else rho_r.masked_fill(solid_mask, 0.0)
    rho_b_safe = rho_b if solid_mask is None else rho_b.masked_fill(solid_mask, 0.0)
    _phi, nhat_x, nhat_y, _mag = _cg_phase_gradient_2d(rho_r_safe, rho_b_safe)

    ci_dot_n = cx * nhat_x.unsqueeze(0) + cy * nhat_y.unsqueeze(0)

    recolor_amp = beta * (rho_r * rho_b / rho_s).unsqueeze(0) * ci_dot_n * w_v
    f_r_out = (rho_r / rho_s).unsqueeze(0) * f_post + recolor_amp
    f_b_out = (rho_b / rho_s).unsqueeze(0) * f_post - recolor_amp

    if solid_mask is not None:
        mask_3d = solid_mask.unsqueeze(0)
        f_r_out = torch.where(mask_3d, f_r, f_r_out)
        f_b_out = torch.where(mask_3d, f_b, f_b_out)

    return f_r_out, f_b_out


# ---------------------------------------------------------------------------
# 3-D stepper (D3Q19; streaming + bounce-back external, library calls)
# ---------------------------------------------------------------------------


def _cg_phase_gradient_3d(rho_r, rho_b):
    n = rho_r + rho_b
    n_safe = torch.clamp(n, min=1e-12)
    phi = (rho_r - rho_b) / n_safe
    dphi_dx = 0.5 * (torch.roll(phi, -1, dims=-1) - torch.roll(phi, 1, dims=-1))
    dphi_dy = 0.5 * (torch.roll(phi, -1, dims=-2) - torch.roll(phi, 1, dims=-2))
    dphi_dz = 0.5 * (torch.roll(phi, -1, dims=-3) - torch.roll(phi, 1, dims=-3))
    mag = torch.sqrt(dphi_dx ** 2 + dphi_dy ** 2 + dphi_dz ** 2)
    mag_safe = torch.clamp(mag, min=1e-12)
    return (phi, dphi_dx / mag_safe, dphi_dy / mag_safe, dphi_dz / mag_safe, mag)


def hybrid_two_tau_step_3d(
    f_r: torch.Tensor,
    f_b: torch.Tensor,
    tau_r: float = 1.0,
    tau_b: float = 1.0,
    beta: float = BETA_PINNED,
    gxf: torch.Tensor | None = None,
    gyf: torch.Tensor | None = None,
    gzf: torch.Tensor | None = None,
    solid_mask: torch.Tensor | None = None,
    visc_mode: str = "mixture",
    pform: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One XH-1 collision step (3-D, D3Q19; streaming is external).

    Same structure as :func:`hybrid_two_tau_step` on the D3Q19 lattice:
    mixture-velocity two-tau collision (Guo half-step shift + full Guo
    source with F = rho * g fields), no A-perturbation, Latva-Kokko
    recolor along the central-difference color-gradient normal, optional
    pressure-form equilibrium correction.

    ``pform`` additionally honours ``axes`` (default "ppp") for the
    near-wall stencil policy of the Korteweg term.
    """
    if f_r.shape != f_b.shape or f_r.dim() != 4 or f_r.shape[0] != 19:
        raise ValueError(f"expect (19, nz, ny, nx) pair, got {tuple(f_r.shape)}")
    if tau_r <= 0.5 or tau_b <= 0.5:
        raise ValueError(f"tau must exceed 0.5 (tau_r={tau_r}, tau_b={tau_b})")
    if visc_mode not in ("mixture", "per_color"):
        raise ValueError(f"visc_mode must be 'mixture' or 'per_color', got {visc_mode!r}")

    device = f_r.device
    dt = f_r.dtype
    c = C_3D.to(device=device)
    cx = c[:, 0].to(dt).view(19, 1, 1, 1)
    cy = c[:, 1].to(dt).view(19, 1, 1, 1)
    cz = c[:, 2].to(dt).view(19, 1, 1, 1)
    w_v = W_EXACT64.to(device=device, dtype=dt).view(19, 1, 1, 1)
    zero = torch.zeros_like(f_r.sum(dim=0))
    gxf = zero if gxf is None else gxf
    gyf = zero if gyf is None else gyf
    gzf = zero if gzf is None else gzf

    rho_r = f_r.sum(dim=0)
    rho_b = f_b.sum(dim=0)
    rho = rho_r + rho_b
    rho_s = rho.clamp(min=1e-12)
    jx = ((f_r + f_b) * cx).sum(dim=0)
    jy = ((f_r + f_b) * cy).sum(dim=0)
    jz = ((f_r + f_b) * cz).sum(dim=0)
    ux = jx / rho_s
    uy = jy / rho_s
    uz = jz / rho_s

    ux_eq = ux + 0.5 * gxf
    uy_eq = uy + 0.5 * gyf
    uz_eq = uz + 0.5 * gzf

    cu = cx * ux_eq.unsqueeze(0) + cy * uy_eq.unsqueeze(0) + cz * uz_eq.unsqueeze(0)
    u_sq = (ux_eq ** 2 + uy_eq ** 2 + uz_eq ** 2).unsqueeze(0)

    corr = None
    if pform is not None:
        psi = pressure_correction_3d(
            phi01_of(f_r, f_b),
            pform["sigma"], pform["width"],
            pform["rho_r_bulk"], pform["rho_b_bulk"],
            pform.get("p0"), pform.get("korteweg", True),
            pform.get("axes", "ppp"),
        )
        corr = _pressure_corr_populations(psi, w_v)

    def _base_feq(rho_field: torch.Tensor) -> torch.Tensor:
        return w_v * rho_field.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq)

    def _guo(rho_field: torch.Tensor, tau_field) -> torch.Tensor:
        fx = rho_field * gxf
        fy = rho_field * gyf
        fz = rho_field * gzf
        ci_f = cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0) + cz * fz.unsqueeze(0)
        u_f = (ux_eq * fx + uy_eq * fy + uz_eq * fz).unsqueeze(0)
        src = w_v * (3.0 * ci_f + 9.0 * cu * ci_f - 3.0 * u_f)
        return (1.0 - 0.5 / tau_field) * src

    if visc_mode == "mixture":
        tau_eff = (rho_r * tau_r + rho_b * tau_b) / rho_s
        feq = _base_feq(rho)
        if corr is not None:
            feq = feq + corr
        f_post = (f_r + f_b) - ((f_r + f_b) - feq) / tau_eff + _guo(rho, tau_eff)
    else:
        def _collide(f_sig, rho_sig, tau, corr_sig):
            feq = _base_feq(rho_sig)
            if corr_sig is not None:
                feq = feq + corr_sig
            return f_sig - (f_sig - feq) / tau + _guo(rho_sig, tau)

        if corr is None:
            f_post = _collide(f_r, rho_r, tau_r, None) + _collide(f_b, rho_b, tau_b, None)
        else:
            frac_r = (rho_r / rho_s).unsqueeze(0)
            frac_b = (rho_b / rho_s).unsqueeze(0)
            f_post = (
                _collide(f_r, rho_r, tau_r, corr * frac_r)
                + _collide(f_b, rho_b, tau_b, corr * frac_b)
            )

    rho_r_safe = rho_r if solid_mask is None else rho_r.masked_fill(solid_mask, 0.0)
    rho_b_safe = rho_b if solid_mask is None else rho_b.masked_fill(solid_mask, 0.0)
    _phi, nhat_x, nhat_y, nhat_z, _mag = _cg_phase_gradient_3d(rho_r_safe, rho_b_safe)

    ci_dot_n = cx * nhat_x.unsqueeze(0) + cy * nhat_y.unsqueeze(0) + cz * nhat_z.unsqueeze(0)

    recolor_amp = beta * (rho_r * rho_b / rho_s).unsqueeze(0) * ci_dot_n * w_v
    f_r_out = (rho_r / rho_s).unsqueeze(0) * f_post + recolor_amp
    f_b_out = (rho_b / rho_s).unsqueeze(0) * f_post - recolor_amp

    if solid_mask is not None:
        mask_4d = solid_mask.unsqueeze(0)
        f_r_out = torch.where(mask_4d, f_r, f_r_out)
        f_b_out = torch.where(mask_4d, f_b, f_b_out)

    return f_r_out, f_b_out


# ---------------------------------------------------------------------------
# Initial conditions (XH battery convention: frac = 0.5 (1 + tanh(d/w_init)),
# w_init = 2.5; pressure-form aware so the rest state is collision-consistent)
# ---------------------------------------------------------------------------


def _equilibrium_rest_2d(rho_field: torch.Tensor, psi: torch.Tensor | None, dtype, device):
    """w_i rho (+ pressure corr) at u = 0, shape (9, ny, nx)."""
    w_v = torch.tensor(_W2D, dtype=dtype, device=device).view(9, 1, 1)
    f = w_v * rho_field.unsqueeze(0)
    if psi is not None:
        f = f + _pressure_corr_populations(psi, w_v)
    return f


def _equilibrium_rest_3d(rho_field: torch.Tensor, psi: torch.Tensor | None, dtype, device):
    w_v = W_EXACT64.to(device=device, dtype=dtype).view(19, 1, 1, 1)
    f = w_v * rho_field.unsqueeze(0)
    if psi is not None:
        f = f + _pressure_corr_populations(psi, w_v)
    return f


def init_droplet_fields_2d(
    n: int,
    radius: float,
    w_init: float,
    *,
    rho_r_bulk: float = 1.0,
    rho_b_bulk: float = 1.0,
    pform: dict | None = None,
    dtype=torch.float64,
    device=torch.device("cpu"),
):
    """Periodic 2-D droplet: red inside, blue outside (P1 convention).

    At equal bulk densities the red/blue densities are rho_r_bulk * frac
    and rho_b_bulk * (1 - frac) (the P1 equal-mass two-color init).  At a
    density contrast the per-color targets follow phi01: rho_r =
    phi01 * rho_t(phi01), rho_b = (1 - phi01) * rho_t(phi01) with
    rho_t the linear mixing (CAC rho_of_phi), and with ``pform`` the rest
    populations include the pressure correction.
    """
    ys = torch.arange(n, dtype=dtype, device=device).view(n, 1)
    xs = torch.arange(n, dtype=dtype, device=device).view(1, n)
    r = ((ys - n / 2.0) ** 2 + (xs - n / 2.0) ** 2).sqrt()
    frac = 0.5 * (1.0 + torch.tanh((radius - r) / w_init))
    if rho_r_bulk == rho_b_bulk:
        rho_r = rho_r_bulk * frac
        rho_b = rho_b_bulk * (1.0 - frac)
    else:
        rho_t = _rho_target(frac, rho_r_bulk, rho_b_bulk)
        rho_r = frac * rho_t
        rho_b = (1.0 - frac) * rho_t
    psi = None
    if pform is not None:
        psi = pressure_correction_2d(
            frac, pform["sigma"], pform["width"],
            pform["rho_r_bulk"], pform["rho_b_bulk"],
            pform.get("p0"), pform.get("korteweg", True),
        )
    return (
        _equilibrium_rest_2d(rho_r, psi, dtype, device),
        _equilibrium_rest_2d(rho_b, psi, dtype, device),
    )


def init_droplet_fields_3d(
    shape: tuple[int, int, int],
    radius: float,
    w_init: float,
    *,
    center: tuple[float, float, float] | None = None,
    rho_r_bulk: float = 1.0,
    rho_b_bulk: float = 1.0,
    pform: dict | None = None,
    dtype=torch.float64,
    device=torch.device("cpu"),
):
    """Periodic 3-D droplet (red sphere inside blue), XH convention."""
    nz, ny, nx = shape
    cz, cy, cx = center if center is not None else (nz / 2.0, ny / 2.0, nx / 2.0)
    z = torch.arange(nz, dtype=dtype, device=device).view(-1, 1, 1)
    y = torch.arange(ny, dtype=dtype, device=device).view(1, -1, 1)
    x = torch.arange(nx, dtype=dtype, device=device).view(1, 1, -1)
    r = ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2).sqrt()
    frac = 0.5 * (1.0 + torch.tanh((radius - r) / w_init))
    if rho_r_bulk == rho_b_bulk:
        rho_r = rho_r_bulk * frac
        rho_b = rho_b_bulk * (1.0 - frac)
    else:
        rho_t = _rho_target(frac, rho_r_bulk, rho_b_bulk)
        rho_r = frac * rho_t
        rho_b = (1.0 - frac) * rho_t
    psi = None
    if pform is not None:
        psi = pressure_correction_3d(
            frac, pform["sigma"], pform["width"],
            pform["rho_r_bulk"], pform["rho_b_bulk"],
            pform.get("p0"), pform.get("korteweg", True),
            pform.get("axes", "ppp"),
        )
    return (
        _equilibrium_rest_3d(rho_r, psi, dtype, device),
        _equilibrium_rest_3d(rho_b, psi, dtype, device),
    )


def init_layers_fields_2d(
    ny: int,
    nx: int,
    *,
    tau_mid: float | None = None,
    w_init: float = 2.0,
    rho_r_bulk: float = 1.0,
    rho_b_bulk: float = 1.0,
    pform: dict | None = None,
    dtype=torch.float64,
    device=torch.device("cpu"),
):
    """Horizontal layers, red on top (frozen poiseuille_two_phase_cg init).

    frac_r = 0.5 (1 - tanh((j - ny/2)/w_init)) with w_init = 2.0 (the
    frozen benchmark's init width).
    """
    j = torch.arange(ny, dtype=dtype, device=device).view(ny, 1)
    frac_r = 0.5 * (1.0 - torch.tanh((j - ny / 2.0) / w_init)).expand(ny, nx).contiguous()
    if rho_r_bulk == rho_b_bulk:
        rho_r = rho_r_bulk * frac_r
        rho_b = rho_b_bulk * (1.0 - frac_r)
    else:
        rho_t = _rho_target(frac_r, rho_r_bulk, rho_b_bulk)
        rho_r = frac_r * rho_t
        rho_b = (1.0 - frac_r) * rho_t
    psi = None
    if pform is not None:
        psi = pressure_correction_2d(
            frac_r, pform["sigma"], pform["width"],
            pform["rho_r_bulk"], pform["rho_b_bulk"],
            pform.get("p0"), pform.get("korteweg", True),
        )
    return (
        _equilibrium_rest_2d(rho_r, psi, dtype, device),
        _equilibrium_rest_2d(rho_b, psi, dtype, device),
    )


# ---------------------------------------------------------------------------
# Measurement helpers (M1/P1 conventions)
# ---------------------------------------------------------------------------


def droplet_sigma_series_2d(
    steps: int,
    radius: float,
    n: int,
    *,
    sigma: float,
    width: float,
    beta: float = BETA_PINNED,
    w_init: float = 2.5,
    rho_r_bulk: float = 1.0,
    rho_b_bulk: float = 1.0,
    pform: dict | None = None,
    sample_every: int = 500,
    dtype=torch.float64,
    device=torch.device("cpu"),
    progress_every: int = 0,
) -> dict:
    """Run one periodic 2-D droplet and return the sigma_rec time series.

    Conventions = probe P1 / M1: p = cs^2 rho in plain mode (mechanical
    pressure in pressure-form mode), bands phi01 > 0.9 / < 0.1, sigma_rec
    = dP * R (2-D cylinder), u_max spurious current, per-color mass ledger.
    """
    from .solver import stream

    f_r, f_b = init_droplet_fields_2d(
        n, radius, w_init, rho_r_bulk=rho_r_bulk, rho_b_bulk=rho_b_bulk,
        pform=pform, dtype=dtype, device=device,
    )
    m_r0, m_b0 = float(f_r.sum()), float(f_b.sum())
    c9 = C_2D.to(dtype).to(device)
    ys = torch.arange(n, dtype=dtype, device=device).view(n, 1)
    xs = torch.arange(n, dtype=dtype, device=device).view(1, n)
    r = ((ys - n / 2.0) ** 2 + (xs - n / 2.0) ** 2).sqrt()
    r_mask = r <= 0.7 * radius
    b_mask = r >= 1.3 * radius

    import time as _time
    t0 = _time.perf_counter()
    series = []
    for step in range(1, steps + 1):
        phi01 = phi01_of(f_r, f_b)
        rho = (f_r + f_b).sum(dim=0).clamp(min=1e-12)
        fx, fy = chem_force2d(phi01, sigma, width)
        gxf, gyf = fx / rho, fy / rho
        f_r, f_b = hybrid_two_tau_step(
            f_r, f_b, tau_r=1.0, tau_b=1.0, beta=beta,
            gxf=gxf, gyf=gyf, pform=pform,
        )
        f_r = stream(f_r)
        f_b = stream(f_b)
        if not bool(torch.isfinite(f_r).all() and torch.isfinite(f_b).all()):
            return {"stable": False, "nan_step": step, "series": series}
        if step % sample_every == 0 or step == 1:
            rho_r = f_r.sum(dim=0)
            rho_b = f_b.sum(dim=0)
            rho_t = (rho_r + rho_b).clamp(min=1e-12)
            phi01 = 0.5 * (1.0 + (rho_r - rho_b) / rho_t)
            p = mechanical_pressure_2d(f_r, f_b, pform)
            win, wout = phi01 > 0.9, phi01 < 0.1
            dp = (float(p[win].mean() - p[wout].mean())
                  if (win.any() and wout.any()) else float("nan"))
            f = f_r + f_b
            u = (f * c9[:, 0].view(9, 1, 1)).sum(dim=0) / rho_t
            v = (f * c9[:, 1].view(9, 1, 1)).sum(dim=0) / rho_t
            series.append({
                "step": step, "dP": dp, "sigma_rec": dp * radius,
                "u_max": float(torch.sqrt(u ** 2 + v ** 2).max()),
                "rho_r_in": float((rho_r * r_mask).sum() / r_mask.sum()),
                "rho_b_out": float((rho_b * b_mask).sum() / b_mask.sum()),
                "mass_r_rel": abs(float(f_r.sum()) - m_r0) / abs(m_r0),
                "mass_b_rel": abs(float(f_b.sum()) - m_b0) / abs(m_b0),
            })
        if progress_every and step % progress_every == 0:
            print(f"[xh-droplet2d] step {step}/{steps}", flush=True)
    return {
        "stable": True,
        "series": series,
        "mass_drift_r_per_step": abs(float(f_r.sum()) - m_r0) / steps / abs(m_r0),
        "mass_drift_b_per_step": abs(float(f_b.sum()) - m_b0) / steps / abs(m_b0),
        "wall_time_s": _time.perf_counter() - t0,
    }


def droplet_diagnostics_3d(f_r, f_b, pform, r_mask, b_mask, gravity_scale=1.0):
    """fp64 macro diagnostics for a 3-D two-color state (D3Q19)."""
    f64 = torch.float64
    fr64, fb64 = f_r.to(f64), f_b.to(f64)
    rho_r = fr64.sum(dim=0)
    rho_b = fb64.sum(dim=0)
    rho = (rho_r + rho_b).clamp(min=1e-12)
    phi01 = 0.5 * (1.0 + (rho_r - rho_b) / rho)
    c = C_3D.to(f64)
    jx = ((fr64 + fb64) * c[:, 0].view(19, 1, 1, 1)).sum(dim=0)
    jy = ((fr64 + fb64) * c[:, 1].view(19, 1, 1, 1)).sum(dim=0)
    jz = ((fr64 + fb64) * c[:, 2].view(19, 1, 1, 1)).sum(dim=0)
    p = mechanical_pressure_3d(fr64, fb64, pform)
    win, wout = phi01 > 0.9, phi01 < 0.1
    dP = (float(p[win].mean() - p[wout].mean())
          if (win.any() and wout.any()) else float("nan"))
    umax = float((jx ** 2 + jy ** 2 + jz ** 2).div(rho * rho).sqrt().max())
    return {
        "rho_r": rho_r, "rho_b": rho_b, "rho": rho, "phi01": phi01,
        "jx": jx, "jy": jy, "jz": jz,
        "u_max": umax, "dP": dP,
        "rho_r_in": float((rho_r * r_mask).sum() / r_mask.sum()),
        "rho_b_out": float((rho_b * b_mask).sum() / b_mask.sum()),
    }
