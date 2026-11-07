"""D2Q9 lattice-Boltzmann solver for a spinning cylinder (Magnus / Flettner rotor).

Physics scope
-------------
Two-dimensional, incompressible, single-phase D2Q9 lattice Boltzmann with

* a **Bouzidi--Firdaouss--Lallemand (BFL) interpolated moving-wall bounce-back**
  (Bouzidi et al. 2001) so that the curved cylinder surface carries the
  rigid-body rotation velocity ``u_w = omega x r`` without a staircase bias
  (Ladd 1994 moving-wall momentum injection);
* an optional **Smagorinsky large-eddy closure** (local eddy viscosity from the
  resolved strain-rate tensor) that keeps the scheme stable when the molecular
  viscosity ``nu = u_in D / Re`` becomes very small (Re >= 1e5);
* user-prescribed **inlet velocity profiles** ``u(y)`` and a zero-gradient
  outlet, so that non-uniform ship-wake inflow can be imposed directly;
* **two independent force estimators**: (i) wall momentum exchange over the
  same BFL links that impose the boundary condition, and (ii) a control-volume
  momentum balance ``F_i = -closed_integral Pi_ij n_j dl`` with the exact LBM
  momentum-flux tensor ``Pi_ij = sum_q f_q c_qi c_qj``.  Agreement between the
  two is a numerical audit of the force extraction, independent of the wall
  model.

Unit conventions: lattice units with ``dx = dt = 1``, reference density 1.
Coefficients are reported as ``C_L = 2 F_y / (rho u_in^2 D)`` and
``C_D = 2 F_x / (rho u_in^2 D)`` (force per unit span), with ``x`` streamwise
and ``y`` transverse.  Positive spin ``omega > 0`` is counter-clockwise and
produces a *positive* (upward) lift for a +x free stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# --------------------------------------------------------------------------- #
# Lattice definition
# --------------------------------------------------------------------------- #
C_LIST: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
)
W_LIST: tuple[float, ...] = (
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
)
OPPOSITE: tuple[int, ...] = (0, 3, 4, 1, 2, 7, 8, 5, 6)
CS2 = 1.0 / 3.0


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class RotorChannelConfig:
    """Geometry / flow parameters of one 2D rotating-cylinder channel.

    The box is ``nx x ny`` cells, the cylinder of diameter ``D = 2 * radius``
    is centred at ``(cx, cy)`` (cell centres on integer coordinates).  ``spin``
    is a tuple of spin ratios ``alpha = omega R / u_in``; all values are
    advanced simultaneously as a batch.

    ``inlet_profile`` is an optional length-``ny`` tuple of ``u(y) / u_in`` used
    to impose a non-uniform (ship-wake) inflow; ``u_in`` remains the reference
    speed used for the force normalisation and for the spin ratio.
    """

    nx: int = 640
    ny: int = 384
    radius: float = 16.0
    u_in: float = 0.08
    re: float = 200.0
    spin: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)
    cx: float | None = None
    cy: float | None = None
    les_cs: float = 0.0
    inlet_profile: tuple[float, ...] | None = None
    n_steps: int = 20000
    warmup: int = 6000
    sample_every: int = 5
    seed: int = 0
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.cx is None:
            self.cx = 8.0 * 2.0 * self.radius
        if self.cy is None:
            self.cy = 0.5 * (self.ny - 1)

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def nu(self) -> float:
        return self.u_in * self.diameter / self.re

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    def validate(self) -> None:
        if self.nx < 32 or self.ny < 32:
            raise ValueError("nx, ny must be >= 32")
        if self.radius < 2.0:
            raise ValueError("radius must be >= 2 lattice cells")
        if self.cx <= 2.0 * self.radius or self.cx >= self.nx - 4.0 * self.radius:
            raise ValueError("cylinder must leave >=2D upstream and >=4D downstream")
        if self.cy <= 1.5 * self.radius or self.cy >= self.ny - 1.5 * self.radius:
            raise ValueError("cylinder must be >=3D away from transverse boundaries")
        if self.tau <= 0.5:
            raise ValueError("tau <= 0.5: viscosity unphysical (reduce u_in*radius or raise Re)")
        if self.les_cs < 0.0:
            raise ValueError("les_cs must be >= 0")
        if self.inlet_profile is not None and len(self.inlet_profile) != self.ny:
            raise ValueError("inlet_profile length must equal ny")
        if self.warmup >= self.n_steps:
            raise ValueError("warmup must be < n_steps")


# --------------------------------------------------------------------------- #
# Geometry (curved-wall link fractions)
# --------------------------------------------------------------------------- #
@dataclass
class RotorGeometry:
    """Pre-computed solid mask and fluid--solid link fractions."""

    solid: torch.Tensor  # (ny, nx) bool, True inside the cylinder
    link: torch.Tensor  # (8, ny, nx) bool, fluid node with wall along q
    delta: torch.Tensor  # (8, ny, nx) float64 wall distance fraction
    nx: int
    ny: int
    cx: float
    cy: float
    radius: float

    @property
    def wall_cells(self) -> int:
        return int(self.link.any(dim=0).sum().item())


def build_geometry(nx: int, ny: int, cx: float, cy: float, radius: float) -> RotorGeometry:
    """Build the solid mask and per-direction BFL wall fractions ``delta``."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    solid = r2 <= radius * radius

    link = torch.zeros((8, ny, nx), dtype=torch.bool)
    delta = torch.full((8, ny, nx), 0.5, dtype=torch.float64)
    for q in range(1, 9):
        cqx, cqy = C_LIST[q]
        neighbour_solid = torch.roll(solid, shifts=(-cqy, -cqx), dims=(0, 1))
        m = (~solid) & neighbour_solid
        link[q - 1] = m
        if not m.any():
            continue
        dx = xx[m] - cx
        dy = yy[m] - cy
        dc = dx * cqx + dy * cqy  # d . c  (negative when pointing at the wall)
        disc = dc * dc - (dx * dx + dy * dy - radius * radius)
        t = -dc - torch.sqrt(disc.clamp_min(0.0))
        delta[q - 1][m] = t.clamp(1e-3, 1.0)
    return RotorGeometry(
        solid=solid, link=link, delta=delta, nx=nx, ny=ny, cx=cx, cy=cy, radius=radius
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 equilibrium for batched fields ``(B, ...)`` -> ``(B, 9, ...)``."""
    shape = [1, 9] + [1] * (rho.ndim - 1)
    w = torch.tensor(W_LIST, dtype=rho.dtype, device=rho.device).view(shape)
    cc = torch.tensor(C_LIST, dtype=rho.dtype, device=rho.device)
    cx = cc[:, 0].view(shape)
    cy = cc[:, 1].view(shape)
    ux_ = ux.unsqueeze(1)
    uy_ = uy.unsqueeze(1)
    cu = cx * ux_ + cy * uy_
    usq = ux_ * ux_ + uy_ * uy_
    return w * rho.unsqueeze(1) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def macroscopic(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Density and velocity from populations ``f`` of shape ``(B, 9, ny, nx)``."""
    cc = torch.tensor(C_LIST, dtype=f.dtype, device=f.device).view(1, 9, 2, 1, 1)
    rho = f.sum(dim=1)
    mom = (f.unsqueeze(2) * cc).sum(dim=1)  # (B, 2, ny, nx)
    ux = mom[:, 0] / rho
    uy = mom[:, 1] / rho
    return rho, ux, uy



def _macro_slice(fs: torch.Tensor, cdev: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Macroscopic fields on a slice whose trailing dims may be (ny,) or (nx,)."""
    rho = fs.sum(dim=1)
    shape = [1, 9] + [1] * (fs.ndim - 2)
    cx = cdev[:, 0].view(shape)
    cy = cdev[:, 1].view(shape)
    ux = (fs * cx).sum(dim=1) / rho
    uy = (fs * cy).sum(dim=1) / rho
    return rho, ux, uy


def _stream(f: torch.Tensor) -> torch.Tensor:
    """Pull streaming for D2Q9 with layout ``(B, 9, ny, nx)``."""
    out = torch.empty_like(f)
    for q in range(9):
        cqx, cqy = C_LIST[q]
        out[:, q] = torch.roll(f[:, q], shifts=(cqy, cqx), dims=(1, 2))
    return out


def _smagorinsky_tau(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    nu: float,
    cs: float,
    tau_min: float = 0.503,
    tau_max: float = 2.0,
) -> torch.Tensor:
    """Local relaxation time with Smagorinsky eddy viscosity (batched)."""
    dux_dx = 0.5 * (torch.roll(ux, -1, dims=2) - torch.roll(ux, 1, dims=2))
    dux_dy = 0.5 * (torch.roll(ux, -1, dims=1) - torch.roll(ux, 1, dims=1))
    duy_dx = 0.5 * (torch.roll(uy, -1, dims=2) - torch.roll(uy, 1, dims=2))
    duy_dy = 0.5 * (torch.roll(uy, -1, dims=1) - torch.roll(uy, 1, dims=1))
    sxx, syy = dux_dx, duy_dy
    sxy = 0.5 * (dux_dy + duy_dx)
    smag = torch.sqrt(torch.clamp(2.0 * (sxx * sxx + syy * syy + 2.0 * sxy * sxy), min=0.0))
    nu_t = (cs * cs) * smag  # Delta = 1 in lattice units
    tau = 0.5 + 3.0 * (nu + nu_t)
    return tau.clamp(tau_min, tau_max)


# --------------------------------------------------------------------------- #
# Main simulation driver
# --------------------------------------------------------------------------- #
def run_rotor_channel(cfg: RotorChannelConfig, keep_fields: bool = False) -> dict:
    """Advance the rotating-cylinder channel and return mean force coefficients.

    Returns a dict with ``spin``, ``cl``, ``cd``, ``cq`` (wall momentum
    exchange), ``cl_cv``, ``cd_cv`` (independent control-volume momentum
    balance), their standard deviations over the sampling window, plus the
    sampled time series and (optionally) the final macroscopic fields.
    """
    cfg.validate()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)
    geom = build_geometry(cfg.nx, cfg.ny, cfg.cx, cfg.cy, cfg.radius)

    spin = torch.tensor(cfg.spin, dtype=dtype, device=device)
    batch = spin.numel()
    ny, nx = cfg.ny, cfg.nx
    # positive spin ratio = clockwise rotation in the (x, y) plane, i.e.
    # positive (upward) Magnus lift on a +x free stream
    omega = -spin * (cfg.u_in / cfg.radius)  # (B,)

    if cfg.inlet_profile is None:
        prof = torch.ones(ny, dtype=dtype, device=device)
    else:
        prof = torch.tensor(cfg.inlet_profile, dtype=dtype, device=device)
    u_prof = (prof * cfg.u_in).view(1, ny).expand(batch, ny).contiguous()
    u_prof_col = u_prof.view(batch, ny, 1).expand(batch, ny, nx)
    v_zero = torch.zeros_like(u_prof_col)

    rho0 = torch.ones((batch, ny, nx), dtype=dtype, device=device)
    f = equilibrium(rho0, u_prof_col, v_zero).clone()
    del u_prof_col, v_zero, rho0

    solid_b = geom.solid.to(device)
    cc = torch.tensor(C_LIST, dtype=dtype, device=device)
    w = torch.tensor(W_LIST, dtype=dtype, device=device)
    opp = list(OPPOSITE)

    # per-direction wall data, flattened to the wall-link list (Nm entries)
    link_idx: list[torch.Tensor] = []
    delta_l: list[torch.Tensor] = []
    uwx_l: list[torch.Tensor] = []
    uwy_l: list[torch.Tensor] = []
    armx_l: list[torch.Tensor] = []
    army_l: list[torch.Tensor] = []
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=dtype, device=device),
        torch.arange(nx, dtype=dtype, device=device),
        indexing="ij",
    )
    for q in range(1, 9):
        cqx, cqy = C_LIST[q]
        m = geom.link[q - 1].to(device)
        idx = m.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        link_idx.append(idx)
        d = geom.delta[q - 1].to(device).to(dtype)[m]
        delta_l.append(d)
        wx = xx[m] + d * cqx
        wy = yy[m] + d * cqy
        uwx_l.append(-omega.view(batch, 1) * (wy.view(1, -1) - cfg.cy))
        uwy_l.append(omega.view(batch, 1) * (wx.view(1, -1) - cfg.cx))
        armx_l.append((wx - cfg.cx).view(1, -1))
        army_l.append((wy - cfg.cy).view(1, -1))
    del yy, xx, m, d, wx, wy

    f_rest = equilibrium(
        torch.ones((batch, ny, nx), dtype=dtype, device=device),
        torch.zeros((batch, ny, nx), dtype=dtype, device=device),
        torch.zeros((batch, ny, nx), dtype=dtype, device=device),
    )
    solid_flat = solid_b.reshape(-1)
    f_rest_flat = f_rest.reshape(batch, 9, -1)[:, :, solid_flat]

    # control-volume contour (rectangle ~0.75 D away from the wall)
    pad = int(max(2, round(0.75 * cfg.diameter)))
    cv_x0 = max(1, int(round(cfg.cx)) - int(cfg.radius) - pad)
    cv_x1 = min(nx - 2, int(round(cfg.cx)) + int(cfg.radius) + pad)
    cv_y0 = max(1, int(round(cfg.cy)) - int(cfg.radius) - pad)
    cv_y1 = min(ny - 2, int(round(cfg.cy)) + int(cfg.radius) + pad)

    series: dict[str, list] = {"step": [], "cl": [], "cd": [], "cq": [], "cl_cv": [], "cd_cv": []}
    n_samp = 0
    acc = torch.zeros((6, batch), dtype=torch.float64, device=device)
    fields: dict = {}
    norm = 2.0 / (cfg.u_in * cfg.u_in * cfg.diameter)
    cq_norm = 0.5 * cfg.u_in * cfg.u_in * cfg.diameter * cfg.diameter
    zeros_b2 = torch.zeros((batch, 2), dtype=dtype, device=device)

    for step in range(cfg.n_steps):
        rho, ux, uy = macroscopic(f)
        if cfg.les_cs > 0.0:
            tau = _smagorinsky_tau(f, rho, ux, uy, cfg.nu, cfg.les_cs)
        else:
            tau = torch.full_like(rho, cfg.tau)
        feq = equilibrium(rho, ux, uy)
        f_post = f - (f - feq) / tau.unsqueeze(1)
        del feq
        rho_post = f_post.sum(dim=1).clamp_min(1e-12)

        f_new = _stream(f_post)

        # ---- BFL interpolated moving-wall bounce-back (wall links only) -----
        force = zeros_b2.clone()
        torque = torch.zeros(batch, dtype=dtype, device=device)
        flat_post = f_post.permute(0, 2, 3, 1).reshape(batch, -1, 9)
        for q in range(1, 9):
            idx = link_idx[q - 1]
            if idx.numel() == 0:
                continue
            cqx, cqy = C_LIST[q]
            qb = opp[q]
            d = delta_l[q - 1]
            fq = flat_post[:, idx, q]
            fqb = flat_post[:, idx, qb]
            up = torch.roll(f_post[:, q], shifts=(cqy, cqx), dims=(1, 2)).reshape(batch, -1)[:, idx]
            # BFL far branch needs the *neighbour* node value of the opposite
            # population: f_opp_post(x_b + e_opp) = f_opp_post(x_b - c_q)
            fqb_nb = torch.roll(f_post[:, qb], shifts=(cqy, cqx), dims=(1, 2)).reshape(batch, -1)[:, idx]
            rho_b = rho_post.reshape(batch, -1)[:, idx]
            cu_wall = cqx * uwx_l[q - 1] + cqy * uwy_l[q - 1]
            corr = -6.0 * w[q] * rho_b * cu_wall
            near = 2.0 * d * fq + (1.0 - 2.0 * d) * up + corr
            far = (fq + (2.0 * d - 1.0) * fqb_nb) / (2.0 * d) + corr / (2.0 * d)
            refl = torch.where(d.view(1, -1) <= 0.5, near, far)
            f_new[:, qb].reshape(batch, -1)[:, idx] = refl
            imp = fq + refl
            force[:, 0] += cqx * imp.sum(dim=1)
            force[:, 1] += cqy * imp.sum(dim=1)
            torque += (imp * (armx_l[q - 1] * cqy - army_l[q - 1] * cqx)).sum(dim=1)
        del flat_post

        # ---- boundaries ------------------------------------------------------
        f_new.reshape(batch, 9, -1)[:, :, solid_flat] = f_rest_flat

        # inlet: non-equilibrium extrapolation from column 1
        rho1, u1x, u1y = _macro_slice(f_new[:, :, :, 1], cc)
        feq0 = equilibrium(torch.ones_like(rho1), u_prof, torch.zeros_like(u_prof))
        f_new[:, :, :, 0] = feq0 + (f_new[:, :, :, 1] - equilibrium(rho1, u1x, u1y))

        # outlet: zero gradient
        f_new[:, :, :, -1] = f_new[:, :, :, -2]

        # transverse far field: equilibrium at the free-stream state
        u_far_cache: dict[int, torch.Tensor] = {}
        for j, src in ((0, 1), (ny - 1, ny - 2)):
            rho_s, ux_s, uy_s = _macro_slice(f_new[:, :, src], cc)
            if j not in u_far_cache:
                u_far_cache[j] = u_prof[:, j].view(batch, 1).expand(batch, nx).contiguous()
            u_far = u_far_cache[j]
            f_new[:, :, j] = equilibrium(torch.ones_like(rho_s), u_far, torch.zeros_like(u_far)) + (
                f_new[:, :, src] - equilibrium(rho_s, ux_s, uy_s)
            )

        f = f_new

        # ---- sampling --------------------------------------------------------
        if step >= cfg.warmup and (step - cfg.warmup) % cfg.sample_every == 0:
            cl_i = force[:, 1] * norm
            cd_i = force[:, 0] * norm
            cq_i = torque / cq_norm
            ccv = _control_volume_force(f, cv_x0, cv_x1, cv_y0, cv_y1)
            cv_cl_i = (ccv[:, 1] * norm).to(dtype)
            cv_cd_i = (ccv[:, 0] * norm).to(dtype)
            vals = [cl_i, cd_i, cq_i, cv_cl_i, cv_cd_i]
            for row, v in enumerate(vals):
                acc[row] += v.double()
                acc[5] += (cl_i.double() ** 2) if row == 0 else 0.0
            n_samp += 1
            series["step"].append(step)
            series["cl"].append(cl_i.tolist())
            series["cd"].append(cd_i.tolist())
            series["cq"].append(cq_i.tolist())
            series["cl_cv"].append(cv_cl_i.tolist())
            series["cd_cv"].append(cv_cd_i.tolist())
            if keep_fields:
                rho_s, ux_s, uy_s = macroscopic(f)
                fields = {
                    "u": ux_s.detach().cpu().numpy(),
                    "v": uy_s.detach().cpu().numpy(),
                    "rho": rho_s.detach().cpu().numpy(),
                    "cv_box": (cv_x0, cv_x1, cv_y0, cv_y1),
                    "solid": solid_b.cpu().numpy(),
                    "cx": cfg.cx,
                    "cy": cfg.cy,
                    "radius": cfg.radius,
                }

    if n_samp == 0:
        raise RuntimeError("no samples collected: increase n_steps or reduce warmup")
    n = float(n_samp)
    mean = (acc[:5] / n).cpu()
    cl_m = mean[0]
    cl_sd = ((acc[5] / n) - cl_m.double() ** 2).clamp_min(0.0).sqrt()

    return {
        "spin": list(cfg.spin),
        "cl": mean[0].tolist(),
        "cd": mean[1].tolist(),
        "cq": mean[2].tolist(),
        "cl_cv": mean[3].tolist(),
        "cd_cv": mean[4].tolist(),
        "cl_std": cl_sd.tolist(),
        "steps": cfg.n_steps,
        "warmup": cfg.warmup,
        "samples": n_samp,
        "nu": cfg.nu,
        "tau": cfg.tau,
        "re": cfg.re,
        "u_in": cfg.u_in,
        "radius": cfg.radius,
        "diameter": cfg.diameter,
        "nx": cfg.nx,
        "ny": cfg.ny,
        "les_cs": cfg.les_cs,
        "mach": cfg.u_in * math.sqrt(3.0),
        "series": series,
        "fields": fields if keep_fields else {},
    }


def _control_volume_force(
    f: torch.Tensor, x0: int, x1: int, y0: int, y1: int
) -> torch.Tensor:
    """Momentum-balance force on the body from the LBM momentum-flux tensor.

    For a steady control volume enclosing only the body,
    ``F_i = -oint Pi_ij n_j dl`` with ``Pi_ij = sum_q f_q c_qi c_qj``
    (this includes pressure and viscous stresses exactly on the lattice).
    Returns ``(B, 2)`` forces in lattice units.
    """
    cc = torch.tensor(C_LIST, dtype=f.dtype, device=f.device)
    pi = torch.zeros((f.shape[0], 2, 2, f.shape[2], f.shape[3]), dtype=f.dtype, device=f.device)
    for q in range(9):
        cqx, cqy = C_LIST[q]
        pi[:, 0, 0] += f[:, q] * cqx * cqx
        pi[:, 0, 1] += f[:, q] * cqx * cqy
        pi[:, 1, 0] += f[:, q] * cqy * cqx
        pi[:, 1, 1] += f[:, q] * cqy * cqy
    force = torch.zeros((f.shape[0], 2), dtype=torch.float64, device=f.device)
    # top face (outward normal +y): F_y += -Pi_yy ; F_x += -Pi_xy
    force[:, 0] += -pi[:, 0, 1, y1, x0 : x1 + 1].sum(dim=-1).double()
    force[:, 1] += -pi[:, 1, 1, y1, x0 : x1 + 1].sum(dim=-1).double()
    # bottom face (normal -y): +Pi
    force[:, 0] += pi[:, 0, 1, y0, x0 : x1 + 1].sum(dim=-1).double()
    force[:, 1] += pi[:, 1, 1, y0, x0 : x1 + 1].sum(dim=-1).double()
    # right face (normal +x)
    force[:, 0] += -pi[:, 0, 0, y0 : y1 + 1, x1].sum(dim=-1).double()
    force[:, 1] += -pi[:, 1, 0, y0 : y1 + 1, x1].sum(dim=-1).double()
    # left face (normal -x)
    force[:, 0] += pi[:, 0, 0, y0 : y1 + 1, x0].sum(dim=-1).double()
    force[:, 1] += pi[:, 1, 0, y0 : y1 + 1, x0].sum(dim=-1).double()
    del cc
    return force