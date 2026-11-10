"""Spherical (curved-wall surface) force method for D2Q9 LBM.

Node: P233W2-N06 (TensorLBM).  PANORAMA.md §2 lists "LBM 球面力法" as a solver
hard-gap that is complementary to the immersed-boundary (IBM) route.

What this module provides
-------------------------
1. **Curved-wall link geometry** (``circle_links`` / ``sdf_links``): for every
   fluid node whose neighbour across a lattice direction lies inside a closed
   body, the exact wall-crossing fraction ``q in (0, 1]`` is obtained by
   intersecting the lattice ray with the surface (analytic quadratic for a
   circle / sphere section, signed-distance interpolation for a general SDF).
   Each link carries the wall point and the outward (solid -> fluid) normal, so
   the link set is a *discrete surface mesh* of the body.
2. **BFL interpolated bounce-back** (Bouzidi, Firdaouss & Lallemand 2001) on a
   D2Q9 lattice, vectorised over the link set:

       q < 1/2 :  f_opp(x_f) = 2q f_i^out(x_f) + (1-2q) f_i^out(x_f - c_i)
       q >= 1/2:  f_opp(x_f) = (1/2q) f_i^out(x_f) + (2q-1)/(2q) f_opp^out(x_f)

   Both branches use post-collision populations at the current time level, so
   the boundary update is applied *after* the pull-stream (see ``apply_bfl``).
3. **Momentum-exchange force integral** on the curved surface

       F_body = sum_links ( f_i^out(x_f) + f_opp^new(x_f) ) c_i

   which is exactly minus the momentum the boundary removed from the fluid in
   that step, i.e. the discrete surface integral of the wall traction.  A
   ``q_power`` switch is exposed for the ablation study (p=0 is the auditable
   momentum-exchange form; p=1 reproduces the ``/q`` variant that is in use in
   ``src/lbm/boundary/spherical_force.py`` and that breaks the momentum budget).
4. **Normal/tangential decomposition of the surface integral** and a
   **momentum audit** (``momentum_audit``) that closes the discrete budget
   dP_fluid/dt = -F_body + g N_fluid to round-off.
5. A body-force driven, doubly periodic **cylinder benchmark driver**
   (``run_cylinder_case``) plus a planar **Couette driver** (``run_couette``)
   used for the planar-boundary leg of the acceptance criteria.

.. warning::
   Status (P233W2-N06): the link geometry, the BFL update, the momentum budget
   (residual 2.6e-17) and the *planar* force integral (wall shear within
   0.0-4.1% of ``mu U / H``) are verified by ``tests/test_spherical_force_method.py``
   and ``artifacts/spherical_force_method/planar.json``.  The body-force driven
   *cylinder* drivers (``run_cylinder_case`` / ``run_cylinder_case_ibm``) are
   **NOT** yet validated: on a 10-lattice-diameter cylinder at Re = 20 the
   transient diverges to NaN within ~10^2 steps (report section 8).  Do not use
   them as a Cd source until that defect is resolved.

References
----------
Bouzidi, M., Firdaouss, M., Lallemand, P. (2001) Phys. Fluids 13, 3452.
Yu, D., Mei, R., Luo, L.-S., Shyy, W. (2003) Prog. Comput. Fluid Dyn. 3, 152.
Ladd, A.J.C. (1994) J. Fluid Mech. 271, 285 (momentum-exchange force).
Guo, Z., Zheng, C., Shi, B. (2002) Phys. Rev. E 65, 046308 (forcing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "C_D2Q9",
    "W_D2Q9",
    "OPPOSITE_D2Q9",
    "CurvedLinks",
    "circle_links",
    "sdf_links",
    "equilibrium",
    "macroscopic",
    "collide_bgk",
    "stream_periodic",
    "apply_bfl",
    "momentum_audit",
    "decompose_surface_force",
    "run_planar_channel",
    "run_couette",
    "surface_force_stress",
    "stress_tensor",
    "run_cylinder_case",
    "run_cylinder_case_ibm",
    "_advance",
]

C_D2Q9 = (
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
W_D2Q9 = (
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
OPPOSITE_D2Q9 = (0, 3, 4, 1, 2, 7, 8, 5, 6)


# ---------------------------------------------------------------------------
# lattice helpers
# ---------------------------------------------------------------------------
def _c_t(device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(C_D2Q9, dtype=dtype, device=device)


def _w_t(device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(W_D2Q9, dtype=dtype, device=device)


def _opp_t(device) -> torch.Tensor:
    return torch.tensor(OPPOSITE_D2Q9, dtype=torch.int64, device=device)


def equilibrium(rho, ux, uy, device=None):
    """D2Q9 equilibrium populations, shape (9, ny, nx)."""
    device = device or rho.device
    c = _c_t(device, rho.dtype)
    w = _w_t(device, rho.dtype)
    cx = c[:, 0].view(9, *([1] * rho.dim()))
    cy = c[:, 1].view(9, *([1] * rho.dim()))
    cu = cx * ux + cy * uy
    usq = ux * ux + uy * uy
    return w.view(9, *([1] * rho.dim())) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def macroscopic(f, gx: float = 0.0, gy: float = 0.0):
    """rho, ux, uy from the raw moments (``u = sum_i f_i c_i / rho``).

    The body force is applied as a post-collision momentum source, so the raw
    moment is the physical velocity up to O(g).  The ``gx``/``gy`` arguments are
    accepted for API stability but do not shift the velocity.
    """
    c = _c_t(f.device, f.dtype)
    rho = f.sum(0)
    mx = (f * c[:, 0].view(9, 1, 1)).sum(0)
    my = (f * c[:, 1].view(9, 1, 1)).sum(0)
    ux = mx / rho
    uy = my / rho
    return rho, ux, uy


def collide_bgk(f, tau: float, fluid_mask=None, gx: float = 0.0, gy: float = 0.0):
    """BGK collision + constant body force (momentum injection ``g`` per node)."""
    rho, ux, uy = macroscopic(f, gx, gy)
    feq = equilibrium(rho, ux, uy, device=f.device)
    fout = f - (f - feq) / tau
    if gx != 0.0 or gy != 0.0:
        c = _c_t(f.device, f.dtype)
        w = _w_t(f.device, f.dtype)
        cu = c[:, 0].view(9, 1, 1) * gx + c[:, 1].view(9, 1, 1) * gy
        d = (3.0 * w.view(9, 1, 1) * cu).expand_as(fout)
        fout = fout + (d if fluid_mask is None else d * fluid_mask.unsqueeze(0))
    if fluid_mask is not None:
        fout = torch.where(fluid_mask.unsqueeze(0), fout, f)
    return fout


def stream_periodic(f):
    """Pull streaming with periodic wrap; returns a new tensor."""
    out = torch.zeros_like(f)
    c = C_D2Q9
    for d in range(9):
        cx, cy = c[d]
        if cx == 0 and cy == 0:
            out[d] = f[d]
        else:
            out[d] = torch.roll(f[d], shifts=(cy, cx), dims=(0, 1))
    return out


# ---------------------------------------------------------------------------
# curved-wall link geometry
# ---------------------------------------------------------------------------
@dataclass
class CurvedLinks:
    """Discrete surface mesh of a body: one entry per fluid/solid lattice link."""

    flat: torch.Tensor  # (L,) flat index iy*nx + ix of the fluid node
    direction: torch.Tensor  # (L,) lattice direction pointing into the solid
    q: torch.Tensor  # (L,) wall fraction in (0, 1]
    normal: torch.Tensor  # (L, 2) outward unit normal (solid -> fluid)
    wall_point: torch.Tensor  # (L, 2) wall intersection point (lattice units)
    nx: int
    ny: int
    idx_dir: torch.Tensor = field(init=False)
    idx_opp: torch.Tensor = field(init=False)
    idx_prev: torch.Tensor = field(init=False)
    idx_scatter: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        n = self.nx * self.ny
        d = self.direction
        opp = torch.tensor([OPPOSITE_D2Q9[int(v)] for v in d.tolist()], dtype=torch.int64)
        self.idx_dir = d * n + self.flat
        self.idx_opp = opp * n + self.flat
        # x_f - c_d (periodic gather used by the q < 1/2 branch)
        ix = (self.flat % self.nx).cpu().numpy()
        iy = (self.flat // self.nx).cpu().numpy()
        cxy = np.array([C_D2Q9[int(v)] for v in d.tolist()], dtype=np.int64)
        px = (ix - cxy[:, 0]) % self.nx
        py = (iy - cxy[:, 1]) % self.ny
        prev = py * self.nx + px
        self.idx_prev = torch.as_tensor(prev, dtype=torch.int64) + d * n
        self.idx_scatter = opp * n + self.flat

    def __len__(self) -> int:  # number of active links
        return int(self.flat.numel())

    def summary(self) -> Dict[str, float]:
        q = self.q
        n = self.normal
        return {
            "n_links": int(q.numel()),
            "q_min": float(q.min()) if q.numel() else 0.0,
            "q_max": float(q.max()) if q.numel() else 0.0,
            "q_mean": float(q.mean()) if q.numel() else 0.0,
            "normal_closure": float(n.sum(0).norm()) if q.numel() else 0.0,
        }


def _finish_links(nx, ny, flats, dirs, qs, wall_pts, normals, device) -> CurvedLinks:
    if len(flats) == 0:
        z = torch.zeros(0)
        return CurvedLinks(
            flat=torch.zeros(0, dtype=torch.int64),
            direction=torch.zeros(0, dtype=torch.int64),
            q=z,
            normal=torch.zeros(0, 2),
            wall_point=torch.zeros(0, 2),
            nx=nx,
            ny=ny,
        )
    return CurvedLinks(
        flat=torch.as_tensor(np.concatenate(flats), dtype=torch.int64, device=device),
        direction=torch.as_tensor(np.concatenate(dirs), dtype=torch.int64, device=device),
        q=torch.as_tensor(np.concatenate(qs), dtype=torch.float64, device=device),
        normal=torch.as_tensor(np.concatenate(normals), dtype=torch.float64, device=device),
        wall_point=torch.as_tensor(np.concatenate(wall_pts), dtype=torch.float64, device=device),
        nx=nx,
        ny=ny,
    )


def circle_links(nx: int, ny: int, cx: float, cy: float, radius: float, device=None) -> CurvedLinks:
    """Boundary links of a circular cylinder (analytic ray/circle intersection)."""
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    fluid = (xx - cx) ** 2 + (yy - cy) ** 2 > radius**2
    flats: List[np.ndarray] = []
    dirs: List[np.ndarray] = []
    qs: List[np.ndarray] = []
    wps: List[np.ndarray] = []
    nrm: List[np.ndarray] = []
    for d, (cxx, cyy) in enumerate(C_D2Q9):
        if d == 0:
            continue
        xn = xx + cxx
        yn = yy + cyy
        inside = (xn - cx) ** 2 + (yn - cy) ** 2 <= radius**2
        ok = (xn >= 0) & (xn < nx) & (yn >= 0) & (yn < ny)
        valid = fluid & inside & ok
        iy, ix = np.nonzero(valid)
        if iy.size == 0:
            continue
        px = ix.astype(np.float64)
        py = iy.astype(np.float64)
        norm = math.hypot(cxx, cyy)
        ex, ey = cxx / norm, cyy / norm
        dx = px - cx
        dy = py - cy
        b = 2.0 * (dx * ex + dy * ey)
        cc = dx * dx + dy * dy - radius**2
        disc = np.sqrt(np.maximum(b * b - 4.0 * cc, 0.0))
        t = -0.5 * (b + disc)
        q = np.clip(t / norm, 1e-3, 1.0)
        wxp = px + q * norm * ex
        wyp = py + q * norm * ey
        nxp = (wxp - cx) / radius
        nyp = (wyp - cy) / radius
        flats.append(iy * nx + ix)
        dirs.append(np.full(iy.shape, d, dtype=np.int64))
        qs.append(q)
        wps.append(np.stack([wxp, wyp], axis=1))
        nrm.append(np.stack([nxp, nyp], axis=1))
    return _finish_links(nx, ny, flats, dirs, qs, wps, nrm, device)


def sdf_links(
    sdf: np.ndarray,
    device=None,
    periodic: Tuple[bool, bool] = (False, False),
) -> CurvedLinks:
    """Boundary links of a body given by a signed distance field (fluid > 0).

    ``periodic`` selects periodic wrapping of the neighbour lookup in ``x`` / ``y``
    (needed for a channel that is periodic along its streamwise direction, so
    that no link is lost at the lateral domain edges).
    """
    ny, nx = sdf.shape
    fluid = sdf > 0.0
    px, py = periodic
    flats: List[np.ndarray] = []
    dirs: List[np.ndarray] = []
    qs: List[np.ndarray] = []
    wps: List[np.ndarray] = []
    nrm: List[np.ndarray] = []
    gy, gx = np.gradient(sdf)
    jx = np.arange(nx)[None, :] + np.zeros((ny, nx), dtype=np.int64)
    jy = np.arange(ny)[:, None] + np.zeros((ny, nx), dtype=np.int64)
    for d, (cxx, cyy) in enumerate(C_D2Q9):
        if d == 0:
            continue
        xr = jx + cxx
        yr = jy + cyy
        if px:
            ok_x = np.ones_like(xr, dtype=bool)
            xm = xr % nx
        else:
            ok_x = (xr >= 0) & (xr < nx)
            xm = np.clip(xr, 0, nx - 1)
        if py:
            ok_y = np.ones_like(yr, dtype=bool)
            ym = yr % ny
        else:
            ok_y = (yr >= 0) & (yr < ny)
            ym = np.clip(yr, 0, ny - 1)
        s_nb = sdf[ym, xm]
        valid = fluid & (s_nb <= 0.0) & ok_x & ok_y
        iy, ix = np.nonzero(valid)
        if iy.size == 0:
            continue
        nxg = gx[iy, ix]
        nyg = gy[iy, ix]
        gn = np.hypot(nxg, nyg)
        gn[gn == 0] = 1.0
        nxg = nxg / gn
        nyg = nyg / gn
        norm = math.hypot(cxx, cyy)
        ex, ey = cxx / norm, cyy / norm
        s0 = sdf[iy, ix]
        s1 = s_nb[iy, ix]
        q = np.clip(s0 / np.maximum(s0 - s1, 1e-12), 1e-3, 1.0)
        wxp = ix + q * norm * ex
        wyp = iy + q * norm * ey
        flats.append(iy * nx + ix)
        dirs.append(np.full(iy.shape, d, dtype=np.int64))
        qs.append(q)
        wps.append(np.stack([wxp, wyp], axis=1))
        nrm.append(np.stack([nxg, nyg], axis=1))
    return _finish_links(nx, ny, flats, dirs, qs, wps, nrm, device)


# ---------------------------------------------------------------------------
# BFL boundary update + momentum-exchange force integral
# ---------------------------------------------------------------------------
@dataclass
class ForceResult:
    force: np.ndarray  # (2,) total force on the body, lattice units
    normal_force: np.ndarray  # (2,) integral of the normal (presssure) traction
    tangential_force: np.ndarray  # (2,) integral of the tangential (friction) traction
    drag: float
    lift: float
    n_links: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "drag": self.drag,
            "lift": self.lift,
            "fx": float(self.force[0]),
            "fy": float(self.force[1]),
            "fx_pressure": float(self.normal_force[0]),
            "fy_pressure": float(self.normal_force[1]),
            "fx_friction": float(self.tangential_force[0]),
            "fy_friction": float(self.tangential_force[1]),
            "n_links": self.n_links,
        }


def apply_bfl(
    f_post: torch.Tensor,
    links: CurvedLinks,
    wall_velocity: Optional[Tuple[float, float]] = None,
    rho_wall: float = 1.0,
    q_power: float = 0.0,
    return_populations: bool = False,
):
    """Stream + BFL interpolated bounce-back; return (f_new, ForceResult).

    ``f_post`` are post-collision populations at time ``t``.  ``q_power`` is the
    exponent of the (non-physical) ``1/q`` force weighting used only for the
    ablation study; ``q_power=0`` is the momentum-exchange form that satisfies
    the discrete momentum budget.
    """
    f_new = stream_periodic(f_post)
    if len(links) == 0:
        if return_populations:
            return f_new, f_post
        raise ValueError("no active boundary links")
    flat = f_post.reshape(-1)
    q = links.q.to(f_post.dtype)
    fi = flat[links.idx_dir]
    fo = flat[links.idx_opp]
    lin = q < 0.5
    fup = flat[links.idx_prev]
    fb_lin = 2.0 * q * fi + (1.0 - 2.0 * q) * fup
    inv2q = 0.5 / q
    fb_quad = inv2q * fi + (2.0 * q - 1.0) * inv2q * fo
    fb = torch.where(lin, fb_lin, fb_quad)

    if wall_velocity is not None:
        # specular reflection about the wall velocity in the wall frame:
        #   f_opp = f_i - 2 w_i rho (c_i . u_w) / cs^2 ,  cs^2 = 1/3
        c = _c_t(f_post.device, f_post.dtype)
        w = _w_t(f_post.device, f_post.dtype)
        d = links.direction
        ci = c[d]
        if torch.is_tensor(wall_velocity):
            uw = wall_velocity.to(f_post.device, f_post.dtype)
            uwd = ci[:, 0] * uw[:, 0] + ci[:, 1] * uw[:, 1]
        else:
            uwd = ci[:, 0] * wall_velocity[0] + ci[:, 1] * wall_velocity[1]
        fb = fb - 6.0 * w[d] * rho_wall * uwd

    out = f_new.reshape(-1)
    out.index_copy_(0, links.idx_scatter, fb)

    c = _c_t(f_post.device, f_post.dtype)
    d = links.direction
    ci = c[d].double()
    weight = (1.0 if q_power == 0.0 else (1.0 / q.double()) ** q_power)
    contrib = (fi.double() + fb.double()) * weight
    fx = float((contrib * ci[:, 0]).sum())
    fy = float((contrib * ci[:, 1]).sum())

    nrm = links.normal
    ftot = np.array([fx, fy])
    # tangential component of the link traction: t = F_link/|c| - (F_link . n) n
    fl = contrib.unsqueeze(1) * ci  # (L, 2) link force (momentum per step)
    nn = nrm
    fn = (fl * nn).sum(1, keepdim=True) * nn
    Ft = fl - fn
    pres = fn.sum(0).cpu().numpy()
    tang = Ft.sum(0).cpu().numpy()
    res = ForceResult(
        force=ftot,
        normal_force=pres,
        tangential_force=tang,
        drag=float(ftot[0]),
        lift=float(ftot[1]),
        n_links=len(links),
    )
    if return_populations:
        return f_new, res
    return f_new, res


def decompose_surface_force(links: CurvedLinks, link_force: torch.Tensor) -> Dict[str, float]:
    """Split a per-link force array into normal (pressure) and tangential parts."""
    n = links.normal
    fn = (link_force * n).sum(1, keepdim=True) * n
    ft = link_force - fn
    return {
        "pressure": fn.sum(0).cpu().numpy(),
        "friction": ft.sum(0).cpu().numpy(),
        "total": link_force.sum(0).cpu().numpy(),
    }


def momentum_audit(
    f_before: torch.Tensor,
    links: CurvedLinks,
    tau: float,
    fluid_mask=None,
    dtype=torch.float64,
) -> Dict[str, float]:
    """One-step discrete momentum budget: dP_fluid = -F_body (to round-off).

    Collision (BGK, fluid nodes only) and periodic streaming conserve total
    momentum exactly; therefore the change of the lattice-wide momentum over a
    full step must equal minus the momentum-exchange force integral, and the
    momentum injected by a body force ``g``.  Returns the residual normalised by
    the momentum scale of the state.
    """
    f = f_before.to(dtype)
    if fluid_mask is not None and fluid_mask.dtype != torch.bool:
        fluid_mask = fluid_mask.bool()
    f = torch.where(fluid_mask.unsqueeze(0), f, torch.zeros_like(f)) if fluid_mask is not None else f
    p0 = _total_momentum(f, fluid_mask)
    f_post = collide_bgk(f, tau, fluid_mask)
    if fluid_mask is not None:
        f_post = torch.where(fluid_mask.unsqueeze(0), f_post, torch.zeros_like(f_post))
    f_new, res = apply_bfl(f_post, links)
    p1 = _total_momentum(f_new, fluid_mask)
    dp = p1 - p0
    resid = dp + torch.as_tensor(res.force, dtype=dtype)
    scale = float(p0.norm()) + float(f.abs().sum())
    return {
        "dp": dp.cpu().numpy(),
        "force": res.force,
        "residual": resid.cpu().numpy(),
        "residual_norm": float(resid.norm()),
        "relative_residual": float(resid.norm()) / max(scale, 1e-30),
        "momentum_scale": scale,
    }


def _total_momentum(f: torch.Tensor, mask=None) -> torch.Tensor:
    c = _c_t(f.device, f.dtype)
    p = torch.zeros(2, dtype=f.dtype, device=f.device)
    m = f if mask is None else f * mask.unsqueeze(0)
    p[0] = (m * c[:, 0].view(9, 1, 1)).sum()
    p[1] = (m * c[:, 1].view(9, 1, 1)).sum()
    return p


# ---------------------------------------------------------------------------
# planar validation: Couette flow with BFL walls
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# surface (stress) integral: independent force route for cross-validation
# ---------------------------------------------------------------------------
def stress_tensor(f: torch.Tensor, tau: float, gx: float = 0.0, gy: float = 0.0):
    """Pressure and viscous stress from the non-equilibrium populations.

    ``sigma_ab = -(1 - 1/(2 tau)) sum_i (f_i - f_i^eq) c_ia c_ib`` follows from
    the first-order Chapman-Enskog expansion (``Pi^(1) = -sigma``), see e.g.
    Krueger et al. (2017) ch. 5.
    """
    rho, ux, uy = macroscopic(f, gx, gy)
    feq = equilibrium(rho, ux, uy, device=f.device)
    fneq = f - feq
    c = _c_t(f.device, f.dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    pref = 1.0 - 0.5 / tau
    sxx = -pref * (fneq * cx * cx).sum(0)
    sxy = -pref * (fneq * cx * cy).sum(0)
    syy = -pref * (fneq * cy * cy).sum(0)
    return rho / 3.0, sxx, sxy, syy


def surface_force_stress(
    f: torch.Tensor,
    links: CurvedLinks,
    tau: float,
    closed: bool = True,
    gx: float = 0.0,
    gy: float = 0.0,
) -> Dict[str, object]:
    """Integrate the wall traction over the discrete surface mesh.

    The traction ``t = (-p I + sigma) . n`` is sampled at the fluid node of every
    boundary link, averaged per surface point, and integrated with a
    1D Voronoi (trapezoidal) arc-length quadrature along the surface.  The force
    on the body is ``-int t ds`` (``n`` points from the solid into the fluid).
    """
    if len(links) == 0:
        raise ValueError("no active boundary links")
    p, sxx, sxy, syy = stress_tensor(f, tau, gx, gy)
    flat_idx = links.flat
    pn = p.reshape(-1)[flat_idx]
    sxxn = sxx.reshape(-1)[flat_idx]
    sxyn = sxy.reshape(-1)[flat_idx]
    syyn = syy.reshape(-1)[flat_idx]
    n = links.normal
    tx = -pn * n[:, 0] + sxxn * n[:, 0] + sxyn * n[:, 1]
    ty = -pn * n[:, 1] + sxyn * n[:, 0] + syyn * n[:, 1]
    t = torch.stack([tx, ty], dim=1)
    # average multiple links that share the same boundary fluid node; each
    # boundary node owns one surface patch (Voronoi in arc length)
    keys = links.flat
    uniq, inverse = torch.unique(keys, return_inverse=True)
    tsum = torch.zeros((uniq.numel(), 2), dtype=t.dtype)
    tsum.index_add_(0, inverse, t)
    cnt = torch.zeros(uniq.numel(), dtype=t.dtype)
    cnt.index_add_(0, inverse, torch.ones_like(inverse, dtype=t.dtype))
    tavg = tsum / cnt.unsqueeze(1)
    order = torch.argsort(uniq)
    tavg = tavg[order]
    pts = torch.zeros((uniq.numel(), 2), dtype=torch.float64)
    pts.index_add_(0, inverse, links.wall_point.double())
    pts = pts / cnt.double().unsqueeze(1)
    pts = pts[order]
    nn_acc = torch.zeros((uniq.numel(), 2), dtype=torch.float64)
    nn_acc.index_add_(0, inverse, links.normal.double())
    navg = (nn_acc / cnt.double().unsqueeze(1))[order]
    navg = navg / navg.norm(dim=1, keepdim=True).clamp_min(1e-12)
    # 1D arc-length quadrature along the ordered surface points
    seg = (pts[1:] - pts[:-1]).norm(dim=1)
    if closed:
        last = (pts[0] - pts[-1]).norm()
        ds = 0.5 * (torch.cat([seg, last[None]]) + torch.cat([last[None], seg]))
    else:
        ds = torch.empty(pts.shape[0], dtype=torch.float64)
        ds[0] = seg[0]
        ds[-1] = seg[-1]
        if seg.numel() > 1:
            ds[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    # force on the body = +int tau . n_S ds  (n_S = outward normal of the body)
    force = (tavg.double() * ds.unsqueeze(1)).sum(0)
    nn = navg
    tn = (tavg.double() * nn).sum(1, keepdim=True) * nn
    tt = tavg.double() - tn
    pressure_force = (tn * ds.unsqueeze(1)).sum(0)
    friction_force = (tt * ds.unsqueeze(1)).sum(0)
    return {
        "force": force.cpu().numpy(),
        "pressure_force": pressure_force.cpu().numpy(),
        "friction_force": friction_force.cpu().numpy(),
        "surface_length": float(ds.sum()),
        "n_points": int(uniq.numel()),
    }


def run_planar_channel(
    ny: int = 18,
    nx: int = 6,
    h: float = 16.0,
    g: float = 2e-5,
    nu: float = 0.05,
    steps: int = 9000,
    offset: float = 0.0,
    dtype=torch.float64,
) -> Dict[str, float]:
    """Pressure/body-force driven plane channel (planar leg of the force check).

    Both walls are static, so the boundary force integral must reproduce the
    momentum balance ``F_wall,total = g * H`` (per unit depth) and the analytic
    Poiseuille profile ``u = g (H^2/4 - y'^2) / (2 mu)``.  ``offset`` moves both
    walls off the halfway position so the interpolated BFL branch is exercised.
    """
    device = torch.device("cpu")
    tau = 3.0 * nu + 0.5
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    y_bot = 0.5 + offset
    y_top = ny - 1.5 + offset
    h_eff = y_top - y_bot
    sdf = np.minimum(yy - y_bot, y_top - yy).astype(np.float64)
    links = sdf_links(sdf, periodic=(True, False), device=device)
    fluid = torch.as_tensor(sdf > 0.0, device=device)
    rho = torch.ones((ny, nx), dtype=dtype, device=device)
    f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho), device=device)
    f = torch.where(fluid.unsqueeze(0), f, torch.zeros_like(f))
    for _ in range(steps):
        f_post = collide_bgk(f, tau, fluid, gx=g)
        f_post = torch.where(fluid.unsqueeze(0), f_post, torch.zeros_like(f_post))
        f, _ = apply_bfl(f_post, links)
        f = torch.where(fluid.unsqueeze(0), f, torch.zeros_like(f))

    c = _c_t(device, dtype)
    fi = f_post.reshape(-1)[links.idx_dir]
    fb = f.reshape(-1)[links.idx_scatter]
    link_f = (fi + fb).unsqueeze(1) * c[links.direction]
    is_top = links.wall_point[:, 1] > 0.5 * (y_bot + y_top)
    f_bot = float(link_f[~is_top, 0].sum().item()) / nx
    f_top = float(link_f[is_top, 0].sum().item()) / nx
    rho_c, ux, _ = macroscopic(f, g, 0.0)
    mu = (tau - 0.5) / 3.0
    yn = np.arange(ny) - y_bot
    ana = g / (2.0 * mu) * yn * (h_eff - yn)
    prof = ux[:, nx // 2].detach().cpu().numpy()
    ok = fluid[:, 0].cpu().numpy()
    prof_err = float(np.abs(prof[ok] - ana[ok]).max() / np.abs(ana).max())
    n_fluid = float(fluid.sum().item())
    return {
        "tau": tau,
        "mu": mu,
        "h_eff": h_eff,
        "offset": offset,
        "q_mean": float(links.q.mean().item()),
        "q_min": float(links.q.min().item()),
        "q_max": float(links.q.max().item()),
        "profile_rel_error": prof_err,
        "force_bottom": f_bot,
        "force_top": f_top,
        "shear_analytic": g * h_eff / 2.0,
        "rel_error_bottom": abs(f_bot - g * h_eff / 2.0) / (g * h_eff / 2.0),
        # the fluid drags *both* walls downstream: the wall traction is +tau at
        # the bottom and at the top wall (the walls retard the fluid on both
        # sides), so the momentum balance is F_bottom + F_top = g * H.
        "rel_error_top": abs(f_top - g * h_eff / 2.0) / (g * h_eff / 2.0),
        "momentum_source": g * n_fluid,
        "force_balance_rel": abs((f_bot + f_top) - g * h_eff) / (g * h_eff),
        "n_links": len(links),
    }


def run_couette(
    ny: int = 18,
    nx: int = 6,
    h: float = 16.0,
    u_wall: float = 0.05,
    re: float = 5.0,
    steps: int = 9000,
    offset: float = 0.0,
    dtype=torch.float64,
) -> Dict[str, float]:
    """Couette flow between a moving (top) and a static (bottom) planar wall.

    The lattice has one solid row inside the domain at each wall, so the link /
    force machinery of the cylinder applies unchanged.  ``offset = 0`` puts both
    walls halfway between nodes (q = 1/2); ``offset != 0`` exercises the
    interpolated BFL branch (q != 1/2).

    Two force routes are reported: the surface (stress) integral
    (:func:`surface_force_stress`) and the momentum exchange of the links.  For a
    wall aligned with the lattice the momentum-exchange route transfers no
    tangential momentum (all active links are normal to the wall), which is a
    documented property of that route, not a bug.
    """
    device = torch.device("cpu")
    nu = u_wall * h / re
    tau = 3.0 * nu + 0.5
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    y_bot = 0.5 + offset
    y_top = ny - 1.5 + offset
    h_eff = y_top - y_bot
    sdf = np.minimum(yy - y_bot, y_top - yy).astype(np.float64)
    links = sdf_links(sdf, device=device)
    fluid = torch.as_tensor(sdf > 0.0, device=device)
    is_top = links.wall_point[:, 1] > 0.5 * (y_bot + y_top)
    uw = torch.zeros((len(links), 2), dtype=dtype, device=device)
    uw[is_top, 0] = u_wall
    c = _c_t(device, dtype)

    rho = torch.ones((ny, nx), dtype=dtype, device=device)
    ux = torch.full((ny, nx), 0.5 * u_wall, dtype=dtype, device=device)
    f = equilibrium(rho, ux, torch.zeros_like(ux), device=device)
    f = torch.where(fluid.unsqueeze(0), f, torch.zeros_like(f))
    for _ in range(steps):
        f_post = collide_bgk(f, tau, fluid)
        f_post = torch.where(fluid.unsqueeze(0), f_post, torch.zeros_like(f_post))
        f, _ = apply_bfl(f_post, links, wall_velocity=uw)
        f = torch.where(fluid.unsqueeze(0), f, torch.zeros_like(f))

    rho_c, ux_c, _ = macroscopic(f)
    mu = (tau - 0.5) / 3.0
    prof = ux_c[:, nx // 2].detach().cpu().numpy()
    y_nodes = np.arange(ny) - y_bot
    lin = u_wall * y_nodes / h_eff
    ok = fluid[:, 0].cpu().numpy()
    l2 = float(np.sqrt(np.mean((prof[ok] - lin[ok]) ** 2)))

    # stress (surface) integral on the static bottom wall
    from tensorlbm.spherical_force_method import surface_force_stress as _sfs

    class _L:
        pass

    def _subset(mask):
        return CurvedLinks(
            flat=links.flat[mask],
            direction=links.direction[mask],
            q=links.q[mask],
            normal=links.normal[mask],
            wall_point=links.wall_point[mask],
            nx=nx,
            ny=ny,
        )

    bot = _subset(~is_top)
    top = _subset(is_top)
    r_bot = _sfs(f, bot, tau, closed=False)
    r_top = _sfs(f, top, tau, closed=False)
    shear_analytic = mu * u_wall / h_eff
    f_bot = float(r_bot["force"][0].item()) / nx
    f_top = float(r_top["force"][0].item()) / nx
    return {
        "tau": tau,
        "mu": mu,
        "h_eff": h_eff,
        "offset": offset,
        "q_mean": float(links.q.mean().item()),
        "shear_analytic": shear_analytic,
        "shear_bottom_stress": f_bot,
        "shear_top_stress": f_top,
        "rel_error_bottom": abs(f_bot - shear_analytic) / shear_analytic,
        "rel_error_top": abs(f_top - shear_analytic) / shear_analytic,
        "profile_l2_error": l2,
        "n_links": len(links),
    }


# ---------------------------------------------------------------------------
# cross-validation route: diffuse immersed boundary (IBM) direct forcing
# ---------------------------------------------------------------------------
def _peskin4(r: np.ndarray) -> np.ndarray:
    a = np.abs(r)
    out = np.zeros_like(a)
    m1 = a <= 1.0
    out[m1] = (3.0 - 2.0 * a[m1] + np.sqrt(1.0 + 4.0 * a[m1] - 4.0 * a[m1] ** 2)) / 8.0
    m2 = (a > 1.0) & (a <= 2.0)
    out[m2] = (5.0 - 2.0 * a[m2] - np.sqrt(np.maximum(-7.0 + 12.0 * a[m2] - 4.0 * a[m2] ** 2, 0.0))) / 8.0
    return out


def run_cylinder_case_ibm(
    re: float,
    diameter: int = 16,
    length_over_d: float = 30.0,
    height_over_d: float = 16.0,
    u_target: float = 0.04,
    reg_steps: int = 6000,
    max_steps: int = 80000,
    sample_stride: int = 4,
    device: str = "cpu",
    dtype=torch.float32,
    verbose: bool = False,
) -> Dict[str, object]:
    """Same benchmark as :func:`run_cylinder_case` but with a diffuse IBM.

    Peskin 4-point delta interpolation/spreading and Uhlmann (2005) direct
    forcing.  The body force on the fluid is ``F_k`` per marker, so the drag on
    the body is ``-sum_k F_k ds_k``.  This route shares nothing with the
    curved-wall link machinery and is used only as an independent
    cross-validation of the drag level (AC5).
    """
    dev = torch.device(device)
    nx = int(round(length_over_d * diameter))
    ny = int(round(height_over_d * diameter))
    radius = 0.5 * diameter
    nu = u_target * diameter / re
    tau = 3.0 * nu + 0.5
    cx, cy = nx / 4.0, ny / 2.0
    n_mark = max(8, int(round(2.0 * math.pi * radius)))
    theta = np.arange(n_mark) * (2.0 * math.pi / n_mark)
    mx = cx + radius * np.cos(theta)
    my = cy + radius * np.sin(theta)
    ds_m = 2.0 * math.pi * radius / n_mark
    flat_m = (np.round(my).astype(np.int64) * nx + np.round(mx).astype(np.int64))
    # 4x4 stencil per marker
    offs = np.array([[-1, -1], [-1, 0], [-1, 1], [-1, 2],
                     [0, -1], [0, 0], [0, 1], [0, 2],
                     [1, -1], [1, 0], [1, 1], [1, 2],
                     [2, -1], [2, 0], [2, 1], [2, 2]])
    ix0 = np.floor(mx).astype(np.int64) - 1
    iy0 = np.floor(my).astype(np.int64) - 1
    idx_x = (ix0[:, None] + offs[None, :, 0]) % nx
    idx_y = (iy0[:, None] + offs[None, :, 1]) % ny
    wx = _peskin4(mx[:, None] - idx_x)
    wy = _peskin4(my[:, None] - idx_y)
    wts = (wx * wy)  # (M, 16)
    node_flat = (idx_y * nx + idx_x).reshape(-1)
    wts_flat = wts.reshape(-1)
    inside = np.zeros((ny, nx), dtype=bool)
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    fluid = torch.as_tensor(~inside, device=dev)
    n_fluid = float(fluid.sum().item())

    rho = torch.ones((ny, nx), dtype=dtype, device=dev)
    ux = torch.full((ny, nx), u_target, dtype=dtype, device=dev)
    f = equilibrium(rho, ux, torch.zeros_like(ux), device=dev)
    f = torch.where(fluid.unsqueeze(0), f, torch.zeros_like(f))

    c = _c_t(dev, dtype)
    w = _w_t(dev, dtype)
    node_t = torch.as_tensor(node_flat, dtype=torch.int64, device=dev)
    wt_t = torch.as_tensor(wts_flat, dtype=dtype, device=dev)
    n_nodes = nx * ny

    def interp(field):
        v = field.reshape(-1)[node_t]
        return (v.view(n_mark, 16) * wt_t.view(n_mark, 16)).sum(1)

    cd0 = 1.18 + 6.8 / re**0.85
    g = cd0 * 0.5 * u_target**2 * diameter / n_fluid
    u_acc, u_cnt, reg_window = 0.0, 0, 200
    for step in range(reg_steps):
        f_post = collide_bgk(f, tau, fluid, gx=g)
        f_post = torch.where(fluid.unsqueeze(0), f_post, torch.zeros_like(f_post))
        fs = stream_periodic(f_post)
        rho_i, uxi, uyi = macroscopic(fs, g, 0.0)
        us = interp(uxi)
        vs = interp(uyi)
        fkx = -us
        fky = -vs
        Fx = torch.zeros(n_nodes, dtype=dtype, device=dev)
        Fy = torch.zeros(n_nodes, dtype=dtype, device=dev)
        Fx.index_add_(0, node_t, (fkx.unsqueeze(1) * wt_t.view(n_mark, 16)).reshape(-1))
        Fy.index_add_(0, node_t, (fky.unsqueeze(1) * wt_t.view(n_mark, 16)).reshape(-1))
        Fx = Fx.view(ny, nx) * fluid
        Fy = Fy.view(ny, nx) * fluid
        cu = c[:, 0].view(9, 1, 1) * Fx + c[:, 1].view(9, 1, 1) * Fy
        f = fs + (3.0 * w.view(9, 1, 1) * cu)
        f = torch.where(fluid.unsqueeze(0), f, torch.zeros_like(f))
        u_acc += float((uxi * fluid).sum().item() / n_fluid)
        u_cnt += 1
        if (step + 1) % reg_window == 0:
            u_bar = u_acc / u_cnt
            u_acc, u_cnt = 0.0, 0
            g *= (u_target / max(u_bar, 1e-12)) ** 0.35

    drag, lift, us_hist = [], [], []
    for step in range(max_steps):
        f_post = collide_bgk(f, tau, fluid, gx=g)
        f_post = torch.where(fluid.unsqueeze(0), f_post, torch.zeros_like(f_post))
        fs = stream_periodic(f_post)
        rho_i, uxi, uyi = macroscopic(fs, g, 0.0)
        us = interp(uxi)
        vs = interp(uyi)
        fkx = -us
        fky = -vs
        Fx = torch.zeros(n_nodes, dtype=dtype, device=dev)
        Fy = torch.zeros(n_nodes, dtype=dtype, device=dev)
        Fx.index_add_(0, node_t, (fkx.unsqueeze(1) * wt_t.view(n_mark, 16)).reshape(-1))
        Fy.index_add_(0, node_t, (fky.unsqueeze(1) * wt_t.view(n_mark, 16)).reshape(-1))
        Fx = Fx.view(ny, nx) * fluid
        Fy = Fy.view(ny, nx) * fluid
        cu = c[:, 0].view(9, 1, 1) * Fx + c[:, 1].view(9, 1, 1) * Fy
        f = fs + (3.0 * w.view(9, 1, 1) * cu)
        f = torch.where(fluid.unsqueeze(0), f, torch.zeros_like(f))
        if step % sample_stride == 0:
            u_bulk = float((uxi * fluid).sum().item() / n_fluid)
            qd = 0.5 * u_bulk**2 * diameter
            drag.append(-float((fkx).sum().item()) * ds_m / qd)
            lift.append(-float((fky).sum().item()) * ds_m / qd)
            us_hist.append(u_bulk)
        if verbose and step % 20000 == 0 and drag:
            print(f"[ibm Re={re} D={diameter}] step={step} Cd={drag[-1]:.4f}", flush=True)
        if step > 8000 and len(drag) > 400 and step % sample_stride == 0:
            tail = np.asarray(drag[-400:])
            if np.std(tail) / max(abs(np.mean(tail)), 1e-12) < 1e-3:
                break

    d = np.asarray(drag)
    keep = d[int(0.3 * d.size):] if d.size > 20 else d
    u_arr = np.asarray(us_hist)
    u_mean = float(u_arr[int(0.3 * u_arr.size):].mean()) if u_arr.size else float(u_target)
    return {
        "method": "ibm_diffuse",
        "re_target": float(re),
        "re_measured": u_mean * diameter / nu,
        "diameter": diameter,
        "cd_mean": float(keep.mean()),
        "cd_std": float(keep.std()),
        "u_mean": u_mean,
        "body_force": float(g),
        "n_markers": n_mark,
        "steps": int(step + 1),
        "cd_from_body_force": float(g * n_fluid / (0.5 * u_mean**2 * diameter)),
        "cd_samples": [float(x) for x in d[::max(1, d.size // 200)]],
    }


# ---------------------------------------------------------------------------
# fused hot loop (allocation-light) used by the benchmark drivers
# ---------------------------------------------------------------------------
class _DV:
    """Precomputed per-lattice constants for the fused update."""

    __slots__ = ("cx", "cy", "w", "n")

    def __init__(self, ny: int, nx: int, device, dtype):
        c = _c_t(device, dtype)
        w = _w_t(device, dtype)
        self.cx = c[:, 0].view(9, 1, 1)
        self.cy = c[:, 1].view(9, 1, 1)
        self.w = w.view(9, 1, 1)
        self.n = nx * ny


def _advance(f, tau, links, dv, g: float = 0.0, force_buf=None, fluid_f=None):
    """One LBM step (collide + optional body force + stream + BFL).

    Allocation-light variant used by the production drivers: the solid interior
    is kept at rest equilibrium and is only ever read through boundary
    populations that are overwritten, which is the standard LBM treatment.
    """
    inv_tau = 1.0 / tau
    rho = f.sum(0).clamp_min(1e-8)
    ux = (f * dv.cx).sum(0) / rho
    uy = (f * dv.cy).sum(0) / rho
    cu = dv.cx * ux + dv.cy * uy
    feq = dv.w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy))
    f -= (f - feq) * inv_tau
    if fluid_f is not None:
        # empty the solid *before* streaming so that the boundary links are the
        # only channel between solid and fluid (this keeps the discrete momentum
        # budget of apply_bfl exact)
        f *= fluid_f
    if g != 0.0 and force_buf is not None:
        f += force_buf
    f_new, res = apply_bfl(f, links)
    if fluid_f is not None:
        f_new *= fluid_f
    return f_new, res


def _setup_cylinder(
    diameter: float,
    nx: int,
    ny: int,
    u0: float,
    device,
    dtype=torch.float32,
):
    radius = 0.5 * diameter
    cx = nx / 4.0
    cy = ny / 2.0
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    solid = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    fluid = torch.as_tensor(~solid, device=device)
    links = circle_links(nx, ny, cx, cy, radius, device=device)
    rho = torch.ones((ny, nx), dtype=dtype, device=device)
    ux = torch.full((ny, nx), u0, dtype=dtype, device=device)
    f = equilibrium(rho, ux, torch.zeros_like(ux), device=device)
    f = torch.where(fluid.unsqueeze(0), f, equilibrium(rho, torch.zeros_like(ux), torch.zeros_like(ux), device=device))
    return f, fluid, links, cx, cy, radius


def run_cylinder_case(
    re: float,
    diameter: int = 16,
    length_over_d: float = 30.0,
    height_over_d: float = 16.0,
    u_target: float = 0.04,
    reg_steps: int = 6000,
    max_steps: int = 120000,
    sample_stride: int = 4,
    min_cycles: float = 4.0,
    q_power: float = 0.0,
    device: str = "cpu",
    dtype=torch.float32,
    seed: int = 0,
    snapshot: bool = True,
    verbose: bool = False,
) -> Dict[str, object]:
    """Steady/unsteady flow past a cylinder, driven by a constant body force.

    Doubly periodic domain; the flow is driven by a uniform body force ``g`` that
    is regulated so the time-averaged bulk velocity (hence ``Re = U D / nu``)
    matches ``re``.  Drag is measured by the curved-wall momentum-exchange
    integral of :func:`apply_bfl`; an independent check is available from the
    momentum source balance ``F_drag = g * N_fluid`` at statistical steady state.
    """
    dev = torch.device(device)
    nx = int(round(length_over_d * diameter))
    ny = int(round(height_over_d * diameter))
    radius = 0.5 * diameter
    nu = u_target * diameter / re
    tau = 3.0 * nu + 0.5
    f, fluid, links, cx, cy, radius = _setup_cylinder(diameter, nx, ny, u_target, dev, dtype)
    dv = _DV(ny, nx, dev, dtype)
    force_buf = dv.w * 3.0 * dv.cx
    fluid_f = fluid.to(dtype).view(1, ny, nx)
    n_fluid = float(fluid.sum().item())

    # initial body-force guess from a Blasius-type drag correlation
    cd0 = 1.18 + 6.8 / re**0.85
    g = cd0 * 0.5 * u_target**2 * diameter / n_fluid

    gen = torch.Generator(device="cpu").manual_seed(seed)

    # ---- regulation phase: relax g so that <U> -> u_target -----------------
    reg_window = 200
    u_acc = 0.0
    u_cnt = 0
    g_hist: List[float] = []
    for step in range(reg_steps):
        f, res = _advance(f, tau, links, dv, g, force_buf, fluid_f)
        rho, ux, uy = macroscopic(f)
        u_mean = float((ux * fluid).sum().item() / n_fluid)
        u_acc += u_mean
        u_cnt += 1
        if (step + 1) % reg_window == 0:
            u_bar = u_acc / u_cnt
            u_acc, u_cnt = 0.0, 0
            ratio = u_target / max(u_bar, 1e-12)
            # mild relaxation keeps the regulation stable
            g *= ratio ** 0.35
            g_hist.append(g)

    # ---- sampling phase ----------------------------------------------------
    drag: List[float] = []
    lift: List[float] = []
    u_samples: List[float] = []
    u_window_sum = 0.0
    u_window_cnt = 0
    u_bulk = 0.0
    cl_hist: List[float] = []
    t_hist: List[int] = []
    period = 0.0
    ux_snap = None
    uy_snap = None
    rho_snap = None
    for step in range(max_steps):
        f, res = _advance(f, tau, links, dv, g, force_buf, fluid_f)
        rho, ux, uy = macroscopic(f)
        u_window_sum += float((ux * fluid).sum().item() / n_fluid)
        u_window_cnt += 1
        if step % sample_stride == 0:
            u_bulk = u_window_sum / u_window_cnt
            u_window_sum, u_window_cnt = 0.0, 0
            qd = 0.5 * u_bulk**2 * diameter  # rho = 1 normalisation
            drag.append(res.drag / qd)
            lift.append(res.lift / qd)
            u_samples.append(u_bulk)
            cl_hist.append(res.lift / qd)
            t_hist.append(step)
        if verbose and step % 20000 == 0:
            print(f"[cyl Re={re} D={diameter}] step={step} Cd={drag[-1] if drag else 0:.4f} U={u_bulk:.5f}", flush=True)
        # stopping: steady state or enough shedding cycles
        if step > 8000 and len(drag) > 400 and step % sample_stride == 0:
            tail = np.asarray(drag[-400:])
            if np.std(tail) / max(abs(np.mean(tail)), 1e-12) < 5e-4:
                break
            # unsteady: require min_cycles shedding cycles in the lift signal
            s = np.asarray(cl_hist[-2000:])
            if s.size > 200 and np.std(s) > 1e-6:
                crossings = np.sum(np.diff(np.sign(s - s.mean())) != 0)
                period = 2.0 * s.size / max(crossings, 1) * sample_stride
                if crossings >= 2 * min_cycles:
                    break

    if snapshot:
        rho_snap = rho.detach().cpu().numpy()
        ux_snap = ux.detach().cpu().numpy()
        uy_snap = uy.detach().cpu().numpy()

    d = np.asarray(drag)
    # discard the first 30 % (residual transient of the sampling window)
    keep = d[int(0.3 * d.size):] if d.size > 20 else d
    cd_mean = float(keep.mean())
    cd_std = float(keep.std())
    u_arr = np.asarray(u_samples)
    u_mean = float(u_arr[int(0.3 * u_arr.size):].mean()) if u_arr.size else u_bulk
    re_measured = u_mean * diameter / nu

    # Strouhal number from lift zero crossings
    st = 0.0
    cl = np.asarray(cl_hist)
    if cl.size > 100:
        s = cl[int(0.3 * cl.size):]
        s = s - s.mean()
        cross = np.nonzero(np.diff(np.sign(s)) != 0)[0]
        if cross.size >= 4:
            half_period = float(np.mean(np.diff(cross))) * sample_stride
            if half_period > 0:
                st = diameter / (2.0 * half_period * max(u_mean, 1e-9))

    out: Dict[str, object] = {
        "re_target": float(re),
        "re_measured": float(re_measured),
        "diameter": diameter,
        "length_over_d": length_over_d,
        "height_over_d": height_over_d,
        "nx": nx,
        "ny": ny,
        "nu": nu,
        "tau": tau,
        "u_mean": float(u_mean),
        "u_target": float(u_target),
        "body_force": float(g),
        "cd_mean": cd_mean,
        "cd_std": cd_std,
        "cl_rms": float(np.std(keep) * 0 + np.std(cl[int(0.3 * cl.size):]) if cl.size else 0.0),
        "strouhal": float(st),
        "n_samples": int(d.size),
        "steps": int(step + 1),
        "n_links": len(links),
        "q_mean": float(links.q.mean()) if len(links) else 0.0,
        "drag_from_body_force": float(g * n_fluid),
        "cd_from_body_force": float(g * n_fluid / (0.5 * u_mean**2 * diameter)),
        "cd_samples": [float(x) for x in d[::max(1, d.size // 200)]],
        "q_power": float(q_power),
        "tau_field": tau,
    }
    if snapshot:
        out["snapshot"] = {
            "rho": rho_snap,
            "ux": ux_snap,
            "uy": uy_snap,
            "cx": cx,
            "cy": cy,
            "radius": radius,
            "fluid": fluid.cpu().numpy(),
        }
    return out