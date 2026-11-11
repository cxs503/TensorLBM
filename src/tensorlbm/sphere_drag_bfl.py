"""Sphere drag coefficient from a D3Q19 LBM solver with BFL interpolated bounce-back.

This module is the *mechanistic* sphere-drag anchor for TensorLBM
(``GAP-LBM-SPHERE``): it generates a sphere geometry on a uniform
Cartesian lattice, applies the **Bouzidi–Firdaouss–Lallemand (BFL)**
quadratic/linear interpolated bounce-back scheme on the curved surface,
integrates the hydrodynamic force with the **momentum-exchange (Ladd)
formula**, and compares the resulting drag coefficient with the classical
Clift–Grace–Weber (CGW 1978) measurements / correlations.

Why BFL interpolated bounce-back (and not Guo extrapolation)?
-------------------------------------------------------------
* The wall position is resolved continuously through the fractional
  distance ``q`` of each cut link, so the hydrodynamic radius is the
  *prescribed* one (no ``0.5``-voxel bias that staircase/half-way schemes
  suffer from).  The scheme is second order in the interpolation stencil.
* It is strictly local: no field extrapolation across the body, no
  iteration to convergence per step, so it stays stable at the
  low-viscosity (Re ~ 1e3) end of the validation range where Guo
  extrapolation is known to produce pressure noise / mass drift.
* The momentum flux needed for the drag is available analytically from the
  same link list, so no additional surface reconstruction is required.

Numerics
--------
* D3Q19 lattice, two-relaxation-time (TRT) collision with magic parameter
  ``Lambda = 3/16`` (stable at ``tau - 0.5 ~ 1e-3``).
* Body force free: the sphere is at rest in a uniform stream, the drag is
  obtained from the momentum exchange on the cut links.
* Inlet: equilibrium at prescribed velocity; outlet: zero-gradient
  extrapolation; lateral boundaries: free-slip (specular) reflection.

Usage (library)::

    from tensorlbm.sphere_drag_bfl import SphereBFLConfig, run_sphere_case
    res = run_sphere_case(SphereBFLConfig(re=100.0, diameter=16, max_steps=20000))
    print(res["cd"], res["cd_ref"], res["cd_err_pct"])

Usage (CLI, entry point)::

    PYTHONPATH=src python -m tensorlbm.sphere_drag_bfl --re 100 --diameter 16
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "SphereBFLConfig",
    "SphereBFLSolver",
    "run_sphere_case",
    "reference_cd",
    "REFERENCE_CD_CGW",
    "schiller_naumann_cd",
    "DEFAULT_RE_SCAN",
    "DEFAULT_DIAMETERS",
    "main",
]

# --------------------------------------------------------------------------- #
# Reference drag data
# --------------------------------------------------------------------------- #

#: Clift, Grace & Weber (1978) "Bubbles, Drops and Particles", Table 5.2:
#: standard drag coefficient of a smooth sphere (experimental compilation).
REFERENCE_CD_CGW: Dict[float, float] = {
    1.0: 26.5,
    2.0: 18.3,
    5.0: 9.80,
    10.0: 4.11,
    20.0: 2.73,
    50.0: 1.57,
    100.0: 1.09,
    200.0: 0.80,
    500.0: 0.55,
    1000.0: 0.44,
}

#: Reynolds numbers of the anchor scan (Clift–Grace–Weber anchor points).
DEFAULT_RE_SCAN: Tuple[float, ...] = (10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)

#: Lattice diameters used for the grid-independence study.
DEFAULT_DIAMETERS: Tuple[int, ...] = (12, 16, 24)


def schiller_naumann_cd(re: float) -> float:
    """Schiller–Naumann (1933) correlation, ``Re <~ 3e5``."""
    return 24.0 / re * (1.0 + 0.15 * re ** 0.687) + 0.42 / (1.0 + 42500.0 / re ** 1.16)


def reference_cd(re: float) -> float:
    """External anchor drag coefficient of a smooth sphere at ``re``.

    Uses the Clift–Grace–Weber tabulated measurement at the tabulated
    Reynolds numbers and log–log linear interpolation in between; outside
    the table the Schiller–Naumann correlation is used (explicitly flagged
    by :func:`reference_source`).
    """
    keys = sorted(REFERENCE_CD_CGW)
    if re in REFERENCE_CD_CGW:
        return REFERENCE_CD_CGW[re]
    if re < keys[0] or re > keys[-1]:
        return schiller_naumann_cd(re)
    lo = max(k for k in keys if k < re)
    hi = min(k for k in keys if k > re)
    t = (math.log(re) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return math.exp(
        math.log(REFERENCE_CD_CGW[lo])
        + t * (math.log(REFERENCE_CD_CGW[hi]) - math.log(REFERENCE_CD_CGW[lo]))
    )


def reference_source(re: float) -> str:
    """Human-readable provenance of :func:`reference_cd` at ``re``."""
    if re in REFERENCE_CD_CGW:
        return "Clift-Grace-Weber(1978) table"
    if min(REFERENCE_CD_CGW) <= re <= max(REFERENCE_CD_CGW):
        return "Clift-Grace-Weber(1978) log-log interpolation"
    return "Schiller-Naumann(1933) correlation"


# --------------------------------------------------------------------------- #
# D3Q19 lattice
# --------------------------------------------------------------------------- #

_D3Q19_DIRS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
    (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
)
_D3Q19_W: Tuple[float, ...] = (
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
)


def _opposite_table() -> Tuple[int, ...]:
    idx = {d: i for i, d in enumerate(_D3Q19_DIRS)}
    return tuple(idx[(-d[0], -d[1], -d[2])] for d in _D3Q19_DIRS)


_OPP: Tuple[int, ...] = _opposite_table()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class SphereBFLConfig:
    """Configuration of a single sphere-drag LBM case (lattice units)."""

    re: float = 100.0
    diameter: int = 16                 # lattice diameters D
    #: free-stream lattice velocity (Mach number ~ 0.05 with c_s^2 = 1/3)
    u_lb: float = 0.05
    #: extents of the box in sphere diameters
    upstream: float = 2.5              # inlet -> sphere centre, in diameters
    downstream: float = 7.0            # sphere centre -> outlet, in diameters
    lateral: float = 3.0               # half-width of the box, in diameters
    max_steps: int = 40000
    #: convergence window / tolerance of the drag plateau
    window: int = 500
    tol: float = 0.01
    sample_interval: int = 10
    #: sponge (damping) layer thickness in front of the outlet, in diameters
    sponge: float = 1.5
    #: maximum relaxation rate of the sponge layer
    sponge_strength: float = 0.2
    #: save the last macroscopic snapshot to ``output_dir/<tag>_fields.pt``
    save_fields: bool = False
    output_dir: str = "artifacts/sphere_drag_bfl"
    device: str = "cpu"
    threads: int = 8
    dtype: str = "float32"

    @property
    def tag(self) -> str:
        return f"re{self.re:g}_D{self.diameter}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # -- derived quantities -------------------------------------------------
    @property
    def nu_lb(self) -> float:
        return self.u_lb * self.diameter / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu_lb + 0.5

    @property
    def mach(self) -> float:
        return self.u_lb * math.sqrt(3.0)

    def grid(self) -> Tuple[int, int, int]:
        """Lattice dimensions (nx, ny, nz) of the flow box."""
        d = float(self.diameter)
        nx = int(round((self.upstream + 1.0 + self.downstream) * d))
        n = int(round(2.0 * self.lateral * d))
        return nx, n, n

    def center(self) -> Tuple[float, float, float]:
        """Sphere centre in lattice coordinates (half-integer transverse)."""
        d = float(self.diameter)
        nx, ny, nz = self.grid()
        return (self.upstream * d + 0.5 * d, 0.5 * (ny - 1), 0.5 * (nz - 1))


# --------------------------------------------------------------------------- #
# Geometry: sphere on a uniform lattice, cut-link fractional distances
# --------------------------------------------------------------------------- #

@dataclass
class SphereLinks:
    """Cut links of a sphere immersed in the D3Q19 lattice."""

    bx: torch.Tensor        # fluid node index (x)
    by: torch.Tensor
    bz: torch.Tensor
    dirs: torch.Tensor      # direction index pointing from fluid node into solid
    q: torch.Tensor         # fractional distance from fluid node to the wall (0, 1]
    solid: torch.Tensor     # (nx, ny, nz) bool mask of solid nodes
    center: Tuple[float, float, float]
    radius: float

    @property
    def n_links(self) -> int:
        return int(self.q.numel())


def sphere_links(
    nx: int, ny: int, nz: int, diameter: float, center: Optional[Sequence[float]] = None
) -> SphereLinks:
    """Build the cut-link list of a sphere centred in the ``y-z`` mid-plane.

    Nodes whose centre lies inside the sphere (``|x - c| < R``) are solid.
    For every fluid node with a solid neighbour in direction ``e_i`` the
    fractional wall distance ``q`` in ``(0, 1]`` is the positive root of
    ``|x + q e_i - c| = R``.
    """
    radius = 0.5 * float(diameter)
    if center is None:
        # half-integer centre -> exact mirror symmetry about the y-z mid-planes
        center = (0.5 * (nx - 1) - 0.5 * diameter, 0.5 * (ny - 1), 0.5 * (nz - 1))
    cx, cy, cz = (float(c) for c in center)

    xs = torch.arange(nx, dtype=torch.float64) - cx
    ys = torch.arange(ny, dtype=torch.float64) - cy
    zs = torch.arange(nz, dtype=torch.float64) - cz
    r2 = (xs[:, None, None] ** 2 + ys[None, :, None] ** 2 + zs[None, None, :] ** 2)
    solid = r2 < radius * radius

    bx: List[torch.Tensor] = []
    by: List[torch.Tensor] = []
    bz: List[torch.Tensor] = []
    dd: List[torch.Tensor] = []
    qq: List[torch.Tensor] = []

    # candidate fluid nodes = solid mask dilated by one node, minus solid.
    idx = solid.nonzero(as_tuple=False)
    e_all = torch.tensor(_D3Q19_DIRS, dtype=torch.float64)
    # every cut link is generated exactly once: it belongs to a unique solid
    # node s and its fluid node is s + e_d for exactly one direction d
    for d_idx in range(1, 19):
        dx, dy, dz = _D3Q19_DIRS[d_idx]
        nx_i = idx[:, 0] + dx
        ny_i = idx[:, 1] + dy
        nz_i = idx[:, 2] + dz
        ok = (
            (nx_i >= 1) & (nx_i <= nx - 2)
            & (ny_i >= 1) & (ny_i <= ny - 2)
            & (nz_i >= 1) & (nz_i <= nz - 2)
        )
        if not bool(ok.any()):
            continue
        cand = torch.stack([nx_i[ok], ny_i[ok], nz_i[ok]], dim=1)
        cand = cand[~solid[cand[:, 0], cand[:, 1], cand[:, 2]]]
        if cand.numel() == 0:
            continue
        # direction from the fluid node into the solid node is -e_d = opp(d)
        link_dir = _OPP[d_idx]
        er = -e_all[d_idx]
        bxx = cand[:, 0].to(torch.float64) - cx
        byy = cand[:, 1].to(torch.float64) - cy
        bzz = cand[:, 2].to(torch.float64) - cz
        b = bxx * er[0] + byy * er[1] + bzz * er[2]
        c0 = bxx ** 2 + byy ** 2 + bzz ** 2 - radius * radius
        # |x + t e - c|^2 = R^2 with |e|^2 = 1 (face) or 2 (edge) link length:
        #   |e|^2 t^2 + 2 (p.e) t + (|p|^2 - R^2) = 0
        ee = float(er[0] ** 2 + er[1] ** 2 + er[2] ** 2)
        disc = b * b - ee * c0
        valid = disc > 0
        if not bool(valid.any()):
            continue
        cand = cand[valid]
        q = ((-b - torch.sqrt(disc)) / ee)[valid]
        keep = (q > 1e-9) & (q <= 1.0 + 1e-9)
        cand = cand[keep]
        q = q[keep].clamp(max=1.0)
        if cand.numel() == 0:
            continue
        bx.append(cand[:, 0].to(torch.long))
        by.append(cand[:, 1].to(torch.long))
        bz.append(cand[:, 2].to(torch.long))
        dd.append(torch.full((cand.shape[0],), link_dir, dtype=torch.long))
        qq.append(q)

    if not bx:
        raise ValueError("sphere_links: empty cut-link set (diameter too small?)")

    bx_t = torch.cat(bx)
    by_t = torch.cat(by)
    bz_t = torch.cat(bz)
    dirs_t = torch.cat(dd)
    q_t = torch.cat(qq).to(torch.float32)

    # (node, direction) pairs are unique by construction
    return SphereLinks(
        bx=bx_t, by=by_t, bz=bz_t, dirs=dirs_t, q=q_t,
        solid=solid, center=(cx, cy, cz), radius=radius,
    )


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #

class SphereBFLSolver:
    """D3Q19 TRT-LBM with BFL interpolated bounce-back around a sphere.

    Field layout is ``f[nx, ny, nz, 19]`` (direction index contiguous), which
    keeps the per-direction streaming copies and the collision arithmetic
    vectorised over the innermost axis.
    """

    def __init__(self, cfg: SphereBFLConfig):
        if cfg.re <= 0:
            raise ValueError("re must be positive")
        if cfg.diameter < 6:
            raise ValueError("diameter must be >= 6 lattice cells")
        if not (0.0 < cfg.u_lb < 0.15):
            raise ValueError("u_lb must be in (0, 0.15) for weakly compressible LBM")
        tau = cfg.tau
        if tau - 0.5 < 1e-4:
            raise ValueError(
                f"tau-0.5={tau - 0.5:.3e} too small: increase diameter or u_lb"
            )
        self.cfg = cfg
        if cfg.threads > 0:
            torch.set_num_threads(cfg.threads)
        self.dtype = getattr(torch, cfg.dtype)
        self.device = torch.device(cfg.device)

        self.nx, self.ny, self.nz = cfg.grid()
        self.links = sphere_links(self.nx, self.ny, self.nz, cfg.diameter, cfg.center())
        self.solid = self.links.solid
        self.fluid_mask = (~self.solid).to(self.dtype)

        e = torch.tensor(_D3Q19_DIRS, dtype=torch.long)
        self.ex = e[:, 0].to(self.device)
        self.ey = e[:, 1].to(self.device)
        self.ez = e[:, 2].to(self.device)
        self.cx = self.ex.to(self.dtype).view(1, 1, 1, 19)
        self.cy = self.ey.to(self.dtype).view(1, 1, 1, 19)
        self.cz = self.ez.to(self.dtype).view(1, 1, 1, 19)
        self.w = torch.tensor(_D3Q19_W, dtype=self.dtype, device=self.device).view(1, 1, 1, 19)
        self.opp = torch.tensor(_OPP, dtype=torch.long, device=self.device)

        # TRT relaxation rates (magic parameter Lambda = 3/16)
        self.s_plus = 1.0 / tau
        self.s_minus = 1.0 / (0.5 + (3.0 / 16.0) / (tau - 0.5))
        self.a_trt = 0.5 * (self.s_plus + self.s_minus)
        self.b_trt = 0.5 * (self.s_plus - self.s_minus)

        # cut-link tensors on device
        L = self.links
        self.bx = L.bx.to(self.device)
        self.by = L.by.to(self.device)
        self.bz = L.bz.to(self.device)
        self.bdir = L.dirs.to(self.device)
        self.bdir_opp = self.opp[self.bdir]
        self.bq = L.q.to(self.dtype).to(self.device)
        self.bcx = self.ex[self.bdir].to(self.dtype)
        self.bcy = self.ey[self.bdir].to(self.dtype)
        self.bcz = self.ez[self.bdir].to(self.dtype)
        self.lt_half = self.bq < 0.5

        # inlet equilibrium profile (constant along the inflow plane)
        up = cfg.u_lb
        cu = 3.0 * self.ex.to(self.dtype) * up
        self._inlet_dirs = [i for i in range(19) if _D3Q19_DIRS[i][0] > 0]
        self._outlet_dirs = [i for i in range(19) if _D3Q19_DIRS[i][0] < 0]
        self._feq_in = (
            torch.tensor(_D3Q19_W, dtype=self.dtype, device=self.device)
            * (1.0 + cu + 4.5 * (self.ex.to(self.dtype) * up) ** 2 - 1.5 * up * up)
        ).view(1, 1, 19)

        # sponge layer: relax the outgoing solution toward the free stream so
        # that acoustic/vortical waves leave the domain without reflection
        n_sp = int(round(cfg.sponge * cfg.diameter))
        w_sp = torch.zeros(self.nx, dtype=self.dtype, device=self.device)
        if n_sp > 0:
            ramp = torch.linspace(0.0, 1.0, n_sp, dtype=self.dtype, device=self.device)
            w_sp[self.nx - n_sp:] = cfg.sponge_strength * ramp ** 2
        self.sponge_weight = w_sp.view(self.nx, 1, 1, 1)
        self._sponge_active = bool(float(w_sp.abs().sum()) > 0)

        # fields: uniform free stream as the initial condition (already a good
        # approximation of the converged inflow/outflow, saves transient steps)
        rho = torch.ones((self.nx, self.ny, self.nz, 1), dtype=self.dtype, device=self.device)
        ux = torch.full((self.nx, self.ny, self.nz, 1), up, dtype=self.dtype, device=self.device)
        uy = torch.zeros_like(ux)
        uz = torch.zeros_like(ux)
        f0 = self.equilibrium(rho, ux, uy, uz)
        self.f = f0.clone()
        self.f_post = torch.empty_like(self.f)
        self.f_new = torch.empty_like(self.f)

        self._force = torch.zeros(3, dtype=self.dtype, device=self.device)
        self.step_count = 0
        self.wall_time = 0.0

    # -- equilibrium / macroscopic -----------------------------------------
    def equilibrium(self, rho, ux, uy, uz):
        """Second-order equilibrium, ``rho/ux/uy/uz`` shaped (nx, ny, nz, 1)."""
        cu = self.cx * ux + self.cy * uy + self.cz * uz
        usq = ux * ux + uy * uy + uz * uz
        return self.w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)

    def macroscopic(self):
        f = self.f
        rho = f.sum(dim=-1, keepdim=True)
        rho_safe = torch.where(rho.abs() > 1e-12, rho, torch.ones_like(rho))
        ux = (f * self.cx).sum(dim=-1, keepdim=True) / rho_safe
        uy = (f * self.cy).sum(dim=-1, keepdim=True) / rho_safe
        uz = (f * self.cz).sum(dim=-1, keepdim=True) / rho_safe
        return rho, ux, uy, uz

    # -- one LBM step -------------------------------------------------------
    def collide(self):
        """TRT collision, ``f -> f_post`` (no allocation of ``f`` copies)."""
        rho, ux, uy, uz = self.macroscopic()
        feq = self.equilibrium(rho, ux, uy, uz)
        df = self.f - feq
        df_opp = df[..., self.opp]
        self.f_post = self.f - self.a_trt * df - self.b_trt * df_opp
        return rho, ux, uy, uz

    def stream(self):
        """Pull-streaming with slicing (no periodic wrap), then boundary conditions."""
        fp = self.f_post
        fn = self.f_new
        nx, ny, nz = self.nx, self.ny, self.nz
        for i in range(19):
            dx, dy, dz = _D3Q19_DIRS[i]
            if dx == 0 and dy == 0 and dz == 0:
                fn[..., i] = fp[..., i]
                continue
            sx = slice(max(0, -dx), nx - max(0, dx))
            sy = slice(max(0, -dy), ny - max(0, dy))
            sz = slice(max(0, -dz), nz - max(0, dz))
            tx = slice(max(0, dx), nx - max(0, -dx))
            ty = slice(max(0, dy), ny - max(0, -dy))
            tz = slice(max(0, dz), nz - max(0, -dz))
            fn[tx, ty, tz, i] = fp[sx, sy, sz, i]
        # boundary conditions are written into the freshly streamed buffer,
        # only then does it become the active field
        self._apply_boundary_conditions()
        # swap: the streamed+BC buffer becomes the active field
        self.f, self.f_new = self.f_new, self.f

    def _apply_boundary_conditions(self):
        f_new = self.f_new
        f_post = self.f_post
        up = self.cfg.u_lb

        # --- inlet (x = 0): non-reflecting velocity inflow.  Only the
        # populations entering the domain (c_x > 0) are prescribed; the
        # outgoing ones keep the value they just streamed out, so pressure
        # waves can leave instead of being reflected by the inlet plane.
        for i in self._inlet_dirs:
            f_new[0, :, :, i] = self._feq_in[0, 0, i]

        # --- outlet (x = nx-1): zero-gradient extrapolation of the incoming
        # populations only (c_x < 0); outgoing ones are already streamed.
        for i in self._outlet_dirs:
            f_new[-1, :, :, i] = f_post[-2, :, :, i]

        # --- lateral y / z: free-slip specular reflection
        self._free_slip()

        # --- BFL interpolated bounce-back on the cut links
        self._bounce_back()

        # --- outlet sponge layer (wave-absorbing)
        if self._sponge_active:
            rho = f_new.sum(dim=-1, keepdim=True)
            rho_safe = torch.where(rho.abs() > 1e-12, rho, torch.ones_like(rho))
            ux = torch.full_like(rho, self.cfg.u_lb)
            zeros = torch.zeros_like(rho)
            feq_target = self.equilibrium(rho_safe, ux, zeros, zeros)
            f_new += self.sponge_weight * (feq_target - f_new)

    def _free_slip(self):
        f_new = self.f_new
        fp = self.f_post
        dirs = _D3Q19_DIRS
        for i in range(19):
            cx, cy, cz = dirs[i]
            # at the y = 0 wall the populations entering the domain have cy = +1;
            # at y = ny-1 they have cy = -1.  Specular reflection: the incoming
            # population equals the mirrored outgoing (post-collision) one.
            if cy > 0:
                j = dirs.index((cx, -cy, cz))
                f_new[:, 0, :, i] = fp[:, 0, :, j]
            elif cy < 0:
                j = dirs.index((cx, -cy, cz))
                f_new[:, -1, :, i] = fp[:, -1, :, j]
            if cz > 0:
                j = dirs.index((cx, cy, -cz))
                f_new[:, :, 0, i] = fp[:, :, 0, j]
            elif cz < 0:
                j = dirs.index((cx, cy, -cz))
                f_new[:, :, -1, i] = fp[:, :, -1, j]

    def _bounce_back(self):
        """BFL interpolated bounce-back + momentum-exchange force."""
        fp = self.f_post
        f_post_b = fp[self.bx, self.by, self.bz, self.bdir]
        bx2 = self.bx - self.bcx.to(torch.long)
        by2 = self.by - self.bcy.to(torch.long)
        bz2 = self.bz - self.bcz.to(torch.long)
        fp_up = fp[bx2, by2, bz2, self.bdir]                    # f_i^* at x - e_i
        fp_opp = fp[self.bx, self.by, self.bz, self.bdir_opp]   # f_opp^* at x

        q = self.bq
        val_half = 2.0 * q * f_post_b + (1.0 - 2.0 * q) * fp_up
        val_far = (1.0 / (2.0 * q)) * f_post_b + ((2.0 * q - 1.0) / (2.0 * q)) * fp_opp
        # thin solid bridges: if the wall-side neighbour is itself solid the
        # q < 1/2 branch has no valid donor -> fall back to the one-sided branch
        up_is_solid = self.solid[bx2, by2, bz2]
        val = torch.where(
            self.lt_half & ~up_is_solid,
            val_half,
            torch.where(up_is_solid, f_post_b, val_far),
        )

        self.f_new[self.bx, self.by, self.bz, self.bdir_opp] = val

        # momentum-exchange (Ladd) force on the sphere:
        #   F = sum_over_cut_links (f_in + f_refl) * e_link
        exchange = f_post_b + val
        fx = (exchange * self.bcx).sum()
        fy = (exchange * self.bcy).sum()
        fz = (exchange * self.bcz).sum()
        self._force = torch.stack([fx, fy, fz])

    # -- diagnostics --------------------------------------------------------
    @property
    def drag_force(self) -> float:
        return float(self._force[0])

    @property
    def drag_coefficient(self) -> float:
        area = math.pi * (self.cfg.diameter ** 2) / 4.0
        return self.drag_force / (0.5 * self.cfg.u_lb ** 2 * area)

    def mass(self) -> float:
        return float(self.f.sum())

    def run(
        self,
        *,
        max_steps: Optional[int] = None,
        window: Optional[int] = None,
        tol: Optional[float] = None,
        sample_interval: Optional[int] = None,
        progress: bool = False,
    ) -> Dict[str, Any]:
        """Integrate until the drag plateau is reached (or ``max_steps``)."""
        cfg = self.cfg
        max_steps = int(max_steps or cfg.max_steps)
        window = int(window or cfg.window)
        tol = float(tol if tol is not None else cfg.tol)
        sample_interval = int(sample_interval or cfg.sample_interval)

        hist: List[Tuple[int, float, float]] = []   # (step, cd, wall_s)
        t0 = time.time()
        converged = False
        cd = float("nan")
        n_window = max(2, window // sample_interval)
        m0 = self.mass()

        for _ in range(max_steps):
            self.collide()
            self.stream()
            self.step_count += 1
            if self.step_count % sample_interval == 0:
                cd = self.drag_coefficient
                hist.append((self.step_count, cd, time.time() - t0))
                if not math.isfinite(cd):
                    break
                if len(hist) >= n_window:
                    tail = [h[1] for h in hist[-n_window:]]
                    mean = sum(tail) / len(tail)
                    spread = (max(tail) - min(tail)) / abs(mean) if mean else float("inf")
                    if spread < tol and self.step_count > 4 * window:
                        converged = True
                        break
            if progress and self.step_count % (20 * sample_interval) == 0:
                print(f"  step {self.step_count}: Cd={cd:.5f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        self.wall_time += time.time() - t0
        rho, ux, uy, uz = self.macroscopic()
        mass_drift = abs(self.mass() - m0) / abs(m0) * 1e6  # ppm

        cd_tail = float("nan")
        if hist:
            n_tail = min(len(hist), max(n_window, 10))
            vals = [h[1] for h in hist[-n_tail:]]
            cd_tail = sum(vals) / len(vals)

        cd_ref = reference_cd(cfg.re)
        return {
            "re": cfg.re,
            "diameter": cfg.diameter,
            "cd": cd_tail,
            "cd_ref": cd_ref,
            "cd_err_pct": abs(cd_tail - cd_ref) / cd_ref * 100.0 if cd_ref else float("nan"),
            "reference_source": reference_source(cfg.re),
            "force_x": self.drag_force,
            "steps": self.step_count,
            "converged": converged,
            "wall_s": self.wall_time,
            "nu_lb": cfg.nu_lb,
            "tau": cfg.tau,
            "u_lb": cfg.u_lb,
            "mach": cfg.mach,
            "mass_drift_ppm": mass_drift,
            "n_cut_links": self.links.n_links,
            "domain": [self.nx, self.ny, self.nz],
            "blockage_pct": (math.pi * cfg.diameter ** 2 / 4.0)
                             / (self.ny * self.nz) * 100.0,
            "cd_history": hist[:: max(1, len(hist) // 200)],
            "finite": bool(math.isfinite(cd_tail) and math.isfinite(mass_drift)),
            "fields": {},
        }


# --------------------------------------------------------------------------- #
# Convenience one-shot driver
# --------------------------------------------------------------------------- #

def run_sphere_case(
    cfg: Optional[SphereBFLConfig] = None,
    *,
    save_fields: bool = False,
    progress: bool = False,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a solver, integrate it and return a JSON-serialisable result dict."""
    if cfg is None:
        cfg = SphereBFLConfig(**overrides)
    elif overrides:
        raise TypeError("pass either cfg or keyword overrides, not both")

    solver = SphereBFLSolver(cfg)
    res = solver.run(progress=progress)

    if save_fields or cfg.save_fields:
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rho, ux, uy, uz = solver.macroscopic()
        path = out / f"{cfg.tag}_fields.pt"
        torch.save(
            {
                "rho": rho.cpu(),
                "ux": ux.cpu(),
                "uy": uy.cpu(),
                "uz": uz.cpu(),
                "solid": solver.links.solid,
                "center": solver.links.center,
                "radius": solver.links.radius,
                "cfg": cfg.to_dict(),
                "cd": res["cd"],
            },
            path,
        )
        res["fields"] = {"path": str(path)}

    res["config"] = cfg.to_dict()
    return res


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tensorlbm.sphere_drag_bfl",
        description="Sphere drag coefficient with BFL interpolated bounce-back "
                    "(D3Q19 TRT-LBM).",
    )
    p.add_argument("--re", type=float, default=100.0, help="Reynolds number")
    p.add_argument("--diameter", type=int, default=16, help="sphere diameter in lattice cells")
    p.add_argument("--u", type=float, default=0.05, help="free-stream lattice velocity")
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--window", type=int, default=500, help="plateau window (steps)")
    p.add_argument("--tol", type=float, default=0.01, help="plateau tolerance (relative)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--output-dir", default="artifacts/sphere_drag_bfl")
    p.add_argument("--save-fields", action="store_true")
    p.add_argument("--out", default=None, help="write result JSON to this path")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = SphereBFLConfig(
        re=args.re,
        diameter=args.diameter,
        u_lb=args.u,
        max_steps=args.max_steps,
        window=args.window,
        tol=args.tol,
        threads=args.threads,
        output_dir=args.output_dir,
        save_fields=args.save_fields,
    )
    res = run_sphere_case(cfg, progress=not args.quiet)
    print(
        f"Re={res['re']:g} D={res['diameter']}  Cd={res['cd']:.4f}  "
        f"ref={res['cd_ref']:.4f} ({res['reference_source']})  "
        f"err={res['cd_err_pct']:.2f}%  steps={res['steps']}  "
        f"converged={res['converged']}  wall={res['wall_s']:.1f}s"
    )
    payload = {k: v for k, v in res.items()}
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=float))
        print(f"wrote {path}")
    return 0 if res["finite"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())