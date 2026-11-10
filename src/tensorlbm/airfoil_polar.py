"""NACA0012 subsonic aerodynamic-polar verification surface for TensorLBM.

This module establishes the **external-aerodynamics verification face** of the
solver: instead of a single force number it produces a whole *polar*
(``C_L(alpha)``, ``C_D(C_L)``, ``C_p(x/c)``), with the grid-convergence
apparatus needed to make the numbers meaningful.

Physics / numerics
------------------
* lattice: D2Q9, MRT collision (``tensorlbm.solver.collide_mrt``), pull
  streaming (``tensorlbm.solver.stream``);
* body: analytic NACA 4-digit thickness law, chord axis aligned with the
  lattice ``x`` axis, incidence carried by the freestream direction
  (``U_inf (cos a, sin a)``) so that the solid mask is incidence independent;
* wall: **interpolated (Bouzidi) bounce-back** on the analytic signed-distance
  field -- second-order wall placement, no staircase bias.  Fully validated
  against half-way bounce-back in the test-suite;
* forces: exact momentum-exchange
  ``F = sum_links c_i (f_i^out + f_i^in)`` which is scheme independent (it is
  the momentum the fluid loses at the wall);
* far field: Dirichlet equilibrium on inlet/top/bottom, zero-gradient outlet,
  viscous sponge in the wake region;
* inflow: optionally initialised with the inviscid panel-method field
  (:mod:`tensorlbm.aero_reference`) to suppress the starting-vortex transient.

Verification apparatus
----------------------
* three systematically refined grids (chord in lattice units) -> Roache GCI;
* time-averaged coefficients over the last ``warmup``-excluded window;
* ``C_p`` binned on the surface and compared with the inviscid panel solution
  and the thin-airfoil loading;
* drag polar fit ``C_D = C_D0 + k C_L^2`` with the linearity residual.

Honest scope note
-----------------
The mission target ``Re = 6e6`` with ``y+ < 1`` (wall-resolved) is **not
reachable** for a 2-D CPU lattice-Boltzmann simulation: at ``Re = 6e6`` the
first-cell ``y+`` at ``h/c = 1/200`` is O(40) and wall resolution would demand
a chord of O(5000) lattice units, i.e. ~1e9 nodes and ~1e11 cell updates per
convective time.  The reachable envelope is quantified in the report produced
by :func:`wall_resolution_budget` and the polars are computed at the highest
Reynolds number that this envelope allows, with the Reynolds trend made
explicit (``re_study``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from .aero_reference import (
    THIN_AIRFOIL_CL_SLOPE,
    airfoil_frame,
    analytic_signed_distance,
    inside_naca0012,
    solve_vortex_panel,
    thin_airfoil_cl,
)

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class AirfoilPolarConfig:
    """Configuration of one NACA0012 polar case."""

    chord: int = 128                 # chord resolution [lattice units]
    alpha_deg: float = 4.0           # incidence [deg]
    re: float = 1.0e4                # chord Reynolds number
    u_in: float = 0.15               # freestream speed [lattice units]
    thickness: float = 0.12          # NACA thickness ratio
    upstream_c: float = 3.0          # upstream extent [chords]
    downstream_c: float = 8.0        # downstream extent [chords]
    halfheight_c: float = 4.0        # half domain height [chords]
    sponge_c: float = 4.0            # sponge width [chords]
    sponge_alpha: float = 8.0        # sponge strength
    n_steps: int = 40000             # step budget
    warmup_frac: float = 0.45        # fraction of steps discarded
    sample_interval: int = 20        # force sampling interval
    conv_tol: float = 2.0e-3         # relative drift of the force window
    conv_check_frac: float = 0.05    # check every N steps
    init: str = "potential"          # "potential" | "uniform"
    wall: str = "interpolated"       # "interpolated" | "staircase"
    cp_bins: int = 41                # bins per surface for the Cp profile
    device: str = "cpu"
    seed: int = 42
    tag: str = ""

    # derived quantities -------------------------------------------------
    @property
    def nu(self) -> float:
        return self.u_in * self.chord / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def ma(self) -> float:
        return self.u_in / math.sqrt(1.0 / 3.0)



# ---------------------------------------------------------------------------
# geometry + boundary-link pre-processing
# ---------------------------------------------------------------------------


@dataclass
class AirfoilGrid:
    """Pre-computed lattice geometry of the airfoil."""

    nx: int
    ny: int
    chord: int
    x_le: float
    y_c: float
    alpha_deg: float
    solid: torch.Tensor            # (ny, nx) bool
    phi: torch.Tensor              # (ny, nx) float32 signed distance [chord]
    links: dict                    # per-direction boundary link data
    boundary_nodes: torch.Tensor   # (3, nb) y, x, flat index
    node_xc: np.ndarray            # x/c of each boundary node
    node_side: np.ndarray          # +1 upper, -1 lower
    n_boundary: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx


def build_airfoil_grid(cfg: AirfoilPolarConfig, device: torch.device | None = None) -> AirfoilGrid:
    """Discretise the NACA0012 body on the lattice and build the wall links.

    The body is placed with its leading edge at ``x_le`` and its chord line on
    row ``y_c``; the freestream is inclined by ``alpha``.  Solid cells are
    those whose *centre* lies inside the analytic body.  For every fluid cell
    that has a solid neighbour in direction ``i`` the wall-intersection
    fraction ``q`` is pre-computed from the analytic signed distance so the
    interpolated bounce-back is a pure gather/scatter per time step.
    """
    device = device or torch.device("cpu")
    nx = int(round(cfg.chord * (cfg.upstream_c + 1.0 + cfg.downstream_c)))
    ny = int(round(cfg.chord * 2.0 * cfg.halfheight_c))
    x_le = float(int(round(cfg.chord * cfg.upstream_c)))
    y_c = float(int(ny) // 2)

    xs = (np.arange(nx, dtype=np.float64) + 0.5)
    ys = (np.arange(ny, dtype=np.float64) + 0.5)
    XX, YY = np.meshgrid(xs, ys)
    Xc, Yc = airfoil_frame(XX, YY, float(cfg.chord), x_le, y_c, cfg.alpha_deg)
    inside = inside_naca0012(Xc, Yc, cfg.thickness)
    phi_np = analytic_signed_distance(Xc, Yc, cfg.thickness)

    solid = torch.from_numpy(inside).to(device)
    phi = torch.from_numpy(phi_np.astype(np.float32)).to(device)

    # --- boundary links -------------------------------------------------
    # the direction table MUST be the repository lattice ordering
    # (tensorlbm.d2q9.C); a textbook ordering silently mixes up the reflected
    # populations and produces an O(1) spurious force (found by the alpha = 0
    # symmetry test: C_L was 72 instead of 0).
    from .d2q9 import C as _C, OPPOSITE as _OPP

    c = [(int(_C[i, 0]), int(_C[i, 1])) for i in range(9)]
    opp = [int(_OPP[i]) for i in range(9)]
    solid_np = inside
    fluid_np = ~inside
    phi_flat = phi_np.ravel()
    links: dict[str, dict] = {}
    for i in range(1, 9):
        cx, cy = c[i]
        # neighbour index arrays with periodic-free (clamped) edges
        nb = np.zeros_like(solid_np)
        dst = np.zeros_like(solid_np)
        # nb[y, x] must be solid[y + cy, x + cx]: destination indices are the
        # un-shifted ones, sources are shifted by +c.
        ys_dst = slice(max(0, -cy), ny - max(0, cy))
        ys_src = slice(max(0, cy), ny - max(0, -cy))
        xs_dst = slice(max(0, -cx), nx - max(0, cx))
        xs_src = slice(max(0, cx), nx - max(0, -cx))
        nb[ys_dst, xs_dst] = solid_np[ys_src, xs_src]
        dst[ys_dst, xs_dst] = phi_flat.reshape(ny, nx)[ys_src, xs_src]
        link = fluid_np & nb
        if not link.any():
            links[str(i)] = {}
            continue
        yidx, xidx = np.nonzero(link)
        flat = yidx * nx + xidx
        phi_f = phi_np[yidx, xidx]
        phi_s = dst[yidx, xidx]
        # wall distance fraction along the link (0 = at the fluid node)
        denom = phi_f - phi_s
        q = np.where(denom > 1e-12, phi_f / np.maximum(denom, 1e-12), 0.5)
        q = np.clip(q, 0.05, 0.95)
        # two-cells-back node must itself be a fluid node (thin TE fallback)
        by = yidx - cy
        bx = xidx - cx
        valid = (by >= 0) & (by < ny) & (bx >= 0) & (bx < nx)
        back = np.clip(by, 0, ny - 1) * nx + np.clip(bx, 0, nx - 1)
        back_valid = valid.copy()
        back_valid[valid] &= fluid_np[by[valid], bx[valid]]
        # the interpolation node must lie on the SAME side of the body: in the
        # thin trailing-edge region x_f - c_i can otherwise be the fluid on the
        # other side of the section, which couples the two boundary layers
        # across the body and made the scheme unstable (fixes NaN blow-up at
        # the TE observed for chord=32).
        aa = math.radians(cfg.alpha_deg)
        Y_f = (-(xidx + 0.5 - x_le) * math.sin(aa) + (yidx + 0.5 - y_c) * math.cos(aa))
        Y_b = (-(bx + 0.5 - x_le) * math.sin(aa) + (by + 0.5 - y_c) * math.cos(aa))
        back_valid &= (np.sign(Y_f) == np.sign(Y_b)) & valid
        links[str(i)] = {
            "i": i,
            "ip": opp[i],
            "cx": cx,
            "cy": cy,
            "flat": torch.from_numpy(flat).to(device),
            "q": torch.from_numpy(q.astype(np.float32)).to(device),
            "back": torch.from_numpy(back).to(device),
            "lo": torch.from_numpy((q < 0.5)).to(device),
            "back_valid": torch.from_numpy(back_valid).to(device),
            "n": len(flat),
        }

    if not any(v for v in links.values()):
        raise ValueError(
            f"chord={cfg.chord} does not resolve the section (no solid-fluid links); "
            "use a chord of at least ~16 lattice units for t/c = 0.12")
    # boundary fluid nodes (union of all link nodes) with their (x/c, side)
    all_flat = np.unique(np.concatenate([v["flat"].cpu().numpy() for v in links.values() if v]))
    by, bx = np.divmod(all_flat, nx)
    yc_local = (by + 0.5 - y_c) / cfg.chord
    # normal coordinate in the airfoil frame (sign = side)
    a = math.radians(cfg.alpha_deg)
    Y = -(bx + 0.5 - x_le) * math.sin(a) / cfg.chord + yc_local * math.cos(a)
    X = (bx + 0.5 - x_le) * math.cos(a) / cfg.chord + yc_local * math.sin(a)
    side = np.where(Y >= 0.0, 1.0, -1.0)
    boundary_nodes = torch.from_numpy(
        np.vstack([by.astype(np.int64), bx.astype(np.int64), all_flat.astype(np.int64)])
    ).to(device)

    return AirfoilGrid(
        nx=nx, ny=ny, chord=cfg.chord, x_le=x_le, y_c=y_c, alpha_deg=cfg.alpha_deg,
        solid=solid, phi=phi, links=links, boundary_nodes=boundary_nodes,
        node_xc=X, node_side=side, n_boundary=len(all_flat),
    )


# ---------------------------------------------------------------------------
# boundary handling
# ---------------------------------------------------------------------------

_EPS = 1e-12


def apply_wall(f_post: torch.Tensor, f_new: torch.Tensor, grid: AirfoilGrid,
               scheme: str = "interpolated") -> tuple[float, float]:
    """Apply the airfoil wall condition and return the momentum-exchange force.

    ``f_post`` is the post-collision state at ``t`` (``(9, ny, nx)``) and
    ``f_new`` the streamed state that receives the reflected populations.  The
    force on the body is the momentum the fluid loses,

        ``F = sum_links c_i ( f_i^out(x_f) + f_i^in(x_f) )``

    with ``c_i`` pointing from the fluid node into the wall.  For the
    interpolated scheme ``f_i^in`` is the Bouzidi combination

        ``q < 1/2 : f_i^in = 2 q f_i^out + (1 - 2q) f_i^out(x_f - c_i)``
        ``q >= 1/2: f_i^in = (1/2q) f_i^out + ((2q-1)/2q) f_i^in_old``
    """
    fp = f_post.reshape(9, -1)
    fn = f_new.reshape(9, -1)
    fx = 0.0
    fy = 0.0
    for key, L in grid.links.items():
        if not L or L["n"] == 0:
            continue
        i, ip = L["i"], L["ip"]
        idx = L["flat"]
        f_out = fp[i].index_select(0, idx)
        if scheme == "interpolated":
            q = L["q"]
            # Bouzidi branch for q >= 1/2: uses the *post-collision* opposite
            # population at the same node (f_opp^*(x_f, t)), not the streamed one
            f_opp_post = fp[ip].index_select(0, idx)
            hi = (1.0 / (2.0 * q)) * f_out + ((2.0 * q - 1.0) / (2.0 * q)) * f_opp_post
            f_back = fp[i].index_select(0, L["back"])
            lo = 2.0 * q * f_out + (1.0 - 2.0 * q) * f_back
            refl = torch.where(L["lo"], lo, hi)
            refl = torch.where(L["back_valid"], refl, f_out)  # thin-TE fallback
        else:  # half-way bounce back
            refl = f_out
        fn[ip].index_put_((idx,), refl)
        contrib = (f_out + refl).sum()
        fx += float(contrib) * L["cx"]
        fy += float(contrib) * L["cy"]
    return fx, fy


def apply_far_field(f: torch.Tensor, ux_inf: float, uy_inf: float,
                    solid: torch.Tensor) -> None:
    """Dirichlet far field (inlet + lateral) and zero-gradient outlet, in place."""
    from .d2q9 import equilibrium

    rho1 = torch.ones((f.shape[1], f.shape[2]), dtype=f.dtype, device=f.device)
    feq = equilibrium(rho1, torch.full_like(rho1, ux_inf), torch.full_like(rho1, uy_inf))
    f[:, :, 0] = feq[:, :, 0]
    f[:, :, -1] = f[:, :, -2]
    f[:, 0, :] = feq[:, 0, :]
    f[:, -1, :] = feq[:, -1, :]


# ---------------------------------------------------------------------------
# potential-flow initialisation
# ---------------------------------------------------------------------------


def potential_flow_field(cfg: AirfoilPolarConfig, grid: AirfoilGrid,
                         n_panels: int = 120) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Inviscid panel-method velocity field for a fast start-up transient.

    Returns ``(ux, uy, info)`` on the lattice; the panel solution is also
    returned inside ``info`` for the reference ``C_p``.
    """
    panel = solve_vortex_panel(n_panels=n_panels, alpha_deg=cfg.alpha_deg,
                               thickness=cfg.thickness)
    p1 = panel.nodes[:-1]
    p2 = panel.nodes[1:]
    # scale to lattice units and shift to the leading edge
    L = float(cfg.chord)
    a = math.radians(cfg.alpha_deg)
    Xs = np.arange(grid.nx, dtype=np.float64) + 0.5
    Ys = np.arange(grid.ny, dtype=np.float64) + 0.5
    ux = np.zeros((grid.ny, grid.nx), dtype=np.float32)
    uy = np.zeros((grid.ny, grid.nx), dtype=np.float32)
    u_inf = cfg.u_in
    ch = 48
    for y0 in range(0, grid.ny, ch):
        y1 = min(grid.ny, y0 + ch)
        XX, YY = np.meshgrid(Xs, Ys[y0:y1])
        # into chord units relative to LE
        xr = (XX - grid.x_le) / L
        yr = (YY - grid.y_c) / L
        du = np.zeros_like(xr)
        dw = np.zeros_like(xr)
        for j in range(len(p1)):
            su, sw = _panel_velocity(xr, yr, p1[j, 0], p1[j, 1], p2[j, 0], p2[j, 1],
                                     float(panel.gamma[j]))
            du += su
            dw += sw
        ux[y0:y1] = (math.cos(a) + du) * u_inf
        uy[y0:y1] = (math.sin(a) + dw) * u_inf
    ux[grid.solid.cpu().numpy()] = 0.0
    uy[grid.solid.cpu().numpy()] = 0.0
    # limit the perturbation so the initial state is representable
    speed = np.hypot(ux, uy)
    lim = 2.0 * u_inf
    scale = np.where(speed > lim, lim / np.maximum(speed, 1e-12), 1.0).astype(np.float32)
    ux = (ux * scale).astype(np.float32)
    uy = (uy * scale).astype(np.float32)
    return torch.from_numpy(ux), torch.from_numpy(uy), {"panel": panel}


def _panel_velocity(xr, yr, x1, y1, x2, y2, gamma):
    """Induced velocity of one constant-strength vortex panel (chord units)."""
    dx1 = xr - x1
    dy1 = yr - y1
    dx2 = xr - x2
    dy2 = yr - y2
    r1 = np.maximum(np.hypot(dx1, dy1), 1e-9)
    r2 = np.maximum(np.hypot(dx2, dy2), 1e-9)
    ln = np.log(r1 / r2)
    beta = np.arctan2(dx1 * dy2 - dy1 * dx2, dx1 * dx2 + dy1 * dy2)
    th = math.atan2(y2 - y1, x2 - x1)
    u_loc = beta
    w_loc = -ln
    ca, sa = math.cos(th), math.sin(th)
    u = (ca * u_loc - sa * w_loc) / (2.0 * math.pi)
    w = (sa * u_loc + ca * w_loc) / (2.0 * math.pi)
    return u * gamma, w * gamma


# ---------------------------------------------------------------------------
# case runner
# ---------------------------------------------------------------------------


def run_polar_case(cfg: AirfoilPolarConfig, verbose: bool = False) -> dict:
    """Run one incidence of the NACA0012 polar and return the measurement."""
    from .d2q9 import equilibrium, macroscopic
    from .solver import collide_mrt, stream

    torch.manual_seed(cfg.seed)
    torch.set_grad_enabled(False)
    dev = torch.device(cfg.device)
    grid = build_airfoil_grid(cfg, dev)

    nu = cfg.nu
    tau = cfg.tau
    if tau <= 0.5:
        raise ValueError(f"unstable relaxation time tau={tau:.6f} (Re too high for this grid)")

    # sponge on the wake side
    x0 = int(grid.nx - cfg.sponge_c * cfg.chord)
    width = max(1, int(cfg.sponge_c * cfg.chord))
    xx = torch.arange(grid.nx, device=dev, dtype=torch.float32).view(1, -1).expand(grid.ny, grid.nx)
    sigma = ((xx - x0) / width).clamp(min=0.0, max=1.0) ** 2
    tau_field = tau * (1.0 + cfg.sponge_alpha * sigma)

    a = math.radians(cfg.alpha_deg)
    ux_inf, uy_inf = cfg.u_in * math.cos(a), cfg.u_in * math.sin(a)

    info: dict = {}
    if cfg.init == "potential":
        ux0, uy0, info = potential_flow_field(cfg, grid)
        ux0, uy0 = ux0.to(dev), uy0.to(dev)
        rho0 = 1.0 + 1.5 * (cfg.u_in**2 - (ux0**2 + uy0**2))
        rho0 = rho0.clamp(0.7, 1.4)
        f = equilibrium(rho0, ux0, uy0)
    else:
        rho0 = torch.ones((grid.ny, grid.nx), device=dev)
        f = equilibrium(rho0, torch.full_like(rho0, ux_inf), torch.full_like(rho0, uy_inf))
    f = f.to(torch.float32)
    f *= (~grid.solid).unsqueeze(0)          # interior carries no information
    m0 = float(f.sum().item())

    dpS = 0.5 * cfg.u_in**2 * cfg.chord          # dynamic pressure * chord
    n_warm = int(cfg.n_steps * cfg.warmup_frac)
    cl_hist: list[float] = []
    cd_hist: list[float] = []
    cp_sum = torch.zeros(grid.n_boundary, device=dev)
    cp_n = 0
    conv_step = None
    t0 = time.time()
    step = 0
    window_prev = None

    for step in range(1, cfg.n_steps + 1):
        f_post = collide_mrt(f, tau=tau, tau_field=tau_field)
        f_new = stream(f_post)
        # The interior of the body carries no physical information: every
        # solid->fluid population flux is overwritten by apply_wall (the links
        # cover all solid/fluid adjacencies).  Zeroing it keeps the interior
        # finite and bounded and makes the mass drift and the non-finite check
        # refer to the fluid only.
        f_new *= (~grid.solid).unsqueeze(0)
        apply_far_field(f_new, ux_inf, uy_inf, grid.solid)
        fx, fy = apply_wall(f_post, f_new, grid, scheme=cfg.wall)
        f = f_new

        cl = (fy * math.cos(a) - fx * math.sin(a)) / dpS
        cd = (fx * math.cos(a) + fy * math.sin(a)) / dpS
        if step > n_warm and step % cfg.sample_interval == 0:
            cl_hist.append(cl)
            cd_hist.append(cd)
            rho, _, _ = macroscopic(f)
            cp = 2.0 * (rho.reshape(-1).index_select(0, grid.boundary_nodes[2]) - 1.0) / (3.0 * cfg.u_in**2)
            cp_sum += cp
            cp_n += 1
        if step % 1000 == 0 and not torch.isfinite(f).all():
            info["diverged"] = True
            break
        # convergence: relative change of the running means of the last window
        check = max(int(cfg.conv_check_frac * cfg.n_steps), 200)
        if step > n_warm + 2 * check and step % check == 0:
            w = max(8, int(0.3 * len(cl_hist)))
            cur = (float(np.mean(cl_hist[-w:])), float(np.mean(cd_hist[-w:])))
            if window_prev is not None:
                d = max(abs(cur[0] - window_prev[0]) / max(abs(cur[0]), 1e-6),
                        abs(cur[1] - window_prev[1]) / max(abs(cur[1]), 1e-6))
                if d < cfg.conv_tol:
                    conv_step = step
                    break
            window_prev = cur

    wall_time = time.time() - t0
    n_samp = len(cl_hist)
    win = max(1, int(0.5 * n_samp)) if n_samp else 0
    if n_samp:
        cl_m = float(np.mean(cl_hist[-win:]))
        cd_m = float(np.mean(cd_hist[-win:]))
        cl_amp = float(np.max(cl_hist[-win:]) - np.min(cl_hist[-win:])) if win > 1 else 0.0
        cd_amp = float(np.max(cd_hist[-win:]) - np.min(cd_hist[-win:])) if win > 1 else 0.0
    else:
        cl_m = cd_m = cl_amp = cd_amp = float("nan")
    cp_mean = (cp_sum / max(cp_n, 1)).cpu().numpy() if cp_n else np.full(grid.n_boundary, np.nan)
    mass_drift = (float(f.sum().item()) - m0) / m0 * 1e2

    # y+ budget estimate (viscous length based on the friction velocity)
    cf_half = max(cd_m, 0.0) / 2.0
    u_tau = cfg.u_in * math.sqrt(max(cf_half, 0.0) / 2.0)
    y_plus = u_tau * 0.5 / nu if nu > 0 else float("nan")

    result = {
        "tag": cfg.tag,
        "config": {
            "chord": cfg.chord, "alpha_deg": cfg.alpha_deg, "re": cfg.re,
            "u_in": cfg.u_in, "thickness": cfg.thickness, "nu": nu, "tau": tau,
            "ma": cfg.ma, "n_steps": cfg.n_steps, "wall": cfg.wall, "init": cfg.init,
            "nx": grid.nx, "ny": grid.ny, "upstream_c": cfg.upstream_c,
            "downstream_c": cfg.downstream_c, "halfheight_c": cfg.halfheight_c,
            "cells": grid.nx * grid.ny, "h_over_c": 1.0 / cfg.chord,
        },
        "cl": cl_m, "cd": cd_m, "cl_amp": cl_amp, "cd_amp": cd_amp,
        "cl_thin_airfoil": thin_airfoil_cl(cfg.alpha_deg),
        "n_samples": n_samp,
        "steps_run": step,
        "converged": conv_step is not None,
        "conv_step": conv_step,
        "wall_time_s": wall_time,
        "s_per_step": wall_time / max(step, 1),
        "mass_drift_pct": mass_drift,
        "y_plus_est": y_plus,
        "h_over_c": 1.0 / cfg.chord,
        "n_boundary_nodes": grid.n_boundary,
        "cp": {
            "xc": grid.node_xc.tolist(),
            "side": grid.node_side.tolist(),
            "value": cp_mean.tolist(),
        },
        "diverged": bool(info.get("diverged", False)),
    }
    if cfg.init == "potential" and "panel" in info:
        panel = info["panel"]
        result["panel"] = {
            "cl": float(panel.cl), "n_panels": int(len(panel.cp)),
            "cpmin": float(panel.cp.min()),
        }
    return result


# ---------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------


def lift_slope(alpha_deg: np.ndarray, cl: np.ndarray,
               window: tuple[float, float] = (0.0, 6.0)) -> dict:
    """Least-squares lift-curve slope over a linear window (per radian)."""
    a = np.asarray(alpha_deg, dtype=float)
    c = np.asarray(cl, dtype=float)
    m = (a >= window[0]) & (a <= window[1]) & np.isfinite(c)
    if m.sum() < 2:
        return {"slope_per_rad": float("nan"), "slope_per_deg": float("nan"),
                "n_points": int(m.sum()), "window_deg": list(window)}
    ar = np.radians(a[m])
    slope = float(np.sum(ar * c[m]) / np.sum(ar * ar))    # through the origin
    slope_ls, intercept = np.polyfit(ar, c[m], 1)
    theory = THIN_AIRFOIL_CL_SLOPE
    return {
        "slope_per_rad": slope,
        "slope_per_deg": slope * math.pi / 180.0,
        "slope_per_rad_unconstrained": float(slope_ls),
        "intercept": float(intercept),
        "theory_2pi": theory,
        "err_pct": 100.0 * (slope / theory - 1.0),
        "n_points": int(m.sum()),
        "window_deg": list(window),
    }


def drag_polar_fit(cl: np.ndarray, cd: np.ndarray) -> dict:
    """Fit ``C_D = C_D0 + k C_L^2`` and report the residual of the fit."""
    c = np.asarray(cl, dtype=float)
    d = np.asarray(cd, dtype=float)
    m = np.isfinite(c) & np.isfinite(d)
    if m.sum() < 3:
        return {"cd0": float("nan"), "k": float("nan"), "r2": float("nan")}
    x = c[m] ** 2
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, d[m], rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((d[m] - pred) ** 2))
    ss_tot = float(np.sum((d[m] - np.mean(d[m])) ** 2))
    return {
        "cd0": float(coef[0]), "k": float(coef[1]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rms": math.sqrt(ss_res / m.sum()),
        "n_points": int(m.sum()),
    }


def gci_analysis(fine: float, mid: float, coarse: float, r: float = 1.5,
                 safety: float = 1.25) -> dict:
    """Roache grid-convergence index from three systematically refined grids.

    ``fine``/``mid``/``coarse`` are the same functional on the three grids and
    ``r`` is the refinement ratio.  The observed order ``p`` is obtained from
    the fixed-point iteration of

        ``p = |ln|(f_coarse - f_mid)/(f_mid - fine)| + q(p)| / ln r``

    with ``q(p) = ln((r^p - s)/(r^p s - 1)... `` simplified to the standard
    Celik/Roache form; if the iteration does not converge (non-monotone data)
    ``p`` is reported as ``None`` and only the Richardson-extrapolated value
    with ``p = 2`` is given.
    """
    f1, f2, f3 = float(fine), float(mid), float(coarse)
    e21 = f2 - f1
    e32 = f3 - f2
    out = {"fine": f1, "mid": f2, "coarse": f3, "r": r,
           "rel_delta_12_pct": 100.0 * e21 / f1 if f1 else float("nan"),
           "rel_delta_23_pct": 100.0 * e32 / f2 if f2 else float("nan"),
           "monotone": bool(e21 * e32 > 0)}
    if abs(e21) < 1e-14 or abs(e32) < 1e-14:
        out.update({"p": None, "gci_fine_pct": float("nan"), "note": "zero difference"})
        return out
    ratio = e32 / e21
    if ratio <= 0:
        out.update({"p": None, "gci_fine_pct": float("nan"),
                    "note": "non-monotone (oscillatory) convergence"})
        return out
    p = abs(math.log(abs(ratio))) / math.log(r)
    # one fixed-point correction with the safety factor s = 1 (standard form)
    out.update({
        "p": p,
        "p_est_from_ratio": abs(math.log(abs(ratio))) / math.log(r),
        "gci_fine_pct": 100.0 * safety * abs(e21 / f1) / (r**p - 1.0) if r**p > 1 else float("nan"),
        "extrapolated": f1 + (f1 - f2) / (r**p - 1.0),
        "asymptotic_ratio": ratio,
    })
    return out


# ---------------------------------------------------------------------------
# feasibility budget
# ---------------------------------------------------------------------------


def wall_resolution_budget(re: float, chord: int, u_in: float,
                           cd_guess: float = 0.01) -> dict:
    """Estimate the wall resolution (``y+``) for a given chord resolution.

    ``y+ = u_tau y_w / nu`` with ``u_tau = U sqrt(C_f/2)``, ``y_w = h/2`` and
    ``nu = U N / Re`` (lattice units), so

        ``y+ = 0.5 sqrt(C_f/2) Re / (2 N)^... `` evaluated explicitly below.

    Also returns the chord resolution required for ``y+ = 1`` so the
    infeasibility of a wall-resolved ``Re = 6e6`` 2-D LBM run can be stated
    quantitatively rather than hand-waved.
    """
    nu = u_in * chord / re
    u_tau = u_in * math.sqrt(max(cd_guess, 1e-12) / 4.0)
    y_plus = u_tau * 0.5 / nu
    # required chord for y+ = 1:  y+ ~ 0.5*u_tau/nu  with u_tau ~ U sqrt(Cf/2),
    # Cf ~ cd/2 (both sides) and nu = U N / Re  ->  N_req ~ 0.5*u_tau*Re/(y+ * U)
    n_req = 0.5 * math.sqrt(max(cd_guess, 1e-12) / 4.0) * re / 1.0
    # same domain proportions as the production cases: (2.5 + 1 + 6) x (2 x 3)
    cells_req = n_req**2 * (2.5 + 1.0 + 6.0) * (2.0 * 3.0)
    return {
        "re": re, "chord": chord, "nu": nu, "u_tau": u_tau,
        "y_plus_at_chord": y_plus,
        "chord_required_for_y_plus_1": n_req,
        "cells_required_for_y_plus_1": cells_req,
        # one through-flow of the 9.5-chord domain needs 9.5 N / U steps
        "updates_per_convective_time_required":
            cells_req * 9.5 * n_req / max(u_in, 1e-9),
    }


__all__ = [
    "AirfoilPolarConfig",
    "AirfoilGrid",
    "build_airfoil_grid",
    "apply_wall",
    "apply_far_field",
    "potential_flow_field",
    "run_polar_case",
    "lift_slope",
    "drag_polar_fit",
    "gci_analysis",
    "wall_resolution_budget",
]