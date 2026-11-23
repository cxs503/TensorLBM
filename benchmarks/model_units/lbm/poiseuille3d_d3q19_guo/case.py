"""Plane Poiseuille flow in 3D, D3Q19 + BGK + Guo force + half-way bounce-back.

Same analytic reference as the 2D unit, with two extra periodic directions:

    u_x(y) = G / (2 nu) * (y + 1/2) * (H - y - 1/2),   H = ny,   nu = (tau - 1/2)/3

This unit exists to prove the framework is not 2D-shaped: the lattice, the field
rank (``(19, nz, ny, nx)``) and the streaming gather all come from the unit's own
manifest, so a 3D model directory builds and is verified on the same CI matrix as
the 2D ones.
"""

from __future__ import annotations

import torch

from tensorlbm import d3q19

_STREAM_CACHE: dict = {}


def _opposite_table(c: torch.Tensor) -> list[int]:
    return [int((c == -c[i]).all(dim=1).nonzero().flatten()[0]) for i in range(c.shape[0])]


def _stream_bounce_back(f: torch.Tensor, c: torch.Tensor, opp: list[int]) -> torch.Tensor:
    """Stream with periodic x/z and half-way bounce-back at the two y walls."""
    q, nz, ny, nx = f.shape
    key = (q, nz, ny, nx)
    if key not in _STREAM_CACHE:
        zs = torch.arange(nz).view(1, nz, 1, 1)
        ys = torch.arange(ny).view(1, 1, ny, 1)
        xs = torch.arange(nx).view(1, 1, 1, nx)
        q_idx = torch.arange(q).view(q, 1, 1, 1).expand(q, nz, ny, nx)
        z_idx = ((zs - c[:, 2].view(q, 1, 1, 1)) % nz).expand(q, nz, ny, nx)
        y_idx = torch.clamp(ys - c[:, 1].view(q, 1, 1, 1), 0, ny - 1).expand(q, nz, ny, nx)
        x_idx = ((xs - c[:, 0].view(q, 1, 1, 1)) % nx).expand(q, nz, ny, nx)
        up = [i for i in range(q) if int(c[i, 1]) > 0]
        down = [i for i in range(q) if int(c[i, 1]) < 0]
        _STREAM_CACHE[key] = (q_idx, z_idx, y_idx, x_idx, up, down)
    q_idx, z_idx, y_idx, x_idx, up, down = _STREAM_CACHE[key]
    out = f[q_idx, z_idx, y_idx, x_idx].clone()
    for i in up:
        out[i, :, 0, :] = f[opp[i], :, 0, :]
    for i in down:
        out[i, :, ny - 1, :] = f[opp[i], :, ny - 1, :]
    return out


def _force_tables(tau: float, gx: float, c: torch.Tensor, w: torch.Tensor, dtype):
    q = c.shape[0]
    cx = c[:, 0].view(q, 1, 1, 1).to(dtype)
    k = (1.0 - 0.5 / tau) * w.view(q, 1, 1, 1) * gx
    return cx, 3.0 * k * cx, 3.0 * k, 9.0 * k * cx


def _collide_guo(f: torch.Tensor, tau: float, gx: float, tabs) -> torch.Tensor:
    cx, a, b, d = tabs
    rho, ux, uy, uz = d3q19.macroscopic3d(f)
    ux = ux + gx / (2.0 * rho)
    feq = d3q19.equilibrium3d(rho, ux, uy, uz)
    force = a - b * ux.unsqueeze(0) + d * (cx * ux.unsqueeze(0))
    return f - (f - feq) / tau + force


def run(params: dict, *, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    dtype = torch.float64
    ny = int(params.get("ny", 16))
    nz = int(params.get("nz", 2))
    nx = int(params.get("nx", 2))
    tau = float(params.get("tau", 0.8))
    gx = float(params.get("force", 1.0e-5))
    steps = int(params.get("steps", 2000))
    init = str(params.get("init", "parabolic"))

    nu = (tau - 0.5) / 3.0
    c = d3q19._c_on(torch.device("cpu"))
    w = d3q19._w_on(torch.device("cpu"), dtype)
    opp = _opposite_table(c)
    shape = (nz, ny, nx)
    rho0 = torch.ones(shape, dtype=dtype)
    y = torch.arange(ny, dtype=dtype) + 0.5
    u_ana = gx / (2.0 * nu) * y * (float(ny) - y)
    if init == "parabolic":
        ux0 = u_ana.view(1, ny, 1).expand(*shape).contiguous()
    else:
        ux0 = torch.zeros(shape, dtype=dtype)
    zero = torch.zeros(shape, dtype=dtype)
    f = d3q19.equilibrium3d(rho0, ux0, zero, zero)

    tabs = _force_tables(tau, gx, c, w, dtype)
    history = []
    for step in range(steps):
        f = _stream_bounce_back(f, c, opp)
        f = _collide_guo(f, tau, gx, tabs)
        if (step + 1) % max(1, steps // 8) == 0:
            _, ux_t, _, _ = d3q19.macroscopic3d(f)
            history.append([step + 1, float(ux_t.abs().max())])

    rho, ux, uy, uz = d3q19.macroscopic3d(f)
    ux_phys = ux + gx / (2.0 * rho)
    prof = ux_phys[0, :, 0]
    inner = slice(1, ny - 1)
    rel_sup = float((prof[inner] - u_ana[inner]).abs().max() / u_ana[inner].max())
    return {
        "kpis": {"u_profile_rel_sup_err": rel_sup},
        "metrics": {
            "ny": ny,
            "nz": nz,
            "nx": nx,
            "tau": tau,
            "nu": nu,
            "force": gx,
            "steps": steps,
            "init": init,
            "u_max_numeric": float(prof[inner].max()),
            "u_max_analytic": float(u_ana[inner].max()),
            "u_max_rel_err": float((prof[inner].max() - u_ana[inner].max()) / u_ana[inner].max()),
            "transverse_max": float(uy.abs().max() + uz.abs().max()),
            "mass_conserved": float(rho.sum() / rho0.sum() - 1.0),
        },
        "curves": {
            "u_profile": {
                "y": y.tolist(),
                "numeric": prof.tolist(),
                "analytic": u_ana.tolist(),
            },
            "convergence": {"step": [h[0] for h in history], "u_max": [h[1] for h in history]},
        },
    }
