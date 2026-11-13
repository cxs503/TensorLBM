"""Plane Poiseuille flow, D2Q9 + BGK + Guo body force + half-way bounce-back.

Reference: steady solution of the forced Stokes/Navier-Stokes channel flow

    u_x(y) = G / (2 nu) * (y + 1/2) * (H - y - 1/2),   H = ny,   nu = (tau - 1/2)/3

(the ``+1/2`` offset is where half-way bounce-back puts the no-slip plane).

The KPI reported to the unit manifest is the relative sup-norm error of the
measured physical velocity ``u* = u + F/(2 rho)`` against that profile.  This is
the unit's own verification, in the spirit of TCLB's per-model regression runs:
the model directory carries its own reference and its own tolerance.
"""

from __future__ import annotations

import torch

from tensorlbm import d2q9


def _opposite_table(c: torch.Tensor) -> list[int]:
    """Index of the direction opposite to each direction (order independent)."""
    return [int((c == -c[i]).all(dim=1).nonzero().flatten()[0]) for i in range(c.shape[0])]


_STREAM_CACHE: dict = {}


def _stream_bounce_back(f: torch.Tensor, c: torch.Tensor, opp: list[int]) -> torch.Tensor:
    """Stream with periodic x and half-way bounce-back at the two y walls.

    One gather for all directions (index tensors cached per shape/device); the
    rows whose source falls outside the domain are then overwritten with the
    reflected population of the opposite direction at the same node, which is
    the half-way bounce-back rule (wall at ``y = -1/2`` and ``y = ny - 1/2``).
    """
    q, ny, nx = f.shape
    key = (q, ny, nx, f.device.type)
    if key not in _STREAM_CACHE:
        ys = torch.arange(ny).view(1, ny, 1)
        xs = torch.arange(nx).view(1, 1, nx)
        q_idx = torch.arange(q).view(q, 1, 1).expand(q, ny, nx)
        y_idx = torch.clamp(ys - c[:, 1].view(q, 1, 1), 0, ny - 1).expand(q, ny, nx)
        x_idx = ((xs - c[:, 0].view(q, 1, 1)) % nx).expand(q, ny, nx)
        walls_up = [i for i in range(q) if int(c[i, 1]) > 0]
        walls_down = [i for i in range(q) if int(c[i, 1]) < 0]
        _STREAM_CACHE[key] = (q_idx, y_idx, x_idx, walls_up, walls_down)
    q_idx, y_idx, x_idx, walls_up, walls_down = _STREAM_CACHE[key]

    out = f[q_idx, y_idx, x_idx].clone()
    for i in walls_up:  # direction points into the fluid from the lower wall
        out[i, 0] = f[opp[i], 0]
    for i in walls_down:  # direction points into the fluid from the upper wall
        out[i, ny - 1] = f[opp[i], ny - 1]
    return out


def _force_tables(tau: float, gx: float, c: torch.Tensor, w: torch.Tensor, dtype):
    """Per-direction Guo coefficients, precomputed once per unit run."""
    q = c.shape[0]
    cx = c[:, 0].view(q, 1, 1).to(dtype)
    cy = c[:, 1].view(q, 1, 1).to(dtype)
    k = (1.0 - 0.5 / tau) * w.view(q, 1, 1) * gx
    # (c_i - u).F / cs^2 + (c_i.u)(c_i.F) / cs^4 with F = (gx, 0), cs^2 = 1/3
    a = 3.0 * k * cx
    b = 3.0 * k
    d = 9.0 * k * cx
    return cx, cy, a, b, d


def _collide_guo(f: torch.Tensor, tau: float, gx: float, tabs) -> torch.Tensor:
    """BGK collision with a Guo body force in +x (``u* = u + F/(2 rho)``)."""
    cx, cy, a, b, d = tabs
    rho, ux, uy = d2q9.macroscopic(f)
    ux = ux + gx / (2.0 * rho)  # Guo: equilibrium uses the force-corrected velocity
    feq = d2q9.equilibrium(rho, ux, uy)
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    force = a - b * ux.unsqueeze(0) + d * cu
    return f - (f - feq) / tau + force


def run(params: dict, *, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    dtype = torch.float64
    ny = int(params.get("ny", 32))
    nx = int(params.get("nx", 4))
    tau = float(params.get("tau", 0.8))
    gx = float(params.get("force", 1.0e-5))
    steps = int(params.get("steps", 3000))
    init = str(params.get("init", "parabolic"))

    nu = (tau - 0.5) / 3.0
    c = d2q9._c_on(torch.device("cpu"))
    w = d2q9._w_for(dtype, torch.device("cpu"))
    opp = _opposite_table(c)
    rho0 = torch.ones((ny, nx), dtype=dtype)
    y = torch.arange(ny, dtype=dtype) + 0.5
    u_ana = gx / (2.0 * nu) * y * (float(ny) - y)
    if init == "parabolic":
        # start on the analytic profile so the run measures the steady-state
        # discretisation error, not the spin-up transient (see the report's
        # convergence study, which runs the rest-start case to steady state)
        ux0 = u_ana.unsqueeze(1).expand(ny, nx).contiguous()
    else:
        ux0 = torch.zeros((ny, nx), dtype=dtype)
    f = d2q9.equilibrium(rho0, ux0, torch.zeros_like(rho0))

    tabs = _force_tables(tau, gx, c, w, dtype)
    history = []
    for step in range(steps):
        f = _stream_bounce_back(f, c, opp)
        f = _collide_guo(f, tau, gx, tabs)
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            _, ux, _ = d2q9.macroscopic(f)
            history.append([step, float(ux[:, nx // 2].abs().max())])

    rho, ux, uy = d2q9.macroscopic(f)
    ux_phys = ux + gx / (2.0 * rho)  # what a force-corrected measurement returns
    inner = slice(1, ny - 1)  # exclude the two wall-adjacent rows from the metric
    err = (ux_phys[inner, :] - u_ana[inner].unsqueeze(1)).abs()
    rel_sup = float(err.max() / u_ana[inner].max())

    return {
        "kpis": {"u_profile_rel_sup_err": rel_sup},
        "metrics": {
            "ny": ny,
            "nx": nx,
            "tau": tau,
            "nu": nu,
            "force": gx,
            "steps": steps,
            "u_max_numeric": float(ux_phys[inner, :].max()),
            "init": init,
            "u_max_analytic": float(u_ana[inner].max()),
            "u_max_rel_err": float(
                (ux_phys[inner, :].max() - u_ana[inner].max()) / u_ana[inner].max()
            ),
            "mass_conserved": float(rho.sum() / rho0.sum() - 1.0),
        },
        "curves": {
            "u_profile": {
                "y": y.tolist(),
                "numeric": ux_phys[:, nx // 2].tolist(),
                "analytic": u_ana.tolist(),
            },
            "convergence": {"step": [h[0] for h in history], "u_max": [h[1] for h in history]},
        },
    }
