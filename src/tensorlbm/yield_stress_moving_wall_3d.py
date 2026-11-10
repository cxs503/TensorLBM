"""D3Q19 yield-stress LBM with **moving walls** and an exact momentum ledger.

Moving-wall treatment
---------------------
Half-way bounce-back with Ladd's velocity correction (Ladd 1994), applied on the
*post-collision* populations at the boundary fluid cells, i.e. the verified
ordering documented in ``docs/yield_stress_bounce_back_convention_20261109.md``
(restore solid cells -> reflect post-collision populations -> stream).  Here the
bounce-back is written as an *overwrite after streaming*, which is algebraically
the same scheme but needs no solid-cell state:

    f_i(x_b, t+1) = f_ci(x_b, t) + 6 w_i rho(x_b) (c_i . u_w)   for every link
                                                                x_b - c_i solid

with ``f_ci`` the post-collision population of the opposite direction and
``u_w`` the local wall velocity.  For a plane wall this is second-order accurate
(wall located half-way between the last fluid cell and the solid cell).

Choosing half-way bounce-back + Ladd (not Bouzidi) is deliberate: every wall in
this task is a *plane* wall lying exactly on a lattice face, the case where
half-way bounce-back is second order and Bouzidi's interpolation reduces to it.

Momentum ledger
---------------
For each bounced link the momentum given to the fluid is
``c_i (f_i^new - f_i^streamed)``.  Splitting the Ladd term out,

    M_wall = M_drag + M_inject
    M_drag   = sum_links c_i (f_ci - f_i^streamed)      (no-slip part, wall drag)
    M_inject = sum_links c_i * 6 w_i rho (c_i . u_w)    (moving-wall injection)

so that at steady state ``M_drag`` on all walls balances the Guo body force
(``sum rho a V``) and, for pure Couette, ``M_drag(top) = -M_drag(bottom)``.
The ledger also checks the exact discrete identity
``P(t+1) - P(t) = M_body + M_wall`` (P = momentum of the fluid cells).

Usage
-----
    from tensorlbm.yield_stress_moving_wall_3d import run_channel_3d

    res = run_channel_3d(
        H=32, ax=1e-5, u_wall=(1.9e-2, 0.0, 0.0), moving_top=True,
        tau_y=8e-5, mu_0=0.1, m=2.0e5,
    )
    res["ux_profile"], res["ledger"]["force_balance"]
"""

from __future__ import annotations

import torch

from .d3q19 import C, OPPOSITE, W_EXACT64, equilibrium3d, macroscopic3d
from .solver3d import stream3d
from .yield_stress_lbm_3d import (
    guo_force_term_3d,
    collide_yield_stress_bgk_forced_3d,
    collide_bingham_bgk_forced_3d,
    collide_hb_bgk_forced_3d,
)

__all__ = [
    "build_channel_walls",
    "build_duct_walls",
    "bounce_back_arrival_mask",
    "apply_moving_wall_bounce_back",
    "MomentumLedger",
    "run_channel_3d",
    "run_duct_3d",
    "shear_rate_profile",
]



def _collide_newtonian_guo(f: torch.Tensor, mu_0: float, ax: float) -> torch.Tensor:
    """Plain BGK + Guo body force (Newtonian limit ``tau_y = 0``)."""
    rho, ux, uy, uz = macroscopic3d(f)
    tau = torch.full_like(rho, 3.0 * mu_0 + 0.5)
    usx = ux + 0.5 * ax
    feq = equilibrium3d(rho, usx, uy, uz)
    f_post = f - (f - feq) / tau.unsqueeze(0)
    return f_post + guo_force_term_3d(f, ax, 0.0, 0.0, tau=tau,
                                      ux_star=usx, uy_star=uy, uz_star=uz)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def build_channel_walls(H: int, nx: int, nz: int, *, device=None):
    """Channel wall mask: solid layers at ``y = 0`` and ``y = H+1``.

    Periodic in ``x`` and ``z``.  Returns ``(wall, shape)`` with
    ``wall`` of shape ``(nz+0, H+2, nx)`` -> ``(nz, H+2, nx)``.
    """
    ny = H + 2
    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall[:, 0, :] = True
    wall[:, -1, :] = True
    return wall


def build_duct_walls(N: int, nx: int, *, device=None):
    """Square-duct wall mask (solid layers at ``y,z = 0`` and ``N+1``)."""
    ny = nz = N + 2
    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall[:, 0, :] = True
    wall[:, -1, :] = True
    wall[0, :, :] = True
    wall[-1, :, :] = True
    return wall


def bounce_back_arrival_mask(wall: torch.Tensor) -> torch.Tensor:
    """Mask of populations that arrive at a **fluid** cell from a solid neighbour.

    ``mask[i, z, y, x]`` is True when ``(z,y,x)`` is fluid and
    ``(z,y,x) - c_i`` is solid, i.e. the streamed population ``f_i`` at that cell
    came out of the wall and must be replaced by the bounce-back value.
    """
    c = C.to(torch.int64)
    mask = torch.zeros((19,) + tuple(wall.shape), dtype=torch.bool,
                       device=wall.device)
    shifted = wall
    for i in range(19):
        sx, sy, sz = int(c[i, 0]), int(c[i, 1]), int(c[i, 2])
        if sx == 0 and sy == 0 and sz == 0:
            continue
        nb = torch.roll(wall, shifts=(sz, sy, sx), dims=(0, 1, 2))
        mask[i] = nb & (~wall)
    return mask


# ---------------------------------------------------------------------------
# moving-wall bounce-back
# ---------------------------------------------------------------------------
def apply_moving_wall_bounce_back(
    f_stream: torch.Tensor,
    f_post: torch.Tensor,
    mask: torch.Tensor,
    *,
    u_wall: tuple[float, float, float] | None = None,
    moving_mask: torch.Tensor | None = None,
    rho: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Overwrite the wall-arriving populations (half-way BB + Ladd).

    Args:
        f_stream: populations after streaming, shape ``(19, nz, ny, nx)``.
        f_post: post-collision populations of the same step.
        mask: output of :func:`bounce_back_arrival_mask` (moving mask is ANDed in).
        u_wall: wall velocity ``(ux, uy, uz)`` (pure Ladd correction velocity).
        moving_mask: boolean ``(nz, ny, nx)``; only these wall links move.
        rho: density used in the Ladd term (default: from ``f_post``).

    Returns:
        ``(f_new, ledger_step)`` with ``ledger_step`` holding
        ``M_drag`` (``2 * sum c_i f_ci``, no-slip part), ``M_inject``
        (``sum c_i * 6 w_i rho (c_i.u_w)``) and ``M_wall = M_drag + M_inject``
        as 3-vectors: the momentum **given to the fluid** by the walls.
    """
    device = f_stream.device
    dtype = f_stream.dtype
    c = C.to(device=device, dtype=dtype)
    opp = OPPOSITE.to(device)

    reflected = f_post[opp]                       # f_ci at every cell
    if rho is None:
        rho = f_post.sum(dim=0)

    if moving_mask is None:
        moving = torch.zeros_like(mask)
    elif moving_mask.dim() == 4:                 # (19, nz, ny, nx) link mask
        moving = mask & moving_mask
    else:                                        # (nz, ny, nx) wall-cell mask
        moving = mask & moving_mask.unsqueeze(0)

    if u_wall is not None and any(float(v) != 0.0 for v in u_wall):
        w = W_EXACT64.to(device=device, dtype=dtype).view(19, 1, 1, 1)
        cu = (c[:, 0].view(19, 1, 1, 1) * float(u_wall[0])
              + c[:, 1].view(19, 1, 1, 1) * float(u_wall[1])
              + c[:, 2].view(19, 1, 1, 1) * float(u_wall[2]))
        correction = 6.0 * w * rho.unsqueeze(0) * cu
        f_bb = torch.where(moving, reflected + correction, reflected)
    else:
        correction = None
        f_bb = reflected

    f_new = torch.where(mask, f_bb, f_stream)

    # momentum bookkeeping (per step)
    def _mom(field, msk):
        """sum_links c_i * field_i over the masked links -> 3-vector"""
        return torch.einsum("i,ia->a", (field * msk.to(dtype)).sum(dim=(1, 2, 3)), c)

    # Exact exchange: the fluid loses c_opp*f_post[opp] into the solid and gains
    # c_i*f_bb[i] from it, hence  M = sum_{links} c_i (f_post[opp] + f_bb[i]).
    M_drag = 2.0 * _mom(reflected, mask)
    if correction is None:
        M_inject = torch.zeros(3, device=device, dtype=dtype)
        M_drag_moving = torch.zeros(3, device=device, dtype=dtype)
    else:
        M_inject = _mom(correction, moving)
        M_drag_moving = 2.0 * _mom(reflected, moving)
    M_wall = M_drag + M_inject
    return f_new, {"M_drag": M_drag, "M_inject": M_inject, "M_wall": M_wall,
                   "M_drag_moving": M_drag_moving}


# ---------------------------------------------------------------------------
# momentum ledger
# ---------------------------------------------------------------------------
class MomentumLedger:
    """Discrete momentum budget of a run.

    Every step the exact exchange ``M_wall`` (momentum given to the fluid by the
    walls), its no-slip part ``M_drag`` and the Ladd part ``M_inject`` are
    accumulated, together with the Guo body-force momentum ``M_body``; the
    identity ``dP = M_body + M_wall`` holds to round-off and is reported as
    ``closure_run`` / ``closure_window``.

    ``force_balance`` (criterion (d)) is evaluated over the **last steady
    window** (the final ``check_every`` steps), because the run-average of the
    wall drag also contains the start-up transient.
    """

    _KEYS = ("M_wall", "M_drag", "M_inject", "M_drag_moving", "M_body",
             "M_wall_moving", "M_wall_static")

    def __init__(self, device=None, dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        self.P0 = None
        self.P = None
        self.steps = 0
        self.run_sum = {k: 0.0 for k in self._KEYS}
        self.win_sum = {k: 0.0 for k in self._KEYS}
        self.last = None          # dict for the last completed window

    def start(self, P: torch.Tensor) -> None:
        self.P0 = P.detach().clone()
        self.P = P.detach().clone()
        self._win_P0 = P.detach().clone()

    def record(self, P: torch.Tensor, ledger_step: dict, dP_body: torch.Tensor) -> None:
        self.P = P
        # per-wall split: moving-wall links vs the rest (static walls)
        step = dict(ledger_step)
        step["M_wall_moving"] = ledger_step["M_drag_moving"] + ledger_step["M_inject"]
        step["M_wall_static"] = ledger_step["M_wall"] - step["M_wall_moving"]
        step["M_body"] = dP_body
        for k in self._KEYS:
            v = float(step[k][0].item())
            self.run_sum[k] += v
            self.win_sum[k] += v
        self.steps += 1

    def checkpoint(self) -> None:
        """Close the current window (called at every convergence check)."""
        self.last = {k: self.win_sum[k] for k in self._KEYS}
        self.last["dP"] = float((self.P - self._win_P0)[0].item())
        self.last["steps"] = self.steps
        self.last["closure"] = (
            (self.last["dP"] - self.last["M_body"]) / self.last["M_wall"]
            if abs(self.last["M_wall"]) > 1e-300 else float("nan"))
        self.last["force_balance"] = (
            self.last["M_wall"] / (-self.last["M_body"])
            if abs(self.last["M_body"]) > 1e-300 else float("nan"))
        self.last["force_balance_drag"] = (
            self.last["M_drag"] / (-self.last["M_body"])
            if abs(self.last["M_body"]) > 1e-300 else float("nan"))
        self.last["couette_balance"] = (
            abs(self.last["M_wall_static"]) / abs(self.last["M_wall_moving"])
            if abs(self.last["M_wall_moving"]) > 1e-300 else float("nan"))
        self._win_P0 = self.P.detach().clone()
        self.win_sum = {k: 0.0 for k in self._KEYS}

    def summary(self) -> dict:
        run = dict(self.run_sum)
        run["dP"] = float((self.P - self.P0)[0].item())
        run["steps"] = self.steps
        run["closure"] = (
            (run["dP"] - run["M_body"]) / run["M_wall"]
            if abs(run["M_wall"]) > 1e-300 else float("nan"))
        run["force_balance"] = (
            run["M_wall"] / (-run["M_body"])
            if abs(run["M_body"]) > 1e-300 else float("nan"))
        run["force_balance_drag"] = (
            run["M_drag"] / (-run["M_body"])
            if abs(run["M_body"]) > 1e-300 else float("nan"))
        run["couette_balance"] = (
            abs(run["M_wall_static"]) / abs(run["M_wall_moving"])
            if abs(run["M_wall_moving"]) > 1e-300 else float("nan"))
        out = {"run": run, "window": self.last,
               # flat aliases used by the acceptance report
               "closure_run": run["closure"],
               "closure_window": (self.last or {}).get("closure", float("nan")),
               "force_balance": (self.last or run)["force_balance"],
               "force_balance_run": run["force_balance"],
               "M_drag_window": (self.last or {}).get("M_drag", run["M_drag"]),
               "M_inject_window": (self.last or {}).get("M_inject", run["M_inject"]),
               "M_body_window": (self.last or {}).get("M_body", run["M_body"]),
               "M_drag_moving_window": (self.last or {}).get(
                   "M_drag_moving", run["M_drag_moving"]),
               "M_wall_window": (self.last or {}).get("M_wall", run["M_wall"]),
               "M_wall_moving_window": (self.last or {}).get(
                   "M_wall_moving", run["M_wall_moving"]),
               "M_wall_static_window": (self.last or {}).get(
                   "M_wall_static", run["M_wall_static"]),
               "force_balance_drag": (self.last or run)["force_balance_drag"],
               "couette_balance": (self.last or run)["couette_balance"]}
        return out


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def shear_rate_profile(ux_profile: torch.Tensor, dy_uniform: bool = True) -> torch.Tensor:
    """|du_x/dy| of a profile sampled at cell centres (central differences)."""
    g = torch.zeros_like(ux_profile)
    g[1:-1] = 0.5 * (ux_profile[2:] - ux_profile[:-2])
    g[0] = ux_profile[1] - ux_profile[0]
    g[-1] = ux_profile[-1] - ux_profile[-2]
    return g.abs()


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------
def _fluid_momentum(f: torch.Tensor, fluid: torch.Tensor) -> torch.Tensor:
    """Total momentum of the fluid cells (``rho u`` summed over the domain)."""
    c = C.to(device=f.device, dtype=f.dtype)
    sums = (f * fluid.unsqueeze(0).to(f.dtype)).sum(dim=(1, 2, 3))
    return torch.einsum("ia,i->a", c, sums)


def run_channel_3d(
    H: int,
    *,
    nx: int = 4,
    nz: int = 4,
    ax: float = 0.0,
    u_wall: tuple[float, float, float] = (0.0, 0.0, 0.0),
    moving_top: bool = True,
    tau_y: float = 0.0,
    mu_0: float = 0.1,
    m: float = 0.0,
    hb_K: float | None = None,
    hb_n: float | None = None,
    max_iter: int = 60000,
    check_every: int = 50,
    tol: float = 1e-10,
    nu_min: float = 1e-9,
    nu_max: float = 1e6,
    tau_min: float = 0.5005,
    tau_max: float | None = None,
    u_init: torch.Tensor | None = None,
    use_neq: bool = True,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    track_ledger: bool = True,
) -> dict:
    """Plane channel: walls at ``y=0`` (fixed) and ``y=H+1`` (optionally moving).

    Periodic in ``x`` and ``z``.  The top wall may be driven at ``u_wall`` and a
    streamwise body force ``ax`` may be applied, giving the full Bingham
    Couette--Poiseuille anchor (one wall moving + pressure gradient).
    """
    if device is None:
        device = torch.device("cpu")
    wall = build_channel_walls(H, nx, nz, device=device)
    mask = bounce_back_arrival_mask(wall)
    fluid = ~wall
    nzc, ny, nxc = wall.shape

    moving_mask = torch.zeros_like(wall)
    if moving_top:
        moving_mask[:, -1, :] = True
    # links that hit the moving wall: arrival mask AND moving wall behind them
    moving_links = torch.zeros_like(mask)
    c = C.to(torch.int64)
    for i in range(19):
        sx, sy, sz = int(c[i, 0]), int(c[i, 1]), int(c[i, 2])
        if sx == 0 and sy == 0 and sz == 0:
            continue
        nb_solid = torch.roll(moving_mask, shifts=(sz, sy, sx), dims=(0, 1, 2))
        moving_links[i] = nb_solid & fluid

    rho = torch.ones((nzc, ny, nxc), dtype=dtype, device=device)
    if u_init is None:
        ux = torch.zeros_like(rho)
    else:
        ux = u_init.to(device=device, dtype=dtype).clone()
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, uy, uz)
    feq_rest = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho),
                             torch.zeros_like(rho))

    ledger = MomentumLedger(device, dtype) if track_ledger else None
    body_dp = torch.zeros(3, device=device, dtype=dtype)
    body_dp[0] = ax * float(fluid.to(dtype).sum().item())   # rho = 1 per cell
    if ledger is not None:
        ledger.start(_fluid_momentum(f, fluid))

    prev = ux.clone()
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        if tau_y <= 0.0 and hb_K is None:
            f_post = _collide_newtonian_guo(f, mu_0, ax)
        elif hb_K is not None and hb_n is not None:
            f_post = collide_hb_bgk_forced_3d(
                f, tau_y, hb_K, hb_n, m, ax,
                nu_min=nu_min, nu_max=nu_max, tau_min=tau_min,
                tau_max=tau_max)
        else:
            f_post = collide_bingham_bgk_forced_3d(
                f, tau_y, mu_0, m, ax,
                nu_min=nu_min, nu_max=nu_max, tau_min=tau_min,
                tau_max=tau_max, use_neq=use_neq)
        f_stream = stream3d(f_post)
        f, lstep = apply_moving_wall_bounce_back(
            f_stream, f_post, mask, u_wall=u_wall, moving_mask=moving_links)
        f = torch.where(wall.unsqueeze(0), feq_rest, f)

        if ledger is not None:
            ledger.record(_fluid_momentum(f, fluid), lstep, body_dp)

        if it % check_every == 0:
            _, ux, _, _ = macroscopic3d(f)
            ux = torch.where(fluid, ux, torch.zeros_like(ux))
            scale = float(ux.abs().max().item())
            du = float((ux - prev).abs().max().item())
            prev = ux.clone()
            if ledger is not None:
                ledger.checkpoint()
            if scale > 0 and du / scale < tol:
                converged = True
                break

    rho, ux, uy, uz = macroscopic3d(f)
    ux = torch.where(fluid, ux, torch.zeros_like(ux))
    profile = ux.mean(dim=(0, 2))[1:-1]          # (H,) mean over x, z
    gamma = shear_rate_profile(profile)

    out = {
        "ux": ux, "uy": uy, "uz": uz, "rho": rho, "f": f,
        "wall": wall, "fluid": fluid, "moving_links": moving_links,
        "ux_profile": profile, "shear_rate_profile": gamma,
        "y_centres": torch.arange(1, H + 1, dtype=dtype) - 0.5,
        "iterations": it, "converged": converged,
        "H": H, "ux_field": ux,
    }
    if ledger is not None:
        out["ledger"] = ledger.summary()
    return out


def run_duct_3d(
    N: int,
    *,
    nx: int = 4,
    ax: float = 1e-5,
    tau_y: float = 0.0,
    mu_0: float = 0.1,
    m: float = 0.0,
    hb_K: float | None = None,
    hb_n: float | None = None,
    max_iter: int = 60000,
    check_every: int = 50,
    tol: float = 1e-10,
    nu_min: float = 1e-9,
    nu_max: float = 1e6,
    tau_min: float = 0.5005,
    tau_max: float | None = None,
    u_init: torch.Tensor | None = None,
    use_neq: bool = True,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
    track_ledger: bool = True,
) -> dict:
    """Square duct ``N x N``: solid walls at ``y,z = 0`` and ``N+1``, periodic x."""
    if device is None:
        device = torch.device("cpu")
    wall = build_duct_walls(N, nx, device=device)
    mask = bounce_back_arrival_mask(wall)
    fluid = ~wall
    nzc, ny, nxc = wall.shape

    rho = torch.ones((nzc, ny, nxc), dtype=dtype, device=device)
    if u_init is None:
        ux = torch.zeros_like(rho)
    else:
        ux = u_init.to(device=device, dtype=dtype).clone()
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, uy, uz)
    feq_rest = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho),
                             torch.zeros_like(rho))

    ledger = MomentumLedger(device, dtype) if track_ledger else None
    body_dp = torch.zeros(3, device=device, dtype=dtype)
    body_dp[0] = ax * float(fluid.to(dtype).sum().item())
    if ledger is not None:
        ledger.start(_fluid_momentum(f, fluid))

    prev = ux.clone()
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        if tau_y <= 0.0 and hb_K is None:
            f_post = _collide_newtonian_guo(f, mu_0, ax)
        elif hb_K is not None and hb_n is not None:
            f_post = collide_hb_bgk_forced_3d(
                f, tau_y, hb_K, hb_n, m, ax,
                nu_min=nu_min, nu_max=nu_max, tau_min=tau_min,
                tau_max=tau_max)
        else:
            f_post = collide_bingham_bgk_forced_3d(
                f, tau_y, mu_0, m, ax,
                nu_min=nu_min, nu_max=nu_max, tau_min=tau_min,
                tau_max=tau_max, use_neq=use_neq)
        f_stream = stream3d(f_post)
        f, lstep = apply_moving_wall_bounce_back(f_stream, f_post, mask)
        f = torch.where(wall.unsqueeze(0), feq_rest, f)

        if ledger is not None:
            ledger.record(_fluid_momentum(f, fluid), lstep, body_dp)

        if it % check_every == 0:
            _, ux, _, _ = macroscopic3d(f)
            ux = torch.where(fluid, ux, torch.zeros_like(ux))
            scale = float(ux.abs().max().item())
            du = float((ux - prev).abs().max().item())
            prev = ux.clone()
            if ledger is not None:
                ledger.checkpoint()
            if scale > 0 and du / scale < tol:
                converged = True
                break

    rho, ux, uy, uz = macroscopic3d(f)
    ux = torch.where(fluid, ux, torch.zeros_like(ux))
    cross = ux.mean(dim=2)[1:-1, 1:-1]          # (nz, ny) -> [iz, iy]
    out = {
        "ux": ux, "rho": rho, "f": f, "wall": wall, "fluid": fluid,
        "ux_cross": cross, "ux_cross_iyz": cross.T,   # [iy, iz] for analytic compare
        "iterations": it, "converged": converged, "N": N,
        # flow rate per unit depth (the duct is periodic in x, so divide by nx)
        "Q": float((ux * fluid.to(dtype)).sum().item()) / nx,
    }
    if ledger is not None:
        out["ledger"] = ledger.summary()
    return out