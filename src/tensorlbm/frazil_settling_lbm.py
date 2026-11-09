"""Lattice-Boltzmann settling column and stratified slurry channel (D2Q5/D2Q9).

This module is the *numerical* half of the frazil-suspension closure in
:mod:`tensorlbm.frazil_suspension`:

* :func:`run_advection_diffusion_check` -- scalar (D2Q5) advection-diffusion
  against the exact damped travelling wave of the advection-diffusion equation;
* :func:`run_settling_column` -- the Kynch / Richardson-Zaki column: the
  concentration field is advected by the local hindered settling velocity
  ``v_s(c)`` and diffused by ``D_eff``; the falling interface must move at the
  analytically known shock speed ``v_s(c0)``;
* :func:`stratified_powerlaw_reference` -- the exact solution (by quadrature) of
  the variable-viscosity power-law momentum balance
  ``d/dy [ K(y) (du/dy)^{n(y)} ] = -a`` for a wall-concentrated frazil layer;
* :func:`run_powerlaw_channel` -- the matching D2Q9 variable-viscosity BGK
  channel (implicit non-equilibrium-stress closure, Guo forcing, half-way
  bounce-back), either uniform or stratified.

Equations
---------
Concentration field (D2Q5, BGK, second-order in ``u``)::

    dc/dt + div(c u_eff) = div(D_eff grad c) + O(u^3),
    u_eff = u_fluid - v_s(c) z_hat

The settling is therefore a *local, concentration-dependent drift* in the
equilibrium distribution; because the drift enters inside the flux divergence,
the scheme transports the conservative Kynch equation rather than a
non-conservative advection form.  The domain of the column is periodic so mass
is conserved to round-off and the measured interface motion is uncontaminated by
boundary fluxes.

Momentum field (D2Q9): the local relaxation time solves, per cell,

    nu = K(y) ( B / (1/2 + 3 nu) )^{n(y)-1},   B = 3||Pi^neq||/(sqrt2 rho),

by Newton iteration (unique root for ``n <= 1``), i.e. the LBM evaluates the
power law at the stress it actually transports -- no finite-difference stencil
for the shear rate, which is what keeps the stratified profile second-order
accurate.  The body force is the Guo (2002) discrete force with per-cell tau.

Lattice units: the closure is written in whatever unit system the caller uses;
``gamma_dot`` is the lattice shear rate, so ``K`` and ``nu`` are lattice numbers.
The physical reading of a lattice run (``nu_lat = mu_r * nu_water_lat`` etc.) is
documented in the artifact produced by ``benchmarks/frazil_suspension_bench.py``.
"""

from __future__ import annotations

import math

import torch

from .d2q9 import OPPOSITE, equilibrium, macroscopic
from .d2q9 import C as C9
from .d2q9 import W as W9
from .frazil_lbm import neq_stress_2d
from .frazil_rheology import FrazilRheologyError
from .frazil_suspension import hindered_settling_velocity
from .passive_scalar import (
    collide_scalar_bgk,
    equilibrium_scalar,
    macroscopic_scalar,
    stream_scalar,
)
from .powerlaw import guo_force_term
from .solver import stream

__all__ = [
    "powerlaw_tau_field",
    "stratified_powerlaw_reference",
    "run_powerlaw_channel",
    "run_advection_diffusion_check",
    "run_settling_column",
]


# ---------------------------------------------------------------------------
# Variable-viscosity power-law collision (per-cell K, n)
# ---------------------------------------------------------------------------


def powerlaw_tau_field(
    f: torch.Tensor,
    consistency_index,
    flow_index,
    *,
    tau_min: float = 0.5001,
    tau_max: float | None = None,
    nu_min: float = 1e-9,
    nu_max: float = 1e4,
    newton_iters: int = 20,
) -> torch.Tensor:
    """Relaxation-time field from the LBM's own non-equilibrium stress.

    Generalisation of :func:`tensorlbm.frazil_lbm.powerlaw_relaxation_time_neq`
    to **per-cell** ``K`` and ``n`` (the stratified case): the scalar equation

        ``nu - K (B / (1/2 + 3 nu))^{n-1} = 0``

    is solved per cell by Newton iteration.  For ``n <= 1`` the residual is
    strictly increasing in ``nu`` so the root is unique; ``n = 1`` converges in
    one step (the Newtonian limit).

    Args:
        f: Distribution tensor ``(9, ny, nx)``.
        consistency_index: Scalar or ``(ny, nx)`` consistency index ``K``.
        flow_index: Scalar or ``(ny, nx)`` power-law index ``n`` in ``(0, 1]``.
        tau_min: Lower bound on ``tau`` (> 0.5).
        tau_max: Optional upper bound on ``tau``.
        nu_min, nu_max: Kinematic-viscosity clamps (lattice units).
        newton_iters: Newton iterations (20 is far beyond convergence).

    Returns:
        ``tau`` field of shape ``(ny, nx)`` clipped to the requested bounds.

    Raises:
        FrazilRheologyError: on a non-positive ``K`` or ``n`` outside ``(0, 1]``.
    """
    rho, _, _, pi_xx, pi_xy, pi_yy = neq_stress_2d(f)
    shape = rho.shape
    K = torch.as_tensor(consistency_index, dtype=f.dtype, device=f.device)
    n = torch.as_tensor(flow_index, dtype=f.dtype, device=f.device)
    if K.ndim == 0:
        K = K.expand(shape)
    if n.ndim == 0:
        n = n.expand(shape)
    if bool((K <= 0).any()):
        raise FrazilRheologyError("consistency_index must be > 0 everywhere")
    if bool(((n <= 0) | (n > 1.0 + 1e-12)).any()):
        raise FrazilRheologyError("flow_index must be in (0, 1] everywhere")

    pi_norm = torch.sqrt(
        torch.clamp(pi_xx * pi_xx + 2.0 * pi_xy * pi_xy + pi_yy * pi_yy, min=0.0)
    )
    B = torch.clamp(3.0 * pi_norm / (math.sqrt(2.0) * rho), min=1e-30)
    nu = torch.full_like(B, float(K.max()))
    for _ in range(int(newton_iters)):
        tau = 0.5 + 3.0 * nu
        g = K * torch.pow(B, n - 1.0) * torch.pow(tau, 1.0 - n)
        dg = 3.0 * (1.0 - n) * g / tau
        nu = nu - (nu - g) / (1.0 - dg)
        nu = torch.clamp(nu, min=nu_min, max=nu_max)
    tau = 0.5 + 3.0 * nu
    tau = tau.clamp(min=tau_min)
    if tau_max is not None:
        tau = tau.clamp(max=tau_max)
    return tau


def _channel_step(
    f: torch.Tensor,
    K: torch.Tensor,
    n: torch.Tensor,
    ax: float,
    *,
    tau_min: float,
    tau_max,
    nu_min: float,
    nu_max: float,
    newton_iters: int,
) -> torch.Tensor:
    """One fused collide + Guo-force step of the variable-viscosity channel.

    Equivalent to :func:`powerlaw_tau_field` followed by
    :func:`_collide_powerlaw_field`, but the macroscopic moment, the
    equilibrium and the force-corrected velocity are computed once instead of
    four times.  This is the inner loop of every channel verification run, so
    the saving matters.
    """
    rho, ux, uy = macroscopic(f)
    ux_star = ux + 0.5 * ax
    feq = equilibrium(rho, ux_star, uy)
    fneq = f - feq
    c = C9.to(device=f.device, dtype=f.dtype)
    w = W9.to(device=f.device, dtype=f.dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    pi_xx = (cx * cx * fneq).sum(0)
    pi_xy = (cx * cy * fneq).sum(0)
    pi_yy = (cy * cy * fneq).sum(0)
    pi_norm = torch.sqrt(
        torch.clamp(pi_xx * pi_xx + 2.0 * pi_xy * pi_xy + pi_yy * pi_yy, min=0.0)
    )
    B = torch.clamp(3.0 * pi_norm / (math.sqrt(2.0) * rho), min=1e-30)
    KB = K * torch.pow(B, n - 1.0)
    nu = torch.full_like(B, nu_max)
    nu = torch.clamp(KB * torch.pow(0.5 + 3.0 * nu, 1.0 - n), min=nu_min, max=nu_max)
    for _ in range(int(newton_iters)):
        tau = 0.5 + 3.0 * nu
        g = KB * torch.pow(tau, 1.0 - n)
        dg = 3.0 * (1.0 - n) * g / tau
        nu = torch.clamp(nu - (nu - g) / (1.0 - dg), min=nu_min, max=nu_max)
    tau = (0.5 + 3.0 * nu).clamp(min=tau_min)
    if tau_max is not None:
        tau = tau.clamp(max=tau_max)
    cu = 3.0 * (cx * ux_star + cy * uy)
    force = (
        w.view(9, 1, 1)
        * rho.unsqueeze(0)
        * ax
        * (1.0 - 0.5 / tau).unsqueeze(0)
        * (3.0 * (cx - ux_star.unsqueeze(0)) + 9.0 * cx * cu)
    )
    return f - (f - feq) / tau.unsqueeze(0) + force


def _collide_powerlaw_field(
    f: torch.Tensor,
    ax: float,
    tau_field: torch.Tensor,
) -> torch.Tensor:
    """BGK collision at a prescribed ``tau`` field plus the Guo body force."""
    rho, ux, uy = macroscopic(f)
    ux_star = ux + 0.5 * ax
    feq = equilibrium(rho, ux_star, uy)
    fi = guo_force_term(f, ax, tau_field, ux_star=ux_star)
    return f - (f - feq) / tau_field.unsqueeze(0) + fi


def _wall_mask(H: int, nx: int, dtype, device) -> torch.Tensor:
    wall = torch.zeros((H + 2, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True
    return wall


def _bounce_back(f_post: torch.Tensor, f_pre: torch.Tensor, wall: torch.Tensor) -> torch.Tensor:
    """Half-way no-slip bounce-back on the wall rows (second-order accurate)."""
    opp = OPPOSITE.to(f_pre.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f_post)


def _stream_scalar_closed(g: torch.Tensor) -> torch.Tensor:
    """D2Q5 streaming in a closed (no-flux) column, periodic in ``x``.

    The grid holds ``H`` fluid rows ``1..H`` between two wall rows ``0`` and
    ``H+1`` (the momentum convention of this repository).  Incoming
    wall-normal populations are half-way bounced back onto the wall-adjacent
    fluid rows, which makes the wall flux *exactly* zero and therefore conserves
    the column's mass to round-off: for a batch settling test that is what keeps
    the Kynch front measurement free of boundary-flux bias.
    """
    ny = g.shape[1]
    out = torch.empty_like(g)
    out[0] = g[0]
    out[1] = torch.roll(g[1], 1, dims=1)  # +x, periodic
    out[3] = torch.roll(g[3], -1, dims=1)  # -x, periodic
    out[2] = 0.0
    out[4] = 0.0
    out[2, 2 : ny - 1, :] = g[2, 1 : ny - 2, :]  # +y interior
    out[4, 1 : ny - 2, :] = g[4, 2 : ny - 1, :]  # -y interior
    out[2, 1, :] = g[4, 1, :]  # bounce at the lower wall
    out[4, ny - 2, :] = g[2, ny - 2, :]  # bounce at the upper wall
    return out


# ---------------------------------------------------------------------------
# Stratified channel: exact reference by quadrature
# ---------------------------------------------------------------------------


def _interp_profile(y_fine: torch.Tensor, profile: torch.Tensor, H: int) -> torch.Tensor:
    """Piecewise-linear interpolation of a cell-centred profile onto ``y_fine``.

    Cell centres sit at ``y = 1 .. H``; the returned value is constant
    (extrapolated) outside that range, i.e. the wall-adjacent value is used at
    the wall plane.
    """
    pos = (y_fine - 1.0).clamp(min=0.0, max=float(H - 1) - 1e-12)
    i0 = torch.floor(pos).to(torch.int64).clamp(0, H - 2)
    frac = pos - i0.to(y_fine.dtype)
    return profile[i0] * (1.0 - frac) + profile[i0 + 1] * frac


def stratified_powerlaw_reference(
    consistency_profile,
    flow_profile,
    ax: float,
    H: int,
    *,
    sub_per_cell: int = 200,
) -> dict:
    """Exact steady power-law channel with wall-normal ``K(y)``, ``n(y)``.

    For fully developed flow the shear stress is fixed by momentum balance,
    ``tau_xy(y) = ax (y - y_c)`` (``rho = 1``), independent of the rheology, so

        ``du/dy = -sign(y - y_c) ( |ax (y - y_c)| / K(y) )^{1/n(y)}``

    and ``u`` follows by quadrature from the lower wall ``y = 1/2``.  The result
    satisfies ``u(1/2) = 0``; the upper-wall condition ``u(H + 1/2) = 0`` holds
    exactly when ``K`` and ``n`` are symmetric about the centreline, which is the
    case used here (a wall-concentrated frazil layer).  Quadrature on
    ``sub_per_cell`` sub-intervals per lattice cell converges to ~1e-7, i.e. far
    below any LBM discretisation error.

    Args:
        consistency_profile: ``K`` on the ``H`` fluid rows (cell centred).
        flow_profile: ``n`` on the ``H`` fluid rows.
        ax: Body-force acceleration (+x).
        H: Channel height in cells.
        sub_per_cell: Fine-grid points per cell.

    Returns:
        dict with ``u`` (cell centres), ``y``, ``u_max``, ``u_top_wall`` (should
        be ~0 for a symmetric stratification), ``flow_rate`` and the profiles
        used.
    """
    K = torch.as_tensor(consistency_profile, dtype=torch.float64).reshape(-1)
    n = torch.as_tensor(flow_profile, dtype=torch.float64).reshape(-1)
    if K.numel() != H or n.numel() != H:
        raise FrazilRheologyError(
            f"profiles must have H = {H} entries, got {K.numel()} and {n.numel()}"
        )
    if bool((K <= 0).any()):
        raise FrazilRheologyError("consistency profile must be > 0")
    yc = 0.5 * float(H + 1)
    npts = int(H * sub_per_cell) + 1
    y_fine = torch.linspace(0.5, H + 0.5, npts, dtype=torch.float64)
    K_fine = _interp_profile(y_fine, K, H)
    n_fine = _interp_profile(y_fine, n, H)
    dy = float(y_fine[1] - y_fine[0])
    sgn = torch.sign(y_fine - yc)
    g = torch.pow(torch.abs(ax * (y_fine - yc)) / K_fine, 1.0 / n_fine)
    # momentum balance: d tau_xy/dy = -a  =>  tau_xy = -a (y - y_c), and
    # du/dy = tau_xy / (K |du/dy|^(n-1)) = -sign(y - y_c) (|tau|/K)^(1/n).
    integrand = -sgn * g
    # cumulative trapezoid from the lower wall, then shift so u(y=0.5) = 0
    cum = torch.zeros_like(y_fine)
    cum[1:] = torch.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dy, dim=0)
    u_fine = cum - cum[0]
    # sample the fine grid at the cell centres y = 1..H
    idx = torch.round((torch.arange(1, H + 1, dtype=torch.float64) - 0.5) / dy).to(torch.int64)
    u_cells = u_fine[idx]
    return {
        "u": u_cells,
        "y": torch.arange(1, H + 1, dtype=torch.float64),
        "u_max": float(u_cells.max()),
        "u_top_wall": float(u_fine[-1]),
        "flow_rate": float(u_cells.mean()),
        "u_fine": u_fine,
        "y_fine": y_fine,
    }


# ---------------------------------------------------------------------------
# D2Q9 variable-viscosity channel driver
# ---------------------------------------------------------------------------


def run_powerlaw_channel(
    *,
    consistency_index,
    flow_index,
    ax: float = 1.0e-3,
    H: int = 24,
    nx: int = 4,
    tau_min: float = 0.5001,
    tau_max: float | None = None,
    nu_min: float = 1e-9,
    nu_max: float = 1e4,
    max_steps: int = 60000,
    check_every: int = 100,
    window: int = 20,
    tol: float = 1e-6,
    avg_steps: int = 300,
    newton_iters: int = 8,
    initial_profile=None,
    dtype=torch.float64,
    device: str | torch.device = "cpu",
) -> dict:
    """Steady channel of a power-law slurry with a wall-normal viscosity profile.

    Args:
        consistency_index: Scalar ``K`` or length-``H`` profile (cell centred).
        flow_index: Scalar ``n`` or length-``H`` profile.
        ax: Uniform body force (+x).
        H, nx: Channel height (cells) and streamwise width (periodic).
        tau_min, tau_max, nu_min, nu_max: Solver clamps.
        max_steps, check_every, window, tol: Steady-state detection on the
            centreline velocity (peak-to-peak spread < ``tol * mean``).
        avg_steps: Number of further steps the returned profile is averaged
            over (suppresses the acoustic residual).
        initial_profile: Optional length-``H`` velocity profile used to
            initialise the run (e.g. the analytic solution).  Only the steady
            state is verified, so initialising from the target removes the
            viscous start-up transient (``~ H^2/nu`` steps) without weakening
            the test.

    Returns:
        dict with ``profile`` (x- and time-averaged ``u`` on the ``H`` fluid
        rows), ``y``, ``tau_max``, ``tau_min``, ``steps``, ``converged``,
        ``drift``, ``profile_2d`` and ``nu_wall``.
    """
    dev = torch.device(device)
    wall = _wall_mask(H, nx, dtype, dev)
    ny = H + 2
    K = torch.as_tensor(consistency_index, dtype=dtype, device=dev)
    n = torch.as_tensor(flow_index, dtype=dtype, device=dev)
    if K.ndim == 1:
        if K.numel() != H:
            raise FrazilRheologyError(f"K profile must have {H} entries, got {K.numel()}")
        K2d = torch.zeros((ny, nx), dtype=dtype, device=dev)
        K2d[1:-1, :] = K.view(H, 1)
        K2d[0, :] = K[0]
        K2d[-1, :] = K[-1]
        K = K2d
    else:
        K = K.expand((ny, nx)).contiguous()
    if n.ndim == 1:
        if n.numel() != H:
            raise FrazilRheologyError(f"n profile must have {H} entries, got {n.numel()}")
        n2d = torch.zeros((ny, nx), dtype=dtype, device=dev)
        n2d[1:-1, :] = n.view(H, 1)
        n2d[0, :] = n[0]
        n2d[-1, :] = n[-1]
        n = n2d
    else:
        n = n.expand((ny, nx)).contiguous()

    u_init = torch.zeros((ny, nx), dtype=dtype, device=dev)
    if initial_profile is not None:
        prof0 = torch.as_tensor(initial_profile, dtype=dtype, device=dev).reshape(-1)
        if prof0.numel() != H:
            raise FrazilRheologyError(
                f"initial_profile must have {H} entries, got {prof0.numel()}"
            )
        u_init[1:-1, :] = prof0.view(H, 1)
    f = equilibrium(
        torch.ones((ny, nx), dtype=dtype, device=dev),
        u_init,
        torch.zeros((ny, nx), dtype=dtype, device=dev),
    )
    def advance(field: torch.Tensor) -> torch.Tensor:
        collided = _channel_step(
            field,
            K,
            n,
            ax,
            tau_min=tau_min,
            tau_max=tau_max,
            nu_min=nu_min,
            nu_max=nu_max,
            newton_iters=newton_iters,
        )
        return stream(_bounce_back(collided, field, wall))

    samples: list[float] = []
    converged = False
    step = 0
    for step in range(1, int(max_steps) + 1):
        f = advance(f)
        if step % check_every == 0:
            _, ux_c, _ = macroscopic(f)
            samples.append(float(ux_c[:, nx // 2].mean()))
            if len(samples) >= window and step >= 20 * check_every:
                recent = samples[-window:]
                mean = sum(recent) / len(recent)
                if (max(recent) - min(recent)) / max(abs(mean), 1e-30) < tol:
                    converged = True
                    break

    prof = torch.zeros(H, dtype=dtype, device=dev)
    for _ in range(int(avg_steps)):
        f = advance(f)
        _, ux_a, _ = macroscopic(f)
        prof += ux_a[1:-1, :].mean(dim=1)
    prof = prof / float(avg_steps)
    _, ux_f, _ = macroscopic(f)
    recent = samples[-window:] if len(samples) >= window else samples
    drift = None
    if recent:
        mean = sum(recent) / len(recent)
        drift = (max(recent) - min(recent)) / max(abs(mean), 1e-30)
    tau_end = powerlaw_tau_field(
        f,
        K,
        n,
        tau_min=tau_min,
        tau_max=tau_max,
        nu_min=nu_min,
        nu_max=nu_max,
        newton_iters=newton_iters,
    )
    _, ux_f, _ = macroscopic(f)
    return {
        "profile": prof.cpu(),
        "y": torch.arange(1, H + 1, dtype=dtype),
        "profile_2d": ux_f.cpu(),
        "tau_max": float(tau_end.max()),
        "tau_min": float(tau_end.min()),
        "nu_wall": 0.5 * (float(tau_end[1, 0]) - 0.5) / 3.0,
        "steps": step,
        "converged": converged,
        "drift": drift,
        "H": H,
        "nx": nx,
        "ax": ax,
    }


def _profile_error(u_num: torch.Tensor, u_exact: torch.Tensor) -> dict:
    """L2-relative and peak-relative profile errors."""
    u_num = torch.as_tensor(u_num, dtype=torch.float64)
    u_exact = torch.as_tensor(u_exact, dtype=torch.float64)
    denom = float(torch.linalg.vector_norm(u_exact))
    l2 = float(torch.linalg.vector_norm(u_num - u_exact)) / denom
    i_max = int(torch.argmax(u_exact))
    peak = abs(float(u_num[i_max] - u_exact[i_max])) / abs(float(u_exact[i_max]))
    return {
        "l2_relative": l2,
        "peak_relative": peak,
        "u_max_numeric": float(u_num.max()),
        "u_max_exact": float(u_exact.max()),
    }


# ---------------------------------------------------------------------------
# Scalar advection-diffusion verification
# ---------------------------------------------------------------------------


def run_advection_diffusion_check(
    *,
    u: float = 0.05,
    D: float = 0.01,
    nx: int = 64,
    ny: int = 4,
    steps: int = 400,
    mode: int = 1,
    dtype=torch.float64,
    device: str | torch.device = "cpu",
) -> dict:
    """D2Q5 advection-diffusion against the exact damped travelling wave.

    Initial condition ``c(x,0) = 1/2 [1 + cos(k x)]`` with ``k = 2 pi mode / nx``
    on a periodic domain.  The exact solution of
    ``dc/dt + u dc/dx = D d^2c/dx^2`` is

        ``c(x,t) = 1/2 + 1/2 cos(k (x - u t)) exp(-D k^2 t)``

    so a single run measures three things at once: the L2 shape error, the
    numerical advection speed (from the phase) and the *numerical* diffusivity
    (from the amplitude decay ``D_num = ln(A_0/A)/(k^2 t)``).

    Returns:
        dict with ``l2_relative``, ``advection_speed_numeric``,
        ``advection_speed_error``, ``diffusivity_numeric``,
        ``diffusivity_relative_error``, ``amplitude_ratio`` and the profiles.
    """
    dev = torch.device(device)
    if D <= 0.0:
        raise FrazilRheologyError(f"diffusivity must be > 0, got {D!r}")
    tau_d = 3.0 * D + 0.5
    k = 2.0 * math.pi * float(mode) / float(nx)
    x = torch.arange(nx, dtype=dtype, device=dev) + 0.5
    c0 = 0.5 * (1.0 + torch.cos(k * x)).view(1, nx).expand(ny, nx).contiguous()
    ux = torch.full((ny, nx), float(u), dtype=dtype, device=dev)
    uy = torch.zeros((ny, nx), dtype=dtype, device=dev)
    g = equilibrium_scalar(c0, ux, uy)
    for _ in range(int(steps)):
        c = macroscopic_scalar(g)
        g = collide_scalar_bgk(g, c, ux, uy, tau_d)
        g = stream_scalar(g)
    c_num = macroscopic_scalar(g).mean(dim=0)
    t = float(steps)
    exact = 0.5 + 0.5 * torch.cos(k * (x - u * t)) * math.exp(-D * k * k * t)
    l2 = float(torch.linalg.vector_norm(c_num - exact)) / float(torch.linalg.vector_norm(exact))
    # c = 1/2 + 1/2 a cos(k(x - u t)); the mean-free mode projection has
    # amplitude a/4 per unit of 2cos, hence the factor 4 below.
    A = 4.0 * float((c_num * torch.cos(k * x)).mean())
    B = 4.0 * float((c_num * torch.sin(k * x)).mean())
    amp = math.hypot(A, B)
    phase = math.atan2(B, A) % (2.0 * math.pi)
    u_num = phase / (k * t)
    d_num = math.log(1.0 / max(amp, 1e-300)) / (k * k * t)
    return {
        "l2_relative": l2,
        "advection_speed_numeric": u_num,
        "advection_speed_error": abs(u_num - u) / abs(u),
        "diffusivity_numeric": d_num,
        "diffusivity_relative_error": abs(d_num - D) / D,
        "amplitude_ratio": amp,
        "amplitude_ratio_exact": math.exp(-D * k * k * t),
        "x": x.cpu(),
        "c_numeric": c_num.cpu(),
        "c_exact": exact.cpu(),
        "steps": int(steps),
        "u": u,
        "D": D,
        "k": k,
        "tau_d": tau_d,
    }


# ---------------------------------------------------------------------------
# Settling column (Kynch / Richardson-Zaki)
# ---------------------------------------------------------------------------


def _interface_height(profile: torch.Tensor, level: float) -> float:
    """Height of the ``c = level`` crossing of a monotone column profile.

    ``profile`` is the x-averaged concentration indexed by height (index 0 at the
    bottom).  Returns the linearly interpolated height where ``c`` crosses
    ``level`` between the highest cell above the level and the next one up.
    """
    above = profile > level
    if not bool(above.any()):
        return float("nan")
    i_top = int(torch.nonzero(above, as_tuple=False)[-1, 0])
    if i_top >= profile.numel() - 1:
        return float(i_top) + 0.5
    c_lo = float(profile[i_top])
    c_hi = float(profile[i_top + 1])
    if c_lo == c_hi:
        return float(i_top) + 0.5
    frac = (c_lo - level) / (c_lo - c_hi)
    return float(i_top) + 0.5 + frac


def run_settling_column(
    concentration_0: float = 0.2,
    *,
    v_s0: float = 0.08,
    exponent: float = 3.55,
    phi_max: float = 0.60,
    D: float = 1.0e-3,
    H: int = 96,
    nx: int = 2,
    init_height_fraction: float = 0.75,
    max_steps: int = 20000,
    check_every: int = 10,
    travel_cells: float = 6.0,
    record_snapshots: int = 3,
    dtype=torch.float64,
    device: str | torch.device = "cpu",
) -> dict:
    """Batch settling column of a frazil suspension (Richardson-Zaki + diffusion).

    A closed column of height ``H`` cells starts homogenised at ``c0`` up to
    ``init_height_fraction * H`` with clear water above.  The scalar field is
    advected by the local drift ``v_y = -v_s(c)`` (RZ law) and diffused with
    ``D``.  The walls are no-flux (half-way bounce-back), so the column mass is
    conserved to round-off and no boundary flux can bias the front.

    For a RZ flux ``f(c) = c v_s(c)`` the concentration step is a Kynch shock
    whose speed is exactly ``v_s(c0)``; the measured interface descent is
    therefore an analytic verification target.

    Args:
        concentration_0: Initial (homogeneous) ice volume fraction.
        v_s0: Single-particle settling velocity in lattice units.
        exponent: RZ hindrance exponent ``m``.
        phi_max: Packing limit.
        D: Effective diffusivity in lattice units.
        H, nx: Column height / width in cells.
        init_height_fraction: Initial slurry depth as a fraction of ``H``.
        max_steps, check_every: Iteration budget and interface-sampling period.
        travel_cells: Interface descent that terminates the measurement.
        record_snapshots: Number of x-averaged profiles kept for the figures.

    Returns:
        dict with ``interface_velocity`` (least-squares fit of ``h(t)``),
        ``interface_velocity_expected`` (``v_s(c0)``), ``relative_error``,
        ``mass_drift``, ``times``, ``heights``, ``profile_final``, ``snapshots``,
        ``steps`` and the run parameters.

    Raises:
        FrazilRheologyError: on ``D <= 0`` or ``c0`` outside ``[0, phi_max)``.
    """

    dev = torch.device(device)
    if D <= 0.0:
        raise FrazilRheologyError(f"diffusivity must be > 0, got {D!r}")
    if float(concentration_0) < 0.0 or float(concentration_0) >= phi_max:
        raise FrazilRheologyError(
            f"concentration_0 must be in [0, {phi_max}), got {concentration_0!r}"
        )
    tau_d = 3.0 * D + 0.5
    c0 = float(concentration_0)
    v_expected = float(
        hindered_settling_velocity(c0, v_s0, exponent=exponent, phi_max=phi_max)
    )
    ny = H + 2
    y = torch.arange(1, H + 1, dtype=dtype, device=dev) + 0.5  # fluid-row centres
    c_init = torch.where(
        y < init_height_fraction * H,
        torch.full_like(y, c0),
        torch.zeros_like(y),
    )
    c2d = torch.zeros((ny, nx), dtype=dtype, device=dev)
    c2d[1:-1, :] = c_init.view(H, 1)
    zero = torch.zeros((ny, nx), dtype=dtype, device=dev)
    g = equilibrium_scalar(c2d, zero, zero)
    mass0 = float(macroscopic_scalar(g).sum())

    times: list[float] = []
    heights: list[float] = []
    snaps: list[tuple[float, torch.Tensor]] = []
    level = 0.5 * c0
    h0 = _interface_height(c_init, level)
    travel = 0.0
    step = 0
    next_snap = 0
    for step in range(1, int(max_steps) + 1):
        c = macroscopic_scalar(g)
        vy = -torch.as_tensor(
            hindered_settling_velocity(
                (c[1:-1, :]).clamp(min=0.0, max=phi_max * (1.0 - 1e-12)),
                v_s0,
                exponent=exponent,
                phi_max=phi_max,
            ),
            dtype=dtype,
            device=dev,
        )
        vy_full = torch.zeros((ny, nx), dtype=dtype, device=dev)
        vy_full[1:-1, :] = vy
        g = collide_scalar_bgk(g, c, zero, vy_full, tau_d)
        g = _stream_scalar_closed(g)
        if step % int(check_every) == 0:
            prof = macroscopic_scalar(g)[1:-1, :].mean(dim=1)
            h = _interface_height(prof, level)
            if math.isnan(h):
                break
            times.append(float(step))
            heights.append(h)
            travel = h0 - h
            if record_snapshots > 0 and next_snap < record_snapshots:
                target = h0 - travel_cells * (next_snap + 1) / float(record_snapshots)
                if h <= target:
                    snaps.append((float(step), prof.detach().cpu()))
                    next_snap += 1
            if travel >= travel_cells:
                break
    c_final = macroscopic_scalar(g)[1:-1, :].mean(dim=1)
    if not snaps:
        snaps.append((float(step), c_final.detach().cpu()))
    mass1 = float(macroscopic_scalar(g)[1:-1, :].sum())
    n_pts = len(times)
    v_meas = float("nan")
    if n_pts >= 3:
        tt = torch.as_tensor(times, dtype=torch.float64)
        hh = torch.as_tensor(heights, dtype=torch.float64)
        tm = tt.mean()
        hm = hh.mean()
        v_meas = -float(((tt - tm) * (hh - hm)).sum() / ((tt - tm).pow(2).sum()))
    return {
        "concentration_0": c0,
        "v_s0": v_s0,
        "exponent": exponent,
        "phi_max": phi_max,
        "diffusivity": D,
        "tau_d": tau_d,
        "interface_velocity": v_meas,
        "interface_velocity_expected": v_expected,
        "relative_error": abs(v_meas - v_expected) / v_expected if n_pts >= 3 else float("nan"),
        "mass_drift": abs(mass1 - mass0) / mass0,
        "times": times,
        "heights": heights,
        "profile_final": c_final.detach().cpu(),
        "snapshots": snaps,
        "y": y.cpu(),
        "steps": int(step),
        "H": H,
        "nx": nx,
        "init_height_fraction": init_height_fraction,
        "base_profile": c_init.cpu(),
    }
