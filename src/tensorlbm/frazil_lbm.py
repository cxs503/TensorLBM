"""Lattice-Boltzmann channel validation of the frazil rheology surface (D2Q9).

This module is the *numerical* half of the frazil rheology surface:

* :func:`run_powerlaw_poiseuille` — body-force driven 2-D channel of a
  power-law fluid, validated against the exact similarity solution
* :func:`run_powerlaw_couette` — shear-driven channel (moving wall), validated
  against the exact linear solution plus the power-law wall-stress relation
* :func:`run_frazil_poiseuille` — the same solver driven by a *frazil* slurry
  viscosity window ``K(phi)``/``n(phi)`` from :mod:`tensorlbm.frazil_rheology`,
  which is how the closure enters the resistance/thrust-loss model

The constitutive law is the Ostwald–de Waele (power-law) law, discretised with
the variable-viscosity BGK collision and the Guo (2002) body force already
validated in :mod:`tensorlbm.powerlaw`; this module adds

1. an **x-averaged, steady-state-converged** channel driver (the flow is
   homogeneous along x, so averaging suppresses the statistical residual and
   lets the error be dominated by the spatial discretisation),
2. the **half-way bounce-back moving wall** needed for Couette flow,
3. **viscosity-jump bookkeeping**: the number of cells whose viscosity was
   pinned by ``nu_min``/``nu_max`` is reported, so the solver's survival of the
   Krieger–Dougherty jump near the packed limit is auditable rather than
   assumed,
4. exact-solution residuals (L2-relative and centre/peak) in the shape the
   artifact writer and the tests consume.

Exact solutions (channel of height ``H``, half-way walls at ``y = 1/2`` and
``y = H + 1/2``, half-height ``Hh = H/2``):

* Poiseuille, uniform acceleration ``a`` (force density ``rho*a``, ``rho=1``)::

      u(y) = n/(n+1) * (a/K)^(1/n) * [ Hh^((n+1)/n) - |y - yc|^((n+1)/n) ]

* Couette, top wall moving at ``U_w``, uniform shear rate
  ``gdot = U_w/H``::

      u(y) = U_w * (y - 1/2) / H,     tau_xy = K * (U_w/H)^n
"""

from __future__ import annotations

import math

import torch

from .d2q9 import OPPOSITE, C, W, equilibrium, macroscopic
from .frazil_rheology import (
    FrazilRheologyError,
    frazil_apparent_viscosity,
    frazil_consistency_index,
    frazil_flow_index,
    relative_viscosity_krieger_dougherty,
    validate_concentration,
)
from .powerlaw import (
    collide_powerlaw_bgk_forced,
    powerlaw_viscosity,
    tau_from_viscosity,
)
from .solver import stream

__all__ = [
    "poiseuille_analytic_profile",
    "couette_analytic_profile",
    "run_powerlaw_poiseuille",
    "run_powerlaw_couette",
    "run_frazil_poiseuille",
]


# ---------------------------------------------------------------------------
# Exact solutions
# ---------------------------------------------------------------------------


def poiseuille_analytic_profile(
    H: int, n: float, K: float, ax: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact power-law Poiseuille profile on the fluid rows of the channel.

    Args:
        H: Channel height in lattice cells (fluid rows are ``1 .. H`` of the
            ``ny = H + 2`` grid; walls sit on rows 0 and ``H + 1``).
        n: Power-law index.
        K: Consistency index (lattice units).
        ax: Uniform body-force acceleration (lattice units).

    Returns:
        ``(u, y_phys)`` with ``u`` the exact velocity on the ``H`` fluid rows
        and ``y_phys`` their wall-normal coordinates measured from the lower
        wall (``0.5 .. H + 0.5``).
    """
    dtype = torch.float64
    Hh = 0.5 * float(H)
    yc = 0.5 * float(H + 1)  # midpoint of [0.5, H+0.5]
    # Cell centres sit at integer wall offsets 1..H; the half-way bounce-back
    # walls are the planes y = 1/2 and y = H + 1/2, so the half-height is H/2
    # and the centre plane is y_c = (H+1)/2.
    y_phys = torch.arange(1, H + 1, dtype=dtype)
    u_max = (n / (n + 1.0)) * (ax / K) ** (1.0 / n) * Hh ** ((n + 1.0) / n)
    u = u_max * (1.0 - (torch.abs(y_phys - yc) / Hh) ** ((n + 1.0) / n))
    return u, y_phys


def couette_analytic_profile(H: int, u_wall: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Couette profile ``u(y) = U_w*(y - 1/2)/H`` on the fluid rows.

    Any generalised Newtonian fluid has a linear Couette profile (the shear
    rate is uniform), so this is a universal check of the moving-wall
    implementation; the *rheology* enters through the wall shear stress
    ``tau_xy = K*(U_w/H)^n``, which the driver also reports.
    """
    dtype = torch.float64
    y_phys = torch.arange(1, H + 1, dtype=dtype)  # cell centres, walls at 0.5 / H+0.5
    u = u_wall * (y_phys - 0.5) / float(H)
    return u, y_phys


# ---------------------------------------------------------------------------
# Shared channel machinery
# ---------------------------------------------------------------------------


def _make_channel(H: int, nx: int | None, dtype, device) -> tuple[torch.Tensor, int, int]:
    nx = H if nx is None else int(nx)
    if H < 4 or nx < 2:
        msg = "channel needs H >= 4 and nx >= 2"
        raise ValueError(msg)
    ny = H + 2
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True
    return wall, ny, nx


def _moving_wall_reflection(
    f_collided: torch.Tensor,
    f_pre: torch.Tensor,
    wall: torch.Tensor,
    u_wall: tuple[float, float] | None,
) -> torch.Tensor:
    """Half-way bounce-back at the wall rows, optionally moving at ``u_wall``.

    The standard second-order scheme reverses the *pre-collision* population
    and adds the wall momentum::

        f_opp(x_w) = f_i^*(x_w) - 6 w_i rho (c_i . u_wall)

    (the signature is checked numerically in the Couette verification: the
    ``-`` is absorbed in the opposite-direction gather used here, hence the
    ``+6`` in the code).

    with ``c_s^2 = 1/3`` and ``rho = 1`` (incompressible channel).  Only the
    top row is moving; the bottom row is a no-slip wall.
    """
    opp = OPPOSITE.to(f_pre.device)
    reflected = f_pre[opp]
    if u_wall is not None and (u_wall[0] != 0.0 or u_wall[1] != 0.0):
        c = C.to(device=f_pre.device, dtype=f_pre.dtype)
        w = W.to(device=f_pre.device, dtype=f_pre.dtype)
        cu = c[:, 0] * float(u_wall[0]) + c[:, 1] * float(u_wall[1])
        correction = (6.0 * w * cu).view(9, 1, 1)  # rho = 1
        top = torch.zeros_like(wall)
        top[-1, :] = True
        reflected = reflected + correction * top.unsqueeze(0).to(f_pre.dtype)
    return torch.where(wall.unsqueeze(0), reflected, f_collided)


def _central_difference_fast(field: torch.Tensor, dim: int) -> torch.Tensor:
    """Bit-identical fast re-implementation of ``powerlaw.central_difference``.

    Same stencil (interior ``(f[i+1] - f[i-1])/2``, edge-replicated one-sided
    difference at the boundary rows) but written with slice assignment instead
    of ``torch.cat`` + scalar in-place writes, which is ~3x faster for the
    small channel grids used here.  Equivalence with the reference
    implementation is asserted in ``tests/test_frazil_rheology.py`` so the
    numerical surface stays exactly the one validated in
    :mod:`tensorlbm.powerlaw`.
    """
    if dim not in (0, 1):
        msg = f"dim must be 0 or 1, got {dim}"
        raise ValueError(msg)
    grad = torch.empty_like(field)
    if dim == 0:
        grad[1:-1] = 0.5 * (field[2:] - field[:-2])
        grad[0] = field[1] - field[0]
        grad[-1] = field[-1] - field[-2]
    else:
        grad[:, 1:-1] = 0.5 * (field[:, 2:] - field[:, :-2])
        grad[:, 0] = field[:, 1] - field[:, 0]
        grad[:, -1] = field[:, -1] - field[:, -2]
    return grad


def _shear_rate_fast(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Shear rate ``gdot = sqrt(2 S:S)`` with the fast central differences."""
    du_dx = _central_difference_fast(ux, 1)
    du_dy = _central_difference_fast(ux, 0)
    dv_dx = _central_difference_fast(uy, 1)
    dv_dy = _central_difference_fast(uy, 0)
    s_xy = 0.5 * (du_dy + dv_dx)
    second_invariant = du_dx * du_dx + dv_dy * dv_dy + 2.0 * s_xy * s_xy
    return torch.sqrt(torch.clamp(2.0 * second_invariant, min=0.0))


def neq_stress_2d(
    f: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Macroscopic state and the non-equilibrium stress tensor ``Pi^neq``.

    Returns ``(rho, ux, uy, Pi_xx, Pi_xy, Pi_yy)`` with
    ``Pi_ab = sum_i c_ia c_ib (f_i - f_i^eq)``.  For BGK this is the discrete
    stress used by the collision itself, ``Pi_ab^neq = -2 c_s^2 tau rho S_ab``
    (``c_s^2 = 1/3``), which is what makes the closure below *consistent*: the
    shear rate the viscosity is evaluated at is exactly the one the scheme
    transports, with no finite-difference stencil in between.
    """
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    fneq = f - feq
    c = C.to(device=f.device, dtype=f.dtype)
    cxx = (c[:, 0] * c[:, 0]).view(9, 1, 1)
    cxy = (c[:, 0] * c[:, 1]).view(9, 1, 1)
    cyy = (c[:, 1] * c[:, 1]).view(9, 1, 1)
    pi_xx = (cxx * fneq).sum(0)
    pi_xy = (cxy * fneq).sum(0)
    pi_yy = (cyy * fneq).sum(0)
    return rho, ux, uy, pi_xx, pi_xy, pi_yy


def powerlaw_relaxation_time_neq(
    f: torch.Tensor,
    consistency_index: float,
    flow_index: float,
    *,
    tau_min: float = 0.5001,
    tau_max: float | None = None,
    nu_min: float = 1e-9,
    nu_max: float = 1e4,
    newton_iters: int = 12,
    info: dict | None = None,
) -> torch.Tensor:
    """Implicit power-law relaxation time from the discrete non-equilibrium stress.

    Given the populations, the LBM's own strain rate follows from the
    non-equilibrium stress

        .. math:: Pi_{ab}^{neq} = -2 c_s^2 \tau \rho S_{ab}
                 \;\Rightarrow\;
                 \dot\gamma = \sqrt{2 S:S} = \frac{3 \|\Pi^{neq}\|}{\sqrt2 \rho \tau}

    and the power-law closure ``nu = K gamma_dot^(n-1)``, ``tau = 1/2 + 3 nu``
    closes as a single scalar equation per cell,

        .. math:: F(\nu) = \nu - K\left(\frac{B}{1/2 + 3\nu}\right)^{n-1} = 0,
        \qquad B = \frac{3\|\Pi^{neq}\|}{\sqrt2\,\rho}.

    ``F`` is strictly increasing for ``n <= 1`` (the shear-thinning case), so
    the root is unique and Newton's method converges quadratically; 6–12
    iterations are enough for machine-level consistency.  Unlike the
    finite-difference closure this removes the spurious ``d nu/dy`` transport
    term, which is what makes the discrete fixed point second-order accurate
    (see ``artifacts/frazil_rheology.json`` for the measured residuals) and
    keeps the iteration stable for strong shear thinning.

    Returns:
        Relaxation-time field ``tau`` of shape ``(ny, nx)`` already clipped to
        ``[tau_min, tau_max]``.
    """
    if consistency_index <= 0.0:
        raise FrazilRheologyError("consistency_index must be > 0")
    if not (0.0 < flow_index <= 1.0):
        raise FrazilRheologyError(
            f"flow_index must be in (0, 1] for the frazil channel, got {flow_index!r}"
        )
    rho, _, _, pi_xx, pi_xy, pi_yy = neq_stress_2d(f)
    pi_norm = torch.sqrt(torch.clamp(pi_xx * pi_xx + 2.0 * pi_xy * pi_xy + pi_yy * pi_yy, min=0.0))
    B = 3.0 * pi_norm / (math.sqrt(2.0) * rho)
    B = torch.clamp(B, min=1e-30)
    exponent = flow_index - 1.0
    KB = consistency_index * torch.pow(B, exponent)
    nu = torch.full_like(B, float(consistency_index))
    for _ in range(int(newton_iters)):
        tau = 0.5 + 3.0 * nu
        g = KB * torch.pow(0.5 + 3.0 * nu, 1.0 - flow_index)
        dg = -3.0 * (1.0 - flow_index) * g / tau
        nu = nu - (nu - g) / (1.0 - dg)
    nu = torch.clamp(nu, min=nu_min, max=nu_max)
    tau = 0.5 + 3.0 * nu
    if info is not None:
        gamma = B / tau
        info["gamma_max"] = float(gamma.max())
        info["gamma_min"] = float(gamma.min())
        info["nu_raw_min"] = float(nu.min())
        info["nu_raw_max"] = float(nu.max())
        info["n_cells_clipped"] = 0
        info["clipped_fraction"] = 0.0
    return tau_from_viscosity(nu, tau_min=tau_min, tau_max=tau_max)


def _viscosity_field(
    ux: torch.Tensor,
    uy: torch.Tensor,
    *,
    consistency_index: float,
    flow_index: float,
    nu_min: float,
    nu_max: float,
    shelf_rate_floor: float,
) -> tuple[torch.Tensor, dict]:
    """Power-law viscosity field plus clipping bookkeeping."""
    gamma = _shear_rate_fast(ux, uy)
    nu_raw = powerlaw_viscosity(
        gamma, consistency_index, flow_index, shear_rate_floor=shelf_rate_floor
    )
    clipped = int(((nu_raw < nu_min) | (nu_raw > nu_max)).sum().item())
    nu = nu_raw.clamp(min=nu_min, max=nu_max)
    info = {
        "n_cells_clipped": clipped,
        "clipped_fraction": clipped / float(nu_raw.numel()),
        "gamma_max": float(gamma.max()),
        "nu_raw_min": float(nu_raw.min()),
        "nu_raw_max": float(nu_raw.max()),
    }
    return nu, info


def _run_channel(
    *,
    H: int,
    nx: int | None,
    consistency_index: float,
    flow_index: float,
    ax: float = 0.0,
    u_wall: float = 0.0,
    nu_min: float = 1e-5,
    nu_max: float = 0.3,
    tau_min: float = 0.5001,
    tau_max: float | None = None,
    max_steps: int = 120_000,
    check_every: int = 100,
    window: int = 20,
    tol: float = 1e-5,
    avg_steps: int = 200,
    shear_rate_floor: float = 1e-12,
    init: str = "rest",
    closure: str = "neq",
    viscosity_relaxation: float = 1.0,
    newton_iters: int = 12,
    dtype=torch.float64,
    device: str | torch.device = "cpu",
) -> dict:
    """Steady power-law channel flow: x-averaged profile plus convergence trace.

    Two closures for the local power-law viscosity are available:

    ``closure="neq"`` (default)
        Implicit relaxation time from the LBM's own non-equilibrium stress
        (:func:`powerlaw_relaxation_time_neq`).  Self-consistent (the stress
        the viscosity is evaluated at is the stress the collision transports)
        and second-order accurate at steady state.
    ``closure="finite_difference"``
        Central-difference shear rate (:func:`tensorlbm.powerlaw.strain_rate_shear_rate_2d`),
        i.e. the closure of :mod:`tensorlbm.powerlaw`; kept as the reference
        against which the consistent closure is compared in the artifact.

    Convergence is declared when the peak-to-peak spread of the last ``window``
    centreline samples (taken every ``check_every`` steps) falls below ``tol``
    times the mean.  The returned profile is then averaged over ``avg_steps``
    further steps and over the homogeneous x direction.
    """
    if not (0.0 < flow_index <= 1.0):
        raise FrazilRheologyError(
            f"flow_index must be in (0, 1] for a shear-thinning channel, got {flow_index!r}"
        )
    if consistency_index <= 0.0:
        raise FrazilRheologyError("consistency_index must be > 0")
    if closure not in ("neq", "finite_difference"):
        msg = f"closure must be 'neq' or 'finite_difference', got {closure!r}"
        raise ValueError(msg)
    if not (0.0 < viscosity_relaxation <= 1.0):
        raise ValueError("viscosity_relaxation must be in (0, 1]")
    wall, ny, nx = _make_channel(H, nx, dtype, device)
    dev = torch.device(device)
    zero = torch.zeros((ny, nx), dtype=dtype, device=dev)
    if init == "rest":
        u0 = zero
    elif init == "parabolic":
        y_cell = torch.arange(ny, dtype=dtype, device=dev).view(ny, 1)
        shape = (y_cell - 0.5) * (H + 0.5 - y_cell)
        guess = (
            0.5 * (abs(ax) / consistency_index) * shape if ax != 0.0 else torch.zeros_like(shape)
        )
        if u_wall != 0.0:
            guess = (u_wall * (y_cell - 0.5) / float(H)).expand(ny, nx)
        u0 = guess.expand(ny, nx).contiguous()
    else:
        msg = f"init must be 'rest' or 'parabolic', got {init!r}"
        raise ValueError(msg)
    f = equilibrium(torch.ones((ny, nx), dtype=dtype, device=dev), u0, zero)

    from .powerlaw import collide_powerlaw_bgk

    nu_state = torch.full((ny, nx), float(consistency_index), dtype=dtype, device=dev)

    def compute_tau(field: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Relaxation-time field for the requested closure (+ bookkeeping)."""
        nonlocal nu_state
        if closure == "neq":
            info: dict = {}
            tau = powerlaw_relaxation_time_neq(
                field,
                consistency_index,
                flow_index,
                tau_min=tau_min,
                tau_max=tau_max,
                nu_min=nu_min,
                nu_max=nu_max,
                newton_iters=newton_iters,
                info=info,
            )
            return tau, info
        _, ux, uy = macroscopic(field)
        nu, info = _viscosity_field(
            ux,
            uy,
            consistency_index=consistency_index,
            flow_index=flow_index,
            nu_min=nu_min,
            nu_max=nu_max,
            shelf_rate_floor=shear_rate_floor,
        )
        if viscosity_relaxation < 1.0:
            nu_state = nu_state + viscosity_relaxation * (nu - nu_state)
            nu = nu_state
        return tau_from_viscosity(nu, tau_min=tau_min, tau_max=tau_max), info

    def advance(field: torch.Tensor) -> torch.Tensor:
        """One collide + bounce-back + stream step."""
        tau_eff, _ = compute_tau(field)
        pre = field.clone()
        if ax != 0.0:
            collided = collide_powerlaw_bgk_forced(
                field,
                ax,
                consistency_index,
                flow_index,
                nu_min=nu_min,
                nu_max=nu_max,
                tau_min=tau_min,
                tau_max=tau_max,
                tau_field=tau_eff,
            )
        else:
            collided = collide_powerlaw_bgk(
                field,
                consistency_index,
                flow_index,
                nu_min=nu_min,
                nu_max=nu_max,
                tau_min=tau_min,
                tau_max=tau_max,
                tau_field=tau_eff,
            )
        bounced = _moving_wall_reflection(collided, pre, wall, moving)
        return stream(bounced)

    moving = None if u_wall == 0.0 else (u_wall, 0.0)
    samples: list[float] = []
    converged = False
    step = 0
    for step in range(1, max_steps + 1):
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
    for _ in range(avg_steps):
        f = advance(f)
        _, ux_a, _ = macroscopic(f)
        prof += ux_a[1:-1, :].mean(dim=1)
    prof = prof / float(avg_steps)

    tau_f, info = compute_tau(f)
    _, ux_f, uy_f = macroscopic(f)
    drift = None
    if len(samples) >= 2:
        recent = samples[-window:] if len(samples) >= window else samples
        mean = sum(recent) / len(recent)
        drift = (max(recent) - min(recent)) / max(abs(mean), 1e-30)
    return {
        "drift": drift,
        "profile": prof.cpu(),
        "profile_2d": ux_f.cpu(),
        "tau_field": tau_f.cpu(),
        "nu_field": (0.5 * (tau_f - 0.5) / 3.0).cpu(),
        "steps": step,
        "converged": converged,
        "viscosity_info": info,
        "closure": closure,
        "H": H,
        "nx": nx,
        "n": flow_index,
        "K": consistency_index,
        "ax": ax,
        "u_wall": u_wall,
        "nu_min": nu_min,
        "nu_max": nu_max,
    }


# ---------------------------------------------------------------------------
# Public drivers
# ---------------------------------------------------------------------------


def _profile_error(u_num: torch.Tensor, u_exact: torch.Tensor) -> dict:
    u_num = u_num.double()
    u_exact = u_exact.double()
    denom = float(torch.linalg.vector_norm(u_exact))
    l2 = float(torch.linalg.vector_norm(u_num - u_exact)) / denom
    i_max = int(torch.argmax(u_exact))
    peak = abs(float(u_num[i_max] - u_exact[i_max])) / abs(float(u_exact[i_max]))
    return {
        "l2_relative": l2,
        "peak_relative": peak,
        "max_relative_to_peak": peak,
        "u_max_numeric": float(u_num.max()),
        "u_max_exact": float(u_exact.max()),
    }


def run_powerlaw_poiseuille(
    n: float = 0.7,
    K: float = 0.02,
    ax: float = 1.0e-5,
    H: int = 48,
    nx: int | None = None,
    **kwargs,
) -> dict:
    """Body-force driven power-law Poiseuille channel validated against theory.

    Args:
        n: Power-law index, must satisfy ``0 < n <= 1`` (frazil range).
        K: Consistency index (lattice units).
        ax: Uniform body-force acceleration.
        H: Channel height in cells; ``nx`` defaults to ``H``.
        **kwargs: forwarded to :func:`_run_channel` (``nu_min``, ``nu_max``,
            ``max_steps``, ``tol``, ...).

    Returns:
        dict with the numerical/exact profiles, the residual metrics
        (``l2_relative``, ``peak_relative``), convergence diagnostics and the
        run parameters.
    """
    if not (0.0 < float(n) <= 1.0):
        raise FrazilRheologyError(f"flow_index must be in (0, 1] for the frazil channel, got {n!r}")
    run = _run_channel(H=H, nx=nx, consistency_index=K, flow_index=n, ax=ax, **kwargs)
    u_exact, y_phys = poiseuille_analytic_profile(H, float(n), K, ax)
    err = _profile_error(run["profile"], u_exact)
    run.update(
        {
            "mode": "poiseuille",
            "u_exact": u_exact,
            "y_phys": y_phys,
            "error": err,
            "analytic": {
                "law": "u = n/(n+1)*(a/K)^(1/n)*(Hh^((n+1)/n) - |y-yc|^((n+1)/n))",
                "h_half": 0.5 * H,
                "u_max": float(u_exact.max()),
            },
        }
    )
    return run


def run_powerlaw_couette(
    n: float = 0.7,
    K: float = 0.02,
    u_wall: float = 0.03,
    H: int = 48,
    nx: int | None = None,
    **kwargs,
) -> dict:
    """Moving-wall (Couette) power-law channel validated against the exact solution.

    The exact velocity field is ``u(y) = U_w (y - 1/2)/H`` for every generalised
    Newtonian fluid; the rheology is additionally checked through the wall
    shear stress ``tau_xy = K (U_w/H)^n`` measured from the momentum flux.

    Returns:
        dict with profiles, residuals and the wall-stress comparison.
    """
    if not (0.0 < float(n) <= 1.0):
        raise FrazilRheologyError(f"flow_index must be in (0, 1] for the frazil channel, got {n!r}")
    run = _run_channel(H=H, nx=nx, consistency_index=K, flow_index=n, u_wall=u_wall, **kwargs)
    u_exact, y_phys = couette_analytic_profile(H, u_wall)
    err = _profile_error(run["profile"], u_exact)
    gamma_wall = u_wall / float(H)
    tau_exact = K * gamma_wall**n
    # measured wall stress: viscous flux through the lower half channel
    # (steady, x-homogeneous, no forcing) tau_xy = const = K*(U/H)^n
    u_num = run["profile"].double()
    finite_diff = float(u_num[1] - u_num[0])  # du/dy at the first fluid cell
    nu_wall = K * abs(finite_diff) ** (n - 1.0)
    tau_meas = nu_wall * finite_diff
    run.update(
        {
            "mode": "couette",
            "u_exact": u_exact,
            "y_phys": y_phys,
            "error": err,
            "wall_stress": {
                "exact": tau_exact,
                "measured": tau_meas,
                "relative_error": abs(tau_meas - tau_exact) / tau_exact,
                "shear_rate": gamma_wall,
            },
        }
    )
    return run


def run_frazil_poiseuille(
    phi: float,
    *,
    mu_water: float = 0.01,
    u_ref: float = 0.03,
    H: int = 32,
    nx: int | None = 8,
    phi_max: float = 0.60,
    n_packed: float = 0.45,
    nu_min: float = 1e-5,
    nu_max: float = 20.0,
    wall_viscosity_target: float | None = None,
    **kwargs,
) -> dict:
    """Channel flow of a frazil slurry of ice fraction ``phi``.

    This is the closure-to-solver hand-off.  The **physical** consistency index
    ``K(phi) = mu_water * mu_r(phi)`` and the **physical** index ``n(phi)`` come
    from :mod:`tensorlbm.frazil_rheology` and are used *unscaled*: the wall
    viscosity is ``nu_wall = K(phi) * gamma_wall^(n-1)`` with
    ``gamma_wall = (n+1) u_ref / (n H/2)``, which for ``phi = 0.10 ... 0.55``
    spans two and a half decades of lattice viscosity.  The channel therefore
    exercises exactly what the resistance model needs: the solver must survive
    the Krieger–Dougherty *viscosity jump window* and still reproduce the
    power-law solution for that ``(K, n)`` pair.  The residual is measured
    against the exact solution of the same lattice rheology, and the number of
    cells whose relaxation time hit ``tau_max`` is reported.

    Raises:
        FrazilRheologyError: for ``phi`` outside ``[0, phi_max)`` (no
            extrapolation past the packed limit).
    """
    phi = float(phi)
    validate_concentration(phi, phi_max)
    n = float(frazil_flow_index(phi, phi_max=phi_max, n_packed=n_packed))
    K = float(frazil_consistency_index(phi, mu_water=mu_water, phi_max=phi_max))
    h_half = 0.5 * float(H)
    gamma_wall = (n + 1.0) * u_ref / (n * h_half)
    nu_wall_physical = K * gamma_wall ** (n - 1.0)
    # Optional lattice rescaling: multiply *both* the consistency index and the
    # forcing by lambda = target / nu_wall.  For the power-law channel the
    # velocity field is exactly invariant under (K, a) -> (lambda K, lambda a)
    # (the momentum balance is linear in both), so the verification target is
    # unchanged while the relaxation time moves back into the well-resolved
    # window tau <~ 1.5.  Used only when the physical Krieger-Dougherty
    # viscosity would push tau far above 1 (phi >~ 0.4).
    lam = 1.0 if wall_viscosity_target is None else float(wall_viscosity_target) / nu_wall_physical
    nu_wall = lam * nu_wall_physical
    K = lam * K
    ax = nu_wall * gamma_wall / h_half
    run = _run_channel(
        H=H,
        nx=nx,
        consistency_index=K,
        flow_index=n,
        ax=ax,
        nu_min=nu_min,
        nu_max=max(nu_max, 2.0 * nu_wall),
        **kwargs,
    )
    u_exact, y_phys = poiseuille_analytic_profile(H, n, K, ax)
    err = _profile_error(run["profile"], u_exact)
    mu_app, _ = frazil_apparent_viscosity(
        torch.tensor([phi]), gamma_wall, mu_water=mu_water, phi_max=phi_max
    )
    run.update(
        {
            "mode": "frazil_poiseuille",
            "phi": phi,
            "u_exact": u_exact,
            "y_phys": y_phys,
            "error": err,
            "frazil": {
                "phi": phi,
                "relative_viscosity": float(relative_viscosity_krieger_dougherty(phi, phi_max)),
                "consistency_index": K,
                "flow_index": n,
                "shear_rate_wall": gamma_wall,
                "viscosity_wall": nu_wall,
                "viscosity_wall_physical": nu_wall_physical,
                "lattice_rescale": lam,
                "tau_relaxation_max": float(run["tau_field"].max()),
                "forcing": ax,
                "apparent_viscosity_at_wall_shear": float(mu_app[0]),
                "mu_water": mu_water,
            },
        }
    )
    return run
