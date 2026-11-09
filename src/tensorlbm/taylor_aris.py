"""Taylor–Aris dispersion of a passive scalar in pressure-driven channel / pipe flow.

Why this module exists
----------------------
P97-N05 asks for a *passive-scalar transport* verification face: the lattice
Boltzmann scalar solver must reproduce, within a few percent, the classical
long-time dispersion law of Taylor (1953) and Aris (1956)

.. math::

    \\partial_t \\phi + u(y)\\,\\partial_x \\phi = D\\,\\nabla^2 \\phi ,

whose cross-sectionally averaged field obeys an effective one-dimensional
diffusion equation with

.. math::

    D_{\\rm eff} = D + \\langle u' \\chi \\rangle ,\\qquad
    D\\,\\nabla_\\perp^2 \\chi = u',\\quad \\partial_n \\chi = 0 ,

where ``u' = u - <u>`` is the shear-induced velocity perturbation and ``chi`` the
Aris auxiliary field.  Two closed forms are used as references:

* plane Poiseuille (half-width ``h``, mean velocity ``U``):
  ``D_eff = D + 2 h^2 U^2 / (105 D)``
* circular pipe (radius ``a``, mean velocity ``U``):
  ``D_eff = D + a^2 U^2 / (48 D)``

Three things are provided, deliberately separated so that the physics and the
numerics can be blamed independently:

1. **Analytic references** — :func:`taylor_aris_channel`,
   :func:`taylor_aris_pipe` (closed forms) and
   :func:`aris_dispersion_coefficient`, which solves the *discrete* Aris
   problem on the very lattice nodes used by the LBM.  The discrete solve
   removes the quadrature ambiguity (a parabola sampled on ``n`` nodes has a
   mean that differs from ``U`` by ``O(h^-2)``), so the LBM residual is pure
   numerical error rather than a bookkeeping mismatch.
2. **A D2Q9 passive-scalar LBM** — :func:`scalar_equilibrium`,
   :func:`scalar_collide`, :func:`scalar_stream` with periodic ``x`` and
   zero-flux (bounce-back) walls in ``y``, diffusivity ``D = c_s^2 (tau-1/2)``.
3. **A dispersion meter** — :func:`simulate_channel_dispersion` tracks the
   second moment of the cross-sectionally averaged plume in a moving frame and
   returns the fitted ``D_eff``.  Controls isolate numerical diffusion:
   ``u_mean = 0`` (pure diffusion, must return ``D``) and ``profile="plug"``
   (uniform advection, no shear, must also return ``D``).

References
----------
Taylor, G. I. (1953). Dispersion of soluble matter in solvent flowing slowly
    through a tube. *Proc. R. Soc. Lond. A* 219, 186–203.
Aris, R. (1956). On the dispersion of a solute in a fluid flowing through a
    tube. *Proc. R. Soc. Lond. A* 235, 67–77.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import torch

from .d2q9 import C, OPPOSITE, W, equilibrium, macroscopic
from .solver import stream

__all__ = [
    "poiseuille_channel_velocity",
    "poiseuille_pipe_velocity",
    "taylor_aris_channel",
    "taylor_aris_pipe",
    "aris_dispersion_coefficient",
    "scalar_equilibrium",
    "scalar_collide",
    "scalar_stream",
    "lbm_poiseuille_profile",
    "DispersionConfig",
    "simulate_channel_dispersion",
]

CS2 = 1.0 / 3.0

#: Direction weights re-normalised so that they sum to exactly one in the
#: working dtype.  The float32 ``W`` of :mod:`tensorlbm.d2q9` sums to
#: ``1 + O(2^-25)``; without this correction the equilibrium carries a
#: ``3e-8`` mass defect per step and a 200-step pure-diffusion run drifts by
#: ``3e-6`` (measured before the fix, see
#: ``tests/test_taylor_aris.py::test_scalar_stream_conserves_mass_with_zero_flux_walls``).
_W_CACHE: dict[Any, torch.Tensor] = {}


def _weights(dtype: torch.dtype) -> torch.Tensor:
    key = dtype
    if key not in _W_CACHE:
        w = W.double()
        w = w / w.sum()
        wq = w.to(dtype)
        for _ in range(4):  # absorb the quantisation residual (< 1 ulp)
            residual = 1.0 - float(wq.double().sum())
            if residual == 0.0:
                break
            i = int(wq.argmax())
            wq[i] = wq[i] + float(residual)
        _W_CACHE[key] = wq.view(9, 1, 1)
    return _W_CACHE[key]

# Closed-form Aris coefficients: D_eff = D + K * (a^2 U^2 / D)
ARIS_K_CHANNEL = 2.0 / 105.0  # half-width h between parallel plates
ARIS_K_PIPE = 1.0 / 48.0  # radius a of a circular pipe


# ── analytic velocity profiles ────────────────────────────────────────────────
def poiseuille_channel_velocity(
    y: np.ndarray | torch.Tensor,
    half_width: float,
    u_mean: float,
    profile: str = "parabolic",
) -> np.ndarray | torch.Tensor:
    """Streamwise velocity of plane Poiseuille flow between ``y = ±half_width``.

    ``profile="parabolic"``: ``u = (3/2) U (1 - (y/h)^2)``
    ``profile="plug"``: uniform ``u = U`` (control: no shear, no Taylor term)
    """
    if profile not in ("parabolic", "plug"):
        raise ValueError(f"unknown profile: {profile}")
    if isinstance(y, torch.Tensor):
        if profile == "plug":
            return torch.full_like(y, float(u_mean))
        return 1.5 * u_mean * (1.0 - (y / half_width) ** 2)
    y = np.asarray(y, dtype=np.float64)
    if profile == "plug":
        return np.full_like(y, float(u_mean))
    return 1.5 * u_mean * (1.0 - (y / half_width) ** 2)


def poiseuille_pipe_velocity(
    r: np.ndarray, radius: float, u_mean: float, profile: str = "parabolic"
) -> np.ndarray:
    """Axisymmetric Poiseuille velocity in a circular pipe of radius ``a``.

    ``u = 2 U (1 - (r/a)^2)`` (parabolic) or ``u = U`` (plug).
    """
    r = np.asarray(r, dtype=np.float64)
    if profile == "plug":
        return np.full_like(r, float(u_mean))
    if profile == "parabolic":
        return 2.0 * u_mean * (1.0 - (r / radius) ** 2)
    raise ValueError(f"unknown profile: {profile}")


def taylor_aris_channel(diffusivity: float, half_width: float, u_mean: float) -> float:
    """Taylor–Aris effective diffusivity between parallel plates (closed form)."""
    if diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive")
    return diffusivity + ARIS_K_CHANNEL * half_width**2 * u_mean**2 / diffusivity


def taylor_aris_pipe(diffusivity: float, radius: float, u_mean: float) -> float:
    """Taylor–Aris effective diffusivity in a circular pipe (closed form)."""
    if diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive")
    return diffusivity + ARIS_K_PIPE * radius**2 * u_mean**2 / diffusivity


# ── discrete Aris auxiliary problem ───────────────────────────────────────────
def aris_dispersion_coefficient(
    velocity: Sequence[float] | np.ndarray,
    diffusivity: float,
    spacing: float = 1.0,
) -> dict[str, float]:
    """Solve ``D ∇²χ = u'`` with no-flux side walls on the given node set.

    Parameters
    ----------
    velocity :
        Streamwise velocity sampled on the transverse nodes (``n`` values,
        uniform spacing, first and last node *inside* the fluid; the walls sit
        half a cell outside at ``±n/2``).
    diffusivity :
        Molecular diffusivity ``D`` in lattice units.
    spacing :
        Node spacing (lattice units, normally 1).

    Returns
    -------
    dict with ``d_eff`` (``D + <u' χ>``), ``mean_velocity``,
    ``taylor_term = <u' χ>`` and the ``residual`` of the discrete equation
    (a solver-quality check, not a physics check).
    """
    u = np.asarray(velocity, dtype=np.float64)
    n = u.size
    if n < 3:
        raise ValueError("need at least three transverse nodes")
    h2 = spacing**2
    u_mean = float(u.mean())
    u_prime = u - u_mean

    # Discrete Laplacian with no-flux (odd reflection) at both walls:
    #   chi_{-1} = chi_0,  chi_n = chi_{n-1}
    A = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        A[j, j] = -2.0
        A[j, j + 1 if j + 1 < n else j] += 1.0
        A[j, j - 1 if j - 1 >= 0 else j] += 1.0
    A /= h2

    # Pin the null space (constant mode) before solving.
    # Quasi-steady Aris problem:  D ∇²χ = -u'  (so that D_eff = D + <u'χ> > D).
    A = A + np.ones((n, n)) / n
    chi = np.linalg.solve(A, -u_prime / diffusivity)
    chi = chi - chi.mean()
    taylor_term = float(np.mean(u_prime * chi))
    residual = float(np.max(np.abs(diffusivity * (A - np.ones((n, n)) / n) @ chi + u_prime)))
    return {
        "d_eff": diffusivity + taylor_term,
        "taylor_term": taylor_term,
        "mean_velocity": u_mean,
        "residual": residual,
        "n_nodes": n,
    }


# ── D2Q9 passive-scalar LBM ──────────────────────────────────────────────────
def scalar_equilibrium(
    phi: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor
) -> torch.Tensor:
    """``g_i^eq = w_i φ (1 + 3 c_i·u)`` — first-order advection equilibrium."""
    c = C.to(device=phi.device, dtype=phi.dtype)
    w = _weights(phi.dtype).to(phi.device)
    cu = c[:, 0].view(9, 1, 1) * ux + c[:, 1].view(9, 1, 1) * uy
    return w * phi.unsqueeze(0) * (1.0 + 3.0 * cu)


def scalar_collide(
    g: torch.Tensor, phi: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, tau_d: float
) -> torch.Tensor:
    """BGK relaxation of the scalar populations; ``D = c_s^2 (tau_d - 1/2)``."""
    geq = scalar_equilibrium(phi, ux, uy)
    return g - (g - geq) / tau_d


def scalar_stream(
    g: torch.Tensor, wall_rows: torch.Tensor | None = None
) -> torch.Tensor:
    """Periodic-in-``x`` streaming with zero-flux half-way walls in ``y``.

    The fluid nodes are rows ``0 … ny-1``; the walls sit half a cell outside at
    ``y = -1/2`` and ``y = ny - 1/2``.  Populations that would stream out of
    the domain are reflected at the wall node (half-way bounce-back,
    ``g_i(x_w, t+1) = g_{opp(i)}^post(x_w, t)``), which is the zero-flux
    (adiabatic) boundary condition for a passive scalar and conserves the total
    scalar mass exactly.  The ``wall_rows`` argument is accepted for API
    symmetry with :func:`tensorlbm.thermal.temperature_stream` but has no
    effect: the reflection is applied wherever the target row would fall
    outside the domain.
    """
    ny, nx = g.shape[1], g.shape[2]
    out = torch.empty_like(g)
    c = C.tolist()
    opp = OPPOSITE.tolist()
    for i, (cx, cy) in enumerate(c):
        # out(x) = in(x - c): a +x population must arrive one cell downstream
        src = torch.roll(g[i], shifts=cx, dims=1) if cx else g[i]
        if cy == 0:
            out[i] = src
        elif cy > 0:
            if ny - cy > 0:
                out[i, cy:] = src[: ny - cy]
            out[i, :cy] = g[opp[i], :cy]
        else:
            k = -cy
            if ny - k > 0:
                out[i, :-k] = src[k:]
            out[i, -k:] = g[opp[i], -k:]
    return out


# ── optional: velocity field from a real LBM Poiseuille solve ────────────────
def lbm_poiseuille_profile(
    ny: int,
    nx: int = 4,
    tau: float = 0.8,
    body_force: float = 1e-5,
    steps: int = 40000,
    tol: float = 1e-9,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Steady plane-Poiseuille profile from the library's own D2Q9 solver.

    Used as an end-to-end alternative to imposing the analytic parabola: the
    momentum field is then a genuine LBM solution (verified/poiseuille_2d
    covers the accuracy of that solve; here it only feeds the scalar solve).

    Guo forcing with a uniform streamwise body force, half-way bounce-back on
    the first/last rows; the returned ``u`` is a numpy array of length ``ny``
    (velocity on the fluid nodes, zero on the wall rows).
    """
    from .thermal import collide_bgk_force, pre_streaming_bounce_back

    device = torch.device(device)
    nx = max(int(nx), 1)
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True
    rho = torch.ones((ny, nx), device=device)
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, zero)
    F = torch.zeros((2, ny, nx), device=device)
    F[0] = float(body_force)
    prev = None
    step = 0
    for step in range(1, steps + 1):
        f_pre = f
        f = collide_bgk_force(f, tau, F)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = stream(f)
        if step % 1000 == 0:
            _, ux, _ = macroscopic(f)
            u = ux[ny // 2]
            if prev is not None and float((u - prev).abs().max()) < tol * max(float(u.abs().max()), 1e-30):
                break
            prev = u.clone()
    _, ux, _ = macroscopic(f)
    return {
        "u": ux[:, 0].detach().cpu().numpy(),
        "steps": step,
        "body_force": float(body_force),
        "tau": float(tau),
        "nu": (tau - 0.5) / 3.0,
        "u_max": float(ux.max().item()),
    }


# ── dispersion meter ─────────────────────────────────────────────────────────
@dataclass
class DispersionConfig:
    """Configuration of a Taylor–Aris dispersion measurement."""

    nx: int = 256
    ny: int = 17  # odd: one node on the centreline; walls at y = ±(ny/2)
    diffusivity: float = 0.02
    u_mean: float = 0.04
    steps: int = 40000
    profile: Literal["parabolic", "plug"] = "parabolic"
    init_width: float = 2.0
    method: Literal["variance", "mode"] = "mode"
    mode_waves: float = 1.0  # single-Fourier-mode method: waves across nx
    report_every: int = 100
    fit_start_frac: float = 0.5
    fit_end_frac: float = 1.0
    noise_floor_frac: float = 2e-3  # mode method: drop samples below this fraction of A0
    mode_amplitude: float = 0.4  # mode method: initial amplitude of the cos(kx) mode
    dtype: str = "float32"
    auto_window: bool = True
    renorm: bool = True  # mode method: rescale the field to a fixed amplitude
    device: str = "cpu"
    seed_x: float | None = None
    velocity_field: Sequence[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _cross_section_mean(g: torch.Tensor) -> torch.Tensor:
    return g.sum(dim=0).mean(dim=0)


def _plume_moments(phi_x: torch.Tensor, nx: int) -> tuple[float, float]:
    """Centroid and variance of a periodic 1-D plume, made wrap-safe.

    The profile is first circularly rolled so that its maximum sits at
    ``nx // 2``; the second moment is then evaluated with unwrapped
    ``x - nx//2`` coordinates.  Without the roll a plume straddling ``x = 0``
    is split into two pieces and the variance jumps by ``O(nx^2)`` (this is
    exactly the artefact that produced the negative ``D_eff`` in the first
    development run).  Requirement: the plume must stay narrower than the
    domain (``sigma < nx/6``), which the driver checks.
    """
    prof = phi_x.double()
    m0 = prof.sum()
    if float(m0) <= 0.0:
        raise RuntimeError("plume has zero mass")
    k = int(torch.argmax(prof).item())
    center = nx // 2
    prof = torch.roll(prof, shifts=center - k)
    dx = torch.arange(nx, dtype=torch.float64, device=prof.device) - center
    var = float((prof * dx**2).sum() / m0)
    return float(k), var


def simulate_channel_dispersion(cfg: DispersionConfig) -> dict[str, Any]:
    """Run the D2Q9 scalar LBM in plane Poiseuille flow and fit ``D_eff``.

    Two equivalent measurement protocols are implemented, because they fail in
    different ways:

    ``method="mode"`` (default, recommended)
        Release ``φ = 1 + A cos(kx)`` (uniform across the channel, ``k n_x =
        2π·waves``) and follow the amplitude decay of the cross-sectionally
        averaged field.  For ``t >> h²/D`` the slow rate is ``D_eff k²``; the
        instantaneous rate ``-d ln a / dt / k²`` converges to ``D_eff`` from
        below.  Because a single Fourier mode is periodic by construction there
        is no domain-size/aliasing bias and a short box suffices.
    ``method="variance"``
        Classical Taylor release test: a narrow Gaussian plume spreads as
        ``Var(t) = Var(0) + 2 D_eff t``.  Requires ``sigma << n_x`` (checked via
        the returned ``sigma_over_nx``); the plume is kept centred by
        whole-cell rolls in a moving frame.

    Both return the fitted ``d_eff`` together with the discrete Aris reference
    (:func:`aris_dispersion_coefficient`) computed on the very velocity profile
    that the scalar sees, the closed-form channel value and diagnostics
    (``mass_drift``, ``r2``).
    """
    device = torch.device(cfg.device)
    nx, ny = int(cfg.nx), int(cfg.ny)
    if ny % 2 == 0:
        raise ValueError("ny must be odd so a node sits on the centreline")
    half_width = 0.5 * ny  # half-way walls at ±ny/2
    D = float(cfg.diffusivity)
    if D <= 0.0:
        raise ValueError("diffusivity must be positive")
    tau_d = 3.0 * D + 0.5

    y = torch.arange(ny, dtype=torch.float64) - 0.5 * (ny - 1)
    if cfg.velocity_field is not None:
        u_np = np.asarray(cfg.velocity_field, dtype=np.float64)
        if u_np.size != ny:
            raise ValueError("velocity_field length must equal ny")
    else:
        u_np = np.asarray(
            poiseuille_channel_velocity(y.numpy(), half_width, cfg.u_mean, cfg.profile)
        )
    tdt = torch.float64 if cfg.dtype == "float64" else torch.float32
    ux = (
        torch.tensor(u_np, dtype=tdt, device=device)
        .view(ny, 1)
        .expand(ny, nx)
        .contiguous()
    )
    uy = torch.zeros((ny, nx), dtype=tdt, device=device)

    x = torch.arange(nx, dtype=torch.float64, device=device)
    if cfg.method == "mode":
        k = 2.0 * math.pi * float(cfg.mode_waves) / nx
        phi_x0 = 1.0 + float(cfg.mode_amplitude) * torch.cos(k * x)
        phi = phi_x0.to(tdt).view(1, nx).expand(ny, nx).contiguous()
    elif cfg.method == "variance":
        x0 = 0.5 * nx if cfg.seed_x is None else float(cfg.seed_x)
        sigma0 = float(cfg.init_width)
        gauss = torch.exp(-0.5 * ((x - x0) / sigma0) ** 2)
        gauss = gauss / gauss.sum()
        phi = gauss.to(tdt).view(1, nx).expand(ny, nx).contiguous()
    else:
        raise ValueError(f"unknown method: {cfg.method}")
    g = scalar_equilibrium(phi, ux, uy)

    wall_rows = torch.zeros(ny, dtype=torch.bool, device=device)
    wall_rows[0] = True
    wall_rows[-1] = True

    mass0 = float(phi.double().sum()) / ny
    u_mean_local = float(u_np.mean())
    drift_accum = 0.0
    hist: list[tuple[int, float, float]] = []
    a_ref = None
    a_prev = None
    step_prev = 0
    for step in range(1, cfg.steps + 1):
        phi_cur = g.sum(dim=0)
        g = scalar_collide(g, phi_cur, ux, uy, tau_d)
        g = scalar_stream(g, wall_rows)
        drift_accum += u_mean_local
        if drift_accum >= 1.0:  # exact Galilean shift in whole cells
            shift = int(drift_accum)
            g = torch.roll(g, shifts=-shift, dims=2)
            drift_accum -= shift
        if step % cfg.report_every == 0:
            phi_x = _cross_section_mean(g)
            if cfg.method == "mode":
                amp = _mode_amplitude(phi_x, k)
                if a_ref is None:
                    a_ref = amp
                rate = float("nan")
                if a_prev is not None and a_prev > 0.0 and amp > 0.0:
                    rate = -math.log(amp / a_prev) / ((step - step_prev) * k**2)
                hist.append((step, amp, rate))
                # If the field is rescaled below, the amplitude that the next
                # interval starts from is ``a_ref``, not the measured ``amp``;
                # comparing against ``amp`` made the rate vanish identically.
                a_prev, step_prev = amp, step
                if cfg.renorm and amp > 0.0:
                    a_prev = a_ref
                    # The equation is linear, so rescaling the perturbation is
                    # exact; without it the amplitude reaches round-off long
                    # before the transverse transient (t ~ h^2/D) has decayed,
                    # which makes the asymptotic rate unmeasurable at high Pe.
                    # Only the *deviation from the x-averaged background* is
                    # rescaled (rescaling the whole field instead was observed
                    # to amplify undamped transverse modes until the meter went
                    # unstable at tau_d -> 0.5).
                    phi_bar = float(phi_x.double().sum()) / nx
                    base = scalar_equilibrium(
                        torch.full_like(phi_cur, phi_bar), ux, uy
                    )
                    g = base + (g - base) * (a_ref / amp)
            else:
                _, var = _plume_moments(phi_x, nx)
                hist.append((step, var, float("nan")))

    phi_x = _cross_section_mean(g)
    mass1 = float(phi_x.double().sum())
    mass_drift = abs(mass1 - mass0) / mass0

    t = np.array([h[0] for h in hist], dtype=np.float64)
    q = np.array([h[1] for h in hist], dtype=np.float64)
    rate = np.array([h[2] for h in hist], dtype=np.float64)
    t_min_phys = 3.0 * half_width**2 / D  # transverse equilibration, 3 time constants
    if cfg.method == "mode" and cfg.renorm:
        thr = max(t_min_phys, 0.2 * t[-1])
        win = t >= thr
        window_ok = bool(win.sum() >= 5)
        if not window_ok:  # transient not over: fall back and flag it
            win = t >= 0.5 * t[-1]
        rr = rate[win]
        rr = rr[np.isfinite(rr)]
        d_eff_fit = float(rr.mean()) if rr.size else float("nan")
        rate_scatter = float(rr.std() / abs(d_eff_fit)) if rr.size and d_eff_fit else float("nan")
        r2 = 1.0 - rate_scatter**2 if rate_scatter == rate_scatter else float("nan")
        tf, qf = t[win], q[win]
        var_final = sigma_final = float("nan")
        fit_note = (
            f"renormalised constant-rate fit over t>={tf[0]:.0f}"
            if window_ok
            else "renormalised rate, transient NOT resolved (window too short)"
        )
    else:
        i0 = int(round(len(t) * cfg.fit_start_frac))
        i1 = max(i0 + 4, int(round(len(t) * cfg.fit_end_frac)))
        if cfg.method == "mode" and cfg.auto_window:
            # The amplitude reaching the round-off floor makes log-slopes
            # meaningless (observed as a spurious rate *increase* in the last
            # window of the first development runs).
            below = np.nonzero(q < cfg.noise_floor_frac * q[0])[0]
            t_noise = float(t[below[0]]) if below.size else float(t[-1])
            t_min = max(t_min_phys, 0.15 * t_noise)
            i0 = int(np.searchsorted(t, t_min))
            i1 = int(np.searchsorted(t, t_noise))
            i0 = min(i0, max(0, len(t) - 6))
            if i1 - i0 < 5:
                i1 = len(t)
        tf, qf = t[i0:i1], q[i0:i1]
        if cfg.method == "mode":
            slope, intercept = np.polyfit(tf, np.log(qf), 1)
            r2 = _r2(np.log(qf), slope * tf + intercept)
            d_eff_fit = -float(slope) / (k**2)
        else:
            slope, intercept = np.polyfit(tf, qf, 1)
            r2 = _r2(qf, slope * tf + intercept)
            d_eff_fit = 0.5 * float(slope)
        var_final = float(qf[-1]) if cfg.method == "variance" else float("nan")
        sigma_final = math.sqrt(max(var_final, 0.0)) if var_final == var_final else float("nan")
        fit_note = "log/slope fit"

    disc = aris_dispersion_coefficient(u_np, D)
    analytic = float("nan") if cfg.velocity_field is not None else (
        taylor_aris_channel(D, half_width, cfg.u_mean) if cfg.profile == "parabolic" else D
    )

    return {
        "d_eff": d_eff_fit,
        "d_eff_discrete": disc["d_eff"],
        "d_eff_analytic": analytic,
        "taylor_term_discrete": disc["taylor_term"],
        "d_molecular": D,
        "tau_d": tau_d,
        "method": cfg.method,
        "pe": float(abs(u_np.mean()) * half_width / D),
        "u_mean_imposed": float(cfg.u_mean),
        "u_mean_profile": u_mean_local,
        "half_width": half_width,
        "ny": ny,
        "nx": nx,
        "steps": cfg.steps,
        "profile": cfg.profile,
        "mass_drift": mass_drift,
        "variance_final": var_final,
        "sigma_final": sigma_final,
        "sigma_over_nx": sigma_final / nx if sigma_final == sigma_final else float("nan"),
        "r2": r2,
        "rate_scatter": float(rate_scatter) if cfg.method == "mode" and cfg.renorm else float("nan"),
        "fit_t0": float(tf[0]),
        "fit_t1": float(tf[-1]),
        "fit_n": int(tf.size),
        "fit_note": fit_note,
        "transient_resolved": bool(cfg.method != "mode" or not cfg.renorm or window_ok),
        "dtype": cfg.dtype,
        "renorm": bool(cfg.renorm),
        "rel_err_vs_discrete": (d_eff_fit - disc["d_eff"]) / disc["d_eff"],
        "rel_err_vs_analytic": (d_eff_fit - analytic) / analytic
        if analytic == analytic and analytic != 0
        else float("nan"),
        "history": {"t": t.tolist(), "q": q.tolist(), "rate": rate.tolist()},
    }


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination (1.0 for a perfectly predicted constant)."""
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def _mode_amplitude(phi_x: torch.Tensor, k: float) -> float:
    """Amplitude of the ``cos(kx)`` Fourier component of a 1-D profile."""
    x = torch.arange(phi_x.numel(), dtype=torch.float64, device=phi_x.device)
    ph = phi_x.double()
    a = 2.0 * float((ph * torch.cos(k * x)).sum()) / ph.numel()
    b = 2.0 * float((ph * torch.sin(k * x)).sum()) / ph.numel()
    return math.hypot(a, b)
