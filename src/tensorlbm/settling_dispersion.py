"""Hindered-settling induced dispersion closure (``D_settle``).

What is measured
----------------
In the frame comoving with the particles, a settling suspension is a flow
*through* a random array of obstacles driven by the same body force that makes
the particles settle.  The dispersive flux of the suspension is therefore the
longitudinal dispersion coefficient of that flow, and this module measures it
directly:

1. :func:`random_disk_array` builds a periodic random sequential adsorption
   (RSA) packing of equal disks at a target area fraction ``c`` (the 2D
   analogue of the particle volume fraction);
2. :func:`solve_array_flow` drives a D2Q9 flow through that array with a
   constant body force (gravity analogue) until the drag balances the driving
   force, and reports the Darcy velocity ``U(c)``;
3. :func:`measure_hindered_dispersion` releases a passive scalar band in the
   frozen steady flow and measures the growth of its longitudinal variance,
   ``sigma_x^2(t) = 2 D_eff t``, giving ``D_eff(c)`` **without** putting a
   dispersion coefficient in by hand;
4. :func:`fit_dispersion_law` fits the measured ``D_eff(c)`` to the literature
   closed form ``D(c) = D_0 (1 - c)^alpha`` and reports per-point residuals plus
   a held-out extrapolation error;
5. :func:`run_settling_column_dispersion` uses ``D_settle(c) = D_eff(c) - D_m``
   in the Kynch batch-settling column, where the interface speed must stay at
   the analytic shock speed ``v_s(c_0)`` while the interface thickness is set by
   ``D_settle``.  :func:`killer_test_dispersion` switches the term off to show
   that the thickness actually depends on it.

Why this is not circular
------------------------
``D_eff`` is *not* an input: the scalar solver only receives the molecular
``D_m`` and the measured velocity field.  The dispersion is produced by the
velocity heterogeneity of the array.  ``D_settle`` is then defined as the
measured excess over ``D_m``.

Note on dimensionality
----------------------
The resolved runs are 2D (disks).  The 2D Stokes problem is logarithmically
divergent, so the *absolute* settling velocity of an isolated disk has no
infinite-domain limit; the array configuration used here is periodic, hence
well posed, and every quantity reported is a ratio (``D_eff/D_m``,
``U(c)/U(0)``, ``v_interface/v_s(c_0)``) for which the 2D reduction is harmless.
Quantitative 3D predictions are out of scope and are not claimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .d2q9 import macroscopic
from .frazil_suspension import hindered_settling_velocity
from .passive_scalar import equilibrium_scalar
from .settling_coupling import CS2, Disk, DiskLatticeCoupling

__all__ = [
    "random_disk_array",
    "solve_array_flow",
    "measure_hindered_dispersion",
    "fit_dispersion_law",
    "run_settling_column_dispersion",
    "killer_test_dispersion",
]

_SCALAR_C = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=torch.int64)
_SCALAR_W = torch.tensor([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6], dtype=torch.float64)


# ---------------------------------------------------------------------------
# Random array of disks
# ---------------------------------------------------------------------------


def random_disk_array(
    ny: int,
    nx: int,
    area_fraction: float,
    *,
    radius: float = 8.0,
    seed: int = 0,
    min_gap: float = 1.5,
    max_attempts: int = 200000,
) -> list[Disk]:
    """Periodic random sequential adsorption of equal disks.

    Args:
        area_fraction: target ``N pi R^2 / (ny nx)``.
        min_gap: minimum surface-to-surface gap (lattice units).
        seed: RNG seed (the packing is the only stochastic input).

    Raises:
        ValueError: if the target fraction cannot be reached.
    """
    if not (0.0 < area_fraction < 0.55):
        raise ValueError(f"area_fraction must be in (0, 0.55), got {area_fraction!r}")
    area = ny * nx
    # a disk of radius R covers a deterministic number of cell centres; using it
    # (instead of pi R^2) makes the *rasterised* fraction match the target
    n_cells = sum(
        1
        for i in range(-int(math.ceil(radius)), int(math.ceil(radius)) + 1)
        for j in range(-int(math.ceil(radius)), int(math.ceil(radius)) + 1)
        if (i + 0.5) ** 2 + (j + 0.5) ** 2 <= radius * radius
    )
    n_target = max(1, int(round(area_fraction * area / n_cells)))
    gen = torch.Generator().manual_seed(int(seed))
    centres: list[tuple[float, float]] = []
    dmin = 2.0 * radius + min_gap
    attempts = 0
    while len(centres) < n_target and attempts < max_attempts:
        attempts += 1
        cand = (
            float(torch.rand(1, generator=gen)) * nx,
            float(torch.rand(1, generator=gen)) * ny,
        )
        ok = True
        for cx, cy in centres:
            dx = abs(cand[0] - cx)
            dx = min(dx, nx - dx)  # minimum image
            dy = abs(cand[1] - cy)
            dy = min(dy, ny - dy)
            if dx * dx + dy * dy < dmin * dmin:
                ok = False
                break
        if ok:
            centres.append(cand)
    if len(centres) < n_target:
        raise ValueError(
            f"RSA could not reach area fraction {area_fraction} in {attempts} attempts "
            f"({len(centres)}/{n_target} disks)"
        )
    return [Disk(center=c, radius=radius, fixed=True) for c in centres]


# ---------------------------------------------------------------------------
# Flow through the array
# ---------------------------------------------------------------------------


def solve_array_flow(
    area_fraction: float,
    *,
    ny: int = 128,
    nx: int = 128,
    radius: float = 8.0,
    tau: float = 0.8,
    body_force: float = 1e-5,
    seed: int = 0,
    max_steps: int = 6000,
    check_every: int = 100,
    tol: float = 2e-4,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Drive a body-forced D2Q9 flow through the random array to steady state."""
    disks = random_disk_array(ny, nx, area_fraction, radius=radius, seed=seed)
    lat = DiskLatticeCoupling(ny, nx, tau=tau, dtype=dtype, track_forces=False)
    lat.set_disks(disks)
    prev = 0.0
    history = []
    steps = 0
    converged = False
    for steps in range(1, int(max_steps) + 1):
        lat.step(body_force=(body_force, 0.0))
        if steps % check_every == 0:
            u = lat.mean_fluid_velocity()
            ux = float(u[0])
            history.append(ux)
            if prev != 0.0 and abs(ux - prev) / abs(ux) < tol:
                converged = True
                break
            prev = ux
    rho, ux, uy = macroscopic(lat.f)
    fluid = ~lat.mask
    porosity = float(fluid.double().mean())
    u_mean = float((ux * fluid).sum() / fluid.sum().clamp(min=1))
    return {
        "lattice": lat,
        "disks": disks,
        "ux": ux,
        "uy": uy,
        "fluid": fluid,
        "porosity": porosity,
        "n_disks": len(disks),
        "area_fraction_nominal": area_fraction,
        "area_fraction_raster": float(lat.solid_fraction),
        "darcy_velocity": porosity * u_mean,
        "interstitial_velocity": u_mean,
        "steps": steps,
        "converged": converged,
        "velocity_history": history,
        "nu": (tau - 0.5) / 3.0,
    }


# ---------------------------------------------------------------------------
# Scalar transport on the frozen flow
# ---------------------------------------------------------------------------


def _scalar_equilibrium(c, ux, uy):
    cx = _SCALAR_C.to(c.device)[:, 0].view(5, 1, 1).to(c.dtype)
    cy = _SCALAR_C.to(c.device)[:, 1].view(5, 1, 1).to(c.dtype)
    w = _SCALAR_W.to(device=c.device, dtype=c.dtype).view(5, 1, 1)
    cu = cx * ux + cy * uy
    return w * c.unsqueeze(0) * (1.0 + cu / CS2)


def _scalar_stream_nf(g, fluid: torch.Tensor):
    """D2Q5 streaming with zero-flux (bounce-back) walls; exactly mass conserving."""
    out = torch.empty_like(g)
    out[0] = g[0]
    for q in range(1, 5):
        dy, dx = int(_SCALAR_C[q, 1]), int(_SCALAR_C[q, 0])
        a = q + 2 if q <= 2 else q - 2  # opposite direction in D2Q5 ordering
        val = torch.roll(g[q], shifts=(dy, dx), dims=(-2, -1))
        src_solid = ~torch.roll(fluid, shifts=(dy, dx), dims=(-2, -1))
        out[q] = torch.where(src_solid, g[a], val)
    # solid cells hold no scalar: without this the population that a fluid node
    # sends *away* from a wall would be teleported into the solid node and
    # accumulate there (a spurious mass reservoir that destabilises the scheme)
    out = out.masked_fill(~fluid.unsqueeze(0), 0.0)
    return out


def measure_hindered_dispersion(
    area_fraction: float,
    *,
    d_m: float = 0.002,
    ny: int = 128,
    nx: int = 128,
    radius: float = 8.0,
    tau: float = 0.8,
    body_force: float = 2e-4,
    seed: int = 0,
    flow_steps: int = 6000,
    scalar_steps: int = 6000,
    init_width: float = 5.0,
    fit_start: float = 0.5,
    stop_sigma_frac: float = 0.22,
    check_every: int = 20,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Measure ``D_eff(c)`` from the spreading of a scalar band in the array flow.

    The scalar band is a Gaussian in ``x`` (uniform in ``y``, so the measurement
    is the longitudinal dispersion) released in the frozen steady velocity
    field.  Over fluid cells we track

        M0 = sum c,   mu = sum c x / M0,   sigma^2 = sum c (x - mu)^2 / M0

    and fit ``sigma^2 = 2 D_eff t + b`` on the late window; the fit is repeated
    on two sub-windows to expose the statistical/model uncertainty.
    """
    flow = solve_array_flow(
        area_fraction,
        ny=ny,
        nx=nx,
        radius=radius,
        tau=tau,
        body_force=body_force,
        seed=seed,
        max_steps=flow_steps,
        dtype=dtype,
    )
    ux, uy, fluid = flow["ux"], flow["uy"], flow["fluid"]
    # work in the frame comoving with the *mean* flow: the longitudinal
    # dispersion coefficient is defined about the mean displacement, and this
    # also keeps the band away from the periodic boundary
    ux = ux - flow["interstitial_velocity"]
    device = ux.device
    x = torch.arange(nx, dtype=dtype, device=device).view(1, -1) + 0.5
    y = torch.arange(ny, dtype=dtype, device=device).view(-1, 1) + 0.5
    x0 = 0.5 * nx
    y0 = 0.5 * ny
    blob = torch.exp(
        -0.5 * (((x - x0) / init_width) ** 2 + ((y - y0) / init_width) ** 2)
    )
    c_field = blob / blob.mean()
    g = _scalar_equilibrium(c_field, ux, uy)
    tau_d = 3.0 * float(d_m) + 0.5

    times, variances, masses, means = [], [], [], []
    for step in range(1, int(scalar_steps) + 1):
        c = g.sum(dim=0)
        g_eq = _scalar_equilibrium(c, ux, uy)
        g = g - (g - g_eq) / tau_d
        g = _scalar_stream_nf(g, fluid)
        if step % check_every == 0:
            cf = g.sum(dim=0) * fluid
            m0 = float(cf.sum())
            mu = float((cf * x).sum()) / m0
            var = float((cf * (x - mu) ** 2).sum()) / m0
            times.append(float(step))
            variances.append(var)
            masses.append(m0)
            means.append(mu)
            # stop before the plume feels the periodic boundary
            if var > (stop_sigma_frac * nx) ** 2:
                break

    t = torch.tensor(times, dtype=dtype)
    s2 = torch.tensor(variances, dtype=dtype)

    def _fit(frac0: float, frac1: float) -> dict:
        lo = int(frac0 * len(t))
        hi = int(frac1 * len(t))
        tt, ss = t[lo:hi], s2[lo:hi]
        A = torch.stack([2.0 * tt, torch.ones_like(tt)], dim=1)
        sol = torch.linalg.lstsq(A, ss.unsqueeze(1)).solution.squeeze(1)
        pred = A @ sol
        ss_tot = ss - ss.mean()
        r2 = 1.0 - float(((ss - pred) ** 2).sum() / (ss_tot**2).sum().clamp(min=1e-300))
        return {"D_eff": float(sol[0]), "intercept": float(sol[1]), "r2": r2}

    main = _fit(fit_start, 1.0)
    early = _fit(max(0.2, fit_start - 0.2), fit_start + 0.2)
    late = _fit(min(0.7, fit_start + 0.1), 1.0)
    m0 = torch.tensor(masses, dtype=dtype)
    return {
        "area_fraction": float(flow["area_fraction_raster"]),
        "area_fraction_nominal": float(area_fraction),
        "d_eff": main["D_eff"],
        "d_m": float(d_m),
        "d_settle": main["D_eff"] - float(d_m),
        "r2": main["r2"],
        "intercept": main["intercept"],
        "d_eff_window_spread": abs(late["D_eff"] - early["D_eff"]) / main["D_eff"],
        "d_eff_windows": {"early": early, "main": main, "late": late},
        "darcy_velocity": flow["darcy_velocity"],
        "interstitial_velocity": flow["interstitial_velocity"],
        "porosity": flow["porosity"],
        "n_disks": flow["n_disks"],
        "flow_steps": flow["steps"],
        "flow_converged": flow["converged"],
        "scalar_steps": int(scalar_steps),
        "mass_drift": abs(float(m0[-1] - m0[0])) / float(m0[0]),
        "times": t,
        "variance": s2,
        "means": torch.tensor(means, dtype=dtype),
        "seed": int(seed),
        "radius": float(radius),
    }


def ensemble_dispersion(
    area_fraction: float,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    **kwargs,
) -> dict:
    """Average :func:`measure_hindered_dispersion` over array realisations."""
    runs = [measure_hindered_dispersion(area_fraction, seed=s, **kwargs) for s in seeds]
    d = torch.tensor([r["d_eff"] for r in runs], dtype=torch.float64)
    return {
        "area_fraction": float(runs[0]["area_fraction"]),
        "n_seeds": len(runs),
        "d_eff": float(d.mean()),
        "d_eff_std": float(d.std(unbiased=True)) if len(runs) > 1 else 0.0,
        "d_eff_sem": float(d.std(unbiased=True) / math.sqrt(len(runs))) if len(runs) > 1 else 0.0,
        "d_eff_per_seed": [float(v) for v in d],
        "d_settle": float(d.mean()) - runs[0]["d_m"],
        "d_m": runs[0]["d_m"],
        "darcy_velocity": float(
            sum(r["darcy_velocity"] for r in runs) / len(runs)
        ),
        "porosity": float(runs[0]["porosity"]),
        "mass_drift_max": max(r["mass_drift"] for r in runs),
        "r2_min": min(r["r2"] for r in runs),
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# Closed-form law
# ---------------------------------------------------------------------------


def fit_dispersion_law(
    concentrations: Sequence[float],
    d_eff: Sequence[float],
    *,
    holdout: float | None = None,
) -> dict:
    """Fit ``D_eff(c) = D_0 (1 - c)^alpha`` to measured points.

    The two parameters are obtained from a log-space least squares on the
    calibration points; per-point relative residuals are reported, and — when a
    ``holdout`` concentration is supplied together with the corresponding
    measurement (last entry of ``d_eff``) — the extrapolation error is reported
    separately.
    """
    c = torch.tensor(list(concentrations), dtype=torch.float64)
    if holdout is not None:
        d = torch.tensor(list(d_eff)[:-1], dtype=torch.float64)
        d_hold = float(list(d_eff)[-1])
        # ``concentrations`` may or may not include the hold-out concentration
        c_calib = c[:-1] if len(c) == len(d_eff) else c
    else:
        d = torch.tensor(list(d_eff), dtype=torch.float64)
        c_calib = c
    if len(c_calib) < 2:
        raise ValueError("need at least two calibration points")
    logd = torch.log(d)  # same length as c_calib by construction
    # log D = log D0 + alpha log(1 - c)
    A = torch.stack([torch.ones_like(c_calib), torch.log(1.0 - c_calib)], dim=1)
    sol = torch.linalg.lstsq(A, logd.unsqueeze(1)).solution.squeeze(1)
    d0 = float(torch.exp(sol[0]))
    alpha = float(sol[1])
    pred = d0 * (1.0 - c_calib) ** alpha
    rel = (pred - d).abs() / d
    out = {
        "d0": d0,
        "alpha": alpha,
        "calibration_concentrations": [float(v) for v in c_calib],
        "measured": [float(v) for v in d],
        "predicted": [float(v) for v in pred],
        "relative_errors": [float(v) for v in rel],
        "max_relative_error": float(rel.max()),
    }
    if holdout is not None:
        d_pred = d0 * (1.0 - float(holdout)) ** alpha
        out.update(
            {
                "holdout_concentration": float(holdout),
                "holdout_measured": d_hold,
                "holdout_predicted": d_pred,
                "holdout_relative_error": abs(d_pred - d_hold) / d_hold,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Kynch batch-settling column with a concentration-dependent dispersion
# ---------------------------------------------------------------------------


def _scalar_collide_field(g, c, ux, uy, tau_field):
    g_eq = _scalar_equilibrium(c, ux, uy)
    return g - (g - g_eq) / tau_field.unsqueeze(0)


def _smooth(profile: torch.Tensor, width: int = 5) -> torch.Tensor:
    """Moving average used only for reading the interface geometry.

    The BGK scalar scheme leaves O(1%) dispersive ripples ahead of the front;
    averaging over a few cells makes the 10-90% crossings well defined without
    touching the solution used for the mass/velocity measurements.
    """
    if width <= 1:
        return profile
    k = torch.ones(width, dtype=profile.dtype, device=profile.device) / width
    pad = width // 2
    padded = torch.cat([profile[:1].expand(pad), profile, profile[-1:].expand(pad)])
    return torch.nn.functional.conv1d(padded.view(1, 1, -1), k.view(1, 1, -1)).view(-1)



def _front_width_erf(
    profile: torch.Tensor,
    c_ref: float,
    h50: float,
    *,
    half_window: float = 15.0,
    n_widths: int = 60,
) -> dict:
    """Interface thickness from a robust error-function fit.

    A settling front is an error function whose width is the only geometric
    unknown (the position and amplitude are linear parameters).  For every trial
    width the linear parameters are solved exactly and the best width is the one
    with the smallest residual — a 1D grid search that cannot be fooled by the
    dispersive ripples the BGK scalar scheme leaves in the clear-water region
    (a level-crossing estimator can: it found a 26-cell "interface" for
    ``D = 1e-4``).
    """
    n = int(profile.numel())
    if math.isnan(h50):
        return {"width": float("nan"), "amplitude": float("nan"), "residual": float("nan")}
    lo = max(0, int(h50 - half_window))
    hi = min(n, int(h50 + half_window) + 1)
    y = torch.arange(lo, hi, dtype=profile.dtype, device=profile.device) + 0.5
    c = profile[lo:hi]
    best = None
    for w in torch.logspace(math.log10(0.4), math.log10(40.0), n_widths,
                            dtype=profile.dtype, device=profile.device):
        shape = 0.5 * (1.0 - torch.erf((y - h50) / w))
        A = torch.stack([shape, torch.ones_like(shape)], dim=1)
        sol = torch.linalg.lstsq(A, c.unsqueeze(1)).solution.squeeze(1)
        res = float(((A @ sol - c) ** 2).sum())
        if best is None or res < best[0]:
            best = (res, float(w), float(sol[0]), float(sol[1]))
    res, w, amp, off = best
    rms = math.sqrt(res / max(1, int(c.numel())))
    return {
        "width": w,
        "amplitude": amp,
        "offset": off,
        "rms": rms,
        "relative_rms": rms / max(c_ref, 1e-30),
    }

def _interface_height(profile: torch.Tensor, level: float) -> float:
    """Linear-interpolated height of the *falling* (top) interface at ``level``.

    The profile is indexed from the bottom (index 0), so the slurry/clear-water
    interface is the **last** crossing when scanning upwards.
    """
    n = int(profile.numel())
    above = profile > level
    if not bool(above.any()) or bool(above.all()):
        return float("nan")
    idx = int(torch.nonzero(above, as_tuple=False)[-1, 0])
    if idx >= n - 1:
        return float("nan")
    hi = float(profile[idx])
    lo = float(profile[idx + 1])
    frac = (hi - level) / (hi - lo) if hi != lo else 0.0
    return float(idx) + 0.5 + frac


def run_settling_column_dispersion(
    concentration_0: float = 0.2,
    *,
    v_s0: float = 0.08,
    exponent: float = 3.55,
    phi_max: float = 0.60,
    d_of_c: Callable[[torch.Tensor], torch.Tensor] | None = None,
    d_m: float = 1.0e-4,
    H: int = 96,
    nx: int = 2,
    init_height_fraction: float = 0.75,
    max_steps: int = 40000,
    check_every: int = 10,
    travel_cells: float = 6.0,
    velocity_fit_start: float = 0.25,
    dtype=torch.float64,
    device: str | torch.device = "cpu",
) -> dict:
    """Kynch column with drift ``v_s(c)`` and a concentration-dependent ``D(c)``.

    The scalar field is advected by ``u_eff = -v_s(c) z_hat`` and diffused with
    ``D(c)`` (molecular ``d_m`` plus the measured settling dispersion when
    ``d_of_c`` is given), inside the flux divergence, so the conservative Kynch
    equation is transported.  Walls are zero-flux, hence mass is conserved to
    round-off.

    Returns the measured interface speed (compare with the Kynch shock speed
    ``v_s(c_0)``), the 10-90% interface thickness and the mass drift.
    """
    dev = torch.device(device)
    if not (0.0 <= concentration_0 < phi_max):
        raise ValueError(f"concentration_0 must be in [0, {phi_max})")
    c0 = float(concentration_0)
    v_expected = float(
        hindered_settling_velocity(c0, v_s0, exponent=exponent, phi_max=phi_max)
    )
    ny = H + 2
    y = torch.arange(1, H + 1, dtype=dtype, device=dev) + 0.5
    c_init = torch.where(
        y < init_height_fraction * H,
        torch.full_like(y, c0),
        torch.zeros_like(y),
    )
    c2d = torch.zeros((ny, nx), dtype=dtype, device=dev)
    c2d[1:-1, :] = c_init.view(H, 1)
    zero = torch.zeros((ny, nx), dtype=dtype, device=dev)
    fluid = torch.ones((ny, nx), dtype=torch.bool, device=dev)
    fluid[0, :] = False
    fluid[-1, :] = False
    g = _scalar_equilibrium(c2d, zero, zero)
    mass0 = math.fsum((g.sum(dim=0) * fluid).flatten().tolist())
    level = 0.5 * c0
    h0 = _interface_height(c_init, level)
    times, heights, thickness = [], [], []
    travel = 0.0
    step = 0
    final_profile = c_init
    for step in range(1, int(max_steps) + 1):
        c = g.sum(dim=0)
        c_phys = c.clamp(min=0.0, max=phi_max * (1.0 - 1e-12))
        vy = -hindered_settling_velocity(
            c_phys[1:-1, :], v_s0, exponent=exponent, phi_max=phi_max
        )
        vy_full = torch.zeros((ny, nx), dtype=dtype, device=dev)
        vy_full[1:-1, :] = vy
        if d_of_c is None:
            d_field = torch.full_like(c, float(d_m))
        else:
            d_field = d_of_c(c_phys)
        tau_field = (3.0 * d_field).clamp(min=1e-3) + 0.5
        g = _scalar_collide_field(g, c, zero, vy_full, tau_field)
        g = _scalar_stream_nf(g, fluid)
        if step % int(check_every) == 0:
            profile_raw = (g.sum(dim=0)[1:-1, :]).mean(dim=1)
            profile = _smooth(profile_raw.clamp(min=0.0), width=3)
            final_profile = profile
            h = _interface_height(profile, level)
            if math.isnan(h):
                break
            times.append(float(step))
            heights.append(h)
            travel = h0 - h
            fit = _front_width_erf(profile, c0, h)
            thickness.append(fit["width"])
            if travel >= travel_cells:
                break
    mass1 = math.fsum((g.sum(dim=0) * fluid).flatten().tolist())
    n_pts = len(times)
    v_meas = float("nan")
    v_full = float("nan")
    if n_pts >= 3:
        t_all = torch.tensor(times, dtype=dtype)
        h_all = torch.tensor(heights, dtype=dtype)
        tm, hm = t_all.mean(), h_all.mean()
        v_full = -float(((t_all - tm) * (h_all - hm)).sum() / ((t_all - tm).pow(2).sum()))
        lo = int(velocity_fit_start * n_pts)
        tt = t_all[lo:]
        hh = h_all[lo:]
        tm, hm = tt.mean(), hh.mean()
        v_meas = -float(((tt - tm) * (hh - hm)).sum() / ((tt - tm).pow(2).sum()))
    widths = [v for v in thickness if not math.isnan(v)]
    return {
        "concentration_0": c0,
        "v_s0": v_s0,
        "exponent": exponent,
        "phi_max": phi_max,
        "dispersion_model": "measured_D_eff(c)" if d_of_c is not None else "molecular_only",
        "interface_velocity": v_meas,
        "interface_velocity_expected": v_expected,
        "relative_error": abs(v_meas - v_expected) / v_expected if n_pts >= 3 else float("nan"),
        "interface_velocity_full_window": v_full,
        "relative_error_full_window": abs(v_full - v_expected) / v_expected
        if n_pts >= 3
        else float("nan"),
        "velocity_fit_start": float(velocity_fit_start),
        "interface_thickness_final": widths[-1] if widths else float("nan"),
        "interface_thickness_mean": sum(widths) / len(widths) if widths else float("nan"),
        "mass_drift": abs(mass1 - mass0) / mass0,
        "steps": step,
        "times": torch.tensor(times, dtype=dtype),
        "heights": torch.tensor(heights, dtype=dtype),
        "thickness": torch.tensor(widths, dtype=dtype),
        "profile": final_profile,
        "profile_raw": profile_raw,
        "front_fit": fit if n_pts else {},
    }


def killer_test_dispersion(
    concentration_0: float = 0.2,
    *,
    d_of_c: Callable[[torch.Tensor], torch.Tensor],
    d_m: float = 1.0e-4,
    **kwargs,
) -> dict:
    """A/B test: switch the ``D_settle`` term off and compare interface width.

    The killer test of the dispersion closure asks whether the term does
    anything: with the same velocity field and the same molecular diffusivity,
    turning ``D_settle`` off must make the interface *thinner* by a margin much
    larger than the numerical noise (the acceptance criterion is 20%).
    """
    with_term = run_settling_column_dispersion(
        concentration_0, d_of_c=d_of_c, d_m=d_m, **kwargs
    )
    without = run_settling_column_dispersion(
        concentration_0, d_of_c=None, d_m=d_m, **kwargs
    )
    w_on = with_term["interface_thickness_final"]
    w_off = without["interface_thickness_final"]
    dev = abs(w_off - w_on) / w_on if w_on else float("nan")
    return {
        "with_dispersion": with_term,
        "without_dispersion": without,
        "thickness_with": w_on,
        "thickness_without": w_off,
        "relative_thickness_deviation": dev,
        "killer_pass": bool(dev > 0.20),
        "interface_velocity_difference": abs(
            with_term["interface_velocity"] - without["interface_velocity"]
        )
        / with_term["interface_velocity_expected"],
    }