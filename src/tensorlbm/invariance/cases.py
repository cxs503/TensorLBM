"""Case studies for the mesh-orientation / Galilean invariance certificate.

Four physical problems, all with an exact or reference answer:

``periodic_channel``
    Force-driven plane channel with the walls at *arbitrary* sub-cell
    positions, fully periodic in the streamwise direction.  This is the
    textbook lattice-Boltzmann resolution test: no inflow/outflow, no pressure
    gradient, exact parabolic solution, so the observed order isolates the
    *wall treatment* (criterion C1a/C1b and the negative control C6).
``poiseuille``
    The same parabola rotated by ``theta = 0/15/30/45°`` relative to the
    lattice.  An unbounded inclined strip cannot be periodic on a rectangular
    lattice, so the ends are closed with velocity Dirichlet faces whose data is
    the exact developed profile and the wall shear is balanced by a body force
    (zero pressure gradient, ``drive="body_force"``).  ``drive="pressure"``
    reproduces the historical velocity-driven variant, which carries a
    streamwise density gradient and is kept only as a diagnosed reference.
``cylinder``
    Flow past a circular cylinder at ``Re = 100 / 200`` in a doubly periodic
    box, driven along an axis at angle ``theta`` to the lattice.  Drag comes
    from the exact discrete momentum budget (mean body force x fluid area),
    Strouhal number from a wake probe.
``taylor_green``
    Decaying Taylor-Green vortex, optionally superposed with a uniform
    translation ``U`` (Galilean boost).  Reports the decay-rate deviation and
    the superposition defect (criteria C4/C5).

Every function returns a plain ``dict`` so that the driver can serialise it
without any hidden state; no randomness is used anywhere (criterion C7).
"""

from __future__ import annotations

import math

import numpy as np

from .diagnostics import (
    band_errors,
    density_variation,
    effective_wall_offset,
    wall_slip,
)
from .geometry import channel_boundary_cells, circle, poiseuille_profile, tilted_channel
from .solver import D2Q9Solver, VelocityDirichlet

CS2 = 1.0 / 3.0
CS = math.sqrt(CS2)


# --------------------------------------------------------------- case 1: channel
class PeriodicChannel:
    """Plane channel of width ``width`` periodic in ``x``, walls at ``y0``/``y1``.

    ``offset`` shifts both walls (and the channel axis) by a fraction of a
    lattice spacing, i.e. it moves the wall away from the lattice midpoint.
    The signed distance function is analytic, exactly as for the inclined
    channel, so the same interpolated bounce-back is exercised.
    """

    def __init__(self, nx: int, ny: int, width: float, offset: float = 0.0,
                 center: tuple[float, float] | None = None) -> None:
        self.nx, self.ny = int(nx), int(ny)
        if center is None:
            center = (0.5 * self.nx, 0.5 * self.ny)
        cx, cy = center
        self.y0 = cy - 0.5 * width + offset
        self.y1 = cy + 0.5 * width + offset
        Y = (np.arange(self.ny) + 0.5)[:, None]
        dl = Y - self.y0
        du = self.y1 - Y
        self.sdf = np.broadcast_to(np.minimum(dl, du), (self.ny, self.nx)).copy()
        gy = np.where(dl < du, 1.0, -1.0)
        self.gy = np.broadcast_to(gy, (self.ny, self.nx)).copy()
        self.gx = np.zeros((self.ny, self.nx))
        self.solid = self.sdf <= 0.0
        self.theta_rad = 0.0
        self.axis = (1.0, 0.0)
        self.normal = (0.0, 1.0)
        self.meta = {"kind": "periodic_channel", "width": float(width),
                     "offset": float(offset), "center": (cx, cy),
                     "y0": self.y0, "y1": self.y1}

    @property
    def fluid(self) -> np.ndarray:
        return ~self.solid


def _body_force(width: float, nu: float, u_mean: float) -> float:
    """Force density that drives ``u_mean`` through a channel of width ``width``."""
    return 12.0 * nu * u_mean / (width * width)


def _parabolic_profile(geom, u_mean: float, nu: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-pressure-gradient Poiseuille field for a strip/wall geometry."""
    width = float(geom.meta["width"])
    G = _body_force(width, nu, u_mean)
    d = geom.sdf
    u = (G / (2.0 * nu)) * d * (width - d)
    zeros = np.zeros_like(u)
    return u.copy(), zeros


def periodic_channel_case(
    width: float,
    *,
    offset: float = 0.0,
    u_mean: float = 0.01,
    nu: float = 0.1,
    bounce_back: str = "interpolated",
    collision: str = "bgk",
    steps_factor: float = 20.0,
    nx: int = 8,
    dtype=np.float64,
) -> dict:
    """One resolution of the force-driven periodic-channel resolution test.

    ``offset`` is the sub-cell position of the walls relative to the lattice
    nodes (``0`` = walls exactly half-way between rows).  ``bounce_back=
    "halfway"`` freezes ``q = 0.5`` and is the negative control: it is blind to
    the sub-cell wall position by construction.
    """
    ny = int(round(width)) + 8
    geom = PeriodicChannel(nx, ny, width, offset)
    tau = 0.5 + 3.0 * nu
    solver = D2Q9Solver(geom, tau=tau, bounce_back=bounce_back, collision=collision,
                        dtype=dtype)
    ux_ex, uy_ex = _parabolic_profile(geom, u_mean, nu)
    solver.initialise(ux_ex, np.zeros_like(ux_ex))
    G = _body_force(width, nu, u_mean)
    steps = int(steps_factor * width * width)
    for _ in range(steps):
        solver.step(G, 0.0)
    rho, ux, uy = solver.macroscopic()
    err = np.hypot(ux - ux_ex, uy)
    m = geom.fluid
    l2 = float(np.sqrt(np.mean(err[m] ** 2)))
    slip = wall_slip(geom, ux, uy, ux_ex, uy_ex, trim=0.0)
    delta = effective_wall_offset(geom, slip["mean_slip"], u_mean, nu)
    dudn = _body_force(width, nu, u_mean) * width / (2.0 * nu)
    return {
        "case": "periodic_channel",
        "width": float(width),
        "offset": float(offset),
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "nu": nu,
        "u_mean": u_mean,
        "steps": steps,
        "bounce_back": bounce_back,
        "collision": collision,
        "l2_error": l2,
        "linf_error": float(err[m].max()),
        "rel_l2": l2 / (1.5 * u_mean),
        "wall_slip": slip["mean_slip"],
        "wall_slip_rel": slip["mean_slip"] / (1.5 * u_mean),
        "wall_offset_cells": delta,
        "wall_offset_rel": abs(delta) / float(width),
        "wall_shear_theory": dudn,
        "density": density_variation(rho, m),
        "bands": band_errors(geom, ux, uy, ux_ex, uy_ex, trim=0.0),
    }


def poiseuille_case(
    theta_deg: float,
    width: int,
    *,
    u_mean: float = 0.01,
    nu: float = 0.1,
    bounce_back: str = "interpolated",
    collision: str = "bgk",
    steps_factor: float = 30.0,
    trim: float = 0.25,
    drive: str = "body_force",
    dtype=None,
) -> dict:
    """One resolution of the inclined-channel Poiseuille verification.

    ``drive="body_force"`` adds the force that balances the wall shear, so the
    exact solution has a *uniform density* (zero pressure gradient) and the
    measured error belongs to the wall treatment.  ``drive="pressure"`` drops
    the force (the parabola is then sustained by a streamwise pressure gradient
    which, in a weakly compressible lattice gas, drags a density gradient
    ``Delta rho ~ 12 nu u_mean/H`` and an unrelated velocity deviation along
    with it); it is retained as a diagnosed reference.
    """
    if drive not in ("body_force", "pressure"):
        raise ValueError(f"unknown drive {drive!r}")
    if dtype is None:
        dtype = np.float64
    th = math.radians(theta_deg)
    nx = int(round(6 * width))
    ny = int(round(nx * math.tan(th) + width / math.cos(th) + 2 * width)) + 2
    geom = tilted_channel(nx, ny, theta_deg, width)
    tau = 0.5 + 3.0 * nu
    solver = D2Q9Solver(geom, tau=tau, dtype=dtype, bounce_back=bounce_back,
                        collision=collision)
    ux_ex, uy_ex = poiseuille_profile(geom, u_mean)
    solver.initialise(ux_ex, uy_ex)
    inlet, outlet = channel_boundary_cells(geom)
    solver.set_velocity_dirichlet([
        VelocityDirichlet(inlet, ux_ex, uy_ex, (0, 1)),
        VelocityDirichlet(outlet, ux_ex, uy_ex, (0, -1)),
    ])
    ax, ay = geom.axis
    G = _body_force(width, nu, u_mean) if drive == "body_force" else 0.0
    fx = np.full((ny, nx), G * ax, dtype=dtype)
    fy = np.full((ny, nx), G * ay, dtype=dtype)
    steps = int(steps_factor * width * width)
    for _ in range(steps):
        solver.step(fx, fy)
    rho, ux, uy = solver.macroscopic()
    mask = geom.fluid.copy()
    k = max(1, int(trim * nx))
    mask[:, :k] = False
    mask[:, -k:] = False
    ex = (ux - ux_ex)[mask]
    ey = (uy - uy_ex)[mask]
    l2 = float(np.sqrt(np.mean(ex * ex + ey * ey)))
    linf = float(np.max(np.hypot(ux - ux_ex, uy - uy_ex)[mask]))
    flux_ex = float((ux_ex * ax + uy_ex * ay)[inlet].sum())
    flux_num = float((ux * ax + uy * ay)[inlet].sum())
    slip = wall_slip(geom, ux, uy, ux_ex, uy_ex, trim=trim)
    return {
        "case": "poiseuille",
        "theta_deg": theta_deg,
        "width": width,
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "nu": nu,
        "u_mean": u_mean,
        "steps": steps,
        "bounce_back": bounce_back,
        "collision": collision,
        "drive": drive,
        "l2_error": l2,
        "linf_error": linf,
        "rel_l2": l2 / (1.5 * u_mean),
        "flux_error_rel": abs(flux_num - flux_ex) / abs(flux_ex),
        "trim_frac": trim,
        "wall_slip": slip["mean_slip"],
        "wall_offset_cells": effective_wall_offset(geom, slip["mean_slip"], u_mean, nu),
        "density": density_variation(rho, mask),
        "bands": band_errors(geom, ux, uy, ux_ex, uy_ex, trim=trim),
    }


def observed_order(errors: list[float], widths: list[float]) -> list[float]:
    """Observed convergence orders ``p = log(e_i/e_{i+1}) / log(h_i/h_{i+1})``."""
    orders = []
    for i in range(len(errors) - 1):
        a, b = errors[i], errors[i + 1]
        ha, hb = widths[i], widths[i + 1]
        if a <= 0 or b <= 0:
            orders.append(float("nan"))
        else:
            orders.append(float(math.log(a / b) / math.log(hb / ha)))
    return orders


# --------------------------------------------------------------------- cylinder
def _strouhal(t: list[float], sig: list[float], diameter: float, u_mean: float):
    """Dominant frequency of the wake signal -> Strouhal number."""
    if len(sig) < 32:
        return float("nan"), float("nan"), sig
    dt = float(np.mean(np.diff(t)))
    x = np.asarray(sig, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, d=dt)
    spec[0] = 0.0
    k = int(np.argmax(spec))
    if 0 < k < len(spec) - 1:  # parabolic peak refinement
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        delta = 0.5 * (a - c) / (a - 2 * b + c + 1e-30)
    else:
        delta = 0.0
    df = freqs[1] - freqs[0]
    f_peak = float(freqs[k] + delta * df)
    st = f_peak * diameter / u_mean
    return st, f_peak, sig


def cylinder_case(
    theta_deg: float,
    re: float,
    *,
    diameter: float = 16.0,
    domain_per_diameter: float = 12.0,
    u_target: float = 0.05,
    collision: str = "trt",
    bounce_back: str = "interpolated",
    spinup_periods: float = 20.0,
    sample_periods: float = 15.0,
    sample_every: int = 5,
    seed_amplitude: float = 0.05,
    st_guess: float = 0.165,
    initial: str = "potential",
    transient_d: float | None = None,
    sample_d: float | None = None,
    dtype=np.float32,
) -> dict:
    """Flow past a circular cylinder in a doubly periodic box, driven at angle theta.

    Two-phase driving avoids the momentum-inertia problem of the doubly
    periodic box (the drag-free response time of the box is
    ``M/(dF/du) ~ 1e4`` steps) **without** damping the shedding:

    1. *spin-up*: the body-force density is updated every step as
       ``f_n = rho (u_target - u_mean,n-1)``, which pins the bulk momentum to
       the target and brings the wake up on the convective time scale;
    2. *production*: the force is frozen at its spin-up mean and the flow
       evolves freely for ``sample_periods`` shedding periods; in the
       statistically steady state the mean drag is exactly ``|f| N_fluid``
       (discrete momentum budget of the periodic box).

    The wake is seeded by a deterministic, body-frame-localised transverse
    velocity perturbation so that the shedding phase is reproducible and
    orientation independent.
    """
    # deprecated (P170W2-N08) interface: transient/sample windows in units of
    # D/U.  They map onto shedding periods through the Strouhal guess.
    if transient_d is not None:
        spinup_periods = float(transient_d) / st_guess
    if sample_d is not None:
        sample_periods = float(sample_d) / st_guess
    th = math.radians(theta_deg)
    axis = np.array([math.cos(th), math.sin(th)])
    nrm = np.array([-math.sin(th), math.cos(th)])
    n = int(round(domain_per_diameter * diameter))
    radius = 0.5 * diameter
    geom = circle(n, n, radius, theta_deg=theta_deg)
    nu = u_target * diameter / re
    tau = 0.5 + 3.0 * nu
    solver = D2Q9Solver(geom, tau=tau, collision=collision, bounce_back=bounce_back,
                        dtype=dtype)
    fluid = geom.fluid
    n_fluid = float(fluid.sum())

    # initial field: uniform inflow (+ potential-flow correction around the disk)
    # plus a deterministic, body-frame-localised symmetry-breaking seed.  The
    # potential-flow part removes most of the impulsive-start transient that a
    # doubly periodic box would otherwise keep alive for L^2/nu steps.
    X = (np.arange(n) + 0.5)[None, :]
    Y = (np.arange(n) + 0.5)[:, None]
    cx, cy = geom.meta["center"]
    xb = (X - cx) * axis[0] + (Y - cy) * axis[1]      # streamwise body coord
    yb = (X - cx) * nrm[0] + (Y - cy) * nrm[1]        # wall-normal body coord
    r2 = np.maximum(xb * xb + yb * yb, radius * radius * 1.000001)
    uxb = np.full_like(xb, u_target)
    uyb = np.zeros_like(xb)
    if initial == "potential":
        cosf = xb / np.sqrt(r2)
        sinf = yb / np.sqrt(r2)
        ur = u_target * cosf * (1.0 - radius ** 2 / r2)
        ut = -u_target * sinf * (1.0 + radius ** 2 / r2)
        uxb = u_target + ur * cosf - ut * sinf
        uyb = ur * sinf + ut * cosf
    sigma = 0.35 * diameter
    seed = seed_amplitude * u_target * np.exp(
        -((xb - 0.8 * diameter) ** 2 + yb ** 2) / (sigma * sigma))
    ux0 = uxb + seed * nrm[0]
    uy0 = uyb + seed * nrm[1]
    solver.initialise(ux0, uy0)

    period = 1.0 / (st_guess * u_target / diameter)   # shedding period in steps
    n_trans = int(spinup_periods * period)
    n_sample = int(sample_periods * period)

    fx_field = np.zeros((n, n), dtype=dtype)
    fy_field = np.zeros((n, n), dtype=dtype)
    probe = (int(round(cy + 3.0 * diameter * axis[1] + 0.6 * diameter * nrm[1])),
             int(round(cx + 3.0 * diameter * axis[0] + 0.6 * diameter * nrm[0])))
    rho = np.ones((n, n))
    ux = ux0.copy()
    uy = uy0.copy()
    u_hist: list[float] = []
    fmag_hist: list[float] = []
    fmag = 0.0
    for it in range(n_trans):
        # the controller gain is 1 (it pins the volume-mean velocity in one
        # step), so the velocity has to be read *every* step: with a stale
        # value the loop gain becomes (update interval) and the force oscillates
        # instead of settling on the drag.
        rho, ux, uy = solver.macroscopic()
        umean_prev = float((ux * axis[0] + uy * axis[1])[fluid].mean())
        fmag = (u_target - umean_prev)
        fx_field[fluid] = fmag * axis[0]
        fy_field[fluid] = fmag * axis[1]
        solver.step(fx_field, fy_field)
        if it > 0.5 * n_trans and it % 5 == 0:
            u_hist.append(umean_prev)
            fmag_hist.append(fmag)
    f_frozen = float(np.mean(fmag_hist)) if fmag_hist else 0.0
    fx_field[fluid] = f_frozen * axis[0]
    fy_field[fluid] = f_frozen * axis[1]

    signal_t, signal_u, u_s, p_s = [], [], [], []
    rho, ux, uy = solver.macroscopic()
    for it in range(n_sample):
        solver.step(fx_field, fy_field)
        if it % sample_every == 0:
            rho, ux, uy = solver.macroscopic()
            signal_t.append(float(it))
            signal_u.append(float(ux[probe] * nrm[0] + uy[probe] * nrm[1]))
            u_s.append(float((ux * axis[0] + uy * axis[1])[fluid].mean()))
            p_s.append(float(((rho * ux) * axis[0] + (rho * uy) * axis[1])[fluid].sum()))
    u_mean = float(np.mean(u_s))
    # momentum-drift correction: F_drag = f*N_fluid - dP/dt, with dP/dt from a
    # least-squares fit of the production-window momentum (unbiased even when the
    # window is not perfectly steady)
    tt = np.arange(len(p_s), dtype=float) * sample_every
    dp_dt_fit = float(np.polyfit(tt, np.array(p_s), 1)[0]) if len(p_s) > 2 else 0.0
    f_drag = abs(f_frozen) * n_fluid - dp_dt_fit
    cd = abs(f_frozen) * n_fluid / (0.5 * diameter * u_mean ** 2)
    cd_corrected = f_drag / (0.5 * diameter * u_mean ** 2)
    st, f_peak, _ = _strouhal(signal_t, signal_u, diameter, u_mean)
    # momentum drift of the production window (must be ~0 in steady statistics)
    p0, p1 = p_s[0], p_s[-1]
    dp_dt = (p1 - p0) / max(1, len(p_s))
    return {
        "case": "cylinder",
        "theta_deg": theta_deg,
        "re": re,
        "diameter": diameter,
        "domain": [n, n],
        "domain_per_diameter": domain_per_diameter,
        "u_mean": u_mean,
        "u_target": u_target,
        "nu": nu,
        "tau": tau,
        "re_actual": u_mean * diameter / nu,
        "fmag_frozen": f_frozen,
        "fmag_spinup_mean": f_frozen,
        "cd": cd_corrected,
        "cd_from_body_force": cd,
        "cd_momentum_drift": dp_dt_fit,
        "initial": initial,
        "momentum_drift_per_sample": dp_dt,
        "momentum_drift_rel": dp_dt / (abs(f_frozen) * n_fluid + 1e-30),
        "momentum_drift_fit_rel": dp_dt_fit / (abs(f_frozen) * n_fluid + 1e-30),
        "st": st,
        "st_peak_freq": f_peak,
        "probe": list(probe),
        "collision": collision,
        "bounce_back": bounce_back,
        "steps_spinup": n_trans,
        "steps_sample": n_sample,
        "period_estimate": period,
        "cd_rms_fluctuation": float(np.std([
            f_frozen * n_fluid / (0.5 * diameter * u * u) for u in u_s if u > 0])),
        "probe_signal": [round(v, 10) for v in signal_u],
    }


# ------------------------------------------------------------------ Taylor-Green
def tg_exact(n: int, u0: float, nu: float, time: float, boost: float = 0.0,
             boost_axis: tuple[float, float] = (1.0, 0.0)):
    """Analytic boosted Taylor-Green field on an ``n x n`` periodic box."""
    k = 2.0 * math.pi / n
    x = (np.arange(n) + 0.5)[None, :]
    y = (np.arange(n) + 0.5)[:, None]
    amp = u0 * math.exp(-2.0 * nu * k * k * time)
    ax, ay = boost_axis
    ux = amp * np.sin(k * x) * np.cos(k * y) + boost * ax
    uy = -amp * np.cos(k * x) * np.sin(k * y) + boost * ay
    return np.broadcast_to(ux, (n, n)).copy(), np.broadcast_to(uy, (n, n)).copy()


class _PeriodicBox:
    """Trivial geometry wrapper: a fully periodic box with no solid cells."""

    def __init__(self, n: int, angle_deg: float = 0.0) -> None:
        self.nx = self.ny = n
        self.solid = np.zeros((n, n), dtype=bool)
        self.sdf = np.full((n, n), 1e6)
        self.gx = np.zeros((n, n))
        self.gy = np.zeros((n, n))
        th = math.radians(angle_deg)
        self.theta_rad = th
        self.axis = (math.cos(th), math.sin(th))
        self.normal = (-math.sin(th), math.cos(th))
        self.meta = {"kind": "periodic_box", "n": n, "angle_deg": angle_deg}

    @property
    def fluid(self):
        return ~self.solid


def _rotated_tg_exact(n: int, u0: float, nu: float, time: float, angle_deg: float):
    """Taylor-Green vortex whose wave vector is rotated by ``angle_deg``."""
    k = 2.0 * math.pi / n
    x = (np.arange(n) + 0.5)[None, :]
    y = (np.arange(n) + 0.5)[:, None]
    th = math.radians(angle_deg)
    # rotate coordinates into the vortex frame
    xr = math.cos(th) * x + math.sin(th) * y
    yr = -math.sin(th) * x + math.cos(th) * y
    amp = u0 * math.exp(-2.0 * nu * k * k * time)
    ur = amp * np.sin(k * xr) * np.cos(k * yr)
    vr = -amp * np.cos(k * xr) * np.sin(k * yr)
    ux = math.cos(th) * ur - math.sin(th) * vr
    uy = math.sin(th) * ur + math.cos(th) * vr
    return np.broadcast_to(ux, (n, n)).copy(), np.broadcast_to(uy, (n, n)).copy()


def taylor_green_case(
    ma: float,
    *,
    n: int = 64,
    u0: float = 0.02,
    nu: float = 0.01,
    steps: int = 6000,
    periods: int = 1,
    angle_deg: float = 0.0,
    collision: str = "bgk",
    fit_start: int = 500,
    sample_every: int = 10,
    dtype=np.float64,
) -> dict:
    """Boosted Taylor-Green decay; returns decay rate + superposition residual.

    The boost magnitude is chosen as ``U = periods * n / steps`` so that the
    vortex is translated by an exact integer number of cells in the periodic
    box at the final time; the analytic and the discrete un-boosted fields then
    coincide in space and the superposition residual can be evaluated without
    any interpolation.  ``angle_deg`` rotates the vortex wave vector relative to
    the lattice, which is the *bulk* orientation-invariance probe (no walls).
    """
    boost = periods * n / steps if ma > 0 else 0.0
    ma_actual = boost / CS if ma > 0 else 0.0
    box = _PeriodicBox(n, angle_deg)
    solver = D2Q9Solver(box, tau=0.5 + 3.0 * nu, collision=collision, dtype=dtype)
    ux0, uy0 = _rotated_tg_exact(n, u0, nu, 0.0, angle_deg) if angle_deg else tg_exact(n, u0, nu, 0.0)
    if boost:
        ax = math.cos(math.radians(angle_deg))
        ay = math.sin(math.radians(angle_deg))
        ux0 = ux0 + boost * ax
        uy0 = uy0 + boost * ay
    solver.initialise(ux0, uy0)
    ts, energies = [], []
    for step in range(1, steps + 1):
        solver.step()
        if step % sample_every == 0 and step >= fit_start:
            rho, ux, uy = solver.macroscopic()
            mx = float((rho * ux).mean())
            my = float((rho * uy).mean())
            e = 0.5 * float(np.mean((ux - mx) ** 2 + (uy - my) ** 2))
            ts.append(float(step))
            energies.append(max(e, 1e-30))
    slope, intercept = np.polyfit(ts, np.log(energies), 1)
    decay = -float(slope)                     # d ln(E)/dt = -4 nu k^2 (1 + ...)
    ux_m, uy_m = solver.velocity()
    if angle_deg:
        ux_e, uy_e = _rotated_tg_exact(n, u0, nu, float(steps), angle_deg)
        if boost:
            ax = math.cos(math.radians(angle_deg))
            ay = math.sin(math.radians(angle_deg))
            ux_e = ux_e + boost * ax
            uy_e = uy_e + boost * ay
    else:
        ux_e, uy_e = tg_exact(n, u0, nu, float(steps), boost)
    res_ax = float(np.max(np.abs(ux_m - ux_e)))
    res_ay = float(np.max(np.abs(uy_m - uy_e)))
    out = {
        "case": "taylor_green",
        "ma": ma_actual,
        "boost": boost,
        "periods": periods,
        "angle_deg": angle_deg,
        "n": n,
        "u0": u0,
        "nu": nu,
        "steps": steps,
        "decay_rate": decay,
        "theory_decay": 4.0 * nu * (2.0 * math.pi / n) ** 2,
        "residual_abs": max(res_ax, res_ay),
        "final_energy": float(energies[-1]),
        "collision": collision,
        "fit_start": fit_start,
        "times": ts[:: max(1, len(ts) // 40)],
        "energies": energies[:: max(1, len(energies) // 40)],
    }
    out["residual_relU"] = out["residual_abs"] / boost if boost > 0 else float("nan")
    out["decay_rel_deviation"] = float("nan")
    return out


def galilean_superposition_residual(ma: float, **kwargs) -> dict:
    """Defect of ``u(U) - u(0) - U``, i.e. the Galilean invariance of the *scheme*.

    Two runs (boosted and un-boosted) are advanced from the same analytic
    initial field; the boost is chosen so that the vortex has moved an integer
    number of cells at the final step, so the two velocity fields can be
    compared cell by cell without interpolation.  The defect is reported
    normalised by the boost ``U`` (a scheme-level statement), by the solution
    amplitude (a physical statement) and through the decay-rate deviation
    ``(lambda(U) - lambda(0))/lambda(0)`` used by the Ma^2 law.
    """
    n = int(kwargs.get("n", 64))
    steps = int(kwargs.get("steps", 12000))
    periods = int(kwargs.get("periods", 1))
    u0 = float(kwargs.get("u0", 0.02))
    nu = float(kwargs.get("nu", 0.01))
    angle_deg = float(kwargs.get("angle_deg", 0.0))
    collision = kwargs.get("collision", "bgk")
    fit_start = int(kwargs.get("fit_start", 500))
    sample_every = int(kwargs.get("sample_every", 20))
    dtype = kwargs.get("dtype", np.float64)
    boost = periods * n / steps if ma > 0 else 0.0
    th = math.radians(angle_deg)
    ax, ay = math.cos(th), math.sin(th)
    out = {"case": "galilean", "n": n, "steps": steps, "periods": periods,
           "u0": u0, "nu": nu, "angle_deg": angle_deg, "boost": boost,
           "ma": boost / CS if boost > 0 else 0.0}
    fields, decays, energies = [], [], []
    for b in (boost, 0.0):
        box = _PeriodicBox(n, angle_deg)
        solver = D2Q9Solver(box, tau=0.5 + 3.0 * nu, collision=collision, dtype=dtype)
        if angle_deg:
            ux0, uy0 = _rotated_tg_exact(n, u0, nu, 0.0, angle_deg)
        else:
            ux0, uy0 = tg_exact(n, u0, nu, 0.0)
        solver.initialise(ux0 + b * ax, uy0 + b * ay)
        ts, es = [], []
        for step in range(1, steps + 1):
            solver.step()
            if step % sample_every == 0 and step >= fit_start:
                rho, ux, uy = solver.macroscopic()
                mx = float((rho * ux).mean())
                my = float((rho * uy).mean())
                ts.append(float(step))
                es.append(max(0.5 * float(np.mean((ux - mx) ** 2 + (uy - my) ** 2)), 1e-30))
        slope, _ = np.polyfit(ts, np.log(es), 1)
        decays.append(-float(slope))
        energies.append(es[:: max(1, len(es) // 40)])
        fields.append(solver.velocity())
        out["times"] = ts[:: max(1, len(ts) // 40)]
    (ux_b, uy_b), (ux_0, uy_0) = fields
    d = np.hypot(ux_b - ux_0 - boost * ax, uy_b - uy_0 - boost * ay)
    out.update({
        "superposition_residual": float(np.max(d)),
        "superposition_residual_relU": float(np.max(d)) / boost if boost > 0 else float("nan"),
        "superposition_residual_rms_relU":
            float(np.sqrt(np.mean(d ** 2))) / boost if boost > 0 else float("nan"),
        "superposition_residual_rel_amplitude": float(np.max(d)) / u0,
        "decay_rate": decays[0],
        "decay_rate_base": decays[1],
        "theory_decay": 4.0 * nu * (2.0 * math.pi / n) ** 2,
        "decay_rel_deviation": float((decays[0] - decays[1]) / decays[1]),
        "energy_tail": energies[0][-1],
        "energy_base_tail": energies[1][-1],
    })
    return out


# ------------------------------------------------------- bulk orientation probe
def tg_wave_exact(n: int, u0: float, nu: float, time: float, m1: int, m2: int):
    """Decaying vortex/shear wave with an *integer* wave vector ``(m1, m2)``.

    ``psi = A cos(kx x) cos(ky y)`` with ``k = 2 pi (m1, m2) / n`` gives
    ``u = d psi/dy``, ``v = -d psi/dx``; the velocity amplitude decays as
    ``exp(-nu |k|^2 t)`` so the kinetic energy decays at ``lam = 2 nu |k|^2``.
    Integer ``m1, m2`` keep the field periodic, which is what allows the wave
    vector (and hence the lattice orientation relative to the flow) to be
    rotated by arbitrary commensurate angles: ``(1,0)`` -> 0 deg,
    ``(2,1)`` -> 26.57 deg, ``(1,1)`` -> 45 deg, ``(3,1)`` -> 18.43 deg,
    ``(1,3)`` -> 71.57 deg.
    """
    kx = 2.0 * math.pi * m1 / n
    ky = 2.0 * math.pi * m2 / n
    kmag = math.hypot(kx, ky)
    amp = u0 * math.exp(-nu * (kx * kx + ky * ky) * time)
    x = (np.arange(n) + 0.5)[None, :]
    y = (np.arange(n) + 0.5)[:, None]
    if kmag == 0.0:
        raise ValueError("wave vector must be non-zero")
    u = -amp * (ky / kmag) * np.cos(kx * x) * np.sin(ky * y)
    v = amp * (kx / kmag) * np.sin(kx * x) * np.cos(ky * y)
    return np.broadcast_to(u, (n, n)).copy(), np.broadcast_to(v, (n, n)).copy()


def bulk_orientation_case(
    m1: int,
    m2: int,
    *,
    n: int = 64,
    u0: float = 0.02,
    nu: float = 0.01,
    steps: int = 6000,
    sample_every: int = 20,
    fit_start: int = 500,
    collision: str = "bgk",
    dtype=np.float64,
) -> dict:
    """Decay-rate ratio ``lam / (2 nu |k|^2)`` of one commensurate orientation.

    No walls are involved: this isolates the *bulk* lattice anisotropy of the
    solver under a rotation of the flow relative to the lattice.
    """
    k2 = (2.0 * math.pi / n) ** 2 * (m1 * m1 + m2 * m2)
    box = _PeriodicBox(n)
    solver = D2Q9Solver(box, tau=0.5 + 3.0 * nu, collision=collision, dtype=dtype)
    ux0, uy0 = tg_wave_exact(n, u0, nu, 0.0, m1, m2)
    solver.initialise(ux0, uy0)
    ts, es = [], []
    for step in range(1, steps + 1):
        solver.step()
        if step % sample_every == 0 and step >= fit_start:
            _, ux, uy = solver.macroscopic()
            es.append(max(0.5 * float(np.mean(ux ** 2 + uy ** 2)), 1e-30))
            ts.append(float(step))
    slope, _ = np.polyfit(ts, np.log(es), 1)
    lam = -float(slope)
    theory = 2.0 * nu * k2
    ux_e, uy_e = tg_wave_exact(n, u0, nu, float(steps), m1, m2)
    ux_m, uy_m = solver.velocity()
    return {
        "case": "bulk_orientation",
        "m1": m1, "m2": m2,
        "angle_deg": float(math.degrees(math.atan2(m2, m1))),
        "n": n, "u0": u0, "nu": nu, "steps": steps,
        "k_squared": k2,
        "decay_rate": lam,
        "theory_decay": theory,
        "decay_ratio": lam / theory,
        "rel_deviation": abs(lam - theory) / theory,
        "residual_max": float(max(np.max(np.abs(ux_m - ux_e)), np.max(np.abs(uy_m - uy_e)))),
        "times": ts[:: max(1, len(ts) // 40)],
        "energies": es[:: max(1, len(es) // 40)],
    }
