"""Driven contact-line geometries and the run/measurement loop.

Two geometries are provided, both realised in the wall-bounded Shan-Chen channel
of :mod:`tensorlbm.contact_line.engine`:

``sessile``
    A droplet resting on a wettable bottom wall, driven along the wall by a
    streamwise body force ``gx``.  Once the viscous drag balances the drive the
    droplet translates at a steady speed ``U``; its front and rear contact lines
    then have the *advancing* and *receding* dynamic angles.  This is the
    geometry that yields the Cox-Voinov / Tanner scaling of the node.

``meniscus``
    A liquid plug in a plate channel whose meniscus spans the two walls.  A body
    force drives the meniscus downstream (forced imbibition); the generic
    ``x_front(t)`` history is the adjacent Washburn-type check and the second,
    geometrically independent, macroscopic length for the log-term analysis.

Measurement chain (all programmatic, no hand-copied numbers)
-----------------------------------------------------------
1. ``rho(x)`` from the distribution;
2. liquid-phase centroid by **circular mean** of the liquid mass (the channel is
   periodic in ``x``), unwrapped in time so a droplet can slide indefinitely;
3. level-set points at ``rho_m`` for the angles (:mod:`theta_fit`);
4. steady velocity from a linear fit of the unwrapped centroid over the
   measurement window, with the fit residual as its uncertainty;
5. ``Ca = mu U / sigma`` with ``mu = rho_l (tau - 1/2)/3`` (BGK) and ``sigma``
   **measured** from the Laplace law in this node (not taken from a table).

Why an *achievable* capillary-number window exists
-------------------------------------------------
The lattice surface tension of the scheme is small (``sigma ~ 0.056``) while the
lattice viscosity is ``O(0.3)``, so ``U = Ca sigma / mu`` is ``O(10^-3)`` at
``Ca = 10^-2`` and ``O(10^-5)`` at ``Ca = 10^-4`` — the low-``Ca`` end approaches
the residual (spurious) current of the pseudopotential model.  That is precisely
why the node carries an explicit **resolvability gate** (criterion C5, see
:func:`tensorlbm.contact_line.scaling.resolvability_gate`): data points whose
dynamic angle change is not resolved above the measurement uncertainty are
*reported* but excluded from the scaling fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..d2q9 import equilibrium, macroscopic
from ..multiphase import collide_sc_single_component, psi_exp
from ..solver import stream
from .engine import (
    Channel,
    WallStepper,
    meniscus_state,
    sessile_drop_state,
)
from .theta_fit import (
    AngleEstimate,
    TangentAngle,
    wall_contact_angle,
    angle_estimate,
    interface_points,
    level_fraction,
    local_tangent_angle,
)

__all__ = [
    "DriveResult",
    "StaticAngle",
    "capillary_number",
    "centroid",
    "liquid_mobility",
    "measure_contact_angles",
    "meniscus_angles",
    "measure_interface_thickness",
    "measure_surface_tension",
    "run_driven_drop",
    "run_meniscus",
    "static_contact_angle",
    "steady_velocity",
]


# ---------------------------------------------------------------------------
# Basic observables
# ---------------------------------------------------------------------------


def liquid_mobility(tau: float, rho_l: float) -> float:
    """Dynamic viscosity of the liquid in lattice units (BGK, ``nu = (tau-1/2)/3``)."""
    return float(rho_l) * (float(tau) - 0.5) / 3.0


def capillary_number(velocity: float, mu: float, sigma: float) -> float:
    """``Ca = mu U / sigma`` with all quantities in lattice units."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return abs(float(mu) * float(velocity)) / float(sigma)


def centroid(
    rho: torch.Tensor,
    solid: torch.Tensor,
    *,
    threshold: float,
    previous: float | None = None,
) -> tuple[float, float]:
    """Streamwise centroid of the dense phase in a periodic channel.

    Args:
        rho:       Density field.
        solid:     Solid mask (excluded).
        threshold: Density above which a cell counts as liquid.
        previous:  Previous (unwrapped) column; used to unwrap the periodic
                   angular mean so that the returned coordinate is continuous.

    Returns:
        ``(x_unwrapped, y_centroid)``.
    """
    mask = (rho > float(threshold)) & (~solid)
    if not bool(mask.any()):
        return float("nan"), float("nan")
    nx = rho.shape[1]
    mass = rho[mask]
    xs = torch.arange(nx, dtype=torch.float32, device=rho.device).view(1, nx).expand_as(rho)[mask]
    ys = torch.arange(rho.shape[0], dtype=torch.float32, device=rho.device).view(-1, 1).expand_as(rho)[mask]
    ang = 2.0 * math.pi * xs / nx
    c = float((mass * torch.cos(ang)).sum())
    s = float((mass * torch.sin(ang)).sum())
    phi = math.atan2(s, c)
    x = (phi % (2.0 * math.pi)) * nx / (2.0 * math.pi)  # in [0, nx)
    y = float((mass * ys).sum() / mass.sum())
    if previous is not None:
        # Unwrap: choose the branch closest to the previous sample.
        k = round((previous - x) / nx)
        x = x + k * nx
    return x, y


def steady_velocity(
    t: list[float], x: list[float], fraction: float = 0.5
) -> tuple[float, float, float]:
    """Slope of an ``x(t)`` history over its last ``fraction`` (steady window).

    Returns:
        ``(U, r2, u_stderr)`` — velocity, coefficient of determination and the
        standard error of the slope.
    """
    n = len(t)
    if n < 4:
        raise ValueError("need at least 4 samples for a velocity fit")
    k0 = int(math.floor((1.0 - float(fraction)) * n))
    tt = torch.tensor(t[k0:], dtype=torch.float64)
    xx = torch.tensor(x[k0:], dtype=torch.float64)
    tt = tt - tt.mean()
    xx = xx - xx.mean()
    sxx = float((tt * tt).sum())
    if sxx <= 0.0:
        raise ValueError("degenerate time window")
    slope = float((tt * xx).sum()) / sxx
    intercept = float(xx.mean())
    resid = xx - (slope * tt + intercept)
    dof = max(len(t) - k0 - 2, 1)
    s2 = float((resid * resid).sum()) / dof
    stderr = math.sqrt(max(s2 / sxx, 0.0))
    ss_tot = float((xx * xx).sum())
    r2 = 1.0 - float((resid * resid).sum()) / ss_tot if ss_tot > 0.0 else float("nan")
    return slope, r2, stderr


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class StaticAngle:
    """Static (``Ca -> 0``) contact angle for one wall wetting fraction."""

    wall_fraction: float
    theta_deg: float
    unc_deg: float
    unc_differential_deg: float
    rms: float
    radius_fit: float
    contact_half_width: float
    n_points: int
    to_dict_extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        d = {
            "wall_fraction": self.wall_fraction,
            "theta_deg": self.theta_deg,
            "unc_deg": self.unc_deg,
            "unc_differential_deg": self.unc_differential_deg,
            "rms": self.rms,
            "radius_fit": self.radius_fit,
            "contact_half_width": self.contact_half_width,
            "n_points": self.n_points,
        }
        d.update(self.to_dict_extra)
        return d


@dataclass
class DriveResult:
    """Time series and steady-state metrics of a driven droplet run."""

    gx: float
    wall_fraction: float
    dt_sample: int
    t: list[float] = field(default_factory=list)
    x_cm: list[float] = field(default_factory=list)
    y_cm: list[float] = field(default_factory=list)
    mass: list[float] = field(default_factory=list)
    front_x: list[float] = field(default_factory=list)
    rear_x: list[float] = field(default_factory=list)
    theta_a: list[float] = field(default_factory=list)
    theta_r: list[float] = field(default_factory=list)
    theta_a_unc: list[float] = field(default_factory=list)
    theta_r_unc: list[float] = field(default_factory=list)
    theta_a_circle: list[float] = field(default_factory=list)
    theta_r_circle: list[float] = field(default_factory=list)
    trace_t: list[float] = field(default_factory=list)
    trace_x: list[float] = field(default_factory=list)
    velocity: float = float("nan")
    velocity_stderr: float = float("nan")
    velocity_r2: float = float("nan")
    mass_drift: float = float("nan")
    extra: dict[str, object] = field(default_factory=dict)

    def angle(self, which: str, last: int | None = None) -> tuple[float, float]:
        """Mean and standard error of an angle series over its last ``last`` samples."""
        vals = self.theta_a if which == "A" else self.theta_r
        if last is not None:
            vals = vals[-int(last) :]
        if not vals:
            raise ValueError(f"no {which} angle samples")
        tt = torch.tensor(vals, dtype=torch.float64)
        mean = float(tt.mean())
        stderr = float(tt.std(unbiased=True) / math.sqrt(tt.numel())) if tt.numel() > 1 else 0.0
        return mean, stderr

    def angle_unc(self, which: str, last: int | None = None) -> float:
        """Largest reported measurement uncertainty of the angle samples."""
        vals = self.theta_a_unc if which == "A" else self.theta_r_unc
        if last is not None:
            vals = vals[-int(last) :]
        return max(vals) if vals else float("nan")

    def capillary(self, mu: float, sigma: float) -> float:
        return capillary_number(self.velocity, mu, sigma)

    def to_dict(self) -> dict[str, object]:
        return {
            "gx": self.gx,
            "wall_fraction": self.wall_fraction,
            "dt_sample": self.dt_sample,
            "velocity": self.velocity,
            "velocity_stderr": self.velocity_stderr,
            "velocity_r2": self.velocity_r2,
            "mass_drift": self.mass_drift,
            "samples": len(self.t),
            "theta_a": self.theta_a,
            "theta_r": self.theta_r,
            "theta_a_circle": self.theta_a_circle,
            "theta_r_circle": self.theta_r_circle,
            "theta_a_unc": self.theta_a_unc,
            "theta_r_unc": self.theta_r_unc,
            "t": self.t,
            "x_cm": self.x_cm,
            "front_x": self.front_x,
            "rear_x": self.rear_x,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# Surface tension (Laplace, measured in this node)
# ---------------------------------------------------------------------------


def measure_interface_thickness(
    *,
    box: tuple[int, int] = (64, 64),
    G: float = 5.0,
    tau: float = 1.0,
    rho_l: float = 1.957,
    rho_v: float = 0.1596,
    n_steps: int = 4000,
    device: str = "cpu",
) -> dict[str, float]:
    """Equivalent thickness of a planar liquid/vapour interface.

    A flat interface is relaxed in a fully periodic box and its thickness is
    measured from the profile ``rho(x)`` as the **10%-90% width** (linear
    interpolation of the two threshold crossings, in lattice units).  This is the
    length by which the Cox-Voinov log term is expected to be cut off in a
    diffuse-interface model, so it is the natural yardstick for criterion C3.

    Returns:
        Dict with ``thickness`` (10-90% width), ``rho_l_bulk``, ``rho_v_bulk``,
        ``n_steps`` and the two threshold positions.
    """
    ny, nx = int(box[0]), int(box[1])
    dev = torch.device(device)
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32, device=dev),
        torch.arange(nx, dtype=torch.float32, device=dev),
        indexing="ij",
    )
    # Two flat interfaces (periodic): x < nx/2 liquid, else vapour.
    dist = torch.abs(xx - nx / 2.0) - nx / 4.0
    rho0 = rho_v + (rho_l - rho_v) * 0.5 * (1.0 - torch.tanh(dist / 1.5))
    f = equilibrium(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    for _ in range(int(n_steps)):
        f = collide_sc_single_component(f, G, tau=tau, psi_fn=psi_exp, forcing="edm")
        f = stream(f)
    rho = macroscopic(f)[0]
    rho_x = rho.mean(dim=0)  # profile along x (averaged over the periodic y)
    lo = rho_x[: nx // 4].mean()
    hi = rho_x[nx // 2 - nx // 8 : nx // 2 + nx // 8].mean()
    r_l = float(hi) if float(hi) >= float(lo) else float(lo)
    r_v = float(lo) if float(hi) >= float(lo) else float(hi)
    span = max(r_l - r_v, 1e-12)
    norm = ((rho_x - r_v) / span).clamp(0.0, 1.0)

    def crossing(level: float) -> float:
        """First x where ``norm`` crosses ``level`` (linear interpolation)."""
        for k in range(nx - 1):
            a, b = float(norm[k]), float(norm[k + 1])
            if a < level <= b:
                return float(k) + (level - a) / max(b - a, 1e-12)
            if a >= level > b:  # descending interface (other half of the box)
                return float(k) + (a - level) / max(a - b, 1e-12)
        return float("nan")

    x10, x90 = crossing(0.1), crossing(0.9)
    thickness = abs(x90 - x10) if math.isfinite(x10) and math.isfinite(x90) else float("nan")
    return {
        "thickness": float(thickness),
        "x10": float(x10),
        "x90": float(x90),
        "rho_l_bulk": r_l,
        "rho_v_bulk": r_v,
        "n_steps": float(n_steps),
    }


def measure_surface_tension(
    *,
    radii: tuple[float, ...] = (14.0, 20.0, 26.0),
    box: int = 96,
    G: float = 5.0,
    tau: float = 1.0,
    rho_l: float = 1.957,
    rho_v: float = 0.1596,
    n_steps: int = 6000,
    device: str = "cpu",
) -> dict[str, float]:
    """Measure the lattice surface tension from the 2D Laplace law.

    A static droplet in a *periodic* box has ``dp = sigma / R`` (2D), where
    ``dp`` is obtained from the mechanical pressure of the SC scheme,
    ``p(rho) = rho cs^2 + (G_lib/2) cs^2 psi^2``  (library sign convention, see
    :mod:`tensorlbm.sc_eos_coexistence`).  ``sigma`` is the slope of ``dp`` vs
    ``1/R``.

    Returns:
        Dict with ``sigma``, ``r2``, ``intercept`` and the per-radius points.
    """
    ch = Channel(nx=box, ny=box, wall_rows=1, device=device)
    dev = torch.device(device)
    xx, yy = ch.grid(dev)
    psi_t = psi_exp(torch.tensor(rho_l, dtype=torch.float32))
    psi_l = float(psi_t)
    cs2 = 1.0 / 3.0
    points = []
    for R in radii:
        dist = torch.sqrt((xx - box / 2.0) ** 2 + (yy - box / 2.0) ** 2) - float(R)
        rho0 = rho_v + (rho_l - rho_v) * 0.5 * (1.0 - torch.tanh(dist / 1.5))
        f = equilibrium(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0))
        # periodic box: no solid cells and no wall pseudo-potential
        for _ in range(int(n_steps)):
            f = collide_sc_single_component(f, G, tau=tau, psi_fn=psi_exp, forcing="edm")
            f = stream(f)
        rho = macroscopic(f)[0]
        inside = rho[torch.sqrt((xx - box / 2.0) ** 2 + (yy - box / 2.0) ** 2) < R - 6.0]
        outside = rho[torch.sqrt((xx - box / 2.0) ** 2 + (yy - box / 2.0) ** 2) > R + 8.0]
        r_in = float(inside.mean())
        r_out = float(outside.mean())
        p_in = r_in * cs2 - 0.5 * G * cs2 * float(psi_exp(torch.tensor(r_in))) ** 2
        p_out = r_out * cs2 - 0.5 * G * cs2 * float(psi_exp(torch.tensor(r_out))) ** 2
        points.append({"R": float(R), "rho_in": r_in, "rho_out": r_out, "dp": p_in - p_out})

    inv_r = torch.tensor([1.0 / p["R"] for p in points], dtype=torch.float64)
    dp = torch.tensor([p["dp"] for p in points], dtype=torch.float64)
    A = torch.stack([inv_r, torch.ones_like(inv_r)], dim=1)
    sol = torch.linalg.lstsq(A, dp.unsqueeze(1)).solution.squeeze(1)
    sigma = float(sol[0])
    fit = A @ sol
    ss_res = float(((dp - fit) ** 2).sum())
    ss_tot = float(((dp - dp.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "sigma": sigma,
        "intercept": float(sol[1]),
        "r2": r2,
        "psi_liquid": psi_l,
        "points": points,  # type: ignore[dict-item]
    }


# ---------------------------------------------------------------------------
# Static contact angle (calibration of wall_fraction -> theta_e)
# ---------------------------------------------------------------------------


def static_contact_angle(
    *,
    wall_fraction: float,
    channel: Channel | None = None,
    radius: float = 14.0,
    G: float = 5.0,
    tau: float = 1.0,
    rho_l: float = 1.957,
    rho_v: float = 0.1596,
    relax_steps: int = 12000,
    device: str = "cpu",
    compile_step: bool = False,
    wall_field: torch.Tensor | None = None,
) -> StaticAngle:
    """Relax a sessile drop to equilibrium and measure its contact angle.

    Args:
        wall_fraction: ``psi_w / psi(rho_l)`` for a uniform wall (ignored when
                       ``wall_field`` is given).
        channel:       Channel geometry (defaults to ``160x72`` with 2 wall rows).
        radius:        Initial droplet radius.
        relax_steps:   Steps used to reach mechanical equilibrium.
        wall_field:    Optional per-cell wall pseudopotential field (used by the
                       hysteretic-wall study).

    Returns:
        :class:`StaticAngle`.
    """
    ch = channel or Channel(nx=160, ny=72, wall_rows=2, device=device)
    stepper = WallStepper(
        ch,
        G=G,
        tau=tau,
        rho_l=rho_l,
        rho_v=rho_v,
        wall_fraction=wall_fraction,
        compile_step=compile_step,
    )
    if wall_field is not None:
        stepper.set_wall_psi(wall_field)
    f, _ = sessile_drop_state(ch, radius, rho_l=rho_l, rho_v=rho_v)
    f = stepper.run(f, int(relax_steps))
    rho = stepper.density(f)
    level = level_fraction(rho_l, rho_v)
    pts = interface_points(rho, level, stepper.solid)
    est = angle_estimate(pts, ch.y_wall(), side="cap")
    try:
        tang = local_tangent_angle(pts, ch.y_wall(), side="front", centre_x=float(pts[:, 0].median()))
        tangent_deg = tang.theta_deg
        tangent_unc = tang.unc_deg
    except ValueError:  # pragma: no cover - flat/undetectable interface
        tangent_deg = float("nan")
        tangent_unc = float("nan")
    sp = stepper.spurious_current(f)
    # Wall-adjacent layer density: the ratio rho(first fluid row)/rho_l is the
    # direct diagnostic of wall depletion (a depleted layer means the liquid is
    # separated from the wall by a vapour-like film, see the campaign report).
    first_row = rho[ch.wall_rows, :]
    # Maximum over the first fluid row = the wall-adjacent layer density under the
    # droplet (where no liquid touches the wall at all the row is vapour-like and
    # the maximum simply reports that).
    wall_layer = float(first_row.max())
    interior = float(rho[stepper.solid.logical_not()][rho[stepper.solid.logical_not()] > level].mean())
    return StaticAngle(
        wall_fraction=float(wall_fraction),
        theta_deg=est.theta_deg,
        unc_deg=est.unc_deg,
        unc_differential_deg=math.hypot(est.boot_std_deg, est.window_std_deg),
        rms=est.rms,
        radius_fit=est.radius_fit,
        contact_half_width=est.contact_half_width,
        n_points=est.n_points,
        to_dict_extra={
            "spurious_max": sp["max"],
            "spurious_mean": sp["mean"],
            "theta_tangent_deg": tangent_deg,
            "theta_tangent_unc_deg": tangent_unc,
            "wall_layer_density": wall_layer,
            "wall_layer_ratio": wall_layer / rho_l,
            "interior_density": interior,
        },
    )


# ---------------------------------------------------------------------------
# Angle measurement for a running driven state
# ---------------------------------------------------------------------------


def measure_contact_angles(
    rho: torch.Tensor,
    solid: torch.Tensor,
    y_wall: float,
    *,
    level: float,
    centre_x: float,
    bootstrap: int = 24,
    windows: tuple[float, ...] = (6.0, 8.0, 10.0, 12.0),
    seed: int = 0,
) -> tuple[TangentAngle, TangentAngle, AngleEstimate, AngleEstimate, torch.Tensor]:
    """Measure the advancing (front) and receding (rear) angles of a moving drop.

    Two independent estimators are returned for every side:

    * ``local tangent`` (:class:`~tensorlbm.contact_line.theta_fit.TangentAngle`)
      — the primary estimator for a *driven* droplet, because a sliding drop is
      not a circular cap (front compressed, rear stretched);
    * ``windowed circle fit``
      (:class:`~tensorlbm.contact_line.theta_fit.AngleEstimate`) — the classical
      estimator, kept as an independent cross-check (criterion C8).

    The interface points are first re-centred on the droplet centroid so that a
    droplet straddling the periodic boundary is handled correctly.

    Returns:
        ``(front_tangent, rear_tangent, front_circle, rear_circle, points)``.
    """
    pts = interface_points(rho, level, solid)
    if pts.numel() == 0:
        raise ValueError("empty interface")
    nx = rho.shape[1]
    shift = torch.remainder(pts[:, 0] - float(centre_x) + 0.5 * nx, nx) - 0.5 * nx
    pts_s = torch.stack([shift, pts[:, 1]], dim=1)
    xc = 0.0
    front_t = local_tangent_angle(pts_s, y_wall, side="front", centre_x=xc)
    rear_t = local_tangent_angle(pts_s, y_wall, side="rear", centre_x=xc)
    front_c = angle_estimate(
        pts_s, y_wall, side="front", centre_x=xc, bootstrap=bootstrap, windows=(8.0, 10.0, 12.0, 14.0), seed=seed
    )
    rear_c = angle_estimate(
        pts_s, y_wall, side="rear", centre_x=xc, bootstrap=bootstrap, windows=(8.0, 10.0, 12.0, 14.0), seed=seed + 1
    )
    return front_t, rear_t, front_c, rear_c, pts_s


# ---------------------------------------------------------------------------
# Driven droplet
# ---------------------------------------------------------------------------


def run_driven_drop(
    *,
    gx: float,
    wall_fraction: float = 0.7,
    channel: Channel | None = None,
    radius: float = 14.0,
    G: float = 5.0,
    tau: float = 1.0,
    rho_l: float = 1.957,
    rho_v: float = 0.1596,
    n_steps: int = 24000,
    relax_steps: int = 8000,
    sample_every: int = 200,
    angle_from: float = 0.5,
    angle_every: int = 8,
    compile_step: bool = False,
    wall_model: object | None = None,
    device: str = "cpu",
) -> DriveResult:
    """Drive a sessile droplet along the wall and measure its dynamic angles.

    Args:
        gx:            Body-force acceleration (drive strength).
        wall_fraction: Uniform wall wetting ``psi_w / psi(rho_l)``.
        channel:       Geometry (default ``160x72``, 2 wall rows).
        radius:        Initial droplet radius.
        n_steps:       Number of *driven* lattice steps.
        relax_steps:   Steps of free relaxation (``gx = 0``) applied first, so
                       the droplet starts from its static equilibrium shape and
                       the transient is not confused with the steady state.
        sample_every:  Centroid sampling interval.
        angle_from:    Fraction of the run after which angles are measured.
        angle_every:   Angle measurement every ``angle_every`` centroid samples.
        wall_model:    Optional object with ``bind(stepper)`` and
                       ``update(rho, stepper)`` used to implement a heterogeneous
                       / hysteretic wall (see :mod:`tensorlbm.contact_line.hysteresis`).

    Returns:
        :class:`DriveResult` with the time series and the steady velocity.
    """
    ch = channel or Channel(nx=160, ny=72, wall_rows=2, device=device)
    stepper = WallStepper(
        ch,
        G=G,
        tau=tau,
        rho_l=rho_l,
        rho_v=rho_v,
        wall_fraction=wall_fraction,
        compile_step=compile_step,
    )
    if wall_model is not None:
        wall_model.bind(stepper)  # type: ignore[attr-defined]

    f, rho0 = sessile_drop_state(ch, radius, rho_l=rho_l, rho_v=rho_v)
    level = level_fraction(rho_l, rho_v)
    mass0 = float((rho0 * (~stepper.solid)).sum())

    static: dict[str, float] = {}
    if relax_steps > 0:
        f = stepper.run(f, int(relax_steps))
        rho_s = stepper.density(f)
        cx_s = float(centroid(rho_s, stepper.solid, threshold=level)[0])
        try:
            ft, rt, fc, rc, _ = measure_contact_angles(
                rho_s, stepper.solid, ch.y_wall(), level=level, centre_x=cx_s, bootstrap=16
            )
            static = {
                "theta_tangent": ft.theta_deg,
                "theta_tangent_unc": ft.unc_deg,
                "theta_circle": fc.theta_deg,
                "radius_fit": fc.radius_fit,
                "contact_half_width": fc.contact_half_width,
            }
        except ValueError:  # pragma: no cover
            static = {}

    res = DriveResult(gx=float(gx), wall_fraction=float(wall_fraction), dt_sample=int(sample_every))
    prev_x: float | None = None
    shift_ref = 0.0
    angle_failures: list[str] = []
    for n in range(1, int(n_steps) + 1):
        f = stepper.step(f, gx)
        if wall_model is not None and n % int(angle_every * sample_every) == 0:
            wall_model.update(stepper.density(f), stepper)  # type: ignore[attr-defined]
        if n % int(sample_every) != 0:
            continue
        rho = stepper.density(f)
        x, y = centroid(rho, stepper.solid, threshold=level, previous=prev_x)
        if prev_x is None:
            shift_ref = x
            x = 0.0
        else:
            x = x - shift_ref
        prev_x = x + shift_ref
        res.t.append(float(n))
        res.x_cm.append(float(x))
        res.y_cm.append(float(y))
        res.mass.append(float((rho * (~stepper.solid)).sum()))
        # contact-line positions relative to the centroid
        pts = interface_points(rho, level, stepper.solid)
        if pts.numel() > 0:
            nx = ch.nx
            cx_now = float((prev_x) % nx)
            shift = torch.remainder(pts[:, 0] - cx_now + 0.5 * nx, nx) - 0.5 * nx
            res.front_x.append(float(shift.max()))
            res.rear_x.append(float(shift.min()))
        else:  # pragma: no cover
            res.front_x.append(float("nan"))
            res.rear_x.append(float("nan"))

        if n >= int(angle_from * n_steps) and len(res.t) % int(angle_every) == 0:
            try:
                front, rear, front_c, rear_c, _ = measure_contact_angles(
                    rho,
                    stepper.solid,
                    ch.y_wall(),
                    level=level,
                    centre_x=float(prev_x % ch.nx),
                    seed=len(res.theta_a),
                )
            except ValueError as exc:  # pragma: no cover - degenerate state
                if len(angle_failures) < 5:
                    angle_failures.append(str(exc))
                continue
            res.theta_a.append(front.theta_deg)
            res.theta_r.append(rear.theta_deg)
            res.theta_a_unc.append(front.unc_deg)
            res.theta_r_unc.append(rear.unc_deg)
            res.theta_a_circle.append(front_c.theta_deg)
            res.theta_r_circle.append(rear_c.theta_deg)

    if len(res.t) > 4:
        u, r2, se = steady_velocity(res.t, res.x_cm, fraction=0.5)
        res.velocity = u
        res.velocity_r2 = r2
        res.velocity_stderr = se
        if res.mass:
            res.mass_drift = (res.mass[-1] - res.mass[0]) / max(abs(mass0), 1e-12)
    res.extra["angle_failures"] = angle_failures
    res.extra["static"] = static
    res.extra["spurious"] = stepper.spurious_current(f)
    res.extra["mu_liquid"] = liquid_mobility(tau, rho_l)
    res.extra["tau"] = float(tau)
    res.extra["level"] = level
    return res


# ---------------------------------------------------------------------------
# Meniscus in a plate channel
# ---------------------------------------------------------------------------


def meniscus_angles(
    rho: torch.Tensor,
    solid: torch.Tensor,
    channel: Channel,
    *,
    level: float,
    x_centre: float,
) -> dict[str, float]:
    """Dynamic contact angles of a channel meniscus at the bottom and top walls.

    Args:
        rho:      Density field.
        solid:    Solid mask.
        channel:  Channel geometry.
        level:    Mid-density level.
        x_centre: Streamwise position used to re-centre the point cloud (the
                  meniscus may have advanced past the periodic boundary).

    Returns:
        Dict with ``bottom``, ``bottom_unc``, ``top``, ``top_unc``, ``mean``.
    """
    pts = interface_points(rho, level, solid)
    if pts.numel() == 0:
        raise ValueError("empty interface")
    nx = channel.nx
    shift = torch.remainder(pts[:, 0] - float(x_centre) + 0.5 * nx, nx) - 0.5 * nx
    ps = torch.stack([shift, pts[:, 1]], dim=1)
    bottom = wall_contact_angle(ps, channel.y_wall(), below=True)
    top = wall_contact_angle(ps, channel.y_top(), below=False)
    return {
        "bottom": bottom.theta_deg,
        "bottom_unc": bottom.unc_deg,
        "top": top.theta_deg,
        "top_unc": top.unc_deg,
        "mean": 0.5 * (bottom.theta_deg + top.theta_deg),
        "mean_unc": 0.5 * math.hypot(bottom.unc_deg, top.unc_deg),
    }


def run_meniscus(
    *,
    gx: float,
    wall_fraction: float = 0.7,
    channel: Channel | None = None,
    fill_x: float | None = None,
    G: float = 5.0,
    tau: float = 1.0,
    rho_l: float = 1.957,
    rho_v: float = 0.1596,
    n_steps: int = 12000,
    sample_every: int = 100,
    compile_step: bool = False,
    device: str = "cpu",
) -> dict[str, object]:
    """Drive a meniscus along a plate channel (forced imbibition).

    The meniscus position is located per time sample as the **mid-density
    crossing on the channel centreline** (the meniscus bows, so the centreline
    is the standard reference for the bulk front).

    Returns:
        Dict with ``t``, ``x_front``, ``velocity`` (fit over the second half) and
        ``angles`` (wall contact angles at the two walls, degrees).
    """
    ch = channel or Channel(nx=200, ny=48, wall_rows=2, device=device)
    stepper = WallStepper(
        ch,
        G=G,
        tau=tau,
        rho_l=rho_l,
        rho_v=rho_v,
        wall_fraction=wall_fraction,
        compile_step=compile_step,
    )
    level = level_fraction(rho_l, rho_v)
    f, _ = meniscus_state(ch, ch.nx * 0.35 if fill_x is None else fill_x, rho_l=rho_l, rho_v=rho_v)
    t: list[float] = []
    x_front: list[float] = []
    row = ch.ny // 2
    prev: float | None = None
    for n in range(1, int(n_steps) + 1):
        f = stepper.step(f, gx)
        if n % int(sample_every) != 0:
            continue
        rho = stepper.density(f)
        line = rho[row]
        above = (line > level).nonzero()
        if above.numel() == 0:  # pragma: no cover
            continue
        xf = float(above.max())
        if prev is not None:
            # unwrap (the front advances monotonically to the right)
            while xf < prev - ch.nx / 2:
                xf += ch.nx
        prev = xf
        t.append(float(n))
        x_front.append(xf)
    vel, r2, se = steady_velocity(t, x_front, fraction=0.5) if len(t) > 4 else (float("nan"),) * 3
    rho = stepper.density(f)
    angles = meniscus_angles(rho, stepper.solid, ch, level=level, x_centre=float(x_front[-1]))
    return {
        "t": t,
        "x_front": x_front,
        "velocity": vel,
        "velocity_r2": r2,
        "velocity_stderr": se,
        "angles": angles,
        "gx": float(gx),
        "wall_fraction": float(wall_fraction),
        "channel": {"nx": ch.nx, "ny": ch.ny, "wall_rows": ch.wall_rows},
    }