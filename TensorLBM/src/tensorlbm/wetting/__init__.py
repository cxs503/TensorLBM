"""Wetting/diffuse-interface model verification for LBM.

Implements the three-anchor verification:
1. Static contact angle vs Young equation
2. Capillary number scaling (Cox-Voinov law)
3. Young-Laplace pressure difference

All measurements are programmatic (no hand-copied numbers).
"""

from __future__ import annotations

import math

import torch

from ..d2q9 import equilibrium, macroscopic
from ..multiphase import (
    RHO_L_DEFAULT,
    RHO_V_DEFAULT,
    SIGMA_DEFAULT,
    psi_exp,
    step_sc,
)

__all__ = [
    "Channel",
    "sessile_drop_initial",
    "droplet_initial",
    "measure_contact_angle",
    "measure_young_laplace",
    "wall_psi_from_contact_angle",
    "contact_angle_from_wall_psi",
    "run_static_contact",
    "run_young_laplace",
    "run_capillary_scaling",
]

# ---------------------------------------------------------------------------
# Channel geometry
# ---------------------------------------------------------------------------


class Channel:
    """Rectangular channel with no-slip walls on bottom and top.

    Periodic in x, walls at y=0 and y=ny-1.
    """

    def __init__(
        self,
        nx: int = 128,
        ny: int = 64,
        wall_rows: int = 2,
        device: str = "cpu",
    ):
        self.nx = nx
        self.ny = ny
        self.wall_rows = wall_rows
        self.device = device

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    def solid_mask(self) -> torch.Tensor:
        """Boolean (ny, nx) mask, True inside solid rows."""
        solid = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
        solid[: self.wall_rows, :] = True
        solid[self.ny - self.wall_rows :, :] = True
        return solid

    def grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (xx, yy) coordinate fields."""
        yy, xx = torch.meshgrid(
            torch.arange(self.ny, dtype=torch.float32, device=self.device),
            torch.arange(self.nx, dtype=torch.float32, device=self.device),
            indexing="ij",
        )
        return xx, yy

    def y_wall(self) -> float:
        """Height of bottom wall plane (half-way bounce-back location)."""
        return float(self.wall_rows) - 0.5


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def _tanh_profile(
    shape: tuple[int, int],
    distance: torch.Tensor,
    rho_in: float,
    rho_out: float,
    width: float = 1.5,
) -> torch.Tensor:
    """tanh two-phase profile across a signed distance field."""
    return rho_out + (rho_in - rho_out) * 0.5 * (1.0 - torch.tanh(distance / width))


def sessile_drop_initial(
    channel: Channel,
    radius: float = 15.0,
    center_x: float | None = None,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    width: float = 1.5,
) -> torch.Tensor:
    """Initialise a circular droplet resting on the bottom wall.

    Returns distribution f at rest, shape (9, ny, nx).
    """
    dev = torch.device(channel.device)
    solid = channel.solid_mask()
    xx, yy = channel.grid()
    cx = channel.nx / 2.0 if center_x is None else float(center_x)
    cy = channel.y_wall() + radius
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - radius
    rho = _tanh_profile(channel.shape, dist, rho_l, rho_v, width)
    rho = torch.where(solid, torch.full_like(rho, rho_l), rho)
    zero = torch.zeros_like(rho)
    return equilibrium(rho, zero, zero)


def droplet_initial(
    nx: int = 64,
    ny: int = 64,
    radius: float = 15.0,
    center: tuple[float, float] | None = None,
    rho_l: float = RHO_L_DEFAULT,
    rho_v: float = RHO_V_DEFAULT,
    width: float = 1.5,
    device: str = "cpu",
) -> torch.Tensor:
    """Initialise a circular droplet in a periodic box (no walls).

    Returns distribution f at rest, shape (9, ny, nx).
    """
    dev = torch.device(device)
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32, device=dev),
        torch.arange(nx, dtype=torch.float32, device=dev),
        indexing="ij",
    )
    cx, cy = (nx / 2.0, ny / 2.0) if center is None else center
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - radius
    rho = _tanh_profile((ny, nx), dist, rho_l, rho_v, width)
    zero = torch.zeros_like(rho)
    return equilibrium(rho, zero, zero)


# ---------------------------------------------------------------------------
# Wall wetting: psi_w <-> contact angle mapping
# ---------------------------------------------------------------------------

#: Bulk liquid pseudopotential for G=5, psi_exp
PSI_L = psi_exp(torch.tensor(RHO_L_DEFAULT)).item()
#: Bulk vapour pseudopotential
PSI_V = psi_exp(torch.tensor(RHO_V_DEFAULT)).item()


def wall_psi_from_contact_angle(theta_deg: float) -> float:
    """Convert target contact angle to wall pseudopotential.

    Uses the SC wetting relation:
        cos(theta) = (psi_w - psi_v) / (psi_l - psi_v)

    Inverted: psi_w = psi_v + cos(theta) * (psi_l - psi_v)

    WARNING -- this is *not* a validated angle setter.  Measured sessile-drop
    angles at the returned ``psi_w`` deviate from the requested angle by
    +40..+90 deg (2 grids x 3 target angles, CASE-MCMP-CONTACT-ANGLE).  The
    measured wall law is ~4.2x steeper in ``f_w = psi_w/psi_l`` with the neutral
    wall near ``f_w ~ 0.51`` (coarse) / ``0.56`` (fine) instead of
    ``psi_v/psi_l = 0.172``; prescribing an angle requires a same-resolution
    calibration curve (``tests/benchmarks/wetting_contact_angle/wetting_bench.py``
    ``calibration_curve`` / ``invert_calibration``).  See
    ``docs/CASE-MCMP-CONTACT-ANGLE.md`` and
    ``docs/WETTING_CONTACT_ANGLE_BENCHMARK.md``.
    """
    theta_rad = math.radians(theta_deg)
    cos_t = math.cos(theta_rad)
    return PSI_V + cos_t * (PSI_L - PSI_V)


def contact_angle_from_wall_psi(wall_psi: float) -> float:
    """Convert wall pseudopotential to expected contact angle.

    Inverse of wall_psi_from_contact_angle.
    """
    cos_t = (wall_psi - PSI_V) / (PSI_L - PSI_V)
    cos_t = max(-1.0, min(1.0, cos_t))
    return math.degrees(math.acos(cos_t))


# ---------------------------------------------------------------------------
# Contact angle measurement
# ---------------------------------------------------------------------------


def measure_contact_angle(
    rho: torch.Tensor,
    solid_mask: torch.Tensor,
    threshold: float | None = None,
) -> dict[str, float]:
    """Measure static contact angle from density field via circle fit.

    Fits a circle to the mid-density contour and computes the contact angle
    from the circle geometry: cos(theta) = z_c / R_fit.

    Returns dict with theta_deg, cos_theta, R_fit, z_c, y_wall, n_contour.
    """
    if threshold is None:
        threshold = 0.5 * (RHO_L_DEFAULT + RHO_V_DEFAULT)

    ny, nx = rho.shape
    y_wall = float(solid_mask.sum(dim=1).argmax().item())  # first solid row

    # Find contour points at threshold
    fluid = ~solid_mask
    # Get contour by finding cells near threshold
    diff = (rho - threshold).abs()
    # Use a band around the threshold
    band = diff < 0.15 * (RHO_L_DEFAULT - RHO_V_DEFAULT)
    band = band & fluid

    if not band.any():
        return {"theta_deg": float("nan"), "cos_theta": float("nan"),
                "R_fit": float("nan"), "z_c": float("nan"),
                "y_wall": y_wall, "n_contour": 0, "status": "no_contour"}

    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float64, device=rho.device),
        torch.arange(nx, dtype=torch.float64, device=rho.device),
        indexing="ij",
    )
    pts_x = xx[band].double()
    pts_y = yy[band].double()

    if len(pts_x) < 10:
        return {"theta_deg": float("nan"), "cos_theta": float("nan"),
                "R_fit": float("nan"), "z_c": float("nan"),
                "y_wall": y_wall, "n_contour": len(pts_x), "status": "too_few_points"}

    # Circle fit: minimize sum((x-xc)^2 + (y-yc)^2 - R^2)^2
    # Linearized: x^2 + y^2 = 2*xc*x + 2*yc*y + (R^2 - xc^2 - yc^2)
    A = torch.stack([2 * pts_x, 2 * pts_y, torch.ones_like(pts_x)], dim=1)
    b = pts_x ** 2 + pts_y ** 2

    # Least squares
    sol, _, _, _ = torch.linalg.lstsq(A, b)
    xc, yc, c_val = sol[0], sol[1], sol[2]
    R_fit = math.sqrt(c_val + xc ** 2 + yc ** 2)

    # Contact angle: cos(theta) = (yc - y_wall) / R_fit
    # (yc is measured from top, y_wall from top)
    z_c = yc - y_wall
    cos_theta = z_c / R_fit if R_fit > 0 else 0.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta_deg = math.degrees(math.acos(cos_theta))

    return {
        "theta_deg": theta_deg,
        "cos_theta": cos_theta,
        "R_fit": R_fit,
        "z_c": z_c,
        "y_wall": y_wall,
        "n_contour": len(pts_x),
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Young-Laplace pressure measurement
# ---------------------------------------------------------------------------


def measure_young_laplace(
    rho: torch.Tensor,
    radius: float,
    threshold: float | None = None,
) -> dict[str, float]:
    """Measure pressure difference across a droplet interface.

    For a 2D cylindrical droplet: Delta_p * R = sigma (Laplace law).
    Returns dict with dp, R_eq, sigma_laplace, rho_in, rho_out.
    """
    if threshold is None:
        threshold = 0.5 * (RHO_L_DEFAULT + RHO_V_DEFAULT)

    ny, nx = rho.shape
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32, device=rho.device),
        torch.arange(nx, dtype=torch.float32, device=rho.device),
        indexing="ij",
    )
    cx, cy = nx / 2.0, ny / 2.0
    r_field = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Inner and outer regions
    inner = r_field < 0.5 * radius
    outer = r_field > 1.5 * radius

    rho_in = float(rho[inner].mean()) if inner.any() else RHO_L_DEFAULT
    rho_out = float(rho[outer].mean()) if outer.any() else RHO_V_DEFAULT

    # Pressure from SC equation of state: p = rho*cs^2 + G*cs^2/2 * psi^2
    cs2 = 1.0 / 3.0
    G = 5.0
    psi_in = psi_exp(torch.tensor(rho_in)).item()
    psi_out = psi_exp(torch.tensor(rho_out)).item()
    p_in = rho_in * cs2 + G * cs2 / 2.0 * psi_in ** 2
    p_out = rho_out * cs2 + G * cs2 / 2.0 * psi_out ** 2
    dp = p_in - p_out

    # 2D Laplace: sigma = dp * R
    sigma_laplace = dp * radius

    return {
        "dp": dp,
        "R_eq": radius,
        "sigma_laplace": sigma_laplace,
        "rho_in": rho_in,
        "rho_out": rho_out,
        "p_in": p_in,
        "p_out": p_out,
    }


# ---------------------------------------------------------------------------
# Run functions
# ---------------------------------------------------------------------------


def run_static_contact(
    theta_target: float,
    nx: int = 128,
    ny: int = 64,
    radius: float = 15.0,
    steps: int = 8000,
    sample: int = 500,
    G: float = 5.0,
    tau: float = 1.0,
    device: str = "cpu",
) -> dict:
    """Relax a sessile drop and measure its static contact angle.

    Returns dict with theta_target, theta_measured, theta_error_deg,
    wall_psi, steps, converged, history.
    """
    channel = Channel(nx=nx, ny=ny, wall_rows=2, device=device)
    solid = channel.solid_mask()
    wall_psi = wall_psi_from_contact_angle(theta_target)

    f = sessile_drop_initial(channel, radius=radius)
    mass0 = float((f.sum(dim=0) * (~solid)).sum())

    history = []
    converged = False
    for step in range(1, steps + 1):
        f = step_sc(f, G=G, tau=tau, solid_mask=solid, wall_psi=wall_psi)

        if step % sample == 0:
            rho = f.sum(dim=0)
            if not torch.isfinite(rho).all():
                return {"status": "diverged", "step": step}

            meas = measure_contact_angle(rho, solid)
            history.append({"step": step, **meas})

            if len(history) >= 3 and step >= 3000:
                d1 = abs(history[-1]["theta_deg"] - history[-2]["theta_deg"])
                d2 = abs(history[-2]["theta_deg"] - history[-3]["theta_deg"])
                if d1 < 0.5 and d2 < 0.5:
                    converged = True
                    break

    rho = f.sum(dim=0)
    meas = measure_contact_angle(rho, solid)
    mass_now = float((rho * (~solid)).sum())

    return {
        "theta_target": theta_target,
        "theta_measured": meas["theta_deg"],
        "theta_error_deg": meas["theta_deg"] - theta_target,
        "wall_psi": wall_psi,
        "theta_expected": contact_angle_from_wall_psi(wall_psi),
        "steps": step,
        "converged": converged,
        "mass_drift": abs(mass_now - mass0) / mass0 if mass0 > 0 else float("nan"),
        "R_fit": meas["R_fit"],
        "n_contour": meas["n_contour"],
        "history": history,
        "status": meas.get("status", "ok"),
    }


def run_young_laplace(
    radius: float,
    nx: int = 64,
    ny: int = 64,
    steps: int = 6000,
    sample: int = 500,
    G: float = 5.0,
    tau: float = 1.0,
    device: str = "cpu",
) -> dict:
    """Relax a droplet and measure Young-Laplace pressure difference.

    Returns dict with radius, dp, sigma_laplace, sigma_expected,
    error_pct, steps, converged, history.
    """
    f = droplet_initial(nx=nx, ny=ny, radius=radius, device=device)
    mass0 = float(f.sum())

    history = []
    converged = False
    for step in range(1, steps + 1):
        f = step_sc(f, G=G, tau=tau)

        if step % sample == 0:
            rho = f.sum(dim=0)
            if not torch.isfinite(rho).all():
                return {"status": "diverged", "step": step}

            meas = measure_young_laplace(rho, radius)
            history.append({"step": step, **meas})

            if len(history) >= 3 and step >= 3000:
                d1 = abs(history[-1]["dp"] - history[-2]["dp"])
                d2 = abs(history[-2]["dp"] - history[-3]["dp"])
                if d1 < 1e-5 and d2 < 1e-5:
                    converged = True
                    break

    rho = f.sum(dim=0)
    meas = measure_young_laplace(rho, radius)
    mass_now = float(f.sum())

    sigma_expected = SIGMA_DEFAULT
    error_pct = abs(meas["sigma_laplace"] - sigma_expected) / sigma_expected * 100.0

    return {
        "radius": radius,
        "dp": meas["dp"],
        "sigma_laplace": meas["sigma_laplace"],
        "sigma_expected": sigma_expected,
        "error_pct": error_pct,
        "rho_in": meas["rho_in"],
        "rho_out": meas["rho_out"],
        "steps": step,
        "converged": converged,
        "mass_drift": abs(mass_now - mass0) / mass0 if mass0 > 0 else float("nan"),
        "history": history,
        "status": "ok",
    }


def run_capillary_scaling(
    theta_target: float = 90.0,
    gx_values: list[float] | None = None,
    nx: int = 160,
    ny: int = 72,
    radius: float = 18.0,
    steps: int = 12000,
    sample: int = 500,
    G: float = 5.0,
    tau: float = 1.0,
    device: str = "cpu",
) -> dict:
    """Run driven droplet at multiple body forces to verify Ca scaling.

    For each gx, measures steady velocity U and computes Ca = mu*U/sigma.
    The dynamic contact angle should follow Cox-Voinov:
        theta_d^3 = theta_s^3 + 9*Ca*ln(L/L_s)

    Returns dict with results per gx.
    """
    if gx_values is None:
        gx_values = [0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]

    channel = Channel(nx=nx, ny=ny, wall_rows=2, device=device)
    solid = channel.solid_mask()
    wall_psi = wall_psi_from_contact_angle(theta_target)

    # Static angle reference
    static_result = run_static_contact(
        theta_target, nx=nx, ny=ny, radius=radius,
        steps=6000, sample=500, G=G, tau=tau, device=device,
    )
    theta_static = static_result["theta_measured"]

    # Viscosity: mu = rho_l * (tau - 0.5) / 3
    mu = RHO_L_DEFAULT * (tau - 0.5) / 3.0
    sigma = SIGMA_DEFAULT

    results = []
    for gx in gx_values:
        f = sessile_drop_initial(channel, radius=radius)
        x_prev = float(channel.nx / 2)
        x_history = []
        t_history = []
        theta_history = []

        for step in range(1, steps + 1):
            f = step_sc(f, G=G, tau=tau, solid_mask=solid,
                        wall_psi=wall_psi, gx=gx)

            if step % sample == 0:
                rho = f.sum(dim=0)
                if not torch.isfinite(rho).all():
                    break

                # Centroid tracking
                threshold = 0.5 * (RHO_L_DEFAULT + RHO_V_DEFAULT)
                mask = (rho > threshold) & (~solid)
                if mask.any():
                    nx_ch = channel.nx
                    mass = rho[mask]
                    xs = torch.arange(nx_ch, dtype=torch.float64, device=rho.device).view(1, nx_ch).expand_as(rho)[mask]
                    ang = 2.0 * math.pi * xs / nx_ch
                    c = float((mass * torch.cos(ang)).sum())
                    s = float((mass * torch.sin(ang)).sum())
                    phi = math.atan2(s, c)
                    x = (phi % (2.0 * math.pi)) * nx_ch / (2.0 * math.pi)
                    # Unwrap
                    k = round((x_prev - x) / nx_ch)
                    x = x + k * nx_ch
                    x_prev = x
                    x_history.append(x)
                    t_history.append(float(step))

                # Contact angle
                meas = measure_contact_angle(rho, solid)
                if meas["status"] == "ok":
                    theta_history.append(meas["theta_deg"])

        # Steady velocity from linear fit
        if len(x_history) >= 4:
            tt = torch.tensor(t_history[-8:], dtype=torch.float64)
            xx = torch.tensor(x_history[-8:], dtype=torch.float64)
            tt = tt - tt.mean()
            xx = xx - xx.mean()
            sxx = float((tt * tt).sum())
            if sxx > 0:
                velocity = float((tt * xx).sum()) / sxx
            else:
                velocity = 0.0
        else:
            velocity = 0.0

        Ca = mu * abs(velocity) / sigma if sigma > 0 else 0.0
        theta_dynamic = theta_history[-1] if theta_history else theta_static

        results.append({
            "gx": gx,
            "velocity": velocity,
            "Ca": Ca,
            "theta_dynamic": theta_dynamic,
            "theta_static": theta_static,
            "n_samples": len(x_history),
        })

    return {
        "theta_static": theta_static,
        "mu": mu,
        "sigma": sigma,
        "results": results,
    }
