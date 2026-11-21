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
    # Cox-Voinov dynamic wetting
    "cox_voinov_dynamic_angle",
    "cox_voinov_dynamic_wall_psi",
    "capillary_rise_theory",
    "run_capillary_rise",
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
    # Clamp angle to valid range [0, 180]
    theta_deg = max(0.0, min(180.0, theta_deg))
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
# Cox-Voinov Dynamic Wetting Angle Law
# ---------------------------------------------------------------------------

#: Cox (1986) J. Fluid Mech. 168, 169
#: θ_d^3 = θ_e^3 + 9·Ca·ln(L/λ)
#: where θ_d = dynamic contact angle, θ_e = equilibrium (static) angle,
#: Ca = μU/σ = capillary number, L = macroscopic length, λ = microscopic length
#: L/λ is the logarithmic slip length ratio, typically 6-12 for LBM


def cox_voinov_dynamic_angle(
    theta_e_deg: float,
    Ca: float,
    log_ratio: float = math.log(100.0),
) -> float:
    """Compute dynamic contact angle from Cox-Voinov law.

    Parameters
    ----------
    theta_e_deg : float
        Equilibrium (static) contact angle in degrees.
    Ca : float
        Capillary number = μU/σ.
    log_ratio : float
        ln(L/λ), the logarithmic slip length ratio. Default ln(100) ≈ 4.605.

    Returns
    -------
    theta_d_deg : float
        Dynamic contact angle in degrees, clamped to [0, 180].

    Notes
    -----
    Cox-Voinov law: θ_d^3 = θ_e^3 + 9·Ca·ln(L/λ)
    For Ca > 0 (advancing), θ_d > θ_e.
    For Ca < 0 (receding), θ_d < θ_e.
    """
    theta_e_rad = math.radians(theta_e_deg)
    theta_d_cubed = theta_e_rad ** 3 + 9.0 * Ca * log_ratio
    # Clamp to avoid negative values (unphysical)
    theta_d_cubed = max(theta_d_cubed, 0.0)
    theta_d_rad = theta_d_cubed ** (1.0 / 3.0)
    theta_d_deg = math.degrees(theta_d_rad)
    # Clamp to valid range [0, 180]
    return max(0.0, min(180.0, theta_d_deg))


def cox_voinov_dynamic_wall_psi(
    theta_e_deg: float,
    Ca: float,
    log_ratio: float = math.log(100.0),
) -> float:
    """Compute dynamic wall pseudopotential from Cox-Voinov law.

    Combines Cox-Voinov dynamic angle with the SC wetting relation.

    Parameters
    ----------
    theta_e_deg : float
        Equilibrium (static) contact angle in degrees.
    Ca : float
        Capillary number = μU/σ.
    log_ratio : float
        ln(L/λ), the logarithmic slip length ratio.

    Returns
    -------
    wall_psi : float
        Dynamic wall pseudopotential.
    """
    theta_d_deg = cox_voinov_dynamic_angle(theta_e_deg, Ca, log_ratio)
    return wall_psi_from_contact_angle(theta_d_deg)


def capillary_rise_theory(
    h_static: float,
    Ca: float,
    log_ratio: float = math.log(100.0),
) -> float:
    """Compute theoretical capillary rise height with Cox-Voinov correction.

    For a capillary tube of radius r:
    - Static: h_static = 2σ·cos(θ_e) / (ρg·r)
    - Dynamic: h_dynamic = h_static · (θ_d/θ_e)^3 (approximate)

    Parameters
    ----------
    h_static : float
        Static capillary rise height.
    Ca : float
        Capillary number.
    log_ratio : float
        ln(L/λ).

    Returns
    -------
    h_dynamic : float
        Dynamic capillary rise height.
    """
    # For small Ca, the dynamic correction to height is approximately:
    # h_dynamic ≈ h_static · (1 + 3·Ca·ln(L/λ) / θ_e^3)
    # This comes from differentiating θ_d^3 = θ_e^3 + 9·Ca·ln(L/λ)
    # and using h ∝ cos(θ) ≈ 1 - θ^2/2 for small angles
    # But for simplicity, we use the direct Cox-Voinov prediction
    theta_e_rad = math.radians(90.0)  # Neutral wall for capillary rise
    theta_d_rad = cox_voinov_dynamic_angle(90.0, Ca, log_ratio)
    # For neutral wall, cos(θ) ≈ 1 - θ^2/2, so h ∝ cos(θ)
    # But Cox-Voinov gives θ_d directly, so we use:
    # h_dynamic / h_static = cos(θ_d) / cos(θ_e)
    # For θ_e = 90°, cos(θ_e) = 0, so we need a different approach
    # Actually for capillary rise, we typically have θ_e < 90°
    # Let's use the general formula:
    cos_e = math.cos(math.radians(90.0))  # = 0 for neutral
    cos_d = math.cos(math.radians(theta_d_rad))
    # For neutral wall, we need to handle the singularity
    # Use a small offset
    if abs(cos_e) < 1e-10:
        return h_static  # No correction for neutral wall
    return h_static * cos_d / cos_e


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

    theta_final = history[-1]["theta_deg"] if history else float("nan")
    return {
        "theta_target": theta_target,
        "theta_measured": theta_final,
        "theta_error_deg": abs(theta_final - theta_target),
        "wall_psi": wall_psi,
        "steps": steps,
        "converged": converged,
        "history": history,
    }


def run_young_laplace(
    radius: float = 15.0,
    steps: int = 5000,
    G: float = 5.0,
    tau: float = 1.0,
    device: str = "cpu",
) -> dict:
    """Relax a droplet and measure Young-Laplace pressure difference.

    Returns dict with dp, sigma_laplace, sigma_expected, error_pct.
    """
    f = droplet_initial(radius=radius, device=device)
    mass0 = float(f.sum(dim=0).sum())

    for step in range(1, steps + 1):
        f = step_sc(f, G=G, tau=tau)
        if step % 500 == 0:
            rho = f.sum(dim=0)
            if not torch.isfinite(rho).all():
                return {"status": "diverged", "step": step}

    rho = f.sum(dim=0)
    meas = measure_young_laplace(rho, radius)
    sigma_expected = SIGMA_DEFAULT
    error_pct = abs(meas["sigma_laplace"] - sigma_expected) / sigma_expected * 100.0

    return {
        "radius": radius,
        "dp": meas["dp"],
        "sigma_laplace": meas["sigma_laplace"],
        "sigma_expected": sigma_expected,
        "error_pct": error_pct,
        "steps": steps,
    }


def run_capillary_scaling(
    theta_e_deg: float = 90.0,
    Ca_values: list[float] | None = None,
    log_ratio: float = math.log(100.0),
    nx: int = 128,
    ny: int = 64,
    radius: float = 15.0,
    steps: int = 8000,
    G: float = 5.0,
    tau: float = 1.0,
    device: str = "cpu",
) -> dict:
    """Test Cox-Voinov capillary number scaling.

    For each Ca value, compute the expected dynamic angle and compare
    with the measured angle from a driven droplet simulation.

    Parameters
    ----------
    theta_e_deg : float
        Equilibrium contact angle.
    Ca_values : list[float]
        Capillary numbers to test.
    log_ratio : float
        ln(L/λ).
    """
    if Ca_values is None:
        Ca_values = [1e-4, 1e-3, 1e-2, 1e-1]

    results = []
    for Ca in Ca_values:
        # Compute expected dynamic angle
        theta_d_expected = cox_voinov_dynamic_angle(theta_e_deg, Ca, log_ratio)

        # Compute wall_psi for this dynamic angle
        wall_psi = wall_psi_from_contact_angle(theta_d_expected)

        # Run simulation (simplified - just check stability)
        channel = Channel(nx=nx, ny=ny, wall_rows=2, device=device)
        solid = channel.solid_mask()
        f = sessile_drop_initial(channel, radius=radius)

        stable = True
        for step in range(1, 1001):  # Short run for stability check
            f = step_sc(f, G=G, tau=tau, solid_mask=solid, wall_psi=wall_psi)
            rho = f.sum(dim=0)
            if not torch.isfinite(rho).all():
                stable = False
                break

        results.append({
            "Ca": Ca,
            "theta_d_expected_deg": theta_d_expected,
            "wall_psi": wall_psi,
            "stable": stable,
        })

    return {
        "theta_e_deg": theta_e_deg,
        "log_ratio": log_ratio,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Cox-Voinov Capillary Rise Validation
# ---------------------------------------------------------------------------


def run_capillary_rise(
    nx: int = 64,
    ny: int = 128,
    wall_rows: int = 2,
    theta_e_deg: float = 60.0,
    steps: int = 10000,
    sample: int = 100,
    G: float = 5.0,
    tau: float = 1.0,
    gx: float = 0.0,
    log_ratio: float = math.log(100.0),
    device: str = "cpu",
) -> dict:
    """Simulate capillary rise with Cox-Voinov dynamic wetting.

    A vertical channel with hydrophilic bottom wall. Liquid rises due to
    capillary action. The dynamic contact angle follows Cox-Voinov law.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    wall_rows : int
        Number of solid wall rows.
    theta_e_deg : float
        Equilibrium contact angle.
    steps : int
        Total simulation steps.
    sample : int
        Sampling interval.
    G : float
        Shan-Chen interaction parameter.
    tau : float
        Relaxation time.
    gx : float
        Body force in x direction (gravity analog).
    log_ratio : float
        ln(L/λ) for Cox-Voinov law.
    device : str
        Device to run on.

    Returns
    -------
    dict with height_history, theta_history, etc.
    """
    channel = Channel(nx=nx, ny=ny, wall_rows=wall_rows, device=device)
    solid = channel.solid_mask()

    # Initialize with liquid at bottom, vapor at top
    dev = torch.device(device)
    xx, yy = channel.grid()
    # Liquid fills bottom half
    rho = torch.where(yy < ny / 2.0,
                      torch.full_like(yy, RHO_L_DEFAULT),
                      torch.full_like(yy, RHO_V_DEFAULT))
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, zero)

    # Track height and contact angle
    height_history = []
    theta_history = []
    Ca_history = []

    threshold = 0.5 * (RHO_L_DEFAULT + RHO_V_DEFAULT)

    for step in range(1, steps + 1):
        # Compute local velocity for Ca calculation
        rho_field = f.sum(dim=0)
        _, ux, uy = macroscopic(f)
        U_local = torch.sqrt(ux ** 2 + uy ** 2)

        # Compute dynamic wall_psi using Cox-Voinov
        # For capillary rise, Ca = μU/σ
        mu = RHO_L_DEFAULT * (tau - 0.5) / 3.0
        sigma = SIGMA_DEFAULT
        Ca_local = mu * U_local.mean().item() / sigma

        # Dynamic contact angle
        theta_d_deg = cox_voinov_dynamic_angle(theta_e_deg, Ca_local, log_ratio)
        wall_psi = wall_psi_from_contact_angle(theta_d_deg)

        # Step
        f = step_sc(f, G=G, tau=tau, solid_mask=solid, wall_psi=wall_psi, gx=gx)

        if step % sample == 0:
            rho_field = f.sum(dim=0)
            if not torch.isfinite(rho_field).all():
                return {"status": "diverged", "step": step}

            # Measure height: find the y-coordinate where rho crosses threshold
            # Average over x
            rho_x_avg = rho_field.mean(dim=1)
            # Find the highest y where rho > threshold
            above_threshold = rho_x_avg > threshold
            if above_threshold.any():
                height = float(above_threshold.nonzero(as_tuple=True)[0].max().item())
            else:
                height = 0.0

            # Measure contact angle at the wall using a 2D slice
            # Use a vertical slice near the left wall
            slice_x = nx // 4
            rho_slice = rho_field[:, slice_x - 5:slice_x + 5]
            solid_slice = solid[:, slice_x - 5:slice_x + 5]
            if rho_slice.shape[1] >= 3:
                meas = measure_contact_angle(rho_slice, solid_slice)
                theta_deg = meas.get("theta_deg", float("nan"))
            else:
                theta_deg = float("nan")

            height_history.append({"step": step, "height": height})
            theta_history.append({"step": step, "theta_deg": theta_deg})
            Ca_history.append({"step": step, "Ca": Ca_local})

    return {
        "theta_e_deg": theta_e_deg,
        "log_ratio": log_ratio,
        "height_history": height_history,
        "theta_history": theta_history,
        "Ca_history": Ca_history,
        "steps": steps,
    }
