"""Statistics accumulation and analysis for turbulent channel flow.

Computes mean velocity profiles, Reynolds stresses, TKE, energy spectra,
and SGS stress distributions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class StatsAccumulator:
    """Accumulates turbulence statistics from channel flow simulation.

    Averages over homogeneous directions (x, z) and time.
    """

    def __init__(self, ny: int, device: str = "cpu"):
        self.ny = ny
        self.device = device
        self.count = 0

        # Accumulators (wall-normal profiles)
        self.sum_u = torch.zeros(ny, device=device, dtype=torch.float64)
        self.sum_uu = torch.zeros(ny, device=device, dtype=torch.float64)
        self.sum_vv = torch.zeros(ny, device=device, dtype=torch.float64)
        self.sum_ww = torch.zeros(ny, device=device, dtype=torch.float64)
        self.sum_uv = torch.zeros(ny, device=device, dtype=torch.float64)
        self.sum_mass = torch.zeros(1, device=device, dtype=torch.float64)

    def accumulate(self, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
                   rho: Optional[torch.Tensor] = None) -> None:
        """Accumulate statistics from a single time snapshot.

        Parameters
        ----------
        ux, uy, uz : torch.Tensor, shape (nz, ny, nx)
            Velocity components.
        rho : torch.Tensor, optional
            Density field. If None, assumed constant = 1.
        """
        # Average over homogeneous directions (x, z)
        u_mean = ux.mean(dim=(0, 2))  # (ny,)
        v_mean = uy.mean(dim=(0, 2))
        w_mean = uz.mean(dim=(0, 2))

        # Fluctuations
        up = ux - u_mean.view(1, -1, 1)
        vp = uy - v_mean.view(1, -1, 1)
        wp = uz - w_mean.view(1, -1, 1)

        # Reynolds stresses (averaged over x, z)
        uu = (up ** 2).mean(dim=(0, 2))
        vv = (vp ** 2).mean(dim=(0, 2))
        ww = (wp ** 2).mean(dim=(0, 2))
        uv = (up * vp).mean(dim=(0, 2))

        # Accumulate
        self.sum_u += u_mean.to(torch.float64)
        self.sum_uu += uu.to(torch.float64)
        self.sum_vv += vv.to(torch.float64)
        self.sum_ww += ww.to(torch.float64)
        self.sum_uv += uv.to(torch.float64)

        if rho is not None:
            self.sum_mass += rho.mean().to(torch.float64)
        else:
            self.sum_mass += 1.0

        self.count += 1

    def get_profiles(self) -> dict:
        """Compute time-averaged profiles.

        Returns
        -------
        dict with keys: U, uu, vv, ww, uv, TKE (all shape (ny,))
        """
        if self.count == 0:
            raise ValueError("No samples accumulated")

        n = float(self.count)
        U = self.sum_u / n
        uu = self.sum_uu / n
        vv = self.sum_vv / n
        ww = self.sum_ww / n
        uv = self.sum_uv / n

        # TKE = 0.5 * (uu + vv + ww)
        tke = 0.5 * (uu + vv + ww)

        return {
            "U": U.cpu().numpy(),
            "uu": uu.cpu().numpy(),
            "vv": vv.cpu().numpy(),
            "ww": ww.cpu().numpy(),
            "uv": uv.cpu().numpy(),
            "TKE": tke.cpu().numpy(),
            "n_samples": self.count,
        }


def compute_profiles(
    ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
    u_tau: float, nu: float, ny: int,
) -> dict:
    """Compute wall-unit profiles from instantaneous fields.

    Parameters
    ----------
    ux, uy, uz : torch.Tensor, shape (nz, ny, nx)
        Velocity components.
    u_tau : float
        Friction velocity.
    nu : float
        Kinematic viscosity.
    ny : int
        Number of wall-normal points.

    Returns
    -------
    dict with y+, U+, uu+, vv+, ww+, uv+, TKE+ profiles.
    """
    # Wall-normal coordinate (distance from bottom wall)
    y = torch.arange(ny, dtype=torch.float64)
    h = (ny - 2) / 2.0  # half-height

    # Distance from nearest wall
    y_from_wall = torch.minimum(y, torch.tensor(ny - 1) - y)

    # Wall units
    y_plus = y_from_wall * u_tau / nu

    # Average over homogeneous directions
    U = ux.mean(dim=(0, 2)).to(torch.float64)
    V = uy.mean(dim=(0, 2)).to(torch.float64)
    W = uz.mean(dim=(0, 2)).to(torch.float64)

    # Fluctuations
    up = ux - U.view(1, -1, 1)
    vp = uy - V.view(1, -1, 1)
    wp = uz - W.view(1, -1, 1)

    # Reynolds stresses
    uu = (up ** 2).mean(dim=(0, 2)).to(torch.float64)
    vv = (vp ** 2).mean(dim=(0, 2)).to(torch.float64)
    ww = (wp ** 2).mean(dim=(0, 2)).to(torch.float64)
    uv = (up * vp).mean(dim=(0, 2)).to(torch.float64)

    # Normalize by u_tau
    U_plus = U / u_tau
    uu_plus = uu / (u_tau ** 2)
    vv_plus = vv / (u_tau ** 2)
    ww_plus = ww / (u_tau ** 2)
    uv_plus = uv / (u_tau ** 2)
    tke_plus = 0.5 * (uu_plus + vv_plus + ww_plus)

    return {
        "y_plus": y_plus.numpy(),
        "U_plus": U_plus.cpu().numpy(),
        "uu_plus": uu_plus.cpu().numpy(),
        "vv_plus": vv_plus.cpu().numpy(),
        "ww_plus": ww_plus.cpu().numpy(),
        "uv_plus": uv_plus.cpu().numpy(),
        "TKE_plus": tke_plus.cpu().numpy(),
    }


def compute_energy_spectrum(
    ux: torch.Tensor,
    dx: float = 1.0,
    n_planes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute streamwise energy spectrum E_uu(k_x).

    Parameters
    ----------
    ux : torch.Tensor, shape (nz, ny, nx)
        Streamwise velocity field.
    dx : float
        Grid spacing in streamwise direction.
    n_planes : int
        Number of wall-normal planes to average over (center of channel).

    Returns
    -------
    k : np.ndarray
        Wavenumber array.
    E_uu : np.ndarray
        Energy spectrum (averaged over planes and spanwise direction).
    """
    nz, ny, nx = ux.shape

    # Select planes in the center of the channel
    center = ny // 2
    half_range = max(1, n_planes // 2)
    y_start = max(0, center - half_range)
    y_end = min(ny, center + half_range)

    # Average over selected planes and spanwise direction
    ux_slice = ux[:, y_start:y_end, :]  # (nz, n_sel, nx)
    ux_avg = ux_slice.mean(dim=(0, 1))  # (nx,)

    # Remove mean
    ux_prime = ux_avg - ux_avg.mean()

    # FFT
    ux_hat = np.fft.rfft(ux_prime.cpu().numpy())

    # Energy spectrum
    E_uu = np.abs(ux_hat) ** 2 / nx

    # Wavenumber
    k = np.fft.rfftfreq(nx, d=dx) * 2 * np.pi

    # Remove k=0
    k = k[1:]
    E_uu = E_uu[1:]

    return k, E_uu


def compute_energy_spectrum_3d(
    ux: torch.Tensor,
    dx: float = 1.0,
    dz: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D energy spectrum E(k_x, k_z).

    Parameters
    ----------
    ux : torch.Tensor, shape (nz, ny, nx)
        Streamwise velocity field.
    dx, dz : float
        Grid spacing.

    Returns
    -------
    kx, kz : np.ndarray
        Wavenumber arrays.
    E : np.ndarray, shape (nkz, nkx)
        2D energy spectrum.
    """
    nz, ny, nx = ux.shape

    # Average over wall-normal direction
    ux_avg = ux.mean(dim=1).cpu().numpy()  # (nz, nx)

    # Remove mean
    ux_prime = ux_avg - ux_avg.mean()

    # 2D FFT
    ux_hat = np.fft.rfft2(ux_prime)

    # Energy spectrum
    E = np.abs(ux_hat) ** 2 / (nx * nz)

    # Wavenumbers
    kx = np.fft.rfftfreq(nx, d=dx) * 2 * np.pi
    kz = np.fft.fftfreq(nz, d=dz) * 2 * np.pi

    return kx, kz, E


def compute_sgs_stress(
    f: torch.Tensor,
    tau: float,
    C_s: float,
    nu: float,
    u_tau: float,
) -> dict:
    """Compute SGS stress distribution.

    Parameters
    ----------
    f : torch.Tensor, shape (19, nz, ny, nx)
        Distribution functions.
    tau : float
        Relaxation time.
    C_s : float
        Smagorinsky constant.
    nu : float
        Molecular viscosity.
    u_tau : float
        Friction velocity.

    Returns
    -------
    dict with SGS stress profiles.
    """
    from tensorlbm.d3q19 import C
    from tensorlbm import macroscopic3d, equilibrium3d

    device = f.device
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    fneq = f - feq

    # Compute non-equilibrium stress tensor
    # Pi_ij = sum_alpha fneq_alpha * c_alpha_i * c_alpha_j
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    r3 = rho.view(nz, ny, nx)

    cx = C[:, 0].to(device).view(-1, 1, 1, 1)
    cy = C[:, 1].to(device).view(-1, 1, 1, 1)
    cz = C[:, 2].to(device).view(-1, 1, 1, 1)

    fneq3 = fneq.view(19, nz, ny, nx)

    # SGS stress components (averaged over x, z)
    tau_xx = (fneq3 * cx * cx).sum(0).mean(dim=(0, 2)) / r3.mean(dim=(0, 2))
    tau_yy = (fneq3 * cy * cy).sum(0).mean(dim=(0, 2)) / r3.mean(dim=(0, 2))
    tau_xy = (fneq3 * cx * cy).sum(0).mean(dim=(0, 2)) / r3.mean(dim=(0, 2))

    # Total stress = molecular + SGS
    # Molecular stress from velocity gradients
    # SGS stress = -2 * nu_t * S_ij

    # Estimate SGS viscosity from Smagorinsky model
    # |S| from non-equilibrium
    pi_norm = torch.sqrt(
        2.0 * (tau_xy ** 2) + tau_xx ** 2 + tau_yy ** 2
    )

    # SGS viscosity (iterative solution)
    nu_t = torch.zeros_like(pi_norm)
    for _ in range(3):
        nu_t = (C_s ** 2) * pi_norm / (2.0 * (nu + nu_t)).clamp(min=1e-12)

    # SGS stress magnitude
    sgs_mag = 2.0 * nu_t * pi_norm

    # Normalize by u_tau^2
    tau_xx_plus = tau_xx / (u_tau ** 2)
    tau_yy_plus = tau_yy / (u_tau ** 2)
    tau_xy_plus = tau_xy / (u_tau ** 2)
    sgs_mag_plus = sgs_mag / (u_tau ** 2)

    # Wall-normal coordinate
    y = torch.arange(ny, dtype=torch.float64)
    h = (ny - 2) / 2.0
    y_from_wall = torch.minimum(y, torch.tensor(ny - 1, dtype=torch.float64) - y)
    y_plus = y_from_wall * u_tau / nu

    return {
        "y_plus": y_plus.numpy(),
        "tau_xx_plus": tau_xx_plus.cpu().numpy(),
        "tau_yy_plus": tau_yy_plus.cpu().numpy(),
        "tau_xy_plus": tau_xy_plus.cpu().numpy(),
        "sgs_mag_plus": sgs_mag_plus.cpu().numpy(),
        "nu_t_over_nu": (nu_t / nu).cpu().numpy(),
    }


def fit_inertial_subrange(
    k: np.ndarray,
    E: np.ndarray,
    k_range: Optional[tuple[float, float]] = None,
) -> dict:
    """Fit -5/3 slope in the inertial subrange.

    Parameters
    ----------
    k : np.ndarray
        Wavenumber array.
    E : np.ndarray
        Energy spectrum.
    k_range : tuple, optional
        Wavenumber range for fitting.

    Returns
    -------
    dict with slope, r_squared, and k_range used.
    """
    if k_range is None:
        # Use middle portion of log(k) range
        k_pos = k[k > 0]
        E_pos = E[k > 0]
        if len(k_pos) < 5:
            return {"slope": 0.0, "r_squared": 0.0, "k_range": [0, 0], "n_points": 0}

        log_k = np.log10(k_pos)
        k_min = 10 ** (log_k.min() + np.ptp(log_k) * 0.15)
        k_max = 10 ** (log_k.max() - np.ptp(log_k) * 0.15)
    else:
        k_min, k_max = k_range
        k_pos = k[(k >= k_min) & (k <= k_max) & (k > 0)]
        E_pos = E[(k >= k_min) & (k <= k_max) & (k > 0)]

    mask = (k > 0) & (E > 0)
    if k_range:
        mask &= (k >= k_min) & (k <= k_max)

    if mask.sum() < 3:
        return {"slope": 0.0, "r_squared": 0.0, "k_range": [k_min, k_max], "n_points": 0}

    log_k = np.log10(k[mask])
    log_E = np.log10(E[mask])

    # Linear fit: log(E) = slope * log(k) + const
    coeffs = np.polyfit(log_k, log_E, 1)
    slope = float(coeffs[0])

    # R²
    E_fit = 10 ** np.polyval(coeffs, log_k)
    ss_res = np.sum((log_E - np.log10(E_fit)) ** 2)
    ss_tot = np.sum((log_E - log_E.mean()) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-20)

    return {
        "slope": slope,
        "r_squared": float(r_squared),
        "k_range": [float(k_min), float(k_max)],
        "n_points": int(mask.sum()),
    }


def compute_gci(
    phi_coarse: float,
    phi_medium: float,
    phi_fine: float,
    r: float,
    p: float = 2.0,
) -> dict:
    """Compute Grid Convergence Index (GCI).

    Parameters
    ----------
    phi_coarse, phi_medium, phi_fine : float
        Quantity of interest on coarse, medium, fine grids.
    r : float
        Grid refinement ratio.
    p : float
        Observed order of convergence.

    Returns
    -------
    dict with GCI values and convergence status.
    """
    # Relative errors
    eps_fine = abs((phi_fine - phi_medium) / phi_fine) * 100
    eps_medium = abs((phi_medium - phi_coarse) / phi_medium) * 100

    # GCI (safety factor 1.25 for fine grid, 3.0 for medium)
    gci_fine = 1.25 * eps_fine / (r ** p - 1)
    gci_medium = 3.0 * eps_medium / (r ** p - 1)

    # Observed order of convergence
    p_obs = np.log(abs((phi_fine - phi_medium) / (phi_medium - phi_coarse))) / np.log(r)

    # Convergence status
    if abs(p_obs - p) < 0.5:
        status = "monotonic"
    elif p_obs > 0:
        status = "oscillatory"
    else:
        status = "divergent"

    return {
        "phi_fine": phi_fine,
        "phi_medium": phi_medium,
        "phi_coarse": phi_coarse,
        "r": r,
        "p_observed": float(p_obs),
        "eps_fine_pct": float(eps_fine),
        "eps_medium_pct": float(eps_medium),
        "gci_fine_pct": float(gci_fine),
        "gci_medium_pct": float(gci_medium),
        "convergence_status": status,
    }
