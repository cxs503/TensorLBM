"""RAO (Response Amplitude Operator) computation and spectral response analysis.

Provides tools for:
1. Computing RAO from time-domain simulation results (FFT-based)
2. Computing RAO from frequency-domain hydrodynamic data
3. Spectral response analysis (SRA): S_response(ω) = |RAO(ω)|² · S_wave(ω)
4. Statistical response metrics (significant amplitude, max response, etc.)
5. Comparison with experimental RAO data

The RAO is defined as the complex transfer function between wave excitation
and body response:

.. math::

    RAO(\\omega) = \\frac{X(\\omega)}{F_{exc}(\\omega)}
                  = \\frac{1}{-(M + A(\\omega))\\omega^2 + jB(\\omega)\\omega + C}

where X(ω) is the complex response amplitude and F_exc(ω) is the complex
excitation force amplitude.

References:
    Faltinsen, O.M. (1990). "Sea Loads on Ships and Offshore Structures."
    Cambridge University Press.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

_trapezoid = getattr(torch, "trapezoid", torch.trapz)
_DOF_COUNT = 6


@dataclass
class RAOData:
    """Container for RAO data across frequencies and DOFs.

    Attributes:
        omega: Angular frequency array [rad/s], shape (N_freq,).
        rao_amplitude: |RAO(ω)| for each DOF, shape (N_freq, 6).
        rao_phase: Phase angle of RAO [rad], shape (N_freq, 6).
        rao_complex: Complex RAO, shape (N_freq, 6).
        dof_labels: Labels for the 6 DOFs.
        inactive_dofs: DOFs excluded from the solve because they carry no
            inertia / damping / restoring term (unmodelled directions).
    """

    omega: torch.Tensor
    rao_amplitude: torch.Tensor  # (N_freq, 6)
    rao_phase: torch.Tensor  # (N_freq, 6)
    rao_complex: Optional[torch.Tensor] = None
    dof_labels: list = field(
        default_factory=lambda: ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]
    )
    inactive_dofs: tuple = field(default_factory=tuple)


def compute_rao_from_hydrodynamics(
    omega: torch.Tensor,
    added_mass: torch.Tensor,
    damping: torch.Tensor,
    stiffness: torch.Tensor,
    mass_matrix: torch.Tensor,
    excitation_amplitude: torch.Tensor,
) -> RAOData:
    """Compute RAO from frequency-domain hydrodynamic coefficients.

    For each frequency ω:
        RAO(ω) = H(ω) · F_exc(ω)

    where H(ω) = [-(M + A(ω))ω² + jωB(ω) + C]⁻¹

    Args:
        omega: Frequency array, shape (N_freq,).
        added_mass: A(ω), shape (N_freq, 6, 6).
        damping: B(ω), shape (N_freq, 6, 6).
        stiffness: C matrix, shape (6, 6).
        mass_matrix: M matrix, shape (6, 6).
        excitation_amplitude: |F_exc(ω)| per DOF, shape (N_freq, 6).

    Returns:
        RAOData with amplitude and phase.
    """
    N_freq = omega.shape[0]
    # float64 throughout: the response is a ratio of the excitation to a
    # difference of large terms, so single precision loses ~1e-3 relative
    # accuracy near the cancellation dip above resonance.
    mass_total = mass_matrix.to(torch.float64)
    stiff = stiffness.to(torch.float64)
    rao_complex = torch.zeros(N_freq, _DOF_COUNT, dtype=torch.complex128)
    inactive: set[int] = set()

    for i in range(N_freq):
        w = float(omega[i])
        # Dynamic stiffness matrix: Z(ω) = -(M + A(ω))ω² + jωB(ω) + C
        Z = (
            -(mass_total + added_mass[i].to(torch.float64)) * w**2
            + 1j * w * damping[i].to(torch.float64)
            + stiff
        )
        F_exc = excitation_amplitude[i].to(torch.complex128)
        # A DOF with no inertia, damping and restoring entry has an identically
        # vanishing row/column in Z, so the full 6x6 system is singular and
        # torch.linalg.inv used to raise _LinAlgError for every partial-DOF
        # model (e.g. pure heave).  Solve the active block only and leave the
        # unmodelled directions at zero response.
        row_mag = Z.abs().max(dim=1).values
        scale = float(row_mag.max())
        if scale > 0.0:
            active = row_mag > 1e-12 * scale
            index = torch.nonzero(active, as_tuple=False).squeeze(-1)
            inactive.update(
                torch.nonzero(~active, as_tuple=False).squeeze(-1).tolist()
            )
        else:  # Z == 0 (ω = 0 with no restoring force): response undefined
            index = torch.zeros(0, dtype=torch.long)
        if index.numel() == 0:
            continue
        # RAO_active = Z_active⁻¹ · F_exc
        Z_sub = Z[index][:, index]
        try:
            rao_complex[i, index] = torch.linalg.solve(Z_sub, F_exc[index])
        except Exception:  # pragma: no cover - singular sub-system at ω = 0
            rao_complex[i, index] = torch.linalg.lstsq(Z_sub, F_exc[index]).solution

    rao_amplitude = torch.abs(rao_complex)
    rao_phase = torch.angle(rao_complex)

    return RAOData(
        omega=omega,
        rao_amplitude=rao_amplitude,
        rao_phase=rao_phase,
        rao_complex=rao_complex,
        inactive_dofs=tuple(sorted(inactive)),
    )


def compute_rao_from_timeseries(
    t: torch.Tensor,
    wave_elevation: torch.Tensor,
    response: torch.Tensor,
    dof: int = 2,
    window: str = "hann",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute RAO from time-domain simulation using cross-spectral analysis.

    Uses the cross-spectrum method:
        RAO(ω) = S_{ηx}(ω) / S_{ηη}(ω)

    where S_{ηx} is the cross-spectral density between wave elevation η
    and response x, and S_{ηη} is the auto-spectral density of η.

    Args:
        t: Time array, shape (N,).
        wave_elevation: η(t), shape (N,).
        response: Response time series for one DOF, shape (N,).
        dof: DOF index (for labeling).
        window: Window function type ('hann', 'hamming', 'none').

    Returns:
        Tuple of (omega, rao_amplitude, rao_phase).
    """
    N = t.shape[0]
    dt = t[1] - t[0] if len(t) > 1 else torch.tensor(1.0)

    # Apply window
    if window == "hann":
        win = torch.hann_window(N, periodic=False)
    elif window == "hamming":
        win = torch.hamming_window(N, periodic=False)
    else:
        win = torch.ones(N)

    eta_win = wave_elevation * win
    resp_win = response * win

    # FFT
    N_fft = N
    ETA = torch.fft.rfft(eta_win, n=N_fft)
    RESP = torch.fft.rfft(resp_win, n=N_fft)

    # Frequency array
    freqs = torch.fft.rfftfreq(N_fft, d=dt.item())
    omega = 2.0 * math.pi * freqs

    # Cross-spectral density
    S_eta_x = ETA.conj() * RESP  # cross-spectrum
    S_eta_eta = (ETA.conj() * ETA).real  # auto-spectrum

    # Avoid division by zero
    S_eta_eta_safe = torch.clamp(S_eta_eta, min=1e-30)

    # RAO = S_{ηx} / S_{ηη}
    rao_complex = S_eta_x / S_eta_eta_safe
    rao_amplitude = torch.abs(rao_complex)
    rao_phase = torch.angle(rao_complex)

    # Bins where the excitation carries no energy hold only round-off noise, so
    # the ratio is meaningless (it used to blow up to O(1/clamp) and dominate
    # any downstream argmax).  Report zero response there instead.
    peak_energy = float(S_eta_eta.max())
    valid = (
        S_eta_eta > 1e-8 * peak_energy if peak_energy > 0.0 else torch.zeros_like(S_eta_eta, dtype=torch.bool)
    )
    rao_amplitude = torch.where(valid, rao_amplitude, torch.zeros_like(rao_amplitude))
    rao_phase = torch.where(valid, rao_phase, torch.zeros_like(rao_phase))

    return omega, rao_amplitude, rao_phase


def spectral_response_analysis(
    omega: torch.Tensor,
    wave_spectrum: torch.Tensor,
    rao_amplitude: torch.Tensor,
    dof: int = 2,
) -> dict:
    """Perform Spectral Response Analysis (SRA).

    Computes the response spectrum and statistical metrics:
        S_response(ω) = |RAO(ω)|² · S_wave(ω)

    Args:
        omega: Frequency array [rad/s].
        wave_spectrum: Wave energy spectrum S(ω) [m²·s].
        rao_amplitude: |RAO(ω)| for one DOF, shape (N_freq,).
        dof: DOF index.

    Returns:
        Dictionary with:
        - 'omega': frequency array, sorted ascending (input order restored
          nowhere: the returned grid is the sorted one actually integrated)
        - 'response_spectrum': S_r(ω)
        - 'm0': zeroth spectral moment
        - 'm2': second spectral moment
        - 'significant_amplitude': 2√m0 (significant single amplitude)
        - 'mean_period': 2π√(m0/m2) (mean zero-crossing period)
        - 'max_response_estimate': 1.86 × significant amplitude (for 3h storm)
    """
    # The moments are trapezoidal integrals that must run from low to high
    # frequency: an unsorted ω used to give a negative m0 and hence a NaN
    # significant amplitude.
    if omega.numel() > 1 and not bool(torch.all(omega[1:] > omega[:-1])):
        order = torch.argsort(omega)
        omega = omega[order]
        wave_spectrum = wave_spectrum[order]
        rao_amplitude = rao_amplitude[order]

    # Response spectrum
    S_response = rao_amplitude**2 * wave_spectrum

    # Spectral moments
    m0 = _trapezoid(S_response, omega)  # zeroth moment
    m2 = _trapezoid(S_response * omega**2, omega)  # second moment
    m4 = _trapezoid(S_response * omega**4, omega)  # fourth moment

    # Statistical metrics
    significant_amplitude = 2.0 * torch.sqrt(m0)
    mean_period = 2.0 * math.pi * torch.sqrt(m0 / m2) if m2 > 0 else torch.tensor(float("inf"))

    # Maximum response estimate (Rayleigh distribution, 1000 cycles ≈ 3h)
    n_cycles = 1000
    max_factor = math.sqrt(2.0 * math.log(n_cycles))
    max_response_estimate = significant_amplitude * max_factor / 2.0

    return {
        "omega": omega,
        "response_spectrum": S_response,
        "m0": m0,
        "m2": m2,
        "m4": m4,
        "significant_amplitude": significant_amplitude,
        "mean_period": mean_period,
        "max_response_estimate": max_response_estimate,
    }


def compare_rao_with_experiment(
    omega_sim: torch.Tensor,
    rao_sim: torch.Tensor,
    omega_exp: torch.Tensor,
    rao_exp: torch.Tensor,
    dof: int = 2,
    tolerance_relative: float = 0.15,
) -> dict:
    """Compare simulated RAO with experimental data.

    Interpolates simulation results to experimental frequencies and
    computes error metrics.

    Args:
        omega_sim: Simulation frequency array.
        rao_sim: Simulation RAO amplitude, shape (N_sim,).
        omega_exp: Experimental frequency array.
        rao_exp: Experimental RAO amplitude, shape (N_exp,).
        dof: DOF index for labeling.
        tolerance_relative: Relative tolerance for pass/fail (default 15%).

    Returns:
        Dictionary with:
        - 'omega_exp': experimental frequencies
        - 'rao_sim_interp': interpolated simulation RAO
        - 'rao_exp': experimental RAO
        - 'relative_error': |sim - exp| / |exp| at each frequency
        - 'mean_relative_error': mean relative error
        - 'max_relative_error': max relative error
        - 'rmse': root mean square error
        - 'nrmse': normalized RMSE (by range of experimental data)
        - 'pass': bool, True if mean relative error < tolerance
    """
    import numpy as np

    # Interpolate simulation RAO to experimental frequencies
    rao_sim_interp = torch.from_numpy(
        np.interp(omega_exp.numpy(), omega_sim.numpy(), rao_sim.numpy())
    )

    # Error metrics
    abs_error = torch.abs(rao_sim_interp - rao_exp)
    rao_exp_safe = torch.clamp(torch.abs(rao_exp), min=1e-10)
    relative_error = abs_error / rao_exp_safe

    mean_rel_error = relative_error.mean()
    max_rel_error = relative_error.max()
    rmse = torch.sqrt((abs_error**2).mean())

    exp_range = rao_exp.max() - rao_exp.min()
    nrmse = rmse / exp_range if exp_range > 0 else torch.tensor(float("inf"))

    passed = mean_rel_error.item() < tolerance_relative

    return {
        "omega_exp": omega_exp,
        "rao_sim_interp": rao_sim_interp,
        "rao_exp": rao_exp,
        "relative_error": relative_error,
        "mean_relative_error": mean_rel_error,
        "max_relative_error": max_rel_error,
        "rmse": rmse,
        "nrmse": nrmse,
        "pass": passed,
        "tolerance": tolerance_relative,
    }


def compute_natural_frequencies(
    mass_matrix: torch.Tensor,
    added_mass_zero: torch.Tensor,
    stiffness: torch.Tensor,
) -> torch.Tensor:
    """Compute natural frequencies of the floating body.

    Solves the eigenvalue problem:
        ω_n² = eigenvalues of (M + A(0))⁻¹ C

    Args:
        mass_matrix: Body mass matrix M, shape (6, 6).
        added_mass_zero: Added mass at zero frequency A(0), shape (6, 6).
        stiffness: Hydrostatic stiffness C, shape (6, 6).

    Returns:
        Natural frequencies ω_n [rad/s], shape (6,).  DOFs with no inertia
        (unmodelled directions, singular mass matrix) come back as NaN instead
        of raising a ``_LinAlgError``.
    """
    M_total = (mass_matrix + added_mass_zero).to(torch.float64)
    C = stiffness.to(torch.float64)
    n_dof = M_total.shape[0]
    row_mag = M_total.abs().max(dim=1).values
    scale = float(row_mag.max())
    if scale > 0.0:
        active = row_mag > 1e-12 * scale
    else:
        active = torch.zeros(n_dof, dtype=torch.bool)
    index = torch.nonzero(active, as_tuple=False).squeeze(-1)
    omega_n = torch.full((n_dof,), float("nan"), dtype=torch.float64)
    if index.numel():
        M_sub = M_total[index][:, index]
        C_sub = C[index][:, index]
        # ω² = eig(M⁻¹C), solved rather than formed by an explicit inverse.
        # For a decoupled body this keeps ω_n[i] on DOF i; for a coupled body the
        # order follows the eigenvalue routine (same contract as before).
        try:
            eigenvalues = torch.linalg.eigvals(
                torch.linalg.solve(M_sub, C_sub)
            ).real
        except Exception:  # pragma: no cover - singular active block
            eigenvalues = torch.linalg.eigvals(torch.linalg.pinv(M_sub) @ C_sub).real
        omega_n[index] = torch.sqrt(torch.clamp(eigenvalues, min=0.0))
    if mass_matrix.is_floating_point():
        return omega_n.to(mass_matrix.dtype)
    return omega_n


def rao_peak_detection(
    omega: torch.Tensor,
    rao_amplitude: torch.Tensor,
    dof: int = 2,
) -> dict:
    """Detect resonance peaks in the RAO.

    Args:
        omega: Frequency array.
        rao_amplitude: RAO amplitude for one DOF.
        dof: DOF index.

    Returns:
        Dictionary with peak frequencies and amplitudes.
    """
    # Simple peak detection: local maxima
    peaks = []
    for i in range(1, len(rao_amplitude) - 1):
        if rao_amplitude[i] > rao_amplitude[i - 1] and rao_amplitude[i] > rao_amplitude[i + 1]:
            peaks.append(
                {
                    "omega": omega[i].item(),
                    "period": 2.0 * math.pi / omega[i].item() if omega[i] > 0 else float("inf"),
                    "amplitude": rao_amplitude[i].item(),
                }
            )

    # Sort by amplitude (descending)
    peaks.sort(key=lambda p: p["amplitude"], reverse=True)

    return {
        "peaks": peaks,
        "n_peaks": len(peaks),
        "dominant_omega": peaks[0]["omega"] if peaks else None,
        "dominant_period": peaks[0]["period"] if peaks else None,
        "dominant_amplitude": peaks[0]["amplitude"] if peaks else None,
    }
