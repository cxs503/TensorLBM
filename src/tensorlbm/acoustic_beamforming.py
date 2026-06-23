"""Acoustic beamforming and noise source identification for TensorLBM.

Implements microphone-array signal processing to localise aeroacoustic noise
sources from CFD (LBM) simulations.  This corresponds to the **acoustic maps**
and **noise source identification** capability in PowerFlow.

Method overview
---------------
The classical **Delay-and-Sum (DAS) beamformer** steers the microphone array
to each candidate source location by applying time delays (or phase shifts in
the frequency domain) to compensate for the propagation time from each source
point to each microphone, then sums the signals coherently:

    b(x_s) = Σ_m  w_m p_m(t − r_ms/c_0)

where:
  x_s   – steering (candidate source) position
  p_m   – pressure time-series at microphone m
  w_m   – shading weight (uniform, Hamming, Hann, Dolph–Chebyshev)
  r_ms  – distance from source x_s to microphone m
  c_0   – speed of sound

In the frequency domain the beamformer becomes:

    B(x_s, f) = w^H e(x_s, f)  where  e_m = exp(−i 2π f r_ms / c_0)

The **source power map** (acoustic map) is formed by steering to each grid
point and computing the output power spectral density.

Advanced options
----------------
* **DAMAS** (Dougherty 2005) – iterative deconvolution of the DAS output to
  remove side-lobe contamination and sharpen the source map.
* **Clean-SC** (Sijtsma 2007) – iterative subtraction of the strongest source
  contribution for sparse source identification.

References
----------
Johnson, D. H. & Dudgeon, D. E. (1993). *Array Signal Processing*. Prentice Hall.
Brooks, T. F. & Humphreys, W. M. (2006). A deconvolution approach for the
    mapping of acoustic sources (DAMAS) determined from phased microphone arrays.
    *J. Sound Vib.* 294, 856–879.
Sijtsma, P. (2007). CLEAN based on spatial source coherence. *Int. J. Aeroacoust.*
    6, 357–374.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch

__all__ = [
    "MicrophoneArray",
    "BeamformingConfig",
    "BeamformingResult",
    "das_beamformer",
    "compute_source_map",
    "clean_sc",
    "damas_deconvolve",
    "run_acoustic_beamforming",
]

# Speed of sound in air at 20 °C [m/s]
_C0_DEFAULT: float = 343.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MicrophoneArray:
    """Microphone positions and measured pressure signals.

    Attributes
    ----------
    positions:
        (M, 3) tensor of microphone positions in physical coordinates [m].
    signals:
        (M, N_t) tensor of pressure fluctuation time-series [Pa].
    dt:
        Time step between samples [s].
    """
    positions: torch.Tensor       # (M, 3)
    signals: torch.Tensor         # (M, N_t)
    dt: float = 1e-4

    @property
    def n_mics(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.signals.shape[1])


@dataclass
class BeamformingConfig:
    """Configuration for the beamforming analysis."""
    # Source scan grid
    scan_x: tuple[float, float, int] = (-1.0, 1.0, 20)  # (min, max, n_pts)
    scan_y: tuple[float, float, int] = (-1.0, 1.0, 20)
    scan_z: float = 0.0                    # 2-D scan plane z-position [m]
    # Acoustic settings
    c0: float = _C0_DEFAULT               # speed of sound [m/s]
    # Frequency band of interest
    f_min: float = 100.0                   # Hz
    f_max: float = 5000.0                  # Hz
    # Window / shading
    shading: Literal["uniform", "hamming", "hann"] = "hann"
    # Algorithm
    method: Literal["das", "clean_sc", "damas"] = "das"
    n_iter_clean: int = 10                 # iterations for CLEAN-SC / DAMAS
    clean_loop_gain: float = 0.9           # CLEAN-SC loop gain


@dataclass
class BeamformingResult:
    """Output of a beamforming analysis."""
    source_map: list[list[float]]          # 2-D power map (n_y, n_x) in dB
    x_grid: list[float]
    y_grid: list[float]
    peak_x: float
    peak_y: float
    peak_spl_db: float
    dynamic_range_db: float
    dominant_frequency_hz: float
    method: str


# ---------------------------------------------------------------------------
# Utility: shading weights
# ---------------------------------------------------------------------------

def _shading_weights(n_mics: int, shading: str) -> torch.Tensor:
    """Return (n_mics,) real shading weights, normalised to unit sum."""
    if shading == "hamming":
        w = torch.hamming_window(n_mics, periodic=False, dtype=torch.float64)
    elif shading == "hann":
        w = torch.hann_window(n_mics, periodic=False, dtype=torch.float64)
    else:
        w = torch.ones(n_mics, dtype=torch.float64)
    return w / w.sum()


# ---------------------------------------------------------------------------
# DAS beamformer (frequency domain)
# ---------------------------------------------------------------------------

def das_beamformer(
    array: MicrophoneArray,
    source_positions: torch.Tensor,
    f_min: float = 100.0,
    f_max: float = 5000.0,
    c0: float = _C0_DEFAULT,
    shading: str = "hann",
) -> torch.Tensor:
    """Frequency-domain Delay-and-Sum beamformer.

    Parameters
    ----------
    array:
        Microphone array with signals.
    source_positions:
        (S, 3) tensor of steering positions (scan grid points).
    f_min, f_max:
        Frequency band [Hz].
    c0:
        Speed of sound [m/s].
    shading:
        Aperture shading type.

    Returns
    -------
    (S,) tensor of integrated output power [Pa²] for each steering position.
    """
    M = array.n_mics
    Nt = array.n_samples
    dt = array.dt

    # FFT of all microphone signals
    P = torch.fft.rfft(array.signals.double(), n=Nt, dim=1)   # (M, Nf)
    freqs = torch.fft.rfftfreq(Nt, d=dt)                       # (Nf,)

    # Frequency band mask
    freq_mask = (freqs >= f_min) & (freqs <= f_max)
    P_band = P[:, freq_mask]                         # (M, Nf_band)
    freqs_band = freqs[freq_mask]                    # (Nf_band,)

    w = _shading_weights(M, shading).to(P.device)   # (M,)

    # Source positions: (S, 3)
    S_pos = source_positions.double()    # (S, 3)
    mic_pos = array.positions.double()   # (M, 3)

    # Distances r_ms: (S, M)
    diff = S_pos.unsqueeze(1) - mic_pos.unsqueeze(0)  # (S, M, 3)
    r_ms = diff.norm(dim=2) + 1e-12                    # (S, M)

    power = torch.zeros(S_pos.shape[0], dtype=torch.float64)

    for fi, f in enumerate(freqs_band):
        f_val = f.item()
        k = 2.0 * math.pi * f_val / c0

        # Steering vector: e_m(s) = exp(-i k r_ms) / r_ms  (far-field normalised)
        phase = -k * r_ms                         # (S, M)
        e = torch.exp(1j * torch.tensor(0.0, dtype=torch.float64) + 1j * phase)
        # NOTE: torch complex via polar
        e = torch.polar(torch.ones_like(phase), phase)   # (S, M)

        # Weighted sum: b(s) = Σ_m w_m e_m*(s) P_m(f)
        P_f = P_band[:, fi]                       # (M,) complex
        weighted = w.unsqueeze(0) * e.conj() * P_f.unsqueeze(0)  # (S, M)
        b = weighted.sum(dim=1)                   # (S,) complex

        power += b.abs() ** 2

    return power   # (S,) total power in frequency band


# ---------------------------------------------------------------------------
# 2-D source map
# ---------------------------------------------------------------------------

def compute_source_map(
    array: MicrophoneArray,
    cfg: BeamformingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the 2-D acoustic source power map.

    Returns (power_map, x_grid, y_grid) where power_map is (Ny, Nx).
    """
    x_min, x_max, nx = cfg.scan_x
    y_min, y_max, ny = cfg.scan_y

    x_grid = torch.linspace(x_min, x_max, nx, dtype=torch.float64)
    y_grid = torch.linspace(y_min, y_max, ny, dtype=torch.float64)

    xx, yy = torch.meshgrid(x_grid, y_grid, indexing="xy")  # (Nx, Ny)
    zz = torch.full_like(xx, cfg.scan_z)

    source_pos = torch.stack([
        xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)
    ], dim=1)   # (Nx*Ny, 3)

    power = das_beamformer(
        array,
        source_pos,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        c0=cfg.c0,
        shading=cfg.shading,
    )   # (Nx*Ny,)

    power_map = power.reshape(nx, ny).T   # (Ny, Nx)
    return power_map, x_grid, y_grid


# ---------------------------------------------------------------------------
# CLEAN-SC
# ---------------------------------------------------------------------------

def clean_sc(
    power_map: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    array: MicrophoneArray,
    cfg: BeamformingConfig,
    n_iter: int = 10,
    loop_gain: float = 0.9,
) -> torch.Tensor:
    """CLEAN-SC: iterative subtraction of the dominant source contribution.

    Simple spatial version: subtract a fraction of the peak beam pattern
    (PSF) centred on the strongest source at each iteration.
    """
    residual = power_map.clone()
    clean_map = torch.zeros_like(power_map)

    # Build a simple Gaussian PSF for the array
    dx = (x_grid[-1] - x_grid[0]) / max(len(x_grid) - 1, 1)
    psf_sigma = max(dx * 2.0, 1e-3)

    xx, yy = torch.meshgrid(x_grid, y_grid, indexing="xy")   # (ny, nx)

    for _ in range(n_iter):
        peak_val = residual.max()
        if peak_val <= 0:
            break
        peak_idx = residual.argmax()
        peak_row = peak_idx // residual.shape[1]
        peak_col = peak_idx % residual.shape[1]
        x_peak = x_grid[peak_col]
        y_peak = y_grid[peak_row]

        # Gaussian PSF centred at peak
        psf = torch.exp(
            -((xx - x_peak)**2 + (yy - y_peak)**2) / (2.0 * psf_sigma**2)
        )
        psf = psf / psf.max()

        contribution = loop_gain * peak_val * psf
        clean_map += contribution
        residual = (residual - contribution).clamp(min=0.0)

    return clean_map + residual * 0.1   # add residual floor


# ---------------------------------------------------------------------------
# DAMAS (simplified iterative)
# ---------------------------------------------------------------------------

def damas_deconvolve(
    power_map: torch.Tensor,
    n_iter: int = 50,
) -> torch.Tensor:
    """Simplified DAMAS deconvolution via Gauss-Seidel iteration.

    Solves A q = b where A is a Toeplitz PSF matrix and b is the DAS map.
    This version uses a simplified diagonal-dominant approximation.
    """
    q = power_map.clone().clamp(min=0.0)
    b = power_map.clone()

    # Simplified: use spatial Gaussian blur as the PSF operator A
    kernel_size = min(5, min(power_map.shape) // 2 * 2 + 1)
    sigma = 1.5
    # Build separable Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float64) - kernel_size // 2
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
    kernel = kernel.unsqueeze(0).unsqueeze(0).float()

    for _ in range(n_iter):
        q_f = q.float().unsqueeze(0).unsqueeze(0)
        Aq = torch.nn.functional.conv2d(
            q_f, kernel, padding=kernel_size // 2
        ).squeeze()
        q = (q + b.float() - Aq).clamp(min=0.0)

    return q.double()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_acoustic_beamforming(
    array: MicrophoneArray,
    cfg: BeamformingConfig,
) -> BeamformingResult:
    """Run the full acoustic beamforming analysis.

    Computes DAS source map and optionally applies CLEAN-SC or DAMAS.
    """
    power_map, x_grid, y_grid = compute_source_map(array, cfg)

    # Apply post-processing method
    if cfg.method == "clean_sc":
        power_map = clean_sc(
            power_map, x_grid, y_grid, array, cfg,
            n_iter=cfg.n_iter_clean,
            loop_gain=cfg.clean_loop_gain,
        )
    elif cfg.method == "damas":
        power_map = damas_deconvolve(power_map, n_iter=cfg.n_iter_clean)

    # Convert to dB
    p_ref = 20e-6   # 20 µPa reference
    power_floor = max(float(power_map.max()) * 1e-10, (p_ref**2) * 1e-6)
    spl_map = 10.0 * torch.log10(power_map.clamp(min=power_floor) / p_ref**2)

    # Peak location
    peak_idx = spl_map.argmax()
    peak_row = int(peak_idx // spl_map.shape[1])
    peak_col = int(peak_idx % spl_map.shape[1])
    peak_x = float(x_grid[peak_col])
    peak_y = float(y_grid[peak_row])
    peak_spl = float(spl_map.max())
    dyn_range = float(spl_map.max() - spl_map.min())

    # Dominant frequency from the first microphone spectrum
    P0 = torch.fft.rfft(array.signals[0].double(), n=array.n_samples)
    freqs = torch.fft.rfftfreq(array.n_samples, d=array.dt)
    band = (freqs >= cfg.f_min) & (freqs <= cfg.f_max)
    if band.any():
        dom_freq = float(freqs[band][P0[band].abs().argmax()])
    else:
        dom_freq = 0.0

    return BeamformingResult(
        source_map=spl_map.tolist(),
        x_grid=x_grid.tolist(),
        y_grid=y_grid.tolist(),
        peak_x=peak_x,
        peak_y=peak_y,
        peak_spl_db=peak_spl,
        dynamic_range_db=dyn_range,
        dominant_frequency_hz=dom_freq,
        method=cfg.method,
    )
