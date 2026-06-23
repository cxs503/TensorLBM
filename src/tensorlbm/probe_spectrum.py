"""Spectral analysis of probe time-history signals.

Provides FFT-based Power Spectral Density (PSD) analysis, dominant-frequency
extraction, and Strouhal-number estimation for flow probe signals recorded
during LBM simulations.  This is a standard post-processing capability in
commercial tools such as PowerFlow and XFlow.

Background
----------
Given a discrete time series q[n] sampled at a uniform interval Δt, the
one-sided power spectral density is estimated as::

    PSD[k] = 2 |FFT[k]|² / (N · f_s)     k = 1, …, N/2

where N is the number of samples and f_s = 1/Δt is the sampling frequency.
Hanning windowing is applied to reduce spectral leakage.

For periodic vortex shedding the dominant Strouhal number is::

    St = f_peak · D / U_ref

References
----------
Welch, P.D. (1967). "The use of fast Fourier transform for the estimation of
    power spectra." *IEEE Trans. Audio Electroacoust.* AU-15, 70–73.
Williamson, C.H.K. (1996). "Vortex dynamics in the cylinder wake."
    *Annu. Rev. Fluid Mech.* 28, 477–539.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

__all__ = [
    "ProbeSpectrum",
    "compute_probe_spectrum",
    "dominant_peaks",
    "strouhal_number",
    "welch_psd",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProbeSpectrum:
    """Container for spectral analysis results of a single probe signal.

    Attributes
    ----------
    frequencies:
        One-sided frequency array [Hz or LBM steps⁻¹].
    psd:
        Power spectral density at each frequency.
    peak_frequencies:
        Sorted list of dominant peak frequencies (highest PSD first).
    peak_psd:
        PSD values at the dominant peaks.
    f_nyquist:
        Nyquist frequency = 0.5 / dt.
    n_samples:
        Number of time samples used.
    dt:
        Time-step interval.
    signal_rms:
        RMS of the fluctuating component (mean subtracted).
    strouhal:
        Strouhal number St = f_peak * D / U_ref, or None if not provided.
    """

    frequencies: list[float]
    psd: list[float]
    peak_frequencies: list[float]
    peak_psd: list[float]
    f_nyquist: float
    n_samples: int
    dt: float
    signal_rms: float
    strouhal: float | None = None


# ---------------------------------------------------------------------------
# Core DSP helpers
# ---------------------------------------------------------------------------

def _hanning_window(n: int, device: torch.device) -> torch.Tensor:
    """Return a Hanning window of length n."""
    k = torch.arange(n, dtype=torch.float64, device=device)
    return 0.5 * (1.0 - torch.cos(2.0 * math.pi * k / (n - 1)))


def welch_psd(
    signal: torch.Tensor,
    dt: float = 1.0,
    *,
    n_segment: int | None = None,
    overlap: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate one-sided PSD via Welch's averaged-periodogram method.

    Parameters
    ----------
    signal:
        1-D real time series tensor of shape ``(N,)``.
    dt:
        Sampling interval (physical or LBM units).
    n_segment:
        Number of samples per segment.  Defaults to ``len(signal) // 4``,
        rounded up to a power of two.
    overlap:
        Fractional overlap between consecutive segments (0–1).

    Returns
    -------
    freqs : torch.Tensor
        One-sided frequency array, shape ``(n_segment // 2 + 1,)``.
    psd : torch.Tensor
        One-sided PSD, same shape as *freqs*.
    """
    signal = signal.double()
    n = len(signal)
    if n_segment is None:
        seg = max(8, n // 4)
        # round up to next power of two
        seg = 1 << (seg - 1).bit_length()
    else:
        seg = int(n_segment)

    step = max(1, int(seg * (1.0 - overlap)))
    window = _hanning_window(seg, signal.device)
    win_power = float((window ** 2).mean())

    f_s = 1.0 / dt
    freqs = torch.fft.rfftfreq(seg, d=dt).to(signal.device)

    accumulated = torch.zeros(len(freqs), dtype=torch.float64, device=signal.device)
    count = 0
    start = 0
    while start + seg <= n:
        chunk = signal[start : start + seg] - signal[start : start + seg].mean()
        windowed = chunk * window
        spec = torch.fft.rfft(windowed)
        power = (spec.abs() ** 2) / (f_s * seg * win_power)
        # double one-sided (exclude DC and Nyquist)
        power[1:-1] *= 2.0
        accumulated += power
        count += 1
        start += step

    if count == 0:
        # fallback: single segment with zero-padding
        padded = torch.zeros(seg, dtype=torch.float64, device=signal.device)
        padded[: min(n, seg)] = signal[: min(n, seg)]
        spec = torch.fft.rfft(padded - padded.mean())
        power = (spec.abs() ** 2) / (f_s * seg)
        power[1:-1] *= 2.0
        accumulated = power
        count = 1

    psd = accumulated / count
    return freqs, psd


def _find_peaks(
    values: torch.Tensor,
    n_peaks: int = 5,
    *,
    min_prominence: float = 0.0,
) -> tuple[list[int], list[float]]:
    """Locate local maxima in a 1-D tensor by simple prominence filtering."""
    vals = values.tolist()
    n = len(vals)
    candidates: list[tuple[float, int]] = []
    for i in range(1, n - 1):
        if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
            prominence = vals[i] - min(vals[i - 1], vals[i + 1])
            if prominence >= min_prominence:
                candidates.append((vals[i], i))

    candidates.sort(key=lambda x: x[0], reverse=True)
    indices = [c[1] for c in candidates[:n_peaks]]
    heights = [c[0] for c in candidates[:n_peaks]]
    return indices, heights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_probe_spectrum(
    signal: Sequence[float] | torch.Tensor,
    dt: float = 1.0,
    *,
    n_segment: int | None = None,
    overlap: float = 0.5,
    n_peaks: int = 5,
    diameter: float | None = None,
    u_ref: float | None = None,
) -> ProbeSpectrum:
    """Compute the PSD and dominant peaks for a probe time series.

    Parameters
    ----------
    signal:
        1-D time series (e.g. lift coefficient, pressure, or velocity).
    dt:
        Sampling interval (time step or physical dt).
    n_segment:
        Welch segment length.
    overlap:
        Welch segment overlap fraction.
    n_peaks:
        Number of dominant peaks to report.
    diameter:
        Characteristic length D for Strouhal calculation.
    u_ref:
        Reference velocity U for Strouhal calculation.

    Returns
    -------
    ProbeSpectrum
        Structured result with frequencies, PSD, peaks, and Strouhal.
    """
    if not isinstance(signal, torch.Tensor):
        sig_t = torch.tensor(list(signal), dtype=torch.float64)
    else:
        sig_t = signal.double().flatten()

    n = len(sig_t)
    sig_t_centered = sig_t - sig_t.mean()
    signal_rms = float(sig_t_centered.pow(2).mean().sqrt())

    freqs, psd = welch_psd(sig_t, dt=dt, n_segment=n_segment, overlap=overlap)

    freqs_list = freqs.tolist()
    psd_list = psd.tolist()

    # skip DC component (index 0) when finding peaks
    psd_no_dc = psd.clone()
    if len(psd_no_dc) > 1:
        psd_no_dc[0] = 0.0

    peak_idxs, peak_heights = _find_peaks(psd_no_dc, n_peaks=n_peaks)
    peak_freqs = [freqs_list[i] for i in peak_idxs]

    f_peak = peak_freqs[0] if peak_freqs else 0.0
    st: float | None = None
    if diameter is not None and u_ref is not None and u_ref > 0.0 and f_peak > 0.0:
        st = f_peak * diameter / u_ref

    return ProbeSpectrum(
        frequencies=freqs_list,
        psd=psd_list,
        peak_frequencies=peak_freqs,
        peak_psd=peak_heights,
        f_nyquist=0.5 / dt,
        n_samples=n,
        dt=dt,
        signal_rms=signal_rms,
        strouhal=st,
    )


def dominant_peaks(
    spectrum: ProbeSpectrum,
    n: int = 3,
) -> list[dict[str, float]]:
    """Return the top-n dominant frequency peaks as a list of dicts."""
    top = min(n, len(spectrum.peak_frequencies))
    return [
        {"frequency": spectrum.peak_frequencies[i], "psd": spectrum.peak_psd[i]}
        for i in range(top)
    ]


def strouhal_number(
    f_peak: float,
    diameter: float,
    u_ref: float,
) -> float:
    """Compute Strouhal number St = f * D / U."""
    if u_ref <= 0.0:
        raise ValueError("u_ref must be positive")
    return f_peak * diameter / u_ref
