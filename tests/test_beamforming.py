"""Unit tests for the acoustic beamforming module.

Tests cover:
  1. DAS localisation of a single monopole source
  2. DAS localisation of two spatially separated sources
  3. CLEAN-SC deconvolution producing a cleaner map than DAS
  4. DAMAS deconvolution producing a sharper map than DAS
  5. compute_source_map basic structure / shapes
  6. run_acoustic_beamforming end-to-end return type

Synthetic data is generated with the free-space Green's function so that the
true source positions are known exactly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.acoustic_beamforming import (
    BeamformingConfig,
    BeamformingResult,
    MicrophoneArray,
    _shading_weights,
    clean_sc,
    compute_source_map,
    damas_deconvolve,
    das_beamformer,
    run_acoustic_beamforming,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C0 = 343.0  # speed of sound [m/s]


# ---------------------------------------------------------------------------
# Helper: synthetic signal generation
# ---------------------------------------------------------------------------

def create_synthetic_signals(
    source_pos: torch.Tensor,
    mic_positions: torch.Tensor,
    freq: float,
    dt: float,
    n_samples: int,
    c0: float = C0,
) -> torch.Tensor:
    """Generate pressure signals at each microphone using the Green's function.

    p_m(t) = A * sin(2*pi*f*(t - r_m/c0)) / r_m

    where r_m = |source - mic_m| is the distance from the source to
    microphone *m*.

    Parameters
    ----------
    source_pos : (3,) or (S, 3) tensor
        Source position(s).  When 2-D, signals from all sources are
        superposed coherently.
    mic_positions : (M, 3) tensor
        Microphone positions.
    freq : float
        Source frequency [Hz].
    dt : float
        Time step [s].
    n_samples : int
        Number of time samples.
    c0 : float
        Speed of sound [m/s].

    Returns
    -------
    (M, n_samples) tensor of pressure signals [Pa].
    """
    t = torch.arange(n_samples, dtype=torch.float64) * dt

    if source_pos.dim() == 1:
        source_pos = source_pos.unsqueeze(0)

    M = mic_positions.shape[0]
    signals = torch.zeros(M, n_samples, dtype=torch.float64)

    for s in range(source_pos.shape[0]):
        r = (mic_positions.double() - source_pos[s].double()).norm(dim=1)
        delay = (r / c0).unsqueeze(1)          # (M, 1)
        amp = (1.0 / r).unsqueeze(1)           # (M, 1)
        signals += amp * torch.sin(
            2.0 * math.pi * freq * (t.unsqueeze(0) - delay)
        )

    return signals


def _make_linear_array(
    n_mics: int = 8,
    spacing: float = 0.1,
    y: float = 2.0,
    dt: float = 1e-4,
    n_samples: int = 1024,
    freq: float = 1000.0,
    source_pos: torch.Tensor | None = None,
    c0: float = C0,
) -> MicrophoneArray:
    """Build a linear microphone array (along x) with synthetic source signals."""
    positions = torch.zeros(n_mics, 3, dtype=torch.float64)
    positions[:, 0] = torch.linspace(
        -(n_mics - 1) * spacing / 2, (n_mics - 1) * spacing / 2, n_mics
    )
    positions[:, 1] = y

    if source_pos is None:
        source_pos = torch.tensor([0.0, 0.0, 0.0])

    signals = create_synthetic_signals(source_pos, positions, freq, dt, n_samples, c0)
    return MicrophoneArray(positions=positions, signals=signals, dt=dt)


def _find_peak(power_map: torch.Tensor, x_grid: torch.Tensor, y_grid: torch.Tensor):
    """Return (peak_x, peak_y) for a (ny, nx) power map."""
    peak_idx = power_map.argmax()
    peak_row = int(peak_idx // power_map.shape[1])
    peak_col = int(peak_idx % power_map.shape[1])
    return float(x_grid[peak_col]), float(y_grid[peak_row])


def _find_peaks_nms(
    smap: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    min_dist: float = 0.3,
    n_peaks: int = 2,
):
    """Find local maxima with non-maximum suppression.

    More robust than iterative masking when sidelobe ridges connect sources.
    """
    ny, nx = smap.shape
    candidates = []
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            v = smap[i, j]
            if (
                v >= smap[i - 1, j]
                and v >= smap[i + 1, j]
                and v >= smap[i, j - 1]
                and v >= smap[i, j + 1]
            ):
                candidates.append((float(x_grid[j]), float(y_grid[i]), float(v)))
    candidates.sort(key=lambda p: p[2], reverse=True)

    peaks = []
    for px, py, _val in candidates:
        if all(math.hypot(px - ex, py - ey) >= min_dist for ex, ey, _ in peaks):
            peaks.append((px, py, _val))
            if len(peaks) >= n_peaks:
                break
    return [(p[0], p[1]) for p in peaks]


def _dynamic_range_db(pmap: torch.Tensor) -> float:
    """Peak-to-mean dynamic range in dB (higher = cleaner map)."""
    peak = float(pmap.max())
    mean = float(pmap.mean())
    if mean <= 0:
        return float("inf")
    return 10.0 * math.log10(peak / mean)


def _spot_size(pmap: torch.Tensor) -> int:
    """Number of grid points above half the peak power (smaller = sharper)."""
    peak = float(pmap.max())
    if peak <= 0:
        return 0
    return int((pmap >= 0.5 * peak).sum().item())


# ---------------------------------------------------------------------------
# Test 1: DAS single source
# ---------------------------------------------------------------------------

def test_das_single_source():
    """DAS beamformer localises a single monopole source within 0.05 m.

    Source at (0.5, 0.3, 0); 8-mic linear array at y=2.0, spacing 0.1 m;
    dt=1e-4, 1024 samples, 1000 Hz.
    """
    source_pos = torch.tensor([0.5, 0.3, 0.0])
    array = _make_linear_array(
        n_mics=8, spacing=0.1, y=2.0, dt=1e-4, n_samples=1024,
        freq=1000.0, source_pos=source_pos,
    )

    cfg = BeamformingConfig(
        scan_x=(0.2, 0.8, 31),
        scan_y=(0.0, 0.6, 31),
        scan_z=0.0,
        c0=C0,
        f_min=800.0,
        f_max=1200.0,
        shading="hann",
        method="das",
    )

    power_map, x_grid, y_grid = compute_source_map(array, cfg)
    peak_x, peak_y = _find_peak(power_map, x_grid, y_grid)

    err = math.hypot(peak_x - 0.5, peak_y - 0.3)
    assert err < 0.05, (
        f"DAS peak ({peak_x:.3f}, {peak_y:.3f}) is {err:.3f} m from true "
        f"source (0.5, 0.3)"
    )


# ---------------------------------------------------------------------------
# Test 2: DAS two sources
# ---------------------------------------------------------------------------

def test_das_two_sources():
    """DAS finds both peaks for two well-separated incoherent sources.

    Two sources at (-0.4, 0.3) and (0.4, 0.3) — 0.8 m apart, well beyond
    the array's Rayleigh resolution.  Different frequencies (1500 / 2500 Hz)
    prevent coherent interference artefacts.
    """
    source1 = torch.tensor([-0.4, 0.3, 0.0])
    source2 = torch.tensor([0.4, 0.3, 0.0])
    freq1, freq2 = 1500.0, 2500.0

    n_mics, spacing, dt, n_samples = 16, 0.1, 1e-4, 1024
    mic_positions = torch.zeros(n_mics, 3, dtype=torch.float64)
    mic_positions[:, 0] = torch.linspace(
        -(n_mics - 1) * spacing / 2, (n_mics - 1) * spacing / 2, n_mics
    )
    mic_positions[:, 1] = 2.0

    signals = (
        create_synthetic_signals(source1, mic_positions, freq1, dt, n_samples)
        + create_synthetic_signals(source2, mic_positions, freq2, dt, n_samples)
    )
    array = MicrophoneArray(positions=mic_positions, signals=signals, dt=dt)

    cfg = BeamformingConfig(
        scan_x=(-0.7, 0.7, 41),
        scan_y=(0.0, 0.6, 21),
        scan_z=0.0,
        c0=C0,
        f_min=1200.0,
        f_max=2800.0,
        shading="hann",
        method="das",
    )

    power_map, x_grid, y_grid = compute_source_map(array, cfg)

    # Find peaks via local-maxima + NMS (robust to sidelobe ridges)
    peaks = _find_peaks_nms(
        power_map.numpy(),
        x_grid.numpy(),
        y_grid.numpy(),
        min_dist=0.3,
        n_peaks=2,
    )

    sources = [(-0.4, 0.3), (0.4, 0.3)]

    # Each true source must have a detected peak within 0.1 m
    for sx, sy in sources:
        best = min(math.hypot(px - sx, py - sy) for px, py in peaks)
        assert best < 0.1, (
            f"No peak within 0.1 m of source ({sx:.2f}, {sy:.2f}); "
            f"detected peaks: {peaks}"
        )

    # Must find at least 2 distinct peaks
    assert len(peaks) >= 2, f"Expected ≥2 peaks, got {len(peaks)}: {peaks}"


# ---------------------------------------------------------------------------
# Test 3: CLEAN-SC
# ---------------------------------------------------------------------------

def test_clean_sc():
    """CLEAN-SC localises the source and yields a cleaner map than DAS.

    Verified by: (a) peak position within 0.05 m of true source, and
    (b) peak-to-mean dynamic range higher than DAS.
    """
    source_pos = torch.tensor([0.5, 0.3, 0.0])
    array = _make_linear_array(
        n_mics=8, spacing=0.1, y=2.0, dt=1e-4, n_samples=1024,
        freq=1000.0, source_pos=source_pos,
    )

    cfg = BeamformingConfig(
        scan_x=(0.2, 0.8, 31),
        scan_y=(0.0, 0.6, 31),
        scan_z=0.0,
        c0=C0,
        f_min=800.0,
        f_max=1200.0,
        shading="hann",
        method="clean_sc",
        n_iter_clean=10,
        clean_loop_gain=0.9,
    )

    das_map, x_grid, y_grid = compute_source_map(array, cfg)
    clean_map = clean_sc(
        das_map, x_grid, y_grid, array, cfg,
        n_iter=cfg.n_iter_clean,
        loop_gain=cfg.clean_loop_gain,
    )

    # --- Peak position ---
    peak_x, peak_y = _find_peak(clean_map, x_grid, y_grid)
    err = math.hypot(peak_x - 0.5, peak_y - 0.3)
    assert err < 0.05, (
        f"CLEAN-SC peak ({peak_x:.3f}, {peak_y:.3f}) is {err:.3f} m from "
        f"true source (0.5, 0.3)"
    )

    # --- Dynamic range should be higher (cleaner map) ---
    dr_das = _dynamic_range_db(das_map)
    dr_clean = _dynamic_range_db(clean_map)
    assert dr_clean > dr_das, (
        f"CLEAN-SC dynamic range ({dr_clean:.2f} dB) not higher than "
        f"DAS ({dr_das:.2f} dB)"
    )


# ---------------------------------------------------------------------------
# Test 4: DAMAS
# ---------------------------------------------------------------------------

def test_damas():
    """DAMAS deconvolution localises the source and sharpens the map.

    Uses a 16-mic array at 2000 Hz with uniform shading and a large scan
    grid to keep the source far from edges (the simplified DAMAS conv2d
    operator uses zero-padding, which can amplify edge values over many
    iterations).  Verified by: (a) peak position within 0.05 m, and
    (b) half-max spot size smaller than DAS.
    """
    source_pos = torch.tensor([0.5, 0.3, 0.0])
    array = _make_linear_array(
        n_mics=16, spacing=0.1, y=2.0, dt=1e-4, n_samples=1024,
        freq=2000.0, source_pos=source_pos,
    )

    cfg = BeamformingConfig(
        scan_x=(-1.5, 2.5, 61),
        scan_y=(-1.5, 2.5, 61),
        scan_z=0.0,
        c0=C0,
        f_min=1800.0,
        f_max=2200.0,
        shading="uniform",
        method="damas",
        n_iter_clean=2,
    )

    das_map, x_grid, y_grid = compute_source_map(array, cfg)
    damas_map = damas_deconvolve(das_map, n_iter=cfg.n_iter_clean)

    # --- Peak position ---
    peak_x, peak_y = _find_peak(damas_map, x_grid, y_grid)
    err = math.hypot(peak_x - 0.5, peak_y - 0.3)
    assert err < 0.05, (
        f"DAMAS peak ({peak_x:.3f}, {peak_y:.3f}) is {err:.3f} m from "
        f"true source (0.5, 0.3)"
    )

    # --- Spot size should be smaller (sharper map) ---
    ss_das = _spot_size(das_map)
    ss_damas = _spot_size(damas_map)
    assert ss_damas < ss_das, (
        f"DAMAS spot size ({ss_damas}) not smaller than DAS ({ss_das})"
    )


# ---------------------------------------------------------------------------
# Test 5: compute_source_map basic
# ---------------------------------------------------------------------------

def test_source_map_basic():
    """compute_source_map returns (power_map, x_grid, y_grid) with correct shapes.

    The function returns a 3-tuple of tensors (not a BeamformingResult);
    the latter is only produced by :func:`run_acoustic_beamforming`.
    """
    # Source off broadside so the linear array can resolve it in x
    source_pos = torch.tensor([0.5, 0.3, 0.0])
    array = _make_linear_array(
        n_mics=8, spacing=0.1, y=2.0, dt=1e-4, n_samples=1024,
        freq=1000.0, source_pos=source_pos,
    )

    cfg = BeamformingConfig(
        scan_x=(0.2, 0.8, 20),
        scan_y=(0.0, 0.6, 20),
        scan_z=0.0,
        c0=C0,
        f_min=800.0,
        f_max=1200.0,
        shading="hann",
        method="das",
    )

    result = compute_source_map(array, cfg)
    assert isinstance(result, tuple) and len(result) == 3
    power_map, x_grid, y_grid = result

    # Shapes: power_map is (ny, nx)
    assert power_map.shape == (20, 20)
    assert x_grid.shape == (20,)
    assert y_grid.shape == (20,)

    # All finite
    assert torch.isfinite(power_map).all()
    assert torch.isfinite(x_grid).all()
    assert torch.isfinite(y_grid).all()

    # Grid endpoints match config
    assert abs(float(x_grid[0]) - 0.2) < 1e-10
    assert abs(float(x_grid[-1]) - 0.8) < 1e-10
    assert abs(float(y_grid[0]) - 0.0) < 1e-10
    assert abs(float(y_grid[-1]) - 0.6) < 1e-10

    # Peak should be near the true source — coarser 20×20 grid → 0.1 m tol
    peak_x, peak_y = _find_peak(power_map, x_grid, y_grid)
    err = math.hypot(peak_x - 0.5, peak_y - 0.3)
    assert err < 0.1, (
        f"Peak ({peak_x:.3f}, {peak_y:.3f}) is {err:.3f} m from true "
        f"source (0.5, 0.3)"
    )


# ---------------------------------------------------------------------------
# Test 6: run_acoustic_beamforming end-to-end
# ---------------------------------------------------------------------------

def test_run_acoustic_beamforming():
    """run_acoustic_beamforming returns a fully populated BeamformingResult."""
    source_pos = torch.tensor([0.5, 0.3, 0.0])
    array = _make_linear_array(
        n_mics=8, spacing=0.1, y=2.0, dt=1e-4, n_samples=1024,
        freq=1000.0, source_pos=source_pos,
    )

    cfg = BeamformingConfig(
        scan_x=(0.2, 0.8, 31),
        scan_y=(0.0, 0.6, 31),
        scan_z=0.0,
        c0=C0,
        f_min=800.0,
        f_max=1200.0,
        shading="hann",
        method="das",
    )

    result = run_acoustic_beamforming(array, cfg)

    assert isinstance(result, BeamformingResult)
    assert result.method == "das"

    # source_map is a 2-D list (ny, nx)
    assert len(result.source_map) == 31
    assert len(result.source_map[0]) == 31

    assert len(result.x_grid) == 31
    assert len(result.y_grid) == 31

    # Peak near true source
    err = math.hypot(result.peak_x - 0.5, result.peak_y - 0.3)
    assert err < 0.05, (
        f"Peak ({result.peak_x:.3f}, {result.peak_y:.3f}) is {err:.3f} m "
        f"from true source (0.5, 0.3)"
    )

    # Scalar attributes are finite and sensible
    assert math.isfinite(result.peak_spl_db)
    assert math.isfinite(result.dynamic_range_db)
    assert math.isfinite(result.dominant_frequency_hz)
    assert result.peak_spl_db > 0
    assert result.dynamic_range_db > 0
    # Dominant frequency should be near 1000 Hz
    assert abs(result.dominant_frequency_hz - 1000.0) < 50.0


# ---------------------------------------------------------------------------
# Bonus: shading weights
# ---------------------------------------------------------------------------

def test_shading_weights():
    """Uniform / hamming / hann shading produce valid normalised weights."""
    n_mics = 12

    for shading in ("uniform", "hamming", "hann"):
        w = _shading_weights(n_mics, shading)

        # All weights non-negative
        assert (w >= 0).all(), f"{shading}: weights must be non-negative, got {w}"

        # Normalised to unit sum
        assert math.isclose(float(w.sum()), 1.0, abs_tol=1e-12), (
            f"{shading}: weights must sum to 1, got {float(w.sum()):.6f}"
        )

        # Correct length
        assert w.shape == (n_mics,), f"{shading}: wrong shape {w.shape}"

    # Uniform: all equal
    w_u = _shading_weights(n_mics, "uniform")
    assert torch.allclose(w_u, torch.full_like(w_u, 1.0 / n_mics)), (
        "uniform weights should all be 1/M"
    )

    # Hamming / Hann: tapered (edge < centre)
    w_h = _shading_weights(n_mics, "hamming")
    w_n = _shading_weights(n_mics, "hann")
    assert w_h[0] < w_h[n_mics // 2], "hamming: edge should be < centre"
    assert w_n[0] < w_n[n_mics // 2], "hann: edge should be < centre"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
