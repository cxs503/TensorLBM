"""Generate verification figures for P259W1-N10 rotating acoustics campaign."""

import math
import os
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.acoustics_rotating import (
    RetardedTimeSolver,
    StationaryMonopole,
    StationaryDipole,
    RotatingMonopole,
    RotatingDipole,
    PulsatingSphere,
    compute_doppler_shift,
    compute_bpf_peaks,
    compute_analytical_pressure,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _encode_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Figure 1: Retarded time convergence
# ---------------------------------------------------------------------------


def fig_retarded_time_convergence():
    """Show Newton iteration convergence for moving source."""
    solver = RetardedTimeSolver(max_iter=10, tol=1e-10)
    obs = np.array([100.0, 0.0, 0.0])
    c0 = 343.0

    def traj(t):
        return np.array([10.0 * t, 0.0, 0.0])

    t_obs_list = np.linspace(0.1, 2.0, 50)
    residuals = []

    for t_obs in t_obs_list:
        _, res = solver.solve(obs, traj, t_obs, c0=c0)
        residuals.append(res)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(t_obs_list, residuals, "o-", markersize=4, color="#1f77b4")
    ax.axhline(y=1e-8, color="red", linestyle="--", alpha=0.7, label="1e-8 threshold")
    ax.set_xlabel("Observation time t (s)")
    ax.set_ylabel("Retarded time residual |Δt_r|")
    ax.set_title("Retarded-Time Newton Solver Convergence\n(moving source, v=10 m/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIG_DIR, "retarded_time_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: Monopole distance decay (1/r)
# ---------------------------------------------------------------------------


def fig_monopole_distance_decay():
    """Verify 1/r decay of monopole pressure amplitude."""
    src = StationaryMonopole(
        position=np.array([0.0, 0.0, 0.0]),
        source_strength=1e-3,
        frequency=500.0,
        rho0=1.225,
    )
    c0 = 343.0

    distances = np.linspace(1.0, 50.0, 100)
    amplitudes = []

    for r in distances:
        times = np.linspace(0, 1.0 / 500.0, 200)
        p_vals = np.array([src.pressure_at(np.array([r, 0.0, 0.0]), t, c0=c0) for t in times])
        amplitudes.append(np.max(np.abs(p_vals)))

    amplitudes = np.array(amplitudes)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(distances, amplitudes, "o-", markersize=4, color="#1f77b4", label="Computed")
    # Reference 1/r line
    ref = amplitudes[0] * distances[0] / distances
    ax.loglog(distances, ref, "r--", alpha=0.7, label="1/r reference")
    ax.set_xlabel("Distance r (m)")
    ax.set_ylabel("Peak pressure amplitude (Pa)")
    ax.set_title("Monopole Distance Decay: p ∝ 1/r")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    path = os.path.join(FIG_DIR, "monopole_distance_decay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: Dipole directivity pattern
# ---------------------------------------------------------------------------


def fig_dipole_directivity():
    """Show cos(θ) directivity of dipole source."""
    src = StationaryDipole(
        position=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        force_amplitude=1.0,
        frequency=500.0,
    )
    c0 = 343.0
    r = 10.0
    t = 0.005

    angles = np.linspace(0, 2 * math.pi, 200)
    pressures = []

    for theta in angles:
        obs = np.array([r * math.cos(theta), r * math.sin(theta), 0.0])
        pressures.append(src.pressure_at(obs, t, c0=c0))

    pressures = np.array(pressures)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Cartesian plot
    ax1.plot(np.degrees(angles), pressures, "-", color="#1f77b4")
    ax1.plot(np.degrees(angles), np.cos(angles) * np.max(np.abs(pressures)), "r--", alpha=0.7, label="cos(θ)")
    ax1.set_xlabel("Angle θ (degrees)")
    ax1.set_ylabel("Pressure (Pa)")
    ax1.set_title("Dipole Directivity: p ∝ cos(θ)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Polar plot
    ax2 = fig.add_subplot(122, projection="polar")
    ax2.plot(angles, np.abs(pressures), "-", color="#1f77b4")
    ax2.set_title("Dipole Polar Pattern")

    path = os.path.join(FIG_DIR, "dipole_directivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4: Doppler frequency shift
# ---------------------------------------------------------------------------


def fig_doppler_shift():
    """Doppler shift vs source velocity."""
    f_src = 1000.0
    c0 = 343.0
    obs_pos = np.array([100.0, 0.0, 0.0])
    source_pos = np.array([0.0, 0.0, 0.0])

    velocities = np.linspace(-100, 100, 200)
    f_obs = []

    for v in velocities:
        f = compute_doppler_shift(
            f_src, np.array([v, 0.0, 0.0]), obs_pos, source_pos, c0=c0
        )
        f_obs.append(f)

    f_obs = np.array(f_obs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(velocities, f_obs, "-", color="#1f77b4", linewidth=2)
    ax.axhline(y=f_src, color="red", linestyle="--", alpha=0.7, label="Source frequency")
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Source velocity (m/s)")
    ax.set_ylabel("Observed frequency (Hz)")
    ax.set_title(f"Doppler Frequency Shift (f_src = {f_src} Hz, c₀ = {c0} m/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIG_DIR, "doppler_shift.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 5: BPF spectrum for 4-blade propeller
# ---------------------------------------------------------------------------


def fig_bpf_spectrum():
    """BPF spectrum showing fundamental and harmonics."""
    rpm = 1200.0
    n_blades = 4
    bpf = n_blades * rpm / 60.0  # 80 Hz

    dt = 1.0 / 8000.0
    T = 8000
    times = np.arange(T) * dt

    # Realistic propeller noise: BPF + harmonics with decay
    signal = np.zeros(T)
    for h in range(1, 6):
        amp = 0.01 / (h ** 1.5)  # harmonic decay
        signal += amp * np.sin(2 * math.pi * h * bpf * times)

    # Add small broadband noise
    signal += 0.0005 * np.random.randn(T)

    peaks = compute_bpf_peaks(signal, dt, rpm, n_blades, max_harmonic=6)

    # FFT spectrum
    window = np.hanning(T)
    fft_vals = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(T, d=dt)
    psd = np.abs(fft_vals) ** 2 / (T * T)
    psd[1:-1] *= 2.0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(freqs, psd, "-", color="#1f77b4", alpha=0.7, linewidth=0.5, label="PSD")

    # Mark BPF peaks
    for p in peaks:
        ax.axvline(x=p.frequency, color="red", linestyle="--", alpha=0.5)
        ax.annotate(
            f"{p.harmonic}×BPF\n{p.frequency:.1f} Hz",
            xy=(p.frequency, psd[np.argmin(np.abs(freqs - p.frequency))]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            color="red",
        )

    ax.axvline(x=bpf, color="green", linestyle=":", alpha=0.7, label=f"BPF = {bpf:.0f} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (Pa²/Hz)")
    ax.set_title(f"Propeller Noise Spectrum (Z={n_blades}, RPM={rpm:.0f})")
    ax.set_xlim(0, 600)
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIG_DIR, "bpf_spectrum.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 6: U^6 scaling
# ---------------------------------------------------------------------------


def fig_u6_scaling():
    """Acoustic power U^6 scaling for Curle dipole."""
    velocities = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    powers = []
    base_freq = 100.0

    from tensorlbm.acoustics_rotating import StationaryDipole

    for U in velocities:
        freq = base_freq * U
        src = StationaryDipole(
            position=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            force_amplitude=U * U,
            frequency=freq,
        )
        obs = np.array([10.0, 0.0, 0.0])
        n_cycles = 20
        T_sig = n_cycles / freq
        n_steps = max(500, int(T_sig / 0.0001))
        times = np.linspace(0, T_sig, n_steps)
        pressures = np.array([src.pressure_at(obs, t) for t in times])
        p_rms = np.sqrt(np.mean(pressures ** 2))
        powers.append(p_rms ** 2)

    powers = np.array(powers)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(velocities, powers, "o-", markersize=8, color="#1f77b4", label="Computed")

    # Reference U^6 line
    ref = powers[0] * (velocities / velocities[0]) ** 6
    ax.loglog(velocities, ref, "r--", alpha=0.7, label="U⁶ reference")

    # Fit
    log_v = np.log(velocities)
    log_p = np.log(powers)
    coeffs = np.polyfit(log_v, log_p, 1)
    exponent = coeffs[0]
    r_squared = 1 - np.sum((log_p - np.polyval(coeffs, log_v)) ** 2) / np.sum((log_p - np.mean(log_p)) ** 2)

    ax.set_xlabel("Flow velocity U (m/s)")
    ax.set_ylabel("Acoustic power P ∝ p_rms²")
    ax.set_title(f"Curle Dipole U⁶ Scaling\nExponent = {exponent:.2f}, R² = {r_squared:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    path = os.path.join(FIG_DIR, "u6_scaling.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 7: Rotating source pressure time series
# ---------------------------------------------------------------------------


def fig_rotating_source_time_series():
    """Pressure time series from rotating monopole at different observer angles."""
    src = RotatingMonopole(
        radius=0.5,
        omega_rot=2 * math.pi * 20,  # 20 rev/s
        source_strength=1e-3,
        frequency=500.0,
        rho0=1.225,
    )
    c0 = 343.0
    r_obs = 10.0

    angles = [0, math.pi / 4, math.pi / 2]
    labels = ["0° (upstream)", "45°", "90° (broadside)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for angle, label, color in zip(angles, labels, colors):
        obs = np.array([r_obs * math.cos(angle), r_obs * math.sin(angle), 0.0])
        times = np.linspace(0, 0.1, 500)
        pressures = np.array([src.pressure_at(obs, t, c0=c0) for t in times])
        ax.plot(times * 1000, pressures, "-", color=color, linewidth=1, label=label)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Acoustic pressure (Pa)")
    ax.set_title("Rotating Monopole: Pressure Time Series at Different Observer Angles")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIG_DIR, "rotating_time_series.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("Generating verification figures...")
    paths = []
    for fn in [
        fig_retarded_time_convergence,
        fig_monopole_distance_decay,
        fig_dipole_directivity,
        fig_doppler_shift,
        fig_bpf_spectrum,
        fig_u6_scaling,
        fig_rotating_source_time_series,
    ]:
        p = fn()
        paths.append(p)
        print(f"  → {p}")
    print(f"\nGenerated {len(paths)} figures in {FIG_DIR}")
