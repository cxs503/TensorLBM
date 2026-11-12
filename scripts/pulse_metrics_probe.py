"""Measure the 2-D LBM acoustic pulse against the exact Hankel reference."""
import json
import time

import numpy as np

from tensorlbm.acoustic_pulse_2d import (
    CS,
    anisotropy_residual,
    envelope_exponent,
    exact_pulse_radial,
    fd_wave_pulse,
    radial_profile,
    simulate_pulse_2d,
    travel_time_speed,
    waveform_error,
)

PROBES = (30.0, 50.0, 70.0, 90.0)


def measure(sigma, nx=384, n_steps=180, tau=0.52, sample_every=2):
    t0 = time.time()
    sim = simulate_pulse_2d(
        nx=nx, ny=nx, tau=tau, n_steps=n_steps, sigma=sigma,
        amplitude=1e-4, sample_every=sample_every,
        probe_radii=PROBES, snapshots_at=(120, n_steps),
    )
    wall = time.time() - t0
    times = sim["times"]
    radii = sim["probe_radii"]
    # like-for-like reference: same radii, same sampled times
    ex_series = np.array([
        [float(exact_pulse_radial(np.array([r]), float(t), sigma, 1e-4)[0]) for r in radii]
        for t in times
    ])
    sp_sim = travel_time_speed(times, sim["probe_series"], radii)
    sp_ex = travel_time_speed(times, ex_series, radii)
    env_sim = envelope_exponent(radii, np.max(np.abs(sim["probe_series"]), axis=0), r_lo=30.0)
    env_ex = envelope_exponent(radii, np.max(np.abs(ex_series), axis=0), r_lo=30.0)

    out = {
        "sigma": sigma, "nx": nx, "n_steps": n_steps, "tau": tau,
        "nu": sim["nu"], "wall_s": wall, "finite": sim["finite"],
        "max_mach": sim["max_speed"] / CS,
        "front_radius": sim["front_radius"],
        "speed": {"sim": sp_sim["c_meas"], "exact": sp_ex["c_meas"],
                  "cs_theory": CS,
                  "err_sim_vs_cs_pct": 100.0 * (sp_sim["c_meas"] - CS) / CS,
                  "err_sim_vs_exact_pct": 100.0 * (sp_sim["c_meas"] - sp_ex["c_meas"]) / sp_ex["c_meas"],
                  "sim_spread_pct": sp_sim["c_spread_pct"]},
        "envelope": {"sim": env_sim, "exact": env_ex,
                     "exp_err_pct": 100.0 * (env_sim["exponent"] - env_ex["exponent"]) / abs(env_ex["exponent"])},
        "waveform": {},
        "anisotropy_pct": {},
    }
    for step in sorted(sim["snapshots"]):
        p = sim["snapshots"][step]
        r, mean, _ = radial_profile(p, rmax=0.9 * sim["nx"] / 2)
        ref = exact_pulse_radial(r, float(step), sigma, 1e-4)
        hi = 0.80 * CS * step  # resolved region: inside the light cone
        out["waveform"][str(step)] = waveform_error(mean, ref, r=r, r_lo=3.0, r_hi=hi)
        out["anisotropy_pct"][str(step)] = anisotropy_residual(p, r_lo=10.0, r_hi=hi)
    # reference self-validation: FD wave-equation solver vs Hankel integral
    fd = fd_wave_pulse(n=257, sigma=8.0, amplitude=1.0, courant=0.5, n_steps=100)
    rr = np.arange(0.0, 60.5, 0.5)
    ex = exact_pulse_radial(rr, fd["t"], 8.0, 1.0, c=1.0)
    line = fd["field"][128, 128:]
    fdi = np.interp(rr, np.arange(line.size, dtype=float), line)
    m = rr <= 0.8 * fd["t"]
    out["fd_crosscheck"] = {
        "t": fd["t"],
        "l2_rel_pct": 100.0 * float(np.linalg.norm(fdi[m] - ex[m]) / np.linalg.norm(ex[m])),
        "peak_rel_pct": 100.0 * float(abs(np.max(np.abs(fdi[m])) - np.max(np.abs(ex[m]))) / np.max(np.abs(ex[m]))),
    }
    return out, sim


if __name__ == "__main__":
    tiers = {}
    for sigma in (16.0, 12.0, 8.0):
        m, sim = measure(sigma)
        tiers[str(sigma)] = m
        print(json.dumps(m, indent=1, default=float))
    with open("runs/pulse_metrics.json", "w") as fh:
        json.dump({"tiers": tiers}, fh, indent=1, default=float)
    print("WROTE runs/pulse_metrics.json")