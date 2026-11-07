"""Free-decay dispersion test of the phase-field wave tank (no wavemaker).

A sinusoidal standing wave of known wavenumber ``k`` is released from rest; the
tank oscillates at its natural frequency ``omega(k)``.  Measuring the period
gives the *measured* phase speed ``c = omega/k``, which is compared with the
closed form ``c^2 = (g_eff/k) tanh(kh)``.  This isolates the dispersion relation
of the free surface from any wavemaker/absorber bookkeeping, which is exactly
what the two-point phase-difference estimate struggles with when the gauges are
close to the relaxation zone.

Usage::

    PYTHONPATH=src python benchmarks/verified/wave_tank/free_decay.py --kh 1.5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.wave_tank.campaign import ART, DEPTH, wave_of_config  # noqa: E402
from tensorlbm.wave_tank.tank import WaveTank, WaveTankConfig  # noqa: E402


class FreeDecayTank(WaveTank):
    """Tank with the wavemaker and absorber switched off."""

    def step_tank(self, t: float) -> None:  # noqa: D102
        self.step()


def run(kh: float, eps: float = 0.08, n_steps: int = 1100) -> dict:
    cfg = WaveTankConfig(
        kh=kh,
        eps=eps,
        h=DEPTH[kh],
        g=6.5e-3,
        nu=0.008,
        absorb=False,
        ramp_periods=0.0,
        warm_periods=0.0,
        meas_periods=0.0,
        sample_every=1,
    )
    cfg.probe_frac = (0.20, 0.37)
    w = cfg.wave  # built with g_eff
    tank = FreeDecayTank(cfg)
    ny, nx = tank.shape[1], tank.shape[2]
    y = torch.arange(ny, dtype=tank.dtype).view(1, ny, 1)
    x = torch.arange(nx, dtype=tank.dtype).view(1, 1, nx)
    a0 = 0.35
    surf = cfg.h + a0 * torch.cos(w.k * x)
    phi0 = 0.5 + 0.5 * torch.tanh(2.0 * (surf - y) / cfg.width)
    phi0 = phi0.expand(tank.shape[0], ny, nx).contiguous()
    tank.initialize(phi0)
    tank._set_hydrostatic_pressure()
    tank.y_ref = float(tank.elevation().mean())
    cols = [int(round(p)) for p in tank.geo["probes"]]
    ts, et = [], []
    for it in range(n_steps):
        tank.step_tank(float(it))
        ts.append(float(it))
        et.append(tank.elevation()[cols[1]])
    et = np.asarray(et)
    ts = np.asarray(ts)
    # skip the initial transient, then FFT with a Hann window
    seg = et[int(0.25 * n_steps):]
    spec = np.abs(np.fft.rfft((seg - seg.mean()) * np.hanning(seg.size)))
    fr = np.fft.rfftfreq(seg.size, d=1.0)
    i = int(np.argmax(spec[1:])) + 1
    # parabolic refinement of the peak
    y0, y1, y2 = spec[i - 1], spec[i], spec[i + 1]
    off = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if (y0 - 2 * y1 + y2) != 0 else 0.0
    f_meas = (fr[i] + off * (fr[1] - fr[0]))
    omega_meas = 2.0 * math.pi * f_meas
    c_meas = omega_meas / w.k
    g_eff_meas = omega_meas ** 2 / (w.k * math.tanh(w.k * cfg.h))
    out = {
        "kh": w.kh,
        "eps": w.eps,
        "h": cfg.h,
        "k": w.k,
        "wavelength": w.wavelength,
        "g": cfg.g,
        "g_wave_theory": cfg.g_wave,
        "omega_theory": w.omega,
        "period_theory": w.period,
        "period_measured": 1.0 / f_meas,
        "celerity_theory": w.celerity,
        "celerity_measured": c_meas,
        "celerity_error_rel": abs(c_meas - w.celerity) / w.celerity,
        "g_eff_measured": g_eff_meas,
        "g_eff_over_g": g_eff_meas / cfg.g,
        "amplitude_initial": a0,
        "amplitude_final": float(np.std(seg) * math.sqrt(2.0)),
        "n_steps": n_steps,
        "nx": int(nx),
        "ny": int(ny),
    }
    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}),
          flush=True)
    os.makedirs(ART, exist_ok=True)
    np.savez_compressed(os.path.join(ART, f"freedecay_kh{kh:.1f}.npz"), t=ts, eta=et,
                        config=np.array(json.dumps(cfg.to_dict())),
                        metrics=np.array(json.dumps(out)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kh", type=float, default=1.5)
    ap.add_argument("--eps", type=float, default=0.08)
    ap.add_argument("--steps", type=int, default=1100)
    a = ap.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("WT_THREADS", "4")))
    run(a.kh, a.eps, a.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())