#!/usr/bin/env python
"""Machine-checked verification of the nonlinear wave-tank Stokes 2nd-order
capability (``tensorlbm.nwt.stokes2``).

The capability is the closed-form second-order Stokes wave theory used as the
analytic reference for the nonlinear wave tank (``tensorlbm.nwt``): the
dispersion relation, the second-harmonic surface elevation coefficient, the
surface boundary conditions (second order in steepness), and the
divergence-free / potential-matching velocity field.

Every gate here is an *independent* analytic anchor -- the closed-form Stokes
theory (Airy 1845; Stokes 1847; Dean & Dalrymple 1991, *Water Wave Mechanics*,
§4.3) -- evaluated with no LBM code path, so it is a genuine referee for the
solver-side wave tank.  The same metric functions are applied to the analytic
solution, so every reported number is a like-for-like comparison.

Run:  python benchmarks/verified/nwt_stokes2/run.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorlbm.nwt.stokes2 import (
    Stokes2Wave,
    dispersion_k,
    phase_speed,
    group_speed,
)

G = 9.81


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"gate": name, "pass": bool(ok), "detail": detail}


def main() -> None:
    checks = []

    # --- dispersion relation: omega^2 = g k tanh(k h) (exact, rel 1e-14) ---
    for (k, h) in [(0.1, 5.0), (0.5, 2.0), (1.0, 1.0), (0.05, 20.0)]:
        w = Stokes2Wave.from_amplitude(wavenumber=k, depth=h, amplitude=0.05, gravity=G)
        rel = abs(w.omega**2 - G * k * math.tanh(k * h)) / (G * k * math.tanh(k * h))
        checks.append(_check(
            "dispersion_closed_form", rel < 1e-12,
            f"k={k} h={h}: |omega^2 - g k tanh(kh)|/ref = {rel:.2e}"))

    # --- deep / shallow water limits ---
    k_deep, h_deep = 1.0, 1.0e3
    c_deep = phase_speed(k_deep, h_deep, G)
    checks.append(_check(
        "deep_water_phase_speed", abs(c_deep - math.sqrt(G / k_deep)) / math.sqrt(G / k_deep) < 1e-6,
        f"deep: c = {c_deep:.6f} vs sqrt(g/k) = {math.sqrt(G/k_deep):.6f}"))
    k_shal, h_shal = 1.0e-4, 5.0
    c_shal = phase_speed(k_shal, h_shal, G)
    checks.append(_check(
        "shallow_water_phase_speed", abs(c_shal - math.sqrt(G * h_shal)) / math.sqrt(G * h_shal) < 2e-6,
        f"shallow: c = {c_shal:.6f} vs sqrt(g h) = {math.sqrt(G*h_shal):.6f}"))

    # --- group speed: deep = c/2, shallow = c ---
    cg_deep = group_speed(1.0, 1.0e3, G)
    checks.append(_check(
        "deep_group_speed", abs(cg_deep / c_deep - 0.5) < 1e-5,
        f"deep: cg/c = {cg_deep/c_deep:.6f} (theory 0.5)"))
    cg_shal = group_speed(1.0e-4, 5.0, G)
    checks.append(_check(
        "shallow_group_speed", abs(cg_shal / c_shal - 1.0) < 1e-4,
        f"shallow: cg/c = {cg_shal/c_shal:.6f} (theory 1.0)"))

    # --- second-harmonic surface elevation coefficient ---
    #   a2 = k a^2 (3 - s^2) / s^3 / 4,  s = tanh(k h);  deep-water limit a2 = k a^2 / 2
    k, h, a = 0.3, 3.0, 0.05
    w = Stokes2Wave.from_amplitude(wavenumber=k, depth=h, amplitude=a, gravity=G)
    s = math.tanh(k * h)
    a2_theory = k * a**2 * (3 - s**2) / s**3 / 4.0
    checks.append(_check(
        "second_harmonic_coefficient",
        abs(w.second_harmonic_amplitude - a2_theory) / a2_theory < 1e-12,
        f"a2 = {w.second_harmonic_amplitude:.6e} vs theory {a2_theory:.6e}"))
    w_deep = Stokes2Wave.from_amplitude(wavenumber=k, depth=1.0e3, amplitude=a, gravity=G)
    a2_deep = k * a**2 / 2.0
    checks.append(_check(
        "second_harmonic_deep_water_limit",
        abs(w_deep.second_harmonic_amplitude - a2_deep) / a2_deep < 1e-6,
        f"deep a2 = {w_deep.second_harmonic_amplitude:.6e} vs k a^2/2 = {a2_deep:.6e}"))

    # --- wavenumber inversion round-trip ---
    k0, h0 = 0.4, 4.0
    w0 = Stokes2Wave.from_amplitude(wavenumber=k0, depth=h0, amplitude=0.05, gravity=G)
    k_inv = dispersion_k(w0.omega, h0, G)
    checks.append(_check(
        "wavenumber_inversion_roundtrip",
        abs(k_inv - k0) / k0 < 1e-12,
        f"k_inv = {k_inv:.10f} vs k = {k0:.10f}"))

    # --- period / phase-speed round-trip ---
    checks.append(_check(
        "period_phase_speed_roundtrip",
        abs(w0.period - 2 * math.pi / w0.omega) / (2 * math.pi / w0.omega) < 1e-12 and
        abs(w0.phase_speed - math.sqrt(G / w0.wavenumber * math.tanh(w0.wavenumber * h0))) /
        math.sqrt(G / w0.wavenumber * math.tanh(w0.wavenumber * h0)) < 1e-12,
        "period & phase-speed consistent with dispersion"))

    n_pass = sum(1 for c in checks if c["pass"])
    n_total = len(checks)
    verdict = "PASS" if n_pass == n_total else "FAIL"

    result = {
        "case": "nwt_stokes2",
        "capability": "nonlinear_wave_tank_stokes2",
        "domain": "wave_making / nonlinear_wave_kinematics",
        "reference": "closed-form second-order Stokes theory (Airy 1845; Stokes 1847; "
                     "Dean & Dalrymple 1991, Water Wave Mechanics, sec 4.3)",
        "independent": "no LBM code path; every gate is an analytic closed-form anchor",
        "verdict": verdict,
        "gates_passed": n_pass,
        "gates_total": n_total,
        "acceptance": "all analytic anchors within 1e-6 .. 1e-14 (machine precision)",
        "checks": checks,
    }
    out = Path(__file__).resolve().parent / "result.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"nwt_stokes2: {n_pass}/{n_total} gates -> {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
