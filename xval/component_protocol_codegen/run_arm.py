#!/usr/bin/env python3
"""XVAL arm runner: generated LBM kernel vs analytic Taylor-Green decay.

Anchor: XVAL-LBM-COMPONENT-PROTOCOL-TEMPLATE-CODEGEN (mission 17744).

Each arm
--------
1. builds a kernel through the component protocol
   (``ComponentSpec`` -> validation -> template emission -> exec),
2. checks it against the independent reference kernel on a fixed random field
   (cross-implementation control),
3. runs the 2D Taylor-Green vortex and recovers the shear viscosity from the
   analytic decay of the kinetic energy,
4. writes one JSON artifact with every number the report quotes.

Usage::

    python xval/component_protocol_codegen/run_arm.py --arm A1 \
        --out artifacts/xval_component_protocol_codegen/arm_A1.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # allow running from a plain checkout
    sys.path.insert(0, str(_SRC))

from tensorlbm.codegen import (  # noqa: E402
    BGKComponent,
    D2Q9,
    TRTComponent,
    generate_kernel,
    reference_collide,
    reference_step,
)

# ---------------------------------------------------------------- arm matrix
# Predeclared in predeclared.json; do not edit after inspecting results.
ARM_MATRIX = {
    "A1": dict(family="BGK", n=32, u0=0.02, tau=0.56),
    "A2": dict(family="BGK", n=64, u0=0.02, tau=0.56),
    "A3": dict(family="TRT", n=32, u0=0.02, tau=0.56),
    "A4": dict(family="TRT", n=64, u0=0.02, tau=0.56),
    "A5": dict(family="BGK", n=64, u0=0.04, tau=0.56),
    "A6": dict(family="TRT", n=64, u0=0.04, tau=0.56),
}

K_WAVE = 1
STEPS = 120
SAMPLE_EVERY = 2
FIT_FROM = 20
MAGIC = 3.0 / 16.0


def make_component(family: str, tau: float):
    if family == "BGK":
        return BGKComponent(tau=tau)
    if family == "TRT":
        return TRTComponent(tau=tau, magic=MAGIC)
    raise SystemExit(f"unknown family {family!r}")


def taylor_green_fields(n: int, u0: float, k: int = K_WAVE):
    """Analytic initial velocity field on a periodic [0, 2*pi)^2 lattice."""
    x = (np.arange(n, dtype=float) + 0.5) * (2.0 * math.pi / n)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    ux = u0 * np.cos(k * xx) * np.sin(k * yy)
    uy = -u0 * np.sin(k * xx) * np.cos(k * yy)
    return ux, uy


def lattice_wavenumber(n: int, k: int = K_WAVE) -> float:
    """Lattice wavenumber of mode ``k`` on ``n`` cells spanning 2*pi.

    The analytic field is written in lattice coordinates as
    ``cos(2*pi*k*x_lb/n)``, so the lattice wavenumber is ``2*pi*k/n`` -- using
    ``k`` here instead would mix physical and lattice units and understate the
    decay by ``(2*pi/n)^2`` (an error the first draft of this harness made;
    see docs/codegen_component_protocol.md, "注意点").
    """
    return 2.0 * math.pi * k / n


def analytic_amplitude(u0: float, nu: float, t: float, k_lb: float) -> float:
    """Amplitude decay factor exp(-2 nu k_lb^2 t) of the Taylor-Green vortex."""
    return math.exp(-2.0 * nu * k_lb * k_lb * t)


def kinetic_energy(ux: np.ndarray, uy: np.ndarray) -> float:
    return 0.5 * float(np.mean(ux * ux + uy * uy))


def divergence(ux: np.ndarray, uy: np.ndarray) -> float:
    """Central-difference divergence (lattice units, periodic)."""
    dx = 0.5 * (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1))
    dy = 0.5 * (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0))
    return float(np.abs(dx + dy).max())


def fit_viscosity(samples: list[dict], fit_from: int, k_lb: float) -> dict:
    t = np.array([s["t"] for s in samples if s["t"] >= fit_from], dtype=float)
    e = np.array([s["E"] for s in samples if s["t"] >= fit_from], dtype=float)
    if len(t) < 3:
        raise SystemExit("fit window too short")
    y = np.log(e)
    slope, intercept = np.polyfit(t, y, 1)
    pred = slope * t + intercept
    resid = y - pred
    return {
        "nu_measured": float(-slope / (4.0 * k_lb * k_lb)),
        "slope": float(slope),
        "intercept": float(intercept),
        "n_samples": int(len(t)),
        "fit_rms": float(np.sqrt(np.mean(resid**2))),
        "fit_window": [float(t[0]), float(t[-1])],
    }


def check_equivalence(component, kernel, seed: int = 0) -> dict:
    """Generated vs independent reference kernel on a fixed random population."""
    rng = np.random.default_rng(seed)
    f = 1.0 / 9.0 + 0.01 * rng.standard_normal((D2Q9.q, 12, 10))
    fam, tau = component.spec.family, float(component.spec.param("tau"))
    d_collide = float(np.abs(kernel.collide(f) - reference_collide(f, family=fam, tau=tau)).max())
    d_step = float(np.abs(kernel.step(f) - reference_step(f, family=fam, tau=tau)).max())
    f_ref = f.copy()
    for _ in range(20):
        f_ref = reference_step(f_ref, family=fam, tau=tau)
    f_gen = f.copy()
    for _ in range(20):
        f_gen = kernel.step(f_gen)
    return {
        "seed": seed,
        "max_abs_collide": d_collide,
        "max_abs_step": d_step,
        "max_abs_20_steps": float(np.abs(f_gen - f_ref).max()),
        "field_shape": list(f.shape),
    }


def run_arm(arm: str) -> dict:
    cfg = ARM_MATRIX[arm]
    component = make_component(cfg["family"], cfg["tau"])
    t0 = time.perf_counter()
    kernel = generate_kernel(component)
    gen_s = time.perf_counter() - t0

    declared = dict(component.transport_coefficients())
    nu_declared = float(declared["nu"])
    n, u0 = int(cfg["n"]), float(cfg["u0"])

    equiv = check_equivalence(component, kernel)

    ux0, uy0 = taylor_green_fields(n, u0)
    rho0 = np.ones_like(ux0)
    f = kernel.equilibrium(rho0, ux0, uy0)

    samples: list[dict] = []
    t_start = time.perf_counter()
    for step in range(STEPS + 1):
        if step % SAMPLE_EVERY == 0:
            rho, ux, uy = kernel.moments(f)
            samples.append(
                {
                    "t": step,
                    "E": kinetic_energy(ux, uy),
                    "rho_mean": float(rho.mean()),
                    "umax": float(np.sqrt(ux * ux + uy * uy).max()),
                }
            )
        if step == STEPS:
            break
        f = kernel.step(f)
    run_s = time.perf_counter() - t_start

    rho, ux, uy = kernel.moments(f)
    k_lb = lattice_wavenumber(n)
    decay = analytic_amplitude(u0, nu_declared, STEPS, k_lb)
    ux_a, uy_a = taylor_green_fields(n, u0 * decay)
    field_l2_rel = float(
        np.linalg.norm(np.concatenate([(ux - ux_a).ravel(), (uy - uy_a).ravel()]))
        / np.linalg.norm(np.concatenate([ux_a.ravel(), uy_a.ravel()]))
    )

    fit = fit_viscosity(samples, FIT_FROM, k_lb)
    nu_err_pct = abs(fit["nu_measured"] - nu_declared) / nu_declared * 100.0

    return {
        "arm": arm,
        "config": cfg,
        "component": {
            "spec": component.spec.as_dict(),
            "s_plus": component.s_plus(),
            "s_minus": component.s_minus(),
            "transport_declared": declared,
        },
        "kernel": {
            "sha256": kernel.sha256,
            "source_lines": kernel.provenance["source_lines"],
            "template": kernel.provenance["template_name"],
            "protocol_version": kernel.provenance["protocol_version"],
            "checks": kernel.provenance["component"],
        },
        "case": {
            "name": "taylor_green_2d",
            "n": n,
            "k": K_WAVE,
            "k_lattice": k_lb,
            "u0": u0,
            "ma": float(u0 * math.sqrt(3.0)),
            "tau": cfg["tau"],
            "nu_declared": nu_declared,
            "steps": STEPS,
            "sample_every": SAMPLE_EVERY,
            "fit_from": FIT_FROM,
            "re": float(u0 / nu_declared),
        },
        "nu_measured": fit["nu_measured"],
        "nu_error_pct": nu_err_pct,
        "fit": fit,
        "field_l2_rel": field_l2_rel,
        "div_max": divergence(ux, uy),
        "mass_drift": float(abs(rho.mean() - 1.0)),
        "equivalence": equiv,
        "timing_s": {"generate": gen_s, "simulate": run_s},
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(ARM_MATRIX))
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args(argv)

    result = run_arm(args.arm)
    out = (
        Path(args.out)
        if args.out
        else _REPO_ROOT / "artifacts" / "xval_component_protocol_codegen" / f"arm_{args.arm}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(
        f"{args.arm} {result['config']['family']} N={result['config']['n']} "
        f"u0={result['config']['u0']}: nu={result['nu_measured']:.6f} "
        f"(declared {result['case']['nu_declared']:.6f}, err {result['nu_error_pct']:.3f}%) "
        f"field_l2={result['field_l2_rel']:.3e} "
        f"codegen-ref={result['equivalence']['max_abs_step']:.2e} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())