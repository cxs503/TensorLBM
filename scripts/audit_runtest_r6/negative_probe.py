#!/usr/bin/env python3
"""AUDIT-RUNTEST-6 item 3 — negative-input probes for TensorLBM (executed, not read).

Question asked of each probe: *does an invalid physical or structural input get
rejected with a diagnostic, or does the library silently return numbers /
non-finite values?*  Expectations are declared in ``PROBES`` **before** the run
(predeclared, per the experimental-design discipline) and are justified by
physics (negative lattice viscosity, zero density, non-positive channel height)
or by an in-repo precedent (sibling modules that already validate the same
quantity — e.g. ``boundary/partial_slip_wall.py`` raises for ``tau <= 0.5``).

Classification of an observation
  REJECTED   ValueError/TypeError with a message or a custom-domain exception
  LOW_LEVEL  Python built-in failure (ZeroDivisionError/IndexError/broadcast
             silently wrong) => the guard is missing, the crash just happens
             to happen
  SILENT     call returns finite, plausible-looking output for invalid input
  SILENT_NAN returns non-finite values, no exception raised

One positive control (``N09``) is a module *known* to validate: it must come
back REJECTED, otherwise the probe harness itself is broken and no negative
result from this file may be believed.

Usage:  PYTHONPATH=src python scripts/audit_runtest_r6/negative_probe.py [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# probe bodies: each returns a short string describing the returned value
# (they are only reached when the library did NOT reject the input)
# --------------------------------------------------------------------------


def _bgk_bad_tau(tau: float) -> str:
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.solver import collide_bgk

    ny, nx = 8, 12
    rho = torch.ones(ny, nx)
    ux = torch.full((ny, nx), 0.01)
    uy = torch.zeros(ny, nx)
    f = equilibrium(rho, ux, uy)
    out = collide_bgk(f, tau)
    return f"shape={tuple(out.shape)} finite={bool(torch.isfinite(out).all())} max|f|={out.abs().max():.3e}"


def _equilibrium_negative_rho() -> str:
    from tensorlbm.d2q9 import equilibrium

    ny, nx = 8, 12
    rho = torch.full((ny, nx), -1.0)
    ux = torch.full((ny, nx), 0.01)
    uy = torch.zeros(ny, nx)
    out = equilibrium(rho, ux, uy)
    return f"finite={bool(torch.isfinite(out).all())} sum={out.sum():.4f}"


def _equilibrium_bad_shapes() -> str:
    from tensorlbm.d2q9 import equilibrium

    ny, nx = 8, 12
    rho = torch.ones(ny, nx)
    ux = torch.full((ny,), 0.01)  # (ny,)  -- should be (ny, nx)
    uy = torch.zeros(nx)  # (nx,)  -- silently broadcasts to (ny, nx)
    out = equilibrium(rho, ux, uy)
    return f"shape={tuple(out.shape)} finite={bool(torch.isfinite(out).all())}"


def _zou_he_zero_density() -> str:
    from tensorlbm.boundaries import zou_he_outlet_pressure
    from tensorlbm.d2q9 import equilibrium

    ny, nx = 8, 12
    f = equilibrium(torch.ones(ny, nx), torch.full((ny, nx), 0.01), torch.zeros(ny, nx))
    out = zou_he_outlet_pressure(f, 0.0)
    return f"finite={bool(torch.isfinite(out).all())} max|f|={out.abs().max():.3e}"


def _engine_zero_viscosity() -> str:
    from tensorlbm.general_sim import (
        GeneralSimConfig,
        GeneralSimEngine,
        GeometryConfig,
        GeometrySource,
        PhysicsConfig,
        SolverConfig,
    )

    cfg = GeneralSimConfig(
        name="neg-zero-nu",
        geometry=GeometryConfig(source=GeometrySource.PARAMETRIC_SPHERE, sphere_radius=0.5),
        physics=PhysicsConfig(viscosity=0.0, inlet_velocity=0.05, reference_length=1.0),
        solver=SolverConfig(resolution=8, max_steps=1),
    )
    re = cfg.reynolds_number  # inf / ZeroDivisionError -- no validation upstream
    eng = GeneralSimEngine(cfg)
    return f"engine built, Re={re}, uc.tau={getattr(eng.uc, 'tau', 'n/a')}"


def _engine_zero_resolution() -> str:
    from tensorlbm.general_sim import (
        GeneralSimConfig,
        GeneralSimEngine,
        GeometryConfig,
        GeometrySource,
        PhysicsConfig,
        SolverConfig,
    )

    cfg = GeneralSimConfig(
        name="neg-zero-res",
        geometry=GeometryConfig(source=GeometrySource.PARAMETRIC_SPHERE, sphere_radius=0.5),
        physics=PhysicsConfig(viscosity=0.01, inlet_velocity=0.05, reference_length=1.0),
        solver=SolverConfig(resolution=0, max_steps=1),
    )
    GeneralSimEngine(cfg)
    return "engine built with resolution=0"


def _engine_negative_steps() -> str:
    from tensorlbm.general_sim import (
        GeneralSimConfig,
        GeneralSimEngine,
        GeometryConfig,
        GeometrySource,
        PhysicsConfig,
        SolverConfig,
    )

    cfg = GeneralSimConfig(
        name="neg-steps",
        geometry=GeometryConfig(source=GeometrySource.PARAMETRIC_SPHERE, sphere_radius=0.5),
        physics=PhysicsConfig(viscosity=0.01, inlet_velocity=0.05, reference_length=1.0),
        solver=SolverConfig(resolution=8, max_steps=-5),
    )
    GeneralSimEngine(cfg)
    return "engine built with max_steps=-5"


def _engine_unknown_collision() -> str:
    from tensorlbm.general_sim import CollisionModel

    CollisionModel("not_a_collision_model")
    return "enum accepted an unknown member"


def _smagorinsky_negative_cs() -> str:
    from tensorlbm.cumulant import smagorinsky_effective_tau_d3q19

    f = torch.ones(19, 6, 6, 6) / 19.0
    out = smagorinsky_effective_tau_d3q19(f, tau=0.6, C_s=-0.5)
    if torch.is_tensor(out):
        return f"finite={bool(torch.isfinite(out).all())} min={out.min():.4f} max={out.max():.4f}"
    return f"returned {type(out).__name__}: {out}"


# --- positive control: this module is known (by source read) to validate ----
def _partial_slip_wall_tau() -> str:
    from tensorlbm.boundary.partial_slip_wall import nu_from_tau

    return f"nu_from_tau(0.4) returned {nu_from_tau(0.4)}"


# CLI-level probes (subprocess, each with its own timeout)
def _cli_poiseuille_zero_H() -> str:
    out = REPO / "runs/audit_runtest_r6/neg_poiseuille_H0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    cp = subprocess.run(
        [sys.executable, "benchmarks/verified/poiseuille_2d/run.py", "single", "0", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    tail = (cp.stdout + cp.stderr).strip().splitlines()[-1:] or [""]
    return f"rc={cp.returncode} wrote_result={out.exists()} last_line={tail[0][:140]!r}"


def _cli_sphere_D0() -> str:
    out = REPO / "runs/audit_runtest_r6/neg_sphere_D0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    cp = subprocess.run(
        [sys.executable, "benchmarks/verified/sphere_re100/run.py", str(out), "--D", "0", "--steps", "1"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    tail = (cp.stdout + cp.stderr).strip().splitlines()[-1:] or [""]
    return f"rc={cp.returncode} wrote_result={out.exists()} last_line={tail[0][:140]!r}"


PROBES = [
    ("N00", "POSITIVE CONTROL: tau<=0.5 through a validating module", "boundary.partial_slip_wall.nu_from_tau(0.4)", "REJECT", _partial_slip_wall_tau, None),
    ("N01", "core BGK collision, tau=0.4 (negative lattice viscosity)", "solver.collide_bgk(tau=0.4)", "REJECT", _bgk_bad_tau, 0.4),
    ("N02", "core BGK collision, tau=0.0 (division by zero)", "solver.collide_bgk(tau=0.0)", "REJECT", _bgk_bad_tau, 0.0),
    ("N03", "equilibrium at negative density", "d2q9.equilibrium(rho=-1)", "REJECT", _equilibrium_negative_rho, None),
    ("N04", "equilibrium with wrong (broadcastable) velocity shapes", "d2q9.equilibrium(rho(ny,nx), ux(ny,), uy(nx,))", "REJECT", _equilibrium_bad_shapes, None),
    ("N05", "Zou-He outlet with zero prescribed density", "boundaries.zou_he_outlet_pressure(rho_out=0)", "REJECT", _zou_he_zero_density, None),
    ("N06", "engine config with viscosity=0 (Re=inf)", "GeneralSimConfig(viscosity=0) -> engine", "REJECT", _engine_zero_viscosity, None),
    ("N07", "engine config with resolution=0", "SolverConfig(resolution=0) -> engine", "REJECT", _engine_zero_resolution, None),
    ("N08", "engine config with max_steps=-5", "SolverConfig(max_steps=-5) -> engine", "REJECT", _engine_negative_steps, None),
    ("N09", "unknown collision model name (control: enum must reject)", "CollisionModel('not_a_collision_model')", "REJECT", _engine_unknown_collision, None),
    ("N10", "negative Smagorinsky constant", "cumulant.smagorinsky_effective_tau_d3q19(cs=-0.5)", "REJECT", _smagorinsky_negative_cs, None),
    ("N11", "benchmark CLI: Poiseuille channel height H=0", "poiseuille_2d/run.py single 0", "REJECT", _cli_poiseuille_zero_H, None),
    ("N12", "benchmark CLI: sphere diameter D=0", "sphere_re100/run.py --D 0", "REJECT", _cli_sphere_D0, None),
]


def classify(exc: BaseException | None, ret: str | None) -> tuple[str, str]:
    if exc is not None:
        name = type(exc).__name__
        msg = str(exc).replace("\n", " ")[:200]
        if name in {"ValueError", "TypeError", "KeyError", "AssertionError", "FileNotFoundError"}:
            return "REJECTED", f"{name}: {msg}"
        return "LOW_LEVEL", f"{name}: {msg}"
    r = ret or ""
    if r.startswith("rc="):  # CLI probe: the child process reported its own fate
        rc = int(r.split()[0][3:])
        if rc == 0:
            return ("SILENT_NAN" if "finite=False" in r else "SILENT"), r
        if "ValueError" in r or "TypeError" in r:
            return "REJECTED", r
        return "LOW_LEVEL", r
    if "finite=False" in r:
        return "SILENT_NAN", r
    return "SILENT", r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "runs/audit_runtest_r6/negative_probe.json"))
    args = ap.parse_args()

    rows = []
    for pid, what, call, expected, fn, arg in PROBES:
        exc: BaseException | None = None
        ret: str | None = None
        try:
            ret = fn(arg) if arg is not None else fn()
        except BaseException as e:  # noqa: BLE001 - the observation IS the point
            exc = e
        kind, detail = classify(exc, ret)
        rows.append(
            {
                "id": pid,
                "probe": what,
                "call": call,
                "expected": expected,
                "observed": kind,
                "detail": detail,
                "verdict": "PASS" if kind == "REJECTED" else "FAIL",
                "traceback": traceback.format_exc() if (exc is not None and kind == "LOW_LEVEL") else None,
            }
        )
        print(f"{pid} [{kind:10s}] {what} -> {detail}", flush=True)

    n_pass = sum(r["verdict"] == "PASS" for r in rows)
    summary = {
        "n_probes": len(rows),
        "n_rejected": sum(r["observed"] == "REJECTED" for r in rows),
        "n_silent": sum(r["observed"] == "SILENT" for r in rows),
        "n_silent_nan": sum(r["observed"] == "SILENT_NAN" for r in rows),
        "n_low_level": sum(r["observed"] == "LOW_LEVEL" for r in rows),
        "harness_control_ok": all(
            r["observed"] == "REJECTED" for r in rows if r["id"] in {"N00", "N09"}
        ),
        "pass": n_pass,
        "fail": len(rows) - n_pass,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())