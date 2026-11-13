#!/usr/bin/env python3
"""XVAL-LBM-CHAPMAN-ENSKOG-AUTO: automatic Chapman-Enskog vs measured transport.

What this harness decides
-------------------------
For a given (stencil, collision operator) the symbolic Chapman-Enskog module
:mod:`tensorlbm.chapman_enskog` predicts

  * ``nu``    -- kinematic shear viscosity,
  * ``zeta``  -- bulk (second) viscosity,
  * ``nu_L``  -- ``2(D-1)/D nu + zeta``, the coefficient that damps sound.

This harness *measures* those three numbers from the repository solvers and
checks every prediction against the measurement with the owner accuracy gate
(``< 15 %``).  It is a cross-validation, not a plausibility check: the symbolic
derivation consumes only ``(c_q, w_q, collision rates)`` and never sees a
simulation, and the measurement never sees the derivation.

Measurements (all use ``src/tensorlbm`` solver code, nothing is re-implemented):
  * shear wave 2D   -- ``tensorlbm.solver.collide_bgk`` / ``collide_mrt`` + ``stream``
  * shear wave 3D   -- ``tensorlbm.collision_viscosity_audit.run_collision_viscosity_audit``
  * sound wave 2D/3D -- same 2D solver and ``tensorlbm.solver3d`` (D3Q19)

Outputs (written next to this file unless ``--outdir`` is given):
  ``result.json``, ``table.csv``, ``convergence.csv``

Reproduce::

    PYTHONPATH=src python3 benchmarks/xval_chapman_enskog_auto/run_xval.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import tensorlbm  # noqa: E402
import tensorlbm.chapman_enskog as ce  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, collide_mrt, stream  # noqa: E402

ANCHOR = "XVAL-LBM-CHAPMAN-ENSKOG-AUTO"
GATE_PCT = 15.0
DTYPE = torch.float64


# --------------------------------------------------------------------------- #
# measurement primitives
# --------------------------------------------------------------------------- #
def _fit_log_decay(t: np.ndarray, amp: np.ndarray) -> dict[str, float]:
    """Least-squares ``log|amp| = a + b t`` with slope standard error."""
    t = np.asarray(t, dtype=float)
    y = np.log(np.abs(amp))
    A = np.vstack([np.ones_like(t), t]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(t) - 2, 1)
    s2 = float((resid**2).sum() / dof)
    cov = s2 * float(np.linalg.inv(A.T @ A)[1, 1])
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": float(coef[1]), "slope_stderr": math.sqrt(max(cov, 0.0)), "r2": r2}


def _collide_2d(kind: str, tau: float, s_e: float | None):
    if kind == "bgk":
        return lambda f: collide_bgk(f, tau)

    def _mrt(f):
        if s_e is None:
            return collide_mrt(f, tau)
        return collide_mrt(f, tau, s_e=s_e)

    return _mrt


def measure_shear_2d(nx: int, ny: int, tau: float, *, kind: str = "bgk", s_e: float | None = None,
                     steps: int = 1200, amp: float = 1.0e-3, sample_every: int = 10,
                     fit_fraction: float = 0.5) -> dict[str, float]:
    """Decay rate of a transverse (shear) wave: ``amp ~ exp(-nu k^2 t)``."""
    torch.manual_seed(0)
    k = 2.0 * math.pi / nx
    x = torch.arange(nx, dtype=DTYPE)
    profile = torch.sin(k * x).view(1, nx).expand(ny, nx)
    rho = torch.ones((ny, nx), dtype=DTYPE)
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, amp * profile)
    collide = _collide_2d(kind, tau, s_e)
    ts, amps = [], []
    start = int(steps * (1.0 - fit_fraction))
    for step in range(1, steps + 1):
        f = stream(collide(f))
        if step >= start and step % sample_every == 0:
            _, _, uy = macroscopic(f)
            # modal amplitude of the sin component (periodic projection)
            amps.append(float((uy * profile).mean().item() / (profile * profile).mean().item()))
            ts.append(step)
    fit = _fit_log_decay(np.array(ts), np.array(amps))
    nu = -fit["slope"] / k**2
    nu_err = fit["slope_stderr"] / k**2
    # non-modal energy for a sanity check that the wave stays a single mode
    _, ux, _ = macroscopic(f)
    return {
        "nu": nu,
        "nu_uncertainty": nu_err,
        "r2": fit["r2"],
        "k": k,
        "steps": steps,
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "collision": kind,
        "residual_ux_max": float(ux.abs().max().item()),
    }


def measure_sound_2d(nx: int, ny: int, tau: float, *, kind: str = "bgk", s_e: float | None = None,
                     steps: int = 600, amp: float = 1.0e-4, sample_every: int = 5,
                     fit_fraction: float = 0.5, cs_ref: float = 1.0 / math.sqrt(3.0)) -> dict[str, float]:
    """Attenuation and dispersion of a right-going acoustic wave.

    ``rho' ~ exp(-nu_L k^2 t / 2)`` with ``nu_L = 2(D-1)/D nu + zeta``;
    the phase speed gives ``c_s``.
    """
    torch.manual_seed(0)
    k = 2.0 * math.pi / nx
    x = torch.arange(nx, dtype=DTYPE)
    cos_kx = torch.cos(k * x).view(1, nx).expand(ny, nx)
    sin_kx = torch.sin(k * x).view(1, nx).expand(ny, nx)
    rho = 1.0 + amp * cos_kx
    ux = cs_ref * amp * cos_kx
    uy = torch.zeros_like(rho)
    f = equilibrium(rho, ux, uy)
    collide = _collide_2d(kind, tau, s_e)
    norm = float((cos_kx * cos_kx).mean().item())
    start = int(steps * (1.0 - fit_fraction))
    ts, amps, phases = [], [], []
    for step in range(1, steps + 1):
        f = stream(collide(f))
        if step >= start and step % sample_every == 0:
            r, _, _ = macroscopic(f)
            d = r - 1.0
            R = float((d * cos_kx).mean().item() / norm)
            I = float((d * sin_kx).mean().item() / norm)
            ts.append(step)
            amps.append(math.hypot(R, I))
            phases.append(math.atan2(I, R))
    fit = _fit_log_decay(np.array(ts), np.array(amps))
    nu_l = -2.0 * fit["slope"] / k**2
    nu_l_err = 2.0 * fit["slope_stderr"] / k**2
    unwrapped = np.unwrap(np.array(phases))
    t = np.asarray(ts, dtype=float)
    A = np.vstack([np.ones_like(t), t]).T
    omega = float(np.linalg.lstsq(A, unwrapped, rcond=None)[0][1])
    return {
        "nu_L": nu_l,
        "nu_L_uncertainty": nu_l_err,
        "omega": omega,
        "cs_measured": abs(omega) / k,
        "r2": fit["r2"],
        "k": k,
        "steps": steps,
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "collision": kind,
    }


def measure_shear_3d(collision_model: str = "bgk", tau: float = 0.8, wavelength: int = 32,
                     transverse: int = 4, steps: int = 200) -> dict[str, float]:
    """Shear-wave viscosity from the repository's own D3Q19 audit module."""
    from tensorlbm.collision_viscosity_audit import (
        CollisionViscosityAuditConfig,
        run_collision_viscosity_audit,
    )

    cfg = CollisionViscosityAuditConfig(
        collision_model=collision_model,
        tau=tau,
        wavelength_cells=wavelength,
        transverse_cells=transverse,
        steps=steps,
        dtype="float64",
    )
    out = run_collision_viscosity_audit(cfg)
    res = out["result"]
    return {
        "nu": float(res["recovered_kinematic_viscosity"]),
        "nu_uncertainty": float("nan"),
        "r2": float("nan"),
        "k": 2.0 * math.pi / wavelength,
        "steps": steps,
        "tau": tau,
        "collision": collision_model,
        "harness_reported_target": float(res["target_kinematic_viscosity"]),
        "harness_reported_error_pct": float(res["relative_error_pct"]),
    }


def measure_sound_3d(nx: int, tau: float, *, steps: int = 400, amp: float = 1.0e-4,
                     sample_every: int = 4, fit_fraction: float = 0.5,
                     cs_ref: float = 1.0 / math.sqrt(3.0)) -> dict[str, float]:
    """D3Q19 plane sound wave along y in a (1, nx, 1) periodic box."""
    from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
    from tensorlbm.solver3d import collide_bgk3d, stream3d

    k = 2.0 * math.pi / nx
    y = torch.arange(nx, dtype=DTYPE)
    shape = (1, nx, 1)
    cos_ky = torch.cos(k * y).view(shape)
    sin_ky = torch.sin(k * y).view(shape)
    rho = 1.0 + amp * cos_ky
    uy = cs_ref * amp * cos_ky
    zero = torch.zeros_like(rho)
    f = equilibrium3d(rho, zero, uy, zero)
    norm = float((cos_ky * cos_ky).mean().item())
    start = int(steps * (1.0 - fit_fraction))
    ts, amps, phases = [], [], []
    for step in range(1, steps + 1):
        f = stream3d(collide_bgk3d(f, tau))
        if step >= start and step % sample_every == 0:
            r = macroscopic3d(f)[0]
            d = r - 1.0
            R = float((d * cos_ky).mean().item() / norm)
            I = float((d * sin_ky).mean().item() / norm)
            ts.append(step)
            amps.append(math.hypot(R, I))
            phases.append(math.atan2(I, R))
    fit = _fit_log_decay(np.array(ts), np.array(amps))
    nu_l = -2.0 * fit["slope"] / k**2
    unwrapped = np.unwrap(np.array(phases))
    A = np.vstack([np.ones_like(np.asarray(ts, float)), np.asarray(ts, float)]).T
    omega = float(np.linalg.lstsq(A, unwrapped, rcond=None)[0][1])
    return {
        "nu_L": nu_l,
        "nu_L_uncertainty": 2.0 * fit["slope_stderr"] / k**2,
        "cs_measured": abs(omega) / k,
        "r2": fit["r2"],
        "k": k,
        "steps": steps,
        "nx": nx,
        "tau": tau,
        "collision": "bgk-d3q19",
    }


# --------------------------------------------------------------------------- #
# derivation table
# --------------------------------------------------------------------------- #
def derive_all(taus: tuple[float, ...], s_e_list: tuple[float, ...]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tau in taus:
        for stencil, q in ((ce.D2Q9, 9), (ce.D3Q19, 19)):
            d = ce.derive_equivalent_equation(stencil, ce.bgk(tau, q))
            out[f"{stencil.name}-BGK-tau{tau:g}"] = d.as_dict() | {
                "nu_L": ce.longitudinal_viscosity(d),
            }
        d = ce.derive_equivalent_equation(ce.D2Q9, ce.d2q9_mrt(tau))
        out[f"D2Q9-MRT-tau{tau:g}"] = d.as_dict() | {"nu_L": ce.longitudinal_viscosity(d)}
        for s_e in s_e_list:
            d = ce.derive_equivalent_equation(ce.D2Q9, ce.d2q9_mrt(tau, s_e=s_e))
            out[f"D2Q9-MRT-tau{tau:g}-s_e{s_e:g}"] = d.as_dict() | {
                "nu_L": ce.longitudinal_viscosity(d),
            }
    d = ce.derive_equivalent_equation(ce.D3Q27, ce.bgk(0.8, 27))
    out["D3Q27-BGK-tau0.8"] = d.as_dict() | {"nu_L": ce.longitudinal_viscosity(d)}
    d = ce.derive_equivalent_equation(ce.D2Q9, ce.trt(ce.D2Q9, 1.9, 1.25))
    out["D2Q9-TRT-wp1.9-wm1.25"] = d.as_dict() | {"nu_L": ce.longitudinal_viscosity(d)}
    return out


def check(check_id: str, quantity: str, ref: float, got: float, *, ref_source: str,
          tol_pct: float = GATE_PCT, extra: dict | None = None, scale: float | None = None) -> dict:
    """Relative-error gate.

    When the reference is *exactly* zero (e.g. the MRT bulk viscosity at
    ``1/s_e = 1/2``) a relative error is undefined; the error is then formed
    against an explicit physical scale (the shear viscosity of the same
    configuration) instead of silently producing ``nan`` and failing the gate on
    arithmetic rather than on physics.
    """
    if ref != 0:
        err = (got - ref) / ref * 100.0
    elif scale:
        err = (got - ref) / scale * 100.0
    else:
        err = float("nan")
    row = {
        "id": check_id,
        "quantity": quantity,
        "ref": ref,
        "got": got,
        "err_pct": err,
        "abs_err_pct": abs(err),
        "tol_pct": tol_pct,
        "pass": bool(abs(err) <= tol_pct),
        "ref_source": ref_source,
    }
    if extra:
        row |= extra
    if ref == 0 and scale:
        row["scale_for_zero_ref"] = scale
        row["ref_source"] = ref_source + f" (zero reference: error scaled by nu={scale:g})"
    return row


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--quick", action="store_true", help="shorter runs for smoke testing")
    args = ap.parse_args(argv)
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    torch.set_default_dtype(DTYPE)
    taus = (0.6, 0.8, 1.2)
    s_e_list = (1.2, 1.64, 2.0)
    if args.quick:
        taus, s_e_list = (0.8,), (1.64,)

    derivations = derive_all(taus, s_e_list)
    checks: list[dict] = []
    measurements: dict[str, dict] = {}
    convergence: list[dict] = []

    # ---- arm A: D2Q9 BGK shear viscosity, two resolutions, three tau ------ #
    for tau in taus:
        ref = derivations[f"D2Q9-BGK-tau{tau:g}"]["nu"]
        for nx in (32, 64):
            m = measure_shear_2d(nx, 8, tau, steps=400 if args.quick else 1200)
            measurements[f"shear2d_bgk_tau{tau:g}_n{nx}"] = m
            checks.append(check(f"A-shear2d-tau{tau:g}-n{nx}", "nu (D2Q9 BGK)",
                                ref, m["nu"], ref_source="chapman_enskog.derive_equivalent_equation(D2Q9, BGK)",
                                extra={"r2": m["r2"], "fit_uncertainty": m["nu_uncertainty"]}))
            convergence.append({"arm": "A", "nx": nx, "tau": tau, "ref": ref, "got": m["nu"],
                                "err_pct": (m["nu"] - ref) / ref * 100.0})

    # ---- arm B: D2Q9 MRT shear viscosity (rate routing check) ------------- #
    for tau in taus:
        ref = derivations[f"D2Q9-MRT-tau{tau:g}"]["nu"]
        for s_e in s_e_list:
            m = measure_shear_2d(32, 8, tau, kind="mrt", s_e=s_e, steps=400 if args.quick else 1200)
            measurements[f"shear2d_mrt_tau{tau:g}_se{s_e:g}"] = m
            checks.append(check(f"B-shear2d-mrt-tau{tau:g}-se{s_e:g}", "nu (D2Q9 MRT)",
                                ref, m["nu"],
                                ref_source="chapman_enskog.derive_equivalent_equation(D2Q9, d2q9_mrt)"))
            # the derived nu must not depend on the energy-mode rate
            assert abs(ref - derivations[f"D2Q9-BGK-tau{tau:g}"]["nu"]) < 1e-12

    # ---- arm C: D3Q19 BGK shear viscosity (repo audit module) ------------- #
    for tau in taus:
        ref = derivations[f"D3Q19-BGK-tau{tau:g}"]["nu"]
        for wl in (32, 64):
            m = measure_shear_3d("bgk", tau, wavelength=wl)
            measurements[f"shear3d_bgk_tau{tau:g}_n{wl}"] = m
            checks.append(check(f"C-shear3d-tau{tau:g}-n{wl}", "nu (D3Q19 BGK)",
                                ref, m["nu"],
                                ref_source="chapman_enskog.derive_equivalent_equation(D3Q19, BGK)"))
            convergence.append({"arm": "C", "nx": wl, "tau": tau, "ref": ref, "got": m["nu"],
                                "err_pct": (m["nu"] - ref) / ref * 100.0})

    # ---- arm D: D2Q9 acoustic wave -- sound speed, bulk viscosity --------- #
    tau = 0.8
    d_bgk = derivations[f"D2Q9-BGK-tau{tau:g}"]
    for nx in (32, 64):
        m = measure_sound_2d(nx, 4, tau, steps=300 if args.quick else 600)
        measurements[f"sound2d_bgk_tau{tau:g}_n{nx}"] = m
        checks.append(check(f"D1-sound-speed-bgk-n{nx}", "c_s (D2Q9 BGK)",
                            d_bgk["cs"], m["cs_measured"],
                            ref_source="chapman_enskog: cs = sqrt(sum w c_x^2)"))
        checks.append(check(f"D2-sound-atten-bgk-n{nx}", "nu_L = nu + zeta (D2Q9 BGK)",
                            d_bgk["nu_L"], m["nu_L"],
                            ref_source="chapman_enskog: nu_L = 2(D-1)/D nu + zeta"))
    # MRT: bulk viscosity is tuned by the energy-mode rate -> nu_L changes by 3x
    for s_e in s_e_list:
        d_mrt = derivations[f"D2Q9-MRT-tau{tau:g}-s_e{s_e:g}"]
        m = measure_sound_2d(32, 4, tau, kind="mrt", s_e=s_e, steps=300 if args.quick else 600)
        measurements[f"sound2d_mrt_tau{tau:g}_se{s_e:g}"] = m
        checks.append(check(f"D3-sound-atten-mrt-se{s_e:g}", "nu_L = nu + zeta (D2Q9 MRT)",
                            d_mrt["nu_L"], m["nu_L"],
                            ref_source="chapman_enskog.derive_equivalent_equation(D2Q9, MRT)"))
        checks.append(check(f"D4-zeta-mrt-se{s_e:g}", "zeta (D2Q9 MRT)",
                            d_mrt["zeta"], m["nu_L"] - d_bgk["nu"],
                            ref_source="chapman_enskog zeta, isolated via nu_L - nu_measured",
                            extra={"note": "nu_L measured minus derived shear nu"},
                            scale=d_bgk["nu"]))
        convergence.append({"arm": "D-MRT", "nx": 32, "tau": tau, "s_e": s_e,
                            "ref": d_mrt["nu_L"], "got": m["nu_L"],
                            "err_pct": (m["nu_L"] - d_mrt["nu_L"]) / d_mrt["nu_L"] * 100.0})

    # ---- arm E: D3Q19 acoustic wave -- dimension-general nu_L -------------- #
    d3 = derivations[f"D3Q19-BGK-tau{tau:g}"]
    for nx in (32, 64):
        m = measure_sound_3d(nx, tau, steps=200 if args.quick else 400)
        measurements[f"sound3d_bgk_tau{tau:g}_n{nx}"] = m
        checks.append(check(f"E1-sound-atten-3d-n{nx}", "nu_L = (4/3)nu + zeta (D3Q19 BGK)",
                            d3["nu_L"], m["nu_L"],
                            ref_source="chapman_enskog: nu_L = 2(D-1)/D nu + zeta, D=3"))
        checks.append(check(f"E2-sound-speed-3d-n{nx}", "c_s (D3Q19 BGK)",
                            d3["cs"], m["cs_measured"],
                            ref_source="chapman_enskog: cs = sqrt(sum w c_x^2)"))
        convergence.append({"arm": "E", "nx": nx, "tau": tau, "ref": d3["nu_L"],
                            "got": m["nu_L"], "err_pct": (m["nu_L"] - d3["nu_L"]) / d3["nu_L"] * 100.0})

    passed = sum(1 for c in checks if c["pass"])
    worst = max(checks, key=lambda c: c["abs_err_pct"])
    result = {
        "anchor": ANCHOR,
        "mission": 17672,
        "gate_pct": GATE_PCT,
        "verdict": "PASS" if passed == len(checks) else "FAIL",
        "checks_passed": passed,
        "checks_total": len(checks),
        "worst_check": worst,
        "provenance": {
            "git_head": _git_head(),
            "tensorlbm_module": str(Path(tensorlbm.__file__).resolve()),
            "chapman_enskog_sha256": hashlib.sha256(
                (REPO_ROOT / "src/tensorlbm/chapman_enskog.py").read_bytes()).hexdigest(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "dtype": "float64",
            "machine": platform.machine(),
        },
        "derivations": derivations,
        "checks": checks,
        "measurements": measurements,
        "convergence": convergence,
        "repro_cmd": "PYTHONPATH=src python3 benchmarks/xval_chapman_enskog_auto/run_xval.py",
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))

    with (outdir / "table.csv").open("w") as fh:
        fh.write("id,quantity,ref,got,err_pct,abs_err_pct,tol_pct,pass,ref_source\n")
        for c in checks:
            fh.write(f"{c['id']},{c['quantity']},{c['ref']:.12g},{c['got']:.12g},"
                     f"{c['err_pct']:.6f},{c['abs_err_pct']:.6f},{c['tol_pct']},{c['pass']},"
                     f"{c['ref_source']}\n")
    with (outdir / "convergence.csv").open("w") as fh:
        fh.write("arm,nx,tau,ref,got,err_pct%s\n" % (",s_e" if any("s_e" in c for c in convergence) else ""))
        for c in convergence:
            se = f",{c.get('s_e', '')}" if any("s_e" in x for x in convergence) else ""
            fh.write(f"{c['arm']},{c['nx']},{c['tau']},{c['ref']:.12g},{c['got']:.12g},"
                     f"{c['err_pct']:.6f}{se}\n")

    print(f"[{ANCHOR}] {result['verdict']}  {passed}/{len(checks)} checks within {GATE_PCT}%")
    for c in checks:
        flag = "ok " if c["pass"] else "FAIL"
        print(f"  {flag} {c['id']:<32} ref={c['ref']:<12.6g} got={c['got']:<12.6g} err={c['err_pct']:+.3f}%")
    print(f"worst: {worst['id']} {worst['err_pct']:+.3f}%")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
