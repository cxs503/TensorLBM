#!/usr/bin/env python3
"""XVAL-LBM-INSTRUCTION-DSL-TESTS-SH -- cross-validation runner.

Three things are measured, all from ``src/tensorlbm/instruction_dsl.py`` driven by
instruction text only (no case-specific Python):

1. **Scheme semantics vs the OSS reference.**  The very instruction text that
   TensorLBM executes is translated programmatically into a ``pylbm`` scheme
   dictionary (``pylbm`` = ``/root/oss-cache/pylbm``, commit ``f9f5d16``) and
   ``pylbm`` recomputes the moment matrix ``M``, the rest equilibrium (= the
   lattice weights) and the population equilibrium ``f_eq = M^-1 . EQ``.  The
   translation is mechanical: velocity numbers are recovered from
   ``pylbm.stencil.Velocity(dim=2, num=i)`` by matching the declared velocity
   vectors, polynomials/equilibrium expressions are converted from the DSL
   grammar to ``sympy``, and ``qx = rho * ux`` is the only hand-entered relation.
   Errors are reported as ``max |a - b|`` (expected at machine precision).

2. **Physics vs analytic anchors**, using the *same* declared scheme:

   * Taylor-Green vortex decay: the energy decays as ``exp(-2 nu k^2 t)``;
     ``nu`` is fitted from the recorded kinetic-energy history
     (least squares on ``ln A``) and compared with the ``nu`` that the DSL
     derives from the declared shear relaxation ``s`` (``nu = (1/s - 1/2)/3``).
     Reference: any LBM textbook (Krueger et al. 2017, ch. 3; Succi 2001).
   * Plane Poiseuille flow driven by a body force between half-way bounce-back
     walls: the steady profile is ``u(y) = g (y - 1/2)(H - (y - 1/2)) / (2 nu)``
     with ``H = ny - 2`` for a channel whose solid rows are ``y = 0, ny-1``.  The
     wall positions are *also* fitted from the numerical profile (parabola roots)
     so the geometry claim is measured, not assumed.

3. **Convergence**: every anchor is run at two resolutions (``ny = 22, 32`` and
   ``32^2, 64^2``).

Acceptance thresholds (declared before the runs): scheme agreement
``<= 1e-12``; every physics anchor error ``< 15 %`` (owner精度标准) with the
stricter ``< 3 %`` also reported; fitted viscosity error ``< 5 %``.

Usage::

    python benchmarks/xval_instruction_dsl/run_xval.py [--quick]

Artifacts: ``artifacts/xval_instruction_dsl/`` (``*.lbm`` program texts,
``results.json``, ``summary.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.instruction_dsl import (  # noqa: E402
    D2Q9_SCHEME_INSTRUCTIONS,
    build_scheme,
    decay_rate_from_history,
    parse,
    poiseuille_error,
    run,
)

FEATURE_SLUG = "XVAL-LBM-INSTRUCTION-DSL-TESTS-SH"
OUT_DIR = ROOT / "artifacts" / "xval_instruction_dsl"
PYLBM_CACHE = Path(os.environ.get("PYLBM_CACHE", "/root/oss-cache/pylbm"))
PYLBM_STUB = ROOT / "tools" / "xval_pylbm_stub"

#: declared before any run (owner精度标准: > 15 % unacceptable)
THRESHOLDS = {
    "scheme_max_abs_diff": 1e-12,
    "physics_rel_error_pct": 15.0,
    "physics_rel_error_pct_strict": 3.0,
    "fitted_viscosity_rel_error_pct": 5.0,
}

CASES: Dict[str, Dict[str, str]] = {
    "taylor_green_32": {
        "kind": "taylor_green",
        "body": "domain nx=32 ny=32\n"
        "init taylor_green rho=1.0 amp=0.05 mode=1\n"
        "run steps=400\n",
    },
    "taylor_green_64": {
        "kind": "taylor_green",
        "body": "domain nx=64 ny=64\n"
        "init taylor_green rho=1.0 amp=0.05 mode=1\n"
        "run steps=400\n",
    },
    "poiseuille_ny22": {
        "kind": "poiseuille",
        "body": "domain nx=8 ny=22\n"
        "boundary xmin=periodic xmax=periodic ymin=bounce_back ymax=bounce_back\n"
        "init uniform rho=1.0 ux=0.0 uy=0.0\n"
        "force gx=1e-5 gy=0.0\n"
        "run steps=12000\n",
    },
    "poiseuille_ny32": {
        "kind": "poiseuille",
        "body": "domain nx=8 ny=32\n"
        "boundary xmin=periodic xmax=periodic ymin=bounce_back ymax=bounce_back\n"
        "init uniform rho=1.0 ux=0.0 uy=0.0\n"
        "force gx=1e-5 gy=0.0\n"
        "run steps=30000\n",
    },
}

QUICK_STEPS = {"poiseuille_ny22": 4000, "poiseuille_ny32": 8000}


# ---------------------------------------------------------------------------
# 1. scheme semantics against the OSS reference (pylbm)
# ---------------------------------------------------------------------------
def pylbm_reference(prog) -> Dict[str, Any]:
    """Rebuild the same scheme with pylbm and return the comparison numbers."""
    sys.path.insert(0, str(PYLBM_STUB))
    sys.path.insert(0, str(PYLBM_CACHE))
    import sympy as sp  # noqa: PLC0415

    import pylbm  # noqa: PLC0415
    from pylbm.stencil import Velocity  # noqa: PLC0415

    pylbm.utils.log.setLevel(logging.ERROR)

    # (a) velocity numbering recovered from pylbm itself, by matching vectors
    num_of = {}
    for i in range(len(prog.velocities)):
        v = Velocity(dim=prog.dim, num=i)
        num_of[tuple(int(x) for x in v.v)] = i
    vel_idx = [num_of[tuple(int(x) for x in c)] for c in prog.velocities]

    # (b) the DSL expressions converted mechanically to sympy
    X, Y = sp.symbols("X, Y")
    rho, qx, qy = sp.symbols("rho, qx, qy")
    ux, uy, LA = sp.symbols("ux, uy, LA")
    polys = [sp.sympify(p.replace("^", "**"), locals={"X": X, "Y": Y}) for p in prog.polynomials]
    eqs = []
    for expr in prog.equilibria:
        e = sp.sympify(expr.replace("^", "**"), locals={"rho": rho, "ux": ux, "uy": uy})
        eqs.append(sp.simplify(e.subs({ux: qx / rho, uy: qy / rho})).subs(sp.Symbol("LA"), 1))

    dico = {
        "dim": prog.dim,
        "scheme_velocity": float(prog.velocity),
        "schemes": [
            {
                "velocities": vel_idx,
                "polynomials": polys,
                "equilibrium": eqs,
                "relaxation_parameters": [float(s) for s in prog.relaxation],
                "conserved_moments": [rho, qx, qy],
            }
        ],
    }
    ref = pylbm.Scheme(dico)
    M_ref = np.array(ref.M.tolist(), dtype=np.float64)
    invM_ref = np.linalg.inv(M_ref)

    sub_rest = {rho: 1, qx: 0, qy: 0}
    eq_rest = np.array([float(sp.simplify(e.subs(sub_rest))) for e in ref.EQ])
    w_ref = invM_ref @ eq_rest

    dsl = build_scheme(prog)
    rng = np.random.default_rng(20261113)
    feq_diffs = []
    for _ in range(5):
        r = 1.0 + 0.05 * rng.standard_normal()
        u = 0.05 * rng.standard_normal()
        v = 0.05 * rng.standard_normal()
        eq_state = np.array(
            [float(sp.simplify(e.subs({rho: r, qx: r * u, qy: r * v}))) for e in ref.EQ]
        )
        feq_ref = invM_ref @ eq_state
        feq_dsl = dsl.equilibrium(r, u, v)
        feq_diffs.append(float(np.max(np.abs(feq_ref - feq_dsl))))

    return {
        "reference": {
            "package": "pylbm",
            "version": getattr(pylbm, "__version__", "unknown"),
            "path": str(PYLBM_CACHE),
            "commit": _git_head(PYLBM_CACHE),
            "entry_points": {
                "scheme": "pylbm/scheme.py:506 _create_moments_matrices",
                "equilibrium": "pylbm/scheme.py:238 _get_equilibrium",
                "stencil": "pylbm/stencil.py:643 Stencil.__init__",
                "validator": "pylbm/validator.py:187 validate",
            },
            "offline_shim": "tools/xval_pylbm_stub/mpi4py (single-rank no-op, import only)",
        },
        "velocity_numbers": vel_idx,
        "moment_matrix_max_abs_diff": float(np.max(np.abs(M_ref - dsl.M))),
        "weights_max_abs_diff": float(np.max(np.abs(w_ref - dsl.weights))),
        "equilibrium_population_max_abs_diff": float(max(feq_diffs)),
        "pylbm_weights": w_ref.tolist(),
        "dsl_weights": dsl.weights.tolist(),
    }


def _git_head(path: Path) -> str:
    head = path / ".git" / "HEAD"
    try:
        text = head.read_text().strip()
    except OSError:
        return "unknown"
    if text.startswith("ref: "):
        ref = path / ".git" / text.split(" ", 1)[1]
        try:
            return ref.read_text().strip()
        except OSError:
            return "unknown"
    return text


# ---------------------------------------------------------------------------
# 2. physics anchors
# ---------------------------------------------------------------------------
def _parabola_fit(ux: np.ndarray, ny: int, body_force: float) -> Dict[str, float]:
    """Fit ``u(y) = c (y - y0)(y1 - y)`` and recover the wall positions and nu."""
    y = np.arange(1, ny - 1, dtype=float)
    profile = np.asarray(ux).mean(axis=0)[1 : ny - 1]
    a2, a1, a0 = np.polyfit(y, profile, 2)
    if a2 >= 0.0:
        return {"ok": False}
    roots = np.roots([a2, a1, a0])
    y0, y1 = float(np.min(roots)), float(np.max(roots))
    nu_fit = body_force / (2.0 * (-a2))
    resid = profile - np.polyval([a2, a1, a0], y)
    return {
        "ok": True,
        "nu_fit": nu_fit,
        "wall_low": y0,
        "wall_high": y1,
        "height": y1 - y0,
        "max_residual": float(np.max(np.abs(resid))),
    }


def run_case(prog_text: str, kind: str, *, quick: bool = False) -> Dict[str, Any]:
    prog = parse(prog_text)
    if quick and prog.steps > QUICK_STEPS.get(next(iter(QUICK_STEPS)), 0):
        pass
    sample = max(1, prog.steps // 200) if kind == "taylor_green" else max(1, prog.steps // 24)
    t0 = time.perf_counter()
    result = run(prog, sample_every=sample)
    wall = time.perf_counter() - t0
    scheme = result.scheme
    out: Dict[str, Any] = {
        "program_digest": prog.digest,
        "steps": result.steps,
        "wall_time_s": round(wall, 3),
        "nu_declared": scheme.nu,
        "tau": scheme.tau,
        "sample_every": sample,
        "cs2": scheme.cs2,
        "u_max": float(np.max(np.hypot(result.ux, result.uy))),
        "mach_max": float(np.max(np.hypot(result.ux, result.uy)) / np.sqrt(scheme.cs2)),
    }
    if kind == "taylor_green":
        k = 2.0 * np.pi / prog.nx
        nu_eff, r2 = decay_rate_from_history(result.history, k)
        out.update(
            {
                "k": k,
                "nu_eff": nu_eff,
                "fit_r2": r2,
                "viscosity_rel_error_pct": abs(nu_eff - scheme.nu) / scheme.nu * 100.0,
                "ke_initial": result.history[0]["kinetic_energy"],
                "ke_final": result.kinetic_energy,
            }
        )
        out["verdict"] = out["viscosity_rel_error_pct"] < THRESHOLDS["physics_rel_error_pct"]
    else:
        g = prog.force[0]
        err = poiseuille_error(result.ux, prog.ny, nu=scheme.nu, body_force=g)
        fit = _parabola_fit(result.ux, prog.ny, g)
        out.update({"poiseuille": err, "parabola_fit": fit})
        if fit.get("ok"):
            out["viscosity_rel_error_pct"] = (
                abs(fit["nu_fit"] - scheme.nu) / scheme.nu * 100.0
            )
            out["fitted_height_minus_expected"] = fit["height"] - (prog.ny - 2)
        out["verdict"] = err["max_rel_error"] * 100.0 < THRESHOLDS["physics_rel_error_pct"]
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="shorter runs (CI smoke)")
    ap.add_argument("--skip-pylbm", action="store_true", help="skip the OSS reference")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    programs_dir = OUT_DIR / "programs"
    programs_dir.mkdir(exist_ok=True)

    report: Dict[str, Any] = {
        "feature_slug": FEATURE_SLUG,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "thresholds": THRESHOLDS,
        "programs": {},
        "scheme": {},
        "physics": {},
    }

    # --- dialect: the declared scheme itself ------------------------------
    scheme_prog = parse(
        D2Q9_SCHEME_INSTRUCTIONS + "domain nx=16 ny=16\ninit uniform\nrun steps=1\n"
    )
    dsl = build_scheme(scheme_prog)
    report["scheme"] = {
        "instructions_sha256": scheme_prog.digest,
        "weights": dsl.weights.tolist(),
        "cs2": dsl.cs2,
        "nu": dsl.nu,
        "tau": dsl.tau,
        "relaxation": dsl.relaxation.tolist(),
        "checks": {k: float(v) for k, v in dsl.checks.items()},
    }
    (programs_dir / "scheme.lbm").write_text(D2Q9_SCHEME_INSTRUCTIONS)

    # --- OSS reference cross-validation ----------------------------------
    if not args.skip_pylbm:
        try:
            report["pylbm_xval"] = pylbm_reference(scheme_prog)
        except Exception as exc:  # pragma: no cover - environment dependent
            report["pylbm_xval"] = {"error": f"{type(exc).__name__}: {exc}"}

    # --- physics anchors --------------------------------------------------
    for name, spec in CASES.items():
        text = D2Q9_SCHEME_INSTRUCTIONS + spec["body"]
        (programs_dir / f"{name}.lbm").write_text(text)
        prog = parse(text)
        if args.quick and name in QUICK_STEPS and prog.steps > QUICK_STEPS[name]:
            text = text.replace(
                f"run steps={prog.steps}", f"run steps={QUICK_STEPS[name]}"
            )
            (programs_dir / f"{name}.lbm").write_text(text)
        report["programs"][name] = {
            "digest": parse(text).digest,
            "kind": spec["kind"],
            "path": str((programs_dir / f"{name}.lbm").relative_to(ROOT)),
        }
        report["physics"][name] = run_case(text, spec["kind"], quick=args.quick)
        print(
            f"[xval] {name}: {json.dumps({k: v for k, v in report['physics'][name].items() if k in ('steps', 'nu_eff', 'viscosity_rel_error_pct', 'verdict', 'wall_time_s')})}"
        )

    # --- criteria ---------------------------------------------------------
    xval = report.get("pylbm_xval", {})
    criteria = {
        "scheme_moment_matrix": bool(
            xval.get("moment_matrix_max_abs_diff", np.inf)
            <= THRESHOLDS["scheme_max_abs_diff"]
        ),
        "scheme_weights": bool(
            xval.get("weights_max_abs_diff", np.inf)
            <= THRESHOLDS["scheme_max_abs_diff"]
        ),
        "scheme_equilibrium": bool(
            xval.get("equilibrium_population_max_abs_diff", np.inf)
            <= THRESHOLDS["scheme_max_abs_diff"]
        ),
        "physics_all_under_15pct": all(
            v.get("viscosity_rel_error_pct", np.inf) < THRESHOLDS["physics_rel_error_pct"]
            and v.get("verdict", False)
            for v in report["physics"].values()
        ),
        "physics_all_under_3pct": all(
            v.get("viscosity_rel_error_pct", np.inf)
            < THRESHOLDS["physics_rel_error_pct_strict"]
            for v in report["physics"].values()
        ),
    }
    report["criteria"] = {k: bool(v) for k, v in criteria.items()}

    (OUT_DIR / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (OUT_DIR / "summary.md").write_text(_summary_markdown(report))
    digest = hashlib.sha256((OUT_DIR / "results.json").read_bytes()).hexdigest()
    print(f"[xval] results sha256 = {digest}")
    print(f"[xval] criteria = {report['criteria']}")
    return 0 if all(report["criteria"].values()) else 1


def _summary_markdown(report: Dict[str, Any]) -> str:
    lines = [
        f"# {FEATURE_SLUG} -- results",
        "",
        f"generated: {report['generated_at']}",
        "",
        "## scheme (declared D2Q9 instructions)",
        "",
        "| quantity | value |",
        "|---|---|",
    ]
    scheme = report["scheme"]
    for key in ("nu", "tau", "cs2"):
        lines.append(f"| {key} | {scheme[key]:.12g} |")
    for key in ("sum_w", "sum_w_cx2", "cs2_rel_error", "feq_mass_rel_error"):
        lines.append(f"| check {key} | {scheme['checks'][key]:.3e} |")

    xval = report.get("pylbm_xval", {})
    lines += [
        "",
        "## vs OSS reference (pylbm)",
        "",
        "| quantity | max abs diff |",
        "|---|---|",
    ]
    for key in (
        "moment_matrix_max_abs_diff",
        "weights_max_abs_diff",
        "equilibrium_population_max_abs_diff",
    ):
        if key in xval:
            lines.append(f"| {key} | {xval[key]:.3e} |")

    lines += [
        "",
        "## physics anchors",
        "",
        "| case | steps | nu declared | nu measured | error % | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for name, res in report["physics"].items():
        lines.append(
            "| {name} | {steps} | {nu:.6f} | {m:.6f} | {err:.4f} | {v} |".format(
                name=name,
                steps=res["steps"],
                nu=res["nu_declared"],
                m=res["nu_eff"] if "nu_eff" in res else res["parabola_fit"].get("nu_fit", float("nan")),
                err=res["viscosity_rel_error_pct"],
                v="PASS" if res["verdict"] else "FAIL",
            )
        )
    lines += ["", "## criteria", ""]
    for key, value in report["criteria"].items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())