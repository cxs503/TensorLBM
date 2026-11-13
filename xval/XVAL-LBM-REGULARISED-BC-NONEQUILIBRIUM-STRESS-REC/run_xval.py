"""XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC — one-shot cross-validation.

Arms
----
``identity``    : the closure's invariants on synthetic boundary states +
                  a scalar transcription of the OSS routine (src/Grid.cpp:386-466)
                  run against the production vectorised implementation.
``steady``      : plane Couette and body-force Poiseuille, regularised closure vs
                  the repo's half-way bounce back, analytic reference.
``oscillating`` : Stokes' second problem (oscillating wall), last-period DFT,
                  amplitude/phase vs the analytic Stokes layer, two periods.

Usage
-----
    PYTHONPATH=src python xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/run_xval.py --arm all

Each arm writes ``artifacts/<arm>.json`` plus a row in ``artifacts/summary.json``;
nothing is tuned after the fact (thresholds live in PREREG.md and are copied
verbatim into the artifacts by :func:`thresholds`).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm import regularised_bc as rb  # noqa: E402

OUT = Path(__file__).resolve().parent / "artifacts"
OWNER_PCT = 15.0
TARGET_PCT = 3.0


def thresholds() -> dict:
    return {
        "owner_acceptance_pct": OWNER_PCT,
        "target_pct": TARGET_PCT,
        "identity_rel": 1e-13,
        "ghost_moment_rel": 1e-13,
        "oss_transcription_rel": 1e-12,
        "steady_l2_rel": TARGET_PCT / 100.0,
        "oscillating_l2_rel": OWNER_PCT / 100.0,
        "stability": "any non-finite population or |rho-1| > 0.05 is a declared failure",
    }


# ---------------------------------------------------------------------------
# OSS transcription (scalar, per node; src/Grid.cpp:386-466)
# ---------------------------------------------------------------------------


def oss_regularised_bc_node(spec, f_node, rho, ux, uy, normal, kind="velocity"):
    """Literal scalar port of LIFE ``GridClass::regularisedBC`` for one D2Q9 node.

    Mirrors the OSS loop structure: fplus/fzero -> Zou/He closure -> anti
    symmetric completion -> off-equilibrium stress -> second-order rebuild.
    """
    c = spec.c
    w = spec.w
    cs2 = spec.cs2
    ndir = int(np.argmax(np.abs(normal)))
    nsign = int(normal[ndir])
    f = f_node.copy()

    fplus = sum(f[v] for v in range(9) if c[v, ndir] == -nsign)
    fzero = sum(f[v] for v in range(9) if c[v, ndir] == 0)
    u = np.array([ux, uy], dtype=float)
    if kind == "velocity":
        rho = (2.0 * fplus + fzero) / (1.0 - float(normal @ u))

    def feq(v):
        cu = c[v, 0] * u[0] + c[v, 1] * u[1]
        return rho * w[v] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * float(u @ u))

    for v in range(9):
        if c[v, ndir] == nsign:
            f[v] = feq(v) + (f[spec.opposite[v]] - feq(spec.opposite[v]))

    sxx = syy = sxy = 0.0
    for v in range(9):
        fn = f[v] - feq(v)
        sxx += c[v, 0] * c[v, 0] * fn
        syy += c[v, 1] * c[v, 1] * fn
        sxy += c[v, 0] * c[v, 1] * fn
    for v in range(9):
        f[v] = feq(v) + (w[v] / (2.0 * cs2 * cs2)) * (
            (c[v, 0] ** 2 - cs2) * sxx
            + (c[v, 1] ** 2 - cs2) * syy
            + 2.0 * c[v, 0] * c[v, 1] * sxy
        )
    return f


def arm_identity() -> dict:
    spec = rb.D2Q9
    rng = np.random.default_rng(20261113)
    rows = []
    for normal in (np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])):
        f = np.ones((9, 4, 3)) * spec.w[:, None, None] + 0.01 * rng.standard_normal((9, 4, 3))
        rho_in, u_in = rb.macroscopic(spec, f)
        f_face = f
        u_wall = (0.03, -0.01)
        out = rb.regularised_face(spec, f_face, normal, kind="velocity", u_wall=u_wall)
        # closure values recomputed exactly as the implementation does
        cn = spec.c @ normal
        fplus = f_face[cn == -1].sum(axis=0)
        fzero = f_face[cn == 0].sum(axis=0)
        u = np.zeros((2,) + f_face.shape[1:])
        u[0], u[1] = u_wall
        rho = (2.0 * fplus + fzero) / (1.0 - float(normal @ np.array(u_wall)))
        rho_out, u_out = rb.macroscopic(spec, out)
        fc = rb.complete_unknowns(spec, f_face, normal, rho=rho, u=u)
        s_in = rb.nonequilibrium_stress(spec, fc, rho=rho, u=u)
        s_out = rb.nonequilibrium_stress(spec, out, rho=rho, u=u)
        m3_out = rb.third_moment(spec, out - rb.equilibrium(spec, rho, u))
        m3_in = rb.third_moment(spec, fc - rb.equilibrium(spec, rho, u))
        # OSS scalar transcription vs production vectorised code
        worst = 0.0
        for j in range(f.shape[1]):
            for i in range(f.shape[2]):
                ref = oss_regularised_bc_node(
                    spec, f[:, j, i], 0.0, 0.0, 0.0, normal, kind="velocity"
                )
                # the scalar port needs the OSS wall velocity: redo with wall u
                ref = _oss_with_wall(spec, f[:, j, i], normal, u_wall)
                worst = max(worst, float(np.max(np.abs(ref - out[:, j, i]))))
        rows.append(
            {
                "normal": normal.tolist(),
                "rho_rel": float(np.max(np.abs(rho_out - rho)) / np.max(np.abs(rho))),
                "u_abs": float(np.max(np.abs(u_out - u))),
                "stress_rel": float(np.max(np.abs(s_out - s_in)) / np.max(np.abs(s_in))),
                "ghost_before": float(np.max(np.abs(m3_in))),
                "ghost_after": float(np.max(np.abs(m3_out))),
                "oss_transcription_abs": worst,
            }
        )
    return {
        "arm": "identity",
        "thresholds": thresholds(),
        "rows": rows,
        "worst_stress_rel": max(r["stress_rel"] for r in rows),
        "worst_ghost_after": max(r["ghost_after"] for r in rows),
        "worst_oss_transcription_abs": max(r["oss_transcription_abs"] for r in rows),
        "pass": bool(
            max(r["stress_rel"] for r in rows) < 1e-13
            and max(r["ghost_after"] for r in rows) < 1e-13
            and max(r["oss_transcription_abs"] for r in rows) < 1e-12
        ),
    }


def _oss_with_wall(spec, f_node, normal, u_wall):
    """Scalar OSS port with the wall velocity supplied (Zou/He closure included)."""
    c = spec.c
    w = spec.w
    cs2 = spec.cs2
    ndir = int(np.argmax(np.abs(normal)))
    nsign = int(normal[ndir])
    f = f_node.copy()
    fplus = sum(f[v] for v in range(9) if c[v, ndir] == -nsign)
    fzero = sum(f[v] for v in range(9) if c[v, ndir] == 0)
    u = np.asarray(u_wall, dtype=float)
    rho = (2.0 * fplus + fzero) / (1.0 - float(normal @ u))

    def feq(v):
        cu = c[v, 0] * u[0] + c[v, 1] * u[1]
        return rho * w[v] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * float(u @ u))

    for v in range(9):
        if c[v, ndir] == nsign:
            f[v] = feq(v) + (f[spec.opposite[v]] - feq(spec.opposite[v]))
    sxx = syy = sxy = 0.0
    for v in range(9):
        fn = f[v] - feq(v)
        sxx += c[v, 0] * c[v, 0] * fn
        syy += c[v, 1] * c[v, 1] * fn
        sxy += c[v, 0] * c[v, 1] * fn
    for v in range(9):
        f[v] = feq(v) + (w[v] / (2.0 * cs2 * cs2)) * (
            (c[v, 0] ** 2 - cs2) * sxx
            + (c[v, 1] ** 2 - cs2) * syy
            + 2.0 * c[v, 0] * c[v, 1] * sxy
        )
    return f


# ---------------------------------------------------------------------------
# Physics arms
# ---------------------------------------------------------------------------


def _steady_case(case: str, wall_bc: str, ny: int) -> dict:
    nu = 0.1
    u_wall = 0.02
    height = (ny - 1) - (1 if wall_bc == "bounce_back" else 0)
    body_force = 8.0 * nu * u_wall / (height * height)
    cfg = rb.ChannelConfig(
        case=case,
        wall_bc=wall_bc,
        ny=ny,
        nx=2,
        tau=0.8,
        u_wall=u_wall if case == "couette" else 0.0,
        body_force=body_force if case == "poiseuille" else 0.0,
        steps=300000,
        tol=1e-10,
        init="rest",
    )
    rec = rb.run_channel(cfg)
    ux = np.asarray(rec["ux"])
    exact = np.asarray(rec["ux_exact"])
    amp_ratio = float(np.max(np.abs(ux)) / np.max(np.abs(exact)))
    out = {
        "case": case,
        "wall_bc": wall_bc,
        "ny": ny,
        "height": rec["height"],
        "wall_origin_declared": rec["wall_origin"],
        "nu": rec["nu"],
        "body_force": cfg.body_force,
        "u_wall": cfg.u_wall,
        "steps_run": rec["steps_run"],
        "converged": rec["converged"],
        "l2_rel": rec["l2_rel"],
        "linf_abs": rec["linf_abs"],
        "amplitude_ratio": amp_ratio,
        "mass_drift": rec["mass_drift"],
        "fitted_wall_low": rec.get("fitted_wall_low"),
        "fitted_wall_high": rec.get("fitted_wall_high"),
        "y": rec["y"],
        "ux": rec["ux"],
        "ux_exact": rec["ux_exact"],
    }
    out["pass_15pct"] = bool(out["l2_rel"] < OWNER_PCT / 100.0)
    out["pass_3pct"] = bool(out["l2_rel"] < TARGET_PCT / 100.0)
    return out


def arm_steady(resolutions=(16, 32)) -> dict:
    rows = []
    for case in ("couette", "poiseuille"):
        for ny in resolutions:
            for bc in ("regularised", "bounce_back"):
                rows.append(_steady_case(case, bc, ny))
    return {"arm": "steady", "thresholds": thresholds(), "rows": rows}


def _oscillating(wall_bc: str, ny: int, period: int, steps: int, u_wall: float = 0.02) -> dict:
    cfg = rb.ChannelConfig(
        case="oscillating_wall",
        wall_bc=wall_bc,
        ny=ny,
        nx=2,
        tau=0.8,
        u_wall=u_wall,
        omega=2.0 * np.pi / period,
        steps=steps,
    )
    rec = rb.run_channel(cfg)
    amp = np.asarray(rec["amp_abs"]) * np.exp(1.0j * np.asarray(rec["amp_phase"]))
    ref = np.asarray(rec["amp_ref_abs"]) * np.exp(1.0j * np.asarray(rec["amp_ref_phase"]))
    mask = np.abs(ref) > 0.1 * u_wall  # phase is meaningless where the amplitude is ~0
    phase_err = float(np.max(np.abs(np.angle(amp[mask] / ref[mask]))))
    out = {
        "case": "oscillating_wall",
        "wall_bc": wall_bc,
        "ny": ny,
        "period_steps": period,
        "steps_run": rec["steps_run"],
        "nu": rec["nu"],
        "delta_stokes": rec["delta_stokes"],
        "H_over_delta": rec["H_over_delta"],
        "u_wall": u_wall,
        "Ma": u_wall / np.sqrt(1.0 / 3.0),
        "l2_rel_complex": rec["l2_rel_complex"],
        "amp_max_rel": rec["amp_max_rel"],
        "phase_max_abs_masked": phase_err,
        "period_to_period_change": rec["period_to_period_change"],
        "mass_drift": rec["mass_drift"],
        "y": rec["y"],
        "amp_abs": rec["amp_abs"],
        "amp_ref_abs": rec["amp_ref_abs"],
        "amp_phase": rec["amp_phase"],
        "amp_ref_phase": rec["amp_ref_phase"],
    }
    out["pass_15pct"] = bool(out["l2_rel_complex"] < OWNER_PCT / 100.0)
    out["pass_3pct"] = bool(out["l2_rel_complex"] < TARGET_PCT / 100.0)
    return out


def arm_oscillating(periods=(200, 100), ny=48) -> dict:
    rows = []
    for bc in ("regularised", "bounce_back"):
        for period in periods:
            rows.append(_oscillating(bc, ny, period, steps=6 * period))
    return {"arm": "oscillating", "thresholds": thresholds(), "rows": rows}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def write(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def summarize(arms: dict[str, dict]) -> dict:
    rows = []
    ident = arms.get("identity")
    if ident:
        rows.append(
            {
                "arm": "identity/stress-projection",
                "reference": "pre-reconstruction non-equilibrium stress (exact)",
                "got": f"stress_rel={ident['worst_stress_rel']:.3e}, ghost_after={ident['worst_ghost_after']:.3e}",
                "error": ident["worst_stress_rel"],
                "pass_15pct": ident["pass"],
                "pass_3pct": ident["pass"],
            }
        )
        rows.append(
            {
                "arm": "identity/oss-scalar-transcription",
                "reference": "scalar port of LIFE src/Grid.cpp:386-466",
                "got": f"max|diff|={ident['worst_oss_transcription_abs']:.3e}",
                "error": ident["worst_oss_transcription_abs"],
                "pass_15pct": ident["worst_oss_transcription_abs"] < 1e-12,
                "pass_3pct": ident["worst_oss_transcription_abs"] < 1e-12,
            }
        )
    for key in ("steady", "oscillating"):
        for r in arms.get(key, {}).get("rows", []):
            metric = r["l2_rel"] if key == "steady" else r["l2_rel_complex"]
            rows.append(
                {
                    "arm": f"{key}/{r['case']}/{r['wall_bc']}",
                    "ny": r["ny"],
                    "extra": f"period={r.get('period_steps', '-')}" if key == "oscillating" else "",
                    "reference": f"analytic {r['case']}",
                    "error": metric,
                    "error_pct": 100.0 * metric,
                    "pass_15pct": r["pass_15pct"],
                    "pass_3pct": r["pass_3pct"],
                }
            )
    return {
        "anchor": "XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC",
        "thresholds": thresholds(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", default="all", choices=("identity", "steady", "oscillating", "all"))
    args = parser.parse_args(argv)
    arms: dict[str, dict] = {}
    if args.arm in ("identity", "all"):
        arms["identity"] = arm_identity()
        write("identity", arms["identity"])
        print("identity:", json.dumps({k: v for k, v in arms["identity"].items() if k != "rows"}, default=str))
    if args.arm in ("steady", "all"):
        arms["steady"] = arm_steady()
        write("steady", arms["steady"])
        for r in arms["steady"]["rows"]:
            print(
                f"steady {r['case']:10s} {r['wall_bc']:12s} ny={r['ny']:3d} "
                f"l2={r['l2_rel']:.3e} wall_fit=({r['fitted_wall_low']}, {r['fitted_wall_high']})"
            )
    if args.arm in ("oscillating", "all"):
        arms["oscillating"] = arm_oscillating()
        write("oscillating", arms["oscillating"])
        for r in arms["oscillating"]["rows"]:
            print(
                f"osc {r['wall_bc']:12s} T={r['period_steps']:3d} "
                f"l2_cplx={r['l2_rel_complex']:.3e} amp={r['amp_max_rel']:.3e} "
                f"phase={r['phase_max_abs_masked']:.3e} drift={r['period_to_period_change']:.2e}"
            )
    if arms:
        write("summary", summarize(arms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())