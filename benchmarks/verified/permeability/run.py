#!/usr/bin/env python3
"""W4-A benchmark: single-phase permeability of a 2D periodic square array of
solid cylinders against the Sangani & Acrivos (1982) multipole-series drag.

Reference (locked in NOTES.md section 1, corrected in the section 3
amendment of 2026-09-21 — see NOTES for the full adjudication chain):
    Sangani's f is the PRESSURE-form drag f = (G*L^2)/(mu*U_D), which
    includes the phantom pressure acting on the cylinder's own projected
    area.  A fluid-masked body force a produces the identical velocity
    field as a uniform pressure gradient G = a (the force inside the solid
    is inert because u = 0 there), so the measured
        k_sim = nu * <u_x>_fluid / a = L^2 / ((1-phi) * f)
    and the error criterion is err = |k_sim / k_ref - 1| with
    k_ref = d^2/((1-phi)*f_table).  The frozen-prereg channel
    (k_ref = d^2/f_table) is still computed and reported as
    err_prereg for transparency.

All physics kernels come from the tensorlbm library:
    collide/stream      -> tensorlbm.solver            (D2Q9, periodic stream)
    bounce-back         -> tensorlbm.boundaries
    body force          -> tensorlbm.turbulent_channel._apply_body_force_2d
    equilibrium/moments -> tensorlbm.d2q9
Only the geometry (boolean cylinder-array mask, periodic folded distance,
style of tensorlbm.porous_media.make_random_cylinder_medium) is built here;
geometry is not a physics kernel.

Step order follows the library's own turbulent_channel loop:
    collide -> stream -> (fluid-masked) body force -> bounce-back.

Subcommands
    case    run one (phi, d) case, write case_phi{phi}_d{d}[_fs{scale}].json
    report  aggregate case files into result.json + README.md (zero hand
            transcription; README numbers are printed from result.json)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from compile_route import (  # noqa: E402
    add_compile_mode_arg,
    compile_mode_from_args,
    ensure_tensorlbm_importable,
    route_step,
)

ensure_tensorlbm_importable()

import torch  # noqa: E402

from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402
from tensorlbm.turbulent_channel import _apply_body_force_2d  # noqa: E402

# --------------------------------------------------------------------------
# Registered constants (NOTES.md section 2; must match the preregistration)
# --------------------------------------------------------------------------
TAU = 1.0
NU = (TAU - 0.5) / 3.0  # = 1/6 exactly for tau = 1
RE_TARGET = 0.05  # Re_cell = a * k_lu * R / nu^2, solved for a

# Sangani & Acrivos 1982 Table 1 (double transcription, NOTES sections 1.2/1.3)
SANGANI_TABLE: list[tuple[float, float]] = [
    (0.05, 15.56),
    (0.10, 24.83),
    (0.20, 51.53),
    (0.30, 102.90),
    (0.40, 217.89),
    (0.50, 532.55),
    (0.60, 1.763e3),
    (0.70, 1.352e4),
    (0.75, 1.263e5),
]
F_TABLE = {phi: f for phi, f in SANGANI_TABLE}

LADDER: dict[float, list[int]] = {0.3: [26, 52, 104], 0.5: [20, 40, 80]}

SAMPLE_EVERY = 200  # steps between samples
DRIFT_SPAN = 10  # samples in the trailing drift window (= 2000 steps)
DRIFT_TOL = 1e-5  # relative drift threshold
STEADY_REPEATS = 3  # consecutive passing evaluations required
POST_STEADY = 8000  # extra steps after first steady pass (>= S2 needs)
MEAS_SAMPLES = 20  # measurement window = last 20 samples = 4000 steps
S2_WIN_A = 10  # sensitivity S2: last 2000 steps
S2_WIN_B = 40  # sensitivity S2: last 8000 steps


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def library_provenance() -> dict:
    """Record where tensorlbm came from (read-only query, no repo writes)."""
    import tensorlbm

    pkg = Path(tensorlbm.__file__).resolve()
    info: dict = {"tensorlbm_file": str(pkg), "torch_version": torch.__version__}
    repo = pkg.parents[1]
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        info["library_commit"] = head
        info["library_worktree_dirty"] = bool(dirty)
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort
        info["library_commit_error"] = repr(exc)
    return info


def loglog_f_of_phi(phi: float) -> float:
    """Interpolate f(phi) on (ln phi, ln f) over the Sangani table."""
    xs = [math.log(p) for p, _ in SANGANI_TABLE]
    ys = [math.log(f) for _, f in SANGANI_TABLE]
    x = math.log(phi)
    if x < xs[0] or x > xs[-1]:
        raise ValueError(f"phi={phi} outside table range for diagnostic channel B")
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return math.exp(ys[i] + t * (ys[i + 1] - ys[i]))
    raise AssertionError("unreachable")


def cylinder_array_mask(d: int, phi: float, device: torch.device) -> torch.Tensor:
    """Boolean solid mask for one periodic square-array unit cell d x d.

    Geometry only (not a physics kernel): circle of nominal radius
    R = d*sqrt(phi/pi) centred at the cell centre, distance measured with
    periodic folding in both directions, comparisons in float64.
    """
    import numpy as np

    r_real = d * math.sqrt(phi / math.pi)
    idx = np.arange(d, dtype=np.float64)
    dx = np.abs(idx - d / 2.0)
    dx = np.minimum(dx, d - dx)  # periodic fold
    dy = dx.copy()  # square cell, same fold in y
    dist2 = dx[None, :] ** 2 + dy[:, None] ** 2  # (ny, nx)
    mask = dist2 <= r_real * r_real
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def fmt(x: float, digits: int = 6) -> str:
    return f"{x:.{digits}g}"


def run_case(args: argparse.Namespace) -> dict:
    phi = float(args.phi)
    d = int(args.d)
    if phi not in F_TABLE:
        raise SystemExit(f"phi must be one of {sorted(F_TABLE)}, got {phi}")
    device = torch.device(args.device)
    smoke = bool(args.smoke)
    force_scale = float(args.force_scale)

    f_table = F_TABLE[phi]
    r_real = d * math.sqrt(phi / math.pi)
    k_lu = d * d / f_table
    a_body = RE_TARGET * NU * NU / (k_lu * r_real) * force_scale
    max_steps = int(args.max_steps) if args.max_steps else min(40 * d * d, 400_000)
    sample_every = SAMPLE_EVERY
    drift_tol = DRIFT_TOL
    post_steady = POST_STEADY
    if smoke:  # plumbing check only, never a formal result
        max_steps = min(max_steps, 3000)
        sample_every = 100
        drift_tol = 1e-3
        post_steady = 400

    solid = cylinder_array_mask(d, phi, device)
    fluid = ~solid
    n_solid = int(solid.sum().item())
    phi_actual = n_solid / (d * d)

    rho0 = torch.ones(d, d, dtype=torch.float32, device=device)
    u0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, u0, u0, device=device)
    mass0 = float(f.sum().item())
    del rho0, u0

    solid_b = solid.unsqueeze(0)

    def _step(f_t: torch.Tensor) -> torch.Tensor:
        f_t = collide_bgk(f_t, TAU)
        f_t = stream(f_t)
        # Body force must act on fluid nodes only: unmasked injection is
        # reversed by bounce-back on solid nodes, leaving a silent net drive
        # rho*a*L^2*(1-2*phi) (exactly zero at phi=0.5).  NOTES section 2.2.
        f_t = torch.where(solid_b, f_t, _apply_body_force_2d(f_t, a_body))
        return bounce_back_cells(f_t, solid)

    step_fn = route_step(_step, args.compile_mode, name=f"perm[phi{phi}_d{d}]")

    hist_step: list[int] = []
    hist_uxf: list[float] = []
    hist_uyf: list[float] = []
    hist_umax: list[float] = []
    hist_uymax: list[float] = []
    hist_mass: list[float] = []
    recent_drifts: list[float] = []
    steady_step: int | None = None
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        f = step_fn(f)
        if step % sample_every == 0:
            _, ux, uy = macroscopic(f)
            uxf = float(ux[fluid].mean().item())
            uyf = float(uy[fluid].mean().item())
            umax = float(ux[fluid].abs().max().item())
            uymax = float(uy[fluid].abs().max().item())
            mass = float(f.sum().item())
            hist_step.append(step)
            hist_uxf.append(uxf)
            hist_uyf.append(uyf)
            hist_umax.append(umax)
            hist_uymax.append(uymax)
            hist_mass.append(mass)
            if steady_step is None and len(hist_uxf) >= DRIFT_SPAN + 1:
                drift = abs(hist_uxf[-1] - hist_uxf[-1 - DRIFT_SPAN]) / abs(hist_uxf[-1])
                recent_drifts.append(drift)
                if len(recent_drifts) >= STEADY_REPEATS and all(
                    v < drift_tol for v in recent_drifts[-STEADY_REPEATS:]
                ):
                    steady_step = step
        if steady_step is not None and step >= steady_step + post_steady:
            break

    wall = time.perf_counter() - t0
    steps_run = hist_step[-1] if hist_step else 0
    if steady_step is None:
        print(f"WARNING: phi={phi} d={d}: steady criterion not met in {max_steps} steps")

    def _win_mean(vals: list[float], n: int) -> float | None:
        return sum(vals[-n:]) / len(vals[-n:]) if len(vals) >= n else None

    uxf_meas = _win_mean(hist_uxf, MEAS_SAMPLES)
    uxf_w2000 = _win_mean(hist_uxf, S2_WIN_A)
    uxf_w8000 = _win_mean(hist_uxf, S2_WIN_B)
    uyf_meas = _win_mean(hist_uyf, MEAS_SAMPLES)
    umax_meas = _win_mean(hist_umax, MEAS_SAMPLES)
    uymax_meas = _win_mean(hist_uymax, MEAS_SAMPLES)
    assert uxf_meas is not None, "measurement window empty"

    k_sim = NU * uxf_meas / a_body
    # Corrected chain (NOTES section 3 amendment): k_ref = d^2/((1-phi)*f).
    f_sim = d * d / ((1.0 - phi) * k_sim)  # pressure-form drag, nominal phi
    f_sim_actual = d * d / ((1.0 - phi_actual) * k_sim)
    k_ref = d * d / ((1.0 - phi) * f_table)
    signed = k_sim / k_ref - 1.0
    err = abs(signed)
    # frozen-prereg channel A (k_ref = d^2/f_table), transparency only
    k_ref_prereg = d * d / f_table
    err_prereg = abs(k_sim / k_ref_prereg - 1.0)
    u_d_meas = uxf_meas * (1.0 - phi_actual)
    re_achieved = u_d_meas * r_real / NU
    ma = math.sqrt(3.0) * (umax_meas or 0.0)
    uy_ratio = (uymax_meas or 0.0) / abs(uxf_meas) if uxf_meas else float("nan")
    mass_drift = max(abs(m / mass0 - 1.0) for m in hist_mass) if hist_mass else float("nan")
    f_at_actual = loglog_f_of_phi(phi_actual)
    k_ref_b = d * d / ((1.0 - phi_actual) * f_at_actual)
    err_b = abs(k_sim / k_ref_b - 1.0)

    record = {
        "kind": "case",
        "created_utc": utc_now(),
        "smoke": smoke,
        "run": {
            "phi": phi,
            "d": d,
            "tau": TAU,
            "nu": NU,
            "R_real": r_real,
            "k_lu_nominal": k_lu,
            "a_body": a_body,
            "force_scale": force_scale,
            "Re_target": RE_TARGET,
            "sample_every": sample_every,
            "drift_span_steps": DRIFT_SPAN * sample_every,
            "drift_tol": drift_tol,
            "steady_repeats": STEADY_REPEATS,
            "post_steady_steps": post_steady,
            "meas_samples": MEAS_SAMPLES,
            "meas_window_steps": MEAS_SAMPLES * sample_every,
            "max_steps": max_steps,
            "steps_run": steps_run,
            "steady_step": steady_step,
            "steady_detected": steady_step is not None,
            "compile_mode": args.compile_mode,
            "device": str(device),
            "wall_seconds": wall,
        },
        "geometry": {
            "N_solid": n_solid,
            "phi_actual": phi_actual,
            "mask_rule": "periodic-folded (dx^2+dy^2) <= R^2, float64, centre d/2",
        },
        "measurement": {
            "uxf_mean_win4000": uxf_meas,
            "uxf_mean_win2000": uxf_w2000,
            "uxf_mean_win8000": uxf_w8000,
            "uyf_mean_win4000": uyf_meas,
            "umax_mean_win4000": umax_meas,
            "uymax_mean_win4000": uymax_meas,
            "mass_drift_max_rel": mass_drift,
            "u_D_superficial": u_d_meas,
        },
        "derived": {
            "k_sim": k_sim,
            "f_sim": f_sim,
            "f_sim_phi_actual": f_sim_actual,
            "k_ref_nominal": k_ref,
            "err_rel": err,
            "signed_err_rel": signed,
            "err_pct": err * 100.0,
            "k_ref_prereg_A": k_ref_prereg,
            "err_prereg_A_rel": err_prereg,
            "err_prereg_A_pct": err_prereg * 100.0,
            "f_table_at_phi": f_table,
            "f_diagB_at_phi_actual": f_at_actual,
            "k_ref_diagB": k_ref_b,
            "err_B_rel": err_b,
            "err_B_pct": err_b * 100.0,
            "Re_cell_achieved": re_achieved,
            "Ma": ma,
            "uy_ratio_max": uy_ratio,
        },
        "kernels": {
            "collide": "tensorlbm.solver.collide_bgk",
            "stream": "tensorlbm.solver.stream (periodic gather)",
            "body_force": "tensorlbm.turbulent_channel._apply_body_force_2d (fluid-masked)",
            "bounce_back": "tensorlbm.boundaries.bounce_back_cells",
            "moments": "tensorlbm.d2q9.macroscopic",
            "init": "tensorlbm.d2q9.equilibrium(rho=1, u=0)",
            "step_order": "collide -> stream -> masked force -> bounce_back",
            "provenance": library_provenance(),
        },
        "history": {
            "step": hist_step,
            "uxf": hist_uxf,
            "uyf": hist_uyf,
            "umax": hist_umax,
            "uymax": hist_uymax,
            "mass": hist_mass,
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if force_scale == 1.0 else f"_fs{force_scale:g}"
    name = f"case_smoke_phi{phi}_d{d}{tag}.json" if smoke else f"case_phi{phi}_d{d}{tag}.json"
    out = out_dir / name
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[case] phi={phi} d={d} fs={force_scale:g} -> {out}")
    print(
        f"  steady_step={steady_step} steps_run={steps_run} wall={wall:.1f}s "
        f"phi_actual={phi_actual:.6f} k_sim={k_sim:.6g} f_sim={f_sim:.6g} "
        f"k_ref={k_ref:.6g} err={err * 100:.4f}% err_B={err_b * 100:.4f}% "
        f"Re={re_achieved:.4f} Ma={ma:.2e}"
    )
    return record


# --------------------------------------------------------------------------
# report: aggregate case files -> result.json + README.md
# --------------------------------------------------------------------------
def cmd_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    result: dict = {
        "kind": "result",
        "created_utc": utc_now(),
        "benchmark": "permeability_square_array_2d",
        "reference": {
            "source": "Sangani & Acrivos 1982, Int. J. Multiphase Flow 8(2):193-206",
            "doi": "10.1016/0301-9322(82)90029-5",
            "definition": (
                "f = (G*L^2)/(mu*U_D) pressure-form drag (Sangani's "
                "convention: includes the phantom pressure on the "
                "cylinder projection), U_D superficial velocity"
            ),
            "normalization": (
                "k_sim = nu*<u_x>_fluid/a = L^2/((1-phi)*f) "
                "(corrected 2026-09-21, NOTES section 3; frozen "
                "prereg channel A k_ref = d^2/f also reported)"
            ),
            "table": {str(p): v for p, v in SANGANI_TABLE},
            "channels": [
                "Basilisk cylinders.c transcription (verbatim)",
                "MDPI Fluids 2021 6(9):334 Table 1 (digit-identical)",
                "dilute series cross (0.040% at phi=0.05)",
                "own lubrication derivation (0.27% at phi=0.75)",
                "Basilisk printout normalization construct (section 1.5.1)",
            ],
        },
        "settings": {
            "tau": TAU,
            "nu": NU,
            "Re_target": RE_TARGET,
            "sample_every": SAMPLE_EVERY,
            "drift_tol": DRIFT_TOL,
            "meas_window_steps": MEAS_SAMPLES * SAMPLE_EVERY,
            "criterion": "err strictly decreasing with refinement per phi AND finest <= 3%",
            "extrap": "none",
        },
        "ladders": {},
        "sensitivity": {},
        "verdict": {},
    }

    ladders: dict[float, list[dict]] = {}
    for phi, ds in LADDER.items():
        entries = []
        for d in ds:
            p = out_dir / f"case_phi{phi}_d{d}.json"
            if not p.exists():
                print(f"WARNING: missing {p}")
                continue
            rec = json.loads(p.read_text())
            if rec.get("smoke"):
                continue
            r, der, geo = rec["run"], rec["derived"], rec["geometry"]
            entries.append(
                {
                    "d": d,
                    "R_real": r["R_real"],
                    "a_body": r["a_body"],
                    "steps_run": r["steps_run"],
                    "steady_step": r["steady_step"],
                    "steady_detected": r["steady_detected"],
                    "phi_actual": geo["phi_actual"],
                    "N_solid": geo["N_solid"],
                    "k_sim": der["k_sim"],
                    "f_sim": der["f_sim"],
                    "f_sim_phi_actual": der["f_sim_phi_actual"],
                    "k_ref": der["k_ref_nominal"],
                    "err_pct": der["err_pct"],
                    "signed_err_pct": der["signed_err_rel"] * 100.0,
                    "err_prereg_A_pct": der["err_prereg_A_pct"],
                    "err_B_pct": der["err_B_pct"],
                    "Re_cell_achieved": der["Re_cell_achieved"],
                    "Ma": der["Ma"],
                    "uy_ratio_max": der["uy_ratio_max"],
                    "mass_drift_max_rel": rec["measurement"]["mass_drift_max_rel"],
                    "uxf_mean_win2000": rec["measurement"]["uxf_mean_win2000"],
                    "uxf_mean_win8000": rec["measurement"]["uxf_mean_win8000"],
                    "case_file": p.name,
                }
            )
        if entries:
            ladders[phi] = entries

    verdicts: dict[str, dict] = {}
    overall_ok = True
    for phi, entries in ladders.items():
        errs = [e["err_pct"] for e in entries]
        mono = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
        finest = errs[-1] if errs else float("nan")
        all_steady = all(e["steady_detected"] for e in entries)
        ok = mono and finest <= 3.0
        if not all_steady:
            ok = False
        overall_ok = overall_ok and ok
        reasons = []
        if not mono:
            reasons.append("monotonicity violated")
        if finest > 3.0:
            reasons.append(f"finest err {finest:.4f}% > 3%")
        if not all_steady:
            reasons.append("steady criterion not met for at least one grid")
        verdicts[str(phi)] = {
            "errors_pct": errs,
            "monotone_decreasing": mono,
            "finest_err_pct": finest,
            "finest_d": entries[-1]["d"] if entries else None,
            "all_steady": all_steady,
            "pass": ok,
            "reasons": reasons,
        }
    result["ladders"] = {str(p): e for p, e in ladders.items()}
    result["verdict"] = {
        "overall": "PASS" if overall_ok else "FAIL",
        "per_phi": verdicts,
        "extrap": "none",
        "statement": (
            "PASS iff err strictly decreases with refinement for every phi and the "
            "finest grid err <= 3%; nominal geometry only, no effective-radius "
            "recalibration in the criterion (diagnostic channel B reported separately)."
        ),
    }

    # sensitivity S1: force linearity (Stokes) — a x2 must leave k_sim unchanged
    s1: dict[str, dict] = {}
    for p in sorted(out_dir.glob("case_phi*_d*_fs*.json")):
        rec = json.loads(p.read_text())
        if rec.get("smoke"):
            continue
        base = out_dir / f"case_phi{rec['run']['phi']}_d{rec['run']['d']}.json"
        if not base.exists():
            continue
        k0 = json.loads(base.read_text())["derived"]["k_sim"]
        k1 = rec["derived"]["k_sim"]
        s1[p.name] = {
            "k_base": k0,
            "k_scaled": k1,
            "delta_pct": abs(k1 / k0 - 1.0) * 100.0,
            "tol_pct": 0.1,
            "pass": abs(k1 / k0 - 1.0) * 100.0 < 0.1,
        }
    result["sensitivity"]["S1_force_linearity"] = s1

    # sensitivity S2: measurement window (last 2000 vs last 8000 steps)
    s2: dict[str, dict] = {}
    for phi, entries in ladders.items():
        for e in entries[-2:]:
            w2, w8 = e["uxf_mean_win2000"], e["uxf_mean_win8000"]
            if w2 is None or w8 is None:
                continue
            s2[f"phi{phi}_d{e['d']}"] = {
                "uxf_win2000": w2,
                "uxf_win8000": w8,
                "delta_pct": abs(w2 / w8 - 1.0) * 100.0,
                "tol_pct": 0.05,
                "pass": abs(w2 / w8 - 1.0) * 100.0 < 0.05,
            }
    result["sensitivity"]["S2_window"] = s2

    # S3 disclosures live per-case in the ladders (Re, Ma, uy ratio, mass drift)

    res_path = out_dir / "result.json"
    res_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[report] wrote {res_path}")
    for phi, v in verdicts.items():
        print(
            f"  phi={phi}: errs={[round(x, 4) for x in v['errors_pct']]} "
            f"mono={v['monotone_decreasing']} finest={v['finest_err_pct']:.4f}% "
            f"pass={v['pass']}"
        )
    print(f"  VERDICT: {result['verdict']['overall']}")

    write_readme(out_dir, result)


def write_readme(out_dir: Path, result: dict) -> None:
    lines: list[str] = []
    a = lines.append
    a("# W4-A: 2D periodic square array of cylinders — permeability vs Sangani–Acrivos 1982")
    a("")
    a(f"Created (UTC): {result['created_utc']}")
    a("")
    a("## Setup")
    a("")
    a("- D2Q9 BGK, tau = 1.0 (nu = 1/6), fully periodic unit cell d x d, one cylinder")
    a("  of nominal radius R = d*sqrt(phi/pi) at the cell centre (periodic-folded mask).")
    a("- Body force on fluid nodes only; step order collide -> stream -> masked force ->")
    a("  bounce-back. All kernels from tensorlbm (see case JSON `kernels` block).")
    a(f"- Re_target = {result['settings']['Re_target']}; a_body = Re*nu^2/(k_lu*R).")
    a("- Normalization (corrected, NOTES section 3): k_sim = nu*<u_x>_fluid/a maps to")
    a("  L^2/((1-phi)*f) because Sangani's f is the pressure-form drag G*L^2/(mu*U_D)")
    a("  and the fluid-masked body force a is equivalent to G = a. The frozen-prereg")
    a("  channel (k_ref = d^2/f) is reported as err_prereg_A for transparency.")
    a("- Diagnostic channel B re-references to phi_actual (log-log table interpolation);")
    a("  disclosed only, never re-judged.")
    a("- All numbers below are printed programmatically from result.json (zero hand")
    a("  transcription).")
    a("")
    a("## Verdict")
    a("")
    v = result["verdict"]
    a(f"**{v['overall']}** — {v['statement']}")
    a("")
    a("| phi | ladder err (%) | monotone | finest err (%) | pass |")
    a("|-----|----------------|----------|----------------|------|")
    for phi, vv in v["per_phi"].items():
        errs = ", ".join(f"{x:.4f}" for x in vv["errors_pct"])
        a(
            f"| {phi} | {errs} | {vv['monotone_decreasing']} | "
            f"{vv['finest_err_pct']:.4f} | {vv['pass']} |"
        )
    a("")
    a("## Ladder detail")
    a("")
    for phi, entries in result["ladders"].items():
        a(f"### phi = {phi}")
        a("")
        a(
            "| d | R_real | a_body | phi_actual | k_sim | f_sim | k_ref | err (%) | "
            "signed (%) | preregA (%) | err_B (%) | steps | steady | Re | Ma | uy_ratio | mass drift |"
        )
        a(
            "|---|--------|--------|------------|-------|-------|-------|---------|------------|"
            "------------|-----------|-------|-------|-----|-----|----------|------------|"
        )
        for e in entries:
            a(
                f"| {e['d']} | {e['R_real']:.4f} | {e['a_body']:.6e} | "
                f"{e['phi_actual']:.6f} | {e['k_sim']:.6g} | {e['f_sim']:.6g} | "
                f"{e['k_ref']:.6g} | "
                f"{e['err_pct']:.4f} | {e['signed_err_pct']:+.4f} | "
                f"{e['err_prereg_A_pct']:.4f} | {e['err_B_pct']:.4f} | "
                f"{e['steps_run']} | {e['steady_detected']} | {e['Re_cell_achieved']:.4f} | "
                f"{e['Ma']:.2e} | {e['uy_ratio_max']:.3f} | "
                f"{e['mass_drift_max_rel']:.2e} |"
            )
        a("")
    a("## Sensitivity")
    a("")
    a("### S1 force linearity (a x2, Stokes: k_sim must be invariant, tol 0.1%)")
    a("")
    s1 = result["sensitivity"]["S1_force_linearity"]
    if s1:
        a("| case | k_base | k_scaled | delta (%) | pass |")
        a("|------|--------|----------|-----------|------|")
        for name, s in s1.items():
            a(
                f"| {name} | {s['k_base']:.6g} | {s['k_scaled']:.6g} | "
                f"{s['delta_pct']:.5f} | {s['pass']} |"
            )
    else:
        a("(not run)")
    a("")
    a("### S2 measurement window (last 2000 vs last 8000 steps, tol 0.05%)")
    a("")
    s2 = result["sensitivity"]["S2_window"]
    if s2:
        a("| case | uxf(2000) | uxf(8000) | delta (%) | pass |")
        a("|------|-----------|-----------|-----------|------|")
        for name, s in s2.items():
            a(
                f"| {name} | {s['uxf_win2000']:.8e} | {s['uxf_win8000']:.8e} | "
                f"{s['delta_pct']:.5f} | {s['pass']} |"
            )
    else:
        a("(not run)")
    a("")
    a("### S3 achieved values")
    a("")
    a("Reported per grid in the ladder tables above (Re_cell, Ma, uy symmetry ratio,")
    a("max relative mass drift).")
    a("")
    a("## Reference")
    a("")
    r = result["reference"]
    a(f"- {r['source']}, DOI {r['doi']}")
    a(f"- {r['definition']}; {r['normalization']}")
    a(f"- Table (phi -> f): {r['table']}")
    a("- Cross channels: " + "; ".join(r["channels"]))
    a("- Extrapolation: none. Nominal geometry criterion only; diagnostic channel B")
    a("  (phi_actual via log-log table interpolation) is disclosed, never re-judged.")
    a("")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {out_dir / 'README.md'}")


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "Single-phase permeability benchmark: 2D periodic square array of "
            "cylinders vs the Sangani & Acrivos (1982) drag table. "
            "Use 'case' to run one (phi, d) grid, 'report' to aggregate."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_case = sub.add_parser("case", help="run one (phi, d) case and write its case JSON")
    p_case.add_argument(
        "--phi", type=float, required=True, help=f"solid area fraction, one of {sorted(F_TABLE)}"
    )
    p_case.add_argument(
        "--d", type=int, required=True, help="lattice size of the periodic unit cell (square d x d)"
    )
    p_case.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="directory for case JSON output (default ./out)",
    )
    p_case.add_argument(
        "--device", default="cuda:5", help="torch device (default cuda:5, campaign-reserved)"
    )
    p_case.add_argument(
        "--force-scale",
        type=float,
        default=1.0,
        help="multiply a_body by this factor (S1 sensitivity; "
        "k_sim must be invariant in the Stokes regime)",
    )
    p_case.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="override max steps (0 = registered min(40*d^2, 400000))",
    )
    p_case.add_argument(
        "--smoke",
        action="store_true",
        help="plumbing check only (short run, marked smoke, non-formal)",
    )
    add_compile_mode_arg(p_case)

    p_rep = sub.add_parser("report", help="aggregate case JSONs into result.json + README.md")
    p_rep.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="directory holding case JSONs (default ./out)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.compile_mode = compile_mode_from_args(args)
    if args.command == "case":
        run_case(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
