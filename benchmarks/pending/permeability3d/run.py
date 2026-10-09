#!/usr/bin/env python3
"""W6-A benchmark: single-phase Stokes permeability of a 3D periodic
simple-cubic array of solid spheres against Zick & Homsy (1982).

Reference (locked in NOTES.md section 1 BEFORE any run; two-source cross):
    K(phi) = F/(6*pi*mu*U*a)  (U = SUPERFICIAL velocity, a = sphere radius)
Mapping (NOTES section 2): F = rho*a_body*L^3 (phantom projected-area
pressure included), U = (1-phi)*<ux>_fluid, mu = rho*nu, so
    K_sim = a_body*N^3 / (6*pi*nu*a_nom*(1-phi)*<ux>_fluid)
    k_ref = N^3 / (6*pi*a_nom*K_table)
    err   = |k_sim/k_ref - 1|   (nominal geometry primary; phi_actual and
    recalibration channels are diagnostic only, never verdict-bearing).

All physics kernels are library calls from the pinned read-only worktree
(cf5709db; NOTES section 4):
    collide            -> tensorlbm.solver3d.collide_bgk3d
    stream (periodic)  -> tensorlbm.solver3d.stream3d
    bounce-back        -> tensorlbm.boundaries3d.bounce_back_cells_3d
    equilibrium        -> tensorlbm.d3q19.equilibrium3d
    macroscopic        -> tensorlbm.d3q19.macroscopic3d
The single driver-level kernel is ``apply_body_force_3d`` below: the literal
D3Q19 transcription of ``tensorlbm.turbulent_channel._apply_body_force_2d``
(first-order Guo/Luo force f += w_i*3*rho*c_ix*a_x; injects exactly rho*a
per step because sum_i w_i c_ix^2 = 1/3).  It is applied with CALLER-SIDE
fluid masking ``torch.where(solid, f, forced)`` — the same disclosed
precedent as the archived 2D benchmark (unmasked injection on solid nodes
is reversed by bounce-back; 2D NOTES section 2.2).  The library has no 3D
single-phase body-force helper.

Step order (mirrors the library drainage loop and the archived 2D loop):
    collide -> stream -> (fluid-masked) body force -> bounce-back.

Subcommands
    case    run one (phi, N) case, write case_phi{phi}_N{N}[_fs{fs}][_tau{tau}][_smoke].json
    report  aggregate case files into result.json (machine-written verdicts)
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

WORKTREE_SRC = "/nfs/wangxi/worktrees/bm_w5/src"
sys.path.insert(0, WORKTREE_SRC)

import tensorlbm  # noqa: E402

assert tensorlbm.__file__.startswith(WORKTREE_SRC), (
    f"tensorlbm imported from {tensorlbm.__file__}, expected prefix {WORKTREE_SRC}"
)

import torch  # noqa: E402

from tensorlbm.boundaries3d import bounce_back_cells_3d  # noqa: E402
from tensorlbm.d3q19 import W_EXACT64, equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.d3q19 import C as D3Q19_C  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402

# --------------------------------------------------------------------------
# Registered constants (NOTES.md sections 2-3; must match preregistration)
# --------------------------------------------------------------------------
TAU = 1.0
NU = (TAU - 0.5) / 3.0  # = 1/6 exactly at tau = 1
RE_TARGET = 0.05  # Re_p = U*(2a)/nu, superficial U, nominal a
TOL = 0.03  # verdict tolerance on the finest tier

# Zick & Homsy 1982 SC table, two-source locked (NOTES section 1.4).
# Primary values = Holmes 4-digit literature column; Basilisk transcription
# in parentheses where it differs (<=0.015% at all protocol points).
ZH_TABLE: list[tuple[float, float]] = [
    (0.027, 2.0077),  # A: 2.008
    (0.064, 2.8102),  # A: 2.810
    (0.125, 4.292),  # A: 4.292
    (0.216, 7.4423),  # A: 7.442
    (0.343, 15.402),  # A: 15.4
    (0.45, 28.09),  # A: 28.1
    (0.5236, 41.99),  # A: 42.1 — divergent row, EXCLUDED from protocol
]
K_TABLE = dict(ZH_TABLE)
PHI_SET = [0.125, 0.216, 0.343]  # table-exact protocol set (NOTES 1.4)
LADDER: dict[float, list[int]] = {p: [64, 96, 128] for p in PHI_SET}

SAMPLE_EVERY = 200  # steps between samples
DRIFT_SPAN = 10  # samples in trailing drift window (= 2000 steps)
DRIFT_TOL = 1e-5  # relative drift threshold
STEADY_REPEATS = 3  # consecutive passing evaluations required
POST_STEADY = 8000  # extra steps after first steady pass
MEAS_SAMPLES = 20  # measurement window = last 20 samples = 4000 steps
MAX_STEPS_CAP = 250_000
MAX_STEPS_COEF = 20  # max_steps = min(20*N^2, 250000)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def library_provenance() -> dict:
    """Record where tensorlbm came from (read-only query, no repo writes)."""
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


def nominal_radius(n: int, phi: float) -> float:
    """a_nom = N*(3*phi/(4*pi))^(1/3) — NOTES sections 1.5 / 2."""
    return n * (3.0 * phi / (4.0 * math.pi)) ** (1.0 / 3.0)


def k_ref_lu(n: int, phi: float, k_table: float) -> float:
    """Closed-form lattice-unit permeability from the table: L^3/(6*pi*a*K)."""
    a = nominal_radius(n, phi)
    return n**3 / (6.0 * math.pi * a * k_table)


def loglog_k_of_phi(phi: float) -> float:
    """Diagnostic-only log-log interpolation of the locked table."""
    xs = [math.log(p) for p, _ in ZH_TABLE]
    ys = [math.log(k) for _, k in ZH_TABLE]
    x = math.log(phi)
    if x < xs[0] or x > xs[-1]:
        raise ValueError(f"phi={phi} outside table range")
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return math.exp(ys[i] + t * (ys[i + 1] - ys[i]))
    raise AssertionError("unreachable")


def sphere_array_mask_3d(n: int, phi: float, device: torch.device) -> torch.Tensor:
    """Boolean solid mask for one periodic simple-cubic unit cell N^3.

    Geometry only (not a physics kernel): sphere of nominal radius
    a = N*(3*phi/(4*pi))^(1/3) centred at (N/2, N/2, N/2); distances
    periodic-folded in all three directions; comparisons in float64.
    Style follows tensorlbm.porous_media.make_random_cylinder_medium /
    the archived 2D cylinder_array_mask.
    """
    import numpy as np

    a_real = nominal_radius(n, phi)
    idx = np.arange(n, dtype=np.float64)
    dd = np.abs(idx - n / 2.0)
    dd = np.minimum(dd, n - dd)  # periodic fold
    dist2 = dd[:, None, None] ** 2 + dd[None, :, None] ** 2 + dd[None, None, :] ** 2  # (nz, ny, nx)
    mask = dist2 <= a_real * a_real
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def apply_body_force_3d(f: torch.Tensor, a_x: float) -> torch.Tensor:
    """First-order Guo/Luo streamwise body force, D3Q19 form.

    Literal transcription of tensorlbm.turbulent_channel._apply_body_force_2d
    (D2Q9) to D3Q19: f_new = f + w_i * 3 * rho * c_ix * a_x.
    Injects exactly rho*a_x of x-momentum per step (sum_i w_i c_ix^2 = 1/3).
    Masking to fluid nodes is done by the caller (disclosed precedent).
    """
    device = f.device
    c = D3Q19_C.to(device).to(f.dtype)
    w = W_EXACT64.to(device=device, dtype=f.dtype).view(19, 1, 1, 1)
    rho = f.sum(dim=0)
    cx = c[:, 0].view(19, 1, 1, 1)
    return f + w * 3.0 * rho.unsqueeze(0) * cx * a_x


def fmt(x: float, digits: int = 6) -> str:
    return f"{x:.{digits}g}"


def run_case(args: argparse.Namespace) -> dict:
    phi = float(args.phi)
    n = int(args.n)
    tau = float(args.tau)
    if phi not in K_TABLE or phi not in PHI_SET:
        raise SystemExit(f"phi must be one of {PHI_SET}, got {phi}")
    device = torch.device(args.device)
    smoke = bool(args.smoke)
    force_scale = float(args.force_scale)

    k_table = K_TABLE[phi]
    nu_case = (tau - 0.5) / 3.0  # case viscosity (== NU at tau=1, bit-identical)
    a_nom = nominal_radius(n, phi)
    k_lu = k_ref_lu(n, phi, k_table)
    # Re_p = U*(2a)/nu with U = k_lu*a_body/nu  ->  a_body = Re*nu^2/(2a k_lu)
    # (nu_case^2 keeps the tau probe at FIXED Re, isolating BB tau-coupling;
    # identical to the preregistered formula at tau=1)
    a_body = RE_TARGET * nu_case * nu_case / (2.0 * a_nom * k_lu) * force_scale
    max_steps = (
        min(MAX_STEPS_COEF * n * n, MAX_STEPS_CAP) if not args.max_steps else int(args.max_steps)
    )
    sample_every = SAMPLE_EVERY
    drift_tol = DRIFT_TOL
    post_steady = POST_STEADY
    if smoke:  # plumbing check only, never a formal result
        max_steps = min(max_steps, 2000)
        sample_every = 100
        drift_tol = 1e-3
        post_steady = 300

    solid = sphere_array_mask_3d(n, phi, device)
    fluid = ~solid
    n_solid = int(solid.sum().item())
    phi_actual = n_solid / (n**3)

    rho0 = torch.ones(n, n, n, dtype=torch.float32, device=device)
    u0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, u0, u0, u0)
    mass0 = float(f.sum().item())
    del rho0, u0

    solid_b = solid.unsqueeze(0)

    def _step(f_t: torch.Tensor) -> torch.Tensor:
        f_t = collide_bgk3d(f_t, tau)
        f_t = stream3d(f_t)
        # Body force on fluid nodes only (torch.where masking; the force
        # inside the solid is inert and unmasked injection is reversed by
        # bounce-back — archived 2D benchmark NOTES section 2.2).
        f_t = torch.where(solid_b, f_t, apply_body_force_3d(f_t, a_body))
        return bounce_back_cells_3d(f_t, solid)

    hist_step: list[int] = []
    hist_uxf: list[float] = []
    hist_uyf: list[float] = []
    hist_uzf: list[float] = []
    hist_umax: list[float] = []
    hist_mass: list[float] = []
    recent_drifts: list[float] = []
    steady_step: int | None = None
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        f = _step(f)
        if step % sample_every == 0:
            _, ux, uy, uz = macroscopic3d(f)
            uxf = float(ux[fluid].mean().item())
            uyf = float(uy[fluid].mean().item())
            uzf = float(uz[fluid].mean().item())
            umax = float(torch.sqrt(ux * ux + uy * uy + uz * uz)[fluid].max().item())
            mass = float(f.sum().item())
            hist_step.append(step)
            hist_uxf.append(uxf)
            hist_uyf.append(uyf)
            hist_uzf.append(uzf)
            hist_umax.append(umax)
            hist_mass.append(mass)
            if steady_step is None and len(hist_uxf) >= DRIFT_SPAN + 1:
                drift = abs(hist_uxf[-1] - hist_uxf[-1 - DRIFT_SPAN]) / abs(hist_uxf[-1])
                recent_drifts.append(drift)
                if len(recent_drifts) >= STEADY_REPEATS and all(
                    v < drift_tol for v in recent_drifts[-STEADY_REPEATS:]
                ):
                    steady_step = step - DRIFT_SPAN * sample_every
        if steady_step is not None and step >= steady_step + post_steady:
            break

    wall = time.perf_counter() - t0
    steps_run = hist_step[-1] if hist_step else 0
    if steady_step is None:
        print(f"WARNING: phi={phi} N={n}: steady criterion not met in {max_steps} steps")

    def _win_mean(vals: list[float], n_win: int) -> float | None:
        return sum(vals[-n_win:]) / len(vals[-n_win:]) if len(vals) >= n_win else None

    uxf_meas = _win_mean(hist_uxf, MEAS_SAMPLES)
    uyf_meas = _win_mean(hist_uyf, MEAS_SAMPLES)
    uzf_meas = _win_mean(hist_uzf, MEAS_SAMPLES)
    umax_meas = _win_mean(hist_umax, MEAS_SAMPLES)
    assert uxf_meas is not None, "measurement window empty"

    # ---- primary channel (nominal geometry; NOTES sections 2-3) ----
    k_sim = nu_case * uxf_meas * (1.0 - phi) / a_body
    k_ref = k_ref_lu(n, phi, k_table)
    K_sim = a_body * n**3 / (6.0 * math.pi * nu_case * a_nom * (1.0 - phi) * uxf_meas)
    signed = k_sim / k_ref - 1.0
    err = abs(signed)

    # ---- diagnostic channel: phi_actual geometry (NOT verdict-bearing) ----
    k_table_actual = loglog_k_of_phi(phi_actual)
    a_stair = (3.0 * phi_actual / (4.0 * math.pi)) ** (1.0 / 3.0) * n
    K_sim_actual = (
        a_body * n**3 / (6.0 * math.pi * nu_case * a_stair * (1.0 - phi_actual) * uxf_meas)
    )
    err_actualphi = abs(K_sim_actual / k_table_actual - 1.0)

    u_super_actual = uxf_meas * (1.0 - phi_actual)
    re_achieved = u_super_actual * 2.0 * a_nom / nu_case
    ma = math.sqrt(3.0) * (umax_meas or 0.0)
    mass_drift = max(abs(m / mass0 - 1.0) for m in hist_mass) if hist_mass else float("nan")

    record: dict = {
        "kind": "case",
        "created_utc": utc_now(),
        "smoke": smoke,
        "run": {
            "phi": phi,
            "N": n,
            "tau": tau,
            "nu": nu_case,
            "a_nom": a_nom,
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
            "device": str(device),
            "dtype": "float32",
            "wall_seconds": wall,
        },
        "geometry": {
            "N_solid": n_solid,
            "phi_actual": phi_actual,
            "a_nom": a_nom,
            "mask_rule": (
                "periodic-folded (dx^2+dy^2+dz^2) <= a_nom^2, float64, centre (N/2,N/2,N/2)"
            ),
        },
        "measurement": {
            "uxf_mean_win4000": uxf_meas,
            "uyf_mean_win4000": uyf_meas,
            "uzf_mean_win4000": uzf_meas,
            "umax_mean_win4000": umax_meas,
            "mass_drift_max_rel": mass_drift,
            "u_superficial_actualphi": u_super_actual,
        },
        "derived": {
            "K_table": k_table,
            "K_sim": K_sim,
            "k_sim": k_sim,
            "k_ref": k_ref,
            "err_primary": err,
            "signed_err_primary": signed,
            "K_sim_actualphi_diag": K_sim_actual,
            "K_table_at_phi_actual_diag": k_table_actual,
            "err_actualphi_diag": err_actualphi,
            "Re_p_achieved": re_achieved,
            "Ma_max": ma,
        },
        "history": {
            "sample_steps": hist_step,
            "uxf": hist_uxf,
            "umax": hist_umax,
            "mass": hist_mass,
        },
        "library": library_provenance(),
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    suffix = f"_tau{tau:g}" if tau != TAU else ""
    suffix += f"_fs{force_scale:g}" if force_scale != 1.0 else ""
    suffix += "_smoke" if smoke else ""
    suffix += f"_{tag}" if tag else ""
    name = out / f"case_phi{phi}_N{n}{suffix}.json"
    name.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        f"phi={phi} N={n} fs={force_scale:g} tau={tau:g} -> {name.name}\n"
        f"  steps_run={steps_run} steady_step={steady_step} wall={wall:.1f}s\n"
        f"  phi_actual={phi_actual:.6f} uxf={uxf_meas:.6e}\n"
        f"  K_sim={K_sim:.5f} K_table={k_table} err={err * 100:.3f}% "
        f"(actualphi diag {err_actualphi * 100:.3f}%)\n"
        f"  Re_p={re_achieved:.4f} Ma={ma:.2e} mass_drift={mass_drift:.2e}"
    )
    return record


# --------------------------------------------------------------------------
# report aggregation
# --------------------------------------------------------------------------


def load_cases(out_dir: Path, formal_fs: float) -> dict:
    cases = []
    for p in sorted(out_dir.glob("case_*.json")):
        c = json.loads(p.read_text())
        c["case_file"] = p.name
        cases.append(c)
    formal, fs1, probe, precheck, smokes = [], [], [], [], []
    for c in cases:
        if c.get("smoke"):
            smokes.append(c)
            continue
        r = c["run"]
        fs = float(r["force_scale"])
        tau = float(r["tau"])
        key = (r["phi"], r["N"])
        if fs == formal_fs and tau == TAU:
            formal.append(c)
        if fs == 1.0 and tau == TAU:
            fs1.append(c)
        if tau != TAU:
            probe.append(c)
        if key == (0.343, 128) and tau == TAU:
            precheck.append(c)
    return {
        "formal": formal,
        "fs1": fs1,
        "tau_probe": probe,
        "fs_precheck_pool": precheck,
        "smoke": smokes,
    }


def ladder_verdict(tiers: list[dict]) -> dict:
    """Pre-registered verdict rule (NOTES section 3, frozen)."""
    errs = [t["err"] for t in tiers]
    ns = [t["N"] for t in tiers]
    mono = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    finest_ok = errs[-1] <= TOL
    fallback = False
    verdict: str
    if mono and finest_ok:
        verdict = "PASS"
    elif len(errs) >= 3 and errs[-2] > errs[-1] and errs[-1] <= TOL and errs[0] <= errs[1]:
        # pre-registered fallback: non-monotonicity confined to the coarsest
        # tier (staircase-dominated) and finest two strictly decreasing
        fallback = True
        verdict = "PASS_WITH_DISCLOSURE"
    else:
        verdict = "FAIL"
    return {
        "tiers_N": ns,
        "tiers_err": errs,
        "strictly_monotonic": mono,
        "finest_err_le_tol": finest_ok,
        "fallback_applied": fallback,
        "verdict": verdict,
    }


def amended_verdict(tiers: list[dict]) -> dict:
    """Amendment #1 verdict rule (NOTES 5.2, locked before the N=160 runs).

    Envelope-convergence rule over N=64/96/128/160 at formal fs:
    PASS iff err(160) <= 3% AND err(160) < err(64) AND no intermediate
    tier exceeds err(64).
    """
    errs = [t["err"] for t in tiers]
    ns = [t["N"] for t in tiers]
    c1 = errs[-1] <= TOL
    c2 = errs[-1] < errs[0]
    c3 = all(e <= errs[0] for e in errs[1:-1])
    return {
        "tiers_N": ns,
        "tiers_err": errs,
        "cond_i_finest_le_tol": c1,
        "cond_ii_finest_lt_coarsest": c2,
        "cond_iii_intermediate_le_coarsest": c3,
        "verdict": "PASS" if (c1 and c2 and c3) else "FAIL",
    }


def make_report(args: argparse.Namespace) -> dict:
    out = Path(args.out_dir)
    formal_fs = float(args.formal_fs)
    pools = load_cases(out, formal_fs)

    by_phi: dict[float, list[dict]] = {}
    for c in pools["formal"]:
        by_phi.setdefault(c["run"]["phi"], []).append(c)

    ladder_summary: dict[str, dict] = {}
    per_phi_verdict: dict[str, str] = {}
    amended_summary: dict[str, dict] = {}
    per_phi_amended: dict[str, str] = {}
    for phi in sorted(by_phi):
        cases = sorted(by_phi[phi], key=lambda c: c["run"]["N"])
        have = sorted(c["run"]["N"] for c in cases)
        tiers = [
            {
                "N": c["run"]["N"],
                "case_file": c["case_file"],
                "phi_actual": c["geometry"]["phi_actual"],
                "steps_run": c["run"]["steps_run"],
                "steady_detected": c["run"]["steady_detected"],
                "K_sim": c["derived"]["K_sim"],
                "K_table": c["derived"]["K_table"],
                "k_sim": c["derived"]["k_sim"],
                "k_ref": c["derived"]["k_ref"],
                "err": c["derived"]["err_primary"],
                "err_actualphi_diag": c["derived"]["err_actualphi_diag"],
                "Re_p_achieved": c["derived"]["Re_p_achieved"],
                "Ma_max": c["derived"]["Ma_max"],
                "mass_drift_max_rel": c["measurement"]["mass_drift_max_rel"],
                "wall_seconds": c["run"]["wall_seconds"],
            }
            for c in cases
        ]
        if not set(LADDER[phi]).issubset(have):
            print(f"WARNING: phi={phi} frozen ladder incomplete: have {have}")
        frozen_tiers = [t for t in tiers if t["N"] in (64, 96, 128)]
        v = ladder_verdict(frozen_tiers)
        ladder_summary[f"phi{phi}"] = {"tiers": frozen_tiers, **v}
        per_phi_verdict[f"phi{phi}"] = v["verdict"]
        amend_tiers = [t for t in tiers if t["N"] in (64, 96, 128, 160)]
        va = amended_verdict(amend_tiers)
        amended_summary[f"phi{phi}"] = {"tiers": amend_tiers, **va}
        per_phi_amended[f"phi{phi}"] = va["verdict"]

    overall = (
        "PASS"
        if all(x == "PASS" for x in per_phi_verdict.values()) and per_phi_verdict
        else (
            "PASS_WITH_DISCLOSURE"
            if per_phi_verdict
            and all(x in ("PASS", "PASS_WITH_DISCLOSURE") for x in per_phi_verdict.values())
            else "FAIL"
        )
    )
    overall_amended = (
        "PASS" if per_phi_amended and all(x == "PASS" for x in per_phi_amended.values()) else "FAIL"
    )

    fs_precheck = None
    pc = sorted(pools["fs_precheck_pool"], key=lambda c: c["run"]["force_scale"])
    if len(pc) >= 2:
        ks = {f"{c['run']['force_scale']:g}": c["derived"]["k_sim"] for c in pc}
        vals = list(ks.values())
        spread = max(abs(v / min(vals) - 1.0) for v in vals)
        fs_precheck = {
            "case": "phi=0.343 N=128",
            "k_sim_by_fs": ks,
            "max_rel_spread": spread,
            "threshold": 0.005,
            "floor_active": spread > 0.005,
        }

    tau_block = None
    if pools["tau_probe"]:
        tau_block = {}
        for c in pools["tau_probe"]:
            tau_block[f"tau{c['run']['tau']:g}"] = {
                "phi": c["run"]["phi"],
                "N": c["run"]["N"],
                "err_primary": c["derived"]["err_primary"],
                "K_sim": c["derived"]["K_sim"],
            }

    fs1_block = {}
    for c in pools["fs1"]:
        fs1_block[f"phi{c['run']['phi']}_N{c['run']['N']}"] = {
            "err_primary": c["derived"]["err_primary"],
            "K_sim": c["derived"]["K_sim"],
            "k_sim": c["derived"]["k_sim"],
        }

    report = {
        "kind": "report",
        "benchmark": "permeability3d_sphere_array_sc_zick_homsy",
        "created_utc": utc_now(),
        "formal_force_scale": formal_fs,
        "tolerance_finest": TOL,
        "reference": {
            "type": "Zick & Homsy 1982 JFM 115:13-26, simple-cubic array",
            "K_def": "F/(6*pi*mu*U*a), U superficial",
            "locked_table_protocol_points": {
                "0.125": 4.292,
                "0.216": 7.4423,
                "0.343": 15.402,
            },
            "two_source_cross_max_rel_diff_protocol_points": 0.00015,
            "sources": [
                "Basilisk src/test/spheres.c zick[7][2] (verbatim, fetched 2026-09-20)",
                "Holmes, Williams & Tilke 2011 DOI 10.1002/nag.898, literature column "
                "(verbatim snippet witnesses; NOTES section 1.3)",
            ],
            "corroboration": "Hasimoto 1959 5-term dilute SC series (NOTES 1.6)",
        },
        "protocol": {
            "lattice": "D3Q19 BGK fp32 periodic unit cell",
            "tau": TAU,
            "nu": NU,
            "step_order": "collide -> stream -> masked body force -> bounce-back",
            "phi_set": PHI_SET,
            "tiers": LADDER,
            "Re_target": RE_TARGET,
            "Ma_limit": 0.05,
            "sample_every": SAMPLE_EVERY,
            "drift_window_steps": DRIFT_SPAN * SAMPLE_EVERY,
            "drift_tol": DRIFT_TOL,
            "post_steady_steps": POST_STEADY,
            "meas_window_steps": MEAS_SAMPLES * SAMPLE_EVERY,
            "max_steps_rule": f"min({MAX_STEPS_COEF}*N^2, {MAX_STEPS_CAP})",
            "verdict_rule": (
                "per phi: err strictly monotonically decreasing N=64->96->128 "
                f"AND err(128)<={TOL:.0%}; pre-registered coarsest-tier fallback "
                "(NOTES section 3)"
            ),
        },
        "fs_precheck": fs_precheck,
        "fs1_double_report": fs1_block,
        "tau_probe": tau_block,
        "ladders": ladder_summary,
        "per_phi_verdict": per_phi_verdict,
        "overall_verdict": overall,
        "amendment1_ladders": amended_summary,
        "per_phi_verdict_amended": per_phi_amended,
        "overall_verdict_amended": overall_amended,
        "disclosures": [
            "phi set deviates from instructed {0.10,0.20,0.30} to table-exact "
            "{0.125,0.216,0.343} (no reference interpolation; NOTES 1.4)",
            "3D body force is a driver-level D3Q19 transcription of "
            "turbulent_channel._apply_body_force_2d with caller-side fluid "
            "masking; library has no 3D single-phase helper (NOTES 4)",
            "staircase effective radius O(1/R): nominal geometry primary; "
            "phi_actual channels diagnostic only",
            "full-way bounce-back tau-sensitivity is literature-standard "
            "+/-2.5%; tau probe quantifies it (verdict-blind)",
            "fp32 injection rounding floor prechecked at (0.343,128) fs in "
            "{1,10,100}; formal fs per pre-registered decision rule",
            "Z&H row 0.5236 diverges between sources (42.1 vs 41.99, 0.26%) "
            "and is excluded from the protocol",
            "Source B witnessed via verbatim search-snippet fragments of the "
            "Wiley page (page itself blocked); NOTES 1.3 caveat",
        ],
        "library": (pools["formal"] or pools["fs1"] or pools["smoke"])[0].get("library"),
    }

    # attach case_file provenance: load_cases already stamped each case dict
    result_path = out / "result.json"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"report -> {result_path}")
    print(json.dumps(per_phi_verdict, indent=2))
    print(f"overall: {overall}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_case = sub.add_parser("case")
    ap_case.add_argument("--phi", type=float, required=True)
    ap_case.add_argument("--n", type=int, required=True)
    ap_case.add_argument("--force-scale", type=float, default=1.0)
    ap_case.add_argument("--tau", type=float, default=TAU)
    ap_case.add_argument("--device", default="cuda:5")
    ap_case.add_argument("--out-dir", default=".")
    ap_case.add_argument("--max-steps", type=int, default=0)
    ap_case.add_argument("--smoke", action="store_true")
    ap_case.add_argument("--tag", default="")
    ap_case.set_defaults(func=run_case)

    ap_rep = sub.add_parser("report")
    ap_rep.add_argument("--out-dir", default=".")
    ap_rep.add_argument("--formal-fs", type=float, default=10.0)
    ap_rep.set_defaults(func=make_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
