#!/usr/bin/env python3
"""W9-C benchmark: Gebart (1992) transverse permeability of periodic cylinder
arrays (square and hexagonal packing), D2Q9 LBM.

References (locked in prereg.md before any formal run):
  primary, square tiers (ALL of them): Sangani & Acrivos 1982 exact
    f-table (pressure-form drag), W4-A double transcription; k_ref_exact
    = pi/(vf*f) * R^2, f in SANGANI_F_SQ below.
  primary, hex near-packing (0.70/0.75 main): Gebart 1992 closed form
    K_perp / R^2 = C1 * (sqrt(vf_max/vf) - 1)**2.5
    square : C1 = 16/(9*pi*sqrt(2)) = 0.4001405849587144, vf_max = pi/4
    hex    : C1 = 16/(9*pi*sqrt(6)) = 0.2310212744396075, vf_max = pi/(2*sqrt3)
    (Darcy/superficial convention; +0.268% vs exact at Vf=0.75 square)
  secondary channel (every tier, disclosed): Gebart closed form, so the
    square ladders simultaneously verify the locked Gebart-vs-exact bias.

Lock channels: (1) Yazdchi, Srivastava & Luding 2011 IJMF Table 1 verbatim
transcription (K/d^2 form, C1 = 4/(9 pi sqrt2) sq / 4/(9 pi sqrt6) hex,
eps_c = 1 - pi/4 / 1 - pi/(2 sqrt3)); (2) citing-docs R^2-form constant
tables (Toulouse thesis 2017, Semantic Scholar PDF, Jia et al.); (3) own
lubrication derivations (square single-throat, hex honeycomb-network
Kirchhoff, k_hex = k_sq/sqrt(3)); (4) Sangani point anchors.  The
alternative constants 4/(9 pi sqrt2)*r^2 and 16/(9 pi sqrt3) are falsified
- see prereg.md section 2.

Observable (Darcy convention, W4-A NOTES section 3 amendment):
    K_sim = (1 - vf_actual) * nu * <u_x>_fluid / a_body     [lu^2]
    err   = |K_sim / K_ref(nominal vf, nominal R) - 1|       (main channel)
    err_B = same but K_ref evaluated at vf_actual with effective radius
            (diagnostic channel, disclosed only, never re-judged)

Geometry (not a physics kernel; mask style of
tensorlbm.porous_media.make_random_cylinder_medium):
    sq : W x W cell, one cylinder of radius R = W*sqrt(vf/pi) centred
         (W4-A inheritance, periodic folded distance).
    hex: W x H supercell, H = round(sqrt(3)*W); cylinders at the four folded
         corners (one cylinder) + the centre (one cylinder) - together one
         triangular lattice with nearest-neighbour distance W;
         R = sqrt(vf*W*H/(2*pi)).

All physics kernels from the tensorlbm library (grep self-check: no
hand-written collide/stream/bounce-back/force kernels in this file):
    collide/stream -> tensorlbm.solver (D2Q9, periodic stream)
    bounce-back    -> tensorlbm.boundaries
    body force     -> tensorlbm.turbulent_channel._apply_body_force_2d
    equilibrium/moments -> tensorlbm.d2q9
Step order (library turbulent_channel loop): collide -> stream ->
(fluid-masked) body force -> bounce-back.

Subcommands
    case    run one (array, vf, W) case, write case JSON
    report  aggregate case files into result.json + README.md
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

_HERE = Path(__file__).resolve().parent
# <repo>/benchmarks (compile_route adapter); tensorlbm from the installed package
sys.path.insert(0, str(_HERE.parents[1]))

import torch  # noqa: E402
from compile_route import (  # noqa: E402
    add_compile_mode_arg,
    compile_mode_from_args,
    route_step,
)

from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402
from tensorlbm.turbulent_channel import _apply_body_force_2d  # noqa: E402

# --------------------------------------------------------------------------
# Registered constants (prereg.md; must match the frozen preregistration)
# --------------------------------------------------------------------------
TAU = 1.0
NU = (TAU - 0.5) / 3.0  # = 1/6 exactly
RE_TARGET = 0.05  # Re_cell = a * k_lu * R / nu^2, solved for a
FS_DEFAULT = 10.0  # force scale (W4-A fs=10 adjudicated protocol)

C1_SQ = 16.0 / (9.0 * math.pi * math.sqrt(2.0))  # 0.4001405849587144
C1_HEX = 16.0 / (9.0 * math.pi * math.sqrt(6.0))  # 0.2310212744396075
VFMAX = {"sq": math.pi / 4.0, "hex": math.pi / (2.0 * math.sqrt(3.0))}
C1 = {"sq": C1_SQ, "hex": C1_HEX}

# Reference structure (prereg section 3; controller adjudication):
#   sq  main = {0.30, 0.40, 0.50, 0.60} vs Sangani exact f-table
#       (table values only; 0.45 dropped - log-log chord interpolation
#       unreliable, chords differ 8.5%)
#   sq  disclosure = 0.70 (exact-gated, non-gating; 0.75 dropped: W=2107
#       = 4.4M cells exceeds shared-GPU budget, near-packing covered by
#       0.70 + hex 0.75)
#   hex main = {0.70, 0.75} vs Gebart (near-packing asymptote accurate by
#       construction + dual lubrication anchoring; NO exact hex table
#       obtainable - 8-channel hunt documented in prereg section 3)
#   hex disclosure = {0.30, 0.45, 0.60} (reference-limited, non-gating)
MAIN_TIERS = {"sq": (0.30, 0.40, 0.50, 0.60), "hex": (0.70, 0.75)}
DISCLOSURE_TIERS = {"sq": (0.70,), "hex": (0.30, 0.45, 0.60)}
LADDER: dict[tuple[str, float], list[int]] = {
    ("sq", 0.30): [32, 64, 128, 192],
    ("sq", 0.40): [42, 84, 168, 252],
    ("sq", 0.50): [60, 120, 238, 357],
    ("sq", 0.60): [96, 192, 384, 573],
    ("sq", 0.70): [216, 432, 864],
    ("hex", 0.30): [28, 56, 113],
    ("hex", 0.45): [40, 81, 162],
    ("hex", 0.60): [64, 128, 258],
    ("hex", 0.70): [100, 198, 396, 594],
    ("hex", 0.75): [132, 265, 530, 795],
}

# Sangani & Acrivos 1982 exact f-table, square arrays: pressure-form drag
# f = G*L^2/(mu*U_D). W4-A double transcription (Basilisk cylinders.c
# sangani[9][2] + Sharaborin/Rogozin/Kasimov, Fluids 2021 6(9):334
# Table 1), bit-identical; asymptotic cross-checks in W4-A NOTES 1.3.
SANGANI_F_SQ = {
    0.05: 15.56,
    0.10: 24.83,
    0.20: 51.53,
    0.30: 102.90,
    0.40: 217.89,
    0.50: 532.55,
    0.60: 1.763e3,
    0.70: 1.352e4,
    0.75: 1.263e5,
}


def sangani_k_over_r2(vf: float) -> float:
    """Exact square-array K/R^2 from the Sangani f-table (Darcy conv.).

    k_ref = L^2/f with phi = pi R^2/L^2  =>  K/R^2 = pi/(phi*f).
    """
    f = SANGANI_F_SQ.get(round(vf, 4))
    if f is None:
        raise ValueError(f"vf={vf} not a Sangani table value {sorted(SANGANI_F_SQ)}")
    return math.pi / (vf * f)


SAMPLE_EVERY = 200
DRIFT_SPAN = 10  # samples in the trailing drift window (= 2000 steps)
DRIFT_TOL = 1e-5
STEADY_REPEATS = 3
POST_STEADY = 8000
MEAS_SAMPLES = 20  # measurement window = last 20 samples = 4000 steps
S2_WIN_A = 10
S2_WIN_B = 40
FP32_MARGIN_MIN = 14.0  # 3*a_body / 2^-24 must exceed this (W4-A criterion)

# Locked reference-deviation disclosure: Gebart(square) vs Sangani exact
# numerics (percent, Gebart/exact - 1, double precision, prereg section 4).
SANGANI_EXACT_BIAS_SQ = {
    0.05: 49.793,
    0.10: 37.951,
    0.20: 25.331,
    0.30: 18.062,
    0.40: 13.212,
    0.50: 9.532,
    0.60: 6.226,
    0.70: 2.978,
    0.75: 0.268,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gebart_k_over_r2(arr: str, vf: float) -> float:
    """Locked Gebart transverse closed form, K/R^2 (Darcy convention)."""
    if not 0.0 < vf < VFMAX[arr]:
        raise ValueError(f"vf={vf} outside (0, vf_max={VFMAX[arr]}) for {arr}")
    return C1[arr] * (math.sqrt(VFMAX[arr] / vf) - 1.0) ** 2.5


def library_provenance() -> dict:
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
    except Exception as exc:  # noqa: BLE001
        info["library_commit_error"] = repr(exc)
    return info


def cylinder_array_mask(arr: str, W: int, H: int, vf: float, device: torch.device) -> torch.Tensor:
    """Boolean solid mask, periodic folded distance, float64 comparisons.

    sq : one cylinder centred at (W/2, H/2).
    hex: min distance to corner site (0,0) or centre site (W/2, H/2).
    """
    import numpy as np

    def fold(ix, ref: float, n: int):
        d = np.abs(ix - ref)
        return np.minimum(d, n - d)

    ix = np.arange(W, dtype=np.float64)
    iy = np.arange(H, dtype=np.float64)
    if arr == "sq":
        r_real = W * math.sqrt(vf / math.pi)
        dx = fold(ix, W / 2.0, W)
        dy = fold(iy, H / 2.0, H)
        dist2 = dx[None, :] ** 2 + dy[:, None] ** 2
        mask = dist2 <= r_real * r_real
    elif arr == "hex":
        r_real = math.sqrt(vf * W * H / (2.0 * math.pi))
        dx0 = fold(ix, 0.0, W)
        dy0 = fold(iy, 0.0, H)
        dxc = fold(ix, W / 2.0, W)
        dyc = fold(iy, H / 2.0, H)
        d2_corner = dx0[None, :] ** 2 + dy0[:, None] ** 2
        d2_centre = dxc[None, :] ** 2 + dyc[:, None] ** 2
        dist2 = np.minimum(d2_corner, d2_centre)
        mask = dist2 <= r_real * r_real
    else:
        raise ValueError(f"array must be sq or hex, got {arr}")
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def nominal_radius(arr: str, W: int, H: int, vf: float) -> float:
    if arr == "sq":
        return W * math.sqrt(vf / math.pi)
    return math.sqrt(vf * W * H / (2.0 * math.pi))


def fmt(x: float, digits: int = 6) -> str:
    return f"{x:.{digits}g}"


# --------------------------------------------------------------------------
# case
# --------------------------------------------------------------------------
def run_case(args: argparse.Namespace) -> dict:
    arr = str(args.array)
    vf = float(args.vf)
    W = int(args.w)
    if arr not in ("sq", "hex"):
        raise SystemExit(f"array must be sq or hex, got {arr}")
    H = W if arr == "sq" else int(round(math.sqrt(3.0) * W))
    device = torch.device(args.device)
    smoke = bool(args.smoke)
    force_scale = float(args.force_scale)

    r_real = nominal_radius(arr, W, H, vf)
    n_cyl = 1 if arr == "sq" else 2
    gap_lu = W - 2.0 * r_real
    # Primary reference per prereg section 3: square -> Sangani exact
    # f-table (all square tiers); hex -> Gebart closed form (no exact hex
    # table obtainable). Secondary Gebart channel always computed too.
    k_ref_gebart_lu = gebart_k_over_r2(arr, vf) * r_real * r_real
    if arr == "sq":
        f_sangani = SANGANI_F_SQ[round(vf, 4)]
        k_ref_exact_lu = sangani_k_over_r2(vf) * r_real * r_real
        k_ref_primary = k_ref_exact_lu
        ref_primary = "sangani_exact_sq"
    else:
        f_sangani = None
        k_ref_exact_lu = None
        k_ref_primary = k_ref_gebart_lu
        ref_primary = "gebart"
    a_body = RE_TARGET * NU * NU / (k_ref_primary * r_real) * force_scale
    fp32_margin = 3.0 * a_body / 2.0**-24
    if fp32_margin < FP32_MARGIN_MIN:
        raise SystemExit(
            f"fp32 injection margin {fp32_margin:.2f} < {FP32_MARGIN_MIN}; "
            "increase force scale or grid"
        )
    max_steps = int(args.max_steps) if args.max_steps else max(60_000, min(40 * W * H, 1_600_000))
    sample_every = SAMPLE_EVERY
    drift_tol = DRIFT_TOL
    post_steady = POST_STEADY
    if smoke:  # plumbing check only, never a formal result
        max_steps = min(max_steps, 3000)
        sample_every = 100
        drift_tol = 1e-3
        post_steady = 400

    solid = cylinder_array_mask(arr, W, H, vf, device)
    fluid = ~solid
    n_solid = int(solid.sum().item())
    vf_actual = n_solid / (W * H)

    rho0 = torch.ones(H, W, dtype=torch.float32, device=device)
    u0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, u0, u0, device=device)
    mass0 = float(f.sum().item())
    del rho0, u0

    solid_b = solid.unsqueeze(0)

    def _step(f_t: torch.Tensor) -> torch.Tensor:
        f_t = collide_bgk(f_t, TAU)
        f_t = stream(f_t)
        # fluid-masked body force (unmasked injection is partly reversed by
        # bounce-back; W4-A NOTES section 2.2)
        f_t = torch.where(solid_b, f_t, _apply_body_force_2d(f_t, a_body))
        return bounce_back_cells(f_t, solid)

    step_fn = route_step(_step, args.compile_mode, name=f"w9c[{arr}{vf}_w{W}]")

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
        print(f"WARNING: {arr} vf={vf} W={W}: steady criterion not met in {max_steps} steps")

    def _win_mean(vals: list[float], n: int) -> float | None:
        return sum(vals[-n:]) / len(vals[-n:]) if len(vals) >= n else None

    uxf_meas = _win_mean(hist_uxf, MEAS_SAMPLES)
    uxf_w2000 = _win_mean(hist_uxf, S2_WIN_A)
    uxf_w8000 = _win_mean(hist_uxf, S2_WIN_B)
    uyf_meas = _win_mean(hist_uyf, MEAS_SAMPLES)
    umax_meas = _win_mean(hist_umax, MEAS_SAMPLES)
    uymax_meas = _win_mean(hist_uymax, MEAS_SAMPLES)
    assert uxf_meas is not None, "measurement window empty"

    # Darcy/superficial convention (prereg): masked body force a_body is
    # equivalent to macroscopic gradient G = a_body; U_D = (1-vf)*<u>_fluid.
    k_sim_fluid = NU * uxf_meas / a_body  # nu*<u>_fluid/a
    K_sim = (1.0 - vf_actual) * k_sim_fluid  # Darcy K [lu^2]
    signed = K_sim / k_ref_primary - 1.0  # primary-gate channel
    err = abs(signed)
    signed_gebart = K_sim / k_ref_gebart_lu - 1.0  # consistency channel
    err_gebart = abs(signed_gebart)
    r_eff = math.sqrt(vf_actual * W * H / (n_cyl * math.pi))
    k_ref_b = gebart_k_over_r2(arr, vf_actual) * r_eff * r_eff
    err_b = abs(K_sim / k_ref_b - 1.0)
    u_d_meas = uxf_meas * (1.0 - vf_actual)
    re_achieved = u_d_meas * r_real / NU
    ma = math.sqrt(3.0) * (umax_meas or 0.0)
    uy_ratio = (uymax_meas or 0.0) / abs(uxf_meas) if uxf_meas else float("nan")
    mass_drift = max(abs(m / mass0 - 1.0) for m in hist_mass) if hist_mass else float("nan")

    record = {
        "kind": "case",
        "created_utc": utc_now(),
        "smoke": smoke,
        "run": {
            "array": arr,
            "vf": vf,
            "tier": "disclosure" if vf in DISCLOSURE_TIERS[arr] else "main",
            "W": W,
            "H": H,
            "tau": TAU,
            "nu": NU,
            "R_real": r_real,
            "n_cylinders": n_cyl,
            "gap_lu": gap_lu,
            "ref_primary": ref_primary,
            "k_ref_primary": k_ref_primary,
            "k_ref_gebart": k_ref_gebart_lu,
            "k_ref_exact": k_ref_exact_lu,
            "f_sangani": f_sangani,
            "a_body": a_body,
            "force_scale": force_scale,
            "fp32_margin": fp32_margin,
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
            "vf_actual": vf_actual,
            "n_cylinders": n_cyl,
            "nn_distance_lu": float(W),
            "mask_rule": (
                "periodic-folded (dx^2+dy^2) <= R^2 to centre (sq) or "
                "min(corner, centre) (hex), float64"
            ),
            "aspect_H_over_W": H / W,
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
            "k_sim_fluid": k_sim_fluid,
            "K_sim_darcy": K_sim,
            "K_ref_primary": k_ref_primary,
            "err_rel": err,
            "signed_err_rel": signed,
            "err_pct": err * 100.0,
            "signed_err_pct": signed * 100.0,
            "K_ref_gebart": k_ref_gebart_lu,
            "signed_gebart_pct": signed_gebart * 100.0,
            "err_gebart_pct": err_gebart * 100.0,
            "K_ref_diagB": k_ref_b,
            "r_eff_diagB": r_eff,
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
    tag = "" if force_scale == FS_DEFAULT else f"_fs{force_scale:g}"
    name = (
        f"case_smoke_{arr}_vf{vf:g}_w{W}{tag}.json"
        if smoke
        else f"case_{arr}_vf{vf:g}_w{W}{tag}.json"
    )
    out = out_dir / name
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[case] {arr} vf={vf:g} W={W} fs={force_scale:g} -> {out}")
    print(
        f"  steady={steady_step} steps={steps_run} wall={wall:.1f}s "
        f"vf_act={vf_actual:.6f} K_sim={K_sim:.6g} K_ref[{ref_primary}]="
        f"{k_ref_primary:.6g} err={err * 100:.4f}% gebart="
        f"{signed_gebart * 100:+.4f}% err_B={err_b * 100:.4f}% "
        f"Re={re_achieved:.4f} Ma={ma:.2e} margin={fp32_margin:.1f}"
    )
    return record


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def cmd_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    result: dict = {
        "kind": "result",
        "created_utc": utc_now(),
        "benchmark": "gebart_transverse_permeability_sq_hex_2d",
        "reference": {
            "primary": {
                "sq": "Sangani & Acrivos 1982 exact f-table (W4-A double "
                "transcription: Basilisk cylinders.c + Fluids 2021 "
                "6(9):334 Table 1, bit-identical); k_ref = "
                "pi/(vf*f)*R^2; ALL square tiers incl. 0.70 disclosure",
                "hex_0.70_0.75": "Gebart 1992 closed form (near-packing "
                "asymptote, dual lubrication anchoring)",
            },
            "secondary_every_tier": "Gebart closed form (consistency "
            "channel; square ladders must reproduce "
            "the locked Gebart-vs-exact bias)",
            "gebart_formula": "K_perp = C1 * R^2 * (sqrt(vf_max/vf) - 1)**2.5",
            "gebart_constants": {
                "sq": {"C1": C1_SQ, "vf_max": VFMAX["sq"]},
                "hex": {"C1": C1_HEX, "vf_max": VFMAX["hex"]},
            },
            "sangani_f_table_sq": SANGANI_F_SQ,
            "convention": "Darcy/superficial (K = mu*U_sup/G)",
            "lock_channels": [
                "Yazdchi/Srivastava/Luding 2011 IJMF Table 1 verbatim (K/d^2 form)",
                "Yazdchi PhD thesis Table 2.1 (Gebart C and eps_c, K/d^2 form)",
                "citing-docs R^2-form tables (Toulouse thesis, SemanticScholar, Jia)",
                "own lubrication derivations (sq single-throat, hex honeycomb Kirchhoff)",
                "Sangani-Acrivos 1982 exact anchors (sq: +0.268% at Vf=0.75)",
            ],
        },
        "reference_deviation_disclosure": {
            "sq_gebart_vs_sangani_exact_pct": SANGANI_EXACT_BIAS_SQ,
            "note": (
                "Gebart is a near-packing asymptote; locked square "
                "deviations vs exact numerics above. Square tiers are "
                "exact-gated, so these deviations are NOT in the gate; "
                "the per-case gebart channel verifies them in situ."
            ),
            "hex_exact_table": "NOT OBTAINABLE - 8-channel hunt documented "
            "in prereg section 3 (S&A 1982, Wang-Sangani "
            "1997, Koch-Ladd 1997, Gebart 1992, Yazdchi "
            "IJMF+thesis, Monash, basilisk.fr, MDPI); "
            "hex main gate = Gebart at near-packing only, "
            "hex dilute tiers reference-limited disclosure",
        },
        "settings": {
            "tau": TAU,
            "nu": NU,
            "Re_target": RE_TARGET,
            "force_scale_default": FS_DEFAULT,
            "sample_every": SAMPLE_EVERY,
            "drift_tol": DRIFT_TOL,
            "meas_window_steps": MEAS_SAMPLES * SAMPLE_EVERY,
            "criterion": "err (vs tier's primary reference) strictly "
            "decreasing with refinement AND finest <= 3% AND "
            "steady detected on every grid",
            "extrap": "none",
        },
        "ladders": {},
        "sensitivity": {},
        "verdict": {},
    }

    ladders: dict[str, list[dict]] = {}
    for (arr, vf), Ws in LADDER.items():
        entries = []
        for W in Ws:
            p = out_dir / f"case_{arr}_vf{vf:g}_w{W}.json"
            if not p.exists():
                print(f"WARNING: missing {p}")
                continue
            rec = json.loads(p.read_text())
            if rec.get("smoke"):
                continue
            r, der, geo = rec["run"], rec["derived"], rec["geometry"]
            entries.append(
                {
                    "array": arr,
                    "vf": vf,
                    "tier": r["tier"],
                    "W": W,
                    "H": r["H"],
                    "R_real": r["R_real"],
                    "gap_lu": r["gap_lu"],
                    "a_body": r["a_body"],
                    "fp32_margin": r["fp32_margin"],
                    "steps_run": r["steps_run"],
                    "steady_step": r["steady_step"],
                    "steady_detected": r["steady_detected"],
                    "vf_actual": geo["vf_actual"],
                    "K_sim": der["K_sim_darcy"],
                    "K_ref_primary": der["K_ref_primary"],
                    "ref_primary": r["ref_primary"],
                    "err_pct": der["err_pct"],
                    "signed_err_pct": der["signed_err_pct"],
                    "gebart_signed_pct": der["signed_gebart_pct"],
                    "err_gebart_pct": der["err_gebart_pct"],
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
            ladders[f"{arr}_vf{vf:g}"] = entries

    verdicts: dict[str, dict] = {}
    per_array: dict[str, dict] = {"sq": {}, "hex": {}}
    for key, entries in ladders.items():
        arr = entries[0]["array"]
        errs = [e["err_pct"] for e in entries]
        mono = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
        finest = errs[-1] if errs else float("nan")
        all_steady = all(e["steady_detected"] for e in entries)
        ok = mono and finest <= 3.0 and all_steady
        reasons = []
        if not mono:
            reasons.append("monotonicity violated")
        if finest > 3.0:
            reasons.append(f"finest err {finest:.4f}% > 3%")
        if not all_steady:
            reasons.append("steady criterion not met for at least one grid")
        v = {
            "errors_pct": errs,
            "monotone_decreasing": mono,
            "finest_err_pct": finest,
            "finest_W": entries[-1]["W"] if entries else None,
            "all_steady": all_steady,
            "pass": ok,
            "reasons": reasons,
            "tier": entries[0]["tier"],
        }
        verdicts[key] = v
        per_array[arr][key] = v

    overall = {}
    for arr in ("sq", "hex"):
        main = {k: v for k, v in per_array[arr].items() if v["tier"] == "main"}
        n_main = len(MAIN_TIERS[arr])
        ok = all(v["pass"] for v in main.values()) and len(main) == n_main
        overall[arr] = {
            "main_tiers": {k: v["pass"] for k, v in main.items()},
            "n_main_expected": n_main,
            "overall": "PASS" if ok else "FAIL",
            "statement": (
                "sq: PASS iff all 4 main tiers (0.30/0.40/0.50/0.60 vs "
                "Sangani exact) pass; hex: PASS iff both near-packing "
                "tiers (0.70/0.75 vs Gebart) pass; disclosure tiers "
                "reported separately, non-gating"
                if arr == "sq"
                else "hex main gate = Gebart at near-packing (exact hex table "
                "unobtainable); dilute hex tiers are reference-limited "
                "disclosure only"
            ),
        }

    for key, v in verdicts.items():
        v["finest_signed_err_pct"] = ladders[key][-1]["signed_err_pct"]

    result["ladders"] = ladders
    result["verdict"] = {
        "overall": {arr: overall[arr]["overall"] for arr in ("sq", "hex")},
        "benchmark_overall": (
            "PASS" if all(overall[a]["overall"] == "PASS" for a in ("sq", "hex")) else "FAIL"
        ),
        "per_array": overall,
        "per_tier": verdicts,
        "statement": (
            "Per tier: err vs the tier's PRIMARY reference (sq: Sangani "
            "exact; hex: Gebart) strictly decreasing with refinement AND "
            "finest <= 3% AND steady on every grid. Main tiers gate the "
            "per-array verdict (sq: 0.30/0.40/0.50/0.60; hex: 0.70/0.75); "
            "disclosure tiers non-gating. Nominal geometry only (no "
            "effective-radius recalibration in the criterion; diagnostic "
            "channel B and per-case Gebart consistency channel disclosed)."
        ),
    }

    # S1: force linearity - fs=20 case must reproduce K_sim (tol 0.1%)
    s1: dict[str, dict] = {}
    for p in sorted(out_dir.glob("case_*_fs*.json")):
        rec = json.loads(p.read_text())
        if rec.get("smoke"):
            continue
        r = rec["run"]
        base = out_dir / f"case_{r['array']}_vf{r['vf']:g}_w{r['W']}.json"
        if not base.exists():
            continue
        k0 = json.loads(base.read_text())["derived"]["K_sim_darcy"]
        k1 = rec["derived"]["K_sim_darcy"]
        s1[p.name] = {
            "K_base": k0,
            "K_scaled": k1,
            "delta_pct": abs(k1 / k0 - 1.0) * 100.0,
            "tol_pct": 0.1,
            "pass": abs(k1 / k0 - 1.0) * 100.0 < 0.1,
        }
    result["sensitivity"]["S1_force_linearity"] = s1

    # S2: measurement window (last 2000 vs last 8000 steps, tol 0.05%)
    s2: dict[str, dict] = {}
    for key, entries in ladders.items():
        for e in entries[-2:]:
            w2, w8 = e["uxf_mean_win2000"], e["uxf_mean_win8000"]
            if w2 is None or w8 is None:
                continue
            s2[f"{key}_w{e['W']}"] = {
                "uxf_win2000": w2,
                "uxf_win8000": w8,
                "delta_pct": abs(w2 / w8 - 1.0) * 100.0,
                "tol_pct": 0.05,
                "pass": abs(w2 / w8 - 1.0) * 100.0 < 0.05,
            }
    result["sensitivity"]["S2_window"] = s2

    res_path = out_dir / "result.json"
    res_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[report] wrote {res_path}")
    for key, v in verdicts.items():
        print(
            f"  {key}: errs={[round(x, 4) for x in v['errors_pct']]} "
            f"mono={v['monotone_decreasing']} finest={v['finest_err_pct']:.4f}% "
            f"pass={v['pass']} [{v['tier']}]"
        )
    for arr in ("sq", "hex"):
        print(f"  OVERALL {arr}: {result['verdict']['overall'][arr]}")
    write_readme(out_dir, result)


def write_readme(out_dir: Path, result: dict) -> None:
    lines: list[str] = []
    a = lines.append
    a("# W9-C: Gebart (1992) transverse permeability - periodic square/hex cylinder arrays")
    a("")
    a(f"Created (UTC): {result['created_utc']}")
    a("")
    a("## Setup")
    a("")
    a("- D2Q9 BGK, tau = 1 (nu = 1/6), fully periodic unit cell.")
    a("- sq: W x W, one centred cylinder (W4-A inheritance). hex: W x H supercell,")
    a("  H = round(sqrt(3) W), cylinders at folded corners + centre (triangular")
    a("  lattice, nn distance W).")
    a("- Body force on fluid nodes only, force scale 10; step order collide ->")
    a("  stream -> masked force -> bounce-back; kernels all from tensorlbm.")
    a("- Darcy convention: K_sim = (1-vf_actual)*nu*<u_x>_fluid/a_body.")
    a("- PRIMARY reference per tier: square -> Sangani-Acrivos 1982 exact")
    a("  f-table (k_ref = pi/(vf*f)*R^2; W4-A double transcription);")
    a("  hex -> Gebart closed form (exact hex table unobtainable, 8-channel")
    a("  hunt in prereg). Gebart is additionally reported for EVERY tier")
    a("  as a consistency channel; diagnostic channel B re-references to")
    a("  vf_actual (disclosed only).")
    a("- Locked Gebart constants: sq C1=16/(9 pi sqrt2), hex 16/(9 pi sqrt6);")
    a("  vf_max = pi/4 / pi/(2 sqrt3). Lock channels in result.json.")
    a("")
    a("## Reference-deviation disclosure (frozen in prereg)")
    a("")
    bias = result["reference_deviation_disclosure"]["sq_gebart_vs_sangani_exact_pct"]
    a("Gebart is a near-packing asymptote. Square deviations vs Sangani exact")
    a("numerics: " + ", ".join(f"Vf={k}: +{v}%" for k, v in bias.items()) + ".")
    a("Square tiers are EXACT-gated so these deviations are not in the gate;")
    a("the per-case gebart channel must reproduce them (physics check on the")
    a("solver + the Gebart formula simultaneously). Hex: no exact table")
    a("obtainable - main gate is Gebart at near-packing only (0.70/0.75),")
    a("dilute hex tiers are reference-limited disclosure.")
    a("")
    a("## Verdict")
    a("")
    v = result["verdict"]
    a(f"Criterion - {v['statement']}")
    a("")
    a("| tier | ladder err (%) | mono | finest (%) | steady | pass | tier class |")
    a("|------|----------------|------|------------|--------|------|------------|")
    for key, vv in v["per_tier"].items():
        errs = ", ".join(f"{x:.4f}" for x in vv["errors_pct"])
        a(
            f"| {key} | {errs} | {vv['monotone_decreasing']} | "
            f"{vv['finest_err_pct']:.4f} | {vv['all_steady']} | {vv['pass']} | {vv['tier']} |"
        )
    a("")
    for arr in ("sq", "hex"):
        a(f"**{arr} overall: {v['overall'][arr]}**")
    a("")
    a("## Ladder detail")
    a("")
    for key, entries in result["ladders"].items():
        a(f"### {key}")
        a("")
        a(
            "| W | H | R | gap_lu | vf_act | K_sim | K_ref(primary) | err (%) | signed (%) | gebart sgn (%) | err_B (%) | steps | steady | Re | Ma | uy_ratio | mass drift |"
        )
        a(
            "|---|---|---|--------|--------|-------|-------|---------|------------|----------------|-----------|-------|--------|-----|-----|----------|------------|"
        )
        for e in entries:
            a(
                f"| {e['W']} | {e['H']} | {e['R_real']:.4f} | {e['gap_lu']:.3f} | "
                f"{e['vf_actual']:.6f} | {e['K_sim']:.6g} | {e['K_ref_primary']:.6g} | "
                f"{e['err_pct']:.4f} | {e['signed_err_pct']:+.4f} | {e['gebart_signed_pct']:+.4f} | "
                f"{e['err_B_pct']:.4f} | "
                f"{e['steps_run']} | {e['steady_detected']} | {e['Re_cell_achieved']:.4f} | "
                f"{e['Ma']:.2e} | {e['uy_ratio_max']:.3f} | {e['mass_drift_max_rel']:.2e} |"
            )
        a("")
    a("## Sensitivity")
    a("")
    a("### S1 force linearity (fs 10 -> 20, K_sim invariant, tol 0.1%)")
    s1 = result["sensitivity"]["S1_force_linearity"]
    if s1:
        a("| case | K_base | K_scaled | delta (%) | pass |")
        a("|------|--------|----------|-----------|------|")
        for name, s in s1.items():
            a(
                f"| {name} | {s['K_base']:.6g} | {s['K_scaled']:.6g} | {s['delta_pct']:.5f} | {s['pass']} |"
            )
    else:
        a("(not run)")
    a("")
    a("### S2 measurement window (last 2000 vs 8000 steps, tol 0.05%)")
    s2 = result["sensitivity"]["S2_window"]
    if s2:
        a("| case | uxf(2000) | uxf(8000) | delta (%) | pass |")
        a("|------|-----------|-----------|-----------|------|")
        for name, s in s2.items():
            a(
                f"| {name} | {s['uxf_win2000']:.8e} | {s['uxf_win8000']:.8e} | {s['delta_pct']:.5f} | {s['pass']} |"
            )
    else:
        a("(not run)")
    a("")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {out_dir / 'README.md'}")


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "W9-C: Gebart transverse permeability benchmark, square + hexagonal "
            "periodic cylinder arrays. 'case' runs one (array, vf, W) grid, "
            "'report' aggregates."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_case = sub.add_parser("case", help="run one case and write its case JSON")
    p_case.add_argument("--array", choices=("sq", "hex"), required=True)
    p_case.add_argument("--vf", type=float, required=True, help="solid volume fraction")
    p_case.add_argument("--w", type=int, required=True, help="cell width in lattice units")
    p_case.add_argument("--out-dir", type=Path, default=Path("out"))
    p_case.add_argument("--device", default="cuda:0")
    p_case.add_argument("--force-scale", type=float, default=FS_DEFAULT)
    p_case.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="override max steps (0 = max(60000, min(40*W*H, 1.6e6)))",
    )
    p_case.add_argument("--smoke", action="store_true")
    add_compile_mode_arg(p_case)

    p_rep = sub.add_parser("report", help="aggregate case JSONs")
    p_rep.add_argument("--out-dir", type=Path, default=Path("out"))
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
