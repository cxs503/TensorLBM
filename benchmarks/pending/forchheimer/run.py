#!/usr/bin/env python3
"""W6-B benchmark: finite-Reynolds drag on a periodic square array of
circular cylinders (Forchheimer / weak-inertial regime) against the locked
external reference (NOTES.md section 2):

    Koch & Ladd 1997 JFM 349:31-66 as transcribed (vector-extracted) in
    Fig. 6 (top) of Forslund et al. 2023, Transp. Porous Med. 148:545-569,
    DOI 10.1007/s11242-023-01966-w  ->  forchheimer_ref_fig6.json

Main criterion (pre-registered):  R_sim(Re) = (a/u)|_Re / (a/u)|_anchor
compared to series A R(Re) = y/y0 (self-normalised), PCHIP in log(Re).
The Stokes anchor runs at Re_darcy ~ 0.6 (NOTES amendment 9.1: at tau=0.55
the fp32 injection floor empties the Stokes window for d >= 104, so the
campaign runs float64 throughout and the anchor sits at Re ~ 0.6 with the
reference-side quadratic correction R_ref(anchor) = 1 + c Re^2 applied,
c = 6.56e-4 from series B's (0.42, 6.34) points; the correction is 0.023%).
Same-(d,tau) anchor cancels tau-dependent wall-location bias to first
order; any affine y-axis normalisation cancels identically.

All physics kernels come from the tensorlbm library (read-only worktree
/nfs/wangxi/worktrees/bm_w5 @ cf5709db3c):
    collide/stream      -> tensorlbm.solver            (D2Q9, periodic stream)
    bounce-back         -> tensorlbm.boundaries
    body force          -> tensorlbm.turbulent_channel._apply_body_force_2d
    equilibrium/moments -> tensorlbm.d2q9
    step compilation    -> tensorlbm.compile_utils.compile_step
Only the geometry (boolean cylinder-array mask, periodic folded distance,
rule identical to verified/permeability) is built here.

Step order follows the library's own turbulent_channel loop (and the
verified/permeability benchmark): collide -> stream -> (fluid-masked)
body force -> bounce-back.

Subcommands
    stokes   one Stokes-anchor case (Re_d ~= 0.6)      -> case_anchor_*.json
    finite   one finite-Re case (secant on a to land Re_ach) -> case_re*_*.json
    stage0   geometry adjudication + fp32 floor precheck -> stage0.json
    report   aggregate cases + reference -> result.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WORKTREE = "/nfs/wangxi/worktrees/bm_w5"
sys.path.insert(0, f"{WORKTREE}/src")

import tensorlbm  # noqa: E402

assert tensorlbm.__file__.startswith(WORKTREE), tensorlbm.__file__

import torch  # noqa: E402

from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.compile_utils import compile_step  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402
from tensorlbm.turbulent_channel import _apply_body_force_2d  # noqa: E402

ART = Path("/nfs/wangxi/runs/bm_widen_w6_20260921/forchheimer")

# ---------------------------------------------------------------- constants
PHI_DESIGN = 0.40  # NOTES 2.2: locked verification-case solid fraction
F_SA_04 = 217.89  # Sangani-Acrivos 1982 pressure-form f(0.40)
RE_ANCHOR = 0.6  # Stokes-branch anchor Reynolds (NOTES 9.1)
C_INERTIAL = 6.56e-4  # R ~ 1 + c Re^2 from series B (0.42 -> 6.34)
SAMPLE_EVERY = 200
DRIFT_SPAN = 10  # samples (= 2000 steps)
DRIFT_TOL = 1e-5
STEADY_REPEATS = 3
POST_STEADY = 8000
MEAS_SAMPLES = 20  # measurement window = last 20 samples
OSC_SAMPLES = 40  # unsteadiness window = last 40 samples

TIERS = {
    "A": {"re": 25.20, "tau": 0.55, "ladder": [52, 104, 208]},
    "B": {"re": 46.88, "tau": 0.55, "ladder": [104, 156, 208]},
    "C": {"re": 103.64, "tau": 0.55, "ladder": [208, 312, 416]},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def library_provenance() -> dict:
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        dev_info = {
            "cuda_device": f"cuda:{idx}",
            "gpu_name": prop.name,
            "gpu_uuid": str(getattr(prop, "uuid", "n/a")),
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
    else:
        dev_info = {"cuda_device": "cpu"}
    return {
        "tensorlbm_file": str(Path(tensorlbm.__file__).resolve()),
        "tensorlbm_commit": "cf5709db3c2bba273c64807e721071de110627d1",
        "torch_version": torch.__version__,
        **dev_info,
    }


def cylinder_array_mask(d: int, phi: float, device) -> torch.Tensor:
    """Boolean solid mask, one periodic square-array unit cell d x d.

    Geometry only (not a physics kernel): circle of nominal radius
    R = d*sqrt(phi/pi) centred at the cell centre, periodic folded
    distance in both directions, float64 comparisons.  Rule identical to
    the verified/permeability benchmark.
    """
    import numpy as np

    r_real = d * math.sqrt(phi / math.pi)
    idx = np.arange(d, dtype=np.float64)
    dd = np.abs(idx - d / 2.0)
    dd = np.minimum(dd, d - dd)  # periodic fold
    dist2 = dd[None, :] ** 2 + dd[:, None] ** 2  # (ny, nx)
    return torch.as_tensor(dist2 <= r_real * r_real, dtype=torch.bool, device=device)


def run_steady(
    d: int,
    tau: float,
    a_body: float,
    device,
    dtype=torch.float64,
    compile_mode=None,
    max_steps: int | None = None,
    u_init: float = 0.0,
    phi_design: float = PHI_DESIGN,
    label: str = "",
    f_init: torch.Tensor | None = None,
) -> dict:
    """Run one forced periodic-array case to steady state; measure plateau."""
    nu = (tau - 0.5) / 3.0
    solid = cylinder_array_mask(d, phi_design, device)
    fluid = ~solid
    solid_b = solid.unsqueeze(0)
    n_solid = int(solid.sum().item())
    phi_s_mask = n_solid / (d * d)
    r_nom = d * math.sqrt(phi_design / math.pi)
    d_p = 2.0 * r_nom  # lattice cells

    if max_steps is None:
        max_steps = min(max(400_000, int(30 * d * d / max(nu, 1e-3))), 6_000_000)

    rho0 = torch.full((d, d), 1.0, dtype=dtype, device=device)
    u0 = torch.full((d, d), float(u_init), dtype=dtype, device=device)
    f = equilibrium(rho0, u0, torch.zeros_like(u0), device=device)
    del rho0, u0
    if f_init is not None:
        f = f_init.to(dtype=dtype, device=device).clone()
    mass0 = float(f.sum().item())

    def _step(f_t: torch.Tensor) -> torch.Tensor:
        f_t = collide_bgk(f_t, tau)
        f_t = stream(f_t)
        # Body force on fluid nodes only (verified/permeability NOTES 2.2:
        # unmasked injection is silently reversed by bounce-back on solid).
        f_t = torch.where(solid_b, f_t, _apply_body_force_2d(f_t, a_body))
        return bounce_back_cells(f_t, solid)

    step_fn = compile_step(_step, compile_mode)

    hist_uxf: list[float] = []
    hist_uxd: list[float] = []
    hist_umax: list[float] = []
    hist_mass: list[float] = []
    recent_drifts: list[float] = []
    steady_step = None
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        f = step_fn(f)
        if step % SAMPLE_EVERY == 0:
            _, ux, _ = macroscopic(f)
            uxf = float(ux[fluid].mean().item())
            uxd = float(ux.mean().item())  # domain (Darcy) mean; solid u = 0
            umax = float(ux.abs().max().item())
            mass = float(f.sum().item())
            hist_uxf.append(uxf)
            hist_uxd.append(uxd)
            hist_umax.append(umax)
            hist_mass.append(mass)
            if steady_step is None and len(hist_uxf) >= DRIFT_SPAN + 1:
                drift = abs(hist_uxf[-1] - hist_uxf[-1 - DRIFT_SPAN]) / max(
                    abs(hist_uxf[-1]), 1e-30
                )
                recent_drifts.append(drift)
                if len(recent_drifts) >= STEADY_REPEATS and all(
                    v < DRIFT_TOL for v in recent_drifts[-STEADY_REPEATS:]
                ):
                    steady_step = step
        if steady_step is not None and step >= steady_step + POST_STEADY:
            break

    wall = time.perf_counter() - t0
    steps_run = len(hist_uxf) * SAMPLE_EVERY

    def wmean(vals, n):
        return sum(vals[-n:]) / len(vals[-n:]) if len(vals) >= n else None

    uxf_m = wmean(hist_uxf, MEAS_SAMPLES)
    uxd_m = wmean(hist_uxd, MEAS_SAMPLES)
    umax_m = wmean(hist_umax, MEAS_SAMPLES)
    tail = hist_uxf[-OSC_SAMPLES:]
    osc = (max(tail) - min(tail)) / abs(uxf_m) if uxf_m else None
    mass_drift = (hist_mass[-1] - mass0) / mass0

    return {
        "label": label,
        "d": d,
        "tau": tau,
        "nu": nu,
        "a_body": a_body,
        "phi_design": phi_design,
        "phi_s_mask": phi_s_mask,
        "porosity_mask": 1.0 - phi_s_mask,
        "d_p_lu": d_p,
        "n_solid": n_solid,
        "steps_run": steps_run,
        "steady_step": steady_step,
        "max_steps": max_steps,
        "uxf_plateau": uxf_m,
        "uxd_plateau": uxd_m,
        "umax_plateau": umax_m,
        "osc_amplitude": osc,
        "mass_drift": mass_drift,
        "re_darcy": uxd_m * d_p / nu,
        "ma_fluid": uxf_m * math.sqrt(3.0) if uxf_m else None,
        "ma_umax": umax_m * math.sqrt(3.0) if umax_m else None,
        "k_hat": nu * uxf_m / a_body if a_body else None,
        "wall_s": wall,
        "hist_uxf_tail": hist_uxf[-40:],
        "_f": f,  # in-memory only; callers must pop before serialising
    }


def anchor_force(d: int, tau: float, re_anchor: float = RE_ANCHOR) -> float:
    """Body force putting the Stokes branch at Re_darcy ~ re_anchor."""
    nu = (tau - 0.5) / 3.0
    k_guess = d * d / F_SA_04
    d_p = 2.0 * d * math.sqrt(PHI_DESIGN / math.pi)
    phi_p = 0.60  # mask porosity ~ 0.60 at phi_s = 0.40 design
    u_fl = re_anchor * nu / (d_p * phi_p)
    return u_fl * nu / k_guess


# ------------------------------------------------------------------ stokes
def cmd_stokes(args) -> None:
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    a = anchor_force(args.d, args.tau, args.re_anchor)
    if args.force_scale:
        a *= args.force_scale
    # warm start the anchor at its Stokes-predicted pore velocity
    u_pred = a * (args.d * args.d / F_SA_04) / ((args.tau - 0.5) / 3.0)
    res = run_steady(
        args.d,
        args.tau,
        a,
        device,
        dtype,
        compile_mode=args.compile_mode,
        u_init=u_pred,
        label=f"anchor[d{args.d}_tau{args.tau:g}]",
    )
    res.pop("_f", None)
    res["mode"] = "anchor"
    res["re_anchor_requested"] = args.re_anchor
    res["force_scale"] = args.force_scale or 1
    res["dtype"] = str(dtype)
    res["ts"] = utc_now()
    res["provenance"] = library_provenance()
    out = ART / f"case_anchor_d{args.d}_tau{args.tau:g}_{args.tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "hist_uxf_tail"}, indent=2))
    print(f"written {out}")


# ------------------------------------------------------------------ finite
def cmd_finite(args) -> None:
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    nu = (args.tau - 0.5) / 3.0

    st = json.loads((ART / args.anchor_case).read_text())
    assert st["d"] == args.d and abs(st["tau"] - args.tau) < 1e-12
    k_hat_st = st["k_hat"]
    phi_p = st["porosity_mask"]
    d_p = st["d_p_lu"]

    u_fl_target = args.re_target * nu / (d_p * phi_p)
    a = u_fl_target * nu / k_hat_st
    a_prev, re_prev = None, None
    iters = []
    f_warm = None
    for it in range(args.max_iter):
        res = run_steady(
            args.d,
            args.tau,
            a,
            device,
            dtype,
            compile_mode=args.compile_mode,
            u_init=min(u_fl_target, 0.05),
            label=f"re{args.re_target:g}[d{args.d}_it{it}]",
            f_init=f_warm,
        )
        f_warm = res.pop("_f")
        re_ach = res["re_darcy"]
        iters.append(
            {
                "iter": it,
                "a": a,
                "re_ach": re_ach,
                "uxf": res["uxf_plateau"],
                "osc": res["osc_amplitude"],
                "steady_step": res["steady_step"],
                "steps": res["steps_run"],
                "wall_s": res["wall_s"],
            }
        )
        print(
            f"[iter {it}] a={a:.6e} -> Re_ach={re_ach:.4f} "
            f"(target {args.re_target}) osc={res['osc_amplitude']}",
            flush=True,
        )
        if abs(re_ach / args.re_target - 1.0) < args.re_tol:
            break
        if a_prev is not None and re_prev is not None and re_ach != re_prev:
            a = a + (args.re_target - re_ach) * (a - a_prev) / (re_ach - re_prev)
        else:
            a = a * (args.re_target / re_ach) ** 1.5
        a_prev, re_prev = iters[-1]["a"], re_ach

    res["mode"] = "finite"
    res["re_target"] = args.re_target
    res["anchor_case"] = args.anchor_case
    res["k_hat_anchor"] = k_hat_st
    res["re_anchor"] = st["re_darcy"]
    res["secant_iters"] = iters
    res["converged_re"] = abs(res["re_darcy"] / args.re_target - 1.0) < args.re_tol
    # R_sim with the anchor-inertiality correction (NOTES 9.1)
    r_anchor = 1.0 + C_INERTIAL * st["re_darcy"] ** 2
    res["R_sim"] = (
        (res["a_body"] / res["uxf_plateau"]) / (st["a_body"] / st["uxf_plateau"]) * r_anchor
    )
    res["R_anchor_correction"] = r_anchor
    res["dtype"] = str(dtype)
    res["ts"] = utc_now()
    res["provenance"] = library_provenance()
    out = ART / f"case_re{args.re_target:g}_d{args.d}_tau{args.tau:g}_{args.tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(
        json.dumps(
            {k: v for k, v in res.items() if k not in ("hist_uxf_tail", "secant_iters")}, indent=2
        )
    )
    print(f"written {out}")


# ------------------------------------------------------------------ stage0
def cmd_stage0(args) -> None:
    device = torch.device(args.device)
    out = {"ts": utc_now(), "provenance": library_provenance()}

    # 0a geometry adjudication: y0 (f-form) = d^2 / k_hat must separate
    # phi_design 0.40 (-> ~218, matching the locked y-intercept 217.4 and
    # S&A f(0.40)=217.89) from 0.4235 (-> ~274).
    print("== stage 0a: geometry adjudication (Stokes, d=128, tau=1.0, fp64) ==")
    adjud = {}
    for phi_des, tag in ((0.40, "phi040"), (0.4235, "phi04235")):
        nu = (1.0 - 0.5) / 3.0
        k_guess = 128 * 128 / F_SA_04
        a = 1e-3 * nu / k_guess  # u ~ 1e-3 -> Re_d ~ 0.33
        r = run_steady(
            128,
            1.0,
            a,
            device,
            torch.float64,
            phi_design=phi_des,
            compile_mode=args.compile_mode,
            label=f"adj[{tag}]",
        )
        r.pop("_f", None)
        y0 = 128 * 128 / r["k_hat"]
        adjud[tag] = {
            "phi_design": phi_des,
            "phi_s_mask": r["phi_s_mask"],
            "k_hat": r["k_hat"],
            "y0_f_form": y0,
            "u_fluid": r["uxf_plateau"],
            "re_darcy": r["re_darcy"],
        }
        print(f"  {tag}: phi_s_mask={r['phi_s_mask']:.6f} k_hat={r['k_hat']:.4f} y0={y0:.2f}")
    out["geometry_adjudication"] = adjud

    # 0b fp32 injection-floor precheck.  Scope (NOTES 9.2): d = 104 only,
    # empirical anchor point; for d >= 104 the anchor force scales as
    # a ~ nu^2 * Re / (k * D_p) ~ d^-3, so fp32 contamination can only
    # worsen with d (quantified analytically in NOTES 9.1/9.2).
    print("== stage 0b: fp32 injection-floor precheck (tau=0.55, d=104) ==")
    fp = {}
    for d in (104,):
        row = {}
        r64 = run_steady(
            d,
            0.55,
            anchor_force(d, 0.55),
            device,
            torch.float64,
            compile_mode=args.compile_mode,
            label=f"fp64[d{d}]",
        )
        r64.pop("_f", None)
        row["k_hat_fp64"] = r64["k_hat"]
        row["a"] = anchor_force(d, 0.55)
        for fs in (1, 10):
            r32 = run_steady(
                d,
                0.55,
                anchor_force(d, 0.55) * fs,
                device,
                torch.float32,
                compile_mode=args.compile_mode,
                label=f"fp32[d{d}_fs{fs}]",
            )
            r32.pop("_f", None)
            row[f"k_hat_fp32_fs{fs}"] = r32["k_hat"]
            row[f"re_fp32_fs{fs}"] = r32["re_darcy"]
        row["rel_err_fp32_fs1"] = row["k_hat_fp32_fs1"] / row["k_hat_fp64"] - 1.0
        row["rel_err_fp32_fs10"] = row["k_hat_fp32_fs10"] / row["k_hat_fp64"] - 1.0
        row["fs_spread"] = row["k_hat_fp32_fs10"] / row["k_hat_fp32_fs1"] - 1.0
        # analytic anchor forces at the ladder tops, for the record
        row["anchor_forces"] = {
            str(dd): anchor_force(dd, 0.55) for dd in (52, 104, 156, 208, 312, 416)
        }
        fp[f"d{d}"] = row
        print(
            f"  d={d}: a={row['a']:.3e} fp64={row['k_hat_fp64']:.6f} "
            f"fp32fs1={row['k_hat_fp32_fs1']:.6f} fp32fs10={row['k_hat_fp32_fs10']:.6f} "
            f"fs_spread={row['fs_spread']:+.2e}"
        )
    out["fp32_precheck"] = fp

    (ART / "stage0.json").write_text(json.dumps(out, indent=2))
    print(f"written {ART / 'stage0.json'}")


# ------------------------------------------------------------------ report
def pchip_interp(tab_x, tab_y):
    """Monotone PCHIP (Fritsch-Carlson slopes) with LOG abscissae: slopes
    and interval lengths are both computed in ln(x), so the Hermite terms
    are dimensionally consistent.  (The original draft mixed linear-x
    slopes with log-x intervals; caught and fixed at verification.  The
    A criterion was insensitive because tier Re values sit on the nodes,
    but the err_B disclosure column was distorted.)"""
    lxs = [math.log(float(x)) for x in tab_x]
    ys = [float(y) for y in tab_y]
    n = len(lxs)
    h = [lxs[i + 1] - lxs[i] for i in range(n - 1)]
    delta = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    m[0], m[-1] = delta[0], delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    def interp(x):
        lx = math.log(x)
        if lx <= lxs[0]:
            i = 0
        elif lx >= lxs[-1]:
            i = n - 2
        else:
            i = max(j for j in range(n - 1) if lxs[j] <= lx)
        hh = lxs[i + 1] - lxs[i]
        t = (lx - lxs[i]) / hh
        h00 = (1 + 2 * t) * (1 - t) ** 2
        h10 = t * (1 - t) ** 2
        h01 = t * t * (3 - 2 * t)
        h11 = t * t * (t - 1)
        return h00 * ys[i] + h10 * hh * m[i] + h01 * ys[i + 1] + h11 * hh * m[i + 1]

    return interp


def sa_f_interp(phi_s: float) -> float:
    """Sangani-Acrivos f(phi) log-log interpolation (verified table)."""
    table = [
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
    xs = [math.log(p) for p, _ in table]
    ys = [math.log(f) for _, f in table]
    x = math.log(phi_s)
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return math.exp(ys[i] + t * (ys[i + 1] - ys[i]))
    raise ValueError(f"phi_s={phi_s} outside S&A table")


def cmd_report(args) -> None:
    ref = json.loads((ART / "forchheimer_ref_fig6.json").read_text())
    pA = [p for p in ref["A_koch_ladd_1997"]["points"] if p["re_darcy"] > 1.0]
    pB = [p for p in ref["B_fdm_2023"]["points"] if p["re_darcy"] > 1.0]
    fA = pchip_interp([p["re_darcy"] for p in pA], [p["R"] for p in pA])
    fB = pchip_interp([p["re_darcy"] for p in pB], [p["R"] for p in pB])

    rows = []
    for tier, cfg in TIERS.items():
        for d in cfg["ladder"]:
            st_f = sorted(ART.glob(f"case_anchor_d{d}_tau{cfg['tau']:g}_main.json"))
            re_f = sorted(ART.glob(f"case_re{cfg['re']:g}_d{d}_tau{cfg['tau']:g}_main.json"))
            if not st_f or not re_f:
                print(f"skip tier {tier} d={d}: missing case files")
                continue
            st = json.loads(st_f[0].read_text())
            rr = json.loads(re_f[0].read_text())
            R_sim = rr["R_sim"]
            errA = abs(R_sim / fA(rr["re_darcy"]) - 1.0)
            errB = abs(R_sim / fB(rr["re_darcy"]) - 1.0)
            phi_s = st["phi_s_mask"]
            # f-form: f = d^2 / ((1 - phi_s) * k_hat)  (the (1-phi_s)
            # factor was missing in the first draft of this column and in
            # stage0a's y0_f_form; both caught at verification, NOTES 10.1)
            y0_sim = st["d"] ** 2 / ((1.0 - phi_s) * st["k_hat"])
            errSA = abs(y0_sim / sa_f_interp(phi_s) - 1.0)
            rows.append(
                {
                    "tier": tier,
                    "re_target": cfg["re"],
                    "d": d,
                    "re_ach": rr["re_darcy"],
                    "R_sim": R_sim,
                    "R_ref_A": fA(rr["re_darcy"]),
                    "err_A": errA,
                    "err_B": errB,
                    "err_SA_stokes": errSA,
                    "y0_f_form": y0_sim,
                    "ma_umax": rr["ma_umax"],
                    "ma_fluid": rr["ma_fluid"],
                    "osc": rr["osc_amplitude"],
                    "mass_drift": rr["mass_drift"],
                    "steady": rr["steady_step"] is not None,
                    "converged_re": rr["converged_re"],
                    "wall_s": rr["wall_s"] + st["wall_s"],
                    "stokes_file": st_f[0].name,
                    "re_file": re_f[0].name,
                }
            )

    verdicts = {}
    for tier, cfg in TIERS.items():
        tr = [r for r in rows if r["tier"] == tier]
        if len(tr) < len(cfg["ladder"]):
            verdicts[tier] = {"tier_verdict": "INCOMPLETE"}
            continue
        tr.sort(key=lambda r: r["d"])
        errs = [r["err_A"] for r in tr]
        finest_ok = errs[-1] <= 0.03
        monotone = all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))
        verdicts[tier] = {
            "errs_by_d": {str(r["d"]): r["err_A"] for r in tr},
            "finest_err": errs[-1],
            "finest_le_3pct": finest_ok,
            "strictly_monotone": monotone,
            "tier_verdict": "PASS" if (finest_ok and monotone) else "FAIL",
        }

    result = {
        "ts": utc_now(),
        "benchmark": "w6b_forchheimer_square_array",
        "reference": {
            "locked": "Koch & Ladd 1997 as transcribed in Forslund et al. 2023 "
            "Fig. 6 (top), vector-extracted; forchheimer_ref_fig6.json",
            "grade": "figure-grade (downgrade disclosed, NOTES 2.3)",
            "series_A_cross_source_spread": "1.3-1.4% mid-range vs FDM series",
        },
        "rows": rows,
        "tier_verdicts": verdicts,
        "overall": "PASS"
        if all(v.get("tier_verdict") == "PASS" for v in verdicts.values())
        else ("FAIL" if verdicts else "INCOMPLETE"),
        "provenance": library_provenance(),
    }
    (ART / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stokes", "finite", "stage0", "report"])
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--compile-mode", default=None)
    ap.add_argument("--d", type=int)
    ap.add_argument("--tau", type=float, default=0.55)
    ap.add_argument("--re-anchor", type=float, default=RE_ANCHOR)
    ap.add_argument("--force-scale", type=float, default=None)
    ap.add_argument("--re-target", type=float)
    ap.add_argument("--anchor-case", default=None)
    ap.add_argument("--re-tol", type=float, default=0.002)
    ap.add_argument("--max-iter", type=int, default=6)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    torch.manual_seed(0)
    if args.cmd == "stokes":
        cmd_stokes(args)
    elif args.cmd == "finite":
        cmd_finite(args)
    elif args.cmd == "stage0":
        cmd_stage0(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
