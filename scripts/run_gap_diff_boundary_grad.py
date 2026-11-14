#!/usr/bin/env python3
"""GAP-DIFFLBM-BOUNDARY-GRAD driver: closure anchors + AD-vs-FD gradient matrix.

Acceptance target (total ledger GAP-DIFFLBM-BOUNDARY-GRAD): three solid-boundary
closures (static half-way bounce-back wall, moving wall / Couette, volume
penalisation solid) times two targets (drag, lift), reverse-mode AD versus
central differences, relative error <= 1e-2.

Two blocks are produced:

``anchors``
    Absolute correctness of the wall closure *and* of the force probe.  A
    gradient check (AD vs FD) can never reveal a wrong force value, because
    both sides would be wrong together, so the probes are pinned to exact
    solutions first:

    * plane Couette flow, momentum-exchange force on the moving wall vs
      ``F = mu U / H * L`` (exact);
    * plane Poiseuille flow, wall force vs ``F = F_body H L / 2`` (exact) plus
      the parabolic profile;
    * DFG 2-D-2 benchmark (Schaefer & Turek 1996): channel of height 4.1 D and
      length 22 D, cylinder D = 0.1 H, parabolic inlet, Re = 100 -> ``C_D`` and
      ``C_L`` versus the published ranges.

``gradient_matrix``
    ``d(drag)/d(theta)`` and ``d(lift)/d(theta)`` for every closure, by AD
    through the full time-stepping operator and by central differences of the
    same observable, at two finite-difference step sizes (and, as sensitivity,
    at a second lattice resolution and a second time horizon).  The natural
    control parameter of each closure is used (inlet incidence angle for the
    static bounce-back wall, wall speed for the Couette wall, body height for
    the penalised solid) plus one parameter shared by all three (inlet speed).

Everything is written to ``artifacts/GAP-DIFFLBM-BOUNDARY-GRAD/``.

Reproduce (full run, ~20 min on 8 cores)::

    python scripts/run_gap_diff_boundary_grad.py --mode full

Fast smoke run (~1 min)::

    python scripts/run_gap_diff_boundary_grad.py --mode quick
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.diff_boundary_grad import DiffChannel2D  # noqa: E402

ART = Path("artifacts/GAP-DIFFLBM-BOUNDARY-GRAD")

CLOSURE_PARAM = {
    "bb_static": ("inlet incidence angle alpha", 0.06, 2.0e-3, 1.0e-3, "rad"),
    "bb_moving": ("bottom wall speed u_w", 0.06, 2.0e-3, 1.0e-3, "lattice"),
    "penalization": ("body centre height y_c", 16.3, 2.0e-2, 1.0e-2, "lattice cell"),
}


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------
def anchor_couette(ny: int = 34, nx: int = 8, uw: float = 0.05, tau: float = 0.8, steps: int = 20000):
    ch = DiffChannel2D(ny, nx, tau=tau, periodic_x=True)
    geom = ch.geometry(-500.0, -500.0, "bb_moving")     # no body, walls only
    f = ch.initial(0.0, 0.0)
    for _ in range(steps):
        f, femit, rho = ch.step(f, geom, u0=0.0, alpha=0.0, closure="bb_moving", uw=(uw, 0.0))
    _rho, ux, _uy = ch.velocity_profile(f)
    y = torch.arange(ny, dtype=torch.float64)
    ana = uw * (ny - 1.5 - y) / (ny - 2.0)             # bottom wall at y = 1/2 moves at +uw
    prof_err = float((ux[1:-1] - ana[1:-1, None]).abs().max() / uw)
    nu = (tau - 0.5) / 3.0
    f_bot, _ = ch.bb_force(femit, rho, geom, uw=(uw, 0.0), links9=geom.bot9.to(torch.float64))
    f_top, _ = ch.bb_force(femit, rho, geom, uw=(uw, 0.0), links9=geom.top9.to(torch.float64))
    f_ana = nu * uw / (ny - 2.0) * nx                   # |mu U/H| * length
    return {
        "name": "plane Couette (moving bottom wall)",
        "reference": "exact: u(y)=U(y-y_wall)/H, wall force |mu U/H L|",
        "params": {"ny": ny, "nx": nx, "uw": uw, "tau": tau, "nu": nu, "steps": steps},
        "profile_max_rel_err": prof_err,
        "force_on_moving_wall": float(f_bot),
        "force_on_fixed_wall": float(f_top),
        "force_analytic_magnitude": f_ana,
        "force_rel_err": abs(abs(float(f_bot)) - f_ana) / f_ana,
        # the fluid resists the moving wall: sign must be opposite to u_w
        "sign_consistent": bool(float(f_bot) < 0 and float(f_top) > 0),
        "profile_got": [float(v) for v in ux[1:-1, 0]],
        "profile_exact": [float(v) for v in ana[1:-1]],
    }


def anchor_poiseuille(ny: int = 34, nx: int = 8, nu: float = 0.1, body_force: float = 2e-5, steps: int = 20000):
    tau = 3.0 * nu + 0.5
    ch = DiffChannel2D(ny, nx, tau=tau, periodic_x=True)
    geom = ch.geometry(-500.0, -500.0, "bb_static")
    f = ch.initial(0.0, 0.0)
    for _ in range(steps):
        f, femit, rho = ch.step(f, geom, u0=0.0, alpha=0.0, closure="bb_static", body_force=body_force)
    _rho, ux, _uy = ch.velocity_profile(f)
    y = torch.arange(ny, dtype=torch.float64)
    h = ny - 2.0
    ana = (body_force / (2.0 * nu)) * (y - 0.5) * (h - (y - 0.5))
    prof_err = float((ux[1:-1] - ana[1:-1, None]).abs().max() / ana.max())
    f_bot, _ = ch.bb_force(femit, rho, geom, uw=(0.0, 0.0), links9=geom.bot9.to(torch.float64))
    f_top, _ = ch.bb_force(femit, rho, geom, uw=(0.0, 0.0), links9=geom.top9.to(torch.float64))
    f_ana = body_force * h / 2.0 * nx                   # momentum balance per wall
    return {
        "name": "plane Poiseuille (static bounce-back walls)",
        "reference": "exact: u(y)=F/(2 nu) y (H-y), wall force F H L / 2",
        "params": {"ny": ny, "nx": nx, "nu": nu, "tau": tau, "body_force": body_force, "steps": steps},
        "profile_max_rel_err": prof_err,
        "force_on_lower_wall": float(f_bot),
        "force_on_upper_wall": float(f_top),
        "force_analytic_per_wall": f_ana,
        "force_rel_err": abs(float(f_bot) - f_ana) / f_ana,
        "sign_consistent": bool(float(f_bot) > 0 and float(f_top) > 0),
        "profile_got": [float(v) for v in ux[1:-1, 0]],
        "profile_exact": [float(v) for v in ana[1:-1]],
    }


# Schafer & Turek DFG 2-D-1 (Re=20, steady) and 2-D-2 (Re=100, unsteady) references
REF_BY_RE = {
    20.0: {"C_D": [5.57, 5.59], "C_D_mid": 5.58, "C_L": 0.0108,
           "note": "2D-1: C_D 5.57-5.59, C_L 0.0108, steady"},
    100.0: {"C_D": [3.22, 3.24], "C_D_mid": 3.23, "C_L": 1.0,
            "note": "2D-2: C_D 3.22-3.24, C_L amplitude ~1.0"},
}


def anchor_dfg(d_cells: float = 10.0, um: float = 0.1, re: float = 100.0,
               steps: int = 60000, sample_from: int = 40000):
    """Schaefer & Turek DFG 2-D-2: H = 4.1 D, L = 22 D, centre (2 D, H/2)."""
    h_cells = 4.1 * d_cells
    ny = int(round(h_cells)) + 2                # + 2 wall rows
    nx = int(round(22.0 * d_cells))
    nu = um * d_cells / re
    tau = 3.0 * nu + 0.5
    ch = DiffChannel2D(ny, nx, tau=tau, radius=d_cells / 2.0, smoothing=0.8)
    yc = (ny - 2) / 2.0 + 0.5                   # centre of the fluid domain
    xc = 2.0 * d_cells
    geom = ch.geometry(xc, yc, "bb_static")
    yy = ch.yy[:, 0].detach()
    hh = ny - 2.0
    u_max = 1.5 * um
    prof = (4.0 * u_max * (yy - 0.5) * (hh - (yy - 0.5)) / hh**2).clamp(min=0.0).view(ny, 1)
    prof[0] = 0.0
    prof[-1] = 0.0
    f = ch.initial(prof, 0.0)
    cd, cl = [], []
    t0 = time.time()
    for n in range(steps):
        f, femit, rho = ch.step(f, geom, u0=prof, alpha=0.0, closure="bb_static")
        if n >= sample_from and (n - sample_from) % 50 == 0:
            fx, fy = ch.bb_force(femit, rho, geom)
            q = 0.5 * um * um * d_cells
            cd.append(float(fx) / q)
            cl.append(float(fy) / q)
    _rho_all, _ux_all, _uy_all = ch.velocity_profile(f)
    _rho_ux = (_ux_all, _uy_all)
    cd_t = torch.tensor(cd)
    cl_t = torch.tensor(cl)
    cd_mean = float(cd_t.mean())
    cl_amp = float(0.5 * (cl_t.max() - cl_t.min()))
    return {
        "name": "DFG 2-D-2 (Schaefer & Turek) cylinder in a channel, Re=100",
        "reference": ("Schafer & Turek 1996 (NNFM 48:547-566) "
                      + REF_BY_RE[re]["note"]),
        "params": {"D_cells": d_cells, "ny": ny, "nx": nx, "um": um, "re": re, "nu": nu,
                   "tau": tau, "blockage": d_cells / (ny - 2.0), "steps": steps,
                   "sample_from": sample_from, "clock_s": round(time.time() - t0, 1),
                   "rho_min": float(_rho_all.min()), "rho_max": float(_rho_all.max()),
                   "u_max": float(torch.hypot(_ux_all, _uy_all).max())},
        "C_D_mean": cd_mean,
        "C_D_ref": REF_BY_RE[re]["C_D"],
        "C_D_rel_err_to_mid": abs(cd_mean - REF_BY_RE[re]["C_D_mid"]) / REF_BY_RE[re]["C_D_mid"],
        "C_L_amplitude": cl_amp,
        "C_L_mean": float(cl_t.mean()),
        "C_L_ref_amplitude": REF_BY_RE[re]["C_L"],
        "C_L_rel_err": abs(cl_amp - REF_BY_RE[re]["C_L"]) / REF_BY_RE[re]["C_L"],
        "strouhal_note": "C_L amplitude taken as half peak-to-peak of the sampled window",
        "samples": {"C_D": cd, "C_L": cl,
                    "t": [sample_from + 50 * k for k in range(len(cd))]},
        "field_snapshot": {
            "note": "last-step ux/uy, every 2nd cell, rows = wall-normal",
            "ux": [[float(v) for v in row] for row in _rho_ux[0][0::2, 0::2]],
            "uy": [[float(v) for v in row] for row in _rho_ux[1][0::2, 0::2]],
            "body": [[int(v) for v in row] for row in geom.body[0::2, 0::2].to(torch.int64)],
        },
    }


# ---------------------------------------------------------------------------
# gradient matrix
# ---------------------------------------------------------------------------
def _observe(ch, closure, theta, base, steps, ck, differentiable):
    th = torch.tensor(float(theta), dtype=torch.float64, requires_grad=differentiable)
    if closure == "penalization":
        return ch.probe(xc=base["xc"], yc=th, u0=base["u0"], alpha=base["alpha"],
                        closure=closure, steps=steps, checkpoint_segment=ck)
    if closure == "bb_moving":
        return ch.probe(xc=base["xc"], yc=base["yc"], u0=base["u0"], alpha=base["alpha"],
                        closure=closure, uw=(th, 0.0), steps=steps, checkpoint_segment=ck)
    return ch.probe(xc=base["xc"], yc=base["yc"], u0=base["u0"], alpha=th,
                    closure=closure, steps=steps, checkpoint_segment=ck)


def grad_entry(ch, closure, param, theta0, h1, h2, base, steps):
    """AD gradient and central differences of (drag, lift) w.r.t. ``param``."""
    ck = max(50, steps // 12)
    th = torch.tensor(float(theta0), dtype=torch.float64, requires_grad=True)
    t0 = time.time()
    if closure == "penalization":
        drag, lift = ch.probe(xc=base["xc"], yc=th, u0=base["u0"], alpha=base["alpha"],
                              closure=closure, steps=steps, checkpoint_segment=ck)
    elif closure == "bb_moving":
        drag, lift = ch.probe(xc=base["xc"], yc=base["yc"], u0=base["u0"], alpha=base["alpha"],
                              closure=closure, uw=(th, 0.0), steps=steps, checkpoint_segment=ck)
    else:
        drag, lift = ch.probe(xc=base["xc"], yc=base["yc"], u0=base["u0"], alpha=th,
                              closure=closure, steps=steps, checkpoint_segment=ck)
    (g_drag,) = torch.autograd.grad(drag, th, retain_graph=True)
    (g_lift,) = torch.autograd.grad(lift, th, retain_graph=False)
    drag_v, lift_v = float(drag), float(lift)
    t_ad = time.time() - t0

    def fd(h):
        dp, lp = [float(v) for v in _observe(ch, closure, theta0 + h, base, steps, 0, False)]
        dm, lm = [float(v) for v in _observe(ch, closure, theta0 - h, base, steps, 0, False)]
        return (dp - dm) / (2.0 * h), (lp - lm) / (2.0 * h)

    out = {
        "closure": closure,
        "parameter": param,
        "theta0": float(theta0),
        "drag": drag_v,
        "lift": lift_v,
        "ad": {"drag": float(g_drag), "lift": float(g_lift)},
        "fd": {},
        "steps": steps,
        "ad_clock_s": round(t_ad, 1),
    }
    for tag, h in (("h1", h1), ("h2", h2)):
        fd_drag, fd_lift = fd(h)
        out["fd"][tag] = {
            "h": h,
            "drag": fd_drag,
            "lift": fd_lift,
            "rel_err_drag": abs(float(g_drag) - fd_drag) / max(abs(fd_drag), 1e-30),
            "rel_err_lift": abs(float(g_lift) - fd_lift) / max(abs(fd_lift), 1e-30),
        }
    best = out["fd"]["h2"]
    out["rel_err_drag"] = best["rel_err_drag"]
    out["rel_err_lift"] = best["rel_err_lift"]
    out["pass_1e2"] = bool(out["rel_err_drag"] <= 1e-2 and out["rel_err_lift"] <= 1e-2)
    return out


def _as_tensor(v):
    return torch.tensor(float(v), dtype=torch.float64)


def gradient_matrix(*, res_tag: str, ny: int, nx: int, d_cells: float, xc: float, yc: float,
                    u0: float, tau: float, alpha0: float, uw0: float, steps: int,
                    param_shift_yc: float = 0.3):
    ch = DiffChannel2D(ny, nx, tau=tau, radius=d_cells / 2.0)
    base = {"xc": xc, "yc": yc, "u0": u0, "alpha": alpha0}
    rows = []
    for closure in ("bb_static", "bb_moving", "penalization"):
        label, theta0, h1, h2, unit = CLOSURE_PARAM[closure]
        if closure == "bb_moving":
            theta0 = uw0
        elif closure == "penalization":
            theta0 = yc + param_shift_yc
        row = grad_entry(ch, closure, label, theta0, h1, h2, base, steps)
        row.update({"resolution": res_tag, "unit": unit, "ny": ny, "nx": nx,
                    "d_cells": d_cells, "u0": u0, "tau": tau, "alpha0": alpha0,
                    "re": u0 * d_cells / ((tau - 0.5) / 3.0)})
        rows.append(row)
        # parameter shared by all three closures: inlet speed
        base_u0 = dict(base)
        row2 = _grad_entry_u0(ch, closure, base_u0, u0, steps, res_tag,
                              {"ny": ny, "nx": nx, "d_cells": d_cells, "tau": tau, "alpha0": alpha0,
                               "uw0": uw0, "yc_theta": theta0})
        if row2 is not None:
            rows.append(row2)
    return rows


def _grad_entry_u0(ch, closure, base, u0, steps, res_tag, meta):
    """Gradient w.r.t. the inlet speed (shared parameter, all closures)."""
    ck = max(50, steps // 12)
    th = torch.tensor(float(u0), dtype=torch.float64, requires_grad=True)
    kw = dict(xc=base["xc"], yc=base["yc"], u0=th, alpha=base["alpha"], closure=closure,
              steps=steps, checkpoint_segment=ck)
    if closure == "bb_moving":
        drag, lift = ch.probe(**kw, uw=(meta["uw0"], 0.0))
    elif closure == "penalization":
        kw["yc"] = meta["yc_theta"]
        drag, lift = ch.probe(**kw)
    else:
        drag, lift = ch.probe(**kw)
    (g_drag,) = torch.autograd.grad(drag, th, retain_graph=True)
    (g_lift,) = torch.autograd.grad(lift, th, retain_graph=False)

    def run(val):
        kw2 = dict(xc=base["xc"], yc=base["yc"], u0=val, alpha=base["alpha"], closure=closure,
                   steps=steps, checkpoint_segment=0)
        if closure == "bb_moving":
            return [float(v) for v in ch.probe(**kw2, uw=(meta["uw0"], 0.0))]
        if closure == "penalization":
            kw2["yc"] = meta["yc_theta"]
        return [float(v) for v in ch.probe(**kw2)]

    out = {"closure": closure, "parameter": "inlet speed u0", "theta0": float(u0), "unit": "lattice",
           "drag": float(drag), "lift": float(lift),
           "ad": {"drag": float(g_drag), "lift": float(g_lift)}, "fd": {},
           "steps": steps, "resolution": res_tag, "ny": meta["ny"], "nx": meta["nx"],
           "d_cells": meta["d_cells"], "tau": meta["tau"], "alpha0": meta["alpha0"],
           "re": meta["d_cells"] * float(u0) / ((meta["tau"] - 0.5) / 3.0)}
    for tag, h in (("h1", 2.0e-3), ("h2", 1.0e-3)):
        dp, lp = run(float(u0) + h)
        dm, lm = run(float(u0) - h)
        fd_drag, fd_lift = (dp - dm) / (2 * h), (lp - lm) / (2 * h)
        out["fd"][tag] = {"h": h, "drag": fd_drag, "lift": fd_lift,
                          "rel_err_drag": abs(float(g_drag) - fd_drag) / max(abs(fd_drag), 1e-30),
                          "rel_err_lift": abs(float(g_lift) - fd_lift) / max(abs(fd_lift), 1e-30)}
    best = out["fd"]["h2"]
    out["rel_err_drag"] = best["rel_err_drag"]
    out["rel_err_lift"] = best["rel_err_lift"]
    out["pass_1e2"] = bool(out["rel_err_drag"] <= 1e-2 and out["rel_err_lift"] <= 1e-2)
    return out


def closure_sensitivity(ny=40, nx=120, d_cells=10.0, xc=36.0, yc=15.0, u0=0.2, tau=0.65,
                        steps=3000):
    """Penalisation interface width versus drag, against the bounce-back body.

    The diffuse penalisation interface is effectively larger than the staircase
    bounce-back body, so the two closures agree in magnitude only; this scan
    quantifies the monotone dependence on ``smoothing`` instead of hiding it.
    """
    kw = dict(xc=xc, yc=yc, u0=u0, alpha=0.0, steps=steps)
    ref = DiffChannel2D(ny, nx, tau=tau, radius=d_cells / 2.0)
    d_bb, l_bb = ref.probe(closure="bb_static", **kw)
    rows = [{"smoothing": None, "closure": "bb_static", "drag": float(d_bb), "lift": float(l_bb),
             "drag_ratio_to_bb": 1.0}]
    for sm in (0.4, 0.8, 1.2):
        ch = DiffChannel2D(ny, nx, tau=tau, radius=d_cells / 2.0, smoothing=sm)
        d, l = ch.probe(closure="penalization", **kw)
        rows.append({"smoothing": sm, "closure": "penalization", "drag": float(d), "lift": float(l),
                     "drag_ratio_to_bb": float(d) / float(d_bb)})
    return {"params": {"ny": ny, "nx": nx, "d_cells": d_cells, "u0": u0, "tau": tau,
                       "steps": steps, "re": u0 * d_cells / ((tau - 0.5) / 3.0)},
            "rows": rows}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("quick", "full"), default="full")
    ap.add_argument("--part", default="all",
                    choices=("all", "anchors", "grad_main", "grad_coarse", "grad_horizon"),
                    help="run one block only and merge it into the existing JSON "
                         "(lets a long matrix be produced in short, independently "
                         "reproducible chunks)")
    ap.add_argument("--dfg-steps", type=int, default=None)
    ap.add_argument("--dfg-sample-from", type=int, default=None)
    ap.add_argument("--grad-steps", type=int, default=None)
    ap.add_argument("--grad-ny", type=int, default=None)
    ap.add_argument("--grad-nx", type=int, default=None)
    ap.add_argument("--grad-dcells", type=float, default=None)
    ap.add_argument("--grad-tau", type=float, default=None)
    ap.add_argument("--anchor-ny", type=int, default=None)
    ap.add_argument("--anchor-steps", type=int, default=None)
    ap.add_argument("--dfg-dcells", type=float, default=None)
    ap.add_argument("--dfg-re", type=float, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    ART.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out) if args.out else ART
    out_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads": args.threads,
        "dtype": "float64",
        "mode": args.mode,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("env", env, flush=True)

    full_steps = 1200 if args.mode == "full" else 200
    if args.grad_steps:
        full_steps = args.grad_steps
    dfg_steps = args.dfg_steps if args.dfg_steps else (60000 if args.mode == "full" else 1500)
    dfg_from = args.dfg_sample_from if args.dfg_sample_from else max(dfg_steps - 20000, 1000)
    json_path = out_dir / f"diff_boundary_grad_{args.mode}.json"
    payload = json.loads(json_path.read_text()) if json_path.exists() else {}

    if args.mode == "quick":
        blocks = {
            "anchors": lambda: {
                "couette": anchor_couette(steps=4000),
                "poiseuille": anchor_poiseuille(steps=4000),
                "dfg": anchor_dfg(d_cells=10.0, steps=dfg_steps, sample_from=dfg_from),
                "closure_sensitivity": closure_sensitivity(steps=600),
            },
            "grad_main": lambda: gradient_matrix(
                res_tag="main", ny=40, nx=120, d_cells=10.0, xc=36.0, yc=15.0,
                u0=0.2, tau=0.65, alpha0=0.06, uw0=0.06, steps=full_steps),
        }
        keymap = {"anchors": "anchors", "grad_main": "gradient_matrix"}
    else:
        ny = args.grad_ny or 44
        nx = args.grad_nx or 140
        dcell = args.grad_dcells or 10.0
        tau_g = args.grad_tau or 0.65
        nu_g = (tau_g - 0.5) / 3.0
        u0_g = 40.0 * nu_g / dcell                       # Re = 40 by construction
        xc_g = round(0.72 * nx)
        yc_g = round(0.34 * ny)
        ny_c = max(int(round(0.83 * ny)), 20)
        nx_c = max(int(round(0.83 * nx)), 60)
        dc_c = max(0.75 * dcell, 4.0)
        u0_c = 40.0 * nu_g / dc_c                         # same Re on the coarse lattice
        xc_c = round(0.72 * nx_c)
        yc_c = round(0.34 * ny_c)
        anc_ny = args.anchor_ny or 34
        anc_steps = args.anchor_steps or 20000
        ddfg = args.dfg_dcells or 10.0
        blocks = {
            "anchors": lambda: {
                "couette": anchor_couette(ny=anc_ny, steps=anc_steps),
                "poiseuille": anchor_poiseuille(ny=anc_ny, steps=anc_steps),
                "dfg": anchor_dfg(d_cells=ddfg, steps=dfg_steps, sample_from=dfg_from,
                                  re=(args.dfg_re or 100.0), um=(0.1 if (args.dfg_re or 100.0) > 50 else 0.2)),
                "closure_sensitivity": closure_sensitivity(steps=min(full_steps, 3000)),
            },
            "grad_main": lambda: gradient_matrix(
                res_tag="main", ny=ny, nx=nx, d_cells=dcell, xc=float(xc_g), yc=float(yc_g),
                u0=u0_g, tau=tau_g, alpha0=0.06, uw0=0.3 * u0_g, steps=full_steps),
            "grad_coarse": lambda: gradient_matrix(
                res_tag="coarse", ny=ny_c, nx=nx_c, d_cells=dc_c, xc=float(xc_c), yc=float(yc_c),
                u0=u0_c, tau=tau_g, alpha0=0.06, uw0=0.3 * u0_c, steps=full_steps),
            "grad_horizon": lambda: gradient_matrix(
                res_tag=f"main-T{full_steps // 2}", ny=ny, nx=nx, d_cells=dcell, xc=float(xc_g),
                yc=float(yc_g), u0=u0_g, tau=tau_g, alpha0=0.06, uw0=0.3 * u0_g,
                steps=max(full_steps // 2, 60)),
        }
        keymap = {"anchors": "anchors", "grad_main": "gradient_matrix",
                  "grad_coarse": "resolution_scan", "grad_horizon": "horizon_scan"}
        payload.setdefault("lattice", {"main": {"ny": ny, "nx": nx, "d_cells": dcell, "u0": u0_g,
                                                "tau": tau_g, "re": 40.0, "xc": xc_g, "yc": yc_g},
                                       "coarse": {"ny": ny_c, "nx": nx_c, "d_cells": dc_c, "u0": u0_c,
                                                  "tau": tau_g, "re": 40.0, "xc": xc_c, "yc": yc_c}})

    parts = [args.part] if args.part != "all" else list(blocks)
    for part in parts:
        print(f"running block {part} ...", flush=True)
        t0 = time.time()
        payload[keymap[part]] = blocks[part]()
        payload.setdefault("env", env)
        payload["env"]["blocks"] = sorted(set(list(payload["env"].get("blocks", [])) + [part]))
        payload["env"]["block_clocks_s"] = dict(payload["env"].get("block_clocks_s", {}),
                                                **{part: round(time.time() - t0, 1)})
        json_path.write_text(json.dumps(payload, indent=2))
        print(f"block {part} done in {time.time() - t0:.0f}s", flush=True)

    for k, v in (("resolution_scan", []), ("horizon_scan", [])):
        payload.setdefault(k, v)
    every = [r for group in (payload.get("gradient_matrix") or [], payload.get("resolution_scan") or [],
                             payload.get("horizon_scan") or []) for r in group]
    payload["summary"] = {
        "n_pairs": len(every),
        "n_pass": sum(1 for r in every if r["pass_1e2"]),
        "worst_rel_err": (max(max(r["rel_err_drag"], r["rel_err_lift"]) for r in every)
                          if every else None),
        "all_pass": bool(every) and all(r["pass_1e2"] for r in every),
        "by_closure": {
            c: {
                "n": sum(1 for r in every if r["closure"] == c),
                "n_pass": sum(1 for r in every if r["closure"] == c and r["pass_1e2"]),
                "worst_rel_err": max(
                    [max(r["rel_err_drag"], r["rel_err_lift"]) for r in every if r["closure"] == c],
                    default=None,
                ),
            }
            for c in ("bb_static", "bb_moving", "penalization")
        },
    }
    every_missing = []
    json_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print("written", json_path, "missing blocks:", every_missing, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())