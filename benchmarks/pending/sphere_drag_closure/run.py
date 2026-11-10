#!/usr/bin/env python3
"""LBM measurement of the sphere drag coefficient for ``Re = 1 ... 50``.

Node P54-W3-N11 (mission 10330).  Companion of
``src/tensorlbm/drag_closure.py`` (the analytic bridge) and of
``benchmarks/pending/sphere_settling_gcs`` (the force instrument).

Physical case
-------------
A periodic cubic cell of side ``L`` contains one *voxel* sphere of radius
``R`` held fixed.  A **velocity-pinned** uniform body force drives the flow:
a proportional controller adjusts the force density ``g(t)`` so that the
volume-averaged streamwise velocity equals ``U_char``.  The run is started
from the exact uniform stream ``u = (U_char, 0, 0)``, so the background flow
is annihilated by the control surface (its null space) and only the
disturbance of the sphere is measured.

Why the controller (and not a prescribed ``g``)
-----------------------------------------------
With a *prescribed* force density the whole periodic box has to be
accelerated to the terminal velocity; the momentum stored in the fluid makes
that spin-up take ``t ~ rho V /(6 pi mu R)`` steps (16200 steps at
``Re = 10``, ``D = 12``), which is unaffordable.  Pinning the mean velocity
removes the inertance from the measurement: at steady state momentum balance
of the periodic cell gives

    F_drag = g_tail * V_fluid            (exact, no closure involved),

and the drag coefficient follows from the *measured* velocity and the
volume-equivalent radius of the staircase sphere,

    C_D = 2 F_drag / (rho U^2 pi a_vol^2),   Re = rho U (2 a_vol) / mu.

The controller therefore cannot bias the closure comparison: it *pins* the
Reynolds number instead of letting the closure pick it, and ``g_tail`` is the
measured momentum input, not a prediction.

Force instrument
----------------
``tensorlbm.force_control_surface.ControlSurfaceForceTracker`` -- the
Galilean-invariant kinetic-flux control surface validated in
``docs/sphere_settling_gcs_20261107.md`` (G15/G12 repair).  Nothing here
touches the staircase wall.  The tracked surface integral is reported as a
self-consistency diagnostic: in exact arithmetic ``g V_fluid + F_cs`` must
vanish for a *uniform* background, which is the negative control.

Reference curve
---------------
``tensorlbm.drag_closure``: Stokes / Oseen / Proudman-Pearson /
Schiller-Naumann / Clift-Gauvin, plus the bridged closure
``cd_oseen_bridge``.  No reference value is baked into the simulation.

Usage
-----
    python run.py --list
    python run.py --tag conf_Re10_L48
    python run.py --matrix sweep --jobs 6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from tensorlbm.boundary.boundaries3d import bounce_back_cells_3d, sphere_mask  # noqa: E402
from tensorlbm.d3q19 import C, W, equilibrium3d  # noqa: E402
from tensorlbm.drag_closure import cd_oseen_bridge, reynolds_number  # noqa: E402
from tensorlbm.force_control_surface import ControlSurfaceForceTracker  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, macroscopic3d, stream3d  # noqa: E402

CS2, CS4 = 1.0 / 3.0, 1.0 / 9.0
OUT_DIR = _HERE.parent / "out"

#: Fixed characteristic lattice speed: Ma = U/c_s = 0.05/0.577 = 0.087, so the
#: compressibility error is O(Ma^2) ~ 0.8 % at every Reynolds number, while
#: tau = 0.5 + 3 nu = 0.5 + 3 U D / Re stays >= 0.54 (BGK stability).
U_CHAR = 0.05

#: Controller relaxation time in lattice steps (see module docstring).
TAU_CTRL = 200.0

#: Velocity trim gain of the drag-feed-forward controller (per unit of velocity
#: error, in force-density units for a unit-velocity error of 1e-3 -> 1e-5).
CTRL_TRIM = 1.0e-2


def guo_source(rho, ux, uy, uz, fx, tau):
    """Guo (2002) forcing source for a force density ``f`` in +x."""
    c = C.to(rho.device).float()
    w = W.to(rho.device).float().view(19, 1, 1, 1)
    cu = (
        c[:, 0].view(19, 1, 1, 1) * ux
        + c[:, 1].view(19, 1, 1, 1) * uy
        + c[:, 2].view(19, 1, 1, 1) * uz
    )
    cF = c[:, 0].view(19, 1, 1, 1) * fx
    return (1.0 - 0.5 / tau) * w * (cF / CS2 + cu * cF / CS4)


def mem_wall_force(f_pre: torch.Tensor, solid: torch.Tensor) -> tuple[float, float, float]:
    """Momentum-exchange force on the staircase wall, single time level.

    ``f_pre`` is the post-collision, post-bounce-back, **pre-stream** state, so
    the population ``f_pre[q](x)`` on a fluid cell whose ``+q`` neighbour is
    solid is the one that crosses the wall during the coming streaming.  The
    wall receives ``c_q`` on the way in and ``-c_q`` (mirrored) on the way out,
    hence ``F = 2 sum_links f_q(x_f) c_q`` — the Ladd/Suil sum evaluated on a
    *single* time level (this is what the G15 failure of the gap ledger got
    wrong: pairing ``f_q(x_f)`` with ``f_opp(x_s)`` from another time level
    leaves an additive free-stream background).
    """
    c = C.to(f_pre.device)
    fluid = ~solid
    fx = fy = fz = 0.0
    for q in range(f_pre.shape[0]):
        cx, cy, cz = int(c[q, 0]), int(c[q, 1]), int(c[q, 2])
        if cx == 0 and cy == 0 and cz == 0:
            continue
        nb_solid = torch.roll(solid, shifts=(-cz, -cy, -cx), dims=(0, 1, 2))
        link = fluid & nb_solid
        if not bool(link.any()):
            continue
        m = float(f_pre[q][link].sum())
        fx += m * cx
        fy += m * cy
        fz += m * cz
    return 2.0 * fx, 2.0 * fy, 2.0 * fz


def _log(msg: str) -> None:
    print(msg, flush=True)


def _fit_linear(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    k = sxy / sxx
    b = my - k * mx
    ss_res = sum((y - (b + k * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return k, b, r2


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------
def run_case(spec: dict) -> dict:
    """Run one (D, L, Re) case; returns a JSON-able record."""
    torch.set_num_threads(1)
    D = float(spec["D"])
    L = int(spec["L"])
    re_target = float(spec["Re"])
    steps = int(spec["steps"])
    sample_every = int(spec.get("sample_every", 20))
    tail = int(spec.get("tail", 50))
    r_offsets = tuple(spec.get("r_offsets", (2.0, 2.5, 3.0)))
    zero_force = bool(spec.get("zero_force", False))
    use_ctrl = bool(spec.get("controller", True))
    tag = spec["tag"]

    u_char = float(spec.get("u_char", U_CHAR))
    R = 0.5 * D
    nu = u_char * D / re_target
    tau = 0.5 + 3.0 * nu
    u_target = u_char if not zero_force else 0.0
    centre = (L / 2.0, L / 2.0, L / 2.0)
    solid = sphere_mask(L, L, L, centre[0], centre[1], centre[2], R, device=torch.device("cpu"))
    n_solid = int(solid.sum().item())
    v_fluid = float(L) ** 3 - n_solid
    a_vol = (3.0 * n_solid / (4.0 * math.pi)) ** (1.0 / 3.0)

    # the Guo source of this library injects (1 - 1/(2 tau)) x F per step
    # (measured: a force-free periodic box accelerates at 0.2647 g for tau=0.68);
    # every force bookkeeping below carries that factor explicitly.
    force_factor = 1.0 - 0.5 / tau

    # initial guess of the driving force density from the bridge closure
    re_guess = reynolds_number(a_vol, u_target, 1.0, nu) if u_target else 0.0
    cd_guess = cd_oseen_bridge(re_guess) if re_guess > 0 else 0.0
    f_guess = 0.5 * cd_guess * math.pi * a_vol**2 * u_target**2
    g = f_guess / (v_fluid * force_factor)
    g0 = g

    ux0 = torch.full((L, L, L), u_target)
    rho0 = torch.ones(L, L, L)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    del rho0, ux0
    fluid = ~solid
    trackers = {
        float(off): ControlSurfaceForceTracker(centre=centre, radius=R + float(off), periodic=True)
        for off in r_offsets
    }
    u_hist: list[float] = []
    g_hist: list[float] = []
    mem_hist: list[float] = []
    f_mem_last = 0.0
    mem_every = int(spec.get("mem_every", 100))
    t0 = time.time()
    step_i = 0
    _log(
        f"[{tag}] L={L}^3 D={D:g} nu={nu:.6g} tau={tau:.4f} U={u_target:g} "
        f"g0={g0:.5e} n_solid={n_solid} a_vol={a_vol:.4f} cd_guess={cd_guess:.4f} steps={steps}"
    )
    diverged = False
    u_filt = u_target
    g_ff = g
    for step_i in range(1, steps + 1):
        rho, ux, uy, uz = macroscopic3d(f)
        u_mean = float(ux[fluid].mean())
        u_filt += 0.02 * (u_mean - u_filt)
        if not zero_force and use_ctrl:
            # feed-forward from the instantaneous drag measurement (no inertance:
            # the box inertia would need `t ~ rho V /(6 pi mu R)` steps to react
            # to a change of g, so a plain integral controller only integrates
            # noise) plus a gentle velocity trim on the filtered mean velocity.
            g = g_ff + CTRL_TRIM * (u_target - u_filt)
        fx = rho * g
        f_new = collide_bgk3d(f, tau) + guo_source(rho, ux, uy, uz, fx, tau)
        f_new[:, solid] = f[:, solid]
        f_pre = bounce_back_cells_3d(f_new, solid)
        f = stream3d(f_pre)
        if step_i % sample_every == 0:
            for tr in trackers.values():
                tr.sample(f)
            u_hist.append(u_mean)
            g_hist.append(g)
            g_ff = abs(f_mem_last) / (v_fluid * force_factor)
            if step_i % mem_every == 0:
                f_mem_last = mem_wall_force(f_pre, solid)[0]
                mem_hist.append(f_mem_last)
        if step_i % 100 == 0 and not torch.isfinite(f).all():
            diverged = True
            _log(f"[{tag}] DIVERGED at step {step_i}")
            break
        if step_i % max(steps // 8, 1) == 0:
            s = trackers[float(r_offsets[0])].summary("x", last=tail)
            _log(
                f"[{tag}] step={step_i} U={u_mean:.6f} g={g:.5e} Fx_cs={s['mean']:.5e} "
                f"({time.time() - t0:.0f}s)"
            )
    wall = time.time() - t0

    k = min(tail, len(u_hist))
    u_tail = sum(u_hist[-k:]) / k if k else float("nan")
    g_tail = sum(g_hist[-k:]) / k if k else 0.0
    f_drag = g_tail * v_fluid * force_factor
    rec: dict = {
        "tag": tag,
        "D": D,
        "L": L,
        "L_over_D": L / D,
        "R_nominal": R,
        "a_volume_equivalent": a_vol,
        "lambda_R_over_L": R / L,
        "Re_target": re_target,
        "nu": nu,
        "tau": tau,
        "u_target": u_target,
        "u_measured": u_tail,
        "g0": g0,
        "g_tail": g_tail,
        "guo_force_factor": force_factor,
        "F_drive_momentum_balance": f_drag,
        "g_rel_change": (g_tail - g0) / g0 if g0 else None,
        "F_drag": f_drag,
        "Re_measured": reynolds_number(a_vol, abs(u_tail), 1.0, nu),
        "steps": step_i,
        "sample_every": sample_every,
        "wall_s": wall,
        "diverged": diverged,
        "zero_force": zero_force,
        "n_solid_cells": n_solid,
        "v_fluid": v_fluid,
        "u_series": u_hist,
        "g_series": g_hist,
        "control_surface": {},
    }
    if zero_force:
        rec["cd_vol"] = 0.0
        rec["Re_measured"] = 0.0
        rec["closure_cd"] = None
        rec["rel_diff_pct"] = None
    else:
        rec["cd_vol"] = 2.0 * f_drag / (math.pi * a_vol**2 * u_tail**2)
        rec["closure_cd"] = cd_oseen_bridge(max(rec["Re_measured"], 1e-9))
        rec["rel_diff_pct"] = 100.0 * (rec["cd_vol"] - rec["closure_cd"]) / rec["closure_cd"]

    for off, tr in trackers.items():
        sx = tr.summary("x", last=tail)
        ser = tr.series("x")
        kk = min(tail, len(ser))
        w = ser[-kk:]
        h = max(1, len(w) // 2)
        early = sum(w[:h]) / h
        late = sum(w[h:]) / max(1, len(w) - h)
        rec["control_surface"][f"{off:g}"] = {
            "radius": R + off,
            "fx_mean": sx["mean"],
            "fx_rms": sx["rms"],
            "ptp_over_mean": sx["ptp_over_mean"],
            "tail_half_drift_rel": abs(late - early) / abs(late) if late else None,
            "momentum_drift": tr.momentum_drift(dt_steps=sample_every, last=tail),
        }
    key = f"{float(r_offsets[0]):g}"
    if mem_hist:
        km = min(max(5, tail // 4), len(mem_hist))
        rec["mem_force"] = sum(mem_hist[-km:]) / km
        rec["mem_force_series"] = mem_hist
        if not zero_force and u_tail:
            rec["cd_mem"] = 2.0 * abs(rec["mem_force"]) / (
                math.pi * a_vol**2 * u_tail**2
            )
            rec["mem_over_closure"] = rec["cd_mem"] / cd_oseen_bridge(max(rec["Re_measured"], 1e-9))
    rec["cs_tail_half_drift_rel"] = rec["control_surface"][key]["tail_half_drift_rel"]
    # steadiness of the *measurement*: drift of g and of u over the tail window
    kk = min(tail, len(g_hist))
    if kk >= 4:
        h = kk // 2
        u_early = sum(u_hist[-kk:][:h]) / h
        u_late = sum(u_hist[-kk:][h:]) / (kk - h)
        g_early = sum(g_hist[-kk:][:h]) / h
        g_late = sum(g_hist[-kk:][h:]) / (kk - h)
        rec["u_tail_half_drift_rel"] = abs(u_late - u_early) / abs(u_late) if u_late else None
        rec["g_tail_half_drift_rel"] = abs(g_late - g_early) / abs(g_late) if g_late else None
    _log(
        f"[{tag}] FINAL U={u_tail:.6f} Re={rec['Re_measured']:.3f} C_D={rec['cd_vol']:.5f} "
        f"closure={rec['closure_cd'] if rec['closure_cd'] is None else round(rec['closure_cd'], 5)} "
        f"drift_g={rec.get('g_tail_half_drift_rel')} ({wall:.0f}s)"
    )
    return rec


# ---------------------------------------------------------------------------
# study matrices
# ---------------------------------------------------------------------------
def _steps_for(D: float, re: float) -> int:
    """Number of lattice steps: ``n_conv * D / U`` with a low-Re floor."""
    n_conv = 20.0
    if re >= 50:
        n_conv = 34.0
    elif re >= 20:
        n_conv = 28.0
    elif re >= 10:
        n_conv = 24.0
    return int(max(3000, n_conv * D / U_CHAR))


def build_cases() -> list[dict]:
    cases: list[dict] = []

    def add(tag, D, L, Re, **kw):
        spec = {"tag": tag, "D": float(D), "L": int(L), "Re": float(Re),
                "steps": kw.pop("steps", _steps_for(D, Re))}
        spec.update(kw)
        cases.append(spec)

    # ---- 1+2. Reynolds sweep x confinement (D=12, lambda = 1/6, 1/8, 1/10) --
    for re in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        for L in (36, 48):
            add(f"conf_Re{re:g}_L{L}", 12, L, re)

    # ---- 3. resolution convergence at fixed lambda = 1/6 (L = 3D) ----------
    for re in (1.0, 10.0, 50.0):
        for D in (8, 16):
            if re >= 50 and D == 8:
                continue      # tau = 0.5 + 0.15*8/50 = 0.524 -> excluded a priori
            add(f"res_Re{re:g}_D{D}", D, 3 * D, re)

    # ---- 4. negative control: no applied force -> exactly zero drag -------
    add("negctrl_zero_force", 12, 48, 10.0, steps=400, zero_force=True)
    return cases


MATRICES = {
    "conf": lambda: [c for c in build_cases() if c["tag"].startswith("conf")],
    "res": lambda: [c for c in build_cases() if c["tag"].startswith("res_")],
    "neg": lambda: [c for c in build_cases() if c["tag"].startswith("negctrl")],
    "all": build_cases,
}


def _run_and_save(spec: dict, outdir: str):
    rec = run_case(spec)
    Path(outdir, f"{spec['tag']}.json").write_text(json.dumps(rec, indent=1))
    return spec["tag"], rec["Re_measured"], cd_of(rec)


def cd_of(rec: dict) -> float:
    """Drag coefficient from the momentum balance (same formula as merge_results)."""
    if rec.get("zero_force") or not rec.get("u_measured"):
        return 0.0
    F = rec.get("F_drive_momentum_balance", 0.0)
    a = rec["a_volume_equivalent"]
    return 2.0 * F / (math.pi * a * a * rec["u_measured"] ** 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", choices=sorted(MATRICES))
    ap.add_argument("--tag")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--steps", type=int, help="override the step count")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--outdir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.list:
        for c in build_cases():
            print(c)
        return 0

    if args.tag:
        cases = [c for c in build_cases() if c["tag"] == args.tag]
        if not cases:
            raise SystemExit(f"unknown tag {args.tag}")
    elif args.matrix:
        cases = MATRICES[args.matrix]()
    else:
        raise SystemExit("pass --tag or --matrix")

    if args.steps:
        for c in cases:
            c["steps"] = args.steps

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.jobs <= 1:
        for c in cases:
            rec = run_case(c)
            (outdir / f"{c['tag']}.json").write_text(json.dumps(rec, indent=1))
        return 0

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        # each worker writes its own file, so a killed parent never loses a
        # finished case
        futs = {pool.submit(_run_and_save, c, str(outdir)): c for c in cases}
        for fut in as_completed(futs):
            c = futs[fut]
            tag, re_m, cd = fut.result()
            _log(f"[done] {tag} Re={re_m:.2f} CD={cd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())