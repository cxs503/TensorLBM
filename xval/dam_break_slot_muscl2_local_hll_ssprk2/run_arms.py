"""XVAL arms for anchor XVAL-LBM-DAM-BREAK-SLOT-MUSCL2-LOCAL-HLL-SSPRK2.

Same 1-D shallow-water dam break (Ritter 1892, dry bed) run by

  A            tensorlbm_muscl2_local_hll_ssprk2   in-house: minmod-MUSCL face
               states + Davis local-speed HLL + SSP-RK2(Heun) + dry-bed
               positivity fallback; dry region h = 0
  A_matched    same in-house solver, initial dry region raised to eps = 3e-3 so
               that its IC mass is *identical* to the OSS arm (isolates the
               scheme difference from the OSS dry-floor requirement)
  B            clawpack_classic_hlle               OSS, same order: Clawpack 5.14
               HLLE Riemann + minmod MUSCL + Hancock predictor-corrector
  B_1st        clawpack_classic_hlle_1storder      OSS stability control: same
               flux, no reconstruction (lim=0), eps = 1e-4
  C            clawpack_sharpclaw_weno5            OSS high order: WENO5 + SSP104
  D            clawpack_classic_roe                OSS alternative flux family:
               Roe + entropy fix, minmod MUSCL

Each arm writes ``xval/dam_break_slot_muscl2_local_hll_ssprk2/artifacts/arm_<name>_nx<N>_cfl<C>[_eps<E>].json``.
Every downstream volume is measured as a *difference* from the arm's own initial
downstream volume, so the mass-recovered front law ``x_f = 27 V/(4 h0)`` (the
integral form of the Ritter solution, used by the frozen matrix case) stays
valid for arms that start with a positive dry floor.

Usage::

    python xval/dam_break_slot_muscl2_local_hll_ssprk2/run_arms.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
RUN_DIR = CASE_DIR / "artifacts"   # versioned next to the harness (runs/ is gitignored)

X_MIN, X_MAX = -4.0, 12.0
H0, G, T_END = 1.0, 1.0, 5.0
DRY_EPS_OSS = 3.0e-3        # smallest dry floor at which OSS 2nd order is stable
DRY_EPS_OSS_SENS = 1.0e-3   # documented *unstable* floor (NaN probe)
DRY_EPS_ROE = 1.0e-2        # Roe+efix needs a thicker layer to stay finite
DRY_EPS_1ST = 1.0e-4


# --------------------------------------------------------------------------- A
def run_tensorlbm(nx: int, cfl: float, t_end: float, eps: float = 0.0) -> dict:
    """In-house arm: library solver, only the initial dry floor is configurable."""
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from tensorlbm.shallow_water import (  # noqa: E402
        ShallowWaterConfig, init_dam_break, ritter_depth, ritter_front,
        swe_step,
    )

    cfg = ShallowWaterConfig(
        scheme="muscl2", nx=nx, x_min=X_MIN, x_max=X_MAX, dam_x=0.0,
        h0=H0, g=G, cfl=cfl, t_end=t_end, sample_every=max(1, nx // 8),
    )
    cfg.validate()
    x = np.linspace(X_MIN, X_MAX, nx)
    U = init_dam_break(cfg)
    if eps > 0.0:
        h = U[0].clone()
        h[h < cfg.h_dry] = eps
        U[0] = h
    h_init = np.asarray(U[0].tolist(), dtype=float)
    down = x > cfg.dam_x
    v0 = float(h_init[down].sum() * cfg.dx)

    dt = cfg.cfl * cfg.dx / (2.0 * cfg.c0)
    n_steps = int(round(t_end / dt))
    t0 = time.perf_counter()
    for _ in range(n_steps):
        U = swe_step(U, cfg, dt)
    elapsed = time.perf_counter() - t0

    t_final = n_steps * dt
    h = np.asarray(U[0].tolist(), dtype=float)
    v_down = float(h[down].sum() * cfg.dx) - v0
    x_f = ritter_front(t_final, H0, G)
    mask = (x >= 0.0) & (x <= x_f)
    h_ref = np.array([ritter_depth(float(xi), t_final, H0, G) for xi in x[mask]])
    return {
        "arm": ("tensorlbm_muscl2_local_hll_ssprk2" if eps == 0.0
                else "tensorlbm_muscl2_icmatched"),
        "tool": "TensorLBM", "nx": nx, "cfl": cfl, "t_final": t_final,
        "x": x.tolist(), "h": h.tolist(), "x_f_analytic": x_f,
        "v_down": v_down, "dry_floor": eps,
        "mass_initial": float(h_init.sum() * cfg.dx),
        "mass_final": float(h.sum() * cfg.dx),
        "steps": n_steps, "dt": dt, "elapsed_s": elapsed,
        "grid": "node",
        "scheme": "minmod-MUSCL face states + Davis local-speed HLL + SSP-RK2(Heun); "
                  "dry-bed positivity fallback alpha 1->0.5->0",
        "finite": bool(np.isfinite(h).all()),
    }


# ---------------------------------------------------------------------- B/C/D
def _claw_solve(riemann_name: str, mode: str, nx: int, cfl: float, t_end: float,
                dry_eps: float, limiter: int = 1, weno: int = 5,
                time_integrator: str = "SSP104") -> dict:
    from clawpack import pyclaw, riemann

    mod = getattr(riemann, riemann_name)
    # compiled f2py riemann modules expose only rp1/cparam, so PyClaw's solver
    # metadata (num_eqn/num_waves) has to be declared from the SWE system size.
    mod.num_eqn = 2
    mod.num_waves = 2

    if mode == "sharpclaw":
        solver = pyclaw.SharpClawSolver1D(mod)
        solver.time_integrator = time_integrator
        solver.weno_order = weno
        solver.limiters = limiter
        solver.char_decomp = 0
    else:
        solver = pyclaw.ClawSolver1D(mod)
        solver.limiters = limiter          # 1 -> minmod MUSCL, 0 -> first order
    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.cfl_desired = cfl

    dom = pyclaw.Domain([X_MIN], [X_MAX], [nx])
    state = pyclaw.State(dom, 2)
    state.problem_data["grav"] = G
    xc = np.asarray(state.grid.x.centers, dtype=float)
    q0 = np.where(xc <= 0.0, H0, dry_eps)
    state.q[0, :] = q0
    state.q[1, :] = 0.0

    ctrl = pyclaw.Controller()
    ctrl.solution = pyclaw.Solution(state, dom)
    ctrl.solver = solver
    ctrl.tfinal = t_end
    ctrl.num_output_times = 1
    ctrl.outdir = f"/tmp/xval_claw_{riemann_name}_{mode}_{limiter}"
    ctrl.output_format = "ascii"
    ctrl.verbose = False
    t0 = time.perf_counter()
    ctrl.run()
    elapsed = time.perf_counter() - t0

    q = np.asarray(ctrl.solution.q, dtype=float)
    dx = float(xc[1] - xc[0])
    h = q[0, :]
    down = xc > 0.0
    v_down = float((h[down] - q0[down]).sum() * dx)

    if mode == "sharpclaw":
        name, scheme = "clawpack_sharpclaw_weno5", \
            f"WENO{weno}+{time_integrator} (SharpClaw, component-wise minmod, char_decomp=0)"
    elif "roe" in riemann_name:
        name, scheme = "clawpack_classic_roe", \
            "MUSCL(minmod)-Hancock + Roe Riemann solver + entropy fix (ClawSolver1D)"
    elif limiter == 0:
        name, scheme = "clawpack_classic_hlle_1storder", \
            "first-order Godunov-Hancock + HLLE Riemann solver (ClawSolver1D)"
    else:
        name, scheme = "clawpack_classic_hlle", \
            "MUSCL(minmod)-Hancock + HLLE Riemann solver (ClawSolver1D)"

    return {
        "arm": name, "tool": "Clawpack-5.14.0",
        "riemann": riemann_name, "nx": nx, "cfl": cfl,
        "t_final": float(ctrl.solution.t),
        "x": xc.tolist(), "h": h.tolist(), "hu": q[1, :].tolist(),
        "v_down": v_down, "dry_floor": dry_eps,
        "mass_initial": float(q0.sum() * dx), "mass_final": float(h.sum() * dx),
        "steps": int(getattr(ctrl.solver, "num_step", -1)),
        "elapsed_s": elapsed, "grid": "cell-center", "scheme": scheme,
        "finite": bool(np.isfinite(h).all()),
        "nan_cells": int((~np.isfinite(h)).sum()),
    }


def _tag(arm: dict) -> str:
    eps = arm.get("dry_floor", 0.0)
    suf = "" if eps in (0.0, DRY_EPS_OSS) else f"_eps{eps:g}"
    return f"arm_{arm['arm']}_nx{arm['nx']}_cfl{arm['cfl']:g}{suf}"


def save(arms: list[dict]) -> list[str]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for a in arms:
        p = RUN_DIR / f"{_tag(a)}.json"
        p.write_text(json.dumps(a, indent=1) + "\n", encoding="utf-8")
        paths.append(str(p.relative_to(REPO_ROOT)))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="judged grids only (nx=200/400) at CFL=0.4")
    args = ap.parse_args()

    written: list[str] = []
    grids = [200, 400] if args.quick else [100, 200, 400, 800]
    for nx in grids:
        arms = [
            run_tensorlbm(nx, 0.4, T_END),
            run_tensorlbm(nx, 0.4, T_END, eps=DRY_EPS_OSS),
            _claw_solve("shallow_hlle_1D", "classic", nx, 0.4, T_END, DRY_EPS_OSS),
            _claw_solve("shallow_hlle_1D", "sharpclaw", nx, 0.4, T_END, DRY_EPS_1ST),
        ]
        if nx in (200, 400):
            arms += [
                # IC matched to the Roe arm: isolates the flux-family question
                run_tensorlbm(nx, 0.4, T_END, eps=DRY_EPS_ROE),
                _claw_solve("shallow_hlle_1D", "classic", nx, 0.4, T_END,
                            DRY_EPS_1ST, limiter=0),
                _claw_solve("shallow_roe_with_efix_1D", "classic", nx, 0.4, T_END,
                            DRY_EPS_ROE),
                # probe: documented OSS instability at a very thin dry floor
                _claw_solve("shallow_hlle_1D", "classic", nx, 0.4, T_END,
                            DRY_EPS_OSS_SENS),
            ]
        written += save(arms)
        print(f"nx={nx:4d} " + "  ".join(
            f"{a['arm'][:26]}={a['elapsed_s']:.2f}s/nan{a.get('nan_cells', 0)}" for a in arms))

    if not args.quick:
        for cfl in (0.2, 0.8):
            arms = [
                run_tensorlbm(400, cfl, T_END),
                run_tensorlbm(400, cfl, T_END, eps=DRY_EPS_OSS),
                _claw_solve("shallow_hlle_1D", "classic", 400, cfl, T_END, DRY_EPS_OSS),
                _claw_solve("shallow_hlle_1D", "sharpclaw", 400, cfl, T_END, DRY_EPS_1ST),
            ]
            written += save(arms)
            print(f"nx= 400 cfl={cfl}: " + "  ".join(f"{a['arm'][:26]}={a['elapsed_s']:.2f}s"
                                                    for a in arms))
        # dry-floor sensitivity of the OSS arm (same scheme, two floors)
        b_higheps = _claw_solve("shallow_hlle_1D", "classic", 400, 0.4, T_END, DRY_EPS_ROE)
        written += save([b_higheps])
        print(f"nx= 400 hlle eps={DRY_EPS_ROE:g}: {b_higheps['elapsed_s']:.2f}s")

    print(f"wrote {len(written)} arm files into {RUN_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())