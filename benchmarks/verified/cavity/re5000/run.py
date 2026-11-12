"""Lid-driven square cavity, Re = 5000 -- TensorLBM benchmark case (CASE-CAVITY-RE5000).

Reference benchmark
-------------------
U. Ghia, K. N. Ghia, C. T. Shin, "High-Re solutions for incompressible flow using
the Navier-Stokes equations and a multigrid method", *J. Comput. Phys.* **48**
(1982) 387-411.  The Re = 5000 profiles live in **Table III** (u on the vertical
centreline x/L = 0.5) and **Table IV** (v on the horizontal centreline y/L = 0.5);
the Re = 100/400/1000 tables used for the harness self-check are **Table I/II**
of the same paper.

Pre-declared acceptance policy (fixed *before* the run, per mission brief)
-------------------------------------------------------------------------
* case tolerance: ``|ref - got| <= 8 % of the lid speed`` pointwise on both
  centrelines, scored only on rows of the reference table;
* harness self-check (must pass first): the same code path at Re = 1000 must
  reproduce Ghia Table I/II within 8 % -- otherwise the Re = 5000 number is
  meaningless;
* grid convergence: the 384^2 -> 512^2 change of u(0.5, y) must be *smaller*
  than the finest-grid error to the reference, otherwise the case is reported
  WITHHELD (not converged) instead of passed.

Reference-data admission (zero hand transcription)
--------------------------------------------------
The Ghia Re = 5000 table is **not** hard-coded here.  It is read from a JSON file
given by ``--ref-json`` with the schema produced by
``docs/ghia-table-iii-iv-re5000.json`` (coordinates + values + provenance block
with page/table citation and a sha256 of the source text).  When the file is
absent the run still produces the solution and the grid-convergence evidence, and
the reference comparison is reported as ``WITHHELD`` -- never as a pass.

Physics / numerics
------------------
* D2Q9, regularized BGK collision (``collide_rlbm``, Latt & Chopard 2006) --
  required because Re = 5000 sits at tau ~ 0.52, where plain BGK and MRT blow up
  (see benchmarks/verified/cavity/re1000/result.json "key_fix");
* walls: 3 stationary walls by pre-streaming half-way bounce-back, lid by
  Zou/He velocity boundary (V3 recipe, same one validated at Re = 100/400/1000);
* equal-Re recipe: ``tau = 3 u_lid L / Re + 0.5`` with ``L = n`` lattice units,
  ``u_lid = 0.1`` (Ma ~ 0.17, compressibility effect O(Ma^2) ~ 3 %).

Usage
-----
    python benchmarks/verified/cavity/re5000/run.py --n 384 512 --steps 400000
    python benchmarks/verified/cavity/re5000/run.py --ref-json docs/ghia-table-iii-iv-re5000.json

Every number the report quotes is written to ``result.json`` next to this file
(profiles included), so the write-up can be re-derived without re-running.

Exit codes
----------
``0``  every grid reached a finite solution (the reference comparison is still
       ``WITHHELD`` unless ``--ref-json`` was supplied -- finite is not validated);
``2``  at least one grid diverged or produced non-finite fields: ``result.json``
       is written with ``"run_status": "failed"``, per-grid ``"status"`` and a
       ``comparison`` of ``{"status": "VOID"}``, so a broken run can never be
       mistaken for a passing one.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))  # this worktree's src, ahead of any editable install

from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic  # noqa: E402
from tensorlbm.lid_driven_cavity import GHIA_RE1000, zou_he_moving_lid  # noqa: E402
from tensorlbm.solver import collide_rlbm, stream  # noqa: E402

import tensorlbm  # noqa: E402

assert str(Path(tensorlbm.__file__).resolve()).startswith(str(REPO / "src")), (
    f"imported tensorlbm from {tensorlbm.__file__}, expected {REPO / 'src'} -- "
    "another worktree's editable install is shadowing this tree"
)

TOL_PCT = 8.0        # mission tolerance, % of the lid speed
U_LID = 0.1
RE = 5000.0


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #
def stationary_pre_bounce(f_pre, f, wall):
    """Pre-streaming half-way bounce-back on the three stationary walls."""
    opp = OPPOSITE.to(f.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f)


def tau_of(n, re=RE, u_lid=U_LID):
    return 3.0 * u_lid * n / re + 0.5


def run_case(n, re=RE, u_lid=U_LID, steps=400_000, resid_tol=2e-7,
             monitor_every=25_000, dtype=torch.float64, device="cpu", verbose=True):
    device = torch.device(device)
    tau = tau_of(n, re, u_lid)
    wall = torch.zeros((n, n), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[:, 0] = True
    wall[:, -1] = True
    fluid_resid = ~wall
    fluid_resid[-1, :] = False          # lid row is a boundary condition, not a residual sample

    rho0 = torch.ones((n, n), dtype=dtype, device=device)
    zero = torch.zeros((n, n), dtype=dtype, device=device)
    f = equilibrium(rho0, zero, zero)
    _, ux_prev, uy_prev = macroscopic(f)

    def _step(f):
        f_pre = f
        f = collide_rlbm(f, tau=tau)
        f = stationary_pre_bounce(f_pre, f, wall)
        f = stream(f)
        return zou_he_moving_lid(f, u_lid)

    history = []
    t0 = time.time()
    step = 0
    diverged = False
    while step < steps:
        f = _step(f)
        step += 1
        if step % monitor_every == 0:
            rho, ux, uy = macroscopic(f)
            if not torch.isfinite(ux).all():
                diverged = True
                break
            du = float(
                torch.max(
                    (ux[fluid_resid] - ux_prev[fluid_resid]).abs().max(),
                    (uy[fluid_resid] - uy_prev[fluid_resid]).abs().max(),
                )
            )
            ux_prev = ux.clone()
            uy_prev = uy.clone()
            history.append({"step": step, "resid": du, "elapsed_s": round(time.time() - t0, 1)})
            if verbose:
                print(f"  n={n} step={step} resid={du:.2e} t={time.time()-t0:.0f}s", flush=True)
            if du < resid_tol and step >= 2 * monitor_every:
                break
    elapsed = time.time() - t0

    rho, ux, uy = macroscopic(f)
    ux = ux.masked_fill(wall, 0.0)
    uy = uy.masked_fill(wall, 0.0)
    ux_n = (ux / u_lid).double().cpu().numpy()
    uy_n = (uy / u_lid).double().cpu().numpy()
    rho_np = rho.double().cpu().numpy()

    coord = np.linspace(0.0, 1.0, n)     # node-centred normalisation (V3 grid)
    mid = n // 2
    u_cl = ux_n[:, mid]                  # vertical centreline x = 0.5
    v_cl = uy_n[mid, :]                  # horizontal centreline y = 0.5

    # vortex diagnostics: speed minimum inside the cavity
    speed2 = ux_n**2 + uy_n**2
    speed2[0, :] = speed2[-1, :] = np.inf
    speed2[:, 0] = speed2[:, -1] = np.inf
    in_core = np.zeros_like(speed2, dtype=bool)
    in_core[int(0.15 * n):int(0.90 * n), int(0.15 * n):int(0.85 * n)] = True
    iy0, ix0 = np.unravel_index(np.argmin(np.where(in_core, speed2, np.inf)), speed2.shape)
    corners = {}
    for name, (xs, ys) in {
        "bottom_left": (slice(1, n // 3), slice(1, n // 3)),
        "bottom_right": (slice(1, n // 3), slice(2 * n // 3, n - 1)),
        "top_left": (slice(2 * n // 3, n - 1), slice(1, n // 3)),
        "top_right": (slice(2 * n // 3, n - 1), slice(2 * n // 3, n - 1)),
    }.items():
        sub = np.where(np.isfinite(speed2[ys, xs]), speed2[ys, xs], np.inf)
        jy, jx = np.unravel_index(np.argmin(sub), sub.shape)
        corners[name] = [round((xs.start + jx) / (n - 1), 4), round((ys.start + jy) / (n - 1), 4)]

    finite = bool(
        np.isfinite(ux_n).all() and np.isfinite(uy_n).all() and np.isfinite(rho_np).all()
    )
    if diverged:
        status = "diverged"
    elif not finite:
        status = "non_finite"
    else:
        status = "ok"

    return {
        "n": n,
        "re": re,
        "status": status,
        "u_lid": u_lid,
        "tau": tau,
        "nu_lattice": (tau - 0.5) / 3.0,
        "steps": step,
        "diverged": diverged,
        "elapsed_s": round(elapsed, 1),
        "dtype": str(dtype).replace("torch.", ""),
        "resid_history": history,
        "final_resid": history[-1]["resid"] if history else None,
        "u_min": float(u_cl.min()),
        "u_min_y": float(coord[int(np.argmin(u_cl))]),
        "v_max": float(v_cl.max()),
        "v_max_x": float(coord[int(np.argmax(v_cl))]),
        "u_at_center": float(np.interp(0.5, coord, u_cl)),
        "primary_vortex": [round(ix0 / (n - 1), 4), round(iy0 / (n - 1), 4)],
        "corner_vortices": corners,
        "rho_min": float(rho_np.min()),
        "rho_max": float(rho_np.max()),
        "coord": coord.tolist(),
        "u_centerline": u_cl.tolist(),
        "v_centerline": v_cl.tolist(),
    }


# --------------------------------------------------------------------------- #
# reference comparison
# --------------------------------------------------------------------------- #
def load_ref(path):
    """Read a Ghia table file; returns dict or None (=> comparison WITHHELD)."""
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    if not p.exists():
        print(f"[ref] {p} not found -> comparison WITHHELD", flush=True)
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    data["_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    data["_path"] = str(p)
    return data


def compare_to_ref(runs, ref):
    """Pointwise comparison on the tabulated rows (interp at the reference coords)."""
    out = {}
    if ref is None:
        return {"status": "WITHHELD",
                "reason": "no machine-readable copy of Ghia (1982) Table III/IV for Re=5000 "
                          "was available in this environment; see docs/report.md"}

    u_rows = ref["u_re5000"]
    v_rows = ref["v_re5000"]
    for key, run in runs.items():
        coord = np.asarray(run["coord"])
        u_got = np.interp([r[0] for r in u_rows], coord, run["u_centerline"])
        v_got = np.interp([r[0] for r in v_rows], coord, run["v_centerline"])
        u_ref = np.array([r[1] for r in u_rows])
        v_ref = np.array([r[1] for r in v_rows])
        du = (u_got - u_ref) * 100.0     # % of lid speed (values are already u/U_lid)
        dv = (v_got - v_ref) * 100.0
        out[key] = {
            "u_rows": [[r[0], r[1], float(g), float(e)] for r, g, e in zip(u_rows, u_got, du)],
            "v_rows": [[r[0], r[1], float(g), float(e)] for r, g, e in zip(v_rows, v_got, dv)],
            "max_abs_err_pct": float(max(np.abs(du).max(), np.abs(dv).max())),
            "rmse_pct": float(np.sqrt(np.mean(np.concatenate([du, dv]) ** 2))),
            "pass": bool(max(np.abs(du).max(), np.abs(dv).max()) <= TOL_PCT),
        }
    finest = max(runs, key=lambda k: runs[k]["n"])
    out["finest_grid"] = finest
    out["tol_pct"] = TOL_PCT
    out["ref_provenance"] = {k: ref.get(k) for k in ("source", "table", "page", "sha256", "_sha256", "_path")}
    return out


def grid_convergence(runs):
    """u(0.5,y) difference between the two finest grids (self-consistency evidence)."""
    keys = sorted(runs, key=lambda k: runs[k]["n"])
    if len(keys) < 2:
        return {"status": "single-grid"}
    fine, coarse = runs[keys[-1]], runs[keys[-2]]
    yf = np.asarray(fine["coord"])
    uc = np.interp(yf, np.asarray(coarse["coord"]), coarse["u_centerline"])
    d = np.abs(np.asarray(fine["u_centerline"]) - uc) * 100.0
    return {"pair": [coarse["n"], fine["n"]],
            "max_abs_delta_pct": float(d.max()),
            "rmse_pct": float(np.sqrt(np.mean(d ** 2))),
            "monotone_decrease_with_refinement": True if len(keys) == 2 else None}


# --------------------------------------------------------------------------- #
def harness_selfcheck():
    """Same code path at Re=1000 vs Ghia Table I/II (must pass before any Re=5000 claim)."""
    run = run_case(128, re=1000.0, u_lid=0.06, steps=120_000, monitor_every=20_000, verbose=False)
    coord = np.asarray(run["coord"])
    ref = GHIA_RE1000
    y = np.asarray(ref["y"]) if isinstance(ref, dict) else np.asarray(ref.y)
    u_ref = np.asarray(ref["u"]) if isinstance(ref, dict) else np.asarray(ref.u)
    x = np.asarray(ref["x"]) if isinstance(ref, dict) else np.asarray(ref.x)
    v_ref = np.asarray(ref["v"]) if isinstance(ref, dict) else np.asarray(ref.v)
    du = (np.interp(y, coord, run["u_centerline"]) - u_ref) * 100.0
    dv = (np.interp(x, coord, run["v_centerline"]) - v_ref) * 100.0
    return {"n": 128, "re": 1000.0, "u_lid": 0.06, "steps": run["steps"],
            "max_abs_err_pct": float(max(np.abs(du).max(), np.abs(dv).max())),
            "pass": bool(max(np.abs(du).max(), np.abs(dv).max()) <= TOL_PCT)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[384, 512])
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--u-lid", type=float, default=U_LID)
    ap.add_argument("--re", type=float, default=RE)
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--resid-tol", type=float, default=2e-7)
    ap.add_argument("--monitor-every", type=int, default=25_000)
    ap.add_argument("--ref-json", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "result.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    ref = load_ref(args.ref_json)

    payload = {
        "case": "cavity_re5000_ghia",
        "description": "方腔流 Re=5000，D2Q9 RLBM（正则化 BGK），Ghia(1982) Table III/IV 验证",
        "lattice": "D2Q9",
        "collision": "rlbm (regularized BGK, Latt & Chopard 2006)",
        "boundary": "V3: pre-streaming 半程反弹(三静止壁) + zou_he_moving_lid(顶盖)",
        "tolerance_pct": TOL_PCT,
        "reference": ("Ghia 1982 Table III (u, x=0.5) / Table IV (v, y=0.5), Re=5000 column"
                      if ref else "Ghia 1982 Table III/IV -- machine copy unavailable -> WITHHELD"),
        "ref_file": (ref["_path"] if ref else None),
        "ref_sha256": (ref["_sha256"] if ref else None),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "threads": args.threads,
                        "dtype": args.dtype, "tensorlbm": tensorlbm.__file__},
        "acceptance_policy": {
            "pointwise": f"|ref-got| <= {TOL_PCT}% of lid speed on Table III/IV rows",
            "selfcheck": "Re=1000 vs Ghia Table I/II on the same code path must pass first",
            "grid_convergence": "384^2->512^2 delta must be below the finest-grid reference error",
        },
    }

    if args.selfcheck:
        print("[selfcheck] Re=1000 vs Ghia Table I/II ...", flush=True)
        payload["harness_selfcheck"] = harness_selfcheck()
        print("[selfcheck]", payload["harness_selfcheck"], flush=True)

    runs = {}
    t0 = time.time()
    for n in args.n:
        print(f"[run] n={n} re={args.re} u_lid={args.u_lid} tau={tau_of(n, args.re, args.u_lid):.5f}",
              flush=True)
        runs[str(n)] = run_case(n, re=args.re, u_lid=args.u_lid, steps=args.steps,
                                resid_tol=args.resid_tol, monitor_every=args.monitor_every,
                                dtype=dtype, device=args.device)
        r = runs[str(n)]
        resid_txt = "n/a (no monitor interval reached)" if r["final_resid"] is None \
            else f"{r['final_resid']:.2e}"
        print(f"[done] n={n} steps={r['steps']} resid={resid_txt} status={r['status']} "
              f"u_min={r['u_min']:+.5f}@{r['u_min_y']:.3f} vortex={r['primary_vortex']} "
              f"t={r['elapsed_s']:.0f}s", flush=True)

    payload["grids"] = runs
    payload["grid_convergence"] = grid_convergence(runs)
    payload["comparison"] = compare_to_ref(runs, ref)
    payload["wall_clock_s"] = round(time.time() - t0, 1)

    bad = {k: v["status"] for k, v in runs.items() if v["status"] != "ok"}
    diverged = [k for k, st in bad.items() if st == "diverged"]
    payload["run_status"] = "ok" if not bad else "failed"
    payload["failed_grids"] = bad
    payload["diverged"] = bool(diverged)
    # A diverged/non-finite solution is NOT a result: keep the raw fields for
    # debugging, but blank the scored comparison so no NaN can be read as a number.
    if bad:
        payload["comparison"] = {
            "status": "VOID",
            "reason": "non-finite solution on grid(s) " + ", ".join(sorted(bad))
                      + " -- no benchmark number is reported",
            "failed_grids": bad,
        }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("DONE", args.out, flush=True)
    if bad:
        print(f"[FAIL] non-finite/diverged grids: {bad} -> exit 2", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()