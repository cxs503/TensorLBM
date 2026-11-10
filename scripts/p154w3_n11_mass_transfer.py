#!/usr/bin/env python
"""P154W3-N11 driver: convective mass transfer verification surface.

Modes
-----
``bench``     report the per-step cost of the advection-diffusion kernels
``calib``     measured numerical diffusivity vs cell Peclet number
``leveque``   Leveque/Graetz linear-shear wall (the ``-1/3`` flux law)
``blasius``   Blasius flat plate mass transfer (``Sh ~ Re^1/2 Sc^1/3``)
``cylinder``  cylinder in cross flow (``Sh`` vs Ranz-Marshall / Hilpert)
``analogy``   Chilton-Colburn ``j_D = j_H = C_f/2`` (frozen + coupled arms)

Every case writes one JSON file under ``artifacts/p154w3_n11/`` so the sweep can
be sharded over the cluster and the numbers stay programmatically injectable
into the report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tensorlbm.convective_mass_transfer as cmt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "p154w3_n11"

# --- case tables (single source of truth for the sweep and for the report) ---

#: Leveque family.  Resolution levels keep the *physical* problem fixed:
#: halving the lattice spacing means ``a -> a/2``, ``D -> 2D``, ``ny -> 2 ny``
#: while the lattice velocity field ``u = a y`` is unchanged, so ``delta_c``
#: doubles in lattice cells (see the report section on grid refinement).
LEVEQUE_LEVELS = {
    "s025": {"nx": 128, "ny": 16, "shear": 0.008, "D": 7.5e-5, "steps": 10000},
    "s050": {"nx": 256, "ny": 32, "shear": 0.004, "D": 1.5e-4, "steps": 18000},
    "s100": {"nx": 512, "ny": 64, "shear": 0.002, "D": 3.0e-4, "steps": 34000},
}
#: base-level variants: BGK collision and a uniform initial field (IC audit)
LEVEQUE_VARIANTS = {
    "bgk": {"nx": 512, "ny": 64, "shear": 0.002, "D": 3.0e-4, "steps": 34000,
            "trt": False, "init": "leveque"},
    "icuniform": {"nx": 128, "ny": 16, "shear": 0.008, "D": 7.5e-5, "steps": 30000,
                  "trt": True, "init": "uniform"},
    "icuniform_s050": {"nx": 256, "ny": 32, "shear": 0.004, "D": 1.5e-4, "steps": 40000,
                       "trt": True, "init": "uniform"},
    "bgk_s050": {"nx": 256, "ny": 32, "shear": 0.004, "D": 1.5e-4, "steps": 18000,
                 "trt": False, "init": "leveque"},
}

BLASIUS_BASE = {"nx": 512, "ny": 192, "u_inf": 0.08, "nu": 0.08}
BLASIUS_SC = (1.0, 10.0, 100.0, 300.0, 1000.0)

CYL_RADIUS = 12.0
CYL_NX, CYL_NY = 312, 192
CYL_CASES = ((10.0, 9000), (20.0, 9000), (40.0, 10000), (80.0, 14000), (100.0, 15000),
             (150.0, 15000), (200.0, 16000))


def _env() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
        "num_threads": torch.get_num_threads(),
    }


def _save(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    payload.setdefault("env", _env())
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"[written] {path}", flush=True)
    return path


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------


def mode_bench(_: argparse.Namespace) -> None:
    rows = []
    for nx, ny, steps in ((128, 64, 60), (256, 100, 60), (512, 192, 40)):
        i = np.arange(ny, dtype=np.float64)
        ux = torch.tensor(np.repeat((0.0015 * i)[:, None], nx, axis=1), dtype=torch.float32)
        uy = torch.zeros((ny, nx), dtype=torch.float32)
        t0 = time.time()
        cmt.solve_frozen_field_ad(ux, uy, 5e-5, steps=steps, check_every=steps, init="profile")
        dt = (time.time() - t0) / steps
        rows.append({"nx": nx, "ny": ny, "ms_per_step": dt * 1e3,
                     "mlups": nx * ny / dt / 1e6})
        print(rows[-1], flush=True)
    _save("bench", {"rows": rows})


def mode_calib(_: argparse.Namespace) -> None:
    rows = []
    for trt in (True, False):
        for d in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
            r = cmt.effective_diffusivity(d, u=0.05, nx=64, steps=3000, trt=trt)
            r["trt"] = trt
            rows.append(r)
            print(f"trt={trt} D={d:.1e} Pe_cell={r['Pe_cell']:8.1f} tau_o={r['tau_o']:.5f} "
                  f"tau_s={r['tau_s']} D_eff/D={r['ratio']:.4f}", flush=True)
    _save("calib", {"u": 0.05, "rows": rows})


def _leveque_report(res: cmt.FrozenFieldResult, shear: float, D: float, label: str,
                    fit_min_delta: float = 2.0) -> dict:
    x = res.x
    delta_c = (9.0 * D * x / shear) ** (1.0 / 3.0)
    ok = np.isfinite(res.sh_x) & (delta_c >= fit_min_delta) & (x <= x[-1] - 8)
    exact = cmt.leveque_local_sh(x, shear, D)
    flux_exact = cmt.leveque_local_h(x, shear, D) * D  # h * D == dG/dy (dimensionless)
    fit = cmt.fit_power_law(x[ok], res.sh_x[ok])
    fit_flux = cmt.fit_power_law(x[ok], res.flux[ok])
    rel = (res.sh_x[ok] / exact[ok] - 1.0) * 100.0
    return {
        "label": label,
        "shear": shear,
        "D": D,
        "nx": int(x.size),
        "ny": int(res.G.shape[0]),
        "steps": res.steps,
        "converged": bool(res.converged),
        "fit_window": {"x_min": float(x[ok].min()), "x_max": float(x[ok].max()),
                       "n": int(ok.sum()), "delta_c_min": float(delta_c[ok].min()),
                       "delta_c_max": float(delta_c[ok].max())},
        "sh_x_fit": fit,
        "flux_fit": fit_flux,
        "sh_err_pct": {"median": float(np.median(rel)), "max_abs": float(np.abs(rel).max()),
                       "at_x_max": float(rel[-1])},
        "delta_c_over_dx": [float(v) for v in delta_c[ok][:: max(int(ok.sum()) // 8, 1)]],
        "sh_x": res.sh_x.tolist(),
        "sh_x_exact": exact.tolist(),
        "delta_c": delta_c.tolist(),
        "history": res.history,
        "collision": res.collision,
        "nan_in_field": bool(np.isnan(res.G).any()),
    }


def mode_leveque(args: argparse.Namespace) -> None:
    table = {**{f"L_{k}": dict(v) for k, v in LEVEQUE_LEVELS.items()},
             **{f"V_{k}": dict(v) for k, v in LEVEQUE_VARIANTS.items()}}
    names = sorted(table)
    sel = names if args.case is None else [names[args.case]]
    if args.total and args.case is not None:
        sel = [names[i] for i in range(args.case, len(names), args.total)]
    for name in sel:
        cfg = table[name]
        init = cfg.get("init", args.init)
        trt = bool(cfg.get("trt", True))
        t0 = time.time()
        res = cmt.solve_shear_channel_ad(nx=cfg["nx"], ny=cfg["ny"], shear=cfg["shear"],
                                         diffusivity=cfg["D"], steps=cfg["steps"],
                                         check_every=max(cfg["steps"] // 17, 200),
                                         tol=1e-5, init=init, trt=trt)
        rep = _leveque_report(res, cfg["shear"], cfg["D"], name)
        rep["wall_seconds"] = time.time() - t0
        rep["init"] = init
        rep["trt"] = trt
        print(f"{name}: p={rep['sh_x_fit']['p']:.5f} (2/3={2/3:.5f}) "
              f"median_err={rep['sh_err_pct']['median']:+.2f}% steps={res.steps}", flush=True)
        _save(f"leveque_{name}", rep)


def _blasius_report(res: cmt.FrozenFieldResult, cfg: dict, sc: float) -> dict:
    u_inf, nu, D = cfg["u_inf"], cfg["nu"], cfg["nu"] / sc
    x = res.x
    delta_c = np.sqrt(nu * x / u_inf) / sc ** (1.0 / 3.0)
    delta_m = 5.0 * np.sqrt(nu * x / u_inf)
    re_x = u_inf * x / nu
    ok = np.isfinite(res.sh_x) & (delta_c >= 2.0) & (x <= x[-1] - 8) & (re_x >= 10.0)
    if int(ok.sum()) < 5:  # fall back to a wider (less resolved) window
        ok = np.isfinite(res.sh_x) & (delta_c >= 1.0) & (x <= x[-1] - 8) & (re_x >= 5.0)
    if int(ok.sum()) < 5:  # field unusable (diverged): record the failure honestly
        return {"Sc": sc, "D": D, "u_inf": u_inf, "nu": nu, "nx": int(x.size),
                "ny": int(res.G.shape[0]), "steps": res.steps, "converged": False,
                "failed": True, "nan_in_field": bool(np.isnan(res.G).any()),
                "collision": res.collision, "sh_x": [], "x": x.tolist(),
                "sh_err_pct": {"median": float("nan"), "max_abs": float("nan")},
                "sh_vs_re_fit": {"p": float("nan"), "A": float("nan"), "r2": float("nan"), "n": 0},
                "fit_window": {"x_min": float("nan"), "x_max": float("nan"), "Re_min": float("nan"),
                               "Re_max": float("nan"), "delta_c_min": float("nan"), "n": 0},
                "delta_c": [], "delta_m": [], "sh_x_exact": [], "history": res.history}
    exact = cmt.blasius_local_sh(x, u_inf, nu, D)
    fit_re = cmt.fit_power_law(re_x[ok], res.sh_x[ok])
    rel = (res.sh_x[ok] / exact[ok] - 1.0) * 100.0
    return {
        "Sc": sc, "D": D, "u_inf": u_inf, "nu": nu,
        "Re_L": float(u_inf * cfg["nx"] / nu),
        "nx": int(x.size), "ny": int(res.G.shape[0]), "steps": res.steps,
        "converged": bool(res.converged),
        "fit_window": {"x_min": float(x[ok].min()), "x_max": float(x[ok].max()),
                       "Re_min": float(re_x[ok].min()), "Re_max": float(re_x[ok].max()),
                       "delta_c_min": float(delta_c[ok].min()), "n": int(ok.sum())},
        "sh_vs_re_fit": fit_re,
        "sh_err_pct": {"median": float(np.median(rel)), "max_abs": float(np.abs(rel).max())},
        "sh_x": res.sh_x.tolist(), "sh_x_exact": exact.tolist(), "x": x.tolist(),
        "delta_c": delta_c.tolist(), "delta_m": delta_m.tolist(),
        "history": res.history,
        "collision": res.collision,
        "nan_in_field": bool(np.isnan(res.G).any()),
    }


def mode_blasius(args: argparse.Namespace) -> None:
    cases = list(BLASIUS_SC)
    sel = cases if args.case is None else [cases[args.case] if args.case < len(cases)
                                           else cases[args.case % len(cases)]]
    for sc in sel:
        cfg = dict(BLASIUS_BASE)
        D = cfg["nu"] / sc
        t0 = time.time()
        res, _, _ = cmt.solve_blasius_plate_ad(nx=cfg["nx"], ny=cfg["ny"], u_inf=cfg["u_inf"],
                                               nu=cfg["nu"], diffusivity=D, steps=args.steps,
                                               check_every=max(args.steps // 8, 200),
                                               tol=1e-6, init="blasius")
        rep = _blasius_report(res, cfg, sc)
        rep["init"] = "blasius-local-shear-leveque"
        rep["wall_seconds"] = time.time() - t0
        print(f"Sc={sc:g}: slope={rep['sh_vs_re_fit']['p']:.4f} (0.5) "
              f"median_err={rep['sh_err_pct']['median']:+.2f}% steps={res.steps}", flush=True)
        _save(f"blasius_Sc{sc:g}", rep)


def mode_cylinder(args: argparse.Namespace) -> None:
    cases = CYL_CASES
    sel = cases if args.case is None else [cases[args.case % len(cases)]]
    for re, steps in sel:
        t0 = time.time()
        res = cmt.run_cylinder_mass_transfer(CYL_NX, CYL_NY, CYL_RADIUS, u_in=0.05, reynolds=re,
                                             scalar_sc=(1.0, 2.0), n_steps=steps,
                                             diagnostic_every=max(steps // 10, 500))
        rep = dict(res)
        rep["wall_seconds"] = time.time() - t0
        rep["RanzMarshall"] = {f"Sc{s['Sc']:g}": cmt.ranz_marshall_mean_sh(re, s["Sc"])
                              for s in res["scalars"]}
        rep["Hilpert"] = {f"Sc{s['Sc']:g}": cmt.hilpert_mean_sh(re, s["Sc"])
                         for s in res["scalars"]}
        rep["ChurchillBernstein"] = {f"Sc{s['Sc']:g}": cmt.churchill_bernstein_mean_sh(re, s["Sc"])
                                    for s in res["scalars"]}
        for s in res["scalars"]:
            rm = cmt.ranz_marshall_mean_sh(re, s["Sc"])
            s["dev_RM_pct"] = (s["Sh"] / rm - 1.0) * 100.0
            s["dev_Hilpert_pct"] = (s["Sh"] / cmt.hilpert_mean_sh(re, s["Sc"]) - 1.0) * 100.0
            s["dev_CB_pct"] = (s["Sh"] / cmt.churchill_bernstein_mean_sh(re, s["Sc"]) - 1.0) * 100.0
        print(f"Re={re:g}: Cd={res['Cd_mean']:.3f} St={res['St']:.3f} "
              + " ".join(f"Sh(Sc{s['Sc']:g})={s['Sh']:.3f} devRM={s['dev_RM_pct']:+.1f}%"
                         for s in res["scalars"]), flush=True)
        _save(f"cylinder_Re{re:g}", rep)


def mode_analogy(args: argparse.Namespace) -> None:
    """j_D = j_H = C_f/2 on the Blasius plate (frozen similarity velocity field)."""
    cfg = dict(BLASIUS_BASE)
    cases = (("Pr=Sc=1", 1.0, 1.0), ("Pr=0.7,Sc=1", 0.7, 1.0), ("Pr=0.7,Sc=2", 0.7, 2.0),
             ("Pr=Sc=100", 100.0, 100.0))
    if args.case is not None:
        cases = (cases[args.case % len(cases)],)
    for label, pr, sc in cases:
        t0 = time.time()
        kw = dict(nx=cfg["nx"], ny=cfg["ny"], u_inf=cfg["u_inf"], nu=cfg["nu"],
                  steps=args.steps, check_every=max(args.steps // 8, 200), tol=1e-6,
                  init="blasius")
        res_d, _, _ = cmt.solve_blasius_plate_ad(diffusivity=cfg["nu"] / sc, **kw)
        res_h, _, _ = cmt.solve_blasius_plate_ad(diffusivity=cfg["nu"] / pr, **kw)
        x = res_d.x
        re_x = cfg["u_inf"] * x / cfg["nu"]
        delta_c = np.sqrt(cfg["nu"] * x / cfg["u_inf"]) / sc ** (1.0 / 3.0)
        ok = np.isfinite(res_d.sh_x) & np.isfinite(res_h.sh_x) & (delta_c >= 2.0) & (x <= x[-1] - 8) & (re_x >= 10.0)
        sh_d = res_d.sh_x[ok]
        sh_h = res_h.sh_x[ok]
        j_d = cmt.chilton_colburn_j(sh_d / (re_x[ok] * sc), sc)
        j_h = cmt.chilton_colburn_j(sh_h / (re_x[ok] * pr), pr)
        cf_half = cmt.blasius_local_cf(x[ok], cfg["u_inf"], cfg["nu"]) / 2.0
        block = {
            "label": label, "Pr": pr, "Sc": sc, "n": int(ok.sum()),
            "collision_D": res_d.collision, "collision_H": res_h.collision,
            "sh_D_median": float(np.median(sh_d)), "sh_H_median": float(np.median(sh_h)),
            "sh_D_vs_analytic_pct": float(np.median(sh_d / cmt.blasius_local_sh(x[ok], cfg["u_inf"], cfg["nu"], cfg["nu"] / sc) - 1.0) * 100.0),
            "sh_H_vs_analytic_pct": float(np.median(sh_h / cmt.blasius_local_sh(x[ok], cfg["u_inf"], cfg["nu"], cfg["nu"] / pr) - 1.0) * 100.0),
            "j_D_median": float(np.median(j_d)), "j_H_median": float(np.median(j_h)),
            "Cf_half_median": float(np.median(cf_half)),
            "jD_vs_jH_pct": float(np.median(j_d / j_h - 1.0) * 100.0),
            "jD_vs_Cf_half_pct": float(np.median(j_d / cf_half - 1.0) * 100.0),
            "x": x[ok].tolist(), "j_D": j_d.tolist(), "j_H": j_h.tolist(),
            "Cf_half": cf_half.tolist(), "wall_seconds": time.time() - t0,
        }
        print(f"{label}: jD/jH-1={block['jD_vs_jH_pct']:+.2f}% "
              f"jD/Cf2-1={block['jD_vs_Cf_half_pct']:+.2f}%", flush=True)
        _save(f"analogy_frozen_{label.replace('=', '').replace(',', '_')}", block)


def mode_analogy_coupled(args: argparse.Namespace) -> None:
    """Chilton-Colburn analogy on an LBM-solved plate (three-way: Cf, j_D, j_H)."""
    out: dict = {}
    # coupled arm: LBM momentum on the plate, scalars on the solved field
    nx, ny, u_in, nu = 384, 160, 0.05, 0.02
    t0 = time.time()
    res = cmt.solve_channel_plate_momentum(nx, ny, u_in, nu, args.coupled_steps,
                                           scalar_ds=(nu, nu / 0.7), scalar_labels=("Sc1", "Pr0.7"),
                                           diagnostic_every=max(args.coupled_steps // 8, 500))
    x = res["x"]
    cf = res["cf_x"]
    re_x = u_in * x / nu
    blocks = []
    for lab in ("Sc1", "Pr0.7"):
        sh = res["scalars"][lab]["sh_x"]
        ratio = 1.0 if lab == "Sc1" else 0.7
        st = sh / (re_x * ratio)
        j = st * ratio ** (2.0 / 3.0)
        ok = np.isfinite(sh) & (x >= 40) & (x <= x[-1] - 20) & (re_x >= 10)
        blocks.append({"label": lab, "j_median": float(np.nanmedian(j[ok])),
                       "sh_x": sh.tolist(), "j_x": j.tolist(), "n": int(ok.sum())})
    ok = np.isfinite(cf) & (x >= 40) & (x <= x[-1] - 20)
    blas = cmt.blasius_local_cf(x[ok], u_in, nu)
    out = {
        "nx": nx, "ny": ny, "u_in": u_in, "nu": nu, "steps": args.coupled_steps,
        "Re_L": u_in * nx / nu,
        "x": x.tolist(), "cf_lbm": cf.tolist(), "cf_blasius": blas.tolist(),
        "cf_err_pct_median": float(np.median((cf[ok] / blas - 1.0) * 100.0)),
        "cf_err_pct_max": float(np.abs((cf[ok] / blas - 1.0) * 100.0).max()),
        "cf_history": res["cf_history"],
        "scalars": blocks,
        "wall_seconds": time.time() - t0,
    }
    print(f"coupled plate: Cf median err {out['cf_err_pct_median']:+.1f}% "
          f"max {out['cf_err_pct_max']:.1f}%", flush=True)
    _save("analogy_coupled", out)


def mode_fields(_: argparse.Namespace) -> None:
    """Dump the concentration fields used for the contour figure (fig6)."""
    lev = dict(LEVEQUE_LEVELS["s050"])
    res = cmt.solve_shear_channel_ad(nx=lev["nx"], ny=lev["ny"], shear=lev["shear"],
                                     diffusivity=lev["D"], steps=lev["steps"],
                                     check_every=max(lev["steps"] // 17, 200), tol=1e-5,
                                     init="leveque")
    x = np.maximum(np.arange(lev["nx"], dtype=float) - 0.5, 0.5)
    delta_c = (9.0 * lev["D"] * x / lev["shear"]) ** (1.0 / 3.0)
    np.savez_compressed(OUT / "field_leveque.npz", G=res.G, x=x, delta_c=delta_c,
                        sh_x=res.sh_x,
                        sh_x_exact=cmt.leveque_local_sh(x, lev["shear"], lev["D"]))
    print(f"[field] leveque {res.G.shape} steps={res.steps}", flush=True)
    cfg = dict(BLASIUS_BASE)
    sc = 100.0
    res2, _, _ = cmt.solve_blasius_plate_ad(nx=256, ny=96, u_inf=cfg["u_inf"], nu=cfg["nu"],
                                            diffusivity=cfg["nu"] / sc, steps=5000,
                                            check_every=500, tol=1e-5, init="blasius")
    x2 = np.maximum(np.arange(256, dtype=float) - 0.5, 0.5)
    np.savez_compressed(OUT / "field_blasius.npz", G=res2.G, x=x2, Sc=sc,
                        u_inf=cfg["u_inf"], nu=cfg["nu"], sh_x=res2.sh_x,
                        sh_x_exact=cmt.blasius_local_sh(x2, cfg["u_inf"], cfg["nu"],
                                                        cfg["nu"] / sc))
    print(f"[field] blasius {res2.G.shape} steps={res2.steps}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["bench", "calib", "leveque", "blasius", "cylinder", "analogy",
                             "analogy_coupled", "fields"])
    ap.add_argument("--case", type=int, default=None)
    ap.add_argument("--total", type=int, default=None)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("P154_THREADS", "4")))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--coupled-steps", type=int, default=8000)
    ap.add_argument("--init", default="leveque")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    {"bench": mode_bench, "calib": mode_calib, "leveque": mode_leveque,
     "blasius": mode_blasius, "cylinder": mode_cylinder,
     "analogy": mode_analogy, "analogy_coupled": mode_analogy_coupled,
     "fields": mode_fields}[args.mode](args)


if __name__ == "__main__":
    main()