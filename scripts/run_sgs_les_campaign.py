#!/usr/bin/env python3
"""Campaign driver for the four-SGS LBM-LES verification surface.

Arms
----
``dit``      decaying isotropic turbulence (CBC 1971) -- four closures x 32/64/128^3
``tgv``      Taylor-Green vortex -- four closures x 64/128^3 at one Re
``dns``      de-aliased pseudo-spectral Taylor-Green DNS reference (96/128^3)
``laminar``  frozen-field laminar degeneracy + Stokes shear-wave dynamic control
``controls`` SGS-off vs plain-BGK identity and the ``nu_t/nu`` grid scaling
``assemble`` merge every per-case JSON into ``runs/sgs_les/report.json`` and the
             machine-checkable acceptance table

Usage
-----
    python scripts/run_sgs_les_campaign.py --arm dit --workers 4
    python scripts/run_sgs_les_campaign.py --arm assemble
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from tensorlbm.turbulence import sgs_verification as sv  # noqa: E402

OUT = REPO / "runs" / "sgs_les"

# Re = 200: the Taylor-Green flow still develops the vortex-stretching
# dissipation peak, while a 96^3 de-aliased DNS is genuinely resolved
# (k_max*eta ~ 2 with the 2/3 rule).  At Re = 400 the same DNS would stop at
# k_max*eta ~ 1.1, i.e. it would not be a reference at all.
RE_CASE = 200.0
TGV_T_END = 12.0
TGV_AMP = 0.06
# the TGV dissipation peak sits at t ~ 6 at Re = 200, so t_end = 8 with
# dt = 0.02 (RK4, 2/3-dealiased) resolves the peak; the values are overridable
# from the environment for convergence studies
DNS_T_END = float(os.environ.get("SGS_DNS_T_END", "8.0"))
DNS_DT = float(os.environ.get("SGS_DNS_DT", "0.02"))


def _save(name: str, data: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)
    print(f"[saved] {path} ({path.stat().st_size / 1e3:.0f} kB)", flush=True)
    return path


def _load(name: str) -> dict:
    return json.loads((OUT / f"{name}.json").read_text())


# ---------------------------------------------------------------- arms
def job_dit(args) -> str:
    model, n = args
    t0 = time.time()
    res = sv.run_dit(model, n)
    res["wall_sec"] = sv.sig(time.time() - t0)
    _save(f"dit_{model}_n{n}", res)
    return f"dit {model} n={n}"


def job_tgv(args) -> str:
    model, n = args
    t0 = time.time()
    res = sv.run_tgv(model, n, re=RE_CASE, t_end=TGV_T_END, amplitude_lat=TGV_AMP)
    res["wall_sec"] = sv.sig(time.time() - t0)
    _save(f"tgv_{model}_n{n}", res)
    return f"tgv {model} n={n}"


def job_dns(n: int) -> str:
    t0 = time.time()
    res = sv.spectral_dns_tgv(n=n, re=RE_CASE, t_end=DNS_T_END, dt=DNS_DT)
    res["wall_sec"] = sv.sig(time.time() - t0)
    _save(f"dns_n{n}", res)
    return f"dns n={n}"


def job_laminar(_: int) -> str:
    deg = sv.laminar_degeneracy(n=32)
    _save("laminar_degeneracy", deg)
    stokes = {}
    for model in sv.SGS_MODELS + ("none",):
        stokes[model] = sv.stokes_shear_wave(model, n=64, steps=4000, record_every=40)
        _save(f"stokes_{model}", stokes[model])
    return "laminar"


def job_controls(_: int) -> str:
    """SGS-off == plain BGK identity check (same kernel, same IC)."""
    from tensorlbm.turbulence.sgs_lbm_kernel import SGSKernel

    # 120 steps: the CBC-DIT at n=32 has tau-0.5 = 3e-4, i.e. plain BGK is
    # outside its stability envelope after ~160 steps (measured), so the
    # identity check must stay well inside that window
    n, steps = 32, 120
    cfg = sv.dit_setup(n)
    u0, _ic = sv.make_dit_ic(n, cfg)

    def run(model, C=None):
        ker = SGSKernel(n)
        ker.set_bc_mode(1)
        ker.set_velocity(u0)
        for _ in range(steps):
            ker.step(model, cfg["tau"], 0.0 if C is None else C, stats=False)
        return ker.get_velocity()

    ref = run("none")
    out = {"n": n, "steps": steps, "tau": sv.sig(cfg["tau"]), "cases": {}}
    for model in ("smagorinsky", "wale", "vreman"):
        u = run(model, 0.0)
        out["cases"][model] = {"max_abs_velocity_diff": sv.sig(float(np.abs(u - ref).max())),
                               "const": 0.0}
    out["max_abs_velocity_diff"] = max(c["max_abs_velocity_diff"] for c in out["cases"].values())
    _save("sgs_off_identity", out)
    return "controls"


# ---------------------------------------------------------------- assemble
def assemble() -> dict:
    dit_results, dit_metrics = {}, {}
    for model in sv.SGS_MODELS:
        for n in (32, 64, 128):
            p = OUT / f"dit_{model}_n{n}.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            dit_results[n if model == sv.SGS_MODELS[0] else (n, model)] = r
    # metrics per (model, n)
    for model in sv.SGS_MODELS:
        for n in (32, 64, 128):
            p = OUT / f"dit_{model}_n{n}.json"
            if p.exists():
                dit_metrics[(model, n)] = sv.dit_spectrum_metrics(json.loads(p.read_text()))

    grid_conv = {}
    for model in sv.SGS_MODELS:
        rs = [json.loads((OUT / f"dit_{model}_n{n}.json").read_text())
              for n in (32, 64, 128) if (OUT / f"dit_{model}_n{n}.json").exists()]
        if len(rs) >= 2:
            grid_conv[model] = sv.grid_convergence_spectra(rs, station=98.0)

    tgv_les = {}
    for model in sv.SGS_MODELS:
        for n in (64, 128):
            p = OUT / f"tgv_{model}_n{n}.json"
            if p.exists():
                tgv_les[(model, n)] = json.loads(p.read_text())
    dns = {}
    for p in sorted(OUT.glob("dns_n*.json")):
        try:
            n = int(p.stem.split("_n")[1])
        except ValueError:  # pragma: no cover - defensive
            continue
        dns[n] = json.loads(p.read_text())
    dns_conv = {}
    if len(dns) >= 2:
        ks = sorted(dns)
        a, b = dns[ks[-2]], dns[ks[-1]]
        dns_conv = {
            "cases": [{"n": dns[k]["n"], "eps_max": dns[k]["eps_max"],
                       "t_peak": dns[k]["t_peak"]} for k in ks],
            "rel_change_eps_max": sv.sig(abs(b["eps_max"] - a["eps_max"]) / a["eps_max"]),
            "rel_change_t_peak": sv.sig(abs(b["t_peak"] - a["t_peak"]) / a["t_peak"]),
        }
    dns_ref = dns[max(dns)] if dns else {"eps_max": float("nan"), "t_peak": float("nan"), "n": 0}

    deg = json.loads((OUT / "laminar_degeneracy.json").read_text())
    stokes = {m: json.loads((OUT / f"stokes_{m}.json").read_text())
              for m in sv.SGS_MODELS + ("none",) if (OUT / f"stokes_{m}.json").exists()}
    sgs_off = json.loads((OUT / "sgs_off_identity.json").read_text())

    # nu_t/nu vs grid: only runs that reached the end of the window can be used
    # (the CBC configuration diverges for the non-static closures, see report)
    pts = []
    for n in (32, 64, 128):
        p = OUT / f"dit_smagorinsky_n{n}.json"
        if not p.exists():
            continue
        res = json.loads(p.read_text())
        if res["completed_steps"] < 0.99 * res["nsteps"]:
            continue
        pts.append((n, float(np.mean(res["time_series"]["nut_over_nu"]))))
    delta_scaling = {"points": [{"n": n, "nut_over_nu_mean": sv.sig(v)} for n, v in pts]}
    if len(pts) >= 2:
        xs = np.log(np.array([p[0] for p in pts], dtype=float))
        ys = np.log(np.array([max(p[1], 1e-300) for p in pts], dtype=float))
        slope = float(np.polyfit(xs, ys, 1)[0])
        delta_scaling["fitted_exponent"] = sv.sig(slope)
        delta_scaling["note"] = ("nu_t = (Cs Delta)^2 |S| with Delta = dx, so a "
                                 "grid-refined run at fixed physical flow must "
                                 "show nu_t/nu ~ n^-2 (Delta -> 0 limit)")

    # per-model, per-grid TGV entries keyed by name for the gate table
    tgv_by_model = {}
    for (model, n), r in tgv_les.items():
        tgv_by_model.setdefault(model, []).append(r)
    primary = {}
    for model, rs in tgv_by_model.items():
        rs = sorted(rs, key=lambda r: r["n"])
        primary[(model, rs[-1]["n"])] = rs[-1]

    crit = sv.evaluate_acceptance(
        dit_metrics={(m, n): v for (m, n), v in dit_metrics.items()},
        grid_conv=next(iter(grid_conv.values())) if grid_conv else {"comparisons": [], "grids": []},
        tgv_dns=dns_ref,
        tgv_dns_conv=dns_conv,
        tgv_les=primary,
        degeneracy=deg,
        stokes=stokes,
        sgs_off=sgs_off,
        delta_scaling=delta_scaling,
        pytest_examples=int(os.environ.get("SGS_PYTEST_CASES", "0")) or None,
    )
    report = {
        "campaign": "p465w1-n05-lbm-les / four-SGS verification ladder",
        "repo": "TensorLBM",
        "re_case": RE_CASE,
        "tgv": {"t_end": TGV_T_END, "amplitude_lat": TGV_AMP},
        "dns": {"t_end": DNS_T_END, "dt": DNS_DT, "cases": list(dns)},
        "dit_setup": next((json.loads((OUT / f"dit_{sv.SGS_MODELS[0]}_n{n}.json")
                                      .read_text())["config"]
                           for n in (64,) if (OUT / f"dit_{sv.SGS_MODELS[0]}_n{n}.json").exists()),
                          {}),
        "dit_metrics": {f"{m}:n{n}": v for (m, n), v in dit_metrics.items()},
        "grid_convergence": {m: v for m, v in grid_conv.items()},
        "tgv_dns": {k: dns_ref.get(k) for k in ("n", "re", "eps_max", "t_peak", "eps_initial")},
        "tgv_dns_convergence": dns_conv,
        "tgv_les": {f"{m}:n{n}": {k: r.get(k) for k in
                                  ("model", "n", "eps_max", "t_peak", "eps_sgs_max",
                                   "eps_sgs_fraction_at_peak", "tau", "nu_lat", "nsteps",
                                   "completed_steps", "blew_up_at_step")}
                    for (m, n), r in primary.items()},
        "laminar_degeneracy": deg,
        "stokes": {k: {kk: v[kk] for kk in ("nu_molecular", "nu_effective", "nu_eff_over_nu",
                                            "nut_over_nu_mean", "amplitude_ratio_final",
                                            "decay_exact_ratio", "t_final")}
                   for k, v in stokes.items()},
        "sgs_off_identity": sgs_off,
        "delta_scaling": delta_scaling,
        "acceptance": crit,
    }
    _save("report", report)
    print(json.dumps(crit["summary"], indent=1))
    for k, v in crit["criteria"].items():
        print(f"{k:32s} {v['verdict']:9s} measured={v['measured']} ref={v['reference']} tol={v['tolerance']}")
    return report


def main() -> int:
    # libgomp is not fork-safe: a forked worker that inherits an initialised
    # OpenMP runtime deadlocks in the thread-pool barrier (observed as workers
    # sleeping in futex_wait with ~0% CPU).  "spawn" gives every worker a clean
    # interpreter, which is also what the C kernel expects.
    mp.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["dit", "tgv", "dns", "laminar", "controls", "assemble"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--models", default=",".join(sv.SGS_MODELS))
    ap.add_argument("--sizes", default="")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    if args.arm == "dit":
        sizes = [int(x) for x in args.sizes.split(",")] if args.sizes else [32, 64, 128]
        jobs = [(m, n) for m in models for n in sizes]
        with Pool(min(args.workers, len(jobs))) as pool:
            for msg in pool.imap_unordered(job_dit, jobs):
                print(f"[done] {msg}", flush=True)
    elif args.arm == "tgv":
        sizes = [int(x) for x in args.sizes.split(",")] if args.sizes else [64, 128]
        jobs = [(m, n) for m in models for n in sizes]
        with Pool(min(args.workers, len(jobs))) as pool:
            for msg in pool.imap_unordered(job_tgv, jobs):
                print(f"[done] {msg}", flush=True)
    elif args.arm == "dns":
        sizes = [int(x) for x in args.sizes.split(",")] if args.sizes else [96, 128]
        for n in sizes:
            print(f"[done] {job_dns(n)}", flush=True)
    elif args.arm == "laminar":
        print(f"[done] {job_laminar(0)}", flush=True)
    elif args.arm == "controls":
        print(f"[done] {job_controls(0)}", flush=True)
    else:
        assemble()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
