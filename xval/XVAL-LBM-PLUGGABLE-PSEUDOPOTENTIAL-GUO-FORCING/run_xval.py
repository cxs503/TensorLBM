"""Cross-validation driver: pluggable pseudopotential forcing, Guo scheme.

Anchor: ``XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING``.
Acceptance (owner precision standard): relative error **< 15 %** against the
external anchor; partial results are reported honestly, never extrapolated.

Arms
----
``repo``  the repository engine under test:
          ``tensorlbm.multiphase.collide_sc_single_component`` with
          ``forcing in {"guo", "edm", "velocity_shift"}`` (torch, float32 default).
``ref``   the independent numpy transcription in :mod:`ref_sc_numpy`.

Anchors (no fitting, no free parameters)
----------------------------------------
A1  Maxwell (equal-area) coexistence densities of the SC equation of state
    (:func:`ref_sc_numpy.maxwell_phase`) -- analytic, one-dimensional, shares
    no code with either lattice arm.
A2  Young-Laplace ``dp = sigma / R`` for a 2-D droplet: linearity of ``dp`` vs
    ``1/R`` over three radii, ``R^2`` of the fit and the relative size of the
    fitted intercept (a non-zero intercept means the measured ``dp`` is not a
    pure Laplace jump).
A3  Momentum-injection identity ``sum_i c_i (f_i^out - f_i^BGK) = F`` that every
    consistent forcing scheme must satisfy exactly (machine precision).
A4  Forcing-scheme independence at fixed ``(G, psi, tau -> geometry)``: the
    *static* droplet solution (bulk densities, ``sigma``) must not depend on
    which pluggable forcing scheme was selected, and must not depend on the
    relaxation time ``tau`` -- ``tau`` sets viscosity only.  The historical
    velocity-shift scheme is expected (Guo 2002; Kupershtokh et al. 2009;
    Krueger et al., *The Lattice Boltzmann Method*, Sec. 9.3) to violate the
    ``tau``-independence because it injects ``tau F`` instead of ``F`` of
    momentum per step at the interface.

Usage
-----
    python run_xval.py --list
    python run_xval.py --all                     # sequential, all cases
    python run_xval.py --shard i/n --out runs    # one slice of the matrix
    python run_xval.py --analyze                 # metrics + figure + results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

import ref_sc_numpy as ref  # noqa: E402

G_LIB = 5.0  # library sign (>0 = attractive); G_eff = -5 in the physical convention
PSI_RHO0 = 1.0
STEPS = 24000
CHECKPOINT = 16000  # dp/u_max must be steady between CHECKPOINT and STEPS

# --------------------------------------------------------------------------
# case matrix
# --------------------------------------------------------------------------


def build_cases() -> list[dict]:
    """The full pre-declared matrix (nothing is added after looking)."""
    cases: list[dict] = []
    # (1) forcing axis x radius axis, the primary Laplace / Maxwell measurement
    for scheme in ("guo", "edm", "velocity_shift"):
        for R in (10.0, 14.0, 18.0):
            cases.append(dict(name=f"repo_{scheme}_R{int(R)}_tau1.0", arm="repo", scheme=scheme,
                              tau=1.0, L=128, R=R, steps=STEPS))
    # (2) independent numpy arm on the same three radii, Guo only
    for R in (10.0, 14.0, 18.0):
        cases.append(dict(name=f"ref_guo_R{int(R)}_tau1.0", arm="ref", scheme="guo",
                          tau=1.0, L=128, R=R, steps=STEPS))
    # (3) tau sensitivity (scheme-independence of the static solution); ref arm
    for scheme in ("guo", "edm", "velocity_shift"):
        for tau in (0.8, 1.2):
            cases.append(dict(name=f"ref_{scheme}_R14_tau{tau}", arm="ref", scheme=scheme,
                              tau=tau, L=128, R=14.0, steps=STEPS))
    # (4) resolution / box sensitivity at fixed physical ratio (ref arm)
    cases.append(dict(name="ref_guo_R21_L192_tau1.0", arm="ref", scheme="guo", tau=1.0,
                      L=192, R=21.0, steps=STEPS))
    cases.append(dict(name="ref_guo_R7_L64_tau1.0", arm="ref", scheme="guo", tau=1.0,
                      L=64, R=7.0, steps=STEPS))
    for i, c in enumerate(cases):
        c["index"] = i
    return cases


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


def run_repo(case: dict) -> dict:
    import torch

    torch.set_num_threads(int(os.environ.get("XVAL_THREADS", "1")))
    from tensorlbm.d2q9 import equilibrium as t_equilibrium
    from tensorlbm.d2q9 import macroscopic as t_macroscopic
    from tensorlbm.multiphase import collide_sc_single_component, psi_exp
    from tensorlbm.solver import stream as t_stream

    mw = ref.maxwell_phase(G_LIB)
    rho0 = ref.droplet_rho(case["L"], case["R"], mw["rho_l"], mw["rho_v"])
    f = t_equilibrium(torch.tensor(rho0, dtype=torch.float32),
                      torch.zeros(case["L"], case["L"]), torch.zeros(case["L"], case["L"]))
    mass0 = float(torch.tensor(rho0).sum())
    series = []
    t_start = time.time()
    for n in range(1, case["steps"] + 1):
        f = collide_sc_single_component(f, G=G_LIB, tau=case["tau"], psi_fn=psi_exp,
                                        forcing=case["scheme"])
        f = t_stream(f)
        if n % 2000 == 0 or n == case["steps"]:
            rho, ux, uy = t_macroscopic(f)
            r = rho.double().numpy()
            uv = torch.sqrt(ux.double() ** 2 + uy.double() ** 2)
            m = ref.measure_droplet(r, G_LIB, ux=ux.double().numpy(), uy=uy.double().numpy())
            series.append(dict(step=n, dp=m["dp"], R_eq=m["R_eq"], r_in=m["r_in"],
                               r_out=m["r_out"], u_max=m["u_max"],
                               mass=float(r.sum()), u_max_all=float(uv.max())))
    rho, ux, uy = t_macroscopic(f)
    m = ref.measure_droplet(rho.double().numpy(), G_LIB,
                            ux=ux.double().numpy(), uy=uy.double().numpy())
    m.pop("rho_field", None)
    return dict(final=m, series=series, mass_drift=float(rho.double().sum()) - mass0,
                wall_s=time.time() - t_start)


def run_ref(case: dict) -> dict:
    mw = ref.maxwell_phase(G_LIB)
    rho0 = ref.droplet_rho(case["L"], case["R"], mw["rho_l"], mw["rho_v"])
    f = ref.equilibrium(rho0, np.zeros_like(rho0), np.zeros_like(rho0))
    mass0 = float(rho0.sum())
    series = []
    t_start = time.time()
    for n in range(1, case["steps"] + 1):
        f = ref.step(f, G_LIB, case["tau"], case["scheme"])
        if n % 2000 == 0 or n == case["steps"]:
            rho, ux, uy = ref.macroscopic(f)
            m = ref.measure_droplet(rho, G_LIB, ux=ux, uy=uy)
            series.append(dict(step=n, dp=m["dp"], R_eq=m["R_eq"], r_in=m["r_in"],
                               r_out=m["r_out"], u_max=m["u_max"], mass=float(rho.sum())))
    rho, ux, uy = ref.macroscopic(f)
    m = ref.measure_droplet(rho, G_LIB, ux=ux, uy=uy)
    m.pop("rho_field", None)
    return dict(final=m, series=series, mass_drift=float(rho.sum()) - mass0,
                wall_s=time.time() - t_start)


def run_case(case: dict, outdir: str) -> str:
    path = os.path.join(outdir, case["name"] + ".json")
    if os.path.exists(path):
        return path
    t0 = time.time()
    res = run_repo(case) if case["arm"] == "repo" else run_ref(case)
    rec = dict(case=case, result=res, maxwell=ref.maxwell_phase(G_LIB),
               wall_s=time.time() - t0, host=os.uname().nodename)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[done] {case['name']:32s} {rec['wall_s']:7.1f}s  dp={res['final']['dp']:+.5f} "
          f"R_eq={res['final']['R_eq']:6.2f} u_max={res['final']['u_max']:.3e}", flush=True)
    return path


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

REL_TOL = 0.15  # owner precision standard


def rel_err(got: float, refv: float) -> float:
    return abs(got - refv) / abs(refv)


def linfit(x: np.ndarray, y: np.ndarray) -> dict:
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ np.array([slope, intercept])
    ss_res = float(((y - yh) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(slope=float(slope), intercept=float(intercept),
                r2=1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0)


def analyse(outdir: str) -> dict:
    recs = {}
    for fn in sorted(os.listdir(outdir)):
        if fn.endswith(".json"):
            with open(os.path.join(outdir, fn)) as fh:
                r = json.load(fh)
            recs[r["case"]["name"]] = r

    mw = ref.maxwell_phase(G_LIB)
    out: dict = {"anchor": {"maxwell": {k: v for k, v in mw.items() if k != "residual"},
                            "G_lib": G_LIB, "rel_tol": REL_TOL},
                 "cases": {k: {"final": v["result"]["final"],
                               "mass_drift": v["result"]["mass_drift"]}
                           for k, v in recs.items()}}

    # ---- A1: Maxwell coexistence densities -------------------------------
    maxwell_rows = {}
    for name, r in recs.items():
        f = r["result"]["final"]
        maxwell_rows[name] = dict(
            rho_in=f["r_in"], rho_out=f["r_out"],
            err_liquid=rel_err(f["r_in"], mw["rho_l"]),
            err_vapour=rel_err(f["r_out"], mw["rho_v"]),
        )
    out["maxwell_check"] = maxwell_rows

    # ---- A2: Laplace linearity per (arm, scheme, tau) --------------------
    laplace = {}
    for name, r in recs.items():
        c = r["case"]
        key = f"{c['arm']}:{c['scheme']}:tau={c['tau']}"
        laplace.setdefault(key, []).append((c["R"], r["result"]["final"]["R_eq"],
                                           r["result"]["final"]["dp"]))
    laplace_out = {}
    for key, rows in laplace.items():
        if len(rows) < 2:
            continue
        Rnom = np.array([r[0] for r in rows], dtype=float)
        Req = np.array([r[1] for r in rows], dtype=float)
        dp = np.array([r[2] for r in rows], dtype=float)
        fit = linfit(1.0 / Req, dp)
        sigmas = dp * Req
        laplace_out[key] = dict(
            points=[dict(R_nominal=float(a), R_eq=float(b), dp=float(d)) for a, b, d in rows],
            sigma_from_slope=fit["slope"], r2=fit["r2"], intercept=fit["intercept"],
            intercept_over_dp_max=float(abs(fit["intercept"]) / np.abs(dp).max()),
            sigma_per_radius=[float(s) for s in sigmas],
            sigma_spread=float((sigmas.max() - sigmas.min()) / sigmas.mean()),
            consistent_15pct=bool((sigmas.max() - sigmas.min()) / sigmas.mean() < REL_TOL),
        )
    out["laplace"] = laplace_out

    # ---- A4: scheme / tau independence ----------------------------------
    def sigma_of(key):
        v = laplace_out.get(key)
        return None if v is None else v["sigma_from_slope"]

    scheme_cmp = {}
    base = {s: sigma_of(f"ref:{s}:tau=1.0") for s in ref.SCHEMES}
    if all(v is not None for v in base.values()):
        refv = base["guo"]
        scheme_cmp["tau1.0_vs_guo"] = {s: dict(
            sigma=base[s], rel_diff=rel_err(base[s], refv) if s != "guo" else 0.0,
            within_15pct=bool(rel_err(base[s], refv) < REL_TOL) if s != "guo" else True)
            for s in ref.SCHEMES}
    tau_cmp = {}
    for s in ref.SCHEMES:
        s08, s10, s12 = (sigma_of(f"ref:{s}:tau=0.8"), sigma_of(f"ref:{s}:tau=1.0"),
                         sigma_of(f"ref:{s}:tau=1.2"))
        if None not in (s08, s10, s12):
            tau_cmp[s] = dict(
                sigma_tau_08=s08, sigma_tau_10=s10, sigma_tau_12=s12,
                max_rel_dev_from_tau1=max(rel_err(s08, s10), rel_err(s12, s10)),
                tau_independent_15pct=bool(max(rel_err(s08, s10), rel_err(s12, s10)) < REL_TOL),
            )
    out["scheme_independence"] = scheme_cmp
    out["tau_independence"] = tau_cmp

    # ---- cross-arm agreement (repo torch vs numpy) ----------------------
    cross = {}
    for R in (10, 14, 18):
        a = recs.get(f"repo_guo_R{R}_tau1.0")
        b = recs.get(f"ref_guo_R{R}_tau1.0")
        if a and b:
            fa, fb = a["result"]["final"], b["result"]["final"]
            cross[f"R{R}"] = dict(
                dp_repo=fa["dp"], dp_ref=fb["dp"], dp_rel_diff=rel_err(fa["dp"], fb["dp"]),
                Req_repo=fa["R_eq"], Req_ref=fb["R_eq"],
                Req_rel_diff=rel_err(fa["R_eq"], fb["R_eq"]),
                u_max_repo=fa["u_max"], u_max_ref=fb["u_max"],
                within_15pct=bool(rel_err(fa["dp"], fb["dp"]) < REL_TOL),
            )
    out["cross_arm"] = cross

    # ---- steadiness (A2 pre-requisite) ---------------------------------
    steadiness = {}
    for name, r in recs.items():
        ser = r["result"]["series"]
        if len(ser) >= 2:
            a = [s for s in ser if s["step"] == CHECKPOINT]
            b = ser[-1]
            if a:
                steadiness[name] = dict(
                    dp_at_checkpoint=a[0]["dp"], dp_final=b["dp"],
                    dp_drift_rel=rel_err(b["dp"], a[0]["dp"]),
                    rho_in_drift=abs(b["r_in"] - a[0]["r_in"]),
                )
    out["steadiness"] = steadiness
    out["summary_md"] = _summary_md(out, recs)
    return out


def _summary_md(out: dict, recs: dict) -> str:
    """Programmatic markdown tables (zero hand-copy of any number)."""
    L = []
    a = out["anchor"]["maxwell"]
    L.append(f"Maxwell anchor (analytic): rho_l={a['rho_l']:.5f} rho_v={a['rho_v']:.5f} "
             f"ratio={a['rho_l']/a['rho_v']:.2f} p*={a['p_star']:.5f}\n")
    L.append("| case | rho_in | err_l vs Maxwell | rho_out | err_v vs Maxwell |")
    L.append("|---|---|---|---|---|")
    for k, v in sorted(out["maxwell_check"].items()):
        L.append(f"| {k} | {v['rho_in']:.4f} | {100*v['err_liquid']:.2f}% | "
                 f"{v['rho_out']:.4f} | {100*v['err_vapour']:.2f}% |")
    L.append("\n| arm:scheme:tau | sigma (slope) | R^2 | intercept/dp_max | sigma spread | <15% |")
    L.append("|---|---|---|---|---|---|")
    for k, v in sorted(out["laplace"].items()):
        L.append(f"| {k} | {v['sigma_from_slope']:.5f} | {v['r2']:.5f} | "
                 f"{100*v['intercept_over_dp_max']:.2f}% | {100*v['sigma_spread']:.2f}% | "
                 f"{'YES' if v['consistent_15pct'] else 'NO'} |")
    if out["scheme_independence"]:
        L.append("\n| scheme (tau=1) | sigma | rel diff vs guo | <15% |")
        L.append("|---|---|---|---|")
        for k, v in out["scheme_independence"]["tau1.0_vs_guo"].items():
            L.append(f"| {k} | {v['sigma']:.5f} | {100*v['rel_diff']:.2f}% | "
                     f"{'YES' if v['within_15pct'] else 'NO'} |")
    if out["tau_independence"]:
        L.append("\n| scheme | sigma(tau=0.8) | sigma(1.0) | sigma(1.2) | max dev | tau-independent |")
        L.append("|---|---|---|---|---|---|")
        for k, v in out["tau_independence"].items():
            L.append(f"| {k} | {v['sigma_tau_08']:.5f} | {v['sigma_tau_10']:.5f} | "
                     f"{v['sigma_tau_12']:.5f} | {100*v['max_rel_dev_from_tau1']:.2f}% | "
                     f"{'YES' if v['tau_independent_15pct'] else 'NO'} |")
    if out["cross_arm"]:
        L.append("\n| R | dp repo | dp ref | rel diff | R_eq rel diff | <15% |")
        L.append("|---|---|---|---|---|---|")
        for k, v in out["cross_arm"].items():
            L.append(f"| {k} | {v['dp_repo']:.5f} | {v['dp_ref']:.5f} | "
                     f"{100*v['dp_rel_diff']:.2f}% | {100*v['Req_rel_diff']:.2f}% | "
                     f"{'YES' if v['within_15pct'] else 'NO'} |")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs"))
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--maxwell-only", action="store_true")
    args = ap.parse_args()

    cases = build_cases()
    if args.list:
        for c in cases:
            print(c["index"], c["name"], c["arm"], c["scheme"], "tau", c["tau"],
                  "L", c["L"], "R", c["R"], "steps", c["steps"])
        print("total", len(cases))
        return 0
    if args.maxwell_only:
        print(json.dumps(ref.maxwell_phase(G_LIB), indent=1))
        return 0
    os.makedirs(args.out, exist_ok=True)
    if args.analyze:
        res = analyse(args.out)
        with open(os.path.join(HERE, "results.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        with open(os.path.join(HERE, "summary.md"), "w") as fh:
            fh.write(res["summary_md"])
        print(json.dumps({k: res[k] for k in ("scheme_independence", "tau_independence",
                                              "cross_arm")}, indent=1))
        return 0

    sel = cases
    if args.shard:
        i, n = args.shard
        sel = [c for c in cases if c["index"] % n == i]
    if args.all:
        sel = cases
    for c in sel:
        run_case(c, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())