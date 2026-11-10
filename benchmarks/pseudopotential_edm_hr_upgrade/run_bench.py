"""Pseudopotential force-format upgrade benchmark (campaign node ``p97w3-n04``).

One driver, three stages:

``laplace``   static droplet, radii R = 6/8/10, four force formats ->
              Young-Laplace fit ``dp = 2 sigma / R`` (R^2, per-radius sigma
              consistency) and the spurious-current floor ``max|u|``.
``box``       same physical droplet in three box sizes -> measures the
              periodic-image ("hindered settling") coefficient, used to correct
              the settling velocities for confinement.
``settle``    buoyant droplet settling vs Hadamard-Rybczynski, lambda =
              0.1/1/10 x R = 6/8/10 x {baseline, upgraded} formats, plus a
              body-force (Re -> 0) ladder at R = 8, lambda = 1.

Everything runs as one job per (stage, case); results are written to
``runs/<case>.json`` so a campaign is resumable and partial results survive.
Aggregate with ``--stage report``.

Usage:
    python run_bench.py --stage laplace --jobs 12
    python run_bench.py --stage settle  --jobs 12
    python run_bench.py --stage report
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO / "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tensorlbm.hadamard_rybczynski import hr_velocity  # noqa: E402
from tensorlbm.pseudopotential_edm3d import (  # noqa: E402
    FORMATS,
    richardson_limit,
    settling_droplet,
    static_droplet,
)

RUNS = HERE / "runs"
ART = REPO / "artifacts" / "pseudopotential_edm_hr_upgrade"

#: physical set-up shared by every case
PHYS = {
    "G_eff": -5.0,  # psi_exp coexistence: rho_l ~ 1.957, rho_v ~ 0.1596 (verified)
    "rho_l": 1.957,
    "rho_v": 0.1596,
    "width": 2.0,  # initial tanh half width (the EOS sets the equilibrium one)
    "tau_v": 2.0,  # vapour relaxation time -> mu_v = rho_v cs^2 (tau_v - 1/2)
}

#: format ladder: baseline (production) -> upgraded (EDM + 6th-order gradient)
#: F2_edm_d3q19 is algebraically identical to F1 (the spatial "difference"
#: form is a no-op on symmetric stencils, asserted in the tests), so it is
#: kept in the module but not re-run here.
LADDER = ("F0_classic", "F3_edm_iso6")

#: Hadamard-Rybczynski ladder and resolution ladder
LAMBDAS = (0.1, 1.0, 10.0)
RADII = (6.0, 8.0, 10.0)

U_TARGET = 1.0e-2  # target settling speed: keeps Re ~ 1e-2 while staying well
#                     above the spurious-current floor


def _mu_v() -> float:
    return PHYS["rho_v"] / 3.0 * (PHYS["tau_v"] - 0.5)


def g_for_target(lam: float, R: float) -> float:
    """Body force giving ``U_HR ~ U_TARGET`` at this (lambda, R)."""
    drho = PHYS["rho_l"] - PHYS["rho_v"]
    mu_v = _mu_v()
    return U_TARGET * 3.0 * mu_v * (2.0 + 3.0 * lam) / (2.0 * drho * R ** 2 * (1.0 + lam))


def _path(case: str) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    return RUNS / f"{case}.json"


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def _task(job: dict) -> dict:
    """Run one case; failures are captured (never silently swallowed)."""
    kind = job["kind"]
    case = job["case"]
    out = _path(case)
    if out.exists() and not job.get("force"):
        return {"case": case, "cached": True, "ok": True}
    t0 = time.time()
    try:
        if kind == "laplace":
            fmt = job["fmt"]
            st, form, forcing = FORMATS[fmt]
            res = static_droplet(
                R=job["R"],
                n=job["n"],
                G_eff=PHYS["G_eff"],
                tau=job["tau"],
                stencil=st,
                form=form,
                forcing=forcing,
                rho_l=PHYS["rho_l"],
                rho_v=PHYS["rho_v"],
                width=PHYS["width"],
                steps=job["steps"],
                check_every=job["check_every"],
                thread_count=job["threads"],
            )
        elif kind == "settle":
            fmt = job["fmt"]
            st, form, forcing = FORMATS[fmt]
            res = settling_droplet(
                R=job["R"],
                n=job["n"],
                lam=job["lam"],
                G_eff=PHYS["G_eff"],
                g=job["g"],
                tau_v=PHYS["tau_v"],
                stencil=st,
                form=form,
                forcing=forcing,
                rho_l=PHYS["rho_l"],
                rho_v=PHYS["rho_v"],
                width=PHYS["width"],
                steps=job["steps"],
                sample_every=job["sample_every"],
                thread_count=job["threads"],
            )
        else:
            raise ValueError(f"unknown job kind {kind!r}")
        res["case"] = case
        res["job"] = job
        res["wall_s"] = round(time.time() - t0, 1)
        save(out, res)
        return {"case": case, "ok": True, "wall_s": res["wall_s"]}
    except Exception as exc:  # noqa: BLE001 - honest failure record
        err = {
            "case": case,
            "ok": False,
            "job": job,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "wall_s": round(time.time() - t0, 1),
        }
        save(_path(case + "__FAILED"), err)
        return {"case": case, "ok": False, "error": err["error"]}


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------


def laplace_jobs(threads: int) -> list[dict]:
    jobs = []
    for fmt in LADDER:
        for R in RADII:
            jobs.append(
                dict(
                    kind="laplace",
                    case=f"laplace_{fmt}_R{R:g}",
                    fmt=fmt,
                    R=R,
                    n=int(round(4.0 * R)),  # box 4 R across
                    tau=1.0,
                    steps=1200,
                    check_every=120,
                    threads=threads,
                )
            )
    return jobs


def box_jobs(threads: int) -> list[dict]:
    """Box-size ladder at R = 6, lambda = 1 (upgraded format), for the image term.

    Together with the R = 6 ladder case (n = 26) this gives three box sizes at
    fixed physics, from which the hindrance coefficient is fitted.
    """
    jobs = []
    for n in (36, 44):
        jobs.append(
            dict(
                kind="settle",
                case=f"box_F3_lam1_R6_n{n}",
                fmt="F3_edm_iso6",
                R=6.0,
                n=n,
                lam=1.0,
                g=g_for_target(1.0, 6.0),
                steps=2000,
                sample_every=50,
                threads=threads,
            )
        )
    return jobs


def re_jobs(threads: int) -> list[dict]:
    """Body-force ladder (Re -> 0 extrapolation) at R = 6, lambda = 1."""
    jobs = []
    g0 = g_for_target(1.0, 6.0)
    for frac in (0.4, 1.0, 2.5):
        jobs.append(
            dict(
                kind="settle",
                case=f"re_F3_lam1_R6_g{frac:g}",
                fmt="F3_edm_iso6",
                R=6.0,
                n=26,
                lam=1.0,
                g=g0 * frac,
                steps=2000,
                sample_every=50,
                threads=threads,
            )
        )
    return jobs


def settle_jobs(threads: int) -> list[dict]:
    """HR ladder: upgraded format over lambda x R, plus the baseline at R = 8."""
    jobs = []
    for fmt in ("F3_edm_iso6", "F0_classic"):
        radii = RADII if fmt == "F3_edm_iso6" else (8.0,)
        for lam in LAMBDAS:
            for R in radii:
                g = g_for_target(lam, 8.0) * (8.0 / R) ** 2  # same target U at every R
                jobs.append(
                    dict(
                        kind="settle",
                        case=f"settle_{fmt}_lam{lam:g}_R{R:g}",
                        fmt=fmt,
                        R=R,
                        n=int(round(4.4 * R)),
                        lam=lam,
                        g=g,
                        steps=2000,
                        sample_every=50,
                        threads=threads,
                    )
                )
    return jobs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _load(case: str) -> dict | None:
    p = _path(case)
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def report() -> dict:
    out: dict = {"stages": {}}

    # ---------------- Laplace + spurious currents ----------------
    lap = {}
    for fmt in LADDER:
        rows = []
        for R in RADII:
            d = _load(f"laplace_{fmt}_R{R:g}")
            if d:
                rows.append(d)
        if len(rows) >= 3:
            inv = np.array([1.0 / r["R_eq"] for r in rows])
            dp = np.array([r["dp"] for r in rows])
            A = np.vstack([inv, np.ones_like(inv)]).T
            coef, *_ = np.linalg.lstsq(A, dp, rcond=None)
            pred = A @ coef
            ss_res = float(((dp - pred) ** 2).sum())
            ss_tot = float(((dp - dp.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            sigma_fit = float(0.5 * coef[0])
            per = [
                {
                    "R0": r["R0"],
                    "R_eq": r["R_eq"],
                    "dp": r["dp"],
                    "sigma_i": 0.5 * r["dp"] * r["R_eq"],
                    "dev_pct": 100.0 * (0.5 * r["dp"] * r["R_eq"] - sigma_fit) / sigma_fit,
                    "u_max": r["u_max"],
                    "u_max_iface": r["u_max_iface"],
                    "u_max_core": r.get("u_max_core", float("nan")),
                }
                for r in rows
            ]
            lap[fmt] = {
                "stencil": rows[0]["stencil"],
                "form": rows[0]["form"],
                "forcing": rows[0]["forcing"],
                "sigma_fit": sigma_fit,
                "r2": r2,
                "intercept": float(coef[1]),
                "max_sigma_dev_pct": float(np.max(np.abs([p["dev_pct"] for p in per]))),
                "u_max": float(np.mean([p["u_max"] for p in per])),
                "u_max_iface": float(np.mean([p["u_max_iface"] for p in per])),
                "per_radius": per,
            }
    out["stages"]["laplace"] = lap

    # ---------------- box / confinement ----------------
    box = []
    d0 = _load("settle_F3_edm_iso6_lam1_R6")
    if d0:
        box.append({"n": d0["n"], "U_rel": d0["U_rel"], "U_hr": d0["U_hr"],
                    "phi_third": ((4.0 / 3.0) * math.pi * d0["R_eq"] ** 3 / d0["n"] ** 3)
                    ** (1.0 / 3.0), "R_eq": d0["R_eq"]})
    for n in (36, 44):
        d = _load(f"box_F3_lam1_R6_n{n}")
        if d:
            phi = (4.0 / 3.0) * math.pi * d["R_eq"] ** 3 / n ** 3
            box.append({"n": n, "U_rel": d["U_rel"], "U_hr": d["U_hr"],
                        "phi_third": phi ** (1.0 / 3.0), "R_eq": d["R_eq"]})
    conf = {"rows": box}
    if len(box) >= 2:
        x = np.array([b["phi_third"] for b in box])
        y = np.array([b["U_rel"] for b in box])
        A = np.vstack([np.ones_like(x), -x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        conf.update(
            {
                "U_inf": float(coef[0]),
                "k_hindrance": float(coef[1]),
                "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            }
        )
    out["stages"]["box"] = conf

    # ---------------- settling vs HR ----------------
    settle = {}
    k_h = conf.get("k_hindrance", float("nan"))
    for fmt in ("F0_classic", "F3_edm_iso6"):
        rows = []
        for lam in LAMBDAS:
            for R in RADII:
                d = _load(f"settle_{fmt}_lam{lam:g}_R{R:g}")
                if not d:
                    continue
                phi = (4.0 / 3.0) * math.pi * d["R_eq"] ** 3 / d["n"] ** 3
                corr = 1.0 / (1.0 - k_h * phi ** (1.0 / 3.0)) if math.isfinite(k_h) else float("nan")
                u_corr = abs(d["U_rel"]) * corr
                rows.append(
                    {
                        "lam": lam,
                        "R0": R,
                        "n": d["n"],
                        "R_eq": d["R_eq"],
                        "U_rel": d["U_rel"],
                        "U_hr": d["U_hr"],
                        "err_raw": abs(d["U_rel"]) / abs(d["U_hr"]) - 1.0,
                        "conf_correction": corr,
                        "U_corr": u_corr,
                        "err_conf": u_corr / abs(d["U_hr"]) - 1.0,
                        "Re": d.get("Re_v", float("nan")),
                        "u_max": d["u_max"],
                        "u_max_over_U_hr": d["u_max"] / abs(d["U_hr"]),
                        "steps": d["steps_run"],
                        "finite": d["finite"],
                        # convergence evidence: last sample vs the window mean.  A
                        # settling run should have U_last ~ U_mean; a monotone
                        # increase means the transient (or a spurious drift) has not
                        # decayed and the value must not be read as U_terminal.
                        "U_last": d["history"][-1]["U_rel"],
                        "U_mean": float(
                            np.mean([h["U_rel"] for h in d["history"][len(d["history"]) // 2:]])
                        ),
                        "converged": bool(
                            abs(d["history"][-1]["U_rel"]) / max(abs(d["U_rel"]), 1e-30) < 1.25
                        ),
                    }
                )
        settle[fmt] = {"rows": rows}
        # resolution extrapolation R -> infinity, per lambda
        extrap = {}
        for lam in LAMBDAS:
            sub = [r for r in rows if r["lam"] == lam]
            if len(sub) >= 3:
                h = [1.0 / r["R0"] for r in sub]
                for key in ("U_rel", "U_corr"):
                    y = [abs(r[key]) for r in sub]
                    fit = richardson_limit(h, y)
                    # error of the R -> infinity limit against the analytic law
                    # (evaluated with the mean measured parameters of the ladder)
                    ur = np.mean([r["U_hr"] for r in sub])
                    fit["err_at_limit_pct"] = 100.0 * (fit["y0"] / ur - 1.0)
                    fit["U_hr_ref"] = float(ur)
                    extrap.setdefault(key, {})[f"lam{lam:g}"] = fit
                # error at the finest resolution (R = 10) and the R = 16 value
                # implied by the fitted 1/R model
                p = extrap["U_corr"][f"lam{lam:g}"]["p"]
                amp = extrap["U_corr"][f"lam{lam:g}"]["amplitude"]
                y0 = extrap["U_corr"][f"lam{lam:g}"]["y0"]
                ur = extrap["U_corr"][f"lam{lam:g}"]["U_hr_ref"]
                extrap.setdefault("err_vs_R", {})[f"lam{lam:g}"] = {
                    str(int(r["R0"])): 100.0 * (abs(r["U_corr"]) / abs(r["U_hr"]) - 1.0)
                    for r in sub
                } | {
                    "16": 100.0 * ((y0 + amp * (1.0 / 16.0) ** p) / ur - 1.0),
                    "inf": 100.0 * (y0 / ur - 1.0),
                }
        settle[fmt]["extrapolation"] = extrap
    out["stages"]["settle"] = settle

    # ---------------- Re -> 0 ladder ----------------
    re_rows = []
    for frac in (0.4, 1.0, 2.5):
        d = _load(f"re_F3_lam1_R6_g{frac:g}")
        if not d:
            continue
        phi = (4.0 / 3.0) * math.pi * d["R_eq"] ** 3 / d["n"] ** 3
        corr = 1.0 / (1.0 - k_h * phi ** (1.0 / 3.0)) if math.isfinite(k_h) else 1.0
        re_rows.append(
            {
                "g_frac": frac,
                "g": d["g"],
                "U_rel": d["U_rel"],
                "U_corr": abs(d["U_rel"]) * corr,
                "U_hr": d["U_hr"],
                "Re": d.get("Re_v", float("nan")),
                "err_conf": abs(d["U_rel"]) * corr / abs(d["U_hr"]) - 1.0,
            }
        )
    re_block = {"rows": sorted(re_rows, key=lambda r: r["g_frac"])}
    if len(re_rows) >= 3:
        x = np.array([abs(r["Re"]) for r in re_block["rows"]])
        y = np.array([r["U_corr"] for r in re_block["rows"]])
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        re_block["U_Re0"] = float(coef[0])
        re_block["slope"] = float(coef[1])
        ur = float(np.mean([r["U_hr"] for r in re_block["rows"]]))
        re_block["err_at_Re0_pct"] = 100.0 * (coef[0] / ur - 1.0)
        re_block["U_hr_ref"] = ur
    out["stages"]["re_to_zero"] = re_block

    save(ART / "summary.json", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="laplace",
                    choices=["laplace", "settle", "box", "re", "all", "report"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.stage == "report":
        out = report()
        print(json.dumps({k: (list(v) if isinstance(v, dict) else v)
                          for k, v in out["stages"].items()}, indent=2)[:4000])
        return 0

    jobs: list[dict] = []
    if args.stage in ("laplace", "all"):
        jobs += laplace_jobs(args.threads)
    if args.stage in ("settle", "all"):
        jobs += settle_jobs(args.threads)
    if args.stage in ("box", "all"):
        jobs += box_jobs(args.threads)
    if args.stage in ("re", "all"):
        jobs += re_jobs(args.threads)
    if args.force:
        for j in jobs:
            j["force"] = True

    print(f"{len(jobs)} cases, {args.jobs} workers", flush=True)
    t0 = time.time()
    with mp.Pool(args.jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(_task, jobs), 1):
            print(f"[{i}/{len(jobs)}] {res}", flush=True)
    print(f"done in {round(time.time() - t0, 1)} s", flush=True)

    if args.stage == "all":
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())