#!/usr/bin/env python
"""Aggregate + analyse + plot the m183 independent sweep of B23 laplace_droplet.

Inputs (written by ``sweep_m183.py``, one job per (dim, R)):
    runs/m183_laplace/{2d,3d}_R*/run.json / profile.json / snapshot.npy

Outputs
    artifacts/m183_laplace_droplet/sweep_result.json   (all numbers, programmatic)
    artifacts/m183_laplace_droplet/fig*.png            (report figures)

Criterion parity: the Laplace fit / pass-fail logic is the case's own
``fit_laplace`` (imported from ``run.py``), applied to *our* rows.  Nothing is
hand-copied; every number in the report is read back from this JSON.

Extra physics check added here (not in the case): the *mechanical* (pressure
tensor) route for the surface tension of the 2D droplets,

    sigma_mech = \\int_0^inf [ P_rr(r) - P_tt(r) ] dr ,
    P_ab = rho c_s^2 delta_ab + (G_eff/2) psi(x) sum_i w_i psi(x+e_i) e_ia e_ib ,

which reduces to the SC EOS p = rho/3 + G_eff psi^2/6 in the isotropic limit
(the benchmark's own EOS convention, G_eff = -5).  This is independent of the
Young-Laplace measurement (it does not use Delta p nor R), so sigma_mech vs
sigma_Laplace is a genuine cross-check of the Laplace-law extraction.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = REPO / "runs" / "m183_laplace"
OUT = REPO / "artifacts" / "m183_laplace_droplet"
OUT.mkdir(parents=True, exist_ok=True)


def load_case():
    spec = importlib.util.spec_from_file_location("laplace_bench_case", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CASE = load_case()
G_EFF = CASE.G_EFF


# --------------------------------------------------------------------------- #
# mechanical (pressure-tensor) route for sigma  -- 2D snapshots, D2Q9
# --------------------------------------------------------------------------- #
def sigma_mech_2d(rho: torch.Tensor) -> dict:
    """sigma = int (P_rr - P_tt) dr for a circular droplet (D2Q9, psi_exp)."""
    L = rho.shape[0]
    psi = CASE.psi_exp(rho.clamp(min=1e-8))
    # D2Q9 velocity set / weights (same as the library's)
    ex = torch.tensor([0, 1, -1, 0, 0, 1, -1, 1, -1], dtype=torch.float32)
    ey = torch.tensor([0, 0, 0, 1, -1, 1, 1, -1, -1], dtype=torch.float32)
    w = torch.tensor([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
                     dtype=torch.float32)
    Pxx = rho / 3.0
    Pyy = rho / 3.0
    Pxy = torch.zeros_like(rho)
    for i in range(9):
        psi_shift = torch.roll(torch.roll(psi, -int(ex[i].item()), dims=1),
                               -int(ey[i].item()), dims=0)
        coef = 0.5 * G_EFF * w[i] * psi * psi_shift
        Pxx = Pxx + coef * ex[i] ** 2
        Pyy = Pyy + coef * ey[i] ** 2
        Pxy = Pxy + coef * ex[i] * ey[i]
    ys = torch.arange(L, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    dx, dy = xx - L / 2.0, yy - L / 2.0
    r = torch.sqrt(dx**2 + dy**2).clamp(min=1e-6)
    nx, ny = dx / r, dy / r
    P_rr = Pxx * nx**2 + Pyy * ny**2 + 2.0 * Pxy * nx * ny
    P_tt = Pxx * ny**2 + Pyy * nx**2 - 2.0 * Pxy * nx * ny
    dP = (P_rr - P_tt)
    # shell average (dr = 1 lattice unit) and cumulative integral
    dr = 1.0
    edges = torch.arange(0.0, L / 2.0 + dr, dr)
    idx = torch.clamp((r / dr).long(), max=len(edges) - 2)
    nb = len(edges) - 1
    s = torch.zeros(nb, dtype=torch.float64).index_add_(0, idx.reshape(-1), dP.double().reshape(-1))
    c = torch.zeros(nb, dtype=torch.float64).index_add_(
        0, idx.reshape(-1), torch.ones(idx.numel(), dtype=torch.float64))
    prof = (s / c.clamp(min=1)).numpy()
    rc = ((edges[:-1] + edges[1:]) / 2.0).numpy()
    sigma = float(np.sum(prof) * dr)
    # half-width where the integrand has decayed (report the integrand range)
    peak = np.abs(prof).max()
    mask = np.abs(prof) > 0.02 * peak
    return {"sigma_mech": sigma, "r": rc.tolist(), "dP": prof.tolist(),
            "support_r": [float(rc[mask].min()), float(rc[mask].max())] if mask.any() else None}


def main() -> None:
    stored = json.loads((HERE / "result.json").read_text(encoding="utf-8"))
    out: dict = {
        "mission": "m183",
        "case": "benchmarks/verified/laplace_droplet",
        "criterion": ("2D dp=sigma/R, 3D dp=2*sigma/R; per-radius sigma within 3% of "
                      "origin-forced fit; R2>=0.999; |intercept|<=3% of slope/Rmax"),
        "driver": "benchmarks/verified/laplace_droplet/sweep_m183.py (independent loop, "
                  "case physics helpers imported, no edits to run.py/result.json)",
        "dims": {},
        "per_run": [],
    }

    for dim, tag in ((2, "2d"), (3, "3d")):
        runs = []
        for d in sorted(RUNS.glob(f"{tag}_R*"), key=lambda p: float(p.name.split("_R")[1])):
            rj = d / "run.json"
            if not rj.exists():
                continue
            r = json.loads(rj.read_text(encoding="utf-8"))
            r["_dir"] = str(d)
            runs.append(r)
            out["per_run"].append({
                "dim": dim, "R_init": r["meta"]["R_init"], "L": r["meta"]["L"],
                "step": r["final"]["step"], "stage": r["final"]["stage"],
                "converged": r["final"]["converged"],
                "elapsed_s": r["final"]["elapsed_s"],
                "ms_per_step": r["final"]["ms_per_step"],
                "dp": r["final"].get("dp"), "dp_std": r["final"].get("dp_std"),
                "R_eq": r["final"].get("R_eq"), "R_eq_std": r["final"].get("R_eq_std"),
                "rho_in": r["final"].get("rho_in"), "rho_out": r["final"].get("rho_out"),
                "max_u": r["final"].get("max_u"),
                "mass_drift": r["final"].get("mass_drift"),
                "sigma_i_run": r["final"].get("sigma_i_this_run"),
                "band_check": r["final"].get("band_check"),
                "selfcheck_vs_case_measure_relerr":
                    r["final"].get("selfcheck_vs_case_measure_relerr"),
            })
        # only converged runs enter the Laplace fit; everything else is listed as
        # excluded with its reason (honest reporting: a truncated run must not
        # silently bend the sigma vs 1/R regression).
        skipped = []
        for r in runs:
            fin = r["final"]
            reason = None
            if not fin.get("dp") or not np.isfinite(fin.get("dp", float("nan"))) or not fin.get("R_eq"):
                reason = "no_steady_measurement"
            elif not fin.get("converged"):
                reason = f"not_converged(stage={fin.get('stage')}, steps={fin.get('step')})"
            if reason:
                skipped.append({"R_init": r["meta"]["R_init"], "reason": reason,
                                "steps": fin.get("step"), "dp": fin.get("dp")})
        usable = [r for r in runs if r["final"].get("dp") and np.isfinite(r["final"]["dp"])
                  and r["final"].get("R_eq") and r["final"].get("converged")]
        aborted = [str(d.name) for d in sorted(RUNS.glob(f"{tag}_R*"))
                   if (d / "history.json").exists() and not (d / "run.json").exists()]
        out.setdefault("excluded_runs", []).extend(
            [dict(s_, dim=dim) for s_ in skipped] + [{"dir": a, "dim": dim, "reason": "aborted_no_run_json"}
                                                     for a in aborted])
        if len(usable) < 2:
            out["dims"][tag] = {"status": "insufficient_radii", "n_runs": len(runs)}
            continue
        rows = [r["row"] for r in usable]
        fit = CASE.fit_laplace(rows, dim)
        fit["dim"] = dim
        fit["n_radii"] = len(usable)
        fit["radii"] = [r["meta"]["R_init"] for r in usable]
        fit["all_converged"] = all(r["row"]["converged"] for r in usable)
        # ---- comparison with the stored (verified) baseline ----
        base = stored["dims"][f"{dim}d"]
        cmp_base = {
            "sigma_eff_stored": base["sigma_eff_fit"],
            "sigma_eff_this_run": fit["sigma_eff_fit"],
            "sigma_eff_rel_diff_pct": abs(fit["sigma_eff_fit"] - base["sigma_eff_fit"])
            / base["sigma_eff_fit"] * 100.0,
            "stored_max_dev_sigma_pct": base["max_dev_sigma_pct"],
            "this_run_max_dev_sigma_pct": fit["max_dev_sigma_pct"],
            "stored_R2": base["R2"], "this_run_R2": fit["R2"],
            "matched_radii": [],
        }
        base_by_R = {float(p["R_init"]): p for p in base["per_radius"]}
        for p in fit["per_radius"]:
            b = base_by_R.get(float(p["R_init"]))
            if b is None:
                continue
            cmp_base["matched_radii"].append({
                "R_init": p["R_init"],
                "sigma_i_this_run": p["sigma_i"], "sigma_i_stored": b["sigma_i"],
                "rel_diff_pct": abs(p["sigma_i"] - b["sigma_i"]) / b["sigma_i"] * 100.0,
                "dp_this_run": p["dp"], "dp_stored": b["dp"],
                "R_eq_this_run": p["R_eq"], "R_eq_stored": b["R_eq"],
                "R_eq_rel_diff_pct": abs(p["R_eq"] - b["R_eq"]) / b["R_eq"] * 100.0,
            })
        # how much does sigma_i itself drift across the available radius range, and
        # how large is the 1/R lever arm?  The |intercept| criterion is very sensitive
        # to the lever arm, so this is reported next to the pass/fail flag.
        sig = [p_["sigma_i"] for p_ in fit["per_radius"]]
        fit["sigma_drift_across_radii_pct"] = (max(sig) - min(sig)) / min(sig) * 100.0
        fit["inv_R_lever_arm"] = [min(1.0 / p_["R_eq"] for p_ in fit["per_radius"]),
                                  max(1.0 / p_["R_eq"] for p_ in fit["per_radius"])]
        base_sig = [p_["sigma_i"] for p_ in base["per_radius"]]
        fit["baseline_sigma_drift_pct"] = (max(base_sig) - min(base_sig)) / min(base_sig) * 100.0
        fit["baseline_inv_R_lever_arm"] = [
            min(1.0 / p_["R_eq"] for p_ in base["per_radius"]),
            max(1.0 / p_["R_eq"] for p_ in base["per_radius"])]
        # trend comparison: fit the *baseline* (stored) dP vs 1/R line and predict dP
        # at our radii -> tests whether the two runs share the same sigma(R) curve
        # even when the radius windows barely overlap (3D case).
        inv_r_b = np.array([1.0 / p_["R_eq"] for p_ in base["per_radius"]])
        y_b = np.array([p_["dp"] for p_ in base["per_radius"]])
        if dim == 3:
            y_b = y_b / 2.0
        A_b = np.vstack([inv_r_b, np.ones_like(inv_r_b)]).T
        a_b, b_b = np.linalg.lstsq(A_b, y_b, rcond=None)[0]
        trend = []
        for p_ in fit["per_radius"]:
            y_ours = p_["dp"] if dim == 2 else p_["dp"] / 2.0
            y_pred = a_b / p_["R_eq"] + b_b
            trend.append({"R_init": p_["R_init"], "R_eq": p_["R_eq"],
                          "y_this_run": float(y_ours), "y_pred_baseline_trend": float(y_pred),
                          "rel_diff_pct": abs(y_ours - y_pred) / abs(y_pred) * 100.0})
        cmp_base["baseline_trend_fit"] = {"slope": float(a_b), "intercept": float(b_b)}
        cmp_base["trend_prediction"] = trend
        cmp_base["max_trend_rel_diff_pct"] = max(t["rel_diff_pct"] for t in trend)
        fit["vs_stored_result_json"] = cmp_base
        fit["band_sensitivity_pct"] = [
            {"R_init": r["meta"]["R_init"],
             "dp_rel_diff_pct": r["final"]["band_check"]["dp_rel_diff_pct"],
             "R_eq_rel_diff_pct": r["final"]["band_check"]["R_eq_rel_diff_pct"]}
            for r in usable if r["final"].get("band_check")]
        fit["history"] = {str(r["meta"]["R_init"]): r["history"] for r in usable}
        # ---- mechanical-route cross-check (2D only: full field snapshots) ----
        if dim == 2:
            mech = []
            for r in usable:
                snap = Path(r["_dir"]) / "snapshot.npy"
                if not snap.exists():
                    continue
                rho = torch.from_numpy(np.load(snap))
                mres = sigma_mech_2d(rho)
                sig_lap = (r["final"]["dp"] * r["final"]["R_eq"])  # 2D: dp*R
                mech.append({
                    "R_init": r["meta"]["R_init"], "R_eq": r["final"]["R_eq"],
                    "sigma_mech": mres["sigma_mech"], "sigma_laplace_i": sig_lap,
                    "rel_diff_pct": abs(mres["sigma_mech"] - sig_lap) / sig_lap * 100.0,
                    "integrand_support_r": mres["support_r"],
                    "_profile": {"r": mres["r"], "dP": mres["dP"]},
                })
            fit["mechanical_route_2d"] = {"method": "sigma=int(P_rr-P_tt)dr, G_eff=-5",
                                          "per_radius": mech}
            if len(mech) >= 2:
                fit["mechanical_route_2d"]["sigma_mech_Rmax"] = mech[-1]["sigma_mech"]
                fit["mechanical_route_2d"]["rel_diff_vs_fit_pct"] = (
                    abs(mech[-1]["sigma_mech"] - fit["sigma_eff_fit"])
                    / fit["sigma_eff_fit"] * 100.0)
        out["dims"][tag] = fit

    # ------------------------------------------- domain-size independence sweep
    dom_runs = []
    for d in sorted(RUNS.parent.joinpath("m183_laplace_domain").glob("*_df*")):
        rj = d / "run.json"
        if not rj.exists():
            continue
        r = json.loads(rj.read_text(encoding="utf-8"))
        fin = r["final"]
        if not fin.get("dp") or not fin.get("converged"):
            continue
        sigma_i = fin["dp"] * fin["R_eq"] if r["meta"]["dim"] == 2 else fin["dp"] * fin["R_eq"] / 2
        dom_runs.append({
            "dim": r["meta"]["dim"], "R_init": r["meta"]["R_init"],
            "domain_factor": r["meta"]["domain_factor"], "L": r["meta"]["L"],
            "steps": fin["step"], "dp": fin["dp"], "R_eq": fin["R_eq"],
            "rho_in": fin.get("rho_in"), "rho_out": fin.get("rho_out"),
            "sigma_i": sigma_i, "elapsed_s": fin["elapsed_s"],
        })
    if dom_runs:
        base_sigma = {}
        for tag, dim in (("2d", 2), ("3d", 3)):
            d = out["dims"].get(tag, {})
            for p_ in d.get("per_radius", []):
                base_sigma[(dim, float(p_["R_init"]))] = p_["sigma_i"]
        for row in dom_runs:
            b = base_sigma.get((row["dim"], float(row["R_init"])))
            row["sigma_i_at_domain_factor_4"] = b
            row["rel_diff_vs_df4_pct"] = (abs(row["sigma_i"] - b) / b * 100.0
                                         if b else None)
        out["domain_independence"] = {
            "note": ("same radius, different periodic-box size L = domain_factor*R; the 4R "
                     "box is the case default, so the deviation is the periodic-image "
                     "sensitivity of sigma -- an uncertainty the stored result does not report"),
            "runs": dom_runs,
            "max_abs_diff_pct": max(r["rel_diff_vs_df4_pct"] for r in dom_runs
                                    if r["rel_diff_vs_df4_pct"] is not None),
        }

    # ---------------------------------------------------------------- verdict
    verdict = {}
    for tag in ("2d", "3d"):
        d = out["dims"].get(tag, {})
        if "pass_laplace" not in d:
            verdict[tag] = {"status": d.get("status", "missing")}
            continue
        verdict[tag] = {
            "pass_laplace": d["pass_laplace"],
            "n_radii": d["n_radii"],
            # a Laplace *linearity* claim needs >=3 radii; with 2 points the fit is
            # trivially exact and must not be reported as a pass.
            "pass_laplace_strict": bool(d["pass_laplace"] and d["n_radii"] >= 3),
            "max_dev_sigma_pct": d["max_dev_sigma_pct"],
            "within_3pct": bool(d["max_dev_sigma_pct"] <= 3.0),
            "R2": d["R2"], "intercept_frac_pct": d["intercept_frac_pct"],
            "sigma_eff_fit": d["sigma_eff_fit"],
            "sigma_eff_rel_diff_vs_stored_pct":
                d["vs_stored_result_json"]["sigma_eff_rel_diff_pct"],
        }
    out["verdict"] = verdict
    (OUT / "sweep_result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    print(f"saved -> {OUT / 'sweep_result.json'}")


if __name__ == "__main__":
    main()