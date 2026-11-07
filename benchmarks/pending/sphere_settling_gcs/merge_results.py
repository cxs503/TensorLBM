#!/usr/bin/env python3
"""Merge the per-study JSONs into ``result.json`` and evaluate the verdict.

Reads ``out_domain.json`` (λ ladder at R=4), ``out_domain6.json`` (λ ladder at
R=6), ``out_refine.json`` (R ladder at λ=1/8), ``out_reynolds.json``,
``out_anchor.json`` / ``out_anchor_long.json`` (exact momentum-balance anchor)
and evaluates the pre-registered criteria of ``prereg.md`` §5 **as amended by
appendix A1**.  Nothing is typed by hand here.

Normalised observable
---------------------
Every ratio is reported as ``K(λ)·U_t^sim/U_t^Stokes`` with ``K`` the Hasimoto
(1959) periodic-array drag factor, so that a perfect solver+instrument gives
**1.000** and the residual is directly comparable between different λ.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def load(name: str) -> list[dict]:
    p = _HERE / name
    if not p.exists():
        return []
    return json.loads(p.read_text())["runs"]


def exp_asymptote(series: list[float], dt: float) -> dict:
    """Fit ``U(t) = U_inf (1 - exp(-t/tau))`` and return the asymptote.

    The settling cell spins up from rest on a single dominant viscous
    relaxation mode, so the terminal velocity is the *asymptote* of the drift
    series, not its tail mean (see prereg appendix A1.1).
    """
    us = [float(v) for v in series if v == v]
    if len(us) < 5:
        return {}
    ts = [dt * (i + 1) for i in range(len(us))]
    best = None
    tau = 50.0
    while tau <= 20000.0:
        a = [1.0 - math.exp(-t / tau) for t in ts]
        saa = sum(x * x for x in a)
        u_inf = sum(x * u for x, u in zip(a, us)) / saa if saa else float("nan")
        sse = sum((u - u_inf * x) ** 2 for x, u in zip(a, us))
        if best is None or sse < best[0]:
            best = (sse, tau, u_inf)
        tau *= 1.05
    sse, tau, u_inf = best
    tail = sum(us[-10:]) / min(10, len(us))
    return {
        "u_asymptote": u_inf,
        "tau_relax_steps": tau,
        "fit_rms": (sse / len(us)) ** 0.5,
        "tail_residual": 1.0 - tail / u_inf if u_inf else None,
    }


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    k = sxy / sxx
    b = my - k * mx
    resid = max(abs(y - (b + k * x)) for x, y in zip(xs, ys))
    return k, b, resid


def normalised(r: dict) -> float:
    """``K(λ)·U_t^sim/U_t^Stokes`` — 1.000 for an exact solver+instrument."""
    return r["hasimoto_factor"] * r["u_terminal_fit"] / r["u_stokes_unbounded"]


def lsq3(A, y):
    """Least squares for a 3-column design matrix (normal equations)."""
    AtA = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(3)] for i in range(3)]
    Aty = [sum(A[k][i] * y[k] for k in range(len(A))) for i in range(3)]
    M = [row[:] + [Aty[i]] for i, row in enumerate(AtA)]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r_: abs(M[r_][c]))
        M[c], M[piv] = M[piv], M[c]
        for r_ in range(3):
            if r_ != c:
                f = M[r_][c] / M[c][c]
                M[r_] = [M[r_][k] - f * M[c][k] for k in range(4)]
    coef = [M[i][3] / M[i][i] for i in range(3)]
    resid = max(abs(y[k] - sum(coef[i] * A[k][i] for i in range(3))) for k in range(len(A)))
    return coef, resid


def main() -> int:
    domain4 = load("out_domain.json")
    domain6 = load("out_domain6.json")
    refine = load("out_refine.json")
    reyn = load("out_reynolds.json")
    anchor = load("out_anchor_long.json") or load("out_anchor.json")
    runs = domain4 + domain6 + refine + reyn + anchor
    for r in runs:
        f = exp_asymptote(r.get("u_series", []), r.get("sample_every", 25))
        if f:
            r["spin_up_fit"] = f
            r["u_terminal_fit"] = f["u_asymptote"]
        else:
            r["u_terminal_fit"] = r["u_measured"]
        r["u_over_stokes_fit"] = r["u_terminal_fit"] / r["u_stokes_unbounded"]
        r["normalised"] = normalised(r)
        # the same configuration can appear in two studies (e.g. R=6,L=48 is both a
        # domain6 point and the anchor run): keep the longest one, drop duplicates
        best: dict[tuple, dict] = {}
        for r in runs:
            key = (r["R"], r["L"], round(r["Re_target"], 6))
            if key not in best or r["steps"] > best[key]["steps"]:
                best[key] = r
        runs = list(best.values())

    out: dict = {
        "case": "sphere_settling_gcs",
        "estimator": "galilean_invariant_kinetic_flux_control_surface",
        "module": "src/tensorlbm/force_control_surface.py",
        "tests": "tests/test_force_control_surface.py",
        "lattice": "D3Q19",
        "collision": "bgk",
        "nu_lattice": runs[0]["nu"] if runs else None,
        "tau": runs[0]["tau"] if runs else None,
        "observable": "normalised = K(lambda) * U_t_sim / U_t_Stokes  (1.000 = exact)",
        "artifacts": {
            "prereg": "benchmarks/pending/sphere_settling_gcs/prereg.md",
            "driver": "benchmarks/pending/sphere_settling_gcs/run.py",
            "extra_study": "benchmarks/pending/sphere_settling_gcs/run_domain6.py",
            "probes": "benchmarks/pending/sphere_settling_gcs/probe.py",
            "tau_probe": "benchmarks/pending/sphere_settling_gcs/probe_tau.json",
            "figures": "benchmarks/pending/sphere_settling_gcs/figures/",
            "report": "docs/sphere_settling_gcs_20261107.md",
        },
        "runs": runs,
    }
    checks: dict = {}

    def ladder(rs, xkey, label, quantity):
        if len(rs) < 2:
            return None
        xs = [(1.0 / r["R"]) if xkey == "inv_R" else r[xkey] for r in rs]
        ys = [r["normalised"] for r in rs]
        k, b, resid = fit(xs, ys)
        return {
            "quantity": quantity,
            "label": label,
            "x": xs,
            "normalised": ys,
            "slope": k,
            "intercept": b,
            "err_pct": 100.0 * (b - 1.0),
            "max_resid": resid,
            "PASS": abs(b - 1.0) <= 0.03 and resid < 0.05,
        }

    j1 = ladder(
        sorted(domain6, key=lambda r: r["lambda_R_over_L"]),
        "lambda_R_over_L",
        "R=6, lambda -> 0",
        "terminal velocity vs analytic Stokes (confinement extrapolated)",
    )
    if j1:
        checks["J1_terminal_velocity"] = j1
    j1b = ladder(
        sorted(domain4, key=lambda r: r["lambda_R_over_L"]),
        "lambda_R_over_L",
        "R=4 (under-resolved body), lambda -> 0",
        "same quantity on the coarse body (reported, not the primary)",
    )
    if j1b:
        checks["J1b_terminal_velocity_R4"] = j1b
    # R=4 is demonstrably under-resolved (prereg A1.3): the extrapolation uses
    # the resolved points only, and the plain R-to-R span is reported alongside
    # so the reader can see how much of the "extrapolation" is extrapolation.
    ref_res = sorted([r for r in refine if r["R"] >= 6.0], key=lambda r: 1.0 / r["R"])
    j2 = ladder(
        ref_res,
        "inv_R",
        "lambda=1/8, 1/R -> 0 over R>=6",
        "terminal velocity vs confinement-corrected Stokes (body resolution)",
    )
    if j2:
        j2["degenerate_fit"] = len(ref_res) == 2
        j2["R_to_R_span_pct"] = (
            100.0 * (ref_res[-1]["normalised"] / ref_res[0]["normalised"] - 1.0)
            if len(ref_res) >= 2
            else None
        )
        j2["R4_normalised_reported_only"] = [
            (r["R"], r["normalised"]) for r in refine if r["R"] < 6.0
        ]
        j2["status"] = "INCONCLUSIVE_DEGENERATE" if j2["degenerate_fit"] else ("PASS" if j2["PASS"] else "FAIL")
        j2["PASS"] = (not j2["degenerate_fit"]) and j2["PASS"]
        checks["J2_grid_extrapolation"] = j2

    # joint (lambda, 1/R) limit over the resolved points
    pts = [r for r in runs if r["R"] >= 6.0 and abs(r["Re_target"] - 0.05) < 1e-12]
    pts = [r for r in pts if abs(r["control_surface"]["2.5"]["fx_total"]) > 0]
    n_lambda = len({round(r["lambda_R_over_L"], 6) for r in pts})
    n_invR = len({round(1.0 / r["R"], 6) for r in pts})
    if len(pts) >= 4 and n_lambda >= 2 and n_invR >= 2:
        A = [[1.0, r["lambda_R_over_L"], 1.0 / r["R"]] for r in pts]
        y = [r["normalised"] for r in pts]
        coef, resid = lsq3(A, y)
        checks["J2b_joint_limit"] = {
            "model": "normalised = c0 + c1*lambda + c2/R",
            "intercept": coef[0],
            "c_lambda": coef[1],
            "c_invR": coef[2],
            "err_pct": 100.0 * (coef[0] - 1.0),
            "max_resid": resid,
            "n_points": len(pts),
            "PASS": abs(coef[0] - 1.0) <= 0.03,
        }

    if anchor:
        ar = anchor[-1]
        a = ar["control_surface"]["2.5"]
        g = ar["g"]
        tau = ar["tau"]
        v_f = ar["L"] ** 3 - ar["n_solid_cells"]
        # Guo (2002) injects (1 - 1/(2 tau)) * F of discrete momentum per step
        # (the remaining 1/(2tau) sits in the rho*u = sum f c + F/2 half-step);
        # the box-CV observer corroborates this convention independently.
        inj = 1.0 - 0.5 / tau
        checks["J3_anchor"] = {
            "F_total": a["fx_total"],
            "continuum_rho_g_Vfluid": g * v_f,
            "err_vs_continuum_pct": 100.0 * (a["fx_total"] / (g * v_f) - 1.0),
            "discrete_injection_rate_per_step": g * inj,
            "discrete_anchor": g * inj * v_f,
            "err_vs_discrete_pct": 100.0 * (a["fx_total"] / (g * inj * v_f) - 1.0),
            "steps": ar["steps"],
            "PASS": abs(a["fx_total"] / (g * inj * v_f) - 1.0) <= 0.01,
        }
        if "box_cv" in ar:
            box = ar["box_cv"]["fx_mean"]
            v_box = (2 * ar["R"] + 8) ** 3 - ar["n_solid_cells"]
            box_corr = box + g * v_box   # body force acting inside the box CV
            checks["J6_cross_instrument"] = {
                "instrument": "control_volume_force.observe_control_volume_force (box CV)",
                "notes": "box value includes the body force inside the box; corrected here",
                "gcs_fx_total": a["fx_total"],
                "box_cv_raw_fx": box,
                "box_cv_body_force_term": g * v_box,
                "box_cv_corrected_fx": box_corr,
                "rel_diff_pct": 100.0 * (box_corr / a["fx_total"] - 1.0),
                "box_vs_discrete_anchor_pct": 100.0 * (box_corr / (g * inj * v_f) - 1.0),
                "PASS": abs(box_corr / a["fx_total"] - 1.0) <= 0.03,
            }

    if runs:
        # detrended oscillation: remove the fitted spin-up mode (prereg A1.1),
        # then the residual is the lattice-level unsteadiness of F(t)
        osc = {}
        for r in runs:
            ser = [float(v) for v in r["control_surface"]["2.5"]["series"]]
            tf = r.get("u_terminal_fit")
            dt = r["sample_every"]
            tau_r = (r.get("spin_up_fit") or {}).get("tau_relax_steps")
            if not ser or not tf or not tau_r:
                continue
            ffit = exp_asymptote(ser, dt)
            f_inf = ffit.get("u_asymptote")
            tau_f = ffit.get("tau_relax_steps")
            shape = [1.0 - math.exp(-dt * (i + 1) / tau_f) for i in range(len(ser))]
            resid = [v - f_inf * x for v, x in zip(ser, shape)]
            res = resid[len(resid) // 2 :]
            mean = f_inf
            osc[r["tag"]] = {
                "mean": mean,
                "F_inf": f_inf,
                "force_tau_relax_steps": tau_f,
                "force_fit_rms": ffit.get("fit_rms"),
                "rms": (sum(v * v for v in res) / len(res)) ** 0.5,
                "ptp": max(res) - min(res),
            }
            osc[r["tag"]]["rms_over_mean"] = osc[r["tag"]]["rms"] / abs(mean) if mean else None
            osc[r["tag"]]["ptp_over_mean"] = osc[r["tag"]]["ptp"] / abs(mean) if mean else None
            r["oscillation_detrended"] = osc[r["tag"]]
        ptp = max(v["ptp_over_mean"] for v in osc.values()) if osc else float("nan")
        checks["J4_oscillation"] = {
            "quantity": "(max-min)/mean of F(t) after removing the spin-up mode",
            "per_run": {k: v["ptp_over_mean"] for k, v in osc.items()},
            "max_ptp_over_mean": ptp,
            "PASS": ptp <= 0.02,
        }
        spread = []
        for r in runs:
            vals = [r["control_surface"][k]["fx_total"] for k in r["control_surface"]]
            spread.append((max(vals) - min(vals)) / (sum(vals) / len(vals)))
        checks["J5_radius_independence"] = {
            "max_spread": max(spread),
            "per_run": {r["tag"]: s for r, s in zip(runs, spread)},
            "PASS": max(spread) <= 0.015,
        }
        gate = {}
        for r in runs:
            f = r.get("spin_up_fit") or {}
            if not f:
                continue
            gate[r["tag"]] = {
                "fit_rms_over_Uinf": f["fit_rms"] / f["u_asymptote"],
                "steps_over_tau": r["steps"] / f["tau_relax_steps"],
                "tail_residual": f["tail_residual"],
            }
        worst_fit = max(g["fit_rms_over_Uinf"] for g in gate.values()) if gate else float("nan")
        worst_cov = min(g["steps_over_tau"] for g in gate.values()) if gate else float("nan")
        checks["steadiness_gate"] = {
            "criterion": "spin-up fit rms/U_inf < 1% and run length >= 1.5 tau_relax",
            "max_fit_rms_over_Uinf": worst_fit,
            "min_steps_over_tau": worst_cov,
            "per_run": gate,
            "PASS": worst_fit < 0.01 and worst_cov >= 1.5,
        }
    if reyn:
        checks["J7_reynolds_scan"] = {
            f"{r['Re_measured']:.4f}": {
                "U_over_Stokes_fit": r["u_over_stokes_fit"],
                "normalised": r["normalised"],
                "drag_over_Stokes_at_U": r["control_surface"]["2.5"][
                    "drag_over_stokes_at_measured_U"
                ],
                "oseen_factor": 1.0 + 3.0 / 8.0 * r["Re_measured"],
            }
            for r in reyn
        }

    flags = [v["PASS"] for v in checks.values() if isinstance(v, dict) and "PASS" in v]
    checks["all_pass"] = all(flags) if flags else False
    out["verdict"] = checks
    (_HERE / "result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(checks, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())