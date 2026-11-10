#!/usr/bin/env python3
"""P187-W1-N08 快速定稿：把已完成算例的窗均值 -> 双参数律 -> 判据数字 -> 图。

用途：扫掠未跑满时也要有**程序化、可审计**的判据 JSON 与图（honest partial）。
所有数字来自 ``runs/p187_w1_n08/*.json``（零手抄）。

    python scripts/finalize_p187_w1_n08.py --min-t 80 --grid 11
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.confined_cylinder_law import (  # noqa: E402
    MISSION_WINDOWS,
    STATIONARY_WINDOWS,
    churchill_bernstein,
    window_means,
)

RUNS = ROOT / "runs" / "p187_w1_n08"
ART = ROOT / "artifacts" / "p187_w1_n08"
FIG = ART / "figures"
METRICS = {"balance": "nu_global", "surface": "nu_local", "bc": "nu_bc"}


def load(min_t: float, grid: int) -> dict[float, dict]:
    out = {}
    for p in sorted(RUNS.glob("hd*_D*.json")):
        r = json.loads(p.read_text())
        if int(round(r["geometry"]["D_eff"])) != grid:
            continue
        if float(r.get("t_reached", 0.0)) < min_t:
            continue
        out[round(float(r["geometry"]["h_over_d"]), 4)] = r
    return out


def fit_law(hd, tau, nu):
    """Nu = Nu_inf (1+a hd^-m)(1+b tau^-beta)；对数残差，小网格播种 + LM。"""
    from scipy.optimize import least_squares

    hd, tau, nu = map(lambda v: np.asarray(v, float), (hd, tau, nu))
    logn = np.log(nu)

    def resid(p):
        n0, a, m, b, be = p
        m = np.clip(m, 0.0, 6.0); be = np.clip(be, 0.0, 6.0)
        return np.log(np.abs(n0 * (1 + a * hd ** (-m)) * (1 + b * tau ** (-be))) + 1e-30) - logn

    best = None
    for m0 in (0.5, 1.0, 1.5):
        for b0 in (0.5, 1.0, 1.5):
            X = np.column_stack([np.ones_like(hd), hd ** (-m0), tau ** (-b0)])
            c, *_ = np.linalg.lstsq(X, logn, rcond=None)
            p0 = [math.exp(c[0]), c[1], m0, c[2], b0]
            try:
                r = least_squares(resid, p0, method="lm", max_nfev=4000)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    p = best.x
    sse = float(np.sum(best.fun ** 2))
    n = nu.size
    return {"nu_inf": float(p[0]), "a": float(p[1]), "m": float(p[2]),
            "b": float(p[3]), "beta": float(p[4])}, sse, n


def fit_additive(hd, tau, nu):
    from scipy.optimize import least_squares
    hd, tau, nu = map(lambda v: np.asarray(v, float), (hd, tau, nu))
    logn = np.log(nu)

    def resid(p):
        n0, a, m, b, be = p
        m = np.clip(m, 0.0, 6.0); be = np.clip(be, 0.0, 6.0)
        return np.log(np.abs(n0 + a * hd ** (-m) + b * tau ** (-be)) + 1e-30) - logn

    best = None
    for m0 in (0.5, 1.0, 1.5):
        for b0 in (0.5, 1.0, 1.5):
            p0 = [float(nu.mean()), 0.0, m0, 0.0, b0]
            try:
                r = least_squares(resid, p0, method="lm", max_nfev=4000)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    p = best.x
    return {"nu_inf": float(p[0]), "a": float(p[1]), "m": float(p[2]),
            "b": float(p[3]), "beta": float(p[4])}, float(np.sum(best.fun ** 2))


def ftest_factor(hd, tau, nu, factor):
    """负控：把该因子置常数后，F 检验其参数项是否显著。"""
    from scipy import stats
    hd = np.asarray(hd, float).copy()
    tau = np.asarray(tau, float).copy()
    nu = np.asarray(nu, float)
    if factor == "hd":
        hd[:] = hd.mean()
    else:
        tau[:] = tau.mean()
    p_full, sse_full, _n = fit_law(hd, tau, nu)
    if factor == "hd":
        p_red, sse_red, _n2 = fit_law(hd, tau, nu)  # 该项无变异 -> a 不可辨
        # 显式去掉 a 项：固定 a=0
        from scipy.optimize import least_squares
        logn = np.log(nu)

        def resid(p):
            n0, b, be = p
            return np.log(np.abs(n0 * (1 + b * tau ** (-be))) + 1e-30) - logn
        r = least_squares(resid, [nu.mean(), p_full["b"], p_full["beta"]], method="lm", max_nfev=4000)
        sse_red, k_red = float(np.sum(r.fun ** 2)), 3
    else:
        from scipy.optimize import least_squares
        logn = np.log(nu)

        def resid(p):
            n0, a, m = p
            return np.log(np.abs(n0 * (1 + a * hd ** (-m))) + 1e-30) - logn
        r = least_squares(resid, [nu.mean(), p_full["a"], p_full["m"]], method="lm", max_nfev=4000)
        sse_red, k_red = float(np.sum(r.fun ** 2)), 3
    df1, df2 = 5 - k_red, max(nu.size - 5, 1)
    num = sse_red - sse_full
    f = (num / df1) / (sse_full / df2) if num > 0 and sse_full > 0 else 0.0
    return {"factor": factor, "F": float(f),
            "p_value": float(1.0 - stats.f.cdf(max(f, 0.0), df1, df2)),
            "sse_full": sse_full, "sse_reduced": sse_red, "df": [df1, df2],
            "significant": bool(1.0 - stats.f.cdf(max(f, 0.0), df1, df2) < 0.05)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=int, default=11)
    ap.add_argument("--min-t", type=float, default=80.0)
    ap.add_argument("--primary", default="balance")
    args = ap.parse_args(argv)
    ART.mkdir(parents=True, exist_ok=True)

    runs = load(args.min_t, args.grid)
    if not runs:
        raise SystemExit("没有满足条件的算例")
    key = METRICS[args.primary]
    alt = METRICS["surface"] if args.primary == "balance" else METRICS["balance"]

    rows, rows_alt = [], []
    for hd, r in sorted(runs.items()):
        for wset, store in ((MISSION_WINDOWS, rows), (STATIONARY_WINDOWS, rows_alt)):
            wm = window_means(r["times"], r[key], wset)
            for w, v in wm.items():
                if np.isfinite(v):
                    store.append({"h_over_d": hd, "tau": float(w[1] - w[0]),
                                  "window": [float(w[0]), float(w[1])], "nu": float(v),
                                  "source": r["label"], "D_eff": r["geometry"]["D_eff"],
                                  "surface_metric": float(
                                      window_means(r["times"], r[alt], [w])[tuple(w)])})

    cb = churchill_bernstein(float(next(iter(runs.values()))["config"]["re"]),
                            float(next(iter(runs.values()))["config"]["pr"]))

    hd = np.array([x["h_over_d"] for x in rows_alt])
    tau = np.array([x["tau"] for x in rows_alt])
    nu = np.array([x["nu"] for x in rows_alt])
    p_mul, sse_mul, n_pts = fit_law(hd, tau, nu)
    p_add, sse_add = fit_additive(hd, tau, nu)
    aic = lambda sse, p: n_pts * math.log(sse / n_pts) + 2 * p
    pred = p_mul["nu_inf"] * (1 + p_mul["a"] * hd ** (-p_mul["m"])) * (1 + p_mul["b"] * tau ** (-p_mul["beta"]))
    rel = pred / nu - 1.0

    # 双向 ANOVA 交互项（对数域）
    y = np.log(nu)
    hdu, tauu = sorted(set(hd)), sorted(set(tau))
    mu = y.mean()
    r_eff = {v: y[hd == v].mean() - mu for v in hdu}
    c_eff = {v: y[tau == v].mean() - mu for v in tauu}
    inter = y - np.array([mu + r_eff[a] + c_eff[b] for a, b in zip(hd, tau)])

    # C2：零堵塞外推 vs C-B，及"最宽域直接用"的朴素基线
    hd_max = max(hdu)
    base_nu = float(np.mean([x["nu"] for x in rows_alt if x["h_over_d"] == hd_max]))
    c2 = {
        "nu_churchill_bernstein": cb,
        "nu_inf_extrapolated": p_mul["nu_inf"],
        "deviation_pct": abs(p_mul["nu_inf"] - cb) / cb * 100,
        "baseline_widest_domain": {"h_over_d": hd_max, "nu": base_nu,
                                   "deviation_pct": abs(base_nu - cb) / cb * 100},
        "pass": bool(abs(p_mul["nu_inf"] - cb) / cb * 100 < 5.0),
        "threshold": "外推 Nu_inf vs C-B 偏差 <5%",
    }

    # C3：留出窗 [25,60] 与 [40,105]，律预测差值 vs 实测
    tw = [(25.0, 60.0), (40.0, 105.0)]
    fit_rows = [x for x in rows if tuple(x["window"]) not in tw]
    c3 = {"test_windows": tw, "n_fit_points": len(fit_rows), "pass": None}
    if len(fit_rows) >= 6:
        p3, _, _ = fit_law([x["h_over_d"] for x in fit_rows], [x["tau"] for x in fit_rows],
                           [x["nu"] for x in fit_rows])
        def obs(win):
            v = [x["nu"] for x in rows if tuple(x["window"]) == win]
            return float(np.mean(v)) if v else float("nan")
        t_short, t_long = 35.0, 65.0
        hd_ref = float(np.mean([x["h_over_d"] for x in fit_rows]))
        pr = lambda t: p3["nu_inf"] * (1 + p3["a"] * hd_ref ** (-p3["m"])) * (1 + p3["b"] * t ** (-p3["beta"]))
        o_s, o_l = obs(tw[0]), obs(tw[1])
        if np.isfinite(o_s) and np.isfinite(o_l):
            obs_diff = (o_s - o_l) / o_l * 100
            pred_diff = (pr(t_short) - pr(t_long)) / pr(t_long) * 100
            c3.update({"observed_diff_pct": obs_diff, "predicted_diff_pct": pred_diff,
                       "abs_diff_pp": abs(pred_diff - obs_diff),
                       "pass": bool(abs(pred_diff - obs_diff) < 0.5),
                       "threshold": "|律预测差值-实测差值| < 0.5pp",
                       "law_from_holdout_fit": p3})

    nc_hd = ftest_factor(hd, tau, nu, "hd")
    nc_tau = ftest_factor(hd, tau, nu, "tau")

    diag = {}
    for hdir, r in sorted(runs.items()):
        d = r["diag"]
        u = float(np.mean([x["mass_in"] for x in d[-30:]])) / r["geometry"]["H_eff"]
        diag[f"hd{hdir:g}"] = {
            "re_nominal": r["config"]["re"],
            "u_mean_measured": u,
            "re_measured": u * r["geometry"]["D_eff"] / r["lattice"]["nu"],
            "nu_balance_tail": r["final"].get("nu_balance_mean_tail"),
            "nu_surface_tail": r["final"].get("nu_local_mean_tail"),
            "nu_bc_tail": r["final"].get("nu_bc_mean_tail"),
            "surface_over_balance": (r["final"].get("nu_local_mean_tail") or float("nan"))
                                    / (r["final"].get("nu_balance_mean_tail") or float("nan")),
            "t_reached": r.get("t_reached"), "ny": r["geometry"]["ny"], "nx": r["geometry"]["nx"],
            "node_steps": r["node_steps"], "elapsed_s": r["elapsed_s"],
        }

    out = {
        "node": "P187-W1-N08",
        "status": "PARTIAL (sweep unfinished; grid level D_eff=%d only)" % args.grid,
        "primary_metric": args.primary,
        "runs_used": {f"hd{k:g}": {"label": v["label"], "t_reached": v.get("t_reached"),
                                   "ny": v["geometry"]["ny"], "nx": v["geometry"]["nx"]}
                      for k, v in runs.items()},
        "n_points_primary": int(nu.size),
        "C1": {
            "dataset": "grid D_eff=%d x stationary windows" % args.grid,
            "n_points": int(nu.size),
            "multiplicative": p_mul, "sse_log": sse_mul,
            "rmse_log_pct": math.sqrt(sse_mul / nu.size) * 100,
            "additive": p_add, "sse_log_additive": sse_add,
            "aic_multiplicative": aic(sse_mul, 5), "aic_additive": aic(sse_add, 5),
            "delta_aic_mul_minus_add": aic(sse_mul, 5) - aic(sse_add, 5),
            "law_max_abs_residual_pct": float(np.max(np.abs(rel)) * 100),
            "law_rms_residual_pct": float(np.sqrt(np.mean(rel ** 2)) * 100),
            "anova_interaction_max_abs_pct": float(np.max(np.abs(inter)) * 100),
            "pass": bool(np.max(np.abs(rel)) * 100 < 2.0 and aic(sse_mul, 5) < aic(sse_add, 5)),
            "threshold": "残差<2% 且 AIC(乘)<AIC(加)",
        },
        "C2": c2,
        "C3": c3,
        "C4": {"pass": None, "drift_pct": None,
               "note": "两级 Δx 未跑完（D_eff=17 扫掠仅到 t≈42 D/U < 60），C4 未评估 = WITHHELD"},
        "C5": {"hd_constant": nc_hd, "tau_constant": nc_tau,
               "pass": bool(nc_hd["p_value"] > 0.05 and nc_tau["p_value"] > 0.05),
               "threshold": "两负控 p>0.05"},
        "physics_diagnostics": diag,
        "estimator_note": ("surface(壁面梯度积分) 与 balance(进出口焓通量) 与 bc(壁面反反弹计量)"
                           " 三者在同一算例内不等（见 physics_diagnostics.surface_over_balance），"
                           "绝对值口径未闭合 -> C2 的绝对基准存疑，已在报告第 5 节如实列出。"),
    }
    out["all_pass"] = bool(all(out[k]["pass"] for k in ("C1", "C2", "C3", "C5") if out[k]["pass"] is not None))

    (ART / "criteria.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (ART / "window_table.json").write_text(json.dumps({"stationary": rows_alt, "mission": rows}, indent=1))

    # ---- 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for hdir, r in sorted(runs.items()):
        t = np.asarray(r["times"], float)
        axes[0].plot(t, np.asarray(r[key], float), lw=0.8, label=f"H/D={hdir:g}")
    axes[0].set_xlabel("t [D/U]"); axes[0].set_ylabel(f"Nu ({args.primary})")
    axes[0].set_title("(a) window-averaged metric history"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    xx = np.linspace(0, 1.0 / min(hdu) * 1.05, 200)
    tau_ref = max(tauu)
    for hd_v in hdu:
        ys = [np.mean([x["nu"] for x in rows_alt if x["h_over_d"] == hd_v and x["tau"] == tau_ref])]
        axes[1].plot([1 / hd_v], ys, "o")
    axes[1].plot(xx, p_mul["nu_inf"] * (1 + p_mul["a"] * xx ** p_mul["m"]) * (1 + p_mul["b"] * tau_ref ** (-p_mul["beta"])),
                 "-", label=f"fit a={p_mul['a']:.3f}, m={p_mul['m']:.3f}")
    axes[1].axhline(cb, color="crimson", ls="--", label=f"C-B {cb:.3f}")
    axes[1].plot([0], [p_mul["nu_inf"]], "*", ms=14, color="green", label="Nu_inf extrapolation")
    axes[1].axhline(base_nu, color="darkorange", ls=":", label="naive baseline (widest)")
    axes[1].set_xlabel("D/H"); axes[1].set_ylabel("Nu")
    axes[1].set_title("(b) blockage law + zero-blockage extrapolation"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    taus_plot = np.linspace(min(tauu), max(tauu), 100)
    for hd_v in hdu:
        sel = [x for x in rows_alt if x["h_over_d"] == hd_v]
        axes[2].plot([x["tau"] for x in sel], [x["nu"] for x in sel], "o")
        axes[2].plot(taus_plot, p_mul["nu_inf"] * (1 + p_mul["a"] * hd_v ** (-p_mul["m"])) *
                     (1 + p_mul["b"] * taus_plot ** (-p_mul["beta"])), ":")
    axes[2].set_xlabel("window length tau [D/U]"); axes[2].set_ylabel("Nu")
    axes[2].set_title("(c) window-truncation law"); axes[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig1_law_overview.png", dpi=140); plt.close(fig)

    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in {
        "n_points": int(nu.size), "H/D": hdu, "tau_values": tauu,
        "nu_inf": p_mul["nu_inf"], "a": p_mul["a"], "m": p_mul["m"],
        "b": p_mul["b"], "beta": p_mul["beta"],
        "law_max_abs_resid_pct": out["C1"]["law_max_abs_residual_pct"],
        "anova_interaction_pct": out["C1"]["anova_interaction_max_abs_pct"],
        "dAIC": out["C1"]["delta_aic_mul_minus_add"], "C1_pass": out["C1"]["pass"],
        "nu_cb": cb, "C2_dev_pct": c2["deviation_pct"], "C2_base_dev_pct": base_nu and c2["baseline_widest_domain"]["deviation_pct"],
        "C2_pass": c2["pass"], "C3": {k: v for k, v in c3.items() if k != "law_from_holdout_fit"},
        "C5_p": [nc_hd["p_value"], nc_tau["p_value"]], "C5_pass": out["C5"]["pass"],
    }.items()}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())