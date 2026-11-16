#!/usr/bin/env python3
"""XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION-BUDGET

Error budget of the delivered calibrated inversion
(``src/tensorlbm/contact_angle_inversion/``, rounds m18090 + m22445).

Why this round exists
---------------------
m18090 validated 3 hold-out angles at one configuration (errors 1.09-2.57 %).
m22445 re-ran it and found the same inversion on the *production* grid at
11.03 % (theta=90).  Neither round asked the two prior questions:

  Q1  Is the angle *estimator* (``contact_angle_from_mask``) itself unbiased?
      Every calibration number is produced by it, so an unknown estimator bias
      is an unknown floor under every quoted error.
  Q2  Is the residual error caused by the LBM physics (Huang 2007 / resolution
      transfer) or by the piecewise-linear *interpolation* between calibration
      nodes?  m22445 attributed it to resolution transfer without testing the
      interpolation term, even though its own Arm A (same grid as the
      calibration, theta=90) already missed by 7.71 %.

Arms
----
  S  synthetic: analytic circular caps of exactly known theta (see
     ``tensorlbm.contact_angle_inversion.synthetic``) measured with the same
     operator the calibration uses -> estimator bias vs theta, radius,
     interface width and sub-lattice phase.  No LBM, no physics.
  D  dense: extra calibration nodes on the coarse grid inside the steep band
     -> leave-one-out interpolation error for the delivered 7-node curve vs a
     refined curve.
  E  eval: new hold-out targets run through both curves with real LBM drops.

Everything in REPORT.md is rendered from the JSON in ``evidence/``; no number
is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

EVID = os.path.join(HERE, "evidence")
GATE_PCT = 15.0          # owner precision standard
STRICT_PCT = 3.0         # the stricter band the platform report asks about


def _dump(name: str, obj) -> str:
    os.makedirs(EVID, exist_ok=True)
    path = os.path.join(EVID, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    print(f"[write] {path}")
    return path


def _load(name: str):
    with open(os.path.join(EVID, name)) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Arm S -- estimator bias against analytic caps
# ---------------------------------------------------------------------------

def arm_synthetic(thetas=None, radii=None, widths=None, phases=None) -> dict:
    import torch
    from tensorlbm.contact_angle_inversion.synthetic import sessile_cap_field
    from tensorlbm.sc_surface_tension import contact_angle_from_mask

    torch.set_num_threads(4)
    thetas = thetas if thetas is not None else list(range(30, 155, 5))
    radii = radii if radii is not None else [6.0, 8.0, 12.0, 15.0, 20.0]
    widths = widths if widths is not None else [2.5]
    phases = phases if phases is not None else [0.0, 0.25, 0.5]

    rho_l, rho_v = 1.957, 0.1596
    mid = 0.5 * (rho_l + rho_v)
    rows = []
    for w in widths:
        for R in radii:
            for th in thetas:
                for ph in phases:
                    f = sessile_cap_field(th, R, wall_rows=2, x_phase=ph,
                                          rho_l=rho_l, rho_v=rho_v, interface_width=w)
                    m = contact_angle_from_mask(f["rho"], f["solid"], mid=mid)
                    ok = m.get("status") == "ok"
                    rows.append({
                        "theta_true": float(th),
                        "radius": float(R),
                        "interface_width": float(w),
                        "x_phase": float(ph),
                        "grid": [int(f["rho"].shape[1]), int(f["rho"].shape[0])],
                        "theta_meas": m.get("theta_deg") if ok else None,
                        "bias_deg": (m["theta_deg"] - th) if ok else None,
                        "bias_pct": (abs(m["theta_deg"] - th) / th * 100.0) if ok else None,
                        "R_fit": m.get("R_fit") if ok else None,
                        "n_rows": m.get("n_rows"),
                        "status": m.get("status"),
                    })
                    print(f"  S th={th:5.1f} R={R:4.1f} w={w} ph={ph:.2f} "
                          f"-> {('%.2f' % m['theta_deg']) if ok else m.get('status')}")

    # aggregate: mean bias per (radius, theta) over phases, at the base width
    agg = {}
    for r in rows:
        if r["bias_deg"] is None:
            continue
        key = f"{r['radius']}|{r['theta_true']}|{r['interface_width']}"
        agg.setdefault(key, []).append(r["bias_deg"])
    summary = []
    for key, vals in agg.items():
        R, th, w = key.split("|")
        summary.append({
            "radius": float(R), "theta_true": float(th), "interface_width": float(w),
            "bias_mean_deg": sum(vals) / len(vals),
            "bias_spread_deg": max(vals) - min(vals),
            "n_ok": len(vals), "n_phase": len(phases),
        })
    summary.sort(key=lambda d: (d["interface_width"], d["radius"], d["theta_true"]))

    measured = [r for r in rows if r["bias_deg"] is not None]
    out = {
        "arm": "S_synthetic_estimator",
        "operator": "tensorlbm.sc_surface_tension.contact_angle_from_mask(mid=0.5*(rho_l+rho_v))",
        "ground_truth": "analytic circular arc, tanh diffuse profile, mask boundary == arc",
        "config": {"thetas": thetas, "radii": radii, "widths": widths, "phases": phases,
                   "rho_l": rho_l, "rho_v": rho_v, "wall_rows": 2},
        "rows": rows,
        "summary": summary,
        "coverage": {
            "n_cases": len(rows),
            "n_measured": len(measured),
            "n_not_measured": len(rows) - len(measured),
            "measurable_theta_range_per_radius": {},
        },
    }
    for R in radii:
        oks = [r["theta_true"] for r in rows
               if r["radius"] == R and r["bias_deg"] is not None]
        out["coverage"]["measurable_theta_range_per_radius"][str(R)] = (
            [min(oks), max(oks)] if oks else None)
    if measured:
        worst = max(measured, key=lambda r: abs(r["bias_deg"]))
        out["worst"] = worst
        out["max_abs_bias_deg"] = abs(worst["bias_deg"])
        out["max_abs_bias_pct"] = worst["bias_pct"]
        out["gate_pct"] = GATE_PCT
        out["within_gate"] = bool(out["max_abs_bias_pct"] < GATE_PCT)
        out["within_strict"] = bool(out["max_abs_bias_pct"] < STRICT_PCT)
    _dump("S_synthetic.json", out)
    return out


# ---------------------------------------------------------------------------
# Arm D -- interpolation (leave-one-out) on the delivered vs refined curve
# ---------------------------------------------------------------------------

def _pchip_slopes(x, y):
    """Fritsch-Carlson monotone-preserving knot slopes (no overshoot)."""
    n = len(x)
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    d = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    return m


def _pchip_eval(x, y, m, xq):
    if xq <= x[0]:
        return y[0]
    if xq >= x[-1]:
        return y[-1]
    i = max(0, min(len(x) - 2, next(k for k in range(len(x) - 1) if x[k + 1] >= xq)))
    h = x[i + 1] - x[i]
    t = (xq - x[i]) / h
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    return h00 * y[i] + h10 * h * m[i] + h01 * y[i + 1] + h11 * h * m[i + 1]


def _linear_cos(fw, cos, x):
    """Piecewise-linear cos(theta) at ``f_w = x`` (clamped at the ends)."""
    if x <= fw[0]:
        return cos[0]
    if x >= fw[-1]:
        return cos[-1]
    for i in range(len(fw) - 1):
        if fw[i] <= x <= fw[i + 1]:
            t = (x - fw[i]) / (fw[i + 1] - fw[i])
            return cos[i] + t * (cos[i + 1] - cos[i])
    return cos[-1]


def _interp_cos(fw, cos, theta_target, method):
    """f_w for a target angle, interpolating in cos space (delivered behaviour)."""
    ct = math.cos(math.radians(theta_target))
    if method == "linear":
        for i in range(len(fw) - 1):
            y0, y1 = cos[i], cos[i + 1]
            if (y0 - ct) * (y1 - ct) <= 0 and abs(y1 - y0) > 1e-12:
                t = (ct - y0) / (y1 - y0)
                return fw[i] + t * (fw[i + 1] - fw[i])
        return None
    # pchip in f_w -> cos, then invert by bisection on the (monotone part) table
    m = _pchip_slopes(fw, cos)
    lo, hi = fw[0], fw[-1]
    f = lambda x: _pchip_eval(fw, cos, m, x)
    increasing = cos[-1] > cos[0]
    a, b = (lo, hi) if increasing else (hi, lo)
    # scan brackets on a fine sweep so a non-monotone table cannot fool us
    grid = [lo + (hi - lo) * k / 2000.0 for k in range(2001)]
    vals = [f(x) for x in grid]
    for k in range(len(grid) - 1):
        if (vals[k] - ct) * (vals[k + 1] - ct) <= 0 and abs(vals[k + 1] - vals[k]) > 1e-14:
            t = (ct - vals[k]) / (vals[k + 1] - vals[k])
            return grid[k] + t * (grid[k + 1] - grid[k])
    return None


def arm_dense(fw_extra, nx=64, ny=32, radius=8.0, steps=5000, threads=4, tag="D_dense") -> dict:
    from tensorlbm.contact_angle_inversion import calibrate

    t0 = time.time()
    cal = calibrate(fw_values=fw_extra, nx=nx, ny=ny, radius=radius,
                    steps=steps, threads=threads)
    out = {
        "arm": "D_dense_calibration",
        "config": {"nx": nx, "ny": ny, "radius": radius, "steps": steps},
        "fw_values": fw_extra,
        "points": [{"f_w": p.f_w, "theta_measured": p.theta_measured,
                    "cos_measured": p.cos_measured, "status": p.status,
                    "drift_deg": p.drift_deg} for p in cal.points],
        "seconds": time.time() - t0,
    }
    out["tag"] = tag
    _dump(tag + ".json", out)
    return out


def arm_budget() -> dict:
    """Leave-one-out interpolation error of the delivered vs the refined curve."""
    base = _load("../../XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION-HOLDOUT/summary.json")
    dense = _load("D_dense.json")

    def curve(points):
        pts = [p for p in points
               if p.get("status", "ok") == "ok"
               and p.get("cos_measured") is not None
               and p.get("theta_measured") is not None
               and p["theta_measured"] == p["theta_measured"]]
        pts.sort(key=lambda p: p["f_w"])
        return ([p["f_w"] for p in pts], [p["cos_measured"] for p in pts],
                [p["theta_measured"] for p in pts])

    delivered = [(p["f_w"], p["theta_measured"], p["cos_measured"])
                 for p in base["calibration"]["points"]]
    refined = [(p["f_w"], p["theta_measured"], p["cos_measured"]) for p in dense["points"]]
    allpts = sorted({round(p[0], 4): {"f_w": p[0], "theta_measured": p[1],
                                      "cos_measured": p[2], "status": "ok"}
                     for p in delivered + refined
                     if p[1] is not None and p[1] == p[1]}.items())
    allpts = [v for _, v in allpts]

    def loo(points, method):
        """Hold one node out, forecast its angle from the remaining nodes.

        The forecast is ``forward(f_w_held)`` on the reduced curve -- *not*
        ``forward(invert(theta))``, which is an identity and measures nothing.
        Rows that would be extrapolations are flagged and excluded from the
        maxima.
        """
        res = []
        for hold in points:
            rest = [p for p in points if p is not hold]
            fw, cos, _ = curve(rest)
            if len(fw) < 2:
                continue
            x = hold["f_w"]
            inside = fw[0] <= x <= fw[-1]
            if method == "linear":
                c = _linear_cos(fw, cos, x)
            else:
                m = _pchip_slopes(fw, cos)
                c = _pchip_eval(fw, cos, m, x)
            pred = math.degrees(math.acos(max(-1.0, min(1.0, c))))
            res.append({
                "f_w_held": x, "theta_true": hold["theta_measured"],
                "theta_pred": pred,
                "err_deg": pred - hold["theta_measured"],
                "err_pct": abs(pred - hold["theta_measured"]) / hold["theta_measured"] * 100.0,
                "extrapolated": not inside,
                "in_dense_band": bool(0.55 <= x <= 0.72),
            })
        return res

    rep = {
        "arm": "D_budget",
        "delivered_curve": {"points": delivered, "n": len(delivered)},
        "refined_curve": {"points": refined, "n": len(refined),
                          "extra": dense["fw_values"]},
        "loo": {
            "delivered_linear": loo([{"f_w": f, "theta_measured": t, "cos_measured": c}
                                     for f, t, c in delivered], "linear"),
            "delivered_pchip": loo([{"f_w": f, "theta_measured": t, "cos_measured": c}
                                    for f, t, c in delivered], "pchip"),
            "refined_linear": loo(allpts, "linear"),
            "refined_pchip": loo(allpts, "pchip"),
        },
    }
    for k, rows in rep["loo"].items():
        scored = [r for r in rows if not r["extrapolated"]]
        band = [r for r in scored if r["in_dense_band"]]
        rep["loo"][k] = {
            "rows": rows,
            "n": len(rows),
            "n_scored": len(scored),
            "n_extrapolated": len(rows) - len(scored),
            "max_err_pct_all": max((r["err_pct"] for r in scored), default=None),
            "max_err_pct_band": max((r["err_pct"] for r in band), default=None),
            "mean_abs_err_pct_all": (sum(r["err_pct"] for r in scored) / len(scored))
                                    if scored else None,
        }
    # ---- measured transfer pair: identical f_w, coarse vs production grid ----
    hold_ev = os.path.join(HERE, "..", "XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION-HOLDOUT",
                           "evidence")
    pair = {}
    for tag in ("A_th090", "B_th090"):
        path = os.path.join(hold_ev, tag + ".json")
        if os.path.exists(path):
            with open(path) as fh:
                d = json.load(fh)
            pair[tag] = {k: d.get(k) for k in ("theta_target", "f_w", "theta_measured",
                                               "err_deg", "err_pct", "config")}
    rep["transfer_pair"] = pair
    if len(pair) == 2 and pair["A_th090"]["f_w"] and pair["B_th090"]["f_w"]:
        same_fw = abs(pair["A_th090"]["f_w"] - pair["B_th090"]["f_w"]) < 1e-6
        rep["transfer_pair"]["same_f_w"] = bool(same_fw)
        if same_fw:
            rep["transfer_pair"]["delta_deg"] = (pair["B_th090"]["theta_measured"]
                                                 - pair["A_th090"]["theta_measured"])

    # ---- estimator term between the two grids (Arm S) ----------------------
    try:
        synth = _load("S_synthetic.json")
        bias = {}
        for R in (8.0, 15.0):
            vals = [r["bias_mean_deg"] for r in synth["summary"]
                    if abs(r["radius"] - R) < 1e-9 and abs(r["theta_true"] - 90.0) < 1e-9]
            if vals:
                bias[str(R)] = vals[0]
        if len(bias) == 2:
            rep["estimator_term_deg"] = {
                "bias_at_R8": bias["8.0"], "bias_at_R15": bias["15.0"],
                "difference": bias["8.0"] - bias["15.0"],
                "note": "same-angle estimator bias at the two grid radii; the coarse "
                        "calibration carries this extra offset when it is used on the "
                        "production grid",
            }
    except FileNotFoundError:
        pass

    # ---- budget at theta = 90 ---------------------------------------------
    if "transfer_pair" in rep and rep["transfer_pair"].get("same_f_w"):
        tot = abs(rep["transfer_pair"]["B_th090"]["err_deg"])
        incon = abs(rep["transfer_pair"]["A_th090"]["err_deg"])
        rep["budget_theta90_deg"] = {
            "total_on_production_grid": tot,
            "in_config_term": incon,
            "residual_transfer_term": tot - incon,
            "estimator_bias_difference": rep.get("estimator_term_deg", {}).get("difference"),
            "shares_pct": {
                "in_config": incon / tot * 100.0,
                "transfer": (tot - incon) / tot * 100.0,
            },
            "interpretation": "the in-config term is present with no transfer at all, "
                              "so it cannot be Huang2007/resolution; it is the angle "
                              "operator plus node spacing",
        }
    rep["gate_pct"] = GATE_PCT
    _dump("D_budget.json", rep)
    return rep


# ---------------------------------------------------------------------------
# Arm E -- end-to-end hold-out with the refined curve
# ---------------------------------------------------------------------------

def arm_eval(targets, fw_table, nx=64, ny=32, radius=8.0, steps=5000,
             threads=4, tag="refined", method="pchip") -> dict:
    """Run real LBM drops at the f_w the (refined) curve prescribes."""
    import torch
    from tensorlbm.contact_angle_inversion import calibrate
    from tensorlbm.contact_line.engine import Channel, WallStepper, sessile_drop_state
    from tensorlbm.sc_surface_tension import contact_angle_from_mask

    torch.set_num_threads(threads)
    rho_l, rho_v = 1.957, 0.1596
    mid = 0.5 * (rho_l + rho_v)
    pts = sorted(fw_table, key=lambda p: p["f_w"])
    fw = [p["f_w"] for p in pts]
    cos = [p["cos_measured"] for p in pts]

    rows, t0 = [], time.time()
    for th in targets:
        f_w = _interp_cos(fw, cos, th, method)
        if f_w is None:
            rows.append({"theta_target": th, "status": "not-in-calibrated-range"})
            continue
        channel = Channel(nx=nx, ny=ny, wall_rows=2, device="cpu")
        solid = channel.solid_mask()
        y_wall = channel.y_wall()
        f, _ = sessile_drop_state(channel, radius=radius, height=0.0)
        stepper = WallStepper(channel, G=5.0, tau=1.0, rho_l=rho_l, rho_v=rho_v,
                              psi_fn=__import__("tensorlbm.sc_surface_tension",
                                                fromlist=["psi_exp"]).psi_exp,
                              wall_fraction=f_w)
        hist = []
        for step in range(1, steps + 1):
            f = stepper.step(f)
            if step % 500 == 0:
                rho = stepper.density(f)
                m = contact_angle_from_mask(rho, solid, mid=mid)
                if m.get("status") == "ok":
                    hist.append([step, m["theta_deg"]])
        rho = stepper.density(f)
        m = contact_angle_from_mask(rho, solid, mid=mid)
        drift = abs(hist[-1][1] - hist[-4][1]) if len(hist) >= 4 else float("nan")
        ok = m.get("status") == "ok" and (not math.isfinite(drift) or drift <= 1.0)
        err = abs(m["theta_deg"] - th) if ok else float("nan")
        rows.append({
            "theta_target": float(th), "f_w": float(f_w), "method": method,
            "theta_measured": m.get("theta_deg") if ok else None,
            "err_deg": err if ok else None,
            "err_pct": (err / th * 100.0) if ok else None,
            "within_gate": bool(ok and err / th * 100.0 < GATE_PCT),
            "within_strict": bool(ok and err / th * 100.0 < STRICT_PCT),
            "drift_deg": drift, "R_fit": m.get("R_fit"), "n_rows": m.get("n_rows"),
            "status": "ok" if ok else m.get("status"),
            "history": hist,
        })
        print(f"  E[{tag}] th={th} f_w={f_w:.4f} -> {rows[-1]['theta_measured']} "
              f"err={rows[-1]['err_pct']}")
    out = {"arm": "E_eval_" + tag, "method": method,
           "config": {"nx": nx, "ny": ny, "radius": radius, "steps": steps},
           "rows": rows, "seconds": time.time() - t0}
    ok_rows = [r for r in rows if r.get("err_pct") is not None]
    out["n_ok"] = len(ok_rows)
    out["max_err_pct"] = max((r["err_pct"] for r in ok_rows), default=None)
    out["n_within_gate"] = sum(1 for r in ok_rows if r["within_gate"])
    out["n_within_strict"] = sum(1 for r in ok_rows if r["within_strict"])
    _dump(f"E_{tag}.json", out)
    return out


def arm_fine(threads=4) -> dict:
    """Production grid (128x64, R=15): dense sweep, then eval on its own curve.

    Tests whether the f_w-insensitive plateau seen at R=8 is a small-drop
    artefact or survives at the production resolution.
    """
    dense = arm_dense([0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68], nx=128, ny=64,
                      radius=15.0, steps=6000, threads=threads, tag="D_dense_fine")
    table = [p for p in dense["points"] if p.get("status") == "ok"]
    for tag, method in (("fine_pchip", "pchip"), ("fine_linear", "linear")):
        arm_eval([90.0, 105.0], table, nx=128, ny=64, radius=15.0, steps=6000,
                 threads=threads, tag=tag, method=method)
    return dense


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["synthetic", "dense", "fine", "budget", "eval", "report"])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if args.cmd == "synthetic":
        arm_synthetic()
    elif args.cmd == "dense":
        arm_dense([0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72], threads=args.threads)
    elif args.cmd == "fine":
        arm_fine(threads=args.threads)
    elif args.cmd == "budget":
        arm_budget()
    elif args.cmd == "eval":
        dense = _load("D_dense.json")
        base = _load("../../XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION-HOLDOUT/summary.json")
        merged = {}
        for p in base["calibration"]["points"] + dense["points"]:
            if p.get("status") == "ok":
                merged[round(p["f_w"], 4)] = p
        table = [merged[k] for k in sorted(merged)]
        # delivered behaviour, for a like-for-like comparison on the same targets
        arm_eval([90.0, 105.0], base["calibration"]["points"], threads=args.threads,
                 tag="delivered", method="linear")
        arm_eval([90.0, 105.0], table, threads=args.threads, tag="refined", method="pchip")
        arm_eval([90.0, 105.0], table, threads=args.threads, tag="refined_linear",
                 method="linear")
    elif args.cmd == "report":
        import report as rep
        rep.build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())