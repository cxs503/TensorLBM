"""Production sweep for XVAL-LBM-MOVING-WALL-WALL-SHEAR-ME-D2Q9-BGK.

Writes per-run ``case_*.json`` and ``summary.json`` under the output directory
(gates M1..M9, D1 diagnostic).  Thresholds are the predeclared ones in
``PREREG.md``; nothing here may be tuned after seeing the numbers.

Run:  python3 run_xval.py [--stage all|core|ablation|diag|finalize] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mwforce as mf  # noqa: E402

DEFAULT_OUT = Path("outputs/cross_validation/moving_wall_wall_shear_me_d2q9_bgk")
ART = DEFAULT_OUT

XU = mf.mwbb  # alias for readability: the merged reference closure/driver

# predeclared reference configuration (PREREG.md)
REF = dict(tau=0.8, u0=0.02, delta_bl=8.0, ny_stokes=50, nx=4)
REF_COUETTE = dict(ny=34, tau=0.8, u0=0.05, nx=4)


def _dump(name: str, obj: dict) -> dict:
    (ART / f"case_{name}.json").write_text(json.dumps(obj, indent=1))
    return obj


def run_core() -> dict:
    out = {}
    # ---- Couette: wall shear at both walls, H = 16 / 32 -------------------
    cases = []
    for ny, tau in ((18, 0.8), (34, 0.8), (34, 1.2), (34, 0.6)):
        t0 = time.time()
        r = mf.run_couette_shear(ny=ny, nx=4, tau=tau, u0=0.05, n_steps=40000, avg_steps=400)
        r["wall_s"] = round(time.time() - t0, 2)
        cases.append(r)
        _dump(f"couette_shear_ny{ny}_tau{tau}", r)
    # ---- Stokes 2nd: ME force history vs analytic viscous traction --------
    st = []
    for tag, kw in (
        ("REF", dict(ny=50, tau=0.8, u0=0.02, delta_bl=8.0, n_periods=6)),
        ("res_d4", dict(ny=26, tau=0.8, u0=0.02, delta_bl=4.0, n_periods=4)),
        ("tau1.2", dict(ny=50, tau=1.2, u0=0.02, delta_bl=8.0, n_periods=4)),
        ("tau1.8", dict(ny=50, tau=1.8, u0=0.02, delta_bl=8.0, n_periods=4)),
    ):
        t0 = time.time()
        r = mf.run_stokes2_shear(nx=4, **kw)
        r["tag"] = tag
        r["wall_s"] = round(time.time() - t0, 2)
        st.append(r)
        _dump(f"stokes2_shear_{tag}", r)
    # ---- Rayleigh impulsive start -----------------------------------------
    ray = []
    for tag, kw in (("REF", dict(nu=None)), ("tau1.2", dict(nu=(1.2 - 0.5) / 3.0))):
        t0 = time.time()
        r = mf.run_rayleigh_shear(
            ny=66, nx=4, tau=0.8, u0=0.02, n_steps=2000, win=(400, 1600), **kw
        )
        r["tag"] = tag
        r["wall_s"] = round(time.time() - t0, 2)
        ray.append(r)
        _dump(f"rayleigh_shear_{tag}", r)
    out.update(couette=cases, stokes=st, rayleigh=ray)
    return out


def run_ablation() -> dict:
    """Same reference cases, wrong wall closures (the discriminator check)."""
    abl = []
    for variant in (
        "ladd_pre_local",
        "ladd_pre_unitrho",
        "eq_wall_node",
        "static_bb",
        "ladd_sign_flip",
    ):
        r = mf.run_stokes2_shear(
            ny=50, nx=4, tau=0.8, u0=0.02, delta_bl=8.0, n_periods=4, variant=variant
        )
        r["tag"] = f"stokes2_{variant}"
        abl.append(r)
        _dump(f"stokes2_abl_{variant}", r)
    cou = []
    for variant in ("ladd_pre_local", "static_bb", "eq_wall_node"):
        r = mf.run_couette_shear(
            ny=34, nx=4, tau=0.8, u0=0.05, n_steps=6000, avg_steps=200, variant=variant
        )
        r["tag"] = f"couette_{variant}"
        cou.append(r)
        _dump(f"couette_abl_{variant}", r)
    return {"stokes": abl, "couette": cou}


def run_normal_diag() -> dict:
    """D1: wall velocity with a normal component -> fluid mass source."""
    runs = []
    for eps in (0.0, 0.002, 0.005, 0.01, 0.02):
        r = mf.run_normal_wall_drift(ny=34, nx=4, tau=0.8, uw_n=eps, n_steps=600)
        runs.append(r)
        _dump(f"normal_drift_{eps}", r)
    pos = [(r["uw_n"], r["rate_per_step"]) for r in runs if r["uw_n"] > 0]
    if len(pos) >= 2:
        x = np.array([p[0] for p in pos])
        y = np.array([p[1] for p in pos])
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        slope, intercept, r2 = float("nan"), float("nan"), float("nan")
    return {
        "runs": runs,
        "fit_rate_vs_uw_n": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": r2,
            "note": "per-step fluid-mass drift rate vs normal wall speed; "
            "slope is NOT -1 (i.e. not the naive rho*u_w*area source) and the "
            "drift is real, which is why u_w.n != 0 is out of contract",
        },
    }


def finalize(core: dict, abl: dict, diag: dict, ident: dict) -> dict:
    gates = []

    def add(gid, desc, value, threshold, ok, unit="%"):
        gates.append(
            {
                "id": gid,
                "desc": desc,
                "value": value,
                "threshold": threshold,
                "unit": unit,
                "verdict": "PASS" if ok else "FAIL",
            }
        )

    ref_cou = next(c for c in core["couette"] if c["ny"] == 34 and c["tau"] == 0.8)
    e_top, e_bot = ref_cou["tau_err_pct"], ref_cou["tau_bot_err_pct"]
    worst = max(abs(e_top), abs(e_bot))
    add(
        "M1",
        "Couette steady wall shear rel. error, worst of both walls (H=32,tau=0.8,U0=0.05)",
        worst,
        15.0,
        worst <= 15.0,
    )
    add(
        "M2",
        "Couette steady force balance |F_bot+F_top|/|F_top|",
        ref_cou["force_balance_rel"],
        1e-6,
        ref_cou["force_balance_rel"] <= 1e-6,
        "-",
    )

    ref_st = next(c for c in core["stokes"] if c["tag"] == "REF")
    add(
        "M3",
        "Stokes-2nd wall-force amplitude error",
        abs(ref_st["amp_err_pct"]),
        15.0,
        abs(ref_st["amp_err_pct"]) <= 15.0,
    )
    add(
        "M4",
        "Stokes-2nd wall-force phase error",
        abs(ref_st["phase_err_deg"]),
        15.0,
        abs(ref_st["phase_err_deg"]) <= 15.0,
        "deg",
    )

    ref_ray = next(c for c in core["rayleigh"] if c["tag"] == "REF")
    add(
        "M5",
        "Rayleigh start-up tau_w(t) max rel. error over t in [400,1600]",
        ref_ray["err_pct_max_in_win"],
        15.0,
        ref_ray["err_pct_max_in_win"] <= 15.0,
    )

    # M6 is normalised by an *analytic* force scale, never by the measured one:
    # a wall closure that produces no force at all (e.g. static_bb) must not be
    # able to pass the identity by dividing a tiny residual by a tinier force.
    def _rel(c):
        case = c["case"]
        if case == "couette_shear":
            scale = abs(c["tau_ana"]) * c["nx"]
        elif case == "stokes2_shear":
            scale = abs(c["amp_closed_form"]) * c["nx"]
        else:  # rayleigh_shear
            scale = max(abs(np.array(c["fx_ana"]))) * c["nx"]
        return c["me_identity_res_abs"] / max(scale, 1e-30)

    allrel = {
        c["case"] + ":" + str(c.get("variant", "")) + str(c.get("ny")): _rel(c)
        for c in (
            core["couette"] + core["stokes"] + core["rayleigh"] + abl["stokes"] + abl["couette"]
        )
    }
    worst_id = max(allrel.values())
    worst_key = max(allrel, key=allrel.get)
    add(
        "M6",
        "ME bookkeeping identity |F_bot+F_top+dP| / tau*A (worst over all runs)",
        {
            "worst": worst_id,
            "run": worst_key,
            "per_run": {k: v for k, v in sorted(allrel.items()) if v > 1e-15},
        },
        1e-12,
        worst_id <= 1e-12,
        "-",
    )
    add(
        "M7",
        "Ladd impulse identity sum_{c_q.n>0} c_q inj_q = rho_w u_w (relative)",
        ident["rel_res"],
        1e-14,
        ident["rel_res"] <= 1e-14,
        "-",
    )

    amap = {c["variant"]: c for c in abl["stokes"]}
    e_ref = abs(amap["ladd_pre_local"]["amp_err_pct"])
    e_eq = (
        abs(amap["eq_wall_node"]["amp_err_pct"])
        if np.isfinite(amap["eq_wall_node"]["amp_err_pct"])
        else float("inf")
    )
    e_static = (
        abs(amap["static_bb"]["amp_err_pct"])
        if np.isfinite(amap["static_bb"]["amp_err_pct"])
        else 100.0
    )
    add(
        "M8",
        "ablation ranking: err(ladd) < err(eq_wall_node) and err(static_bb) >= 50",
        {"ladd": e_ref, "eq_wall_node": e_eq, "static_bb": e_static},
        "strict",
        (e_ref < e_eq and e_static >= 50.0),
        "%",
    )

    n_fail = sum(1 for g in gates if g["verdict"] == "FAIL")
    summary = {
        "anchor": "XVAL-LBM-MOVING-WALL-WALL-SHEAR-ME-D2Q9-BGK",
        "scope": "wall force / momentum exchange of the merged halfway moving-wall "
        "bounce-back closure (D2Q9 + BGK, float64), vs analytic Couette, "
        "Stokes-2nd and Rayleigh references",
        "predeclared": "PREREG.md",
        "closure": "ladd_pre_local (pre-streaming reflection + Ladd term), bit-identical "
        "to benchmarks/cross_validation/moving_wall_halfway_bb_d2q9_bgk/mwbb.py::make_step",
        "gate": gates,
        "verdict": "PASS" if n_fail == 0 else "FAIL",
        "n_fail": n_fail,
        "reference_config": {"couette": REF_COUETTE, "stokes2": REF},
        "couette": [{k: v for k, v in c.items()} for c in core["couette"]],
        "stokes2": [{k: v for k, v in c.items()} for c in core["stokes"]],
        "rayleigh": [{k: v for k, v in c.items()} for c in core["rayleigh"]],
        "ablation_stokes2": [
            {k: v for k, v in c.items() if k not in ("t", "fx_num", "fx_ana")}
            for c in abl["stokes"]
        ],
        "ablation_couette": [{k: v for k, v in c.items() if k != "u_num"} for c in abl["couette"]],
        "impulse_identity": ident,
        "normal_velocity_diagnostic": diag,
    }
    (ART / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _load(prefix: str, key: str | None = None) -> list:
    out = []
    for p in sorted(ART.glob(f"case_{prefix}*.json")):
        out.append(json.loads(p.read_text()))
    if key is not None:
        return sorted(out, key=lambda r: str(r.get(key, "")))
    return out


def main() -> None:
    global ART

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage", default="all", choices=["all", "core", "ablation", "diag", "finalize"]
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    ART = a.out
    ART.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ident = _dump("impulse_identity", mf.impulse_identity())
    print("M7 impulse identity rel_res =", ident["rel_res"], flush=True)
    if a.stage == "finalize":
        core = {
            "couette": _load("couette_shear", "ny"),
            "stokes": _load("stokes2_shear", "tag"),
            "rayleigh": _load("rayleigh_shear", "tag"),
        }
        abl = {"stokes": _load("stokes2_abl"), "couette": _load("couette_abl")}
        diag = json.loads((ART / "case_normal_velocity_diagnostic.json").read_text())
    else:
        core = run_core() if a.stage in ("all", "core") else None
        abl = run_ablation() if a.stage in ("all", "ablation") else None
        diag = (
            _dump("normal_velocity_diagnostic", run_normal_diag())
            if a.stage in ("all", "diag")
            else None
        )
        print("stage", a.stage, "done in", round(time.time() - t0, 1), "s", flush=True)
        if core is None or abl is None or diag is None:
            return
    s = finalize(core, abl, diag, ident)
    print(json.dumps(s["gate"], indent=1))
    print("VERDICT:", s["verdict"], "fails:", s["n_fail"], "total", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
