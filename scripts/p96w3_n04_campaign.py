#!/usr/bin/env python
"""P96W3-N04 战役运行器：热对流定量验证面（RB 临界 Ra / Nu–Ra / 无量纲自洽）。

每个 ``--task`` 独立产出一个 JSON（``artifacts/p96w3_n04/<task>_<tag>.json``），
可任意分片并行（集群 ``--cmd-tpl`` 或本地串行），零手抄：所有数字直接落盘。

用法示例::

    PYTHONPATH=src python scripts/p96w3_n04_campaign.py --task c1-rac --ny 32
    PYTHONPATH=src python scripts/p96w3_n04_campaign.py --task c2 --ra 1e4 --ny 60
    PYTHONPATH=src python scripts/p96w3_n04_campaign.py --task quick
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm import rb_convection_validation as V  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "p96w3_n04"


def _dump(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    payload = dict(payload)
    payload.setdefault("generated_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    payload["script"] = "scripts/p96w3_n04_campaign.py"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float))
    print(f"[wrote] {path}", flush=True)
    return path


# ── C5 负控与极限 ───────────────────────────────────────────────────────────


def task_geom(args) -> dict:
    """几何口径标定：温度/速度的有效间隙（pre vs post 两种壁面处理）。"""
    torch.set_num_threads(args.threads)
    rows = {}
    for bc in ("pre", "post"):
        rows[bc] = V.calibrate_gaps(ny=args.ny, nx=args.nx or 2 * args.ny,
                                    alpha=1.0 / 6.0, steps=args.steps or 4000,
                                    wall_bc=bc)
    return {"rows": rows,
            "criteria": {"d_equal_post": rows["post"]["consistent"],
                         "dT_over_ny_post": rows["post"]["thermal"]["d_over_ny"],
                         "dT_over_ny_pre": rows["pre"]["thermal"]["d_over_ny"],
                         "dv_over_ny_pre": rows["pre"]["velocity"]["d_over_ny"]}}


def task_c5(args) -> dict:
    rows = []
    for ny in (32, 64, 128):
        steps = int(max(4000, 0.5 * ny**2 / (1.0 / 6.0)))
        r = V.no_buoyancy_control(ny=ny, steps=steps)
        r["steps"] = steps
        rows.append(r)
    dhc = [V.no_buoyancy_control(ny=ny, steps=int(max(4000, 0.5 * ny**2 / (1.0 / 6.0))),
                                 case="dhc") for ny in (32, 64)]
    enth = V.adiabatic_enthalpy_drift(ny=32, nx=64, steps=10000)
    return {"rb_no_buoyancy": rows, "dhc_no_buoyancy": dhc,
            "enthalpy_drift": enth,
            "criteria": {
                "nu_error_max": max(r["nu_error"] for r in rows + dhc),
                "profile_linf_over_dT_max": max(r["profile_linf_over_dT"] for r in rows),
                "enthalpy_drift_rel": enth["drift_rel"],
            }}


# ── C6 精度阶（正弦波衰减反演 α）───────────────────────────────────────────


def task_c6(args) -> dict:
    out = V.diffusion_decay_order(ns=(16, 32, 64, 128))
    ratios = [r["ratio"] for r in out["ratios"] if math.isfinite(r["ratio"])]
    return {**out,
            "criteria": {
                "alpha_rel_error_max": max(r["alpha_rel_error_abs"] for r in out["rows"]),
                "error_ratio_min": min(ratios) if ratios else float("nan"),
                "error_ratio_max": max(ratios) if ratios else float("nan"),
            }}


# ── C1 线性稳定性 ──────────────────────────────────────────────────────────


def task_c1_rac(args) -> dict:
    torch.set_num_threads(args.threads)
    kw = dict(ny=args.ny, nx=args.nx, mode=args.mode,
              alpha=1.0 / 6.0,
              steps=args.steps if args.steps > 0 else None,
              sample_every=args.sample_every if args.sample_every > 0 else None)
    coarse = V.critical_rayleigh_measured(ratios=(0.90, 0.95, 1.00, 1.05, 1.10), **kw)
    fine = V.critical_rayleigh_measured(ratios=(0.95, 0.975, 1.00, 1.025, 1.05), **kw)
    return {
        "coarse_step_0.05": coarse,
        "fine_step_0.025": fine,
        "ra_step_sensitivity_pct": 100.0 * abs(coarse["ra_c"] - fine["ra_c"]) / V.RA_CRIT,
        "criteria": {
            "ra_c_error_pct_coarse": coarse["ra_c_error_pct"],
            "ra_c_error_pct_fine": fine["ra_c_error_pct"],
            "kH": coarse["kH"],
        },
    }


def task_c1_kc(args) -> dict:
    torch.set_num_threads(args.threads)
    boxes = None
    if args.modes:
        boxes = []
        for spec in args.modes.split(","):
            nx_s, m_s = spec.split(":")
            boxes.append((int(nx_s), int(m_s)))
    out = V.critical_wavenumber_measured(
        ny=args.ny, ra=float(args.ra), boxes=boxes, alpha=1.0 / 6.0,
        steps=args.steps if args.steps > 0 else None,
        sample_every=args.sample_every if args.sample_every > 0 else None,
    )
    return {**out, "criteria": {"k_c_error_pct": out["k_c_error_pct"]}}


def task_c1_operator(args) -> dict:
    """独立交叉检查：离散格子线性算子（谱方法）给的 Ra_c 与 k_c。

    与本模块时间推进完全独立的方法（Jacobian 特征值 bisection），但共用
    ``thermal``/``solver`` 底层原语与同一几何口径（H=ny、半程壁、
    α=(τ_T−1/2)/3），故可直接对表。
    """
    from tensorlbm.rb_discrete_stability import (
        RBLatticeParams, critical_rayleigh, wavenumber_scan,
    )

    rows = []
    for ny in (32, 64):
        p = RBLatticeParams(ny=ny, nx=2 * ny, tau=0.855, tau_T=1.0)
        res = critical_rayleigh(p)
        rows.append({"ny": ny, "nx": 2 * ny, "a": p.a, "Ra_c": res["Ra_c"],
                     "rel_err_vs_continuous": res["rel_err_vs_continuous"],
                     "history": res["history"]})
    # 算子层面的 k_c：固定 ny=32，扫 nx 得 a_c（离散）
    scan = []
    for nx in (104, 112, 120, 128, 136):
        p = RBLatticeParams(ny=32, nx=nx, Ra=1900.0, tau=0.855, tau_T=1.0)
        from tensorlbm.rb_discrete_stability import growth_rate
        scan.append({"nx": nx, "a": p.a, "gamma_step": growth_rate(p)["gamma_step"]})
    return {"critical": rows, "wavenumber_scan_ny32": scan}


# ── C2 / C3 定常 Nu ────────────────────────────────────────────────────────


def task_c2(args) -> dict:
    torch.set_num_threads(args.threads)
    ra = float(args.ra)
    dtype = torch.float32 if args.fp32 else torch.float64
    out = V.run_to_steady(case="rb", ny=args.ny, nx=args.nx or 2 * args.ny, ra=ra,
                          pr=0.71, ma_max=args.ma_max, dtype=dtype,
                          turnover_target=args.turnover, min_steps=args.min_steps,
                          max_steps=args.max_steps, report_every=args.report_every,
                          verbose=args.verbose)
    lo, hi = V.RB_ROLL_BAND[ra]
    out["band"] = [lo, hi]
    out["nu_in_band"] = bool(lo <= out["nu"] <= hi)
    out["dev_from_band_mid_pct"] = 100.0 * abs(out["nu"] - 0.5 * (lo + hi)) / (0.5 * (lo + hi))
    return out


def task_c3(args) -> dict:
    torch.set_num_threads(args.threads)
    ra = float(args.ra)
    dtype = torch.float32 if args.fp32 else torch.float64
    out = V.run_to_steady(case="dhc", ny=args.ny, ra=ra, pr=0.71, ma_max=args.ma_max,
                          dtype=dtype, turnover_target=args.turnover,
                          min_steps=args.min_steps, max_steps=args.max_steps,
                          report_every=args.report_every, verbose=args.verbose)
    ref = V.DE_VAHL_DAVIS_NU[ra]
    out["nu_ref"] = ref
    out["err_pct"] = 100.0 * abs(out["nu"] - ref) / ref
    return out


# ── C4 无量纲自洽 ──────────────────────────────────────────────────────────


def task_c4(args) -> dict:
    torch.set_num_threads(args.threads)
    ra = float(args.ra)
    out = V.dimensionless_selfconsistency(case=args.case, ny=args.ny, ra=ra,
                                          steps=args.steps, ma_max=args.ma_max)
    return out


# ── quick 组合（本地快速自检）──────────────────────────────────────────────


def task_quick(args) -> dict:
    torch.set_num_threads(args.threads)
    return {
        "c5": task_c5(args),
        "c6": task_c6(args),
        "growth_above": _slim(V.measure_growth_rate(ny=16, nx=32, ra=1.2 * V.RA_CRIT,
                                                    alpha=1.0 / 6.0, steps=6000)),
        "growth_below": _slim(V.measure_growth_rate(ny=16, nx=32, ra=0.8 * V.RA_CRIT,
                                                    alpha=1.0 / 6.0, steps=6000)),
    }


def _slim(d: dict) -> dict:
    d = dict(d)
    d.pop("t", None)
    d.pop("amp", None)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["c1-rac", "c1-kc", "c1-operator", "c2", "c3", "c4",
                             "c5", "c6", "geom", "quick"])
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--nx", type=int, default=0)
    ap.add_argument("--ra", type=str, default="1e4")
    ap.add_argument("--mode", type=int, default=1)
    ap.add_argument("--modes", type=str, default="")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--sample-every", type=int, default=0)
    ap.add_argument("--turnover", type=float, default=150.0)
    ap.add_argument("--min-steps", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=2_000_000)
    ap.add_argument("--report-every", type=int, default=2000)
    ap.add_argument("--ma-max", type=float, default=0.10)
    ap.add_argument("--case", type=str, default="rb")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    t0 = time.time()
    fn = {
        "c1-rac": task_c1_rac, "c1-kc": task_c1_kc, "c1-operator": task_c1_operator,
        "c2": task_c2, "c3": task_c3, "c4": task_c4, "c5": task_c5, "c6": task_c6,
        "quick": task_quick, "geom": task_geom,
    }[args.task]
    result = fn(args)
    tag = args.tag or f"{args.task}"
    result["meta"] = {"task": args.task, "argv": sys.argv[1:],
                      "elapsed_s": round(time.time() - t0, 1),
                      "ny": args.ny, "nx": args.nx, "ra": args.ra}
    _dump(tag, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())