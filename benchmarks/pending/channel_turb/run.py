#!/usr/bin/env python
"""B26: 湍流通道 Re_tau=180 —— u+ 剖面验证（run + 分析 + 判定）。

真实模拟（无外推）：直接调用共性模块 tensorlbm.turbulent_channel 的
run_turbulent_channel（D2Q9 + Smagorinsky LES + 体积力驱动 + 上下壁
流后反弹 + 流向周期），步进与平均完全由库完成；本脚本只做后处理。

参考：
  - DNS: Moser, Kim & Mansour (1999), Phys. Fluids 11, 943
    (数值方法 Kim, Moin & Moser 1987, JFM 177, 133; Re_tau=178.12)
    官方数据库 https://turbulence.ices.utexas.edu/  chan180/profiles/chan180.means
  - 对数律: u+ = (1/0.41) ln(y+) + 5.0

判定（按 benchmarks/problems.md B26）：
  - y+ ∈ [30, 100] 区间 u+ 剖面相对 DNS 的 RMS 误差 ≤ 3% 且
    相对对数律 RMS 误差 ≤ 3%（对数律仅作副基准，DNS 为主基准），
  - ≥2 档网格同参数对比（收敛性证据），
  - 稳态证据（统计窗口内 max|u| 不再单调增长 / 达到力平衡）。

用法:
  run.py analyze <run_dir> [--out result.json] [--wall-offset 0.5]
  run.py scan --out-root <dir> --steps 100000 --avg-start 60000 [--grids "128x32 256x64"]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

import numpy as np

from tensorlbm.turbulent_channel import TurbulentChannelConfig, run_turbulent_channel

KAPPA = 0.41
B_LOG = 5.0
DNS_CSV = Path(__file__).resolve().parent / "dns_ref" / "mkm180.csv"


# --------------------------------------------------------------------------
# DNS reference (MKM1999 / KMM1987 numerics, Re_tau=178.12)
# --------------------------------------------------------------------------
def load_dns(path: Path = DNS_CSV) -> tuple[np.ndarray, np.ndarray]:
    yp, up = [], []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            yp.append(float(row["y_plus"]))
            up.append(float(row["u_plus"]))
    return np.array(yp), np.array(up)


def dns_uplus_at(y_plus: float, dns_yp: np.ndarray, dns_up: np.ndarray) -> float:
    """Linear interpolation of the DNS u+ profile at the given y+."""
    if y_plus <= dns_yp[0] or y_plus >= dns_yp[-1]:
        return float("nan")
    return float(np.interp(y_plus, dns_yp, dns_up))


def log_law_uplus(y_plus: float, kappa: float = KAPPA, B: float = B_LOG) -> float:
    return math.log(y_plus) / kappa + B


# --------------------------------------------------------------------------
# Analysis of a completed run directory
# --------------------------------------------------------------------------
def analyze_run(
    run_dir: Path,
    *,
    dns_path: Path = DNS_CSV,
    y_plus_min: float = 30.0,
    y_plus_max: float = 100.0,
    wall_offset: float = 0.5,
    log_kappa: float = KAPPA,
    log_B: float = B_LOG,
) -> dict:
    """Compute u+ profile, compare with DNS and log law, return metrics dict."""
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    cfg = meta["config"]
    derived = meta["derived"]
    u_tau = float(cfg["u_tau"])
    nu = float(derived["nu"])
    ny = int(cfg["ny"])

    # module profile: rows 1..ny-2, y_plus = row*(u_tau/nu), u_plus = u/u_tau
    rows = []
    with open(run_dir / "velocity_profile.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((float(r["y"]), float(r["y_plus"]), float(r["u_plus"])))

    # corrected wall distance: no-slip wall at y = wall_offset (half-way BB -> 0.5)
    y_plus_corr = [(y - wall_offset) * u_tau / nu for y, _, _ in rows]
    u_plus = [u for _, _, u in rows]

    dns_yp, dns_up = load_dns(dns_path)
    sel = [
        (yp, up, dns_uplus_at(yp, dns_yp, dns_up), log_law_uplus(yp, log_kappa, log_B))
        for yp, up in zip(y_plus_corr, u_plus, strict=False)
        if y_plus_min <= yp <= y_plus_max
    ]
    n_pts = len(sel)
    if n_pts == 0:
        return {"error": "no points in comparison window"}

    up_arr = np.array([s[1] for s in sel])
    dns_arr = np.array([s[2] for s in sel])
    log_arr = np.array([s[3] for s in sel])

    rms_dns = float(np.sqrt(np.mean((up_arr - dns_arr) ** 2)))
    rms_log = float(np.sqrt(np.mean((up_arr - log_arr) ** 2)))
    rel_dns = float(np.mean(np.abs(up_arr - dns_arr) / dns_arr))
    rel_log = float(np.mean(np.abs(up_arr - log_arr) / log_arr))

    # steady-state evidence from diagnostics
    diags = meta.get("diagnostics", [])
    steady = False
    maxu_last = None
    maxu_prev = None
    if len(diags) >= 2:
        maxu_last = float(diags[-1]["max_speed"])
        maxu_prev = float(diags[-2]["max_speed"])
        # not monotone growth in the last two diagnostic samples => plateaued-ish
        steady = maxu_last <= maxu_prev * 1.02

    # measured u_tau from wall-shear balance: tau_w = rho*a_x*H/2  ->  u_tau_meas
    a_x = float(derived["body_force"])
    H = float(derived["H"])
    u_tau_meas = math.sqrt(a_x * H / 2.0)

    # turbulence check from stats (if enabled)
    ts = meta.get("engineering_closure", {}).get("turbulence_statistics_runtime", {})
    turb_check = None
    if ts:
        vv = float(ts.get("vv_mean", 0.0))
        uu = float(ts.get("uu_mean", 0.0))
        uv = float(ts.get("uv_mean", 0.0))
        turb_check = {
            "uu_mean": uu,
            "vv_mean": vv,
            "uv_mean": uv,
            "has_reynolds_stress": abs(uv) > 1e-6 * max(abs(uu), 1e-12),
            "wall_normal_fluctuations": vv > 0.05 * max(uu, 1e-12),
        }

    return {
        "config": {
            "nx": cfg["nx"],
            "ny": cfg["ny"],
            "re_tau": cfg["re_tau"],
            "u_tau": cfg["u_tau"],
            "smagorinsky_cs": cfg["smagorinsky_cs"],
            "n_steps": cfg["n_steps"],
            "averaging_start": cfg["averaging_start"],
        },
        "derived": {"nu": nu, "tau": derived["tau"], "H": H, "body_force": a_x},
        "wall_model": {
            "bb_type": "post-streaming bounce-back",
            "wall_offset_assumed": wall_offset,
            "delta_y_plus_per_cell": u_tau / nu,
        },
        "u_tau_target": u_tau,
        "u_tau_force_balance": u_tau_meas,
        "comparison_window": {
            "y_plus_min": y_plus_min,
            "y_plus_max": y_plus_max,
            "n_points": n_pts,
        },
        "errors": {
            "rms_vs_dns": rms_dns,
            "rms_vs_loglaw": rms_log,
            "mean_rel_vs_dns": rel_dns,
            "mean_rel_vs_loglaw": rel_log,
        },
        "steady_state": {
            "last_max_speed": maxu_last,
            "prev_max_speed": maxu_prev,
            "plateaued": steady,
        },
        "turbulence_check": turb_check,
        "log_law_rms_error_module_window": meta.get("log_law_rms_error"),
        "averaging_samples": meta.get("averaging_samples"),
        "elapsed_s": meta.get("elapsed_s"),
        "profile": {
            "y_plus": [round(v, 4) for v in y_plus_corr],
            "u_plus": [round(v, 6) for v in u_plus],
        },
    }


def verdict(results: list[dict], threshold: float = 0.03) -> dict:
    ok = []
    for r in results:
        if "error" in r:
            ok.append(False)
            continue
        errs = r["errors"]
        passed = errs["rms_vs_dns"] <= threshold and errs["rms_vs_loglaw"] <= threshold
        ok.append(passed)
    return {
        "threshold": threshold,
        "all_within_threshold": all(ok) if ok else False,
        "n_cases": len(results),
        "n_passed": sum(ok),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def cmd_analyze(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    res = analyze_run(run_dir, wall_offset=args.wall_offset)
    out = Path(args.out)
    out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    errs = res.get("errors", {})
    print(json.dumps(res, indent=2, sort_keys=True))
    print(f"-> {out}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for grid in args.grids:
        nx, ny = (int(v) for v in grid.lower().split("x"))
        name = f"retau{args.re_tau:g}_nx{nx}_ny{ny}_steps{args.steps}"
        config = TurbulentChannelConfig(
            nx=nx,
            ny=ny,
            re_tau=args.re_tau,
            u_tau=args.u_tau,
            smagorinsky_cs=args.cs,
            n_steps=args.steps,
            averaging_start=args.avg_start,
            output_interval=args.output_interval,
            output_root=out_root,
            run_name=name,
            seed=args.seed,
            device=args.device,
            overwrite=True,
        )
        t0 = time.time()
        run_dir = run_turbulent_channel(config)
        elapsed = time.time() - t0
        meta_path = run_dir / "run_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["elapsed_s"] = round(elapsed, 1)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        res = analyze_run(run_dir, wall_offset=args.wall_offset)
        res["run_dir"] = str(run_dir)
        res["elapsed_s"] = round(elapsed, 1)
        results.append(res)
        print(
            f"[{grid}] rms_vs_dns={res.get('errors', {}).get('rms_vs_dns')} "
            f"rms_vs_loglaw={res.get('errors', {}).get('rms_vs_loglaw')} steady={res.get('steady_state')}"
        )

    v = verdict(results)
    summary = {
        "case": "B26_channel_turb_retau180",
        "library_module": "tensorlbm.turbulent_channel",
        "verdict": v,
        "results": results,
    }
    out = out_root / "scan_result.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(v, indent=2))
    print(f"-> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="B26 turbulent channel u+ validation")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Analyze an existing run directory")
    a.add_argument("run_dir")
    a.add_argument("--out", default="result.json")
    a.add_argument("--wall-offset", dest="wall_offset", type=float, default=0.5)
    a.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("scan", help="Run the two-grid comparison via run_turbulent_channel")
    s.add_argument("--out-root", dest="out_root", default="/tmp/channel_runs")
    s.add_argument("--grids", nargs="+", default=["128x32", "256x64"])
    s.add_argument("--re-tau", dest="re_tau", type=float, default=180.0)
    s.add_argument("--u-tau", dest="u_tau", type=float, default=0.005)
    s.add_argument("--cs", type=float, default=0.1)
    s.add_argument("--steps", type=int, default=100000)
    s.add_argument("--avg-start", dest="avg_start", type=int, default=60000)
    s.add_argument("--output-interval", dest="output_interval", type=int, default=5000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default="cpu")
    s.add_argument("--wall-offset", dest="wall_offset", type=float, default=0.5)
    s.set_defaults(fn=cmd_scan)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
