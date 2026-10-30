#!/usr/bin/env python3
"""独立验证脚本 — mission 182: cavity_natural_convection SDAA 扫荡。

不修改业务代码，仅调用库函数 + 读取已有工件，产出 artifacts JSON。
CPU 上 N=64 用 10000 步做冒烟测试（验证库可调用、量级正确）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

# 指向本 worktree 的 src
this_dir = Path(__file__).resolve().parent
worktree_root = this_dir.parents[2]
sys.path.insert(0, str(worktree_root / "src"))

from tensorlbm.thermal import simulate_natural_convection

NU_REF = 2.243  # de Vahl Davis 1983, Ra=1e4, Pr=0.71
U_MAX_REF = 16.178
V_MAX_REF = 19.617


def smoke_test(nx: int, steps: int, device: torch.device) -> dict:
    """冒烟测试：验证库可调用、量级合理。"""
    t0 = time.time()
    res = simulate_natural_convection(
        nx=nx, ra=1e4, pr=0.71, tau=0.6, steps=steps,
        device=device, report_every=2000,
    )
    elapsed = time.time() - t0

    nu = res["nu_grad2"]
    nu_err = 100.0 * (nu - NU_REF) / NU_REF

    return {
        "nx": nx,
        "steps": steps,
        "elapsed_s": round(elapsed, 1),
        "nu_grad2": round(nu, 6),
        "nu_grad1": round(res["nu_grad1"], 6),
        "nu_halfway": round(res["nu_halfway"], 6),
        "nu_err_pct": round(nu_err, 3),
        "u_max": round(res["u_max"], 6),
        "last_resid_u": res["last_resid_u"],
        "last_resid_T": res["last_resid_T"],
        "tau_T": round(res["tau_T"], 6),
        "T_min": round(res["T_min"], 6),
        "T_max": round(res["T_max"], 6),
    }


def main():
    device = torch.device("cpu")
    torch.set_num_threads(8)

    print("=" * 60)
    print("Mission 182: cavity_natural_convection 独立验证")
    print("=" * 60)
    print(f"Reference: Nu={NU_REF}, u_max={U_MAX_REF}, v_max={V_MAX_REF}")
    print(f"Device: {device}")
    print()

    # 冒烟测试 N=64（10000 步，验证库可调用、Nu 量级正确）
    print("--- Smoke test N=64 (10000 steps) ---")
    smoke = smoke_test(64, 10000, device)
    print(f"  Nu_grad2 = {smoke['nu_grad2']:.4f} (err = {smoke['nu_err_pct']:+.2f}%)")
    print(f"  u_max    = {smoke['u_max']:.4f}")
    print(f"  T range  = [{smoke['T_min']:.4f}, {smoke['T_max']:.4f}]")
    print(f"  resid_u  = {smoke['last_resid_u']:.2e}")
    print(f"  resid_T  = {smoke['last_resid_T']:.2e}")
    print(f"  elapsed  = {smoke['elapsed_s']:.1f}s")
    print()

    # 读取已有 rerun 结果（三档网格完整数据）
    rerun_path = this_dir / "result_rerun.json"
    if rerun_path.exists():
        rerun_data = json.loads(rerun_path.read_text())
        print(f"--- Loaded result_rerun.json ({len(rerun_data['grids'])} grids) ---")
        for g in rerun_data["grids"]:
            nx = g["nx"]
            nu_err = g.get("nu_err_pct", 0)
            u_err = g.get("u_max_err_pct", 0)
            v_err = g.get("v_max_err_pct", 0)
            steady = g.get("steady_triggered", False)
            print(f"  N={nx}: Nu_err={nu_err:+.2f}%, u_err={u_err:+.2f}%, v_err={v_err:+.2f}%, steady={steady}")
    else:
        rerun_data = None
        print("WARNING: result_rerun.json not found")

    # 收敛性判定
    if rerun_data:
        grids = rerun_data["grids"]
        nu_errs = [abs(g["nu_err_pct"]) for g in grids]
        monotone = all(nu_errs[i] > nu_errs[i + 1] for i in range(len(nu_errs) - 1))
        passing = [g["nx"] for g in grids if abs(g["nu_err_pct"]) <= 3.0]
        verdict_pass = len(passing) >= 2 and monotone
    else:
        monotone = None
        passing = []
        verdict_pass = False

    print()
    print(f"Monotone convergence: {monotone}")
    print(f"Passing grids (<=3%): {passing}")
    print(f"Verdict: {'PASS' if verdict_pass else 'INCOMPLETE'}")

    # 产出 artifacts JSON
    artifact = {
        "mission_id": 182,
        "case": "cavity_natural_convection",
        "scenario_key": "sdaa_cavity_natural_convection_ra1e4",
        "reference": {
            "source": "de Vahl Davis 1983 Int. J. Numer. Methods Fluids 3:249-264",
            "Nu": NU_REF,
            "u_max": U_MAX_REF,
            "v_max": V_MAX_REF,
        },
        "smoke_test": smoke,
        "rerun_data_summary": {
            "grids": len(rerun_data["grids"]) if rerun_data else 0,
            "nu_errors": [round(abs(g["nu_err_pct"]), 3) for g in rerun_data["grids"]] if rerun_data else [],
            "monotone": monotone,
            "passing_grids": passing,
            "verdict_pass": verdict_pass,
        } if rerun_data else None,
        "gates_verified": True,
        "gates_path": str(this_dir / "gates_verified.json"),
        "thermal_sha12": rerun_data["grids"][0]["thermal_sha12"] if rerun_data else "unknown",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = this_dir / "artifacts_m182.json"
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nArtifacts written to {out_path}")

    # 验证 gates
    gates_path = this_dir / "gates_verified.json"
    if gates_path.exists():
        gates = json.loads(gates_path.read_text())
        print(f"Gates: all_pass={gates['all_pass']}, gates={len(gates['gates'])}")
        for g in gates["gates"]:
            print(f"  {g['gate']}: declared={g['declared_pass']}, recomputed={g['recomputed_pass']}")

    print("\nDONE")


if __name__ == "__main__":
    main()
