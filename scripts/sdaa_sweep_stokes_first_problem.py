#!/usr/bin/env python
"""SDAA 验证案例扫荡：``benchmarks/verified/stokes_first_problem``（B24）。

独立运行脚本（**不改业务代码**：不改 ``benchmarks/verified/stokes_first_problem/run.py``，
只 import 它的 ``run_case`` / ``compare_grids``，把模块级 ``DEVICE`` 指向 ``sdaa``）。

案例既有判据口径（README "判定标准" 表，按 NVIDIA 侧原口径，不降判据）：
  G1  剖面最大相对误差 ``max_y |u_num-u_ana|/u_ana``（限 ``u_ana > 5%·U`` 区域）
      ≤ 3%，对每个记录时刻 t=1000/4000/9000 与每档网格 H=100/200 逐条比对。
  G2  网格收敛：H=200 各时刻 max_rel ≤ max(H=100 值, 3%)（``compare_grids`` 原规则），
      并记录两档剖面差（%·U）作一致性证据。

附加诊断（README "结果" 段落中记录的量，非原判据门，这里如实逐条给出 PASS/FAIL 但标注为诊断）：
  D1  波场有限 ``finite == True``（README: finite=True）
  D2  质量漂移 |mass_drift_pct| ≤ 0.1%（README: ~1e-2%）
  D3  计算张量实际位于 SDAA 设备（禁止 CPU 结果冒充）——逐张量断言 + 设备证据行

绕行记录：入口 ``run.py`` 不 import triton；其 ``--compile-mode default`` 在 SDAA 上走
``teco_inductor``（``torch_sdaa._inductor.codegen.teco``），上游未完成，本脚本用
``compile-mode=eager`` 口径跑主判定（与 ``--compile-mode eager`` 逐位一致），并可选跑一次
``default`` 探测记录 ``compile_status``。全程**未使用** ``TRITON_INTERPRET``，因为它与本案例
的可执行路径无关；若探测发现 triton 介入则如实记录。

用法：
    python scripts/sdaa_sweep_stokes_first_problem.py \
        --out artifacts/sdaa_stokes_first_problem.json [--H 100 200] [--steps 1000 4000 9000]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CASE_DIR = REPO / "benchmarks" / "verified" / "stokes_first_problem"

TOL_MAX_REL_PCT = 3.0  # README 判定标准（原口径，不得放宽）
DIAG_MASS_DRIFT_PCT = 0.1  # README 记录 ~1e-2%；本脚本以 0.1% 作诊断上限


# --------------------------------------------------------------------------- #
# 设备证据
# --------------------------------------------------------------------------- #
def device_evidence() -> dict:
    """收集 SDAA 设备证据行（一条机器可读字符串 + 明细）。"""
    import torch

    ev: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_is_available": bool(torch.cuda.is_available()),
    }
    try:
        import torch_sdaa  # noqa: F401

        ev["torch_sdaa_version"] = getattr(torch_sdaa, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover
        ev["torch_sdaa_version"] = f"IMPORT_FAILED: {exc!r}"

    try:
        n = int(torch.sdaa.device_count())
        ev["sdaa_device_count"] = n
        name = torch.sdaa.get_device_name(0)
        ev["sdaa_device_name_0"] = name
        ev["device_evidence_line"] = f"torch.sdaa.get_device_name(0) = {name!r} (count={n})"
        # 一次真实小算子，证明设备可用（不是只有名字）
        probe = torch.randn(8, 8, device="sdaa")
        ev["sdaa_smoke_probe"] = float((probe + 1.0).sum().item())
        ev["sdaa_smoke_probe_device"] = str((probe + 1.0).device)
        try:
            props = torch.sdaa.get_device_properties(0)
            ev["sdaa_device_properties"] = str(props)
        except Exception as exc:
            ev["sdaa_device_properties"] = f"UNAVAILABLE: {exc!r}"
    except Exception as exc:
        ev["device_evidence_line"] = f"SDAA_UNAVAILABLE: {exc!r}"
        ev["sdaa_device_count"] = 0

    try:
        ev["triton_installed"] = importlib.util.find_spec("triton") is not None
    except Exception:
        ev["triton_installed"] = None
    ev["triton_interpret_env"] = __import__("os").environ.get("TRITON_INTERPRET")
    return ev


# --------------------------------------------------------------------------- #
# 加载案例业务代码（只读，不修改）
# --------------------------------------------------------------------------- #
def load_case_module(device: str):
    """把案例 run.py 作为模块导入，并把其模块级 DEVICE 指向给定设备。"""
    import torch

    sys.path.insert(0, str(CASE_DIR.parent))  # 供 run.py 内的 compile_route import
    sys.path.insert(0, str(CASE_DIR))
    spec = importlib.util.spec_from_file_location("stokes_first_run", CASE_DIR / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CASE_DIR / 'run.py'}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stokes_first_run"] = mod
    spec.loader.exec_module(mod)

    target = torch.device(device)
    mod.DEVICE = target  # run_case 内部读模块全局 DEVICE
    return mod, target


def assert_device_tensors(mod, target, H: int, nx: int) -> dict:
    """在真正跑之前，用案例原语构一个态并断言其落在目标设备上。"""
    import torch

    ny = H + 2
    rho0 = torch.ones((ny, nx), dtype=torch.float32, device=target)
    u0 = torch.zeros((ny, nx), dtype=torch.float32, device=target)
    f = mod.equilibrium(rho0, u0, u0)
    _, ux, _ = mod.macroscopic(f)
    out = {
        "f_device": str(f.device),
        "ux_device": str(ux.device),
        "is_target_device": f.device.type == target.type and ux.device.type == target.type,
        "SPECULAR_tensor_device": str(mod.SPECULAR.device),
    }
    return out


# --------------------------------------------------------------------------- #
# 判据比对
# --------------------------------------------------------------------------- #
def evaluate_gates(cases: list[dict], conv: dict, U: float) -> list[dict]:
    """逐条判据 PASS/FAIL（全部数值在此由脚本从跑出的结果算出，零手抄）。"""
    gates: list[dict] = []

    # --- G1: 每 (网格, 时刻) 的剖面最大相对误差 ≤ 3% ---
    for r in cases:
        H = r["H"]
        for s in r["per_step"]:
            val = float(s["max_rel_err_pct"])
            gates.append(
                {
                    "id": "G1",
                    "criterion": "剖面最大相对误差 ≤ 3%（u_ana > 5%·U 区域）",
                    "scope": f"H={H} t={s['t_lb']}",
                    "value": val,
                    "unit": "%",
                    "threshold": f"<= {TOL_MAX_REL_PCT}",
                    "pass": bool(val <= TOL_MAX_REL_PCT),
                    "source": "README 判定标准 / run.py per_step.max_rel_err_pct",
                }
            )

    # --- G2: 网格收敛（compare_grids 原规则 + 剖面一致性证据）---
    err_by_H = {r["H"]: [s["max_rel_err_pct"] for s in r["per_step"]] for r in cases}
    steps = [s["t_lb"] for s in cases[0]["per_step"]]
    lo_H, hi_H = cases[0]["H"], cases[1]["H"]
    for i, t in enumerate(steps):
        el = float(err_by_H[lo_H][i])
        eh = float(err_by_H[hi_H][i])
        bound = max(el, TOL_MAX_REL_PCT)
        row = conv["rows"][i] if i < len(conv.get("rows", [])) else {}
        gates.append(
            {
                "id": "G2",
                "criterion": "网格收敛：H_hi 的 max_rel ≤ max(H_lo 值, 3%)",
                "scope": f"t={t} (H_lo={lo_H} err={el:.4f}%, H_hi={hi_H} err={eh:.4f}%)",
                "value": eh,
                "unit": "%",
                "threshold": f"<= {bound:.4f}",
                "profile_diff_over_U_pct": row.get("max_profile_diff_over_U_pct"),
                "pass": bool(eh <= bound),
                "source": "README 判定标准「网格收敛」/ run.py compare_grids",
            }
        )
    gates.append(
        {
            "id": "G2",
            "criterion": "网格收敛总体（compare_grids().converged）",
            "scope": f"H={lo_H} vs H={hi_H}",
            "value": bool(conv["converged"]),
            "threshold": "True",
            "pass": bool(conv["converged"]),
            "source": "README 判定标准「网格收敛」",
        }
    )

    # --- D1: 有限性诊断 ---
    for r in cases:
        gates.append(
            {
                "id": "D1",
                "criterion": "[诊断] 波场有限 finite == True",
                "scope": f"H={r['H']}",
                "value": bool(r["finite"]),
                "threshold": "True",
                "pass": bool(r["finite"]),
                "source": "README 结果段（非原判据门）",
            }
        )

    # --- D2: 质量漂移诊断 ---
    for r in cases:
        drift = float(r["mass_drift_pct"])
        gates.append(
            {
                "id": "D2",
                "criterion": f"[诊断] |质量漂移| ≤ {DIAG_MASS_DRIFT_PCT}%",
                "scope": f"H={r['H']}",
                "value": drift,
                "unit": "%",
                "threshold": f"<= {DIAG_MASS_DRIFT_PCT}",
                "pass": bool(abs(drift) <= DIAG_MASS_DRIFT_PCT),
                "source": "README 结果段 ~1e-2%（诊断上限由本脚本设定）",
            }
        )
    return gates


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="SDAA sweep: stokes_first_problem")
    ap.add_argument("--device", default="sdaa")
    ap.add_argument("--H", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--steps", type=int, nargs="+", default=[1000, 4000, 9000])
    ap.add_argument("--tau", type=float, default=0.65)
    ap.add_argument("--U", type=float, default=0.05)
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--compile-mode", default="eager")
    ap.add_argument("--out", default=str(REPO / "artifacts" / "sdaa_stokes_first_problem.json"))
    ap.add_argument(
        "--probe-compile-default",
        action="store_true",
        help="额外探测一次 compile-mode=default（SDAA teco_inductor 路径）并记录 compile_status",
    )
    args = ap.parse_args()

    t_start = time.time()
    ev = device_evidence()
    print(f"[device] {ev['device_evidence_line']}", flush=True)
    print(f"[device] torch={ev['torch_version']} torch_sdaa={ev.get('torch_sdaa_version')}", flush=True)

    record: dict = {
        "case": "stokes_first_problem",
        "case_path": str(CASE_DIR.relative_to(REPO)),
        "entry": "benchmarks/verified/stokes_first_problem/run.py",
        "device": args.device,
        "device_evidence_line": ev.get("device_evidence_line"),
        "device_evidence": ev,
        "config": {
            "H_list": args.H,
            "record_steps": args.steps,
            "tau": args.tau,
            "nu_lb": (args.tau - 0.5) / 3.0,
            "U": args.U,
            "nx": args.nx,
            "dtype": "float32",
            "compile_mode": args.compile_mode,
            "tol_max_rel_pct": TOL_MAX_REL_PCT,
        },
        "workaround": {
            "triton_dependency_of_entry": False,
            "TRITON_INTERPRET": ev.get("triton_interpret_env"),
            "triton_interpret_used": False,
            "note": (
                "案例入口 run.py 不 import triton；torch.compile 在 SDAA 上走 teco_inductor "
                "(torch_sdaa._inductor.codegen.teco) 而非 triton，故 TRITON_INTERPRET=1 无意义/不适用。"
                "主判定采用 compile-mode=eager（与 run.py --compile-mode eager 逐位一致），"
                "绕开 compile_route 已记录的 SDAA teco inductor L5 缺口。"
            ),
        },
        "status": "RUNNING",
    }

    try:
        mod, target = load_case_module(args.device)
        if ev.get("sdaa_device_count", 0) < 1:
            raise RuntimeError(f"SDAA unavailable: {ev['device_evidence_line']}")
        pre = assert_device_tensors(mod, target, args.H[0], args.nx)
        record["preflight_device_assert"] = pre
        print(f"[preflight] {pre}", flush=True)
        if not pre["is_target_device"]:
            raise RuntimeError(f"tensors did not land on {target}: {pre}")

        cases: list[dict] = []
        for H in args.H:
            t0 = time.time()
            r = mod.run_case(
                H,
                args.tau,
                args.U,
                args.steps,
                args.nx,
                compile_mode=args.compile_mode,
            )
            wall = time.time() - t0
            r["wall_s"] = round(wall, 2)
            cases.append(r)
            print(
                f"[run] H={H:3d} ny={r['ny']} "
                + "  ".join(
                    f"t{s['t_lb']}={s['max_rel_err_pct']:.4f}%" for s in r["per_step"]
                )
                + f"  finite={r['finite']} mass_drift={r['mass_drift_pct']:.3e}%"
                + f"  wall={wall:.1f}s",
                flush=True,
            )

        conv = mod.compare_grids(cases[0], cases[1], args.U)
        gates = evaluate_gates(cases, conv, args.U)
        all_pass = all(g["pass"] for g in gates)

        # 轻量落盘结果表（只留剖面摘要，避免 JSON 过大；剖面全量放 profiles）
        result_table = []
        for r in cases:
            for s in r["per_step"]:
                result_table.append(
                    {
                        "H": r["H"],
                        "t_lb": s["t_lb"],
                        "delta_lb": s["delta_lb"],
                        "max_rel_err_pct": s["max_rel_err_pct"],
                        "l2_rel_err": s["l2_rel_err"],
                        "max_abs_err_over_U_pct": s["max_abs_err_over_U_pct"],
                        "u_wall_num_over_U": s["u_wall_num"] / args.U,
                        "u_top_num_over_U": s["u_top_num"] / args.U,
                    }
                )

        record.update(
            {
                "status": "OK",
                "all_pass": all_pass,
                "n_pass": sum(1 for g in gates if g["pass"]),
                "n_total": len(gates),
                "result_table": result_table,
                "grid_convergence": {
                    "rows": conv["rows"],
                    "converged": conv["converged"],
                    "max_rel_err_by_H": {str(k): v for k, v in
                                         {r["H"]: [s["max_rel_err_pct"] for s in r["per_step"]]
                                          for r in cases}.items()},
                },
                "gates": gates,
                "per_case": [
                    {
                        "H": r["H"],
                        "ny": r["ny"],
                        "nx": r["nx"],
                        "tau": r["tau"],
                        "nu_lb": r["nu_lb"],
                        "U": r["U"],
                        "Ma": r["Ma"],
                        "mass_drift_pct": r["mass_drift_pct"],
                        "finite": r["finite"],
                        "elapsed_s": r["elapsed_s"],
                        "wall_s": r["wall_s"],
                        "compile_mode": r.get("compile_mode"),
                        "profiles": {
                            str(s["t_lb"]): {
                                "y": s["y_profile"],
                                "u_num": s["u_profile"],
                                "u_ana": s["u_analytic"],
                            }
                            for s in r["per_step"]
                        },
                    }
                    for r in cases
                ],
            }
        )

        # 可选：探测 compile-mode=default（记录 compile_status / 是否 eager 回退）
        if args.probe_compile_default:
            probe = {
                "attempted": True,
                "purpose": "记录 SDAA 上 torch.compile(teco_inductor) 路径是否可用（绕行证据）",
                "probe_only_not_a_criterion": True,
                "note": "H=32/t=50 只是极小探测档，边界层仅 ~1.6 格，其 max_rel 不代表判据",
            }
            try:
                import contextlib
                import io

                buf = io.StringIO()
                t0 = time.time()
                with contextlib.redirect_stdout(buf):
                    rp = mod.run_case(
                        32, args.tau, args.U, [50], args.nx, compile_mode="default"
                    )
                banner = [
                    ln
                    for ln in buf.getvalue().splitlines()
                    if "compile_route" in ln or "compile" in ln.lower()
                ]
                probe.update(
                    {
                        "ok": True,
                        "H": 32,
                        "steps": [50],
                        "requested_compile_mode": rp.get("compile_mode"),
                        "max_rel_err_pct": rp["per_step"][0]["max_rel_err_pct"],
                        "wall_s": round(time.time() - t0, 2),
                        "compile_route_banner": banner,
                        "fell_back_to_eager": any("EAGER FALLBACK" in ln for ln in banner),
                    }
                )
            except Exception as exc:
                probe.update(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": traceback.format_exc().splitlines()[-6:],
                    }
                )
            record["compile_default_probe"] = probe
            print(f"[probe default] {json.dumps({k: v for k, v in probe.items() if k != 'traceback_tail'}, ensure_ascii=False)}", flush=True)

    except Exception as exc:
        record.update(
            {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc().splitlines()[-15:],
                "all_pass": False,
            }
        )
        print(f"[FAIL] {record['error']}", flush=True)

    record["wall_clock_s"] = round(time.time() - t_start, 2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False))

    print(f"[gates] {record.get('n_pass', 0)}/{record.get('n_total', 0)} PASS  all_pass={record.get('all_pass')}")
    for g in record.get("gates", []):
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {g['id']}  {g['scope']}  "
              f"value={g['value']} thr={g['threshold']}")
    print(f"[out] {out}  wall_clock={record['wall_clock_s']}s")
    return 0 if record.get("all_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())