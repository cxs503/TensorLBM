#!/usr/bin/env python3
"""Publish the m18222 audit report (six elements) to tensor_reports."""
import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
HERE = Path("/root/tp-agent/work/m18222/docs/audit_m18222")
S = json.loads((HERE / "audit_summary.json").read_text())
PROV = json.loads((HERE / "provenance.json").read_text())
COST = json.loads((HERE / "cost.json").read_text())
THR = json.loads((HERE / "cost_threads.json").read_text())
FRESH = json.loads((HERE / "fresh.json").read_text())
MUT = json.loads((HERE / "mutation.json").read_text())
SHIP = json.loads((HERE / "evidence_snapshot" / "result.json").read_text())
tgv = {f"{r['arm']}_n{r['n']}": r["final_error"] for r in SHIP["physics"]["tgv"]}
pois = {r["arm"]: r for r in SHIP["physics"]["poiseuille"]}
md = f"""# 交差核验：m18129 XVAL-LBM-PER-MARGIN-ARRAYS-POINTER-SHIFT-STREAMIN（TensorLBM）

**性质**：任务榜交差核验单（非重做）。对象 m18129（succeeded，2026-11-13T12:14:52Z）；审计单 m18222。
**结论**：判级 **L2**（独立功能实现），`tensor_maturity_verdicts#717`，`verdict_by=board-audit`；
基准裁判 `tensor_benchmark_cases` 对该 anchor **0 行**（无机器裁判）。交付为真代码、可复现、误差远优于
15%/3%，但**无生产调用方**（mission 自决 D3=False 不采纳）⇒ L3/L4/L5 不授。三条 medium 发现需返修式订正
（1 条结论过度泛化、1 条工件陈旧字段、1 条报告自相矛盾）。

## ① 案例介绍（对象与物理背景）

m18129 做的是一次**开源实现交叉对标（XVAL）**：CFD-GO/TCLB 的传输（streaming）写法是「每个 margin 块一份
population 数组 + 跨块时移动基址指针，不做逐格下标运算」。任务把该写法移植成 PyTorch 的
`PointerShiftPlan`（每方向块数 = 2^|{{c_a≠0}}|，D2Q9→25 块、D3Q19→61 块），再与 TensorLBM 生产 gather 传输逐位对比。

| 项 | 值 |
|---|---|
| OSS 锚 | CFD-GO/TCLB @`ded67cd768cf7e727bd078af139e3ec7895076e5`（`/tmp/TCLB`，shallow） |
| 物理锚 | Taylor-Green 2D 精确衰减 e^(−2νk²t)；平面 Poiseuille 抛物线 |
| 关键数值参数 | TGV n=32/64, u_max=0.04, τ=0.8（敏感性 τ=0.6/0.8/1.0）, 300/400 步；Poiseuille ny=34, a=1.6e−5, τ=0.8, 8000 步 |
| 判据 | owner 精度门 15%（>15% 不可接受）、内部目标 3%、成本阀 2× |
| 环境 | loongarch64 CPU, torch 2.7.1, 128 线程, 无 GPU, 无 R/nvcc |

## ② 怎么做的（审计方法）

审计**不在** m18129 工作树就地执行，而是搭 `/tmp/audit_m18222/stage`：本工作树 `src/tensorlbm` + **冻结**的
交付文件（`docs/audit_m18222/evidence_snapshot/`）。七支探针：
A2 provenance（对克隆重算 8 个文件 sha256 + 9 段 excerpt 逐字定位）、A3 pytest、A4 变异探针（3 个破坏版计划
测检出能力）、A5 全量复跑（88.1 s，逐字段 diff）、A6 独立计时（warm-up + 9 reps）+ 线程数扫描
（1/8/32/128）、A7 裁判表核对。工具：`docs/audit_m18222/audit_run.py`。

## ③ 计算结果（数字表，程序化注入）

| 量 | 交付值 | 审计复跑 | 相对偏差 |
|---|---|---|---|
| TGV n=32, 300 步 rel-L2 | {tgv['A_n32']:.6e} | {FRESH['physics_compare']['tgv_A_n32_tau0.8_s300']['auditor_rerun']:.6e} | {FRESH['physics_compare']['tgv_A_n32_tau0.8_s300']['rel_delta']:.1e} |
| TGV n=64, 400 步 rel-L2 | {tgv['A_n64']:.6e} | {FRESH['physics_compare']['tgv_A_n64_tau0.8_s400']['auditor_rerun']:.6e} | {FRESH['physics_compare']['tgv_A_n64_tau0.8_s400']['rel_delta']:.1e} |
| Poiseuille u_max 误差 | {pois['A']['umax_rel_err']:.6e} | {FRESH['physics_compare']['pois_A_umax_err']['auditor_rerun']:.6e} | 0.0 |
| 传输 A vs B（14 形状×dtype） | 逐位相同（max_abs_diff=0） | {FRESH['all_bitwise_equal']} | — |
| 变异探针检出率 | — | {MUT['detection_power']['n_mutants_detected']}/{MUT['detection_power']['n_mutants']} | — |
| 成本（3d/48, 128 线程） | 17.10×（慢） | {COST['rows']['3d/48']['slowdown']:.2f}× | 同量级 |
| 成本（3d/48, **1 线程**） | 未测 | {THR['3d/48/thr1']['slow']:.2f}×（**更快**） | 结论反转 |

## ④ 结果图

左：Taylor-Green 两个传输臂的误差随网格，与 15%/3% 门线；右：成本比随 torch 线程数的变化——交付的
「17× 更慢」只在 128 线程成立，1 线程时移植反而快 1.4–5.5×。

## ⑤ 注意点（踩坑/局限）

* **F1（medium）** 结论「OSS formulation 的优势在移植到 torch 后不存在」**过度泛化**：1 线程下 port 更快
  （3d/48 0.18×、2d/256 0.27×），开销随线程数单调恶化（2d/256：1.12×@8 → 9.13×@128）⇒ 是 per-block
  kernel 启动/线程池税，不是 formulation 经济学。D3「本机不采纳」仍成立。
* **F2（medium）** 交付 `result.json` 的 `short_axis_rejected=False` 与交付代码不符（重跑 True，`n>=3` 守卫在）
  ⇒ 工件早于最后一次代码修改；其余字段全部逐位复现，属重生成遗漏而非造数。
* **F3（medium）** 报告 §Tests 写「2 failed, 15 passed」，实测 **17 passed**（stage 5.38 s；mission 树 6.41 s）
  ⇒ 报告与自身工件自相矛盾。
* **F4（low）** 被「bank」的常数偏移不变量只由 flat 执行器与 `check_plan_invariant` 覆盖，`flat_offset` 被破坏
  对 headline 的 `torch.equal(生产, 切片臂)` 不可见。
* **F5（low）** 5 个交付路径在 m18129 树**未跟踪、零提交** ⇒ 不在任何分支/生态仓，自动推送带不走；审计已
  逐字节冻结进本工作树 `evidence_snapshot/`。
* 局限：TCLB 二进制对照 **WITHHELD**（无 R/nvcc，OSS 侧不可执行，报告自述一致）；GPU 面未测；MPI margin
  exchange 未移植。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 量 | 基准 | 误差 | <15%？ | <3%？ |
|---|---|---|---|---|
| TGV n=32 rel-L2 | 解析衰减 | {100*tgv['A_n32']:.3f}% | 是 | 是 |
| TGV n=64 rel-L2 | 解析衰减 | {100*tgv['A_n64']:.3f}% | 是 | 是 |
| Poiseuille u_max | 解析抛物线 | {100*pois['A']['umax_rel_err']:.3f}% | 是 | 是 |
| 传输 A vs B | 生产算子 | 0（逐位） | 是 | 是 |
| TCLB 二进制 vs TensorLBM | — | **WITHHELD**（OSS 侧不可执行） | n/a | n/a |

**是否 <3%：在解析锚上全部是（0.621%/0.092%/0.058%）；对 OSS 二进制无基准，如实判 WITHHELD，不含糊其辞。**
裁判表该 anchor 0 行 ⇒ 15% 门为自施加、由本审计独立复核（15 个物理标量相对偏差 {FRESH['physics_max_rel_delta']:.1e}）。
"""
png = base64.b64encode((HERE / "fig_m18222_evidence.png").read_bytes()).decode()


def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return r.status, json.loads(r.read() or b"{}")


st, d = api("POST", "/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                   "password": os.environ["TENSOR_AGENT_PASS"]})
tok = d.get("access_token")
payload = {
    "title": "交差核验 m18129 XVAL TCLB per-margin arrays / pointer-shift streaming (TensorLBM) — L2",
    "summary": "board audit: L2 confirmed, bitwise-equivalent port, <3% on analytic anchors, no production caller; "
               "3 medium findings (cost claim overstated at 128 threads, stale result.json field, report self-contradiction)",
    "scenario_key": "xval_lbm_streaming",
    "mission_id": 18222,
    "tags": ["audit", "xval", "lbm", "streaming", "TensorLBM", "TCLB", "m18129", "L2"],
    "markdown": md,
    "figures": [{"name": "fig_m18222_evidence.png",
                 "caption": "left: Taylor-Green error of both transport arms vs 15%/3% gates; right: cost ratio vs torch thread count "
                            "(port is faster at 1 thread, the shipped 17x only appears at 128 threads)",
                 "png_base64": png}],
}
print(api("POST", "/tensor/reports", payload, tok))