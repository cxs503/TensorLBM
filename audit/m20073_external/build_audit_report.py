"""m20105 board audit of m20073 (GAP-DIFFLBM-INVERSE-SHAPE): artifacts, figure, report.

Produces audit/m20073_external/fig_audit_m20073.png from the raw JSON artifacts written by
the two independent re-runs (run5/, run20/) and the identifiability probe (probe/), then
publishes the paper-grade report to the platform.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
if not API.endswith("/api/v1"):
    API = API + "/api/v1"
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

run5 = json.loads((HERE / "run5/results.json").read_text())
run20 = json.loads((HERE / "run20/results.json").read_text())
probe = json.loads((HERE / "probe/identifiability.json").read_text())

TOL = 0.03
cy_true = run20["true_params"]["cy"]
r_true = run20["true_params"]["r"]


def series(run):
    h = run["loss_history"]
    it = np.array([x["iter"] for x in h], float)
    return (it,
            np.array([x["cy"] for x in h], float),
            np.array([x["r"] for x in h], float),
            np.array([x["loss"] for x in h], float))


it20, cy20, r20, loss20 = series(run20)
it5, cy5, r5, loss5 = series(run5)

fig, ax = plt.subplots(2, 2, figsize=(14.2, 9.4))

# ---- (a) parameter trajectories, 20-iteration reproduction
ax[0, 0].axhspan(cy_true * (1 - TOL), cy_true * (1 + TOL), color="#2a9d3f", alpha=0.15,
                 label="±3% acceptance band (cy)")
ax[0, 0].axhspan(r_true * (1 - TOL), r_true * (1 + TOL), color="#1d5f8a", alpha=0.15,
                 label="±3% acceptance band (r)")
ax[0, 0].axhline(cy_true, color="#2a9d3f", lw=1.4, ls="--")
ax[0, 0].axhline(r_true, color="#1d5f8a", lw=1.4, ls="--")
ax[0, 0].plot(it5, cy5, "o-", color="#2a9d3f", ms=4, lw=1.8, label="cy (5-iter run)")
ax[0, 0].plot(it5, r5, "s-", color="#1d5f8a", ms=4, lw=1.8, label="r (5-iter run)")
ax[0, 0].plot(it20, cy20, "o-", color="#2a9d3f", ms=3, lw=1.0, alpha=0.55)
ax[0, 0].plot(it20, r20, "s-", color="#1d5f8a", ms=3, lw=1.0, alpha=0.55)
ax[0, 0].annotate("init cy=9.00", (1, cy5[0]), textcoords="offset points", xytext=(6, 6),
                  fontsize=8, color="#1b5e20")
ax[0, 0].annotate("cy drifts AWAY from truth\n(9.00 → 9.31, 12.5% → 16.4%)",
                  (it20[-1], cy20[-1]), textcoords="offset points", xytext=(-140, 8),
                  fontsize=8, color="#c1121f", arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1))
ax[0, 0].set_xlabel("iteration")
ax[0, 0].set_ylabel("parameter value (lattice units)")
ax[0, 0].set_title("(a) reproduced optimisation trajectories (audit re-run, float64, seed 0)\n"
                   "r moves toward truth; cy moves in the WRONG direction", fontsize=10)
ax[0, 0].legend(fontsize=7, loc="center right")
ax[0, 0].grid(alpha=0.3)

# ---- (b) relative error, log scale, vs 3% criterion
ax[0, 1].axhline(100 * TOL, color="k", ls="--", lw=1.4, label="declared target 3%")
ax[0, 1].plot(it5, 100 * np.abs(cy5 - cy_true) / cy_true, "o-", color="#c1121f", ms=4,
              label="cy rel. error")
ax[0, 1].plot(it20, 100 * np.abs(cy20 - cy_true) / cy_true, "o-", color="#c1121f", ms=3,
              lw=1.0, alpha=0.55)
ax[0, 1].plot(it5, 100 * np.abs(r5 - r_true) / r_true, "s-", color="#1d5f8a", ms=4,
              label="r rel. error")
ax[0, 1].plot(it20, 100 * np.abs(r20 - r_true) / r_true, "s-", color="#1d5f8a", ms=3,
              lw=1.0, alpha=0.55)
ax[0, 1].axhline(15, color="#e08c00", ls=":", lw=1.6, label="owner 15% acceptability band")
ax[0, 1].set_yscale("log")
ax[0, 1].set_xlabel("iteration")
ax[0, 1].set_ylabel("relative parameter error [%]  (log)")
ax[0, 1].set_title("(b) parameter error vs iterations: worst-of-two-monotone for cy\n"
                   "20 iters: cy 16.4% (grew), r 5.1% (still > 3%)", fontsize=10)
ax[0, 1].legend(fontsize=8)
ax[0, 1].grid(alpha=0.3, which="both")

# ---- (c) identifiability: loss landscape along cy at r = truth
grid = probe["grid"]
cy_pts = sorted({g["cy"] for g in grid if abs(g["r"] - r_true) < 1e-12})
loss_pts = [next(g["loss"] for g in grid if abs(g["cy"] - c) < 1e-12 and abs(g["r"] - r_true) < 1e-12)
            for c in cy_pts]
ax[1, 0].semilogy(cy_pts, [max(v, 1e-12) for v in loss_pts], "o-", color="#5a189a", ms=7,
                  label="loss(cy, r=r_true) — forward model only, no optimiser")
ax[1, 0].axvspan(cy_true * (1 - TOL), cy_true * (1 + TOL), color="#2a9d3f", alpha=0.18,
                 label="±3% band around true cy")
ax[1, 0].axvline(cy_true, color="k", ls="--", lw=1.2, label="true cy = 8.0")
ax[1, 0].axvline(run20["recovered_params"]["cy"], color="#c1121f", ls=":", lw=1.6,
                 label=f"optimiser endpoint cy = {run20['recovered_params']['cy']:.3f}")
for c, v in zip(cy_pts, loss_pts):
    ax[1, 0].annotate(f"{v:.1e}", (c, max(v, 1e-12)), textcoords="offset points",
                      xytext=(5, 6), fontsize=7)
ax[1, 0].set_xlabel("trial cy [lattice units]  (r fixed at truth 2.30)")
ax[1, 0].set_ylabel("multi-observable loss (log)")
ax[1, 0].set_title("(c) identifiability probe: loss < 1e-4 across cy = 7.0…9.0\n"
                   "flat valley ⇒ cy non-identifiable at 3% (0.24 lattice units)", fontsize=10)
ax[1, 0].legend(fontsize=7)
ax[1, 0].grid(alpha=0.3, which="both")

# ---- (d) observable sensitivity: what the two observables actually see
sub = [g for g in grid if abs(g["r"] - r_true) < 1e-12]
cys = [g["cy"] for g in sub]
rel_drag = [abs(g["rel_drag"]) for g in sub]
rel_cyw = [abs(g["rel_cy_wake"]) for g in sub]
x = np.arange(len(sub))
ax[1, 1].bar(x - 0.2, [max(v, 1e-18) for v in rel_drag], width=0.4, color="#c1121f",
             label="|drag rel. change| w.r.t. truth")
ax[1, 1].bar(x + 0.2, [max(v, 1e-18) for v in rel_cyw], width=0.4, color="#1d5f8a",
             label="|wake-centre rel. change| w.r.t. truth")
ax[1, 1].set_xticks(x)
ax[1, 1].set_xticklabels([f"{c:.1f}" for c in cys])
ax[1, 1].axhline(1e-2, color="k", ls="--", lw=1.3, label="1% measurement noise (mission scenario)")
ax[1, 1].set_yscale("log")
ax[1, 1].set_xlabel("trial cy (r = truth)")
ax[1, 1].set_ylabel("relative observable change (log)")
ax[1, 1].set_title("(d) why cy cannot be recovered: drag is blind to cy (4e-8),\n"
                   "wake centre moves <1% over Δcy = 1 — below the 1% noise floor", fontsize=10)
ax[1, 1].legend(fontsize=7)
ax[1, 1].grid(alpha=0.3, axis="y", which="both")

fig.suptitle("m20105 cross-check audit of m20073 GAP-DIFFLBM-INVERSE-SHAPE (TensorLBM): reproduced runs "
             "show cy drift + non-identifiable observable pair (verdict L2, L3/L4 withheld)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.955))
png = HERE / "fig_audit_m20073.png"
fig.savefig(png, dpi=140)
print("wrote", png)

summary_numbers = {
    "run5": {"cy_err": run5["relative_errors"]["cy"], "r_err": run5["relative_errors"]["r"],
             "loss0": run5["loss_history"][0]["loss"], "lossN": run5["loss_history"][-1]["loss"]},
    "run20": {"cy_err": run20["relative_errors"]["cy"], "r_err": run20["relative_errors"]["r"],
              "loss0": run20["loss_history"][0]["loss"], "lossN": run20["loss_history"][-1]["loss"]},
    "script_success_flag": run20["success"],
    "identifiability_loss_band": [min(loss_pts), max(loss_pts)],
    "optimizer_endpoint": probe["optimizer_endpoint"],
}
(HERE / "audit_numbers.json").write_text(json.dumps(summary_numbers, indent=2))

MD = """
# 交差审计：m20073 GAP-DIFFLBM-INVERSE-SHAPE（TensorLBM，多参数几何反演）

## 1. 任务与判据（对照 anchors 判据，不采信报告自述）

- 合同（mission m20073, task_key=`gap-difflbm-inverse-shape-r1`）：从尾流场恢复 **≥2 个几何参数**，报告自列验收线 **相对误差 ≤3%**、**1% 噪声下 bootstrap 95% CI 覆盖真值**。
- 平台裁判表 `tensor_benchmark_cases` 对 `anchor_slug='GAP-DIFFLBM-INVERSE-SHAPE'` **0 行**（全表 121 行；`anchor_slug ilike '%INVERSE%'` / `'%SHAPE%'` 均无命中）⇒ 本锚**没有任何机器可复核的基准判据**，唯一可用的判据是任务文本自写的 ≤3%。

## 2. 三件套核验（run_artifacts / ref-vs-got / repro_cmd）

| 项 | 交付声称 | 审计实测 | 判 |
|---|---|---|---|
| 代码入库 | `examples/inverse_shape_multiobs.py` 等 5 文件 | `tensor_dev_missions.id=20073.head_commit = **null**`；`git_status` 5 行全 `??`（未跟踪）；`git log --all` 无任何提交触及这些文件 | ❌ 未入库（交付只在工作树里） |
| run_artifacts | 交付文本无工件指针（`--out` 未用） | 工作树无 `artifact/`、无 results.json、无 log | ❌ 缺失 |
| ref vs got 数值 | 5 iter：loss 0.329→0.171，cy 12.6%，r 19.4% | 审计独立重跑逐位复现：loss 3.290495e-01→1.712428e-01，cy=9.0067（12.584%），r=2.7468（19.424%） | ✅ 5-iter 数字**真实可复现** |
| repro_cmd | 文档写 `--iters 150`（集群 cj2492） | `cluster_jobs.id=2492` 状态 **queued**，从未执行，`rc=NULL`、`finished_at=NULL` ⇒ 150-iter 无任何输出 | ❌ 关键算例从未运行 |
| tests | 无 | 仓库 `tests/` 无任何 shape-inverse 测试（既有 `test_app_inverse.py`/`test_autograd_inverse.py` 覆盖的是单参数 τ/r 反演基础设施） | ❌ 无 |
| 报告六段 | docs 3 篇 | 有方法/参数表/5-iter 收敛，但**无 §④ 对照表（ref vs got 逐项）**、**无 §⑤ 分辨率/参数敏感性**、§⑥ 失败模式有 | ⚠️ 3/6 |

## 3. 审计侧独立复现（不是重做，是取证）

自查重跑（float64, CPU, seed 0，命令见 §7）：

| 运行 | iters | loss 起→终 | cy | r | cy 误差 | r 误差 | 脚本 success |
|---|---|---|---|---|---|---|---|
| run5 | 5 | 3.290495e-01 → 1.712428e-01 | 9.0067 | 2.7468 | **12.58%** | **19.42%** | False |
| run20 | 20 | 3.290495e-01 → 9.677986e-03 | 9.3124 | 2.4171 | **16.41%** | **5.09%** | False |

**决定性事实：cy 单调远离真值。** 初值 9.00（真值 8.00，12.50%）→ iter5 12.58% → iter20 16.41%，同期 loss 掉到 9.7e-3。所谓「需要更多迭代」被证伪：loss 已趋于 0 而参数误差在**扩大**，这是**错误的收敛方向**（wrong attractor），不是预算问题。

## 4. 根因：观测量对 cy 不可辨识（不是调参问题）

不走优化器、直接在真值附近扫 `(cy, r)` 的前向模型（`probe/identifiability.json`，r=2.30 固定）：

| cy | drag 相对变化 | wake-centre 相对变化 | loss |
|---|---|---|---|
| 7.0 | +5.6e-16 | −9.46e-03 | 8.96e-05 |
| 7.5 | +5.15e-03 | −4.72e-03 | 4.88e-05 |
| 8.0 | 0（真值） | 0（真值） | 0 |
| 8.5 | +5.15e-03 | +4.50e-03 | 4.68e-05 |
| 9.0 | **−4.44e-08** | +8.69e-03 | 7.55e-05 |

- drag 对 cy **完全盲**（cy=9.0、r=2.30 时 drag 与真值差 4e-8）——交付自己在 `docs/inverse_shape_multiobs_progress.md` 也写了 ∂drag/∂cy≈7e-7，却仍把它当作 (cy,r) 联合系统的一半信息源。
- wake-centre 对 cy 的灵敏度 ≈0.085 lattice-单位/单位（cy 7→9 只移动 7.488→7.625，1.8%）。要把 cy 收到 3%（±0.24 格）需 wake-centre 分辨率 ~2e-2（0.27% 相对），而合同场景本身要求 **1% 噪声** ⇒ 信息量低于噪声底，bootstrap CI 不可能给出 3% 宽的区间。
- 结果：`cy=7.0`（−12.5%）与 `cy=9.0`（+12.5%）的 loss 分别是 9.0e-5 / 7.5e-5，**同量级**。loss 面在 ±1 格里平坦到 1e-4 ⇒ 参数**结构不可辨识**。

## 5. 发现分级（severity / 证据 / 影响面）

- **F1 阻断级（correctness）**：交付的最终结论「method validated, converging, 需更多迭代」被审计复现证伪。20 iter 后 cy 误差由 12.50% 涨到 16.41%，loss 9.7e-3；且脚本自身 `success=False`。证据：`run20/results.json`、图 (a)(b)。
- **F2 阻断级（identifiability）**：双观测系统对 cy 不可辨识（drag 盲 4e-8；wake-centre 灵敏度低于合同噪声底）。证据：`probe/identifiability.json`、图 (c)(d)。修法必须是换可观测量（压力/涡量/多测面）或改几何尺度（当前 xs=26、球在 x=7、测面 x=20，尾迹在测面已近乎恢复均匀流）。
- **F3 高（process）**：交付物未入库（`head_commit=null`，5 文件 untracked），关键算例 `cj2492` 永远 queued ⇒ 无 run_artifacts、无 repro_cmd 可执行记录；「✓ Bootstrap CI (20 samples, 1% noise)」「✓ Achievement of <3% error target」被写在**标题为 Pending 的块里**却打勾，与同文 §"What's Pending"、`docs/inverse_shape_multiobs_progress.md` §3.2（12.6%/19.4%）直接矛盾——doc-only 情形下这不只是乐观，是**矛盾陈述**。bootstrap 从未运行（脚本 `--bootstrap` 需 `--noise>0`，无任何输出工件）。
- **F4 高（adjudication）**：平台裁判表该 anchor **0 行**，无容差可判；mission 被 `[auto-approve]` 判 succeeded 且 `server_quality_audit.gate_valid=false, key_present=false`——验收 gate 本来就投了反对票，被自动批准覆盖。
- **F5 中（claim 表述）**：「Both parameters move in correct direction」为假（cy 反向）；「r: Should reach <3% in ~50 iterations」无外推依据（20 iter 已到 5.09%，方向对，但合同同时要求 cy，整体不达标）。
- **F6 正面**：失败模式**部分如实**——专利性地记录了 wake-field 直接匹配梯度 ~1e-9 的失败（`docs/inverse_shape_wake_failure_report.md`），未夸大 5-iter 数字（数字可逐位复现），诚实度在「数字」层面成立，在「结论」层面不成立。

## 6. 判级（L0–L5，级别=功能实质）

**L2**。理由：存在真代码的端到端多参数反演管线（自写 `run_forward`(SDF 软几何 + differentiable rollout) → 双观测 loss → Adam 更新 (cy,r) → 输出 results.json），审计独立重跑通过，输入→计算→输出链路真实；但

- **不给 L3**（系统集成）：`examples/inverse_shape_multiobs.py` 无任何仓内调用者，未接入 apps/API/CLI；`src/tensorlbm/inverse/*` 是**既有**单参数基础设施，本单未改。
- **不给 L4**（基准验证）：无可判基准行、无收敛算例（cj2492 从未跑）、无误差分析结论成立（误差 12.6→16.4% 反向发散）、无工件。
- **不给 L5**：无外部锚、无新物理/新算法（组合已有 autograd + 软几何）。

附注：若严格要求「交付代码必须已入库」，本单实质仅剩 3 篇 docs ⇒ **L0/L1**；审计按「功能实质」判 L2 并在 evidence 中显式标注未入库与 0 裁判行，交由 maturity_gate 裁定。

## 7. 复现命令（审计侧）

```bash
# 取证副本（原 worktree /root/tp-agent/work/m20073 的 5 个 untracked 文件）
cp /root/tp-agent/work/m20073/examples/inverse_shape_multiobs.py audit/m20073_external/
PYTHONPATH=src python audit/m20073_external/inverse_shape_multiobs.py --iters 5  --log-every 1 --out audit/m20073_external/run5
PYTHONPATH=src python audit/m20073_external/inverse_shape_multiobs.py --iters 20 --log-every 4 --out audit/m20073_external/run20
PYTHONPATH=src python audit/m20073_external/probe_identifiability.py     # 前向模型扫描，不经优化器
```

## 8. 处置

**request changes / reopen**：交付不得以 succeeded 消账。返修要求（最小集）：①换一个对 cy 有 >1% 灵敏度的可观测量（多测面 wake / 压力 / 涡量）并先做 **identifiability 前向扫描**再谈优化；②把脚本与其算例入库并跑出 `run_artifacts`（results.json + 日志），repro_cmd 可执行；③在 `tensor_benchmark_cases` 注册该 anchor 的 ref/metric/tolerance 行，让 got 可被机器判；④报告六段补 §④对照表、§⑤参数敏感性；⑤删除「✓ <3%」「✓ Bootstrap CI」的矛盾勾选，bootstrap 必须真跑（1% 噪声 × N 样本）。
"""


def post(path, obj, token=None):
    hdr = {"Content-Type": "application/json"}
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(API + path, data=json.dumps(obj).encode(),
                                 headers=hdr, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.status, json.loads(r.read().decode())


def login():
    for p in ("/auth/login", "/tensor/auth/login"):
        for payload in ({"username": USER, "password": PASS}, {"user": USER, "password": PASS},
                        {"agent_user": USER, "agent_pass": PASS}):
            try:
                s, d = post(p, payload)
                tok = d.get("access_token") or d.get("token") or (d.get("data") or {}).get("access_token")
                if tok:
                    print("login ok via", p)
                    return tok
            except Exception as e:  # noqa: BLE001
                print("login fail", p, e)
    raise SystemExit("login failed")


if __name__ == "__main__":
    tok = login()
    body = {
        "title": "交差审计 m20073 GAP-DIFFLBM-INVERSE-SHAPE（TensorLBM）：L2，判据不达标且被证伪——cy 误差 12.5%→16.4% 反向发散、双观测对 cy 不可辨识、交付未入库、裁判表 0 行",
        "summary": ("board-audit 复核 m20073：5-iter 数字逐位可复现（loss 0.3295→0.1712，cy 12.58%，r 19.42%），"
                    "但审计 20-iter 重跑显示 cy 由 12.50% 涨到 16.41%、loss 落至 9.68e-3 且脚本 success=False ⇒ "
                    "「需更多迭代」被证伪；前向扫描证明 drag 对 cy 盲（4e-8）、wake-centre 灵敏度低于合同 1% 噪声底 ⇒ "
                    "参数结构不可辨识。三件套：无 run_artifacts、cj2492 永远 queued、head_commit=null 五文件 untracked；"
                    "tensor_benchmark_cases 该 anchor 0 行。判 L2（真代码端到端管线，审计可复现），L3/L4 扣留，reopened。"),
        "scenario_key": "lbm",
        "mission_id": 20105,
        "tags": ["board-audit", "GAP-DIFFLBM-INVERSE-SHAPE", "TensorLBM", "inverse-problem",
                 "identifiability", "maturity-verdict", "L2", "reopened", "m20073"],
        "markdown": MD,
        "figures": [{
            "name": "fig_audit_m20073.png",
            "caption": ("(a) 审计重跑参数轨迹：r→真值方向，cy 由 9.00 漂到 9.31（远离真值 8.00）；"
                        "(b) 相对误差 vs 迭代（log）：cy 12.50%→16.41% 单调增长，r 19.42%→5.09% 仍 >3%；"
                        "(c) 前向模型扫描（不经优化器）：r=2.30 固定，loss 在 cy∈[7,9] 平坦至 <1e-4，"
                        "3% 带内与带外不可分；(d) 观测量灵敏度：drag 对 cy 变化 4e-8（盲），wake-centre 在 Δcy=1 上仅 0.9%，"
                        "低于合同 1% 噪声底。"),
            "png_base64": base64.b64encode(png.read_bytes()).decode(),
        }],
    }
    s, d = post("/tensor/reports", body, tok)
    print("report post:", s, json.dumps(d)[:500])