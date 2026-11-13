#!/usr/bin/env python3
"""Publish the m18155 audit report to the tensor platform (six-section format)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request

API = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PWD = os.environ["TENSOR_AGENT_PASS"]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
XV = os.path.join(ROOT, "xval", "XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED", "figs")


def post(path: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def figure(name: str, caption: str) -> dict:
    with open(os.path.join(XV, name), "rb") as fh:
        return {"name": name, "caption": caption, "png_base64": base64.b64encode(fh.read()).decode()}


MARKDOWN = """## ① 案例介绍

交差核验单（非重做）：对象 = **m17995 XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED**（TensorLBM）。该单把
`lssfau/lbmpy` 2.0 + `lbmpy_walberla` 的 **PackInfo 方向选择性 ghost 交换**（MPI ghost 层只打包真正跨该面的
分布函数；D3Q19 x 面 = 5/19）移植到 TensorLBM 的 x-slab D3Q19 传输上，用 3D Taylor–Green 涡做分解保真度探针。

| 参数 | 值 |
|---|---|
| 网格 / 分解 | 64³ = 262144 cells，x 向切 2 slab（另有 32³、4-rank 预声明未交付） |
| U0 / Ma / τ / ν | 0.04 / 0.0693 / 0.65 / 0.05 |
| 步数 / 精度 | 400 / float32 |
| 传输 | torch.distributed gloo，isend/irecv，1 线程/rank |
| 臂 | serial、full_alloc（shipped 复刻）、full_persistent、selective_alloc、selective_persistent |

## ② 怎么做的（审计方法）

1. 取证：读 `/root/tp-agent/work/m17995` 交付树 + mission 17995 result（`head_commit=null`，全部路径 `??`
   未提交）+ 报告 16918；把交付文件**原样**移植进审计分支 `agent/m18155`（排除 176 MB 的 .pt），使证据可从
   落地 commit 解析。
2. 基准裁判：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug=...`
   → **0 行**（全表 91 行，2026-11-13 核对）。无注册容差，改以该单 `predeclared.json` 的三条预声明判据裁决。
3. 复跑测试：`PYTHONPATH=src pytest tests/test_xval_mpi_ghost_fused.py -q` → **10 passed, 0 skipped**（7.0 s），
   其中 `importorskip("lbmpy...")` 实际执行，逐方向比对 `lbmpy.advanced_streaming.communication._extend_dir`
   （lbmpy 2.0，communication.py:233）。
4. 集成核验：`grep -rn xval_mpi_ghost_fused src/ app/ examples/ benchmarks/` → **0 hit**；
   `src/tensorlbm/multi_gpu.py:928 D3Q19GlooTransport` 未被修改。
5. 复现核验：对交付 `results.json` 重跑图阶段 → **KeyError 'n64_r4'**（交付只含 1/4 配置）。
6. 解析基准复核：新写 `audits/m18155/verify_tgv_reference.py`，monolithic 重算 n=32/64，同一数值场分别对
   `exp(-2νk²t)`（交付式）与 `exp(-3νk²t)` 打分，并从 RMS 序列独立拟合 ν。

## ③ 计算结果（数字全部来自工件，零手抄）

**基准裁决（predeclared 判据 vs got，n64_r2）**

| 指标 | 判据 | got | 结论 |
|---|---|---|---|
| tgv_rel_l2 | <15%（owner） | **24.887%**（5 臂同值） | **FAIL** |
| identity_max_abs vs serial | 精确 0 | **0.0**（4 臂，float32） | PASS |
| byte_ratio | 精确 5/19 = 0.263157894737 | **0.263157894737**（1245184→327680 B/step） | PASS |
| exchange_ms | 非门 | 17.229 → 12.351 ms/step（−30.4%） | 仅供参考 |

矩阵完整性：预声明 4 配置，`results.json`/`table.csv` 仅 **1**（n64_r2），图注却写「全 4 配置」。

**解析基准缺陷复核（本审计新增）**

| 算例 | rel-L2 vs 2νk²（交付式） | rel-L2 vs 3νk²（修正式） | ν_eff@3k² | 对 ν=0.05 |
|---|---|---|---|---|
| n=32, t=200 | 34.706% | **8.713%** | 0.051326 | +2.65% |
| n=64, t=400 | 24.887% | **17.632%** | 0.051697 | +3.39% |

交付初始场 `ux=U0 sin(kx)cos(ky)cos(kz), uy=−U0 cos(kx)sin(ky)cos(kz), uz=0` 每分量波矢 |k|²=3k²，
线性 NS 衰减率应为 3νk²；交付的 `case.py::TGV.decay`（及 `predeclared.json::baseline_control`，跑前写死）
用了 **2νk²（2D TGV 的率）**。同一数值场换正确指数后误差从 24.887% 降到 17.632%——报告 24.887% 里有
**7.25 个百分点是解析式缺陷**。

## ④ 结果图

图 1 = 各臂误差 vs 15%/5%/3% 判据线；图 2 = 每步交换字节与交换耗时（5/19 臂 1245184→327680 B）；
图 3 = u_x 云图与 |分解−单体| 面板（实测 0）。原报告 ④ 承诺的「图1 TGV 衰减曲线」在交付工件中不存在
（`fig1_tgv_decay.png` 缺失，且 `serial.series={}`、`serial.step_ms=null`，图 1 数据源为空）——本报告不补画，
如实记为交付缺陷。

## ⑤ 注意点

1. **基准表是唯一裁判**：本 anchor 0 行，故「基准验证」维度只能判 WITHHELD；报告与工件再漂亮也换不来 L4。
2. **预声明也会写错**：`2νk²` 在跑之前就写进 `predeclared.json`，照样把结论带偏 7.25pp——预声明防的是
   p-hacking，不防公式错误，故仍需独立复核解析式。
3. **FAIL 结论仍成立**：17.632% > 15%，因为 U0·k·t = 1.57 使谐波确实 O(1)；只是归因「解析基准只含线性
   衰减模态」点错根因（主因是 2/3 指数）。
4. 本机无 mpi4py/MPI 实现，OSS 侧是语义移植 + 源码级机器比对，未跑 OSS 二进制（原单自述如实）。
5. 交付未提交（`head_commit=null`），176 MB 场张量未入库；本审计移植了代码/测试/工件以让证据可解析。

## ⑥ 误差判定

| 关键量 | 基准 | 误差 | 判据 | 结论 |
|---|---|---|---|---|
| TGV rel-L2 (n=64, t=400) | 解析 ABC 模态 3νk² | **17.632%**（修正基准）/ 24.887%（交付基准） | <15% | **FAIL（诚实）** |
| 分解引入的场差 | monolithic | **0.0**（逐比特） | =0 | PASS |
| 交换字节比 | shipped 19/19 | 5/19 = 26.32% | 精确 | PASS |
| ν 拟合 | ν=(τ−½)/3 = 0.05 | +3.39% (n=64) / +2.65% (n=32) | — | 印证 3k² 指数 |

**是否 <3%？** 否。分解误差本身是 0（逐比特），但 TGV 相对 L2 在**正确的**解析基准下仍是 17.632%（n=64），
不满足 3%，也不满足 15%。原报告称「<3% 不满足是 LBM 离散误差」不完全准确：其 24.887% 中 7.25pp 来自
解析式指数写错（2νk² vs 3νk²），余下 17.632% 才是有限幅值/离散误差的真实水平。

**判级：L2**（独立功能实现：真代码 1139 LOC + 10/10 测试 + 真实 2-rank gloo 工件 + OSS `_extend_dir`
机器比对）。**L3 不予**（shipped `multi_gpu.py` 未改、0 集成调用，3.8× 字节收益对库用户不可用）；
**L4 不予**（基准表 0 行、物理判据 FAIL、4 配置仅交付 1、图阶段不可复现）；**L5 不适用**。
"""


def main() -> None:
    tok = post("/auth/login", {"username": USER, "password": PWD})
    token = tok.get("access_token") or tok.get("token")
    payload = {
        "title": "交差核验 m17995：lbmpy 选择性 ghost 交换移植判级 L2（并证解析基准 2νk²→3νk² 缺陷，误差 24.887%→17.632%）",
        "summary": "审计 m17995（TensorLBM XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED）：交付为真代码但未接入 shipped transport，基准表 0 行且物理判据 FAIL，判 L2；新发现 TGV 解析式把 3k² 场错用 2νk²，24.887% 中 7.25pp 为参考式缺陷，修正后仍 17.632%>15%（FAIL 结论成立但归因需改）。",
        "scenario_key": "xval-audit",
        "mission_id": 18155,
        "tags": ["audit", "TensorLBM", "ghost-exchange", "lbmpy", "XVAL", "benchmark-adjudication"],
        "markdown": MARKDOWN,
        "figures": [
            figure("fig2_error_vs_analytic.png", "各臂 TGV 误差 vs 15%(owner)/5%/3% 判据线（交付工件，标注为全 4 配置但 results.json 仅 1 配置）"),
            figure("fig3_transport_cost.png", "每步交换字节与交换墙钟：选择性 5/19 臂 1245184→327680 B/step，交换 17.229→12.351 ms"),
            figure("fig4_field_identity.png", "u_x 云图（monolithic）与 |分解−单体| 面板（实测 0）"),
        ],
    }
    res = post("/tensor/reports", payload, token)
    print(json.dumps({k: res[k] for k in res if k in ("id", "status", "title", "qa_status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()