"""Publish the m21930 progress report to the TensorPlatform report API."""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")


def post(path: str, body: dict, token: str | None = None) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as fh:
        return json.load(fh)


def login() -> str:
    out = post("/tensor/auth/login", {
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    })
    for key in ("access_token", "token", "accessToken"):
        if key in out:
            return out[key]
    data = out.get("data") or {}
    for key in ("access_token", "token", "accessToken"):
        if key in data:
            return data[key]
    raise SystemExit(f"no token in login response keys: {list(out)}")


MD = """# XVAL-LBM-SINGLE-NODE-MULTI-GPU-HOST-STAGED-HALO — 线性剪切参考的独立审计与修复（m21930）

## ① 案例介绍
锚点做法取自 ProjectPhysX/FluidX3D 的 single-node multi-GPU host-staged halo（每设备一板、halo 经 host
暂存、host 指针交换代替拷贝）。TensorLBM 侧移植（`HostStagedHalo`，R1–R9）、分解不变性与暂存字节账本
已在前序任务落地；本单审的是它的**物理判据参考值**：D3b 腿以「线性化 D3Q19 BGK 步进算子的横波特征值」
作为精确参考门禁剪切波衰减率。

参数：D3Q19 + BGK，fp64，周期边界；剪切波 `u = (0, a sin kx, 0)`，`a = 1e-4`（线性区），`k = 2π/n`，
n ∈ {16,24,32}，ν ∈ {0.02,0.05,0.10}，τ = 3ν + 1/2，测量窗 t = 20 → 220，设备网格 1×1×1 与 2×2×2。

## ② 怎么做的
1. 独立重建参考：`A = S(I − ω(I − P))`，`S = diag(e^{−i c_q·k})`，`P_qq' = W_q(1 + 3 c_q·c_q')`；
   在 `src/tensorlbm/host_staged_halo.py` 新增 `linearised_step_operator` / `transverse_modes` /
   `discrete_transverse_rate` / `shear_khat2` / `linear_mode_decay_rate`。
2. 对抗式复核已交付驱动：`git blame` 定位、失败复现、逐行重放。
3. 修复后重跑 D3b 腿，并把三条判断变成测试（37 passed，新增 12 条）。

## ③ 计算结果（程序化输出）
| grid | ν | γ_LBM | 精确特征值 | 误差 | ν·k̂² 闭式 | 误差 |
|---|---|---|---|---|---|---|
| 16³ | 0.02 | 0.00312409 | 0.00312353 | +0.018 % | 0.00304482 | +2.60 % |
| 16³ | 0.10 | 0.01554956 | 0.01554956 | +0.000 % | 0.01522369 | +2.14 % |
| 24³ | 0.02 | 0.00137842 | 0.00137851 | −0.007 % | 0.00136297 | +1.13 % |
| 24³ | 0.10 | 0.00687907 | 0.00687907 | +0.000 % | 0.00681498 | +0.94 % |
| 32³ | 0.02 | 0.000773377 | 0.000773508 | −0.017 % | 0.000768589 | +0.62 % |

横波扇区谱共 6 条模：慢剪切对（16³, ν=0.02 时 0.00312/步）＋ 4 条鬼模（0.2396/步，快 77×）。

## ④ 结果图
左：横波扇区谱（剪切对、鬼模、闭式 ν·k̂²，双对数；绿虚线为特征值相对闭式偏差）。
右：D3b 实测衰减率 vs 精确特征值 vs 闭式，逐点标注误差。

## ⑤ 注意点（发现的缺陷）
- **F1（阻断）**：已交付驱动的 `discrete_transverse_rates` 用了未导入的 `D3Q19_W`；NameError 发生在
  `main()` 写第一个 CSV 之前 → 该驱动在落地状态下**写不出任何工件**，文档第 ⑦.3 节数值不可由已提交代码复现。
- **F2（阻断）**：`zip(vals, vecs)` 把特征值配到特征向量矩阵的**行**（numpy 返回列）。按列修正后，同一
  过滤条件返回 **0 个候选** → RuntimeError；文档参考值 0.001371（24³, ν=0.02）既不等于错版输出（无输出），
  也不等于正确特征值 0.0013785。
- **F3**：正确谱上有 6 个横波候选，「取中位数」会选到 77× 快的鬼模。
- **F4**：文档写的 D3b 协议（t = 0→200）与代码协议（t = 20→220）不一致。
- 局限：本机无 GPU，PCIe 暂存未实测；参考值只对**线性化**算子精确，O(a²) 项未做幅值扫描。

## ⑥ 误差判定
- 关键量 1：γ_LBM vs 精确离散特征值 —— 最差 **0.018 %**，是 <3 %（且 ≪15 % 业主带）。
- 关键量 2：γ_LBM vs 闭式 ν·k̂² —— 0.62 %–2.60 %，是 <3 %，但闭式系统性偏低（O(k²) 修正），不能当「精确参考」。
- 关键量 3：模态线性预测 vs 实测 —— 相对差 <1e-15，D3b 腿已只受 O(a²) 非线性限制。
- honest 记录：m21387 的 D3b 数值判为**不可复现**（非「正确」）；FAIL 语义未被粉饰；D1/D2/D3 结论不受影响。
"""


def main() -> int:
    png = Path("docs/figures/xval_host_staged_halo_m21930.png").read_bytes()
    token = login()
    body = {
        "title": "XVAL host-staged halo：线性剪切参考的独立审计与修复（D3b 参考 6 候选 / 行列错配 / NameError）",
        "summary": "审计发现已交付 D3b 参考函数两处阻断缺陷（未导入 D3Q19_W → 驱动写不出任何工件；特征值配到特征向量行 → 修正后 0 候选），参考下沉到 src 并加退化/分离断言与 12 条测试；修复后实测剪切衰减率对精确离散特征值误差 ≤0.018%（<3%），对闭式 ν·k̂² 0.62–2.60%。",
        "scenario_key": "xval/lbm/host_staged_halo",
        "mission_id": 21930,
        "tags": ["XVAL", "LBM", "multi-GPU", "host-staged-halo", "audit", "D3Q19", "shear-mode"],
        "markdown": MD,
        "figures": [{
            "name": "xval_host_staged_halo_m21930.png",
            "caption": "左：线性化 D3Q19 BGK 横波扇区（慢剪切对 vs 4 条鬼模 vs 闭式 ν·k̂²，右轴为特征值相对闭式偏差）；右：D3b 实测衰减率 vs 精确特征值 vs 闭式，逐点误差标注。",
            "png_base64": base64.b64encode(png).decode(),
        }],
    }
    out = post("/tensor/reports", body, token)
    print(json.dumps(out)[:500])
    return 0


if __name__ == "__main__":
    sys.exit(main())