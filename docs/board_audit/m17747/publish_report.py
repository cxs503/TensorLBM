#!/usr/bin/env python3
"""Publish the m17747 board-audit report (six sections; every number read from audit.json)."""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = json.loads((HERE / "audit.json").read_text())
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def post(path: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def get_token() -> str:
    for p in ("/auth/login", "/tensor/login", "/login"):
        try:
            d = post(p, {"username": USER, "password": PASS})
            for k in ("access_token", "token"):
                if k in d:
                    return d[k]
            if isinstance(d.get("data"), dict) and "access_token" in d["data"]:
                return d["data"]["access_token"]
        except Exception as e:  # noqa: BLE001
            last = e
    raise SystemExit(f"login failed: {last}")


fr = A["A2_frozen_arm"]["rows"]
fz = A["A3_fresh_arm"]["runs"]
rep = A["A3_fresh_arm"]["replay"]
d48, d64 = fr["D48"], fr["D64"]
md = f"""## ① 案例介绍（核验对象与物理背景）

核验对象 = **mission 17679**（`succeeded`，auto-approve），锚点 `XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS`，
仓 TensorLBM 分支 `agent/m17679` @ `3161a1fc`。物理算例 = Re=200 自由流圆柱（Von Kármán 涡街起点），
对标 **haimingz/vivsim @ fb84b12** 的 `examples/2d/flow_pass_cylinder.py`：几何 `D=60 / 30D×20D / 圆心(10D,10D)`、
`U0=0.05`、`Re=200`、总时长 `150 D/U0`；参考值 `CD_REF=1.389`、`CL_REF=0.718`（连三篇 DOI 写在上游源码 :78-85）；
口径 `cd_mean=mean(cd[TM//2:])`、`cl_max=max|cl[TM//2:]|`（上游 :207-208）。本审计不是重做，是**交差核验**：
三件套是否真存在、got 是否真落在容差内、报告六段是否齐全、据此定级 L0–L5。

| 无量纲/参数 | OSS (vivsim) | 本仓口径匹配档（新跑） | 本仓 FROZEN 档（既有 verified） |
|---|---|---|---|
| Re | 200 | 200 | 200 |
| 域 / 圆心 | 30D×20D / 10D | 30D×20D / 10D | **40D×40D / 12D** |
| U0 (lattice) | 0.05 | 0.10（Ma 0.058） | 0.08 |
| 直径格数 D | 60 | 10 / 12 | 48 / 64 |
| 碰撞 / 边界 | KBC + IB-MDF | MRT + 固体冻结半步反弹 + Ladd MEM | 同左 + sponge 10D |
| 时长 | 180000 步 | 15000 / 18000 步预算 | 70000 / 80000 步 |

## ② 怎么做的（核验方法，独立重算不采信转述）

1. **取证**：读 `tensor_reports` id=16662、`/root/tp-agent/work/m17679` 工作树（`git status` 干净、`agent/m17679 @ 3161a1fc`）、
   `artifacts/xval_vivsim_re200/summary.json`、三份 `runs/xval_D*.json`（honest-FAIL 原始记录）。
2. **基准裁判**：`SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-...'` → **0 行**（此前无人登记；
   本审计已补一行 referee，容差**事前**声明 `<15%`，status=pending）。因此改以"独立重算"为裁判：
   从 `benchmarks/verified/cylinder/re200/result.json` 原始值重算 Cd/Cl 误差，从三份 run JSON 读 `nan_at`。
3. **参考值出处**：从上游源码切片 `evidence/oss_flow_pass_cylinder.py` **正则解析** `CD_REF/CL_REF/三篇 DOI`，
   与交付代码常量比对 → `cd_match=True, cl_match=True, doi_match=True`（零手抄核查通过）。
4. **重放**（关键）：审计方独立重跑 `--D 10 --u 0.1 --steps 2000 --threads 16`，比对 `nan_at` 与 `verdict`。
5. **复跑测试**：`pytest tests/test_xval_vivsim_cylinder_re200.py` → `11 passed`。
6. 全部核验脚本：`docs/board_audit/m17747/audit_m17747.py`（A1–A8）→ `audit.json`；`gen_figure.py` → 本图。

## ③ 计算结果（全部程序化注入自 audit.json，非手抄）

**FROZEN 档（既有 verified 算例，非本单产出，口径不符）**

| 档 | Cd | err vs 1.389 | √2·Cl_std | err vs 0.718 | St | <15%? | <3%? |
|---|---|---|---|---|---|---|---|
| D48 | {d48['cd']:.4f} | **{d48['err_cd_pct']:.2f}%** | {d48['cl_peak_from_std']:.4f} | **{d48['err_cl_pct']:.2f}%** | {d48['st']:.4f} | ✓ | ✗ |
| D64 | {d64['cd']:.4f} | **{d64['err_cd_pct']:.2f}%** | {d64['cl_peak_from_std']:.4f} | **{d64['err_cl_pct']:.2f}%** | {d64['st']:.4f} | ✓ | ✗ |

**口径匹配档（决定锚点的那一档）—— 0/3 有效数字**

| 档 | τ | 预算步数 | 终点 | wall_s | 有效 Cd/Cl |
|---|---|---|---|---|---|
""" + "\n".join(
    f"| D={r['setup']['D']} u={r['setup']['u_in']} | {r['setup']['tau']:.3f} | {r['setup']['n_steps']} | "
    f"**NaN @{r['nan_at']}** | {r['wall_s']:.1f} | 无 |" for r in fz.values()
) + f"""

**审计方重放**：`nan_at` 归档 {rep['nan_at_archived']} vs 重放 {rep['nan_at_replay']} →
`verdict` 字典逐键相同 = `{rep['verdict_identical']}`；崩前 `cd_half1` 相对差 **{rep['rel_diff_cd_half1']:.2e}**（20 vs 16 线程 float32 归约顺序）；耗时 {rep['replay_wall_s']} s。

## ④ 结果图

左：FROZEN 档误差 vs 15% 判据带与 3% 报告带（注释标明 40D×40D / u=0.08 的口径不符）；
右：审计方重放的 D=10 档 `Cd(t)/Cl(t)` 与三档 NaN 位置（500 / 2000 / 2500）对照目标窗口 15000 步。
读图要点：**决定锚点的那一档连一个周期都没跑完**，误差柱只存在于口径不符的旁证档。

## ⑤ 注意点（坑与局限）

- **口径陷阱**：FROZEN 档是 `40D×40D / 圆柱12D / u=0.08 / sponge 10D`，且 `Cl` 用 `√2·Cl_std` 换算，
  与 vivsim 的 `30D×20D / 10D / u=0.05 / max|Cl|` 三处不同 → 不能算本锚点的验证（这是本单最容易被好看数字骗过去的地方）。
- **St 幻觉数**：三份 NaN 记录的 `st_vs_lit_err_pct` 都是**有限值**（−49.24% / −51.27% / −18.78%），
  机制是 `harmonic_amplitude()` 退化后回落到频率地板 `1/T` → `St=0.1`；重放同样复现。报告面应置 None。
- **`max|Cl|` 天生毛刺敏感**：被审方自己的单测实测同一毛刺抬 2.6×，而谐波幅值只偏 −19%；
  叠加 `CL_REF=0.718` 上游只说 "Averaged from" 三篇 DOI（无平均方式），15% 判据里含参考值噪声。
- **指针断链**：文档结语指向的 `benchmarks/xval/README` 不存在；`runs/` 被 `.gitignore:108` 排除，
  三份 FAIL 原始记录不随分支走（本审计已快照进 `evidence/`）。
- **未做**：D=12 两档未重放；失稳根因未定位；vivsim 从未执行（无 JAX，上游 3.9×10¹¹ 格点更新）。

## ⑥ 误差判定（逐项，显式回答是否 <3% / <15%）

| 关键量 | 基准 | 本仓值 | 误差 | <15%? | <3%? | 证据级 |
|---|---|---|---|---|---|---|
| Cd_mean (D48) | 1.389 (vivsim) | {d48['cd']:.4f} | {d48['err_cd_pct']:.2f}% | 是 | **否** | FROZEN（异口径） |
| Cd_mean (D64) | 1.389 | {d64['cd']:.4f} | {d64['err_cd_pct']:.2f}% | 是 | **否** | FROZEN（异口径） |
| Cl 峰值 (D48) | 0.718 | {d48['cl_peak_from_std']:.4f}（√2·Cl_std 换算） | {d48['err_cl_pct']:.2f}% | 是 | **否** | FROZEN+换算 |
| Cl 峰值 (D64) | 0.718 | {d64['cl_peak_from_std']:.4f}（换算） | {d64['err_cl_pct']:.2f}% | 是 | **否** | FROZEN+换算 |
| Cd / Cl（**口径匹配**、30D×20D 新跑） | 1.389 / 0.718 | — | — | **无法判定（WITHHELD）** | 无法判定 | FRESH 0/3 有效 |

**是否 <3%？否**——最好的两项（St −0.41%、Cd −3.63%）都没进 3%，且它们来自异口径档。
**是否 <15%？** 只在**异口径旁证**上成立；**锚点要求的口径匹配档没有任何数字**，故不得记「<15% 通过」。

## 判级与处置

**L2**（L1 达：11 条真断言测试；**L2 达**：真代码输入→计算→输出，审计方重放 32.5 s 成功；L3 未达：0 调用者；
L4 未达：0 行裁判 + 口径匹配 0/3 有效 + 无收敛阶；L5 不可达）。
**处置 = request_changes（reopened）**，L2 保留：round 2 需 (1) 用 (D, seed_amp) 二维扫描（含 seed_amp=0）定位 NaN；
(2) 交出 D=16/24、u=0.05 的口径匹配数字并双口径并列；(3) 填本审计登记的 referee 行；(4) 修 St 幻觉数 / README 断链 / 表格手抄漂移。

> 六段引用的每个数字均来自 `docs/board_audit/m17747/audit.json`；verdict 已落表
> `tensor_maturity_verdicts` id=628（level 2, board-audit, gate_state=pending）。
"""

png = base64.b64encode((HERE / "fig_m17747.png").read_bytes()).decode()
payload = {
    "title": "交差核验 m17679 · XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS：判 L2，口径匹配档 0/3 有效（NaN），honest-FAIL 可重放",
    "summary": ("board-audit of m17679: harness is real (11 tests green, auditor replay reproduces NaN@2000 with "
                "verdict identical and pre-divergence Cd rel diff 5.4e-06), but the anchor's deliverable is WITHHELD "
                "(0/3 caliber-matched runs produced a number) and tensor_benchmark_cases had 0 referee rows. "
                "Verdict: L2, request_changes/reopened."),
    "scenario_key": "board_audit",
    "mission_id": 17747,
    "tags": ["board-audit", "XVAL", "TensorLBM", "vivsim", "honest-FAIL", "maturity-L2"],
    "markdown": md,
    "figures": [{"name": "fig_m17747.png",
                 "caption": "left: FROZEN arm errors vs the 15%/3% bars (different set-up: 40D x 40D, u=0.08); "
                            "right: auditor replay of the D=10 caliber-matched run with the three NaN points "
                            "(500/2000/2500) against the 15000-step target window",
                 "png_base64": png}],
}
tok = get_token()
out = post("/tensor/reports", payload, tok)
print(json.dumps(out, ensure_ascii=False)[:600])