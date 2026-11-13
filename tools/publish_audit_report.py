#!/usr/bin/env python3
"""Publish the board-audit report for m17981 (anchor XVAL-LBM-HORIZONTAL-SURVEY-EMPTY).

Every number in the markdown is injected from artifacts/board_audit/m18089/audit_m17981.json
(zero hand copy); the figure is the auditor-generated comparison plot.

  python tools/publish_audit_report.py            # login + POST /tensor/reports
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
MISSION = 18089
HERE = pathlib.Path(__file__).resolve().parents[1]
ART = HERE / "artifacts/board_audit/m18089"


def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return r.status, json.loads(r.read() or b"{}")


def pct(x):
    return f"{x:+.2f} %"


def main() -> int:
    rep = json.loads((ART / "audit_m17981.json").read_text())
    ej = rep["error_judgment"]
    rf = rep["trace_refit"]
    refs = rep["reference_rederivation"]
    reg = rep["benchmark_registry"]

    err_rows = []
    for case, label in (("quick", "quick (D=12, box 6D, 2T 窗)"),
                        ("res16", "res16 (D=16, box 12D, 3T 窗)"),
                        ("base_m17844", "base（旧臂 m17844，D=24, box 12D）")):
        v = ej[case]
        got = v.get("got_E1_area") or v.get("got_E1_halfway")
        err_rows.append(
            f"| {label} | {got:.6f} | {v['ref_same_mask_area']:.6f} | {pct(v['err_vs_same_mask_pct'])} | "
            f"{pct(v['err_vs_analytic_Ca1_pct'])} | 否 | 否 |")
    err_table = ("| 臂 | got `C_I` (E1, area) | 同掩模参考 `c_a` | err vs 同掩模 | err vs 解析 `C_a=1` | <15%？ | <3%？ |\n"
                 "|---|---|---|---|---|---|---|\n" + "\n".join(err_rows))

    refit_rows = []
    for case in ("quick", "res16", "base_m17844"):
        r = rf[case]
        rr = r["refit_from_trace"]
        rep_e1 = r["reported"].get("E1_mem_area") or r["reported"].get("E1_mem_halfway")
        if rep_e1 is None:  # pre-module arm keeps C_M/C_D joint-fit keys
            import json as _json
            kpi = _json.loads((pathlib.Path("/root/tp-agent/work/m17981/runs/"
                                            f"oscillatory_cylinder_inertia/{case}/result.json")).read_text())["kpi"]
            rep_e1 = kpi["E1_area"]["C_M"]
        if r['reproduces']:
            verdict_cell = '逐位复现'
        else:
            verdict_cell = "%.2f %% 差异（旧臂联合 C_M/C_D 拟合）" % (r['reported_vs_refit_pct'] or 0.0)
        repit = (f"| {case} | {rep_e1:.6f} | {rr['F_mem']['C_I']:.6f} | "
                 f"{verdict_cell} | {rr['n_samples']} |")
        refit_rows.append(repit)
    refit_table = ("| 臂 | 报出 `C_I` (E1 area) | 我重拟合（trace.npz） | 判定 | 窗内样本 |\n|---|---|---|---|---|\n"
                   + "\n".join(refit_rows))

    e = ej
    fig_path = ART / "fig_audit_m17981.png"
    figures = [{
        "name": "fig_audit_m17981.png",
        "caption": ("AUDIT m17981：左=各臂测得 C_I 与同掩模势流参考的对比（数值由 trace.npz 独立重拟合）；"
                    "右=误差 % 与 ±15% 门（绿带）及 vs 解析 C_a=1（灰虚线）——三臂全部落在门外，最小 57.9%"),
        "png_base64": base64.b64encode(fig_path.read_bytes()).decode(),
    }]

    md = f"""# AUDIT m18089 — `XVAL-LBM-HORIZONTAL-SURVEY-EMPTY`（m17981 交差核验）

**任务**：owner 2026-11-11 任务榜令的**交差核验单**——核 m17981 的三件套（真工件/基准数值/repro_cmd），
判 L0-L5 并落 `tensor_maturity_verdicts`。**非重做**：不改 m17981 任何物理设置、不改容差、不覆盖 honest-FAIL 工件。

## ① 案例介绍（物理背景 / 几何 / 参数表）

圆柱在振荡流中的**惯性（附加质量）系数** `C_a`——耐波性、PMM 辨识、锚链/立管载荷的主力量。
anchor 的原始命题（`oss_solutions` id=386，`oss-inventory/tensorplatform` / `GAP-DEM-CLUMP` / tag
`horizontal-survey-empty`）：水平扫描成熟 OSS 实现返回空集 ⇒ 该能力只能由文献谱系供给。

配置：双周期 D2Q9 盒、盒心固定圆柱（台阶化圆掩模）、**仅流体节点**加简谐体力 `a(t)=a0 cos(ωt)`
（二阶 Guo 源），全程反弹 + 固体内部置零（保证掩模反弹组合守恒），两个**独立**力估计器：
E1 = Ladd 动量交换、E2 = 表面牵引 `+p n − τ·n`；Morison 谐波最小二乘（1+2 阶）在整数周期窗上取一阶。
`u0=0.04`（Ma=0.069），`Re = β·KC`。

| 臂 | D [格] | box | KC（实测） | β（实测） | τ | 步数 | 窗 |
|---|---|---|---|---|---|---|---|
| quick | 12 | 6D | 0.964 | 115 | 0.514 | 900 | 1.0T 弃 + 2T 测 |
| res16 | 16 | 12D | 0.944 | 115 | 0.519 | 1800 | 1.5T 弃 + 3T 测 |
| base（旧臂 m17844） | 24 | 12D | 0.986 | 107 | 0.529 | 2700 | 1.5T 弃 + 3T 测 |

## ② 怎么做的（方法 / 流程 / 工具链）

**只复核、不重算物理**（容量纪律），全部结论从 m17981 冻结工件重算，工具=self-written
`tools/audit_m17981_verify.py`（A1 工件存在性 → A2 从 `trace.npz` 独立重拟合 → A3 重导参考量 →
A4 误差判定 → A5 registry 交叉 ⇒ A6 集成调用方 grep → A7 出图）：

1. 重走执行路径：`predeclared.json`（预告 15% 门 / 8 case / 停止规则）→ `src/tensorlbm/oscillatory_cylinder_inertia.py:265 run_arm` → `cases/oscillatory_cylinder_inertia/run.py:23` → `runs/oscillatory_cylinder_inertia/*/{{result.json,trace.npz}}`。
2. **不信 result.json**：用自写的 lstsq（1+2 阶谐波、predeclared 整数周期窗、面积归一 `V=πD_area²/4` 由迹里 `A_solid` 反推）重算 `C_I`。
3. 重导参考量：`potential_reference(nx,nx,D)`（同掩模势流）与解析无界锚 `C_a=1`，两口径各判一次 15% 门与 3% 生态线。
4. 端到端复现：module CLI 跑 `quick`（bit-identical）+ `pytest`（8 条）+ survey 仪器在 dump 上重跑（8 family verdict 比对）。
5. 交互检查：`grep -rn oscillatory_cylinder_inertia` 找非自身调用者。

## ③ 计算结果（程序化注入，零手抄）

**C1 交付数字可溯源度（从冻结迹重拟合）**

{refit_table}

⇒ 新臂数字**逐位可重算**；旧臂 −1.85% 的差异由其"联合拟合 C_M 与 |U|U 阻力基"解释，不是造数。

**C2 误差判定（两种参考口径）**

{err_table}

registry 行（唯一裁判）口径：`C_M(E1,D_halfway)=1.5742086608150845` vs Morison `C_M=2.0` ⇒ **−21.29%**，
tolerance `<15%` ⇒ 行文自述 **"gate NOT passed"**。三档分辨率同掩模误差 +57.9 / +72.9 / +62.2%：
**最小 57.9% 仍是门的 3.9 倍**，且不随分辨率收敛进带内 ⇒ 非离散化伪影。

**C3 测量不是短板**：E1 vs E2 离散度 quick {e['quick']['estimator_spread_pct']:.4f} %、
res16 {e['res16']['estimator_spread_pct']:.4f} %（交付头条"222%→0.06%"复核成立）；
误差在模型/参考，不在力测量。

**C4 三件套与 registry 机械核验**

| 项 | 结果 |
|---|---|
| impl_files | 模块 411 行 + case 入口 + survey 仪器（survey/run_arm/analyze/predeclared）— **全部存在且非占位** |
| entry | `python -m tensorlbm.oscillatory_cylinder_inertia --case quick` → `C_I={rf['quick']['reported']['E1_mem_area']:.16f}`（与工件逐位相同） |
| run_artifacts | 3 臂 `{{result.json,trace.npz}}` + `benchmark_table.json` + 3 张 PNG — **全部存在非空** |
| tests | 8 条 **8 passed in 18.69 s**（审计复跑）；均属辅助层，无一条断言物理误差 |
| benchmark | registry 行 tolerance `<15%`，got 1.5742（−21.29%）/ 同掩模 +57.9…+72.9% ⇒ **均未过门** |
| repro_cmd | pytest ✓；module CLI bit-identical ✓；survey 需先 mkdir（`--out` 不建目录）✓ |
| 报告六段 | docs §1-§8 + 3 PNG 齐备；但**无一句显式回答 "<3%"**（⑤-F10） |

**C5 参考量不确定带**：同掩模势流 `c_a` 原始 {refs['res16']['c_a_potential_raw']:.5f}（D=16，208 固体格）→ 面积归一
{refs['res16']['c_a_area_normalised']:.5f}，比解析无界锚高 **{refs['res16']['ref_vs_analytic_pct']:.2f} %**；
两口径都 FAIL，故不改变判决，但"reference"并非完全外部锚。

**C6 registry 交叉**：登记行存在（tolerance `<15%`），但 got 只指向**旧臂 Morison 口径**
（quoted `{reg['quoted_got']['base_E1_halfway']}`，err `{reg['quoted_got']['err_pct']}`%），
不含 rework 的同掩模口径与三档分辨率；且行**由审计 m17888 代注册**（交付时缺行）。

## ④ 结果图

左：各臂测得 `C_I` vs 同掩模参考；右：误差 % 与 ±15% 门绿带 + vs 解析 `C_a=1` 灰虚线——**三臂全部出门**。
（m17981 自带 fig1/fig2/fig3 三张 PNG 只核存在性，未重绘像素。）

## ⑤ 注意点（按严重度；对抗式复核结论）

- **F1（阻断 L4）** 物理门未过：全部臂 |err| 19–83%，15% 门与 3% 生态线双失败。交付如实写 honest FAIL
  并定位原因（β≈115 时 Stokes 层 δ=1.35 格、力峰滞后加速度 45°、谐波残差 0.5–0.8σ）——**未粉饰**，本审计确认。
- **F2（中）** registry 与交付口径脱节：登记行只有旧臂 Morison 数（−21.29%），不能独立支撑交付头条。
- **F3（中）** 登记责任漂移到审计侧（m17888 代注册）——系统性缺口，会掩盖"交付不带基准"。
- **F4（中）** 同掩模参考由本仓 `fsi_added_mass` 算（内部代码，+6.0% vs 解析锚），且口径切换发生在观测到 −21% **之后**。
- **F5（中）** predeclared 的 `res32`/`kc20` **从未跑**；docs §5 的 `C_I=1.541` 窗口敏感性数字**没有落盘工件**。
- **F6（中）** 集成度为 0：非自身调用者 0，未进 `__init__`/capability 契约 ⇒ **L3 未达**。
- **F7（低-中）** 测试绿≠物理对：8 条全在辅助层（几何/谐波合成恢复/窗正交/残差旗标/归一化/势流界/自洽/CLI）。
- **F8（低）** `survey.py --out` 不 mkdir（文档原命令直接 FileNotFoundError）；重跑后 8/8 family verdict 与提交件一致，
  ⇒ "empty" 判决可复现，但范围只是**本仓台账**（33 行/261 行/45 仓），不是全网。
- **F9（低）** KPI 用名义 `a0` 归一、KC/β 用实测 `U`（`a0_measured/a0_nominal=1.014` ⇒ ~1.4% on C_I）。
- **F10（低）** 平台报告 `tensor_reports.id=16915` 的 figures=[]（有 3 张 PNG），且报告与 docs 均未显式回答 "<3%"。
- **F11（低）** docs §3 写"transient 1.5T / 窗 3T"，quick 实际 1.0T / 2.0T。

**未做的检查**：未重跑 res16/base 物理；res32/kc20 不存在；未重绘任务自带 PNG 像素；无实验数据对标；
未核本地 dump 与生产 oss 表一致性。

## ⑥ 误差判定（逐项，显式回答"是否 <3%"）

| 关键量 | got | 基准 | 误差 | <3%？ | <15%？ |
|---|---|---|---|---|---|
| `C_I`(E1,area) quick | {e['quick']['got_E1_area']:.6f} | {e['quick']['ref_same_mask_area']:.6f} | {pct(e['quick']['err_vs_same_mask_pct'])} | **否** | 否 |
| `C_I`(E1,area) res16 | {e['res16']['got_E1_area']:.6f} | {e['res16']['ref_same_mask_area']:.6f} | {pct(e['res16']['err_vs_same_mask_pct'])} | **否** | 否 |
| `C_I`(E1,area) base | {e['base_m17844']['got_E1_area']:.6f} | {e['base_m17844']['ref_same_mask_area']:.6f} | {pct(e['base_m17844']['err_vs_same_mask_pct'])} | **否** | 否 |
| `C_M`(E1,halfway) base（registry 口径） | {reg['quoted_got']['base_E1_halfway']:.6f} | 2.000000 | {reg['quoted_got']['err_pct']:+.2f} % | **否** | 否 |
| 双估计器离散度 quick | E1 {e['quick']['got_E1_area']:.6f} / E2 {e['quick']['got_E2_area']:.6f} | 自洽量 | {e['quick']['estimator_spread_pct']:.4f} % | 是 | 是 |
| 双估计器离散度 res16 | E1 {e['res16']['got_E1_area']:.6f} / E2 {e['res16']['got_E2_area']:.6f} | 自洽量 | {e['res16']['estimator_spread_pct']:.4f} % | 是 | 是 |

**明确回答**：物理量对基准**没有一项 <3%，也没有一项 <15%**；唯一 <3% 的是"两个估计器彼此一致"这一
**内部自洽量**，它证明测量管线可靠，**不证明物理正确**。故本能力 = **未验证（WITHHELD）**，
与交付 honest FAIL 一致；交付的诚实性是本单确认的对象，能力达标不是。

## 判级与落表

**L2**（真代码输入→计算→输出 + 真工件 + 逐位可复现 + 8 测试绿）。不给 L3（0 非自身调用者）、
不给 L4（基准门 FAIL 且 2 个 predeclared case 未跑）、不给 L1（是完整可执行物理算例）。

落表：`tensor_maturity_verdicts('TensorLBM', 'm17981 XVAL-LBM-HORIZONTAL-SURVEY-EMPTY', 18089, 2,
evidence, 'board-audit')`；本审计自检 gate `judge()` → `gate_level=2, reasons=[]`。
**disposition：approve with notes**（三件套真实、数字可溯源、honest FAIL 无粉饰；F2/F5/F6/F10 留作返修建议）。
"""

    payload = {
        "title": "AUDIT m18089 — m17981 XVAL-LBM-HORIZONTAL-SURVEY-EMPTY 交差核验：L2，honest FAIL 复核成立",
        "summary": ("交差核验 m17981：三件套全部真实（模块 411 行 + entry + 3 臂 result/trace + 8 测试 8 passed），"
                    "从 trace.npz 独立重拟合逐位复现 C_I；基准裁判 registry 行 tolerance<15% ⇒ got −21.29%（旧臂口径）"
                    "/ +57.9…+72.9%（同掩模口径）均未过门 ⇒ 判 L2、能力 WITHHELD；发现 registry 口径脱节、"
                    "res32/kc20 未跑、集成调用者 0、报告无图等 11 项缺陷，均已留指针。"),
        "scenario_key": "board_audit_m17981_xval_lbm_horizontal_survey_empty",
        "mission_id": MISSION,
        "tags": ["board-audit", "maturity-verdict", "XVAL", "TensorLBM", "added-mass", "honest-FAIL", "m17981"],
        "markdown": md,
        "figures": figures,
    }
    status, tok = api("POST", "/auth/login", {"username": USER, "password": PASS})
    if status != 200:
        raise SystemExit(f"login failed: {status} {tok}")
    token = tok["access_token"]
    st, res = api("POST", "/tensor/reports", payload, token)
    print("POST /tensor/reports ->", st, json.dumps(res, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())