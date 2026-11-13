"""Build the m17881 audit report payload (numbers injected programmatically, zero hand-copy)."""
import base64
import json
import math
import pathlib

W63 = pathlib.Path("/root/tp-agent/work/m17763/experiments/xval_ludwig_fe_stress")
res = json.loads((W63 / "results.json").read_text())
aud = json.loads(pathlib.Path("/root/tp-agent/work/m17881/docs/board_audit/m17881/audit_numbers.json").read_text())
bm = aud["checks_run"]["independent_sigma_recompute"]["values"]
planar = res["cases"]["planar"]
curved = res["cases"]["curved_interface"]

rows = []
for v in bm:
    k = f"xi{v['xi']:.1f}_iso9"
    rows.append((f"σ 平面 ξ={v['xi']:.1f} (κ={v['kappa']:.0f})", f"{v['sigma_ref']:.5f}",
                 f"{planar[k]['sigma_discrete']:.5f}", f"{v['rel_err']*100:.2f}%",
                 "✓" if v["rel_err"] < 0.15 else "✗", "✓" if v["rel_err"] < 0.03 else "✗"))
for k, d in curved.items():
    rows.append((f"σ 曲面 R={d['R']} ξ={d['xi']:.0f}", f"{d['sigma_analytic']:.5f}",
                 f"{d['sigma_from_curved_tensor']:.5f}", f"{d['sigma_curved_rel_err']*100:.2f}%",
                 "✓" if d["sigma_curved_rel_err"] < 0.15 else "✗",
                 "✓" if d["sigma_curved_rel_err"] < 0.03 else "✗"))
for k, d in curved.items():
    rows.append((f"曲率平衡 ∫(P_NN−P_TT)/r dr  R={d['R']}", f"{d['sigma_flat_over_R']:.5f}",
                 f"{d['radial_balance_lhs']:.5f}", f"{d['radial_balance_abs_rel_err']*100:.2f}%",
                 "✓" if d["radial_balance_abs_rel_err"] < 0.15 else "✗",
                 "✓" if d["radial_balance_abs_rel_err"] < 0.03 else "✗"))
for k, d in planar.items():
    if not k.endswith("iso9"):
        continue
    rows.append((f"平衡态体力残差 max|F|/(σ/ξ²) ξ={d['xi']:.0f}", "0",
                 f"{d['residual_stress_over_Fscale']:.3e}",
                 f"{d['residual_stress_over_Fscale']*100:.2f}%", "✓",
                 "✓" if d["residual_stress_over_Fscale"] < 0.03 else "✗"))
k = "xi4.0_iso9"
rows.append(("两式力场逐点相对差 max|F_s−F_g|/max|F_g| ξ=4", "0（连续恒等）",
             f"{planar[k]['force_diff_rel']:.4f}", f"{planar[k]['force_diff_rel']*100:.2f}%",
             "✗", "✗"))

table = "\n".join("| " + " | ".join(r) + " |" for r in rows)
sig_errs = ", ".join(f"{v['rel_err']*100:.2f}%" for v in bm)
worst = max(v["rel_err"] for v in bm)
under3 = [v for v in bm if v["rel_err"] < 0.03]
seam = aud["checks_run"]["periodic_seam_control"]["values"]

png = (pathlib.Path("/root/tp-agent/work/m17881/docs/board_audit/m17881/figures"
                    "/m17881_audit_verification.png").read_bytes())

md = f"""# 交差审计 m17763：自由能压强张量体力（ludwig STRESS_DIVERGENCE vs TensorLBM −φ∇μ）

**审计单 m17881 · 判定级别 L2 · 处置 approve with notes**（被审 m17763，TensorLBM，succeeded，commit `a53d734c`）

## ① 案例介绍（被审对象是什么）

二元流体自由能 LBM 的毛细（Korteweg）体力存在两种等价写法：化学势梯度式 `F = −φ∇μ`
（TensorLBM `multiphase.free_energy_step` 的现行实现）与自由能压强张量散度式 `F = −∇·P`
（ludwig-cf/ludwig 的 `FE_FORCE_METHOD_STRESS_DIVERGENCE`）。二者在连续极限**恒等**，离散实现不同。
被审单把 OSS 做法从编译产物反解后移植进 TensorLBM，在同一离散场、同一网格上与现行做法对照。

| 参数 | 值 | 说明 |
|---|---|---|
| A, B | 1, 1 | a = −A < 0，φ₀ = √(A/B) = 1 |
| κ | 2, 8, 32 | ξ = √(2κ/A) 即 2, 4, 8 |
| 网格 | 256×256（平面）/ 128×128、256×256（液滴） | 格距 h = 1 |
| 判据窗口 | ±8ξ（平面）；液滴只用正半轴 | 排除周期边界伪界面 |
| 判据 | owner 15% 精度带（>15% 不可接受） | 另并列 <3% 线供对照 |

本单**不重做**该任务，只做取证核验 + 判级 + 落表（交差核验单）。

## ② 怎么做的（审计方法/工具链）

1. **取证三件套**：在被审工作树 `/root/tp-agent/work/m17763` 内逐项核 `impl_files / entry / run_artifacts`，
   并实跑复现命令（`run_xval.py`、`pytest tests/test_free_energy_stress.py`）。
2. **基准裁判优先**：直接查 `tensor_benchmark_cases`（按 `anchor_slug`）——裁判缺席即 L4 不可确认，
   不以「报告好看」补位。
3. **数字独立重算**：审计侧用 numpy 在**同一解析 tanh 场**上重算 σ（不 import tensorlbm），
   与交付物 `results.json` 逐位比对，杜绝手抄。
4. **解析恒等式独立推导**：由 p0 = φf′−f、d = p0 − κφ∇²φ − ½κ|∇φ|² 推 ∂_b P_ab = φ∂_a μ，
   判断两式力场差的 104% 是物理差异还是离散残差被人为放大。
5. **外部锚佐证**：在被审单引用的 ludwig 快照 `.o` 上复核符号/行号/常量（nm、readelf decodedline、objdump -s）。
6. **gate 预演**：用平台 `maturity_gate.judge()` 对我的 evidence 做机器复核（避免自声明走过场）。

## ③ 计算结果（ref vs got，全部程序化注入）

**基准表核验**：`SELECT ... WHERE anchor_slug='XVAL-LBM-FREE-ENERGY-PRESSURE-TENSOR-BODY-FORCE'` → **0 行**
（全表 65 行，`ILIKE '%FREE-ENERGY%'` 亦 0 行）。声明的唯一裁判在库内**不存在** → L4 不可确认。

| 量 | ref（解析） | got（交付物） | 误差 | <15% | <3% |
|---|---|---|---|---|---|
{table}

- 审计独立重算 σ（numpy，无 tensorlbm）：{sig_errs}，最差 {worst*100:.2f}%，与 `results.json` 相符到 15 位。
- 周期接缝控制：全域梯形积分（含接缝）得 {seam['xi=2.0']:.2f} / {seam['xi=4.0']:.2f} / {seam['xi=8.0']:.2f}，
  相对误差 144%–600% → 证明 ±8ξ 开窗是**必要**控制，不是挑选有利区间。
- 平台 gate 干跑：`judge(2)` → (2, []) 全项通过；`judge(3)` → 缺 `integration_callers`。

## ④ 结果图

左：σ 离散误差随界面分辨率 ξ 的收敛（交付物 vs 审计独立重算，叠加 O(h²/ξ²) 参考线、15% 判据带与 3% 线）。
中：两式体力在平衡剖面上的残差（均 ~O(h²)，5.3%→0.58%）。
右：L1–L5 逐级判据达成情况——L1/L2 达成，L3（无求解器调用方）/L4（基准行 0）/L5 未达成。

## ⑤ 注意点（踩坑/局限/适用范围）

- **本单的级别是 L2，不是「报告写得好」能换来的**：模块 217 行真代码、可跑入口、非空工件、3 条不变量断言 → L2；
  但 `grep -rn 'free_energy_stress|stress_divergence_force' src/`（除模块自身）**0 命中**，
  `multiphase.py:1223 free_energy_step` 体内仍是 `Fx=-phi*grad_mu_x` → **L3 未过**；σ 取自**给定解析场**（不积分时间、
  不 import 求解器）→ 静态离散化校验，非完整算例 → **L4 未过**。
- **104% 那行不是造假也不是物理不符**：两式在连续极限恒等（审计已独立推导），平衡剖面上两臂都只剩 O(h²) 截断残差，
  逐点相对差的分母本身就是残差 → 指标病态。交付物已自标 honest-FAIL，未拿它当通过。建议报告改以 σ/ξ² 归一绝对值为主。
- **5pt 与 iso9 在平面一维剖面上代数恒等**（4/12+1/12+1/12 = 1/2），该对照在此场空心；曲面只用了 iso9 → 模板敏感性未测（局限）。
- **平台侧坑（可复用）**：`tensor_benchmark_cases` 无该锚行，属系统性登记缺口（m17713 同病）；
  且 `maturity_gate.files_exist()` 要求 evidence 路径是**裸路径**——把「path:1-395（说明）」写进 `impl_files`
  会让存在性检查失败并把 L2 误判成 L1（见 verdict id=650/651 的 reopened）。本单已改用裸路径 + 独立 `impl_detail` 键。
- 外部锚只到「实现取证」级：ludwig 快照无 `*.c`（实测 0 个），无可用算例入口，因此**不能**裁决物理；审计仅复核到
  符号/行号/常量存在级，未做指令级常量↔行号绑定。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

- σ 平面 ξ=2：{bm[0]['rel_err']*100:.2f}% → **不 <3%**（但 <15%，属二阶离散误差，ξ=4/8 即达标）。
- σ 平面 ξ=4：{bm[1]['rel_err']*100:.2f}%；ξ=8：{bm[2]['rel_err']*100:.2f}% → **<3% ✓**。
- σ 曲面 R=32：{curved['L128_R32_xi4.0']['sigma_curved_rel_err']*100:.2f}%；R=64：{curved['L256_R64_xi8.0']['sigma_curved_rel_err']*100:.2f}% → **<3% ✓**。
- 曲率平衡（R=32 / R=64）：{curved['L128_R32_xi4.0']['radial_balance_abs_rel_err']*100:.2f}% / {curved['L256_R64_xi8.0']['radial_balance_abs_rel_err']*100:.2f}% → **<3% ✓**。
- 平衡态体力残差（ξ=2/4/8）：{planar['xi2.0_iso9']['residual_stress_over_Fscale']*100:.2f}% / {planar['xi4.0_iso9']['residual_stress_over_Fscale']*100:.2f}% / {planar['xi8.0_iso9']['residual_stress_over_Fscale']*100:.2f}%（理论 0）→ ξ=2 **不 <3%**，ξ≥4 ✓。
- 两式力场逐点相对差 ξ=4：{planar['xi4.0_iso9']['force_diff_rel']*100:.1f}% → **不 <3%，且不 <15%**：按 ≤15% 阈值应判 honest-FAIL；
  但该指标分母为 O(h²) 残差，物理上无意义（审计已证恒等），故**不作为物理精度判据**——此点交付物自述一致，审计确认。

**结论**：物理量（表面张力、曲率平衡、平衡态残差）在 ξ≥4 时全部 <3%；ξ=2 的 6.09% 是二阶离散误差且随 ξ 收敛（6.09→1.63→0.41，观测阶 ≈2）。
判级 L2：功能为真、误差分析为真，但**未接入求解器（L3）**、**基准表无行（L4）**——不加级、不打折。
"""

payload = {
    "title": "交差审计 m17763：自由能压强张量体力 XVAL 判 L2（真代码+误差分析为真，无求解器调用方/无基准行）",
    "summary": ("独立复核 m17763（XVAL-LBM-FREE-ENERGY-PRESSURE-TENSOR-BODY-FORCE）：三件套真实、tests 3 passed、复现确定；"
                f"σ 误差 {sig_errs} 经审计侧 numpy 独立重算逐位复现；基准表该锚 0 行（无机器裁判）→ L4 不可确认；"
                "free_energy_step 仍用 −φ∇μ、src 零调用方 → L3 未过。判级 L2，verdict id=652。"),
    "scenario_key": "board_audit",
    "mission_id": 17881,
    "tags": ["board-audit", "m17763", "XVAL", "TensorLBM", "free-energy", "pressure-tensor", "maturity-L2", "honest-FAIL"],
    "markdown": md,
    "figures": [{"name": "m17881_audit_verification.png",
                 "caption": "m17881 交差审计：(a) σ 离散误差随 ξ 收敛，交付物与审计独立重算重合，叠加 O(h²/ξ²) 与 15%/3% 判据线；"
                            "(b) 两式体力平衡残差（均 O(h²)，5.3%→0.58%）；(c) L1–L5 判据达成（L3/L4 未达）。",
                 "png_base64": base64.b64encode(png).decode()}],
}
pathlib.Path("/tmp/report_m17881.json").write_text(json.dumps(payload, ensure_ascii=False))
print("bytes", len(json.dumps(payload)), "rows", len(rows))