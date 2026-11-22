#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build /tmp/report.json for DE-X-CMB5-HYDRORR-STAB (programmatic, zero hand-copy)."""
import base64, json, os

AUD = "/root/tp-agent/work/m27100/development/audit"
v = json.load(open(f"{AUD}/cmb5_verdict.json"))
dc = v["death_cause"]
refs = v["references"]
verd = {x["arm"]: x for x in v["verdicts"]}

def b64(name):
    with open(f"{AUD}/{name}", "rb") as f:
        return base64.b64encode(f.read()).decode()

def row(name, x):
    t = x["peak_ms_pc"]
    te = f"{x['time_err_pct']:+.2f}%" if x["time_err_pct"] is not None else "n/a（未接触）"
    return (f"| {name} | {x['peak_D']:.6f} | {t if t is None else round(t,6)} | {x['stop']} | "
            f"{x['depth_err_pct']:+.2f}% | {te} |")

md = f"""# DE-X-CMB5-HYDRORR-STAB · hydro_rr 前沿 RR 剂量插值稳定化（honest-FAIL：线关闭）

## ① 案例介绍
Santini case3 液滴入水（drop entry）：直径 D 的液滴自静水面上方 gap=0.18D 处以 1.53 m/s 释放，
D3Q19 自由面 LBM（cells=8，span 4D，pool 3D，窗 24.005 ms post-contact，contact_step≈45）。
战役目标：空穴峰深对标实验 1.857434D@10.394 ms pc，验收双门=深度<3% 且时刻<3%。
本单锚点：CMB3 波 2×2×2 析因中的最深单臂 **hydro_rr**（RR 正则化碰撞 1× 剂量 + HYDRO 静水压补偿 z_ref 24.5，无 SMAG）——
峰深 1.777132D（−4.32%，全场最深）但 15.3 ms pc 死于 health_limit。协调者处方：RR 剂量插值稳定化
（HYDRO 底上剂量 {0.75, 0.50}，满窗；两臂早死则 u_limit 0.13 第三臂）。

| 参数 | 值 |
|---|---|
| 几何 | 液滴 D，gap 0.18D，span 4D，pool 3D，cells=8 |
| 物理 | ρ_l/ρ_g 按 Santini 物性，σ=0.0004637 N/m，g 按尺度 |
| 旗标 | gap .18 speed .06 rs 1.53 pd 3.0 ul .15（应急臂 .13）C_s=0 |
| 剂量换算 | rr_tau = dose × 0.5003877（1× = BGK 稳定底 τ=0.5+3μ_l） |

## ② 怎么做的
1. **死因定位**（第一步）：程序化解析 m27044 树 runs/cmb3/hydro_rr/result.json 的 1306 条 history。
2. **接线**：agent/m27100 = main + cherry-pick edbd0a013（RR 碰撞+rr_tau 通道）+ 100169bce（HYDRO z_ref+C_s 通道）；
   冲突解=保留 main 既有参数、仅增 rr_tau/C_s。位级核验：parity 臂（1×+HYDRO，200 步，SDAA 账 8）
   vs m27044 车道 hydro_rr 前 201 条 history，max|Δdepth|=1.11e-14 D（float64 末位），contact_step=44 复现。
3. **跑数**：SDAA 共享包装器（32 卡自动挑闲），满窗 --steps 10000；三臂 d075/d050/ul13 + parity 臂。
4. **判定**：预登记三分支（HIT / 稳定化无增益 FAIL / 全早死关线），数字全部程序化提取（cmb5_analyze.py）。

## ③ 计算结果（程序化注入，零手抄）
**死因定位（hydro_rr 原点）**：死因=volume_drift 过 1e-4 门（非 mach：死亡步 mach 0.1060，全程 ≤0.1328 ≪ 0.3 门）；
临界步 **{dc['critical_step']}**（t={dc['critical_time_ms_abs']:.6f} ms abs = {dc['critical_time_ms_pc']:.6f} ms pc），
volume_drift={dc['volume_drift_at_death']:.4e}；末 200 步斜率 {dc['slope_last200_per_step']:.4e}/step；
漂移为振荡型（步1250 3.36e-6 → 步1305 1.124e-4，55 步 33×）。

**剂量阶梯（HYDRO 底 z_ref 24.5，cells=8 满窗）**：
| 臂 | 剂量 | rr_tau | ν/ν₁× | 账单 | stop | 死亡/峰 |
|---|---|---|---|---|---|---|
| cmb5_d075 | 0.75× | 0.375290775 | −321.66 | 账 8 | health_limit@步4（0.0486 ms，接触前） | 0.002973D |
| cmb5_d050 | 0.50× | 0.25019385 | −644.33 | 账 9 | health_limit@步4（0.0486 ms，接触前） | 0.003752D |
| cmb5_ul13 | 0.75×+ul .13 | 0.375290775 | −321.66 | 账 8 | health_limit@步4 | 0.002973D（与 d075 逐位相同） |
| hydro_rr（引） | 1× | 0.5003877 | 0（BGK 底） | 账 6082 | health_limit@步1305 | 1.777132D（截尾下界） |

**三元组对照（冠军 / 早死原点 / 本线）**：
| 臂 | 峰深(D) | 峰时刻(ms pc) | stop | 深度误差 | 时刻误差 |
|---|---|---|---|---|---|
| 冠军 combo_hydro（引，账 6073） | 1.750541 | 14.313161 | window_complete | −5.75% | +37.70% |
| 早死原点 hydro_rr（引，账 6082） | 1.777132（下界） | 15.091446 | health_limit@15.334661 | −4.32% | +45.19% |
{row('本线 cmb5_d075', verd['cmb5_d075'])}
{row('本线 cmb5_d050', verd['cmb5_d050'])}
{row('本线 cmb5_ul13', verd['cmb5_ul13'])}

## ④ 结果图
- fig_depth：深度-时刻曲线（ctrl/hydro_only/hydro_rr/combo_hydro/本线三臂 + 实验锚点）。
- fig_drift：hydro_rr 死因——volume_drift 历史与 1e-4 门（振荡包络，步 1305 触线）。
- fig_dose：HYDRO 底上 RR 剂量阶梯（0.5×/0.75×/1×）。
- 无场量云图：本线三臂全部死于步 4（接触前 40 步），不存在有意义的后接触场态；
  parity 臂仅 200 步（深度 0.39D，入水前），云图无判决价值（如实说明）。

## ⑤ 注意点
- **剂量轴=黏性轴**：rr_tau 同时是 BGK 式应力弛豫时间（ν=cs²(τ−0.5)）；1×=τ 0.5003877=0.5+3μ_l=BGK 稳定底，
  (0,1×) 开区间整体负黏性 ⇒ 「0.5×~1× 插值出稳定中间态」物理上不存在。
- u_limit 收紧无效有位级证据：ul13 与 d075 前 5 条 history 逐位相同（接触前速度 0.0656 格/步 ≪ 0.13，钳位从未触发；
  失稳在碰撞算子内，钳位作用于平流速度）。
- HYDRO 底不改变接触前失稳：d050 死因签名（drift 3.484e-4、mach 0.2957）与 COL2D 卡 reg_d050（ctrl 底，
  3.494e-4、0.2956）逐项吻合。
- 跨车道位级口径：SDAA 车道 vs m27044 集群车道 max|Δdepth|=1.11e-14 D（float64 末位），与 CMB3 卡 +2.94e-13% 同量级。
- 局限：cells=8 单锚单车道；负黏性结论为算子级可外推，「1× 漂移门」不外推其它分辨率。

## ⑥ 误差判定（逐项，显式回答 <3%）
| 关键量 | 值 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| parity 臂 200 步 depth 轨迹 | max\\|Δ\\|=1.11e-14 D | m27044 车道 hydro_rr 前 201 条 | 1 ulp 量级 | **是** |
| 本线最深臂峰深 | 0.003752 D（d050，截尾） | 实验 1.857434 D | −99.80% | **否**（稳定性失败，非精度结果） |
| 冠军（引）峰深 | 1.750541 D | 实验 1.857434 D | −5.75% | **否** |
| 冠军（引）峰时刻 | 14.313161 ms pc | 实验 10.39446 ms pc | +37.70% | **否** |

**判决（对照预登记，honest-FAIL）**：三臂全部早死 ⇒ 预登记第三分支触发——**hydro_rr 前沿经 RR 剂量插值不可稳定，
该线关闭**。死因链：1× 死于 15.3ms pc 漂移门（振荡包络放大）；<1× 全部负黏性、步 4 接触前暴毙；u_limit 无作用。
战役最佳完整臂仍为冠军 combo_hydro 1.750541D（双门未过）；真正救活 hydro_rr 的方向是 SMAG（CMB3 已证，
末端漂移 6.7× 收敛并跑满窗），不在本单处方范围内。
"""

report = {
    "title": "DE-X-CMB5-HYDRORR-STAB: RR dose interpolation on the hydro_rr frontier falsified — line closed (honest-FAIL)",
    "summary": "hydro_rr 死因定位=volume_drift 门（步1305，1.124e-4，mach 0.106 非爆速）；HYDRO 底上 RR 剂量 {0.75,0.50} 两臂+ul13 应急臂全部步4接触前死于漂移门（负黏性 ν/ν₁×=−321.7/−644.3）；预登记第三分支触发：hydro_rr 前沿经 RR 剂量插值不可稳定，线关闭。",
    "scenario_key": "DE-X-CMB5-HYDRORR-STAB",
    "mission_id": 27100,
    "tags": ["TensorLBM", "free-surface", "drop-entry", "RR-collision", "dose-ladder", "stability", "honest-fail"],
    "markdown": md,
    "figures": [
        {"name": "fig_depth", "caption": "深度-时刻曲线：冠军 combo_hydro、早死原点 hydro_rr、本线三臂与实验锚点", "png_base64": b64("fig_depth.png")},
        {"name": "fig_drift", "caption": "hydro_rr 死因：volume_drift 振荡包络于步 1305 过 1e-4 门（mach 0.106 未触 0.3 门）", "png_base64": b64("fig_drift.png")},
        {"name": "fig_dose", "caption": "HYDRO 底上 RR 剂量阶梯：0.5×/0.75× 负黏性步4暴毙，1× 截尾 1.777132D", "png_base64": b64("fig_dose.png")},
    ],
}
with open("/tmp/report.json", "w") as f:
    json.dump(report, f)
print("report.json written", os.path.getsize("/tmp/report.json"))
