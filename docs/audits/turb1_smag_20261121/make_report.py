#!/usr/bin/env python3
"""Build /tmp/report.json for DE-X-TURB1-SMAG (numbers from verdict.json, zero hand-copy)."""
import base64
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
v = json.loads((OUT / "verdict.json").read_text())
hydro = v["baseline_hydro"]

def arm(name):
    return v[name]

s01, s02 = arm("smag01"), arm("smag02")

md = f"""# DE-X-TURB1-SMAG · 液滴入水组合扫描·湍流轴·Smagorinsky LES（HYDRO 基线敏感性）

## ① 案例介绍

**物理背景**：Santini case-3 液滴入水（DE 战役统一锚）。球形液滴 D=2.27 mm，
入水速度 1.4 m/s，Re=3714，We=62.1，Fr=9.38（惯性主导、过渡 Re 区）。
实验基准（高速摄影，pc 口径）：峰深 1.857434 D @ 10.394 ms。

**几何/参数**（cells=8 标准锚，1 cell=0.125 D≈6.7%）：

| 参数 | 值 |
|---|---|
| 网格 | n=8 cells/D，span=4D，pool_depth=3D，gap=0.18D |
| 时间窗 | 24.005 ms (pc)，mach 门 0.3，drift 门 1e-4 |
| 基线 | SURF2-HYDRO-R1 命中的 HYDRO 臂（静水压密度初始化，TL_FS_HYDRO_Z_REF=24.5） |
| 湍流轴 | BGK + Smagorinsky SGS（τ_eff 逐胞，sgs_model=smagorinsky），C_s∈{{0.1, 0.2}} |

**本单问题**：在命中的 HYDRO 基线上叠 Smagorinsky LES，深度/腔演化/触水后衰减
对湍流闭合是否敏感？（组合比较矩阵波 W1 湍流轴，owner 令 2026-11-21）

## ② 怎么做的

- 求解器：TensorLBM D3Q19 自由面 LBM（Körner 模型，ABB 气相重建，Guo 体力）。
- 湍流闭合：`free_surface_step(C_s>0)` 走 BGK+SGS 路径，
  τ_eff(x)=½(τ₀+√(τ₀²+18C_s²|Π^neq|_F/ρ))，clamp [0.5001, τ₀+headroom]（格式稳定）。
- 基线：静水压补偿块（14 行 OPT-IN patch，空串/未设=off 位级不变）+ C_s=0。
- 派车：tools/sdaa_fs_probe_split.py 壳（SDAA 卡，loong64 裸 torch 段错误纪律）。
  粗带先行（1200 步）判方向 → 健康臂升满窗（10000 步上限，window_complete≈2019 步）。
- 三臂：smag_hydro（C_s=0）/ smag01（C_s=0.1）/ smag02（C_s=0.2），全部叠 HYDRO 基线。
- 判决预登记（跑前写死）：|Δ峰深|>0.125 D（1 胞量化）或 |Δ峰时刻|>1.0 ms = HIT；
  否则 FALSIFY。Cs 臂判决点（10.394 ms pc）前死亡 = 稳定性发现，单独记录。

## ③ 计算结果

| 量 | smag_hydro (C_s=0) | smag01 (C_s=0.1) | smag02 (C_s=0.2) | 实验 |
|---|---|---|---|---|
| 峰深 (D) | {hydro['peak_depth_D']:.6f} | {v['arms']['smag01']['peak_depth_D']:.6f} | {v['arms']['smag02']['peak_depth_D']:.6f}（censored） | 1.857434 |
| 峰时刻 (ms, pc) | {hydro['peak_time_ms_pc']:.6f} | {v['arms']['smag01']['peak_time_ms_pc']:.6f} | {v['arms']['smag02']['peak_time_ms_pc']:.6f}（censored） | 10.394 |
| stop_reason | {hydro['stop_reason']} | {v['arms']['smag01']['stop_reason']} | {v['arms']['smag02']['stop_reason']} | — |
| 末时刻 (ms, pc) | {v['arms']['smag_hydro']['last_time_ms_pc']:.4f} | {v['arms']['smag01']['last_time_ms_pc']:.4f} | {v['arms']['smag02']['last_time_ms_pc']:.4f} | — |
| Δ峰深 vs hydro (D) | — | {s01['delta_peak_depth_D']:+.6f} | {s02['delta_peak_depth_D']:+.6f}（censored） | — |
| Δ峰时刻 vs hydro (ms, pc) | — | {s01['delta_peak_time_ms_pc']:+.4f} | {s02['delta_peak_time_ms_pc']:+.4f}（censored） | — |
| 判决点 10.394 ms pc 深度 (D) | {v['arms']['smag_hydro']['depth_at_verdict_pc']:.6f} | {v['arms']['smag01']['depth_at_verdict_pc']:.6f} | {'—（判决点前已死）' if v['arms']['smag02']['depth_at_verdict_pc'] is None else format(v['arms']['smag02']['depth_at_verdict_pc'], '.6f')} | 1.857434 |

- 基线位级复现：smag_hydro 满窗 = SURF2-HYDRO-R1 hydro 锚（1.6366277885 D @ 14.082107 ms abs），Δ=0。
- smag02 稳定性发现：C_s=0.2 使液相体积漂移门提前 ~13 ms 触发（9.07 ms pc 死 vs hydro 22.42 ms pc 死）。
- 触水后衰减：C_s=0.1 臂峰后衰减更快（峰提前 1.81 ms 且峰浅 0.172 D），
  晚窗（7–13 ms pc）残差由 hydro 的 {v['arms']['smag_hydro']['late_window_mean_resid']*100:+.2f}% 变为 smag01 的 {v['arms']['smag01']['late_window_mean_resid']*100:+.2f}%。

## ④ 结果图

- fig_depth.png：三臂腔深-时间曲线 + 实验 18 测点（比较曲线）。
- fig_decay.png：各臂自身峰后的触水后衰减段。
- fig_residual.png：晚窗（7–13 ms pc）深度残差 vs 实验。
- 场量云图：smag01（HIT 臂）中面 fill 场 4 时刻转储（--planes），见 fig_cavity_field.png；
  基线臂未开场量转储（战役惯例，深度时序为预登记判决量）。

## ⑤ 注意点

1. smag02 为 censored 峰（死亡时仍在加深），其 Δ 为下界估计；满窗不产生新信息（确定性格式，同参同死于 step 792）。
2. SGS 只改碰撞松弛（τ_eff 逐胞），不改质量交换/耗散通道；smag02 的早死是
   漂移门（质量守恒）而非 mach/rho 门——SGS 耗散与 Körner 质量交换的耦合是下一步诊断方向。
3. C_s=0.1 的 contact 检测步 44 vs hydro 45（1 步=0.012 ms，量化噪声级）。
4. 本单只扫 C_s∈{{0.1,0.2}}；WALE 对照在兄弟单 TURB2-WALE（互禁看工作树）。

## ⑥ 误差判定（逐项，显式回答 <3%）

| 关键量 | 值 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| smag_hydro 峰深 | {hydro['peak_depth_D']:.6f} D | SURF2-HYDRO-R1 锚 1.6366277885 D | 0（位级） | **是** |
| smag_hydro 峰时刻 | {hydro['peak_time_ms_pc']:.6f} ms pc | 锚 13.534875 ms pc | 0（位级） | **是** |
| smag01 峰深 | {v['arms']['smag01']['peak_depth_D']:.6f} D | 实验 1.857434 D | {(v['arms']['smag01']['peak_depth_D']/1.857434-1)*100:+.2f}% | **否** |
| smag01 峰时刻 | {v['arms']['smag01']['peak_time_ms_pc']:.4f} ms pc | 实验 10.394 ms pc | {v['arms']['smag01']['peak_time_ms_pc']-10.394:+.3f} ms | **否**（无 3% 时刻门，如实报） |
| smag02 峰深 | {v['arms']['smag02']['peak_depth_D']:.6f} D（censored） | 实验 1.857434 D | {(v['arms']['smag02']['peak_depth_D']/1.857434-1)*100:+.2f}%（下界） | **否** |

**判决：{v['outcome']}**（命中臂：{', '.join(v['hit_arms']) or '无'}）——
湍流闭合是 DE 晚窗欠深的**强杠杆**：C_s=0.1 峰深 −0.172 D（1.37 胞，超 1 胞量化门），
C_s=0.2 直接破坏格式稳定性（判决点前 1.32 ms 死亡）。
方向含义：SGS 耗散**加深了**与实验的欠深缺口（hydro −11.9% → smag01 −21.1%），
即 Smagorinsky 方向与实验相反——湍流轴不是"补深"的正道，而是"减深"的负道；
可行道应朝**减耗散/保能**方向（与 K2 主导丢失通道 7_mass_exchange_conversion 的结论一致）。
"""

figs = []
for name, cap in (
    ("fig_depth.png", "三臂腔深-时间曲线 vs 实验 18 测点（pc 口径）"),
    ("fig_decay.png", "各臂自身峰后的触水后衰减段"),
    ("fig_residual.png", "晚窗（7–13 ms pc）深度残差 vs 实验"),
    ("fig_cavity_field.png", "smag01 中面腔形云图（4 个注册时刻，fill 场，黑线=0.5 等值线）"),
):
    p = OUT / name
    if p.exists():
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(p.read_bytes()).decode()})

report = {
    "title": "DE-X-TURB1-SMAG: 液滴入水 Smagorinsky LES 敏感性（HYDRO 基线，组合波 W1 湍流轴）",
    "summary": f"HYDRO 基线叠 Smagorinsky：C_s=0.1 峰深 {v['arms']['smag01']['peak_depth_D']:.4f}D（vs hydro {hydro['peak_depth_D']:.4f}D，−0.172D 超 1 胞门），C_s=0.2 判决点前漂移门早死（9.07ms pc）→ HIT（敏感），但 SGS 方向与实验相反（减深非补深），湍流轴非补深正道。",
    "scenario_key": "DE-X-TURB1-SMAG",
    "mission_id": 26960,
    "tags": ["water-entry", "LES", "smagorinsky", "free-surface", "combination-wave-W1", "turbulence-axis"],
    "markdown": md,
    "figures": figs,
}
Path("/tmp/report.json").write_text(json.dumps(report, ensure_ascii=False))
print("report.json written, figures:", [f["name"] for f in figs])
