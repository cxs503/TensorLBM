"""Build and publish the T5 solo progress report (programmatic numbers)."""
import json
import os
import subprocess

MINE = "/tmp/cj5662x/cj5662/repo/runs/t5_solo/t5_combined8.json"
BASE = "/tmp/t5_baseline8.json"
FIGS = json.load(open("/tmp/t5_figures.json"))

import numpy as np

d = json.load(open(MINE))
h = d["history"]
ts = np.array([r["time_ms"] for r in h])
ds = np.array([r["depth_over_D"] if r.get("depth_over_D") is not None else np.nan for r in h])
valid = ~np.isnan(ds)
peak_d = float(ds[valid].max())
peak_t = float(ts[valid][np.argmax(ds[valid])])
cs = d["contact_step"]
dt = d["scales"]["dt_s"]
death = h[-1]
pc = (death["step"] - 1 - cs) * dt * 1000
impact = d["impact_speed_m_s"]
max_inv = max(r["volume_drift"] for r in h)
max_geom = max(r["geometric_volume_drift"] for r in h)

b = json.load(open(BASE))
bh = b["history"]
bcs = b["contact_step"]
bdt = b["scales"]["dt_s"]
bpc = (bh[-1]["step"] - 1 - bcs) * bdt * 1000
bimpact = b["impact_speed_m_s"]

exp = {1.39643: 0.512426, 6.4: 1.647387, 10.39446: 1.857434, 13.39136: 1.304231}
rows = []
for t, e in exp.items():
    if ts[-1] >= t:
        s = float(np.interp(t, ts, ds))
        rows.append((t, e, s, abs(s - e) / e * 100))
depth_table = "\n".join(
    f"| {t:.3f} | {e:.4f} | {s:.4f} | {err:.1f}% |" for t, e, s, err in rows
)
peak_err = abs(peak_d - 1.857434) / 1.857434 * 100
speed_err = abs(impact / 1.40 - 1) * 100

md = f"""# T5 单打对照 · 单滴入深水池（Santini case-3）定量记录

## ① 案例介绍
单滴入深水池：直径 D=2.27±0.03 mm、撞击速度 U=1.40±0.02 m/s 的水滴（27°C，ρ=997 kg/m³，μ=0.853 mPa·s，σ=71.42 mN/m）落入 40 mm 玻璃立方水槽（池水 30°C）。无量纲数 Re=3715、We=62.1、Fr=88.0。实验（1000 帧/s）观测气穴峰值 **1.8574 D @ 10.394 ms**，共 18 个测点。验收门：逐点与峰值相对误差 ≤3%、两档网格误差严格递减、体积漂移 ≤1e-4、Ma ≤0.3。

## ② 怎么做的
- 求解器：`free_surface_step/D3Q19`（Körner 自由面，常压大气，BGK），D3Q19，float64，单线程。
- 网格：8 cells/D（dx=2.84e-4 m），lattice speed 0.06（dt=1.216e-5 s），span=4D、pool_depth=3D、gap=0.18D。
- 配置：`--simple-exchange --capacity --release-speed 1.53`（组合配置，复现归档 cal8）。
- 工具链：本地诊断（runtime mass ledger 算子归因）+ 集群 n330（loong64）全窗运行（cj5662）。
- 本单补丁：`free_surface_probe.py` 增 `--ledger`（诊断）；`solver3d.py` MRT 秩检查 np→torch（修 loong64 导入断言）。

## ③ 计算结果
| 量 | 值 |
|---|---|
| postcontact_healthy_ms（组合配置, cells=8） | **{pc:.4f}** |
| postcontact_healthy_ms（baseline, cells=8） | **{bpc:.4f}** |
| 终止 | health_limit @ {death['time_ms']:.3f} ms（Ma {death['mach_max']:.4f}） |
| 撞击速度 | {impact:.5f} m/s（对 1.40 误差 {speed_err:.2f}%） |
| 最大库存漂移 | {max_inv:.3e}（门 1e-4） |
| 最大几何体积漂移 | {max_geom:.3e} |
| 全窗峰值（历史） | {peak_d:.4f} D @ {peak_t:.3f} ms |

深度对照（组合配置 vs 实验，np.interp）：
| t (ms) | 实验 | 模拟 | 相对误差 |
|---|---|---|---|
{depth_table}

## ④ 结果图
图1：气穴深度时序（模拟 vs 实验 18 点）。图2：10.397 ms 中平面 fill 快照（归档组合 8 格算例）。

## ⑤ 注意点
- 组合配置死于 13.851 ms（Ma 0.3006），未覆盖末两个实验点（15.4/17.4 ms）。
- 库存漂移根因=单向 L/I 质量交换（interface 记贷、liquid 不记借），是参考模型设计后果，未改。
- 本机工具 120 s 超时，全窗只能上集群；集群首次导入因 numpy LAPACK SVD 坏而失败，已修。
- 首点 0.0389 D 低于实验 3% 分辨率（0.0132 D），归档保留并计败。

## ⑥ 误差判定（关键量 vs 基准，逐项）
| 量 | 模拟 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| 峰值深度 | {peak_d:.4f} D | 1.8574 D | {peak_err:.1f}% | **否** |
| 撞击速度 | {impact:.4f} m/s | 1.40 m/s | {speed_err:.2f}% | 是 |
| 10.394 ms 深度 | {rows[2][2]:.4f} D | 1.8574 D | {rows[2][3]:.1f}% | **否** |
| 13.391 ms 深度 | {rows[3][2]:.4f} D | 1.3042 D | {rows[3][3]:.1f}% | 否（3.4%） |
**结论：3% 定量验收 FAIL（honest-FAIL）。** 稳定性目标（活过 10.4 ms 峰值时刻）达成（{pc:.3f} ms > 10.394 ms）；精度目标未达成（峰值误差 {peak_err:.1f}%）。
"""

report = {
    "title": "T5 单打对照：单滴入深水池全流程（诊断+集群导入修复+实测 postcontact_healthy_ms）",
    "summary": (
        f"单体全流程复现：baseline(cells=8) postcontact_healthy_ms={bpc:.4f} ms；"
        f"组合配置={pc:.4f} ms（复现归档 cal8，深度 15 位一致）。"
        f"峰值深度误差 {peak_err:.1f}%，3% 验收 honest-FAIL。"
        "附带修复 loong64 集群 D3Q19 MRT 导入断言（np→torch 秩检查）。"
    ),
    "scenario_key": "DE_single_drop_water_entry",
    "mission_id": 26274,
    "tags": ["DE", "single_drop", "water_entry", "solo_control", "TensorLBM", "honest-fail"],
    "markdown": md,
    "figures": [
        {
            "name": "crater_depth_t5_solo.png",
            "caption": "气穴深度时序：模拟（组合配置, 8 cells/D）vs Santini case-3 实验 18 点",
            "png_base64": FIGS["fig1"],
        },
        {
            "name": "midplane_fill_10p4ms.png",
            "caption": f"中平面 fill 快照 @ {FIGS['plane_time_ms']:.3f} ms（归档组合 8 格算例）",
            "png_base64": FIGS["fig2"],
        },
    ],
}
json.dump(report, open("/tmp/report.json", "w"), ensure_ascii=False)
print("report.json written; pc=", round(pc, 4), "bpc=", round(bpc, 4))
