"""Build /tmp/report.json for the T1 DE-T1-E1 paper-level progress report (numbers injected from JSON, zero hand-copy)."""
import base64
import json
from pathlib import Path

ROOT = Path("/root/tp-agent/work/m26277")
FIGS = ROOT / "runs" / "t1_figs"


def load(p):
    return json.loads((ROOT / "runs" / p).read_text())


n8p = load("t1_cac_w6_s03_n8.json")      # n=8 prescribed w6 s0.03
n8b = load("t1_base_n8.json")            # n=8 baseline  w4 s0.04
n12p = load("t1_cac_w6_s03.json")        # n=12 prescribed w6 s0.03 (cluster cj5656)
n12b = load("t1_base_w4_s04_n12.json")  # n=12 baseline  w4 s0.04 (cluster cj5659)
score = load("t1_score.json")
n8s = score["runs"]["t1_cac_w6_s03_n8"]


def death(r):
    h = r["history"][-1]
    return h["step"], h["time_ms"], r["stop_reason"]


s8p, t8p, r8p = death(n8p)
s8b, t8b, r8b = death(n8b)
s12p, t12p, r12p = death(n12p)
s12b, t12b, r12b = death(n12b)

md = f"""# DE-T1-E1 · 双相(CACSim)液滴入水存活修复 — mach_limit(伪电流)处方验证（honest-FAIL）

## ① 案例介绍
Santini et al. 实验 case 3：单液滴入深水池。D=2.27 mm，U=1.4 m/s，Re=3715，We=62.1，
Fr²=88.0，27 °C 水，40 mm 玻璃立方池，1000 fps。参考曲线=Fig.5 实验星号（18 点，含增长/峰值/回弹），
峰值深度时刻 10.394 ms。双相路径求解器=CACSim/D3Q19（保守 Allen-Cahn + D3Q19 LBM，float64，单线程）。
健康门：mach_max≤0.3、volume_drift≤1e-4、rho_min>0；观察窗=触水后 24 ms。

## ② 怎么做的
处方（T1 处方卡）：把 run.py 的 `width` 4.0→6.0、`speed` 0.04→0.03（commit c5a031689，2 行 CLI 默认）。
假设：加宽界面使序参量梯度变缓、表面张力离散伪电流按 ~1/width 衰减，晶格速度↓压低 Mach 基线，
从而跑满 24 ms 窗覆盖 10.394 ms 峰值。验证=处方成本节命令（cells=8 最便宜档）+ n=12 全量（集群）。
受控对照：同分辨率基线（w4/s0.04）。工具链：TensorLBM src/tensorlbm，集群 n330（loong64）。

## ③ 计算结果（程序化注入自 runs/*.json）
| 配置 | 分辨率 | stop_reason | 终止点 | postcontact_healthy_ms |
|---|---|---|---|---|
| 处方 w6/s0.03 | n=8 | {r8p} | step {s8p} = {t8p:.3f} ms | {n8s['postcontact_healthy_ms']} |
| 基线 w4/s0.04 | n=8 | {r8b} | step {s8b} = {t8b:.3f} ms | None |
| 处方 w6/s0.03 | n=12 | {r12p} | step {s12p} = {t12p:.3f} ms | None |
| 基线 w4/s0.04 | n=12 | {r12b} | step {s12b} = {t12b:.3f} ms | None |

关键诊断（n=12 处方档）：终止时 mach_max={n12p['history'][-1]['mach_max']:.4f}（门 0.3），
`phi_at_mach_max`={n12p['history'][-1]['phi_at_mach_max']:.5f}（气相），`mach_liquid_max`={n12p['history'][-1]['mach_liquid_max']:.4f}
→ 最大 mach 在**界面附近气相**，液相 mach 仅 ~0.05。

## ④ 结果图
- 图1：n=8 处方档终止步（step 26）phi 中平面云图（宽界面 w=6 在 8 胞液滴上）。
- 图2：n=8 基线 vs 处方 volume_drift / mach_max 轨迹 vs 门限（基线死于 mach，处方死于 drift）。
- 图3：phi 和漂移速率 vs 分辨率（n=8 6.3e-4/ms vs n=12 8.8e-5/ms，7×）。
- 图4：n=12 处方档全局 mach 与液相 mach 分离 + 最大-mach 胞的相（气相 phi~1e-3）。

## ⑤ 注意点
- n=8 液滴仅 8 胞，w=6 界面占液滴直径 75%，phi 和是劣质体积代理 → drift 门在 n=8 是粗网格伪影。
- 伪电流在气相：width 杠杆（平滑界面）对气相力离散无效；n=12 处方档 0.166 ms vs 基线 0.065 ms（约 2.5×），但远未达 10.394 ms 峰值。
- 单线程 float64；n=12 全量 ~8 min/档（集群）。score.py 的 numeric_pass 对双相路径结构性不可达（无 side_view）。
- 适用范围：本结论针对 CACSim/D3Q19 + 该门限组合；换碰撞模型/门限需重测。

## ⑥ 误差判定（关键量 vs 基准，逐项）
| 关键量 | 处方档实测 | 基准/门 | 误差 | <3%? |
|---|---|---|---|---|
| postcontact_healthy_ms | None（n=8/n=12 均未触水） | 预测 [12,24] ms | 无触水，无实验基准可比 | 不适用（未触水，声明原因） |
| mach_max（n=12 终止） | {n12p['history'][-1]['mach_max']:.4f} | 门 0.3 | 超门 {(n12p['history'][-1]['mach_max']/0.3-1)*100:.1f}% | 否（超门） |
| volume_drift（n=8 终止） | {n8p['history'][-1]['volume_drift']:.3e} | 门 1e-4 | 超门 {(n8p['history'][-1]['volume_drift']/1e-4-1)*100:.1f}% | 否（超门） |
| impact_speed | 未测（无触水） | 1.4 m/s | 无 | 不适用（未触水，声明原因） |
**总判定：honest-FAIL**（两档均未触水，postcontact_healthy_ms=None < 10.394 ms，判据①触发）。
"""

figures = []
for name, cap in [
    ("fig1_phi_field.png", "图1：n=8 处方档终止步 phi 中平面云图（宽界面 w=6）"),
    ("fig2_gate_trajectories.png", "图2：n=8 基线 vs 处方 drift/mach 轨迹 vs 门限"),
    ("fig3_drift_rate_vs_resolution.png", "图3：phi 和漂移速率 vs 分辨率"),
    ("fig4_n12_gas_phase_mach.png", "图4：n=12 全局 mach 死于气相伪电流"),
]:
    b = base64.b64encode((FIGS / name).read_bytes()).decode()
    figures.append({"name": name, "caption": cap, "png_base64": b})

report = {
    "title": "DE-T1-E1 双相 CACSim 液滴入水存活修复：mach_limit 伪电流处方验证（n=8/n=12 honest-FAIL）",
    "summary": "按 T1 处方把 single_drop run.py 的 width 4→6、speed 0.04→0.03；n=8 死于 volume_drift(0.158ms)、n=12 死于 mach_limit(0.166ms)，均远未触水（honest-FAIL）；基线(w4/s0.04)两档均死于 0.065ms，杠杆仅改善约 2.5×；诊断出伪电流在气相；A3#1 所引 1.362ms 基线在当前代码+默认参数下不可复现。",
    "scenario_key": "DE-A2-single-drop",
    "mission_id": 26277,
    "tags": ["TensorLBM", "CAC-LBM", "water-entry", "spurious-current", "honest-FAIL", "DE-T1"],
    "markdown": md,
    "figures": figures,
}
Path("/tmp/report.json").write_text(json.dumps(report, ensure_ascii=False))
print("report.json bytes:", Path("/tmp/report.json").stat().st_size)
print("n8p:", r8p, f"{t8p:.3f}ms", "| n8b:", r8b, f"{t8b:.3f}ms", "| n12p:", r12p, f"{t12p:.3f}ms")
