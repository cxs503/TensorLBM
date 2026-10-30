import json
import base64
from pathlib import Path

def img_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

profile_b64 = img_b64('/tmp/stokes2_profile_H100_kH6.png')
conv_b64 = img_b64('/tmp/stokes2_convergence.png')

scan_dir = Path('/root/tp-agent/work/m302/benchmarks/verified/stokes_second_problem/scan_out')
result = json.loads((scan_dir / 'result.json').read_text())

lines = []
lines.append("# Stokes 第二问题（振荡平板）验证报告")
lines.append("")
lines.append("## ① 案例介绍")
lines.append("")
lines.append("**物理背景**：半无限静止流体中，无限平板在自身平面内作谐波振荡 $u_w(t) = U\\cos(\\omega t)$。稳态周期解析解（Batchelor; Landau-Lifshitz §24）：")
lines.append("")
lines.append("$$u(y,t) = U e^{-ky} \\cos(\\omega t - ky), \\quad k = \\sqrt{\\frac{\\omega}{2\\nu}}$$")
lines.append("")
lines.append("幅值随深度指数衰减，相位线性滞后，均由单一穿透波数 $k$ 决定。")
lines.append("")
lines.append("**几何与参数**：")
lines.append("")
lines.append("| 参数 | 符号 | 值 |")
lines.append("|---|---|---|")
lines.append("| 格子模型 | — | D2Q9 |")
lines.append("| 碰撞模型 | — | BGK |")
lines.append("| 松弛时间 | $\\tau$ | 0.8 |")
lines.append("| 格子粘度 | $\\nu_{lb}$ | 0.1 |")
lines.append("| 板速幅值 | $U$ | 0.05 |")
lines.append("| 马赫数 | $Ma$ | 0.0866 |")
lines.append("| 无量纲波数×域高 | $kH$ | 6, 8 |")
lines.append("| 网格尺寸 | $H$ | 50, 100, 200 |")
lines.append("| 周期步数 | $T_{int}$ | 2182–34907 |")
lines.append("| 总步数 | — | 25,767–488,698 |")
lines.append("")
lines.append("**边界条件**：")
lines.append("- 振荡平板 = 顶行 ($y=ny-1$)，每步以 $u_{lid}(t)=U\\cos(\\omega t)$ 调用库移动盖 `zou_he_moving_lid`")
lines.append("- 远场 = 底行 ($y=0$)，自由滑移镜面反射（SPECULAR 置换）")
lines.append("- $x$ 方向周期")
lines.append("")
lines.append("## ② 方法")
lines.append("")
lines.append("**仿真流程**：")
lines.append("1. 初始化：静止平衡态 ($\\rho=1, u=0$)")
lines.append("2. 跳过瞬态：$skip = \\max(10, \\lceil 8\\tau_{diff}/T \\rceil)$ 个整周期（$\\tau_{diff}=H^2/(\\pi^2\\nu)$）")
lines.append("3. 测量窗口：恰好 4 个整周期，每步记录 $\\bar{u}_x(y)$ 的 Fourier 系数")
lines.append("4. 复振幅提取：$\\hat{u}(y) = \\frac{2}{N}\\sum u_x \\cos(\\omega t) - i\\frac{2}{N}\\sum u_x \\sin(\\omega t)$")
lines.append("5. 与解析解比较：幅值比、相位滞后、复振幅相对误差")
lines.append("")
lines.append("**关键设置**：")
lines.append("- 远场必须用自由滑移（镜面反射），无滑移远壁会污染衰减尾部（~2.7% 不可细化地板）")
lines.append("- 编译模式：GPU 上 `torch.compile(default)`，SDAA 上自动 fallback 到 eager")
lines.append("- 所有内核来自库函数，无手写 collide/stream/equilibrium")
lines.append("")
lines.append("**工具链**：")
lines.append("- `tensorlbm.solver.collide_bgk` / `stream`")
lines.append("- `tensorlbm.lid_driven_cavity.zou_he_moving_lid`")
lines.append("- `tensorlbm.d2q9.equilibrium` / `macroscopic`")
lines.append("- 自由滑移：内联 SPECULAR 置换（`f_pre[SPECULAR]`）")
lines.append("")
lines.append("## ③ 计算结果")
lines.append("")
lines.append("| $kH$ | $H$ | 幅值 max | 相位 max (rel; abs) | 复振幅 max | 总步数 |")
lines.append("|---|---|---|---|---|---|")

for r in result['per_grid']:
    lines.append(f"| {r['kH']} | {r['H']} | {r['amp_rel_max_pct']:.3f}% | {r['phase_rel_max_pct']:.3f}% ({r['phase_err_max_deg']:.2f}°) | {r['complex_rel_max_pct']:.3f}% | {r['total_steps']:,} |")

lines.append("")
lines.append("**网格收敛**：")
lines.append("")
lines.append("| $kH$ | $H=50$ | $H=100$ | $H=200$ | 单调下降 |")
lines.append("|---|---|---|---|---|")

for kh_data in result['convergence']['per_kH']:
    errs = kh_data['complex_rel_max_pct']
    dec = '✓' if kh_data['err_decreased'] else '✗'
    lines.append(f"| {kh_data['kH']} | {errs[0]:.3f}% | {errs[1]:.3f}% | {errs[2]:.3f}% | {dec} |")

max_err = max(r['complex_rel_max_pct'] for r in result['per_grid'])
lines.append("")
lines.append(f"**最大误差**：{max_err:.3f}%（$kH=8, H=50$ 粗网格）")
lines.append("")
lines.append("## ④ 结果图")
lines.append("")
lines.append("### 图1：幅值与相位剖面（$H=100, kH=6$）")
lines.append("")
lines.append("![Amplitude and Phase Profile](/tmp/stokes2_profile_H100_kH6.png)")
lines.append("")
lines.append("左：幅值随深度指数衰减，LBM 与解析解 $Ue^{-ky}$ 重合。右：相位滞后 $\\phi=ky$ 线性增长，LBM 精确跟踪。")
lines.append("")
lines.append("### 图2：网格收敛")
lines.append("")
lines.append("![Grid Convergence](/tmp/stokes2_convergence.png)")
lines.append("")
lines.append("两条 $kH$ 线在三档网格上均严格单调收敛，最大误差 0.425% 远低于 3% 门限。")
lines.append("")
lines.append("## ⑤ 注意点")
lines.append("")
lines.append("1. **远场边界选择**：无滑移远壁会污染衰减尾部，产生 ~2.7% 不可细化地板（$kH=6$ 时）。改用自由滑移镜面反射后误差降至 0.3%。这是本 benchmark 的核心工程发现。")
lines.append("2. **库缺口**：`tensorlbm.boundaries` 目前只有 bounce_back（无滑移），无自由滑移入口。本案例内联 SPECULAR 反射（3 行），未改库。")
lines.append("3. **Zou/He 移动盖**：在振荡驱动下表现良好，板近旁行误差 ≤0.1%@H=50，随网格收敛。")
lines.append("4. **质量漂移**：Zou/He 盖在边界节点上调节密度，非严格守恒。$H=200$ 时 drift ~0.35%，对速度相量无影响。")
lines.append("5. **SDAA 编译**：`teco_inductor` 后端不完整，benchmark 自动 fallback 到 eager 模式，数值结果与 CUDA 一致。")
lines.append("")
lines.append("## ⑥ 误差判定")
lines.append("")
lines.append("| 判据 | 门限 | 最大值 | 是否 <3% |")
lines.append("|---|---|---|---|")
lines.append("| 幅值相对误差 | ≤3% | 0.310% | ✓ 是 |")
lines.append("| 相位相对误差 | ≤3% | 0.147% | ✓ 是 |")
lines.append("| 复振幅相对误差 | ≤3% | 0.425% | ✓ 是 |")
lines.append("| 网格收敛（kH=6） | 单调下降 | 0.303→0.109→0.053% | ✓ 是 |")
lines.append("| 网格收敛（kH=8） | 单调下降 | 0.425→0.108→0.020% | ✓ 是 |")
lines.append("")
lines.append("**结论**：全部 6 配置误差均 <3%，两条 $kH$ 线三档网格严格单调收敛。**判定：PASS（VERIFIED）**。")

markdown = '\n'.join(lines)

report = {
    "title": "Stokes Second Problem (Oscillating Plate) — D2Q9 LBM Verification",
    "summary": "Stokes 第二问题（振荡平板）D2Q9 LBM 验证：6/6 配置全部通过 3% 门限，两条 kH 线三档网格单调收敛。最大复振幅误差 0.425%（kH=8, H=50 粗网格），远低于门限。远场必须用自由滑移镜面反射，无滑移远壁产生 ~2.7% 不可细化地板。",
    "scenario_key": "stokes_second_problem",
    "mission_id": 302,
    "tags": ["verified", "analytic", "D2Q9", "BGK", "oscillating_plate", "stokes", "unsteady"],
    "markdown": markdown,
    "figures": [
        {
            "name": "amplitude_phase_profile",
            "caption": "H=100, kH=6: 幅值剖面（左，半对数）与相位滞后剖面（右）。LBM 与解析解精确吻合。",
            "png_base64": profile_b64
        },
        {
            "name": "grid_convergence",
            "caption": "网格收敛：两条 kH 线在三档网格上复振幅最大误差均单调下降，全部远低于 3% 门限。",
            "png_base64": conv_b64
        }
    ]
}

with open('/root/tp-agent/work/m302/artifacts/m302/report.json', 'w') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("Report JSON written to artifacts/m302/report.json")
print(f"Markdown length: {len(markdown)} chars")
print(f"Figures: {len(report['figures'])}")
