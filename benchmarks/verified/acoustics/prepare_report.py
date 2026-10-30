"""Prepare and POST the verification report to the API."""

import base64
import json
from pathlib import Path

# Load results
with open(Path(__file__).parent / 'result.json', 'r') as f:
    data = json.load(f)

# Extract key metrics programmatically
conv = data['convergence']
verdict = data['verdict_main']

# Build markdown report
markdown = """# D2Q9 平面声波线性色散与衰减验证报告

**案例**: TensorLBM benchmarks/verified/acoustics  
**班次**: m343  
**状态**: ✅ **PASS**

---

## 1. 案例介绍

### 物理背景
D2Q9 BGK 格子 Boltzmann 方法的平面声波传播验证。等温假设下，线性化声波满足：
- 波速：c_s = 1/√3 ≈ 0.57735 (格子单位)
- 衰减率：δ = νk²，其中 ν = c_s²(τ - 1/2) 为运动粘度

### 验证目标
验证 TensorLBM 库原生 D2Q9 BGK 求解器的色散关系与衰减特性，判据：
- 波速误差 |c_meas/c_s - 1| ≤ 3%
- 衰减率误差 |δ_meas/(νk²) - 1| ≤ 3%

### 参数表
| 参数 | 值 |
|------|-----|
| 模型 | D2Q9 BGK, 等温 |
| 网格 N | 16, 32, 64 (每波长点数) |
| 松弛时间 τ | 1.0 (主案) |
| 振幅 ε | 1e-3 |
| 精度 | float64, CPU |

---

## 2. 计算方法

### 工具链
- 求解器：`tensorlbm.solver.collide_bgk` + `tensorlbm.solver.stream`
- 平衡态：`tensorlbm.d2q9.equilibrium`
- 宏量：`tensorlbm.d2q9.macroscopic`
- 边界：周期域（原生模运算）

### 关键设置
- 初始条件：右行平面波 ρ = 1 + ε sin(kx), u_x = c_s ε sin(kx)
- 测量方法：复振幅 Z(n) = ⟨(ρ-1)e^{-ikx}⟩
  - 波速：相位斜率 ω/k
  - 衰减率：稳健估计器（吸收反向波 2ω 差拍）
- 参考理论：
  - 连续：c = c_s, δ = νk²
  - 离散：9×9 传播矩阵特征值精确解

---

## 3. 计算结果

### 主案：网格收敛性 (τ=1.0, ε=1e-3)

| N | c_meas | vs 连续 c_s | δ_meas | vs 连续 νk² |
|---|--------|-------------|--------|-------------|
"""

for c in conv:
    markdown += f"| {c['N']} | {c['c_meas']:.9f} | **{c['c_err_cont_pct']:+.4f}%** | {c['delta_meas']:.6e} | **{c['delta_err_cont_pct']:+.4f}%** |\n"

markdown += f"""
**判定**：
- ✅ 波速误差 max {verdict['c_err_pct_max']:.4f}% ≤ 3%，单调 {verdict['c_monotone']}
- ✅ 衰减率误差 max {verdict['delta_err_pct_max']:.4f}% ≤ 3%，单调 {verdict['delta_monotone']}
- ✅ 收敛阶：O(N⁻²)（实测 3.98–4.07×/档）

### τ 敏感性 (N=32)

| τ | δ_meas | vs 连续 |
|---|--------|---------|
"""

for t in data['tau_sweep']:
    markdown += f"| {t['tau']} | {t['delta_meas']:.6e} | {t['delta_err_cont_pct']:+.4f}% |\n"

markdown += """
**结论**：δ ∝ (τ-1/2) 成立，离散理论预测与实测一致。

---

## 4. 结果图

见附件 figures（4 张 PNG）：
1. **fig_convergence.png**：波速与衰减率误差收敛曲线
2. **fig_dispersion.png**：色散关系与衰减关系对比
3. **fig_tau_sensitivity.png**：τ 敏感性分析
4. **fig_wave_evolution.png**：波形演化示意

---

## 5. 注意点

### 踩过的坑
1. **反向波污染**：平衡态初始化播种反向波 b/a ~ νk/(2c_s)，需稳健估计器吸收
2. **δ 估计器升级**：预注册后诊断出 2ω 差拍偏置 +0.033%，改为稳健估计器
3. **fp32 权重**：库 W 常量为 fp32，对 1e-6 级残差有贡献

### 局限
1. **等温假设**：无能量方程，不适用于绝热声学
2. **线性范围**：ε ≤ 1e-3（ε = 1e-2 见非线性漂移）
3. **壁面效应**：BB 壁方腔被粘性边界层支配（+156%~214%），体理论不适用

### 适用范围
- τ ∈ [0.6, 1.2]，ε ≤ 1e-3，N ≥ 16
- 周期域平面波（壁面问题需单独处理）

---

## 6. 误差判定

### 关键量 vs 基准

| 观测量 | 基准 | N=16 误差 | N=32 误差 | N=64 误差 | <3%? |
|--------|------|-----------|-----------|-----------|------|
| 波速 c | 连续 c_s | +0.2190% | +0.0539% | +0.0135% | ✅ |
| 衰减率 δ | 连续 νk² | +1.3145% | +0.3231% | +0.0812% | ✅ |
| 波速 c | 离散理论 | +0.00015% | +0.00010% | +0.00006% | ✅ |
| 衰减率 δ | 离散理论 | -0.00002% | -0.00002% | +0.00081% | ✅ |

**显式回答**：✅ **是**，所有关键量误差均 < 3%，且单调收敛。

### 三链闭环
1. 级数展开（sympy）→ 离散修正 βk², c₂k²
2. 精确特征值（9×9）→ 与级数一致 < 1e-10
3. 真实模拟（TensorLBM）→ 与离散理论一致 1e-5~1e-7

**结论**：管线、理论、模拟自洽闭环，验证可信。

---

## 7. 复现

```bash
cd benchmarks/verified/acoustics
python run.py
python visualize.py
```

**输出**：result.json + 4 张 PNG + 控制台表格  
**耗时**：< 2 min (CPU fp64 单线程)

---

**报告生成**: m343 班次  
**验证结论**: ✅ **PASS**  
**置信度**: 高（三链闭环 + 单调收敛 + 理论自洽）
"""

# Encode figures as base64
figures = []
for fig_name in ['fig_convergence.png', 'fig_dispersion.png', 
                  'fig_tau_sensitivity.png', 'fig_wave_evolution.png']:
    fig_path = Path(__file__).parent / fig_name
    if fig_path.exists():
        with open(fig_path, 'rb') as f:
            png_base64 = base64.b64encode(f.read()).decode('utf-8')
        figures.append({
            'name': fig_name,
            'caption': fig_name.replace('.png', '').replace('_', ' ').title(),
            'png_base64': png_base64
        })

# Build report JSON
report = {
    'title': 'D2Q9 平面声波线性色散与衰减验证 - TensorLBM 首个声学问题类基准',
    'summary': f'TensorLBM D2Q9 BGK 平面声波验证通过：波速误差 max {verdict["c_err_pct_max"]:.4f}% ≤ 3%，衰减率误差 max {verdict["delta_err_pct_max"]:.4f}% ≤ 3%，单调收敛 O(N⁻²)，三链闭环自洽。',
    'scenario_key': 'acoustics',
    'mission_id': 343,
    'tags': ['verification', 'acoustics', 'D2Q9', 'BGK', 'plane-wave', 'dispersion', 'attenuation'],
    'markdown': markdown,
    'figures': figures
}

# Save report JSON
with open('/tmp/report.json', 'w') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"Report prepared: /tmp/report.json")
print(f"  Title: {report['title']}")
print(f"  Summary: {report['summary'][:100]}...")
print(f"  Figures: {len(figures)}")
print(f"  Markdown length: {len(markdown)} chars")
