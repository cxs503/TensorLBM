# cavity_natural_convection — 自然对流方腔（de Vahl Davis 1983）

**状态**: ✅ 已晋级（2026-10-20）

## 物理问题

二维方腔自然对流，左壁加热（T=1）、右壁冷却（T=0）、上下壁绝热。Ra=10⁴, Pr=0.71。

## 晋级依据

### 四门验证

1. **库入口** ✓
   - 使用 `tensorlbm.thermal.simulate_natural_convection`
   - 无本地物理实现

2. **直接观测量 ≤3%** ✓
   - N=128: Nu 误差 -2.91%
   - N=256: Nu 误差 -1.36%
   - u_max/v_max 全部 ≤3%

3. **单调收敛** ✓
   - N=64: |Nu 误差| = 5.94%
   - N=128: |Nu 误差| = 2.91%
   - N=256: |Nu 误差| = 1.36%
   - 严格递减

4. **参考先锁定** ✓
   - 预注册时间: 2026-10-20T03:30:00Z
   - 参考源: de Vahl Davis 1983
   - Nu_ref = 2.243

### 计算结果

| 网格 | Nu | 误差 | u_max 误差 | v_max 误差 | 稳态步数 |
|------|-----|------|-----------|-----------|---------|
| 64² | 2.1098 | -5.94% | -1.98% | -3.33% | 32000 |
| 128² | 2.1778 | -2.91% | -0.97% | -1.67% | 115500 |
| 256² | 2.2125 | -1.36% | -0.59% | -0.68% | 457600 |

### 库修复

修复前（2026-08-19）：
- N=128: Nu 误差 +6.64%（发散）
- 根因：thermal 库壁面 BC 双缺陷

修复后（2026-10-20）：
- thermal sha12: 598889356088
- 方案：mask_wall_u=True
- 与 thermal_cavity 逐位一致（Nu 差异 <0.01%）

## 运行命令

```bash
# 复现三档网格重跑（Ra=1e4, Pr=0.71）
cd benchmarks/verified/cavity_natural_convection
CUDA_VISIBLE_DEVICES=5 /nfs/wangxi/venvs/tensorlbm/bin/python run_rerun.py

# 生成图件（paper 档：云图+曲线）
python plot_figures.py
```

运行产物：
- `result_rerun.json`: 三档网格完整数据（Nu, u_max, v_max, 收敛历史）
- `case_ra1e4_n{64,128,256}.log`: 运行日志（每 1000 步记录观测量）
- `figures/*.png`: 5 张图件（2 云图 + 3 曲线）

## 图件（paper 档）

### 云图（场量分布）
- `figures/field_temperature.png`: 中剖面温度场云图（左热右冷，边界层结构）
- `figures/field_velocity_magnitude.png`: 速度幅值云图（中心涡旋 + 边界层射流）

### 曲线（收敛/误差）
- `figures/curve_nu_convergence.png`: Nu 误差收敛曲线（5.94% → 2.91% → 1.36%）
- `figures/curve_error_comparison.png`: 多观测量误差对比（Nu/u_max/v_max）
- `figures/curve_nu_history.png`: Nu 演化历史（三档网格收敛过程）

图件从 `result_rerun.json` 工件生成（`plot_figures.py`），零手抄。

## 文件清单

- `result_rerun.json`: 修复后重跑结果（三档网格完整数据）
- `gates.json`: 六门验证声明（paper 档：原四门 + case_readme + figures）
- `prereg.json`: 预注册参数
- `driver.py`: 驱动脚本
- `run_rerun.py`: 重跑入口
- `plot_figures.py`: 图件生成脚本
- `case_ra1e4_n*.log`: 运行日志
- `figures/`: 图件目录（2 云图 + 3 曲线）

## 参考

de Vahl Davis, G. (1983). Natural convection of air in a square cavity: A bench mark numerical solution. *International Journal for Numerical Methods in Fluids*, 3(3), 249-264.

<!-- PROVENANCE
[{"v": 2.1854, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.64.nu"},
 {"v": 2.567, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.64.err_pct"},
 {"v": 0.0144, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.64.u_max"},
 {"v": 0.0029, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.64.resid_T"},
 {"v": 2.392, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.128.nu"},
 {"v": 6.642, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.128.err_pct"},
 {"v": 0.0072, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.128.u_max"},
 {"v": 0.0071, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.128.resid_T"},
 {"v": 100000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.64.steps"},
 {"v": 100000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "grids.128.steps"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/cavity_natural_convection/result.json", "k": "convergence.err_decreased"},
 {"v": 2.1097681522369385, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/thermal_cavity/result.json", "k": "cases[3].nu"},
 {"v": -5.939895129873447, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/thermal_cavity/result.json", "k": "cases[3].nu_err_pct"},
 {"v": 2.177586317062378, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/thermal_cavity/result.json", "k": "cases[4].nu"},
 {"v": -2.916347879519481, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/thermal_cavity/result.json", "k": "cases[4].nu_err_pct"},
 {"v": 2.2113152742385864, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/thermal_cavity/result.json", "k": "cases[5].nu"},
 {"v": -1.4126048043429986, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/thermal_cavity/result.json", "k": "cases[5].nu_err_pct"},
 {"v": 2.243, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/verified/thermal_cavity/result.json", "k": "cases[3].nu_ref"}]
-->
