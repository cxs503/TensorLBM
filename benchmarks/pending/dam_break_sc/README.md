# 2D 溃坝 — Martin & Moyce (1952) 验证（SCMP / Shan–Chen 单组分伪势）

**状态：❌ 未达标（2026-09-29，修正参考表后重评估）** — 见 `REPORT_SCMP_vs_FS2D.md` 与
`result.json`。本目录保留真实数据与可复现材料，**未**入库 `benchmarks/verified/`。

## 决定性修正：M&M 参考表此前是错的

`run.py` 旧表 `{T=1→1.5, T=2→2.7}` 与仓库自身其它基准不一致（把 M&M 的 T=3→2.7
错安到 T=2）。**已修正为经典数字化表 `{T=1→1.1, T=2→1.8, T=3→2.7}`**（与
`benchmarks/bench_fs_2d.py` `REF_Z`、`dam_break_3d_mm` 一致）。

> 此修正把 SCMP 的差距从旧表下的 ~+10% 放大到 **+36~65%** —— 旧结论「只剩 ~+10% 本构缺口」
> 是**错误参考表**的假象。

## 物理问题

水柱 a×2a 位于左下角（x<a, y<2a），重力 -y，四周固壁（闭合盒），其余气相。域 6.4a×3.2a。
`X = x_toe/a`；`T_raw = t·√(g/a)`；`T_eff = t·√(g_eff/a)`（`g_eff = 0.9184 g`）。

## 本次评估（真实运行，CPU，a=40/80，g=2e-4，τ=1.0）

### 路线 A — SCMP（对正确基线 1.1/1.8/2.7）

| 配置 | a | X(T=1) T_eff | 误差 T_eff | 误差 T_raw |
|------|---|--------------|-----------|-----------|
| SC94 `psi_exp` 基线 | 40 | 1.750 | **+59.1%** | +54.0% |
| SC94 `psi_exp` 基线 | 80 | 1.702 | **+54.8%** | +50.2% |
| `wall_psi=0.4`（唯一有效杠杆） | 40 | 1.550 | **+40.9%** | +35.9% |
| `wall_psi=0.4` | 80 | 1.632 | +48.4% | +44.2% |
| `psi=sqrt(rho)` | 40 | **NaN @25** | — | — |

- 基线两档网格收敛：T=1 **2.8%**、T=2 **0.9%**（✓）、T=3 **9.9%**（✗）。
- 质量漂移 ≤2e-5，稳定。**但精度差一个数量级 → SCMP 死路。**

### 路线 B — 2D 自由面模块（`free_surface_lbm_2d`）

| 配置 | 稳定 | X(T=1) a=40 | 误差 | 质量漂移 |
|------|------|-------------|------|----------|
| ρ_gas=0.01 legacy | ❌ NaN~100 | — | — | — |
| ρ_gas=0.01（移植 3D wall-BC） | ❌ NaN@29 | — | — | — |
| ρ_gas=0.01 + ABB-clamp | ✅ | 3.175 | +189% | 100% |
| ρ_gas=1.0 legacy | ✅ | 1.767 | +60.6% | **41%（inv 232%）** |
| ρ_gas=1.0 + wall-BC | ✅ | 1.792 | +62.9% | 47% |

- **守恒实测 41% / 232%，并非任务背景所述 2-5%**；物理气相密度下**不稳定**。
- 移植的 3D wall-BC/TOGAS 修复**在 2D 无效**。

## 判定

- 精度 ≤3%（T=1）：**不满足**（2D 最佳 SCMP +36%，FS2D +61%，均 >3%）。
- 两档网格收敛 ≤3%：SCMP 基线 T=1/T=2 满足、T=3 不满足；FS2D 因守恒失败不适用。
- 综合：**未达标**；推荐路线与后续见 `REPORT_SCMP_vs_FS2D.md` §5。

## 运行方式（复现）

```
cd /root/TensorLBM_feat2
PYTHONPATH=src python benchmarks/pending/dam_break_sc/run.py \
    --a 40 --g 2e-4 --steps 1400 --sample-interval 25 --device cpu --compile-mode eager
PYTHONPATH=src python benchmarks/pending/dam_break_sc/run.py \
    --a 80 --g 2e-4 --steps 2200 --sample-interval 25 --device cpu --compile-mode eager
# 变体：--psi-wall 0.4 / --psi sqrt
# 2D 自由面探针：python runs/dam_break_sc_fix/probe_fs2d.py --a 40 --g 2e-4 --rho-gas 1.0
# 3D 自由面旁证：python runs/fs_aprime_20260929/probe.py --a 8 --g 2e-4 --steps 240 --rho_gas 1.0
```

产物：`case_a40.json`/`case_a80.json`、`hist_*.csv`（本目录）+
`runs/dam_break_sc_fix/`（全部新运行）。