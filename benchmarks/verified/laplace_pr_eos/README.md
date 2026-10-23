# M4: PR-EOS 高密度比液滴 Laplace 基准（σ 全自测）

**状态：✅ 已验证（主档 PASS；σ 完全由模拟自身压差测量提取，零外部参考输入）**

## 物理问题

Peng-Robinson EOS 伪势单组分（SCMP）模型在 T_r = 0.55（等面积 Maxwell
共存比 ρ_l/ρ_v = 123.2）下演化周期域圆形液滴至稳态，测量内外机械压差
ΔP = ⟨p_mech⟩_core − ⟨p_mech⟩_far（p_mech = ρcs² − (G/2)·cs²·ψ²），对每档
4 个半径作 ΔP vs 1/R_eff 线性拟合（LSQ、自由截距）提取表面张力 σ；
Laplace 线性成立与 σ 的网格收敛即为验证目标，不对任何文献 σ 值。

## 判定标准（prereg §5(b) 冻结）

| 条款 | 判据 | 阈值/实测 |
|------|------|-----------|
| clause_i | 逐 L 档 LSQ 残差 RMS ≤ 3% · max(ΔP) | 三档全过（实测 ≤ 0.100%） |
| clause_ii | \|σ(L_hi) − σ(L_lo)\| 跨三档非增 | 过（6.27e-3 → 2.93e-3） |
| 稳定性 | 有限、非负密度 | 24/24 stable |
| 质量守恒 | 质量漂移率 ≤ 5e-8/步 | 24/24 过（max 7.83e-9） |
| ψ 定义域 | ρ_max < ρ_cross（clamp 不触发） | 0/24 触发 |
| sim ratio（披露列） | 稳态 ρ_l/ρ_v ≥ 100 | 399.9–556.7 |

主档 = **R/L 保持**（R 随 L 缩放 = 界面网格加密：R = 12/16/20/24 @L=128 →
18/24/30/36 @192 → 24/32/40/48 @256）。冻结文本存在歧义（“每档同 R 阶梯”），
双读在出数前声明、双跑：副档（固定绝对 R 阶梯）为域尺寸敏感性披露，不参与判决。

## 结果（2026-10-01 实测，result.json v1.1）

### 主档（判决档）

| L | σ | 残差 RMS / max(ΔP) | 至下一档 \|Δσ\| |
|---|---|---|---|
| 128 | 0.320226 | 0.0555% | 6.27e-3 |
| 192 | 0.326492 | 0.0999% | 2.93e-3 |
| 256 | 0.329422 | 0.0446% | — |

**clause_i ✅ / clause_ii ✅ → PASS**（门 3%，残差富余 30 倍以上）。
L=128 主/副档同配置 4 对重跑轨迹**逐位相同**（确定性复核）。

### 副档（披露，不判决）

固定 R = 12/16/20/24，L = 128/192/256：σ = 0.320226 / 0.321775 / 0.332676，
|Δσ| = 1.55e-3 → 1.09e-2 **递增 → clause_ii 不满足**（clause_i 仍全过：
0.0555 / 0.0562 / 0.2376%）。机制注记：固定液滴增大域时 σ 随 L 单调升高，
说明该参数点 σ 对域尺寸仍敏感——界面加密收敛（主档）与域尺寸收敛是两个
独立方向，副档即域尺寸方向的如实披露。

## 配置

- D2Q9 SCMP，PR EOS（YS2006 参数 a = 2/49、b = 2/21、R = 1），
  T = 0.0401055（T_r = 0.55，T_c = 0.0729190）
- 修正符号伪势 ψ² = 2(ρcs² − p_EOS)/cs²（clamp ≥ 0），**G = +1**
  （本库后向 gather 约定下 G > 0 为吸引）
- forcing = **edm**（精确差分），τ = 1，float32，周期边界
- 初值 tanh 液滴（界面宽 4），共存密度取等面积 Maxwell 锁值
  ρ_l = 7.99784 / ρ_v = 0.06491（锁值与 ρ_cross = 9.70818 嵌入 run.py）
- 稳态判据：ρ_l/ρ_v 相对漂移 ≤ 1e-5（min 4000 / max 40000 步、采样 1000），
  测量取末 5 采样点平均；R_eff = 等面积半径 √(Σ(ρ−ρ_v)/(π(ρ_l−ρ_v)))

## 依赖

本目录与本 PR 的库修正同源：`tensorlbm.multiphase` 新增修正符号伪势
`psi_eos_carnahan_starling` / `psi_eos_peng_robinson` / `make_psi_eos`
（YS2006 逐字参数），并为 `collide_sc_single_component` 增加 opt-in
`forcing` 参数（默认 `"sc"` 与补丁前实现**逐位相同**；`"edm"` 精确差分、
`"guo"` 完整 Guo-2002 半移位，见函数 docstring）。既有符号的旧
`psi_carnahan_starling` / `psi_peng_robinson` 保持不变（向后兼容）。

## 复现

```bash
cd benchmarks/verified/laplace_pr_eos
python run.py --device cpu --out /tmp/laplace_pr_eos   # GPU: --device cuda:0
```

机器档案：`result.json`（v1.1：24 run 全轨迹 + 判决块 + 确定性记录）。
过程档案（Maxwell 锁、prereg、probe 链）在 5090 服务器
`/nfs/wangxi/runs/bm_mp_20261001/m4_pseudopotential/`。

## 参考

- Yuan & Schaefer (2006) Phys. Fluids 18 042101（EOS 参数表）。
- Kupershtokh (2004)；Li et al. (2012)（EDM forcing）。
- Young (1805)；Laplace (1806)：ΔP = σ/R（2D 圆柱液滴）。
