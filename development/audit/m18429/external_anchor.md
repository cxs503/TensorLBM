# 外部锚点（external_anchor）—— m18366 锚点核验用

核验判据的可溯源外部依据（由 m18366 predeclared.json 声明的出处复核）：

1. **PR-EOS 用于伪势 LBM 的参数**
   Yuan & Schaefer (2006), *Equations of state in the lattice Boltzmann method for multiphase flows*,
   Phys. Fluids **18**, 042101. —— 对应 `src/tensorlbm/sc_eos_coexistence.py` 的 `pr` EOS 参数。
2. **Maxwell 等面积构造（共存对）**
   在比容空间求解 ∫_{v_l}^{v_v} [p(v) − p_sat] dv = 0；本文核验以库内
   `sc_eos_coexistence.maxwell_coexistence` 重算，面积残差 1.39e-13 / 1.05e-12 / 1.22e-11（目标比 10/100/1000），
   说明 ref 侧数值实现可用（见 tables.md T1）。
3. **大密度比 forcing**
   Li, Luo & Li (2012), Phys. Rev. E **86**, 016709 —— exact-difference method（EDM）forcing。
4. **任务文本声称的开源横向参照** `lssfau/lbmpy` 的 `pf-eos-maxwell`
   —— 库内无 vendored 源码/pin hash；`oss_solutions` 中 feature_slug ILIKE '%EOS-MAXWELL%' 或
   '%HIGH-DENSITY%' ⇒ **0 rows**（m18429 机核）。该开源横向参照 **不可溯源**，不构成本单证据。

**结论**：文献级外部锚点（1-3）真实存在且与实现/判据对应；任务③要求的"调研沉降到 oss_solutions"
未完成（0 行），开源横向参照不可溯源。