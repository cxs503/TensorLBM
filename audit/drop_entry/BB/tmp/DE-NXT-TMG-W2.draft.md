# DE-NXT-TMG-W2 Draft

## 背景
TMG线（时间推进与次序）聚焦于LBM求解器的碰撞-流-力时序结构。
当前CAC求解器（cac_lbm3d.py）采用标准序列：`Collide -> Stream -> Boundary`。
力项（表面张力+重力）在碰撞步中计算并应用。

## 问题
- 峰深误差22.7%（1.4352D vs 1.8574D）。
- 早期死亡（质量漂移、Ma>0.3）。
- 参数扫描（CAVSWEEP）证明参数微调无法修复系统性误差。

## 假设
1. **FORCE_HALF**: 当前力项应用方式（全步）引入数值耗散和相位误差。半步步长修正（Guo 2008）可减少界面动力学误差。
2. **SURF_SUB**: 界面区域表面张力力项刚性大，单步时间积分不稳定。界面亚循环（多碰撞/单流）可稳定界面。

## 臂设计
### ARM 1: FORCE_HALF
- 任务：修改 `collide` 方法，使用半步步长速度修正。
- 物理出处：Guo et al. (2008) "Discrete lattice effects on the forcing term in the lattice Boltzmann method".
- 最小改动：在 `collide` 中计算 `u* = u + 0.5 * F * dt`，用 `u*` 计算平衡分布，然后 `u^{n+1} = u* + 0.5 * F * dt`。

### ARM 2: SURF_SUB
- 任务：在界面区域（`0.1 < phi < 0.9`），执行2次碰撞/1次流。
- 物理出处：界面张力力项时间尺度远小于体流动。
- 最小改动：在 `step` 方法中，对界面细胞循环碰撞。

## 预登记
- FORCE_HALF: 预测峰深误差<15%，证伪>20%。
- SURF_SUB: 预测存活时间>15ms，证伪<10ms。
