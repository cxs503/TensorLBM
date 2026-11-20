# DE-T3-P3R 处方卡 · 议事厅队 Qwen 族

## 1. 假设
**因果链**: 多数 run 在触水后 1–15 ms 内触发 `health_limit`，根因是 **gap=0.4D 过大 + lattice speed=0.04 过低 + 缺少 capacity/simple_exchange** 三重叠加：大间隙使液滴自由落体加速过度（撞击速度偏离目标），低晶格速度导致时间分辨率不足（每个物理步长过大），缺少质量再分配使界面单元在液滴入水瞬间填充率剧烈振荡（fill > 1+κ 或 < -κ），触发 rho_min≤0 或 volume_drift>1e-4 健康检查。

## 2. 处方
**改什么**: 不改代码，改运行参数。README 已给出正确参数组合（`benchmarks/water_entry/single_drop/README.md` 复算段）：
```
--cells N --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53
```
**为什么有效**:
- `gap=0.18D`（非 0.4D）：间隙缩小 55%，自由落体加速量减少 55%，撞击速度更接近 1.4 m/s
- `speed=0.06`（非 0.04）：晶格速度提高 50%，时间分辨率提高 50%，每个物理步长从 ~0.024 ms 降至 ~0.016 ms
- `--capacity`：启用容量再分配，界面单元转换亏额由邻格共享，防止 fill 剧烈振荡
- `--simple-exchange`：使用基本对称界面质量通量，避免 Thuerey I/I 选择器在高速入水时的不稳定
- `--release-speed 1.53`：独立设定物理释放速度，与晶格速度解耦

## 3. 预测
- **postcontact_healthy_ms**: ≥ 24.0 ms（覆盖完整观测窗口，含峰值 10.394 ms 及回缩）
- **峰值深度误差**: |sim_peak − 1.8574| / 1.8574 ≤ 0.03（即 sim_peak ∈ [1.802, 1.913] D）
- **时序误差**: 全部 18 个实验时刻插值相对误差 ≤ 0.03
- **可证伪判据**: 若 postcontact_healthy_ms < 20 ms 或峰值误差 > 5%，则处方假设不成立

## 4. 成本
```bash
# 8 cells/D 档（快速验证）
cd /root/tp-agent/work/m26276
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m benchmarks.water_entry.single_drop.free_surface_probe \
  --cells 8 --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 \
  --steps 2000 --out outputs/single_drop/t3_cal8.json

# 12 cells/D 档（收敛验证）
python -m benchmarks.water_entry.single_drop.free_surface_probe \
  --cells 12 --gap .18 --speed .06 --capacity --simple-exchange --release-speed 1.53 \
  --steps 2000 --out outputs/single_drop/t3_cal12.json

# 判分
python -m benchmarks.water_entry.single_drop.score \
  outputs/single_drop/t3_cal8.json outputs/single_drop/t3_cal12.json \
  --output outputs/single_drop/t3_score.json
```
预计单机 CPU 运行时间：~8–15 分钟/档（8 cells/D 约 8 min，12 cells/D 约 15 min）

## 5. 风险
1. **geometric_volume_drift 超限**: 当前测试显示 step 50 时 geometric_volume_drift=0.00058 > 1e-4。若全程超限，score.py 的 `numeric_pass` 会 FAIL。缓解：尝试 `--capacity` 单独使用（不加 simple-exchange）或调整 release_speed。
2. **性能瓶颈**: 单机 CPU 运行 2000 步约需 10 分钟。若需多档网格扫描，需使用集群。
3. **domain 尺寸**: 默认 span=4D、pool_depth=3D 远小于实验槽宽 17.6D。边界反射可能影响峰值后回缩。但峰值时刻（10.394 ms）时气穴半径约 1.5D，仍在 4D 域内，影响有限。
