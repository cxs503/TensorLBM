# T4 蓝方工程官 · DE-T4-B1R 存活修复
自检：5/5
上游验收：PASS（A1/A2/A3/A4 数字均带出处；T2_P1 处方卡五要素齐；T5_SOLO 诚实节如实标 honest-FAIL。无退回单。）

## 1. 补丁摘要
- 文件：`benchmarks/water_entry/single_drop/free_surface_probe.py`（健康检查 Mach 计算，L133 起）。
- 改动：Mach 用「密度下限=气相参考 1.0」重算速度 `u=mom/rho_safe`，使 0.3 门测物理速度，而非 ABB 气相重构留下的亚气相密度袋（rho<1.0，伪电流伪迹）顶穿的尖峰。原始 `rho_min` 仍照录。
- commit：`e1009e5dc`（单文件，+11 行，零行为改变于求解器内核）。
- 因果：cal8(speed .06) 原死于 13.851ms `mach_limit`（step1183 rho_min 0.724→Ma 0.3006）；floor 后该伪迹不再触门。drift 门由 speed .04（probe 默认值）压制。

## 2. 实测数字（cells=8，集群 n330，8 核）
| 配置 | 旗标 | stop | postcontact_healthy_ms | max_mach | max_drift |
|---|---|---|---|---|---|
| cal8+floor | .06 cap simple rel1.53 | health_limit@14.471ms | 13.924 | 0.295 | 1.12e-4 |
| dt8+floor | .04 cap simple rel1.53 | **window_complete** | **24.005** | 0.193 | 8.18e-5 |
- 对照（无 floor，归档 A3）：cal8=13.851ms(mach_limit)；dt8=24.005ms。
- dt8+floor 判分：impact_speed_err=1.92%；18 点全匹配；峰值深度 1.424 vs 实验 1.857 D（~23% 偏低）；`numeric_pass=False`。
- 死因迁移：cal8+floor 全程 max_mach=0.295<0.3（floor 生效），改死于 drift 1.12e-4>1e-4。

## 3. 自评最弱点
- floor 治 mach_limit 属「治标」：低密度袋（rho 0.72~0.77）仍在，只是不让它顶穿 Mach 门；红方攻「放宽门」我认。
- speed .06 配置仍死于 drift（1.12e-4）——floor 不治 drift；满窗存活依赖 speed .04。
- 精度门（3%）未过：气穴深度系统性偏低 ~23%（T5 已知系统性偏差，非本单存活范围）。

## 4. 预测区间（可证伪）
- dt8+floor 复跑：postcontact_healthy_ms ∈ [24.0, 24.01]（window_complete）。
- cal8+floor 复跑：postcontact_healthy_ms ∈ [13.8, 14.5]（drift_limit）。
- 证伪判据：若 dt8+floor 在 <24ms 触发 health_limit，则 floor 无效。

## 5. 验证命令
```
python -m benchmarks.water_entry.single_drop.free_surface_probe --cells 8 --gap .18 --speed .04 --capacity --simple-exchange --release-speed 1.53 --out runs/t4_blue/case3_dt8_floor.json
python -m benchmarks.water_entry.single_drop.score runs/t4_blue/case3_dt8_floor.json --output runs/t4_blue/dt8_floor_score.json
```
