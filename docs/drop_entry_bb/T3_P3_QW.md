# T3-P3 存活修复处方（Qwen 族）

## 1. 假设
**因果链**：默认 span=4.0D 域宽不足 → 触水后径向自由面波在 ~5-8 ms 触及侧壁 → 虚假反射波与入射波叠加 → 局部密度/马赫数越限 → health_limit/mach_limit 提前终止（DE_A3_DIAG 表1：8 cells/D 中仅 case3_wide8(span=8.0D) 存活 24 ms，其余 7 档均 <20 ms）。

## 2. 处方
**改什么**：不修改源码，仅以 `span=8.0D` 重跑 8 cells/D 配置（复用 case3_wide8 参数）。
**怎么改**：
```bash
cd /root/tp-agent/work/m26272/benchmarks/water_entry/single_drop
python run.py --cells 8 --span 8.0 --gap 0.4 --speed 0.04 --out runs/t3_p3_span8.json
```
**为什么能活过 10.4 ms**：case3_wide8 已实证 span=8.0D 可存活 24.005 ms（DE_A3_DIAG §2），覆盖峰值时刻 10.394 ms。本处方复现该配置以验证可重复性，并作为后续网格收敛的基线。

## 3. 预测
- **postcontact_healthy_ms**：≥ 22.0 ms（区间 22.0–24.0 ms）
- **可证伪判据**：若存活 < 15.0 ms 或终止原因为 health_limit/mach_limit/step_limit，则假设不成立。
- **次级预测**：impact_speed_relative_error < 2.0%（case3_wide8 为 0.68%）。

## 4. 成本
**最小验证命令**（~15 min，单节点 CPU）：
```bash
cd /root/tp-agent/work/m26272/benchmarks/water_entry/single_drop
python run.py --cells 8 --span 8.0 --gap 0.4 --speed 0.04 --out runs/t3_p3_span8.json
```
**收敛验证命令**（需 2 档网格，严格递减误差）：
```bash
python run.py --cells 12 --span 8.0 --gap 0.4 --speed 0.04 --out runs/t3_p3_span8_12.json
python score.py --runs runs/t3_p3_span8*.json --output score_t3_p3.json
```

## 5. 风险
- **风险1**：若失效根因非域宽（如界面厚度/速度初始化），span=8.0D 无改善 → 需回退至 DE_A4 LESSONS §1 的"重力双重计力"或"壁面反射"路线。
- **风险2**：span=8.0D 增加计算量约 2×，12 cells/D 收敛验证可能超时 → 设 timeout=3600s。
- **风险3**：若 case3_wide8 的存活是随机波动（非确定性），重复运行可能失败 → 需 ≥2 次独立运行确认。
