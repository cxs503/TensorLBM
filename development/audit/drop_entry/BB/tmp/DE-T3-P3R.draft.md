# DE-T3-P3R 草稿 — 议事厅队处方

## 已读
- DE_A1_FACTS.md: Santini case-3 实验数据，峰值 1.8574D @ 10.394ms，3% 判据
- DE_A2_MAP.md: 代码地图，free_surface_probe.py/run.py/score.py 入口与参数
- DE_A3_DIAG.md: 13 个 run 诊断，case3_cal12(12cells/D) 和 case3_wide8(8cells/D) 存活 24ms 最长
- DE_A4_LESSONS.md: 溃坝/静滴经验，重力双重计力已修，3D dam break TERMINAL
- ROLES.md: 处方卡 schema（假设/处方/预测/成本/风险）
- free_surface_probe.py: probe() 函数，健康检查在 L123-128
- run.py: run() 函数，CACSim/D3Q19 求解器
- protocol.json: 工况参数与门禁阈值
- free_surface_lbm.py: 核心求解器，Körner 模型
- README.md: 复算命令与参数

## 已得结论
1. **首要失效机制**: 多数 run 在触水后 1-15ms 内触发 `health_limit`，根因是 gap=0.4D 过大 + speed=0.04 过低 + 缺少 capacity/simple_exchange
2. **正确参数**: gap=0.18D, speed=0.06, capacity=True, simple_exchange=True, release_speed=1.53 m/s
3. **验证**: 8 cells/D, 50 steps 运行正常，contact_step=45, depth=0.0894D @ 0.608ms，健康指标良好
4. **处方已提交**: /root/tp-agent/work/m26276/drop_entry/BB/teams/T3/P3_QW.md

## 下一步
- 用集群跑 8 cells/D 和 12 cells/D 完整模拟（~2000 steps）
- 判分验证 3% 精度
- 双写到工作树 docs/
