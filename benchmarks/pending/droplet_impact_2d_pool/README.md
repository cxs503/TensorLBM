# 二维液滴入池 · 待验证

本算例复核自由能 D2Q9 液滴在重力驱动下与同种液池形成液桥的时刻。正式入水动力学状态为 `pending_external_validation`。

- **运行与读图**。网格、边界、重力、无驱动对照、参考工况和晋级判据见[案例说明书](../../../docs/benchmarks/droplet_impact.md)。
- **本次判定**。240×160 网格运行 4,500 步。驱动组在 (3,220, 3,230] 步连通，对照组始终分离。实验气穴深度误差待测。[复核记录](result.json)
- **查原始数据**。档案包含七组轨迹、三半径静滴压力、云图和动画。[原始记录](../../../docs/audits/entry_20261118/README.md)

## PROVENANCE

| 量 | 文件 | 键 |
|---|---|---|
| 连通时间窗 | result.json | paired_control.driven_bridge |
| 对照末态间隙 | result.json | quiescent.trajectory.final_centerline_gap |
| 静滴压差拟合 | result.json | static_prerequisite.relative_slope_error |
| 动态实验误差与阻塞项 | result.json | dynamic_reference |
