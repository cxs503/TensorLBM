# 精确部分单元上的均匀共速通量验证

本轮将 [精确移动矩形几何](swept-rectangle-geometry.md) 接入独立 `UniformSweptTransport`：保存广延 populations `Q = 液体面积 × 厚度 × f`，实际计算共享笛卡尔面通量和移动壁面相对通量，再更新 `Qnew = Qold - cartesian_flux - wall_flux`。

壁面采用 diffuse incoming equilibrium，并按 outgoing mass rate 归一化墙面密度使墙面质量通量为零。新出生单元的 Q 来自通量更新，没有单独初始化为壁速平衡；零体积单元仅清除浮点抵消残差，其质量和动量明确记账。体积/开口面按网格交叉事件精确积分，支持实际单元出生与消失。

## 已执行的验证

四组完整原场：静止、斜向共速、反向共速、时间步减半。移动案例确实跨格，斜向共速出生/消失 6/2 个单元、反向 5/3 个。四例完整重启逐位一致；10 项测试通过，包含错误 aperture 原子拒绝、非均匀输入拒绝、checkpoint 篡改拒绝。

独立 NumPy 从每步 Qold、共享面积分和壁面 facets 重建真实通量与 Qnew，未导入输运或几何模块计算参考值。还独立积分几何、核查 GCL、质量/动量、密度和速度：

| 最大指标 | 结果 |
|---|---:|
| Qnew 独立重构差 | 1.67e-16 kg |
| 动量账本残差 | 1.18e-14 N·s |
| 共速密度相对误差 | 6.62e-14 |
| 共速速度误差 | 4.76e-14 m/s |
| 压缩自由能绝对值 | <6.2e-15 J |

[study.json](assets/swept-uniform-transport/study.json) 绑定源码与完整 gzip 原场；[独立审计](assets/swept-uniform-transport/audit-report.json) 另绑定 auditor。重启是 producer 实际运行及测试证明，独立审计核查原场历史，未伪称重跑 class restore。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python examples/swept_uniform_transport/run.py
python examples/swept_uniform_transport/audit.py
```

## 资格范围

本验证器**仅支持均匀共速平衡状态**，由旧广延状态恢复均匀 trace，并拒绝非均匀输入；这些 trace 的全域恢复不是任意流场的重构算法。它实际验证几何与通量在共速/静止状态下的一致性，不能替代任意状态 FV-BGK 求解器。

未实现非均匀重构、BGK 碰撞、小单元稳定化或黏度验证，未认证一般 CFL 正性/能量稳定性、移动圆盘、周期穿界、刚体反馈或破冰。旧整数 mask 及 balanced 方案的跨格失败保持有效。本轮没有将其替换为已合格的流体后端。

下一步是有限邻域小单元合并、任意状态保守重构与真实碰撞，逐步验收正性、几何守恒、动量/能量和时间/空间细化。
