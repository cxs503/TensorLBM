# DE-X-FORCE2-MOMENTUM_VOL Draft

## 已读
- `drop_entry/BB/NEXT/TRACK_FORCE_W1.md`: ARM MOMENTUM_VOL prescription
- `TensorLBM/src/tensorlbm/airfoil_benchmark.py`: Momentum-exchange method (Ladd 1994)
- `TensorLBM/src/tensorlbm/boundary/force_corrected_nebb.py`: Force correction for NEBB
- `TensorLBM/src/tensorlbm/boundary/solver2d.py`: compute_drag_force (wall shear stress)

## 已得结论
- 当前力采样基于动量交换法（Ladd 1994），在边界节点上求和。
- 需要修改为基于控制体动量通量积分的体积法（Chen 1998）。
- 体积法通过包围体积分计算力，对边界位置不敏感。

## 下一步
1. 实现 `compute_force_volume` 函数，基于动量通量积分计算力。
2. 修改内核力采样，将点采样替换为包围体积分。
3. 运行测试，验证峰误差收敛。
