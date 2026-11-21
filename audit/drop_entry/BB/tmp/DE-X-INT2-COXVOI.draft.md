# DE-X-INT2-COXVOI 草稿
## 已读
- TensorLBM/src/tensorlbm/boundary/inamuro_wetnode.py (Inamuro 湿节点边界)
- TensorLBM/src/tensorlbm/wetting/__init__.py (现有润湿模块)
- TensorLBM/tests/test_wetting_three_anchor.py (润湿测试)
- TensorLBM/src/tensorlbm/multiphase/__init__.py (Shan-Chen 多相流)

## 已得结论
- 峰深欠深问题已排除网格/参数/u_limit，确认为物理内核问题
- Cox-Voinov 定律: θ_d^3 = θ_e^3 + 9·Ca·ln(L/λ)
- 需要实现动态接触角模型，替代固定接触角
- 验证方案: 毛细管上升高度随时间演化 vs 理论解

## 已完成
1. 在 wetting/__init__.py 添加 Cox-Voinov 动态接触角计算 ✓
2. 修改 wall_psi_from_contact_angle 增加角度钳位 ✓
3. 编写 test_cox_voinov_validation.py 验证脚本 ✓
4. 公式验证: 误差 < 1e-14% ✓
5. Ca 标度单调性验证: 通过 ✓
6. 交付文档双写 ✓

## 下一步
- 提交集群验证 (cjN)
- 完整 LBM 模拟需要进一步调参 (G, tau, wall_psi 范围)
