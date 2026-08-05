# 不规则波、RAO 与 6DOF 最小解析闭包

## 范围

本闭包提供彼此独立、由调用者提供水动力输入的线性工具：

- `tensorlbm.rigid_body_6dof`：Cummins 型六自由度方程、辐射阻尼到延迟函数的离散变换，以及由给定谱和复数传递量生成的随机相位激励；
- `tensorlbm.rao_analysis`：频域动态刚度 RAO、时序 FFT RAO、谱响应矩和基础比较指标；
- `tests/test_irregular_wave_6dof.py`：使用本地合成 JONSWAP 谱、合成水动力系数及固定随机种子，检查形状、有限性、可复现性、谱响应和有界时域积分。

## 最小直接依赖

实现直接依赖 Python 标准库、PyTorch 和 NumPy；项目测试环境已提供 NumPy。测试内置了 JONSWAP 谱表达式，避免依赖 `wave_bc`，因此不需要连接波浪边界、`free_surface_lbm`、船体或求解器。

## 证据边界

`added_mass`、`damping`、`stiffness`、质量矩阵和激励均是调用者提供或测试中合成的数据。本闭包不从 LBM 流场提取载荷，未耦合任何 hull 或自由面模型，也未使用实验数据。测试通过仅表明线性解析/数值后处理工具在合成输入下自洽；不是 LBM 端到端验证、真实海况预报、数值收敛或物理验收。

此外，`generate_jonswap_excitation` 的 `rao_complex` 是调用者定义的线性复系数；其物理含义应在上游明确（例如力激励传递量），不能自动视作已校准的船体 RAO。

## 运行

```bash
python -m pytest -q tests/test_irregular_wave_6dof.py
```

参考：Cummins (1962)；Faltinsen (1990)。
