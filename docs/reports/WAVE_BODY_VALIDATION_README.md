# 规则波 RAO 解析基准

## 范围

本目录的 `tensorlbm.wave_body_rao` 提供垂直圆柱（spar）在规则 Airy 波中的低阶、线性解析/半解析参考：

- 深水和有限水深色散关系；
- heave 与 pitch 的附加质量、辐射阻尼、Froude–Krylov 激励近似；
- 对应的幅值 RAO。

`tests/test_wave_body_rao_validation.py` 仅检验静态极限、共振峰、高频衰减和色散关系残差。

## 证据边界

这不是 LBM 端到端测试，也没有连接 `free_surface_lbm`、船体几何、波浪边界或任何 solver。附加质量、阻尼和激励均为低频/长波近似，未包含 BEM 水动力、绕射、粘性或非线性自由面效应。通过测试只证明解析实现及其数学不变量自洽，不能作为 CFD/LBM 精度或实验物理验证结论。

## 运行

```bash
python -m pytest -q tests/test_wave_body_rao_validation.py
```

参考：Newman (1977), *Marine Hydrodynamics*；Faltinsen (1990), *Sea Loads on Ships and Offshore Structures*。
