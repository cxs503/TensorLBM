# Wagner 入水砰击解析基准

## 范围

`tensorlbm.wagner_theory` 给出恒速入水问题的 Wagner 型解析/半解析参考函数：

- 二维楔形体湿半宽、压力分布、单位展长总力和无量纲系数；
- 球体早期入水湿半径、简化 added-mass 砰击力和无量纲化；
- 楔形喷射根与球体自由面/空腔的理想化几何函数。

`tests/test_water_entry_slamming.py` 仅验证这些公式的对称性、尺度律、湿区边界和无量纲一致性。

## 证据边界

该模块不运行 LBM，未调用自由表面求解器、球体入水算例或任何流体/刚体耦合。Wagner 理论适用早期、小穿透、势流型近似；喷射根压力奇性、粘性、可压缩性、流动分离、空腔闭合和真实峰值载荷均不在此闭包内。测试通过是解析参考实现的回归证据，不是 LBM 端到端、数值收敛或实验物理验证。

## 运行

```bash
python -m pytest -q tests/test_water_entry_slamming.py
```

参考：Wagner (1932)；Zhao & Faltinsen (1993)；Korobkin (1992)。
