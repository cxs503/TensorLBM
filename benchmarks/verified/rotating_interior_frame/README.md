# 旋转参考系内部流场验证（D2Q9 环隙 + D3Q19 槽道）

**判定: 全部达标** — 四条机器判据见下表；数值由 `make_report.py` 从 `records/rotating_interior_full.json` 注入，报告全文 `docs/ROTATING_INTERIOR_P158W1_N02.md`。

| 判据 | 实测 | 阈值 | 判定 |
|---|---|---|---|
| C1_profile | 0.007 | <= 0.030 | PASS |
| C2_secondary | 4.112e-04 | <= 0.050 | PASS |
| C3_ekman | 0.007 | <= 0.050 | PASS |
| C4_angmom | 2.091e-16 | <= 1.000e-08 | PASS |

## 复现

```bash
OMP_NUM_THREADS=1 python benchmarks/verified/rotating_interior_frame/run.py
OMP_NUM_THREADS=1 python -m pytest tests/test_rotating_interior.py -q
python benchmarks/verified/rotating_interior_frame/make_report.py
```

## 与已有点的区别（纪律：禁止重复 verified/ 已验证案例）

`rotating_ekman`（平板 Ekman 螺旋）、`rotating_taylor_column`、`rotating_source_sink`、
`taylor_couette`（惯性系环隙）均为已验证面；本目录新增的是**旋转系内部流场**：
闭合弯曲几何内的旋转系压力锚（含 Coriolis+离心，判别性证据）与
绕壁法向轴旋转的平面槽道 Ekman 二次流。

## 踩坑记录（详见报告 ⑤）

1. 离心力是纯梯度：只测速度看不到它的实现错误 → 必须有压力锚（丢框架项偏差 78.6 %）。
2. 移动壁 Ladd 修正项在曲壁上不严格守恒质量（1e-5/50 步量级，测试阈值 1e-3 并注明原因）。
3. 楼梯环隙须用 R_eff 反演（房法），否则量的是楼梯而不是求解器。
4. torch 128 核默认线程数对 1e3 格小算子有 ~25 ms/步开销 → 单线程 7 ms/步，numpy 0.3 ms/步。
5. 判据 4 的口径：无散测试场给机器精度；真实流场受 lattice BGK 不精确守恒角动量限制，两个数都列。
