# 旋转参考系内部流场验证面（二）：Guo/EDM 力项、Ekman 厚度标度律、旋转方管二次流

**判定：全部机器判据达标（VERIFIED）** — 完整数值见
`records/rotating_duct_frame_w2.json`，叙述见 `docs/ROTATING_DUCT_P158W2_N01.md`。

本案例是战役节点 `p158w2-n01-lbm-rotating-frame`，覆盖四个**此前未验证**的面：

| 面 | 判据 | 实测 |
|---|---|---|
| S1 力项方案交叉验证（Guo 2002 vs EDM） | 两方案稳态场差 ≤ 2% | 机器精度级（两者一阶恒等，差 O(a²)） |
| S2 Ekman 厚度标度律 $\delta_E\sim\Omega^{-1/2}$ | 双对数斜率 $\lvert p+1/2\rvert\le0.05$，逐点前因子 ≤3% | $p=-0.4962$，前因子 ≤0.83% |
| S3 反向旋转 Taylor–Couette（参考系旋转） | 剖面 ≤3%、力矩恒等式 ≤3%、无虚假径向流 | 1.97% / 1.96% / 3.8e-4 |
| S4 旋转方管 Coriolis 二次流（Dean 对偶） | 与线性 FD 参考解 ≤10%、涡对拓扑、$\Omega\to-\Omega$ 反号 | 4.58% / (1,0) 涡胞 / corr = −1.0000 |

## 复现

```bash
# 全量（单核约 10 分钟）
PYTHONPATH=src python benchmarks/verified/rotating_duct_frame_w2/run.py \
    benchmarks/verified/rotating_duct_frame_w2/records/rotating_duct_frame_w2.json
# 冒烟（缩短参数）
PYTHONPATH=src python benchmarks/verified/rotating_duct_frame_w2/run.py /tmp/quick.json --quick
# 图与文档（由记录程序化生成，零手抄）
PYTHONPATH=src python benchmarks/verified/rotating_duct_frame_w2/make_figures.py
python3 benchmarks/verified/rotating_duct_frame_w2/make_doc.py
```

单元测试：`python -m pytest tests/test_rotating_duct.py -q`

## 已知缺陷（honest，入记录 `known_limitation`）

D3Q19 的**阶梯移动壁**（naive Ladd）在圆壁轴线附近的角点处逐节点注入质量（净注入为零但空间呈
m=2 分布），驱动出虚假 m=2 流：实测质量漂移 2.6e-2、剖面误差 ~14%（内部）/63%（名义），
比同参数下的 D2Q9 环隙（剖面误差 1.1%）差一个量级。已实现「质量守恒变体」（逐节点减去
$\langle \mathbf c_k\cdot\mathbf u_w\rangle$ 加权均值），漂移降到 1.7e-13，但动量滑移伪影仍在。
因此 Taylor–Couette 锚改用已验证的 D2Q9 环隙求解器；D3Q19 核只用于壁面与格点对齐（平面槽道，
实测误差 <1e-5）或无移动壁（方管）的场合。

另：Ladd 动量交换力矩在粗阶梯圆壁上不可用作锚（压力项不精确抵消，残差比解析力矩大两个量级），
力矩改由**剖面恒等式** $T(r)=2\pi\rho\nu L r^2(u'_\theta-u_\theta/r)$ 给出（r-无关性 + 幅值双检）。
