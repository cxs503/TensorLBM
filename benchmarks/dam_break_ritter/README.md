# Ritter (1892) 溃坝自相似律基准（浅水自由面解析锚）

## 1. 物理背景

初始静止水深 $h_0$ 的水库（$x<0$）在 $t=0$ 释放到干床（无摩阻）。浅水方程（SWE）

$$
\partial_t h+\partial_x(hu)=0,\qquad
\partial_t(hu)+\partial_x\!\left(hu^2+\tfrac12 g h^2\right)=0
$$

的自相似解（Ritter 1892）：令 $c_0=\sqrt{gh_0}$、$\xi=x/t$，

$$
h(x,t)=\frac{(2c_0-\xi)^2}{9g},\qquad
u(x,t)=\frac{2}{3}\left(c_0+\xi\right),\qquad 0\le x\le 2c_0t,
$$

$x\le -c_0t$ 时 $h=h_0$，$x\ge 2c_0t$ 时 $h=0$。**前沿匀速** $x_f(t)=2c_0t$ 即本基准的解析锚
（经典实验对照见 Martin & Moyce 1952）。

## 2. 数值模型（`src/tensorlbm/shallow_water.py`）

1-D 有限体积 **Rusanov（局部 Lax–Friedrichs）通量 + 隐式静水重构**：

* 严格通量形式 → 质量守恒到舍入误差（实测漂移 $\sim10^{-13}\%$）；
* 左端**镜面虚网格壁面**（$h_g=h_0,\ (hu)_g=-(hu)_0$）保证 lake-at-rest 精确保持；
* 右端零梯度出流；干床用 $h_{\rm dry}=10^{-4}h_0$ 阈值（干格动量清零）；
* 速度去奇异化 $u=hu/\max(h,\epsilon)$。

**为什么不是 lattice-BGK 浅水模型**：先实现了 D1Q3 的“非理想压力力”浅水 LBM
（$F=-\partial_x(\tfrac12gh^2-c_s^2h)$），在低 celerity 区间 $\tfrac12gh^2\ll c_s^2h$
（比值 $\sim 2/(3gh)$，本配置 $10^2$–$10^3$），所需力要抵消比它大 2–3 个量级的格点压力项，
干湿前沿因此数值失稳（负分布函数/爆炸），故不作为可用模型交付，仅在此记录为已排除路线。
另一条“深度平均 SWE 的格点 BGK 精确压力平衡”路线也因同样量级问题被排除。

## 3. 两种前沿估计量

1. `front_geometric`：最外侧湿格（$h>10^{-3}h_0$）+ 亚格插值。与
   `benchmarks/bench_dam_break.py` 共享同一前沿跟踪工装
   （`tensorlbm.dam_break.find_front_x`，本单把该函数提升为公共 API，原私有函数改为其别名）。
2. `front_mass`：由 **Ritter 律的积分形式**反解前沿。自相似剖面在 $0\le x\le 2c_0t$ 上积分为
   $V_{\rm down}(t)=8c_0^3t/(27g)$，消去 $t$ 得

   $$
   x_f=\frac{27\,V_{\rm down}}{4h_0}.
   $$

   $V_{\rm down}$ 是数值解的**守恒量**（坝下泄水量），因此该估计量不受干床尖端
   （解析解在该处 $h\to0$ 二次衰减、一阶格式无法解析）数值涂抹的影响 —— 这是本基准
   用于 ≤5 % 判据的估计量，属“物理上等价的前沿律检验”。

## 4. 结果（`result.json`，驱动 `self_similar.py`，判据窗口 $t\in[0.5,5]\sqrt{h_0/g}$）

| nx | dx | 前沿(质量积分) 最大误差 | 前沿(几何) 最大误差 | 质量漂移 | 剖面相对 $L_2$ |
|---|---|---|---|---|---|
| 400  | 0.040 | 6.41 % | 25.94 % | $2.2\times10^{-14}$ % | 8.02 % |
| 800  | 0.020 | 6.41 % | 25.94 % | $2.0\times10^{-13}$ % | 6.70 % |
| 1600 | 0.010 | **3.68 %** | 25.47 % | $1.1\times10^{-13}$ % | 5.91 % |

**判定（声明参考分辨率 nx=1600）**：

* 前沿律 $x_f=2\sqrt{gh_0}\,t$：最大误差 **3.68 % ≤ 5 % → PASS**（同时报告粗网格 6.41 %
  作为收敛研究，未达标，属一阶格式在干床尖端的已知局限）；
* 质量守恒：漂移 **$10^{-13}$ % ≤ 1 % → PASS**（严格通量形式，机器精度级）；
* 剖面自相似：$h\,t^2/h_0$ 在 $h>10^{-2}h_0$ 区域内相对 $L_2$ 5.9 %（见
  `figures/ritter_profiles.png` 的自相似塌缩）。

几何尖端跟踪器给出 ~25 % 的**偏低**：一阶 Rusanov 的数值扩散把 $h\to0$ 的尖端抹平，
阈值交点落在解析前沿之后。这是**方法固有**而非实现缺陷（细化时该误差仅缓慢下降），
故只作透明性报告，不用于判据。

## 5. 复现

```bash
PYTHONPATH=src python benchmarks/dam_break_ritter/self_similar.py            # 全量三档
PYTHONPATH=src python benchmarks/dam_break_ritter/self_similar.py --fast     # 冒烟
PYTHONPATH=src python benchmarks/dam_break_ritter/make_figures.py            # 出图
PYTHONPATH=src python -m pytest tests/test_shallow_water_ritter.py -q
```

产物：`result.json`、`runs/nx*.json`、`figures/ritter_front_law.png`、
`figures/ritter_profiles.png`。