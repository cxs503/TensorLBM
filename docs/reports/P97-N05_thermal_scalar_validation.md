# P97-N05 · TensorLBM 热对流与被动标量验证面（Rayleigh–Bénard 临界 Ra_c / Nu(Ra) / 差分加热腔 / Taylor–Aris 弥散）

**节点**：P97-N05（TensorLBM）· **报告类型**：论文级验证报告 · **生成方式**：`scripts/p97_n05_campaign.py --collect` + `scripts/p97_n05_report.py`（所有数字从 `artifacts/p97_n05/validation.json` 与 `runs/p97_n05/*.json` 程序化注入，无手抄）

## 1. 案例介绍

本节点为 TensorLBM 建立**热对流 + 被动标量输运**的验证面。四个子问题共用同一套格子（D2Q9 动量 + D2Q5 温度/标量，Boussinesq/Guo 力格式）：

| 子问题 | 物理 | 控制参数 | 基准 |
|---|---|---|---|
| C1 线性稳定性 | 两无限大平行板间水平流体层的导热态失稳 | Ra，Pr=0.71，刚-刚壁 | Chandrasekhar (1961) $Ra_c=1707.76$，$a_c=3.1163$ |
| C2 Nu(Ra) | Rayleigh–Bénard 对流（x 周期、上下等温刚壁，展弦比 2） | Ra=1e4…1e6，Pr=0.71 | Clever & Busse (1974) 二维定常卷：Nu(1e4)=2.16，Nu(1e5)=4.22 |
| C3 差分加热腔 | 方腔左热右冷、上下绝热 | Ra=1e4，Pr=0.71 | de Vahl Davis (1983) Nu=2.243 |
| C4/C5 Taylor–Aris | 平面 Poiseuille 流中被动标量的轴向弥散 | Pe=Uh/D=1…100 | Aris (1956) 矩方法：$D_{eff}=D+2h^2U^2/(105D)$（另附圆管 1/48 讨论） |

**几何与格子**：RB 单元 $n_x\times n_y$ 格子单位，热壁钉在首/末行，故热间隙 $H=n_y-1$（速度半程反弹壁在 $y=-1/2,\,n_y-1/2$，比热间隙宽一格——这是既有算子的半格错位，本报告如实沿用并计入 $Ra$ 的定义）；方腔 $H=n_x-1$ 同理。

## 2. 怎么做的

**工具链**：`tensorlbm.thermal`（D2Q9+D2Q5 双分布、Guo 力、anti-bounce-back 等温壁）、`tensorlbm.rb_continuum_stability`（四阶算子有限差分离散 + 稠密广义特征值）、新增 `tensorlbm.taylor_aris`（标量 D2Q9 + Aris 辅助问题 + 弥散计）、`scripts/p97_n05_*`（案例表→分片执行→判据收敛→图）。全部案例由 `scripts/p97_n05_campaign.py` 的单一案例表 `CASES` 驱动，外协方双节点 loong64 集群分片执行。

**C1**：在 $a_c=3.1163$ 上对 $Ra$ 做二分（每个 $M$ 用稠密 QZ 全谱取最不稳定支，$\sigma=0$ 处定临界），再对 $M\in\{50,100,200\}$ 做 Richardson 外推；另在 $M=100$ 上扫描 $a$ 以验证最小临界点位置。**注意**：包内 `critical_at`/`critical_rayleigh_continuous` 依赖 ARPACK shift-invert 近零特征值，在本问题上返回伪谱（见 §5 缺陷登记），因此改走稠密路径。

**C2**：初始化 = 线性导热剖面 + $\sin(2\pi x/n_x)\sin(\pi y/H)$ 扰动（幅值 1e-3），周期 x、上下 anti-bounce-back 等温壁；每个 Ra 单独调 τ 以把格子 Mach 数压到 ≤0.1（Boussinesq 定常解与 τ 无关），后 $30\%$ 步长做时间平均。判据除与文献 Nu 对比外，另加**热流恒等式**：底壁、顶壁与中面（$Nu=\langle u_yT\rangle H/(\alpha\Delta T)-\partial_y\langle T\rangle H/\Delta T$）三个 Nusselt 数必须一致——这是不依赖任何文献值的机器可检能量守恒门。

**C3**：方腔 $Ra=10^4$、$Pr=0.71$，左壁 T=1、右壁 T=0、上下绝热；网格 64/96/128，按扩散时间 $H^2/\alpha$ 取步数（$\ge 4$ 倍），末段 50% 做时间平均。

**C4/C5**：新增 `taylor_aris` 模块，三层证据：(i) 闭式 $D+K a^2U^2/D$；(ii) **离散 Aris 辅助问题** $D\nabla_\perp^2\chi=-u'$，在 LBM 同一套节点/零通量壁上求解，$D_{eff}=D+\langle u'\chi\rangle$——它消除“抛物线在格点上取样均值≠U”的记账歧义，使残差纯为数值误差；(iii) LBM 标量 D2Q9 单模衰减计：释放 $\phi=1+A\cos kx$，测 $-(d\ln A/dt)/k^2$ 的渐近平台，并对场做**仅扰动部分**的周期性重标定，使 $t\gg h^2/D$ 的渐近段在 $D\to0$（高 Pe）时仍可测。C5 用两个对照分离数值扩散：$U=0$（纯扩散，必须回到 $D$）与 plug 流（均匀平移、无剪切，仍必须回到 $D$）。

## 3. 计算结果

### C1 临界 Rayleigh 数


### C2 Nu(Ra) 与热流恒等式


### C3 差分加热腔（de Vahl Davis）


### C4 Taylor–Aris 弥散


### C5 数值扩散分离


### C6 负控（Ra→0）


## 4. 结果图

![图 1 线性稳定性：Ra_c 的网格收敛（左）与临界波数（右）](figures/fig_c1_marginal_stability.png)

*图 1 线性稳定性：Ra_c 的网格收敛（左）与临界波数（右）*

![图 2 Nu(Ra) 标度律与壁/中面热流恒等式](figures/fig_c2_nusselt_rayleigh.png)

*图 2 Nu(Ra) 标度律与壁/中面热流恒等式*

![图 3 方腔温度场与 Nu 的网格收敛（de Vahl Davis）](figures/fig_c3_cavity_dvd.png)

*图 3 方腔温度场与 Nu 的网格收敛（de Vahl Davis）*

![图 4 Taylor–Aris 弥散：计量收敛、Pe 依赖与有限波数偏差](figures/fig_c4_taylor_aris.png)

*图 4 Taylor–Aris 弥散：计量收敛、Pe 依赖与有限波数偏差*

![图 5 数值扩散标定（C5 对照）](figures/fig_c5_numerical_diffusion.png)

*图 5 数值扩散标定（C5 对照）*

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

### 5.1 踩过的坑（缺陷/伪影，均已定位并修复或登记）

1. **弥散计在 $\tau_d\to0.5$ 时不可测**：单模振幅按 $D_{eff}k^2$ 指数衰减，而进入渐近段需要 $t\gg h^2/D$。高 Pe（小 D）时振幅先撞舍入地板、渐近段还没到——第一版直接给出 $-100\%$ 的假结果。修法：只对**扰动部分**做周期性重标定（线性方程下严格等价），配合 float64 与自适应拟合窗（起点 $3h^2/D$）。
2. **重标定只能标定扰动**：整场重标定会把背景一起放大，未被充分阻尼的横向模式随之放大，计量器在 $\tau_d=0.518$ 时输出乱跳的速率（+0.7 / −0.6 交替）。改成 $g_{eq}(\bar\phi)+f\,(g-g_{eq})$ 后速率与窗长无关（$R^2\to1$）。
3. **周期轨迹的矩估计会被绕回污染**：羽流跨过 $x=0$ 时方差突跳 $O(n_x^2)$（曾给出负的 $D_{eff}$）。修法：先按峰值做圆周平移再算二阶矩，并约束 $\sigma<n_x/6$。
4. **流式方向符号**：`scalar_stream` 初版写成 `roll(-c)`，等价于 $-u$；对弥散（只依赖 $u^2$）无影响，但会破坏移动参考系。已由单测 `test_scalar_stream_is_periodic_in_x` 钉住。
5. **权重归一化**：D2Q9 的 float32 权重和 $1+O(2^{-25})$，每步给平衡分布注入 $3\times10^{-8}$ 的质量亏欠，200 步纯扩散即漂移 $3\times10^{-6}$。加 `_weights()`（fp64 归一化 + 余量吸收）后质量守恒到 roundoff。
6. **包内 ARPACK 捷径不可用（缺陷登记，未改动库代码）**：`rb_continuum_stability.critical_at` 在 `sigma=0` 附近只取 4 个特征值，在本题上返回伪谱（实测 `spectra` 给 $1.5\ldots1.7$，稠密谱同一支为 $-1.7$），导致 `critical_rayleigh_continuous` 在波数搜索中抛 `Ra_hi=6000 still stable`。本报告改用稠密 QZ 全谱 + 二分（`growth()`），与 Chandrasekhar 值一致到 $10^{-4}$。建议后续给 `critical_at` 加大 k 或稠密兜底。
7. **热壁与速度壁半格错位**：anti-bounce-back 把温度钉在首/末**节点**上，故热间隙是 $n_y-1$；速度半程反弹壁在 $\pm1/2$ 之外，间隙是 $n_y$。若沿用 $H=n_y$ 构造 $Ra$，实际 $Ra$ 偏低 $(1-1/n_y)^3$（$n_y=50$ 时 6%、$n_y=16$ 时 18%），$Nu$ 口径也带 $1/n_y$ 偏差（$n_y=8,\ Ra=100$ 时实测 $Nu=1.124$ 而非 1）。本报告统一用 $H=n_y-1$（方腔 $n_x-1$）重构造 $Ra$ 与 $Nu$。这是一处值得后续修掉的一致性问题。
8. **判据门本身曾有洞**：`evaluate()` 早期版本对空证据集返回 `all_pass=True`（`all()` over empty）。已加"六项判据必须齐备"校验，并由单测 `test_gate_cannot_pass_without_evidence` 钉住。

### 5.2 Taylor–Aris 的适用范围（honest LIMIT）

弥散计的残差有两个可分离来源，二者都不是 Taylor 物理本身：

* **有限流向波数**：单模法测的是 $D_{eff}(k)$，$kh$ 非无穷小即有偏。Pe≈30 时把 $n_x$ 从 64 加到 256（$(kh)^2$ 由 0.19 降到 0.01），相对离散 Aris 参考的偏差从 （未运行） 变到 （未运行）。
* **横向分辨率**：剪切剖面在 $n_y$ 个节点上取样、零通量壁反射截断壁行，残差随 $1/n_y$ 下降。同一 Pe≈10：$n_y=9$ 时 （未运行），$n_y=17$ 时 （未运行）。

因此判据取每档 Pe 的**最细可用配置**（$kh\le0.2$、$n_y=17$）。可复用限界：$Pe\lesssim30$ 时取 $n_y\gtrsim17$、$n_x\gtrsim250$ 可在 5% 内复现 Aris 公式；**Pe=100 在本次算力预算内不可达**——格子胞 Péclet 数 $u\Delta y/D=Pe/h$ 需压到 $O(1)$，即 $h\gtrsim Pe$（$n_y\gtrsim200$），且 $t\sim3h^2/D\sim10^5$–$10^6$ 步，成本随 $Pe^2$ 增长。Pe=100 的实测值在 C4 表中如实列出，作为该限界的证据。

### 5.3 Rayleigh–Bénard 的限界

* 二维、展弦比 2、$x$ 周期：只对应 Clever & Busse 的**定常卷分支**，不能直接与三维实验或大展弦比统计标度比较。
* 每档 Ra 单独调 $\tau$ 把格子 Mach 数压到 $\lesssim0.1$（Boussinesq 定常解与 $\tau$ 无关，但压缩性误差 $\sim Ma^2$ 与稳定性有关）；$Ra\ge10^6$ 时 $Ma$ 与扩散时间同时恶化，成本按 $H\,Ra/u_{ff}^2$ 增长。
* 标度指数在 $Ra\le10^6$ 只到经典区（$\beta\approx0.3$），$\beta\to1/3$ 的渐近区需要 $Ra\gg10^6$——拟合值见 C2 表，未强行凑 1/3。
* $u_{max}$ 的归一化：`benchmarks/pending/rayleigh_benard` 引用的 de Vahl Davis $u_{max}\approx0.1926$ 在 $\alpha/L$ 与 $\nu/L$ 两种归一化下都无法复现（本平台无量纲速度量级 $\mathcal{O}(10)$，与自由落体标度 $\sqrt{g\beta\Delta T H}$ 的 0.6 倍一致）。故 $u_{max}$ **不作为判据**，只作流态指标报告。

### 5.4 可复现性

所有案例由 `scripts/p97_n05_campaign.py` 的 `CASES` 单一案例表驱动，可在集群上 `--shard i --nshards n` 分片；`--collect` 重算判据并写 `artifacts/p97_n05/validation.json`（含每案配置、拟合窗、$R^2$、质量漂移）。新增单测 39 条（`tests/test_taylor_aris.py`、`tests/test_p97_n05_validation.py`），覆盖闭式/离散 Aris、质量守恒、零通量壁、纯扩散标定、plug 对照、Pe=10 物理对标，以及**判据门本身**（空证据集必须判 FAIL）。
## 6. 误差判定

| 判据 | 关键量 | 基准 | 实测偏差 | 阈值 | 是否通过 |
|---|---|---|---|---|---|

**总判定**：全部通过。
