# p223w3-n02：LBM 湍流槽道 Re_τ = 180/395/590 对标验证面

> 本文件由 `benchmarks/p223w3_n02_channel_dns/make_report.py` 程序化生成，
> 全部数字来自 `benchmarks/p223w3_n02_channel_dns/case_data/*.json` 与 `gates.json`，
> 无手抄。求解器 = `src/tensorlbm/turbulence/les_chan.c`（本单修补，见 §2.2）。

## 1. 案例介绍

**物理背景.** 压力驱动平面槽道湍流是壁湍流的第一性原理算例：流向/展向周期、
两壁无滑移、常量体力 `a = 2u_τ²/H` 驱动。稳态流向动量平衡是精确的

```
τ_w = ρ a h,   u_τ = sqrt(a h),   Re_τ = u_τ h / ν,   C_f = 2 (u_τ/U_b)²
```

因此 `τ_w` 与 `u_τ` 由体力路线**闭式**给出，任何偏差都只能来自离散/统计误差，
不存在"标定过"的自由参数——这是本验证面能对标 DNS 的前提。

**几何与参数（格点单位，D3Q19）.** 壁面 = 半程反弹平面 `y = 0.5` 与 `y = H + 0.5`，
`h = H/2`。所有算例取 `u_τ = 0.0125`（时间尺度的旋钮：一个涡周转 = `h/u_τ` 步），
域 `L_x × L_z = 4h × 2h`（DNS 惯用 6.28h × 3.14h，本单为省算力截断，见 §5）。

| case | $Re_\tau$ | $\Delta y^+$ | grid (nx x ny x nz) | cells | $\nu$ | $\tau_0$ | steps | 统计涡周转数 |
|---|---|---|---|---|---|---|---|---|

**DNS 参照数据.** Moser–Kim–Mansour (1999) 原始剖面表，随库发布并带上游 SHA-256
（`src/tensorlbm/data/mkm1999/provenance.json`，上游 `chandata.tar.gz`
sha256 = `1d8be14c…02ed2`）：Re_τ = 178.12 / 392.24 / 587.19。**未**使用
`benchmarks/in99_n03_channel_turb/dns_ref/*.csv`——该文件与 MKM 原表不一致（§5 第 3 条）。

## 2. 怎么做的

### 2.1 数值方法（既有生产内核）

* 碰撞：D3Q19 双松弛（TRT），`s⁺ = 1/τ₀`，`s⁻ = 8(2-s⁺)/(8-s⁺)`（Λ = 3/16）；
* 力项：Guo 体力，TRT 分解 `(1 - s⁻/2)` 前因子（保证 `j_post = j + F` 精确）；
* 壁面：半程反弹（第二序）；SGS：Smagorinsky `C_s = 0.1`，`τ_t = ½(τ₀ + √(τ₀² + 18(C_sΔ)²Q/ρ))`；
* 统计：每个采样步累加 `⟨u_i⟩, ⟨u_iu_j⟩` 全部 (x,z) 与本行 y，两半槽道折叠取平均。

### 2.2 本单定位并修复的两个缺陷（`les_chan.c`）

1. **正则化投影支路在运行点数值发散（默认值缺陷）.** 原默认 `--reg 1`（二阶 Hermite
   投影碰撞）。实测：`τ₀ = 0.5125`（Δy⁺=3）与 `τ₀ = 0.5075`（Δy⁺≈5.6）下，
   `reg=1` 一旦流动转捩即 `ρ → NaN`；同工况 `reg=0`（TRT + Smagorinsky）稳定。
   修复：默认 `--reg 0`，并在 `LesChannelConfig` 中显式暴露 `regularized` 字段
   （此前 `cli_args` 根本没传 `--reg`，即**所有历史战役都跑在被判为发散的支路上**）。
2. **发散时写出非法 JSON（fail-open 缺陷）.** 原实现无条件写 `nan` 到结果 JSON，
   JSON 规范不允许 `nan`，下游 `json.loads` 抛解码错误——数值失败被伪装成格式错误。
   修复：写文件前扫描全部分布函数，非有限即 `exit(2)` 并报"非有限状态，结果未写出"；
   Python 侧 `run_native` 也失败闭合（`status != ok` 直接抛错）。

另外给内核加了中平面 `*_xzm.csv` 切片输出（gate c 的能谱来源）。

### 2.3 战役设计与工具链

```
benchmarks/p223w3_n02_channel_dns/
├── campaign.py      # 算例矩阵 + 单例归约（壁面单位、Cf、能谱、u_τ 三路线）
├── analysis.py      # 纯函数归约库（对数律/Reichardt/应力拟合/能谱/网格收敛）
├── gates.py         # 判据 (a)-(f) 机算 + gates.json
├── make_figures.py  # 7 张图（全部读 case_data/*.json）
└── make_report.py   # 本报告生成器
```

`campaign.py` 用 `src/tensorlbm/turbulence/les_channel.py::run_native` 调原生 C 内核
（gcc -O3 -fopenmp），结果 JSON 自带 provenance（源码/binary sha256、核数、Mcell-steps/s）。

## 3. 计算结果

### 3.0 战役执行台账（每个算例的结局，失败也算结果）

| case | $\Delta y^+$ | grid | steps | wall clock | status | evidence |
|---|---|---|---|---|---|---|
| R180_dy3 | 3/6 | 120x122x60 | 52,800 | - | **diverged** | les_chan diverged: 16416000 of 16689600 populations non-finite (grid 1 |
| R180_dy3c | 3/10 | 72x122x36 | 52,800 | - | **diverged** | les_chan diverged: 5909760 of 6008256 populations non-finite (grid 72x |
| R180_u02 | 3/6 | 120x122x60 | 24,000 | - | **diverged** | les_chan diverged: 16416000 of 16689600 populations non-finite (grid 1 |
| R180_u02c | 3/10 | 72x122x36 | 24,000 | - | **diverged** | les_chan diverged: 5909760 of 6008256 populations non-finite (grid 72x |
| R180_dy4 | 4/8 | 90x92x45 | 43,200 | - | **diverged** | les_chan diverged: 6925500 of 7079400 populations non-finite (grid 90x |
| R180_dy5 | 5/10 | 72x74x36 | 28,800 | - | **diverged** | les_chan diverged: 3545856 of 3644352 populations non-finite (grid 72x |
| R180_dy6 | 6/12 | 60x62x30 | 24,000 | - | **diverged** | les_chan diverged: 2052000 of 2120400 populations non-finite (grid 60x |
| R395_dy8 | - | - | - | - | not run | - |
| R590_dy12 | - | - | - | - | not run | - |

> 结论先说：**7 次湍流尝试无一存活**（`τ0 = 0.5062–0.5200`，`Δy⁺ = 3–6`，
> `Δx⁺ = 6–12`），全部在 2e4–7e4 步之间 `ρ → NaN`——说明这不是单一参数越界，
> 而是当前显式 TRT + 局部 Smagorinsky 内核在壁湍流区的**结构性数值不稳定**
> （见 §5 第 7 条、§8）。因此 Re_τ = 180 的湍流判据与 Re_τ = 395/590
> （成本 ∝ `Re_τ²`，需 1.1e11–3e11 cell-steps，为全部预算的 2–5 倍）
> 一律按 `FAIL/WITHHELD` 记录，不做替代、不降判据。

### 3.1 平均流与摩擦系数

| case | $U_b$ | $Re_b$ | $u_\tau$ (体力) | $u_\tau$ (应力拟合) | 二者散度 | $C_f$ | $C_f$ Dean | vs Dean |
|---|---|---|---|---|---|---|---|---|

### 3.2 对数律（gate a，窗口 y⁺ ∈ [30,200]，κ=0.41，B=5.2）

| case | 点数 | RMS 误差 | 最大误差 | $\kappa$ 拟合 | $B$ 拟合 | DNS 自身 vs 对数律 (参考) |
|---|---|---|---|---|---|---|

> "DNS 自身 vs 对数律"一列是**参照基线**：MKM 数据本身在该窗口内与
> `ln(y⁺)/κ + B` 的最大偏差，说明 5% 判据本身的物理含义（对数律是渐近式，
> 低 Re_τ 下 DNS 自己也有几个百分点偏差）。

### 3.3 雷诺应力（gate b，Re_τ=180）

| case | $u'u'^+_{max}$ | $y^+_{max}$ | DNS $u'u'^+_{max}$ | DNS $y^+$ | 峰值误差 | $-\langle u'v'\rangle^+_{max}$ | $y^+$ | DNS 剪切峰 |
|---|---|---|---|---|---|---|---|---|

### 3.4 能谱惯性子区斜率（gate c）

| case | 拟合斜率 | 频点数 | $k^+$ 区间 | vs -5/3 |
|---|---|---|---|---|

拟合区间 `k⁺ ∈ [0.05, 0.30]` **先验固定**（不随数据调整），上限由 Δx⁺ 决定，
下限由盒长 `L_x⁺ = 4Re_τ` 决定。

### 3.5 网格收敛（gate f，Re_τ = 180 三档）

| case | $\Delta y^+$ | ny | $u_\tau$ (应力拟合) | vs 体力 $u_\tau$ |
|---|---|---|---|---|

### 3.6 层流负控（gate e）

| 层流算例 | steps | $L_2$ 误差 | 最大误差 | $u_{max}$ 数值 | $u_{max}$ 解析 |
|---|---|---|---|---|---|
| LAM_poiseuille | 38,907 | +0.165% | +0.170% | 0.062537 | 0.062431 |

## 4. 结果图

> 图 1–5、7（平均速度剖面、雷诺应力、能谱、Cf、网格收敛、瞬时流场）依赖**完成的
> 湍流算例**，本单 7 次尝试全部发散（§3.0），因此它们无法生成——不是遗漏，
> 而是"没有可信数据就不画"。当前只有层流负控（gate e）有资格出图，见下。

**图 6 层流负控：Poiseuille 解析解（gate e）**

![图 6 层流负控：Poiseuille 解析解（gate e）](benchmarks/p223w3_n02_channel_dns/figures/fig6_laminar_control.png)

## 5. 注意点（踩过的坑、局限、适用范围）

1. **正则化支路是默认值缺陷（§2.2-1）.** 教训：`--reg` 不在 CLI 白名单里被
   `cli_args` 传递，默认值就成了"不可见的实验条件"。这一条足以让历史战役的
   湍流通道结论失效（发散被写进 JSON 当时还会被下游当作数据）。
2. **发散必须失败闭合（§2.2-2）.** 写 `nan` 的 JSON 让"数值失败"变成"格式错误"，
   掩盖了物理失败——诚实 FAIL 的前提是失败**可见**。
3. **`benchmarks/in99_n03_channel_turb/dns_ref/*.csv` 与 MKM 原表不符.**
   其 `uu_plus` 在壁面单调升到 16.57（y⁺=178），而 MKM 原表 `R_uu` 峰值 7.07 @ y⁺=15.3；
   `uv_plus` 甚至为正。若用它做 gate b 会得到荒谬结论。本单改用
   `src/tensorlbm/data/mkm1999/` 原始表（带上游 sha256），并写测试冻结该差异。
4. **算力-精度的耦合（`u_τ` 是时间尺度旋钮）.** `T_turn = h/u_τ`，
   统计步数 ∝ `1/u_τ`；而 `Δy⁺ = u_τ/ν`，壁面分辨率 ∝ `u_τ`。把 `u_τ` 顶到
   压缩性上限 0.0125 买到 2.5× 统计量，代价是 `Δy⁺ ≥ 3` 与 `Ma` 上限 ~0.43——
   本单是**壁面分辨的粗 LES**，不是 DNS 级 `Δy⁺ ≲ 1`。
5. **层流负控必须跑到黏性扩散时间之后.** 内核初值是律-of-the-wall 剖面，
   在低 Re_τ 上与抛物线差别很大：H=16 时 6000 步只有 2.3 个黏性时间，L2 误差 14.5%
   （假 FAIL）；跑满 ~9 个黏性时间后降到 <1%。这张"忘记等收敛"的坑已固化进
   `tests/test_p223w3_n02_channel_dns.py::test_kernel_recovers_the_poiseuille_parabola`。
6. **盒长截断.** `L_x = 4h, L_z = 2h`（DNS 6.28h × 3.14h）：非常大规模运动被砍掉，
   对二阶统计（`u'u'`、`⟨u'v'⟩`）影响小于对能谱低波数端的影响；能谱拟合区间
   下限因此不取 `k⁺ < 0.05`。
7. **稳定包络（硬约束，决定了本单能跑什么）.** 测得：同一状态（`96x122x48`,
   `τ0 = 0.5125`，3000 步）下 `reg=1` 发散、`reg=0` 稳定
   （`artifacts/p223w3_n02_channel_dns/collision_branch_pair.json`）；而 `τ0` 与网格
   分辨率同时收紧时 `reg=0` 也会发散（§3.0 台账：`Δy⁺ = 4/5/6` 与
   `Δx⁺ = 10` 全部发散）。可用的时间是**有限的**：`τ0 = 0.5 + 3u_τ/Δy⁺ ≳ 0.512`
   ⇒ `Δy⁺ ≲ 3`（在压缩性上限 `u_τ ≤ 0.0125` 下）。而壁面分辨网格的成本
   `N ∝ Re_τ²`、步数 `∝ Re_τ/Δy⁺` ⇒ Re_τ = 395/590 需要 1.1e11–3e11 cell-steps，
   是本次全部预算（约 1e11）的 2–5 倍 → 未跑，判据 `WITHHELD`。
8. **统计长度.** 见 §3.1 表：8–12 个涡周转（DNS 惯例 ≥20）。雷诺应力峰值的
   统计不确定度按 `1/√N_ind` 估计约 3–5%，本单在结论里按此读取偏差。

## 6. 误差判定（逐项 vs 基准，是否 <3%）

| gate | 判据 | 实测 | 结论 | 备注 |
|---|---|---|---|---|
| a_log_law | RMS 误差 < 5% | — | **WITHHELD** | 无完成算例 |
| b_reynolds_stress | 峰值幅值误差 < 10% 且 10 <= y+ <= 20 | — | **WITHHELD** | no completed Re_tau=180 case |
| c_energy_spectrum | 斜率 vs -5/3 误差 < 10% | — | **WITHHELD** | 无完成算例 |
| d_friction_coefficient | Cf vs Dean < 5% | — | **WITHHELD** | 无完成算例 |
| e_laminar_control | L2 误差 < 1% | +0.165% | **PASS** | $u_{max}$ 0.062537 vs 0.062431 |
| f_grid_convergence | 三档网格 u_tau 散度 < 3% | — | **WITHHELD** | 无完成算例 |

**是否 <3% 的显式回答**：

* gate (e) 层流 Poiseuille：**是**（L2 = +0.165% < 1% < 3%）；
* gate (f) 网格收敛：无完成网格（判据不可评估）；
* gate (a) 对数律：按 **5%** 判据（不是 3%）判定，见上表；
* gate (d) Cf vs Dean：按 **5%** 判据判定；注意 Dean 关联式自身与 MKM DNS 差
  3–4%（DNS `C_f` = 0.00814 vs Dean 0.00842 @ Re_b ≈ 5600），即该判据的
  系统性下限就在 3–4% 量级；
* gate (b) 雷诺应力峰值：按 **10%** 判据判定；
* gate (c) 能谱斜率：按 **10%** 判据判定。

超出判据的项目一律如实记 `FAIL`，不覆盖、不调参掩盖。

## 7. 证据指针与复现

| 产物 | sha256(前16) |
|---|---|
| `case_data/LAM_poiseuille.json` | d700fc344974fe69 |
| `case_data/gates.json` | 4ae11b5ec0a1f687 |
| `src/tensorlbm/turbulence/les_chan.c` | 84d8432265fbc214 |
| `benchmarks/.../analysis.py` | 1405a75133a7dce5 |
| `benchmarks/.../campaign.py` | 1a4466f8327b829d |
| `benchmarks/.../gates.py | d46eee057d8d7948 |

## 8. 结论（honest FAIL）与后续路线

**本单的诚实结论：湍流段的判据没有一个被证实。** 全部 6 次湍流尝试都以
`ρ → NaN` 发散收场（台账见 §3.0，网格 `60x62x30 … 120x122x60`，
`τ0 = 0.5062 … 0.5200`，每次在 2e4–7e4 步之间失稳）；只有层流负控（gate e）
与解析锚在 1% 内通过。按纪律，**FAIL/WITHHELD 不覆盖、不改判据**。

**已确认的两件事（本单的净增量）：**

1. **两个真实缺陷被定位并修复**（§2.2）。修复前默认碰撞支路在生产运行点上
   必然发散（同一状态 3000 步：`reg=0` 稳定 `u_bulk = 0.2117`，`reg=1` 立刻 NaN，
   见 `artifacts/p223w3_n02_channel_dns/collision_branch_pair.json`）；且发散时
   会把 `nan` 写进结果 JSON，把数值失败伪装成格式错误。任何在这条支路上
   产出的历史湍流数字都应视为不可信，须重跑。
2. **可复现的验证面与失败台账**：`campaign/analysis/gates/probe_stability/
   make_figures/make_report` 六件套 + 46 条 pytest，判据机算（`case_data/gates.json`），
   报告与图零手抄。以后换更稳的碰撞模型，只需重跑 `campaign.py` 再跑
   `gates.py`，判据自动重判。

**后续路线（按性价比排序）**

1. **换稳定碰撞算子**：把 `reg=1` 那套二阶 Hermite 投影真正修对（当前实现
   系数为 4.5·w·(cc−δ/3) 的投影，实测会把该模式欠松弛一半；正确的正则化应
   保留 2/3 阶矩并按 `(1−s)` 全量松弛），或直接换 cumulant/KBC 碰撞
   （仓库里 `turbulence/turbulent_channel3d.py` 的 TRT + 插值反弹是另一条已验证
   可跑湍流的实现，只是纯 torch 太慢，约 1 µs/cell-step）。
2. **把算力花在 `Δ⁺ ≤ 2` 的壁面分辨网格上**，并且统计长度 ≥ 20 个涡周转：
   本单证明"粗网格 + 近 0.5 松弛时间"这条路在显式格式下根本走不通。
3. **Re_τ = 395/590 需要上面的稳定性问题先解决再谈**：壁面分辨成本 ∝ `Re_τ²`，
   395 需 ~1.1e11、590 需 ~3e11 cell-steps，是本次全部预算的 2–5 倍。
4. 顺带修掉 `benchmarks/in99_n03_channel_turb/dns_ref/*.csv`（与 MKM 原表不符），
   否则后续任何用它的 gate b/c 都会得到荒谬结论。

```bash
# 全量复现（每个算例 0.5–2 h，取决于核数）
python3 benchmarks/p223w3_n02_channel_dns/campaign.py --case all --threads 28
python3 benchmarks/p223w3_n02_channel_dns/gates.py
python3 benchmarks/p223w3_n02_channel_dns/make_figures.py
python3 benchmarks/p223w3_n02_channel_dns/make_report.py
# 判据回归（<1 min，全部小网格）
pytest -q tests/test_p223w3_n02_channel_dns.py
```
