# EOS-Maxwell 构造稳定性探针（EOS-Maxwell construction stability probe）

**锚点**：GAP-LBM-MCMP-POISEUILLE 高密度比路判据 · **仓**：TensorLBM ·
**性质**：工装/探针（instrument），不是通过式基准（见 §9 局限与 §11 路径声明）

**一句话结论（判决见 `RESULTS.md`，数字全部机器生成）**：EOS 等面积 Maxwell 构造本身
在机器精度内可解（三档目标 10/100/1000 全部反解成功，面积残差 ~1e-13），但**用该构造
初始化出来的单组分伪势（SCMP）静态液滴不是离散格式的不动点**：三档 × 两种 forcing 的
6 个 tier 全部未过预注册门（0/6 PASS）；可长时间保持的是最低档，且其伪流幅值已越过
预注册 FAIL 门；密度比 100 档"能跑但维持不住"（稳态密度比约为 Maxwell 锁值的 3 倍）；
密度比 1000 档在数十至一百余步内塌缩。塌缩被如实记为结果（含起始步与机制），不重试调参。

---

## 1. 物理问题

单组分 Shan–Chen 伪势（SCMP）+ 状态方程型伪势 ψ：

```
ψ(ρ) = sqrt( 2 (ρ c_s² − p_EOS(ρ; T)) / c_s² )     (clamp ≥ 0)
p_mech(ρ) = ρ c_s² − (G/2) c_s² ψ² ,  本库后向 gather 约定下 G = +1 为吸引
```

库内实现为 `tensorlbm.multiphase.make_psi_eos('pr'|'cs', T)`（Yuan & Schaefer 2006
参数表：PR a=2/49, b=2/21, R=1；CS a=1, b=4, R=1；含符号约定说明）。构造线索来自
开源实现 openLBMPM（`OptimizedD2Q9GPU.py` 的 PR-EOS 伪势段）——**仅作方法线索**，
探针调用的是本库函数，未复制任何外部代码。

**为什么要构造（Maxwell）**：亚临界温度下 EOS 有液/气共存，共存对 (ρ_l, ρ_v) 由
等面积（Maxwell）规则在 v = 1/ρ 空间确定：

```
∫_{v_l}^{v_v} [p(v) − p_sat] dv = 0   ⇔   μ(ρ_v) = μ(ρ_l)
```

本探针**不粘贴任何历史锁值**：调用 `tensorlbm.sc_eos_coexistence.maxwell_coexistence`
（库的等面积求解器），并对 **T_r 做二分**反解出"目标密度比 = 10 / 100 / 1000"对应的
EOS 点（密度比随 T_r 单调，见 `RESULTS.md` §1 与 `tests/test_eos_maxwell_stability.py`
的单调性门）。三档目标全部命中（相对误差 ≤1%），面积残差相对量 ~1e-13…1e-11。

物理问题是**静态液滴**：周期方域 L×L，中心半径 R 的 tanh 液滴（初始宽度 w），液/气
密度用 Maxwell 锁值，零初速。测三件事：**能否保持、伪流多大、密度比维持误差多少**。

## 2. 为什么"构造成立"≠"能跑"（探针的决策对象）

Maxwell 是**连续 EOS** 的陈述；离散 SC 格式能否把该共存对变成**不动点**是另一回事
（伪势力低阶离散 → 界面压力张量误差 → 气相支指数级敏感）。探针的决策是：

> **给定目标密度比，SCMP+PR-EOS 液滴在什么条件下是"可用构造"？不可用时最先失效的
> 观测量是什么？**

**竞争替代方案**：forcing ∈ {EDM（精确差分）, Guo-2002 源项}；力离散 scheme ∈
{standard, pressure_tensor（严格无旋、严格动量守恒）}；构造初值界面宽 w ∈ {4,6,8}；
EOS 点（= 目标密度比）。

**基线与对照**：
- 基线 1 = Maxwell 构造本身（每档解析目标）；
- 基线 2 = **平面界面对照**（liquid slab，周期盒，同一 EOS 点、同一 forcing）：去掉
  曲率与"小蒸汽库"，分离"格式误差"与"液滴几何/有限域"两类原因；
- 基线 3 = **长持时**（本战役 `--longhold-steps 40000` = 2N；N=20000 已覆盖最慢线性时标，
  长持时再加倍以分离"未弛豫"与"真稳态"）：
- 交叉核对 = 与已验证基准 `benchmarks/verified/laplace_pr_eos` 内嵌 Maxwell 锁
  （T_r=0.55, ratio 123.2232）比对（pytest 门，rel ≤1e-3）。

## 3. 预注册判据（先冻结，后出数）

冻结在 `probe_lib.PREREG`，随产物一同写入 `out/results.json`（`prereg` 字段），
判据代码与结果同源可核。

| 判据 | 阈值 | 说明 |
|---|---|---|
| 构造命中 | Maxwell 密度比与目标比相对误差 ≤ 1% | 二分 T_r |
| 构造一致性 | 面积残差 / (p_sat·Δv) ≤ 1e-6 | 等面积闭式核对 |
| 稳定保持 | ≥ **20000** 步无塌缩（tier 档）；boundary 档 4000 步 | N 取 20000 ≈ 蒸汽扩散时间 τ_D≈L²/D 的 2 倍 + 伪流弛豫时间尺度的量级，且使质量漂移门在 5e-8/步下可判 |
| 密度比维持 | \|比_测 − 比_Maxwell\| / 比_Maxwell ≤ **15%** | 平台精度标准（>15% 不可接受） |
| 伪流 | max\|u\|/c_s ≤ **0.05** = PASS；≤ **0.10** = MARGINAL；> 0.10 = **FAIL** | 静态液滴的伪流幅值口径 |
| 质量守恒 | \|Δm/m\|/步 ≤ 5e-8 | 与 `laplace_pr_eos` 同门 |
| ψ 定义域 | 采样尾窗内 ρ_max < ρ_cross（clamp 不触发） | clamp 触发即判失效（独立失效模式） |
| 塌缩判据 | 非有限 / ρ_min < −1e-6·ρ_max（超出 float32 舍入）/ max\|u\| > 0.5 c_s / clamp 触发 | 记录**起始步**与机制 |
| honest-FAIL | 塌缩=合法结果，如实记录，**不得**事后调参重试以制造通过 | 见 §7 |

## 4. 实验设计要点（可核）

- **单位分析单元**：一次 (目标密度比, forcing, 构造参数) 运行。
- **变量**：受控=G=+1、D2Q9、float32、周期边界、L=96、R=20、CPU；基准档 τ=1。
  变化=EOS 点（目标比）、forcing、力离散 scheme、初值宽度 w、**弛豫时间 τ（τ 杠杆臂，
  预注册格点 `PREREG['tau_arm']`，只跑 EDM）**。
- **保持步数 N 的合理性（可核）**：`probe_lib.stability_budget()` 给出三个线性时标
  （声波穿盒 L/c_s、界面宽度黏扩散 w²/ν、液滴半径黏扩散 R²/ν，ν=c_s²(τ−½)），
  N 取到最慢时标的数倍；比值写进 `CRITERION.md` D0 与 pytest
  `test_hold_length_is_justified_by_the_slowest_timescale`。同一预算也用于标注
  τ 杠杆臂里"N 未覆盖该 τ 的黏性时标"的运行（预注册 caveat，不作正面证据）。
- **混淆因子与对策**：①初期声学瞬变 → 采样窗取尾窗 5 个采样点，并另设长持时臂（2N 步）；
  ②有限域蒸汽库不足 → 平面界面对照（蒸汽库近半域）；③未弛豫 → 长持时；
  ④初始化周期接缝伪差（本探针第一版踩到，见 §6-H6）→ 平板构造改为居中 slab。
- **重复性/确定性**：纯确定性（无随机数、无并行原子加），同一配置重跑逐位一致
  （`laplace_pr_eos` 已锁定该库的确定性；本探针沿用同一 CPU/单进程路径）。本战役另给
  两条机器可核的确定性证据：①**线程数臂**（torch 1/2/4/8 线程，float32 归约序改变）
  三个观测列逐位相同（`RESULTS.md` §6c）；②**跨版本复现**——上一轮战役（旧 `probe_lib`
  修订、64 线程环境）与本轮（新修订、并行拆分作业）的 tier 实测比/塌缩步逐位相同。
  ③**微扰臂**：1e-7 与 1e-5 的随机密度扰动（2 个种子）不改变任何观测列（§6d）。
- **环境记录**：`out/results.json.environment`（torch 版本、python、平台、线程数、
  `probe_lib` sha256）+ `driver_sha256` + `generated_utc`。
- **资源与拆分**：单次 tier 20k 步 ≈ 80 s（≈4 ms/步，L=96，CPU）；战役拆两个集群作业
  （A=核心臂 `tiers,boundary,controls,widths,longhold,schemes,repeats`；B=杠杆臂
  `tau,robust,threads`），各写自己的 `out/results.json`，再由 `merge_runs.py` 合并；
  合并拒绝 prereg/代码哈希不同源、tag 重复，并把每个来源作业的
  `generated_utc/wall_s_total/各臂条数` 写进 `provenance.sources`。
  计算外协方 所双节点 loong64 上执行（`run_cluster`，`stage_dir` 本工作树）。

## 5. 结果

**本文件不复制任何数字**：全部结果表由 `plot_figures.py` 从
`out/results.json` 生成到 **`RESULTS.md`**（§1 构造 / §2 tier / §3 长持时 / §4 平面
对照 / §5 力离散杠杆 / §6 构造宽度 / §6c 线程 / §6d 扰动 / **§6e 黏性 τ 杠杆** /
§7 稳定边界 / §8 判决块）；同一脚本另生成 **`CRITERION.md`**（交付给
GAP-LBM-MCMP-POISEUILLE 的判据 D1–D4，含决策表），两份文件里的每个数字都来自
`out/results.json`，无手抄。

图件（`figures/`，均由 `plot_figures.py` 程序化生成）：

| 图 | 内容 |
|---|---|
| `fig1_radial_profiles.png` | (a) 静态液滴**径向密度剖面**（末态，对数轴，虚线=Maxwell 渐近值）；(b) 平面界面**逐行密度剖面**对照 |
| `fig2_spurious_currents.png` | (a) 伪流 max\|u\|/c_s 时间史（含预注册门线 0.05/0.10/0.5）；(b) 实测密度比时间史 vs Maxwell 目标 |
| `fig3_stability_map.png` | **稳定边界图**：横轴=Maxwell 密度比（对数），纵轴=存活步数（对数），圆/方=EDM/Guo，×=塌缩（标注起始步），▲=构造宽度臂，◆=tier |
| `fig4_ratio_maintenance.png` | (a) 实测 vs 目标密度比（±15% 接受带）；(b) 伪流幅值 vs 密度比误差 |
| `fig5_tau_lever.png` | **黏性杠杆臂**：每档目标比下伪流 Ma（左轴，log）与密度比维持误差（右轴）vs 弛豫时间 τ，含预注册门线与 τ 判据决策框；×=该 τ 塌缩 |

判决摘要（定性，数字见 `RESULTS.md` §8）：**6/6 tier 未过门**。
- 目标比 10：可长持（tier N 步、longhold 2N 步均未塌缩），密度比维持误差在
  15% 门内；**但伪流幅值越过 0.10 的 FAIL 门**（同时高于 0.05 的 PASS 门）→ 判 FAIL，
  卡点单一且明确（伪流）。
- 目标比 100：EDM 可跑满 tier/longhold 步数但**稳态密度比约为 Maxwell 锁值的 3 倍**（蒸汽
  被压到锁值以下数倍）；Guo 在第 23 步塌缩 → 判 FAIL（维持不住 + 塌缩两类）。
- 目标比 1000：EDM 第 100 步（max\|u\|>0.5c_s）与 Guo 第 16 步（非有限密度）塌缩；
  长持时臂第 156 步塌缩 → 判 FAIL（塌缩）。
- 力离散杠杆（`pressure_tensor`）：**不救**——ratio 10 档虽存活但维持误差更大，其余档
  更早塌缩（诚实负面结果）。
- 构造宽度臂：非单调——ratio 100 档 w=8 反而在第 500 步因 **ψ clamp 触发**（ρ_max 越过
  ρ_cross）而失效，w=4/6 存活；ratio 1000 档三个宽度都塌缩。
- 黏性 τ 杠杆臂（预注册）：三档的 τ 判据决策分别为 `ratio_held_currents_above_pass_gate`
  （低档：维持得住但伪流地板越门）、`dead_at_all_tau`（中档：两门都不达）、`no_stable_run`
  （高档：每个 τ 都塌缩）；中档的密度比误差随 τ 增大明显下降但始终出 15% 门。
- 平面界面对照：给的**不是** Maxwell 锁值——低档蒸汽平台偏高、高档蒸汽平台低数倍，
  与液滴臂同向同量级 ⇒ 偏差来源是格式而非几何/有限域。

## 6. 根因分析（竞争假设 + 判别证据 + 排名）

**观察**：Maxwell 锁值初始化后，离散 SC 格式演化到的密度对比与锁值系统性不符（低档
偏高、高档蒸汽偏低数倍），且伪流幅值随目标比单调上升，直至塌缩。

| 假设 | 机制 | 判别观测 | 判决 |
|---|---|---|---|
| H1 未弛豫（瞬态） | 蒸汽扩散/声学模未衰减完 | 长持时臂（2N 步）：密度比时间史是否仍在漂 | **否证**：长持时与 tier 步数给出同一平台（ratio 10 / 100 两档） |
| H2 有限域蒸汽库不足 | 闭盒内蒸汽质量不够，只能由液滴补给 | 平面界面对照（蒸汽库 ≈ 半域）是否回到锁值 | **否证**：平面臂同向偏离、量级相当（高档蒸汽同样低数倍） |
| H3 界面压力张量离散误差在气相支被指数放大 | ψ(ρ) 在低密度 ≈ √ρ，SC 力低阶各向异性离散 → 有效饱和压偏移 → 蒸汽密度指数敏感 | 偏差是否随目标比单调增大；是否与几何无关；clamp 是否未触发 | **主导**：偏差随比单调增大、几何无关、clamp 未触发、长持时后不收敛；且随目标比增大伪流同步上升 |
| H4 ψ clamp 越界（ρ>ρ_cross 掉出 EOS 定义域） | 稠密相过冲 → ψ=0 → 有效力突变 | 采样尾窗 ρ_max vs ρ_cross | **仅在构造宽度臂 w=8/ratio100 成立**（该档即因 clamp 失效）；tier 档三档均未触发 |
| H5 forcing 方案（EDM/Guo）本身不稳 | Guo 源项在陡密度梯度下失去正性 | 两方案塌缩起始步与存活比的差 | **部分成立**：Guo 的失效比 EDM 早得多（边界图整体左移），说明方案选择只平移边界、不能消除边界 |
| H7 伪流/维持误差是黏性（τ）伪影 | SC 力与黏性应力平衡，τ 决定伪流地板；若某 τ 能压低伪流而不改维持误差，则“伪流 FAIL”是流态陈述而非构造陈述 | 固定 EOS 点/forcing/盒，扫 τ∈预注册格点，看 Ma 与密度比误差是否可分别移动 | **否证**：τ 杠杆臂逐 run 表（`RESULTS.md` §6e）显示伪流地板在整个 τ 格点上几乎不动，低档密度比误差也不随 τ 移动；中档密度比误差随 τ 增大单调下降（可动）但始终在 15% 门之外，高档每个 τ 都塌缩。⇒ 失效是构造/离散层面的，黏性不能救；τ 只移动“格子选出的共存对”，不移动伪流地板 |
| H6 初始化周期接缝伪差（工具自身缺陷） | "上气下水"单界面在周期边界把 ρ_l 接到 ρ_v 一格内 | 接缝跳变与初始 max\|F\| 对比液滴 | **成立且已修**：单界面版初始 max\|F\| ≈ 液滴同宽度的 6 倍；改为居中 slab 后接缝跳变 ~1e-5、max\|F\| 与液滴同量级（pytest 回归门 `test_planar_control_is_periodic_and_not_force_inflated`） |

**下一步判别测量（本单未做，明确留白）**：
1. 在平面臂的**实测平台对**上测离散机械压（均匀场时 F≡0，故 p_mech=p_EOS；因此界面处
   的压力亏损是待测对象）——即直接量化 H3 的"界面压力亏损"，预期其使气相支饱和压偏移；
2. 采用更高阶各向同性伪势力模板（如 `multiphase_edm` 描述的 6 阶模板）重跑边界图，
   检验 H3 的"离散各向异性"通道；
3. 以有效温度重标定（YB/YS 类做法）替代输入 T，量化"输入 T ≠ 有效 T"偏差。

## 7. honest-FAIL 机制

- 塌缩不是异常，而是**结构化结果**：`run_probe()` 永不因物理发散抛异常，返回
  `{stable: false, collapse_step, error(机制描述), hist(有限部分轨迹)}`；
  `classify()` 给出 FAIL + 原因字符串。
- 起始步**逐分辨率到步**（每步做一次 O(N) 规约检查有限性/正性，成本 ~10%）；
  避免"采样间隔掩盖起始时刻"。
- 不做事后调参重试：门限、初值、步数均在出数前冻结（`PREREG` 随产物落盘）；
  负面杠杆（`pressure_tensor`、宽度臂）照实登记为负面结果。
- 机器可核：`tests/test_eos_maxwell_stability.py::test_honest_fail_returns_structured_record`
  锁定该契约（越界构造必须返回不可用记录且无派生数字）。

## 8. 消费链映射（≥3，机器可核 `consumers.json`）

| 消费者 | 消费的字段 | 支撑的决策 |
|---|---|---|
| **GAP-LBM-MCMP-POISEUILLE** 高密度比路判据（`docs/REPORT_qr_pilot_a.md`、`benchmarks/pending/two_phase_poiseuille`） | `verdict.tiers[*]`（可达比包络 + 伪流地板）、`verdict.boundary[*]`（塌缩起始 vs compare 比）、`verdict.longhold[*]`、`verdict.tau[*]`、**`CRITERION.md` D1–D4**（开跑判据 + 格式偏差修正）、`fig3`/`fig5` | 两相 Poiseuille 在给密度比下能否开跑、用哪个 forcing/τ、伪流噪声地板多少、对比结果须扣多大的格式共存偏差 |
| `benchmarks/verified/laplace_pr_eos`（σ 自测） | `maxwell_tiers`（锁值交叉核对）、`tiers[*].ma_max`、`longhold[*].ratio_error` | 该构造上 Δp–1/R 拟合出的 σ 能否与伪流噪声分离；构造在拟合期间是否守住对比 |
| `src/tensorlbm/sc_eos_coexistence.py`（G3 台账 Maxwell 求解器） | `controls[*].ratio / rho_v / ratio_error`、`boundary[*]` | "Maxwell 对 = 模拟不动点"这一假设能否直接用于审计 SC 耦合声明，还是必须按比逐档实测 |
| `src/tensorlbm/multiphase_benchmarks.py`（`static_droplet` 基准） | `tiers[*].ma_max / u_max`、`fig2` | 其静态液滴伪流列能否外推到高密度比构造 |
| `src/tensorlbm/multiphase.py`（EOS 伪势工厂） | `maxwell_tiers`、`boundary[*]`、`schemes[*]` | EOS 伪势路可宣称的密度比上限与推荐力离散 |

（`consumers.json` 内 5 条均带 anchor 三元组：路径 + 必含字符串；pytest
`test_consumer_anchors_exist` 机器核验，防"映射写了但链断"。）

## 9. 适用范围与局限

- **维度/离散**：2D D2Q9、float32、周期边界、τ=1、G=+1；**3D 未测**（D3Q19/D3Q27 的
  结论不可由此外推）。
- **物性自由度**：单组分 SCMP，无独立黏度比；不涉及两相各自黏度/密度独立控制。
- **平衡口径**：密度比用"液核均值 / 远场均值"，远场取 r ≥ 0.8·L/2 环带；平面臂取
  中央 1/4 液平台与壁侧蒸汽膜。口径写在代码里（`probe_lib.radial_profile` /
  `planar_regions`），未做口径选购（两口径同向结论）。
- **EOS 参数面**：只用库的 YS2006 默认参数（PR），未扫 a/b/R/α(T)。
- **"可达比"是数值可达性，不是物理可信度**：ratio 100 档"能跑"但维持误差 ~3 倍，
  因此本探针把它判 FAIL；不得引用为"支持密度比 100"。
- **伪流门（0.05/0.10）是预注册阈值**，非文献统一标准；引用时须连同门值一起引用。
- **τ 杠杆臂的覆盖度**：为保住 tier 门与杠杆臂可比，τ 臂沿用 N=20000。τ 越小黏性时标
  R²/ν 越长，部分低 τ 档在 N 内未完成黏性弛豫——`tau_lever_summary` 会标出
  `not_relaxation_resolved`（coverage<2），这些运行只当否定证据用；若 Ma 门只在未弛豫
  运行里被"清过"，决策串降级为 `ma_pass_only_in_unrelaxed_runs`（预注册 caveat）。
- **拆分作业**：合并前各作业的产物互不覆盖；若只跑其中一个作业而不合并，`RESULTS.md`
  的相关臂会为空，`CRITERION.md` 的 τ 列显示 `arm_not_run`（不是"通过"）。

## 10. 复现

```bash
# 全战役（单进程 CPU 约 2 h；本战役拆 6 个进程并行跑，见下面的拆分写法）
cd benchmarks/eos_maxwell
python run.py --mode all --out out --device cpu \
    --steps 20000 --boundary-steps 4000 --longhold-steps 40000 \
    --widths 4,6,8 --control-width 4 --control-steps 40000
# 快速自检（CI 规模，<15 s）
python run.py --mode pilot --out /tmp/eosmax_pilot --device cpu --steps 2000
# 战役可拆成两个集群作业跑（各自的 out/results.json），再用 merge_runs.py 合并：
#   job A: --mode tiers,boundary,controls,widths,longhold,schemes,repeats
#   job B: --mode tau,robust,threads
python merge_runs.py --out out/results.json /path/jobA/out/results.json /path/jobB/out/results.json
# 图件 + 生成 RESULTS.md / CRITERION.md
python plot_figures.py            # 读 out/results.json，写 figures/*.png、RESULTS.md、CRITERION.md
# 门（pytest）
cd ../.. && python -m pytest tests/test_eos_maxwell_stability.py -q
```

**构件过期门**：`test_results_artefact_matches_tree_code_hashes` 检查
`out/results.json` 里记录的 `probe_lib.py`/`run.py` sha256 是否等于工作树里文件的
sha256。改了代码不重跑战役，该门立刻失败——防止别人引用与代码不同步的表。
`merge_runs.py` 合并时同样拒绝 prereg/代码哈希不同源、或 tag 重复的产物，并把每个
来源作业的 `generated_utc`、`wall_s_total`、各臂条数、该来源文件自身的 sha256 写进
`provenance.sources`（`maxwell_tiers` 的 bisection **耗时**按构造因作业而异，不算物理量，
合并不因此拒绝；构造值不同则拒绝）。本战役实际拆 6 个驱动进程并行跑
（`--mode tiers` / `longhold` / `widths,schemes` / `tau` / `controls,repeats` /
`boundary,robust,threads`），再合并成一个 `out/results.json`。

产物：`out/results.json`（全轨迹 ~5 MB，可增量续跑 `--resume`）、`RESULTS.md`（机器
生成表）、`figures/*.png`。集群执行走 `run_cluster(stage_dir=<worktree>, repo=TensorLBM)`；
节点无 matplotlib，图件在控制面由 `plot_figures.py` 生成。

## 11. 与任务书写路径的偏差（自主方法声明）

任务书路径为 `tests/benchmarks/eos_maxwell/`。本仓布局是 `benchmarks/` 存战役工件、
`tests/` 存 pytest（`pyproject.toml: testpaths=["tests"]`），且无 `tests/benchmarks/`
目录先例。故按仓约定落为：

- 探针与工件：`benchmarks/eos_maxwell/`（与 `laplace_pr_eos`、`d3q27_re200_stability_probe.py`
  同类目）；
- pytest 门：`tests/test_eos_maxwell_stability.py`。

两处合计即任务书所指"探针 + pytest + 图件"。若归档要求严格路径，可整体移动到
`tests/benchmarks/eos_maxwell/`（仅需改 `tests/test_eos_maxwell_stability.py` 的
`_BENCH` 与本 README 路径）。

## 12. 参考

- Yuan & Schaefer (2006) *Phys. Fluids* 18 042101（EOS 参数表与伪势定义）。
- Shan & Chen (1993) *Phys. Rev. E* 47 1815；(1994) *Phys. Rev. E* 49 2941（SC 伪势与
  两相分离）。
- Kupershtokh (2004)；Li, Luo, Li & Feng (2012)（EDM forcing）。
- Guo, Zheng & Shi (2002) *Phys. Rev. E* 65 046308（源项 forcing 与半移位速度）。
- Maxwell 等面积构造：见 `src/tensorlbm/sc_eos_coexistence.py` 模块文档（含 v=1/ρ 与
  dρ 口径差异说明）。