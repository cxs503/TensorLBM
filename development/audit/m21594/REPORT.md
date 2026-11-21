# 交差审计 m21594：XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM（m21558）

**结论先行**：被判交付 m21558（提交 `1b685d33c`，master `481ac88da` Merge #2475）判定 **L2 / honest-FAIL**。
L2 成立（真代码 + 可复算工件 + 12 单测），L3 不成立（主求解器无 caller），L4 **不可判**：
基准裁判表 `tensor_benchmark_cases` 对该 anchor **0 行**（无 ref/metric/tolerance 可判），
而唯一可独立复算的对照（`analytic.solve_layered` 精确分层解）实测 **u_peak_err=177.78%、
ε∞=253.30%、ε_L2=264.77%**，全部远超 15% owner 门（也远超 3% 出版门）。
交付方 STATUS.md 自认 "FAILS acceptance criteria / Do not mark as verified"，**诚实性成立**，故为 honest-FAIL 而非虚标。

## ① 案例介绍（物理背景 / 几何 / 参数）

两组分 Shan-Chen（MCMP）平面槽道流动：低黏组分 A（τ=0.8, ν=0.1）与高黏组分 B（τ=1.2, ν=0.2333）
在 G=2.0 的排斥型交叉力下分层共存，受 x 向体积力 g_x=1e-4 驱动，稳态应与**分层 Poiseuille 精确解**
（ν_j u_j''=-g 加界面速度/剪应力连续，`analytic.solve_layered` 的 2N×2N 稠密 LU 精确解）逐点吻合。

| 项 | 交付取值 |
|---|---|
| 网格 | ny×nx = 32×4（H=30 lu，上下壁半程反弹） |
| 组分 | A: τ=0.8, ν=0.1000；B: τ=1.2, ν=0.2333 |
| 耦合 | G=2.0，ψ=ρ0(1-e^{-ρ/ρ0}) |
| 强迫 | g_x=0.0001，方案 EDM（另含 Guo / velocity-shift 两支） |
| 步数 | 2000（黏性扩散时标 H²/ν = 9000.0 步） |

## ② 怎么做的（审计流程，独立于交付方的证据链）

1. **取证**：把交付提交 8 个文件只读快照进本审计包 `audit/m21594/snapshot/`（sha256 见 `evidence.json` E1），
   原始工作树 `/srv/tensor/agent/work/m21558` 仅作出处记录；求解器与解析解按**文件路径** hermetic 载入，
   绕开 `pip` editable 安装（否则会静默 import 到别的工作树 m21384）。
2. **三件套核验**：run_artifacts / benchmark 裁判行 / repro_cmd 逐项落真实指针（E2）。
3. **数值复算**：从交付 `quick_validation_results.json` 的 profile 数组按交付自身定义（`run_quick_validation.py:91`，margin=2）
   复算 u_peak_err / ε∞ / ε_L2，与存储值**逐位吻合**（差 0.0e+00）。
4. **独立诊断**：层序口径（D1）、解析起动瞬态参考（D2a）、稳态检验 2000/20000 步（D2）、
   单组分 G=0 对照（D3）、质量与壁格账（D4）。D2/D3 用**被判交付的同一个求解器**跑，经计算门递交（job cj3554）。
5. **判级落表**：`tensor_maturity_verdicts` INSERT（`verdict_m21594.sql`），evidence 八键全为真实指针。

## ③ 计算结果（数字表，程序化注入）

**交付数值复算（唯一裁判口径 = 交付自带的分层解析解）**

| 量 | 交付存储 | 本审计复算 | 独立重算解析峰 | 15% 门 | 3% 出版门 |
|---|---|---|---|---|---|
| u_peak_err | 177.78% | 177.78% | u_pk: 0.200004 vs 0.072000 | FAIL | FAIL |
| ε∞（profile） | 253.30% | 253.30% | — | FAIL | FAIL |
| ε_L2（profile） | 264.77% | 264.77% | — | FAIL | FAIL |

**基准裁判行**：`tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM'` → **0 行**。
同域其他 LBM 槽道 anchor 有 `CASE-SPH-POISEUILLE-MORRIS|passed, GAP-LBM-MCMP-POISEUILLE|pending, XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET|passed`，本 anchor 未登记 ⇒ L4 无裁判可打。

**诊断腿（同一个交付求解器，逐步数）**

| 步数 | u_peak_num | u_peak_err | ε∞ | u_max/(g·t) |
|---|---|---|---|---|
| 2000 | 0.2000 | 178.27% | 244.35% | 1.000 |
| 20000 | not_run | — | — | — |
| 单组分 G=0 对照 | 20000 | 0.2000 | 77.98% | 136.71% | 1.000 |

**解析起动瞬态参考（ν=0.1，H=30，g=1e-4，t=2000）**：正确黏性解峰值 **0.0994**、
壁面制动层 √(νt)=14.1 lu（与 H/2=15 同量级）；
交付解峰值 **0.2000 ≈ g·t = 0.200**，
内部跨流标准差仅 5.36e-03（几乎完全平顶）。

**单组分阴性对照（关键）**：同一求解器 G=0、B 置真空、仅 ρ_A=2/τ=0.8（ν=0.1）时，
2000 步的 u_max/(g·t) = 1.000（u_peak=0.2000），
而 nu=0.1 的正确起动解同期峰值 0.0994 ⇒ **基础求解器在单组分下也不产生黏性制动**，
故交付报告「有效黏度需标定 / 需 10000+ 步 / 界面弥散」三条根因均被否证（缺陷在强迫尺度/黏性项/壁面耦合，与两组分耦合力无关）。

**层序口径**：解析解 `nus=[ν_A,ν_B]` 把 ν=0.1 贴 y=0 壁；数值初始化（`two_component_sc.py:131-132`，注释写 "A in bottom layer"）
实际把 A 放在 **y>15 一侧**（t=0 实测 ρ_A(y=0.5)=0.10、ρ_B(y=0.5)=2.00）。
镜像口径下解析解自身相对偏差 **22.36%** —— 这是与数值无关的系统性口径误差。

**壁格密度**：y=0.5 处 ρ_A = 584（名义 2.0，×280）。

## ④ 结果图

三幅图：**(a)** 交付速度剖面 vs 解析分层解（含镜像口径）vs 解析起动瞬态；
**(b)** u_max(t) 轨迹 vs 自由加速线 u=g·t（含单组分对照）；**(c)** 交付密度剖面（壁格爆量）。
见附图 `fig_m21558_audit.png`（数据全部来自 evidence.json / 交付 profile 数组，无一手抄）。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **报告根因归因不成立**：STATUS.md 把失败归因于「有效黏度未标定 + 需要 10000+ 步 + 界面弥散」。
   但解析起动解显示，ν=0.1 的**正确**求解器在 2000 步就该出现 ~14 lu 的壁面剪切层、峰值仅 0.0994；
   交付解却是 u≈g·t 的平顶（内部 σ=5.4e-03），即**动量扩散在 2000 步内几乎未发生**。
   这不是「黏度标定」，而更接近强迫尺度/黏性项或界面-壁面耦合的实现缺陷 —— 加步数不会自愈（见 20000 步诊断）。
2. **度量定义未预注册**：入口脚本 margin=2、campaign 脚本 margin=3，同一物理量两套定义
   （m=2 → 253.30%，m=3 → 见 evidence.recompute.metric_definition），先跑后挑口径的口子存在。
3. **repro 不 hermetic**：`run_quick_validation.py:10` 硬编码 `/srv/tensor/agent/work/m21558/TensorLBM/src`；
   把该路径换成不存在后，`import tensorlbm...two_component_sc` 直接 ModuleNotFoundError（解析到 m21384 的 editable 安装）。
   另 `two_component_sc.py:138` 用 `torch.randn_like` 破对称且**无 seed**，故 ε_L2 交付 264.77%
   vs 复跑 264.77%，非比特可复现。
4. **12 个单测不构成物理证据**：全部为 equilibrium/ψ 的代数性质、step_count 递增、ux_max 增大、三支强迫不崩，
   **没有一条**把求解器输出与解析解比对并断言 15%。测试绿≠算对。
5. 局限：本审计不改交付代码（交差单非重做），诊断腿的 2000/20000 步对照只用 EDM 一支；未做网格无关性（nx=4 极窄，x 向周期性下影响有限但未量化）。

## ⑥ 误差判定（逐项显式回答「是否 <3%」）

| 关键量 | 参考 | 交付值 | 本审计复算 | 是否 <15% | 是否 <3% |
|---|---|---|---|---|---|
| u_peak_err | 分层解析 u_pk=0.072000 | 0.200004 | 177.78% | **否** | **否** |
| ε∞ | 分层解析剖面 | — | 253.30% | **否** | **否** |
| ε_L2 | 分层解析剖面 | — | 264.77% | **否** | **否** |
| 层序口径偏差 | 镜像解析解 | — | 22.36% | **否** | **否** |
| 瞬态峰值（2000 步） | 起动级数解 0.0994 | 0.2000 | 169.93% | **否** | **否** |

**基准裁判结论**：该 anchor 在 `tensor_benchmark_cases` 无行，L4 无裁判可打；唯一可独立复算对照全部 FAIL。
本项目关键量 **无一项 <3%，也无一项 <15%** —— honest-FAIL，级别 L2，不消账。
