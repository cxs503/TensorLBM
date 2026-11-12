# CASE-MCMP-CONTACT-ANGLE — 壁面接触角的 Young 定律（TensorLBM / SC 伪势壁面）

**锚点**：CASE-MCMP-CONTACT-ANGLE（第 1 轮·任务榜·算例池，mission 15470）
**判据（预注册，动手前定）**：`|θ_measured − θ_Young| < 10°`（每例），失败不得放宽判据、不得丢例。
**判定：honest FAIL**（ref = Young 定律设定角；got = 同一算例的实测角；6/6 例超差，最大 90.00°）。

---

## ① 问题 + ref 出处

**物理**：静止座滴在固体壁面上的平衡接触角由 Young 方程给出

$$\gamma_{lv}\cos\theta = \gamma_{sv}-\gamma_{sl}\qquad(\text{Young, 1805})$$

在 Shan–Chen 伪势（SCMP）模型里，壁面润湿用一个壁面伪势 $\psi_w$ 注入壁面节点的相互作用力，
库内把"给一个目标角"实现成一条**线性映射**（`TensorLBM/src/tensorlbm/wetting/__init__.py:164-184`，
以及 `tests/benchmarks/wetting_contact_angle/wetting_bench.py:145-174`）：

$$\cos\theta_{\rm Young}=\frac{\psi_w-\psi_v}{\psi_l-\psi_v},\qquad \psi_w = f_w\,\psi_l$$

本单要检验的假设 = **这条映射能不能当"接触角设定器"用**（10° 容差）。

**ref 出处与"零手抄"等级（如实声明）**：

| 项 | 出处 | 本单的取数方式 | 证据等级 |
|---|---|---|---|
| Young 方程形式 | T. Young, *Phil. Trans. R. Soc. Lond.* **95**, 65–87 (1805) | 只用作方程形式；无纸面数值 | REAL（方程）+ 页码级核验 **WITHHELD**（离线无网络，原文不可逐页核） |
| $\cos\theta=(\psi_w-\psi_v)/(\psi_l-\psi_v)$ | SC 伪势壁面润湿关系，库内既有实现 + Shan & Chen, *Phys. Rev. E* **47**, 1815 (1993)、**49**, 2941 (1994)；Huang, Thorne, Schaap & Sukop, *Phys. Rev. E* **76**, 066701 (2007)（SC 模型接触角的近似关系） | **不手抄任何数字**：$\psi_l,\psi_v$ 由库自己算（$\psi(\rho)=1-e^{-\rho}$ 在共存密度处求值），$f_w$ 由库函数反解 | REAL（可执行、可复算）；2007 论文表页 = **WITHHELD**（离线不可核） |
| 共存密度 $\rho_l,\rho_v$ | 库内 `RHO_L_DEFAULT/RHO_V_DEFAULT`（`multiphase`） | 由库读出 | REAL |

> 结论先给：**"Young 关系 = 接触角设定器"这一假设被本算例否证**。目标角 110° 甚至**无法用
> Young 关系表达**（需要 $\psi_w<0$，可达上限 $\theta_{\max}(\psi_w\ge0)=101.97°$，由库现算）。

## ② 方法与实现（复用库能力，不另起炉灶）

一条链，全部走库内既有能力（无新物理内核）：

1. **步进器**：`TensorLBM/src/tensorlbm/contact_line/engine.py` `WallStepper`（半程反弹 + 壁 $\psi_w$ + SC 单组分力）；
2. **初始帽**：`sessile_drop_state(radius, height=-R\cos\theta)` —— 近乎精确的圆缺（几何无误置）；
3. **测角**：`tests/benchmarks/wetting_contact_angle/contact_angle.py` 三估计器
   （主估计器 = 液相面积 + 顶点高的**圆帽反演**，不碰壁面耗散层；circle/band 作交叉核对；
   库约定 $\theta_{\rm lib}=180°-\theta_{\rm phys}$ 在 `contact_angle.py:74` 处显式换算）；
4. **工况编排**：`run_all.sh a|b1|b2` + `assemble.py`（判据、hold-out 计划）+ `make_report.py`（图表）；
5. **测量链自证**：`assemble.py:estimator_control()` 用**解析圆帽**（同半径、同 tanh 界面宽）过同一估计器，
   偏差 −1.11…−0.78°（两网格 6 点），证明观测到的角漂移不是测量方法造成的。

阶段设计（预注册）：**A** 用 Young 关系设定 70/90/110°（两套网格）；**B1** 不设定角、扫
$f_w\in[0.30,0.80]$ 得实测 $\theta(f_w)$ 标定曲线（粗网格）；**B2** 用 B1 曲线的分段线性反解，
在**细网格**上 hold-out 复验"标定式处方"；**C** 同一 $f_w$ 两个初始帽（70° / 110°）测亚稳固有不唯一性。

## ③ 参数表（D2Q9，无量纲）

| 参数 | 值 | 说明 |
|---|---|---|
| 网格 coarse | nx=96, ny=64，壁 2 行，R=13 | 8000 步/例 |
| 网格 fine | nx=144, ny=96，壁 2 行，R=20 | 12000 步/例 |
| 碰撞 | BGK，$\tau=1.0$ | $\nu=(\tau-0.5)/3=0.16667$ |
| SC 耦合 | $G=5.0$（吸引） | 单组分伪势 |
| 伪势 | $\psi(\rho)=1-e^{-\rho}$ | |
| 共存密度 | $\rho_l=1.957$, $\rho_v=0.1596$ | 库默认（已由库的共存测量标定） |
| 伪势 | $\psi_l=0.858718$, $\psi_v=0.147516$ | 现算，非手抄 |
| 表面张力 | $\sigma=0.05611$（lattice） | 库内 Laplace 标定值 |
| 润湿旋钮 | $f_w=\psi_w/\psi_l\in[0.30,0.80]$ | 扫描自变量 |
| Young 可达角上限 | $\theta_{\max}(\psi_w\ge0)=101.97°$ | 现算 |
| 无量纲数 | $Bo=\Delta\rho\,g L^2/\sigma=0$（$g=0$，无重力→Young 定律应严格成立）<br>$Oh=\mu/\sqrt{\rho_l\sigma L}=0.273$（$\mu=\rho_l\nu=0.3262$, $L=13$）<br>$Ca=0$、$Ma=0$（静止） | 全部零手抄可复算 |
| 界面宽 | tanh, w=1.5 格 | |

## ④ 对照表（ref vs got，逐项打勾叉）

**A 阶段（Young 设定角，ref = 设定角）**

| 网格 | $f_w$ | ref $\theta_{\rm Young}$ | got $\theta_{\rm LBM}$ | 差 | <10°？ |
|---|---|---|---|---|---|
| coarse | 0.4551 | 70.0 | 111.39 | **+41.39** | ✗ |
| coarse | 0.1718 | 90.0 | 180.00 | **+90.00** | ✗ |
| coarse | —— | 110.0 | OUT_OF_RANGE（需 $\psi_w<0$） | n/a | ✗（不可表达） |
| fine | 0.4551 | 70.0 | 117.70 | **+47.70** | ✗ |
| fine | 0.1718 | 90.0 | 180.00 | **+90.00** | ✗ |
| fine | —— | 110.0 | OUT_OF_RANGE | n/a | ✗ |

**B1 标定扫描（粗网格，$f_w$ 为自变量）**——Young 预测 vs 实测，系统性同号：

| $f_w$ | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.80 |
|---|---|---|---|---|---|---|---|---|---|---|
| $\theta_{\rm Young}$ | 81.09 | 77.57 | 74.01 | 70.37 | 66.65 | 62.83 | 58.87 | 54.73 | 50.37 | 40.67 |
| $\theta_{\rm LBM}$ | 161.47 | 143.49 | 127.94 | 112.90 | 98.01 | 81.40 | 62.68 | 36.30 | NO_INTERFACE | NO_INTERFACE |
| 差 | +80.38 | +65.91 | +53.93 | +42.53 | +31.36 | +18.57 | **+3.81 ✓** | −18.43 | — | — |

实测律（粗网格，最小二乘，$rms(\cos)=0.074$）：$\cos\theta = 5.032\,f_w - 2.575$，
**斜率是 Young 映射的 4.17 倍**，$\theta=90°$ 的中性壁在 $f_w^{*}=0.5118$，而 Young 关系说 $0.1718$。
细化后（补充扫描，$R=20$）：$\theta$=146.21(0.35)/132.03(0.40)/118.92(0.45)/105.73(0.50)/91.76(0.55)/61.31(0.65)/41.35(0.70)，
中性壁漂到 $f_w\approx0.555$。→ **"中性壁 = 汽相伪势"这一条物理假设本身不成立**：
平壁上一行壁 $\psi_w$ 与整条扩散界面的镜像权重不对称，中性点必然高于 $\psi_v$。

**B2 hold-out（标定曲线反解 → 细网格）**

| ref 目标角 | $f_w$（分段反解） | got | 差 | <10°？ |
|---|---|---|---|---|
| 70 | 0.580446 | 89.12 | +19.12 | ✗ |
| 90 | 0.524112 | 106.01 | +16.01 | ✗ |
| 110 | 0.459730 | 121.44 | +11.44 | ✗ |

**汇总（artifact 判定块）**：`verdicts.young = FAIL 0/6, max|Δ|=90.00`；`verdicts.calibrated = FAIL 0/3, max|Δ|=19.12`。

## ⑤ 收敛性与参数敏感性（两点以上，均现算）

| 敏感性轴 | 证据 | 量级 |
|---|---|---|
| **网格分辨率**（同 $f_w$=0.4551） | coarse 111.39° vs fine 117.70° | **6.31°**（artifact `grid_sensitivity`） |
| 网格分辨率（标定曲线整体） | 中性壁 $f_w^*$：0.512(coarse) → 0.555(fine)；$f_w$=0.65 时 36.30° → 61.31° | **24.99°**（湿地端最敏感，趋于膜） |
| **初始帽（亚稳/钉扎）** | 同 $f_w$=0.52：init=70° → 96.03°；init=110° → 107.14° | **11.11° split** |
| 估计器（非物理） | 解析圆帽过同一管线：bias −1.11…−0.78° | ≤1.11°，故上表漂移非测量伪差 |
| 时间收敛 | 每例尾 1/4 段角漂移 <0.01…0.002°（`drift_deg`） | 已收敛（判据 1.0°） |

**注意**：亚稳 split（11.11°）**已超过本单 10° 容差本身**——即使把处方做到完美，只要初始帽不固定，
可复现性下限就在 11° 量级。这是判据设计层面的硬约束，不是"再算细一点"能解决的。

## ⑥ 失败模式与工件

**失败定位（fail_stage）**

- `fail_stage = stage_a_young`：主判据（Young 关系当设定器）6/6 超差，最大 90.00°（$\theta_{\rm Young}=90°$ → 实测 180°，即完全去润湿）。
- `failure_reason`（根因，定量）：① Young 线性映射与真实壁面能量关系的**斜率差 4.17 倍**、中性点 $f_w^*$ 偏 3 倍
  （0.1718 vs 0.5118）；② 分辨率依赖：同 $f_w$ 下角随网格漂 6.3°（湿地端 25.0°）；
  ③ 接触线钉扎/亚稳：同 $f_w$ 不同初始帽裂 11.11°，超过 10° 容差本身。
- `fail_stage = stage_b_verify`（二级假设）：按实测律标定后跨网格迁移仍 0/3 通过（最大 19.12°）→
  **"粗网格标定可迁移到细网格"被否证**；标定必须同分辨率、同初始帽。

**适用/不适用**

- 可用：$\sigma$（Laplace）与几何量测量链；同分辨率同初始帽的**标定式**接触角处方（库内 `invert_calibration`）。
- 不可用：把 `wall_psi_from_contact_angle()` 当"接触角设定器"直接调用（本算例已证系统性偏差 40–90°）。
- 量程限制：$f_w\ge0.70$ 粗网格、$f_w\ge0.80$ 细网格 出现 `NO_INTERFACE`（强润湿端摊成膜，圆帽反演失效）。

**工件（全部在 worktree 内，数字程序化注入）**

| 工件 | 路径 |
|---|---|
| 判定 artifact（所有表/图的唯一数据源） | `artifacts/CASE-MCMP-CONTACT-ANGLE/contact_angle_sweep.json` |
| 逐算例原始记录（20 例 + 亚稳 2 例 + 补充细网格扫描 10 例） | `artifacts/CASE-MCMP-CONTACT-ANGLE/cases/*.json` |
| 图 | `artifacts/CASE-MCMP-CONTACT-ANGLE/figures/{prescribed_vs_measured,calibration_curve,contours_young}.png` |
| 工装生成的报告（表由 JSON 渲染） | `artifacts/CASE-MCMP-CONTACT-ANGLE/GENERATED_REPORT.md` |
| 一键复现 | `artifacts/CASE-MCMP-CONTACT-ANGLE/repro.sh` |

## 生态摩擦（如实记）

1. **库内注释把"设定角"说成能力**：`wetting/__init__.py:164` `wall_psi_from_contact_angle` 三行 docstring 直接写
   "Convert target contact angle to wall pseudopotential"，无任何偏差警告。下游照用即得 40–90° 偏差
   （`docs/P143W2_N01_SC_INTERFACE_TENSION.md` 已独立踩过同一坑，最大 100.9°）。**建议**在库侧加显式告警/指向标定曲线（本单不擅自改库语义，仅记摩擦）。
2. **控制面 shell 120s 收割**：单例 50–100 s、6–10 例并行必然超过 120 s，必须 `setsid nohup` 后台 + 轮询；
   `run_cluster` 的 `stage_dir` 可直接用工作树，但本例 CPU 侧足够，未占用集群。
3. **`results/` 与 `runs/` 在 .gitignore**：工装产物默认不入库 → 交付必须手动搬进 `artifacts/`，否则"跑了但无工件"。
4. **判据与物理下限冲突**：10° 容差 < 亚稳 split 11.11°，算例池在设计容差时未计入钉扎下限。