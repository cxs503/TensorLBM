# XVAL-LBM-YML-DRIVEN-XML-TEMPLATE-PIPELINE — 第 2 轮（调研结合）验收报告

- 锚点：`XVAL-LBM-YML-DRIVEN-XML-TEMPLATE-PIPELINE`（feature_ledger #482，source=`je-santos/MPLBM-UT`）
- 仓库/分支：TensorLBM @ `agent/m18791`，mission 18791
- 验收判据（总账）：对标误差 **< 15 %**（owner 精度标准）
- 机器复核工件：`artifacts/xval_yml_xml_pipeline/report.json`（由 `scripts/xval_yml_xml_pipeline/run_all.py` 程序化生成，判据数字零手抄）
- 复现命令：`python scripts/xval_yml_xml_pipeline/run_all.py`（本地 ~85 s，4 算例）与 `python -m pytest tests/xval/test_yml_xml_pipeline.py -q`（26 passed，51 s）
- 结论：**PASS** —— 最差渗透率相对误差 **0.411 %**，最差速度剖面 L2 相对误差 **0.281 %**，4/4 算例 + XML 转义探针全过（判据 15 %）。

---

## ① 问题设定 + 基准出处

### 1.1 被对标的能力（OSS 侧事实）

`je-santos/MPLBM-UT` 的 `yml-driven-xml-template-pipeline`（`oss_solutions` 原文，入库于
TensorPlatform `docs/oss-inventory/MPLBM-UT.md` §OSS-SOL，行 334–338）：

> 每个算例目录一个 `input.yml` + 一个 runner 脚本；`create_palabos_input_file` 用 f-string 模板
> 把 yml 展开成求解器 XML，再写 mpirun bash 脚本并 subprocess 提交；后处理脚本从固定文本标记
> （`'Pressure difference ='`）解析输出。
> `files: src/python/mplbm_utils/create_palabos_input_file.py:93-250,
> parse_palabos_output.py:4-16, examples/steady_state_relperm_w_argparse/input/2_phase_sim_input.xml:1-57`
> `conventions: 配置层级 geometry/init/fluids/output 四段；输出固定文本 'Pressure difference = x' 作为解析锚点`
> `transfer: fits`

即：**YAML 是唯一真源，XML 是求解器的输入契约，文本标记是后处理契约**；求解器本体（Palabos C++）
不在仓内，管线层没有任何数值门禁（清单同一文档 §"CI"：`not_supported`，全树 162 blob 无 `.github/`）。

### 1.2 本仓要回答的问题（XVAL 问法）

TensorLBM 侧是否存在等价管线？答案是 **部分**：`src/tensorlbm/config_io.py` 只有
「YAML/TOML → dataclass 配置」的一层（209 行，含环境变量覆盖），**没有**「配置 → 求解器输入文档 →
解析回 → 驱动求解 → 文本标记后处理」的闭环，也没有任何「渲染出的文档确实能被求解器正确消费」的
不变式断言。因此本轮的 XVAL 命题是：

> **把 MPLBM-UT 的管线形状（yml→xml→solver→text markers）在 TensorLBM 内重实现，并证明
> 「经渲染-解析往返后驱动出的解」与原解析解一致（误差 < 15 %）。**

### 1.3 基准（external anchor）

| 项 | 内容 | 出处 |
|---|---|---|
| 判据量 | 渗透率 `K`（lattice units），平面 Poiseuille | Poiseuille (1840) 精确解 |
| 解析式 | `K = H²/12` | Succi, *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond* (2001), ch. 5（halfway bounce-back 壁面位于**半链**处，故 `H` = 流体行数） |
| 剖面 | `u(s) = g/(2ν)·s·(H−s)`，`s = y − 1/2` | Krueger et al., *Lattice Boltzmann Method: Principles and Practice* (2017), §5.3（forced Poiseuille + halfway BB） |
| 力项格式 | Guo forcing（二阶） | Guo, Zheng & Shi, *Phys. Rev. E* 65 (2002) 046308 |
| 边界 | halfway bounce-back | 同上 Krueger §5.3；Ladd (1994) |

> 选 `K = H²/12` 作判据的理由：它是**精确**参考（无离散化/实验误差混入），且正是 MPLBM-UT 这类
> 「图像→几何→渗流」管线的目标物理量，误差可直接归因于我们的管线+求解器。

### 1.4 前轮（m18583）教训与本轮的差别

前轮 honest-FAIL，`fail_stage`/`failure_reason` 双缺。本轮的执行纪律：**先跑通一条可复现的
数值链（yml→xml→解→对比）再谈方法学**，把门禁写进脚本返回值（`all_pass` 为假则 `exit 1`），
并在 `report.json` 中留存每个算例的原始 `K`、`K_ref`、迭代数、残差——避免「无数据结论」。

---

## ② 方法与实现

### 2.1 管线五段（`src/tensorlbm/yml_xml_pipeline.py`，489 行）

| 段 | 函数 | 说明 | 相对 MPLBM-UT 的加固点 |
|---|---|---|---|
| 1. 载入 | `load_spec` (L77) | `yaml.safe_load` → dict | 拒绝非 mapping 顶层 |
| 2. 校验 | `validate_spec` (L87) | 四段 schema + 物理一致性 + **Mach 门禁** | 原实现无校验：`omega` 越界、`ny` 与 `solid_rows` 冲突、`out_folder` 路径穿越都会静默产出错误算例 |
| 3. 渲染 | `render_xml` (L211) | `xml.etree.ElementTree` 输出 `case/geometry/init/fluids/output` | 原实现用 f-string 直拼（`create_palabos_input_file.py:209-241`），值含 `&`/`<`/引号即产坏文档；此处走属性转义 |
| 4. 解析回 | `parse_xml` (L248) | ET 解析 → 再次 `validate_spec` | 新增 **round-trip 不变式** `parse_xml(render_xml(s)) == s`（原实现只有单向生成） |
| 5. 驱动+后处理 | `run_channel` (L330)、`format_run_log`/`parse_run_log` (L300/L318) | D2Q9/BGK + Guo forcing + halfway BB；输出固定标记 `# iters = / # permeability = / # u_peak =` | 原实现后处理硬编码 `'Pressure difference ='` 且无断言；此处标记解析失败即抛 `SpecError` |

渲染示例（`artifacts/xval_yml_xml_pipeline/cases/ch_ny24_w1p7.xml`）：

```xml
<case version="1.0" generator="tensorlbm.yml_xml_pipeline">
  <geometry>
    <size nx="4" ny="24" />
    <per x="true" y="false" />
    <solid_row index="0" />
    <solid_row index="23" />
  </geometry>
  <init><fluid rho="1.0" /></init>
  <fluids><omega value="1.7" /><force x="1.4580000000000002e-05" /></fluids>
  <output><run out_folder="ch_ny24_w1p7" it_max="300000" convergence="1e-09" /></output>
</case>
```

### 2.2 求解器方程（`run_channel`，L330–L430）

- 宏观量（Guo 半步力修正）：`ρ = Σ_i f_i`，`ρu = Σ_i c_i f_i + F/2`
- 平衡态：`f_i^eq = w_i ρ [1 + (c_i·u)/c_s² + (c_i·u)²/(2c_s⁴) − u²/(2c_s²)]`，`c_s² = 1/3`
- Guo 力项：`S_i = (1 − 1/(2τ)) w_i [ (c_i − u)/c_s² + (c_i·u)c_i/c_s⁴ ] · F`
- BGK 碰撞：`f_i* = f_i − (f_i − f_i^eq)/τ + S_i`，`τ = 1/ω`，`ν = (τ − 1/2)c_s²`
- 迁移 + halfway BB（Succi 2001 ch.5）：`f_i(x) = f_i*(x − c_i)`；若 `x + c_i` 为固体，则
  `f_{ō(i)}(x) = f_i*(x)`
- 收敛：每 50 步比较 `max|u_x^{n} − u_x^{n−1}| / max|u_x| < tol`（本轮 `tol = 1e-9`）
- 实测：4 算例均在 `res = 9.9e-10` 收敛，迭代 4 650–65 250 步

关键代码指针：力项与碰撞 `yml_xml_pipeline.py:376-386`；迁移/BB `:387-394`；渗透率提取
`K = ū·ν/g`（Darcy，`:406-409`）；参考解 `analytic_profile`/`analytic_permeability` `:437-451`。

> 实现期踩过的坑（保留可复用）：Guo 力项 `(c_i − u)·F` **不可分配**成 `c_i·F − c_i·u`
> （第一版即此错，导致流量恒定、误差 99.5 %）；`f` 在固体格点被清零后 `ρ` 会为 0，
> 须在固体掩码上取 1，否则 NaN 沿迁移扩散。

---

## ③ 参数表（无量纲数完整）

格子 D2Q9，`c_s² = 1/3`，`ρ₀ = 1`，`nx = 4`（x 向周期，仅用于降维）、`ny` 见下，`x` 向常体力 `g`。

| 算例 | `ny` | `H`（流体行） | `ω` | `τ` | `ν = (τ−½)/3` | `g` | `u_peak` | `Ma = u_peak/c_s` | `Re_H = u_peak·H/ν` | 收敛 `tol` |
|---|---|---|---|---|---|---|---|---|---|---|
| ch_ny24_w1p7 | 24 | 22 | 1.7 | 0.58824 | 0.029412 | 1.458e-05 | 0.02988 | 0.0520 | 22.4 | 1e-9 |
| ch_ny36_w1p7 | 36 | 34 | 1.7 | 0.58824 | 0.029412 | 6.106e-06 | 0.02995 | 0.0520 | 34.6 | 1e-9 |
| ch_ny24_w1p7_halfF | 24 | 22 | 1.7 | 0.58824 | 0.029412 | 7.292e-06 | 0.01494 | 0.0260 | 11.2 | 1e-9 |
| ch_ny24_w0p9 | 24 | 22 | 0.9 | 1.11111 | 0.203704 | 1.010e-04 | 0.03000 | 0.0520 | 3.24 | 1e-9 |

`g` 由「固定 `u_peak = 0.03`」反解：`g = u_peak·8ν/H²`；`Ma ≤ 0.052` 使可压缩误差
`~(Ma)² ≈ 0.27 %` 低于判据一个量级。所有 `g` 在 `validate_spec` 内通过 `Ma ≤ 0.1` 门禁。

---

## ④ 对照表（ref vs got + 误差，判据逐条打勾）

| # | 算例 | `K`(LBM) | `K_ref = H²/12` | `|ΔK|/K_ref` | 剖面 L2 相对误差 | 判据 <15 % | round-trip | 标记解析一致 |
|---|---|---|---|---|---|---|---|---|
| 1 | ch_ny24_w1p7 | 40.25519 | 40.33333 | **0.194 %** | **0.271 %** | ✅ | ✅ | ✅ |
| 2 | ch_ny36_w1p7 | 96.25518 | 96.33333 | **0.081 %** | **0.114 %** | ✅ | ✅ | ✅ |
| 3 | ch_ny24_w1p7_halfF | 40.25519 | 40.33333 | **0.194 %** | **0.271 %** | ✅ | ✅ | ✅ |
| 4 | ch_ny24_w0p9 | 40.49897 | 40.33333 | **0.411 %** | **0.281 %** | ✅ | ✅ | ✅ |
| — | XML 转义探针（`out_folder='weird_&_<value>"x\''`） | — | — | — | — | ✅ 往返无损 | ✅ | — |

判据打勾：4/4 算例「渗透率误差 < 15 %」✅、「剖面 L2 < 15 %」✅、「往返不变式」✅、
「文本标记解析值 = 求解器内值」✅（`rel_tol 1e-11`）。**最差 0.411 %，余量 ~36 倍。**

---

## ⑤ 收敛性 / 参数敏感性

**两分辨率收敛**（同 `u_peak`、同 `ω`）：

| `H` | errK | 比值 | 观测阶（`p = log(e₁/e₂)/log(H₂/H₁)`） |
|---|---|---|---|
| 22 | 0.194 % | — | — |
| 34 | 0.081 % | 2.39 | **p ≈ 1.79** |

→ 与 halfway BB + Guo forcing 的**二阶**预期一致（`H₂/H₁ = 1.545`，`1.545² = 2.39`）。
即误差是离散化误差而非求解器缺陷，且随分辨率单调下降。

**参数敏感性**：

| 扫描 | 结果 | 判读 |
|---|---|---|
| 体力 ×0.5（`u_peak` 0.0299 → 0.0149） | `ΔK/K = 5.2e-14` | Darcy 线性成立（机器精度级） |
| `ω` 1.7 → 0.9（`ν` ×6.93） | `ΔK/K = 0.606 %` | `K` 由几何决定，黏性无关；残差来自 `H` 相同时 `ω=0.9` 的 BB 位置误差更显 |
| 收敛容差 | `tol=1e-9` 时 `res=9.9e-10`，四个算例均达标 | 未收敛偏差 ≲ 0.1 %（`H=46,tol=1e-6` 侧测 errK=0.07 %） |

---

## ⑥ 失败模式与工件

| 失败模式 | 触发条件 | 本轮处理 |
|---|---|---|
| Guo 力项分配错误 | 把 `(c_i−u)·F` 写成 `c_i·F − c_i·u` | 实现期实测暴露（流量恒定、errK 99.5 %），已修正并留记录 |
| 固体格点 `ρ=0` → NaN | 对含固体掩码的场直接 `f.sum` 相除 | 宏观量在固体掩码取 `ρ=1,u=0`；迁移后 `f *= fluid` |
| YAML 值含 XML 元字符 | `out_folder` 含 `& < " '` | ET 属性转义 + 往返断言（`test_roundtrip_preserves_xml_metacharacters`） |
| 配置漂移 | yml 段缺失/多余、`omega` 越界、`ny` 与 `solid_rows` 冲突、路径穿越 | `validate_spec` 11 条负例测试（`test_validate_rejects_bad_spec`） |
| 可压缩误差污染判据 | `g` 过大 | `validate_spec` 内 `Ma ≤ 0.1` 硬门禁（负例测试：`force_x=1.0` → 拒绝） |
| 后处理静默失败 | 标记缺失 | `parse_run_log` 抛 `SpecError`（`test_run_log_missing_marker_raises`） |

**工件**（均在 worktree 内）：

| 路径 | 内容 |
|---|---|
| `src/tensorlbm/yml_xml_pipeline.py` | 管线 + 求解器（489 行） |
| `scripts/xval_yml_xml_pipeline/run_all.py` | 验收驱动，`all_pass=False` 时 `exit 1` |
| `tests/xval/test_yml_xml_pipeline.py` | 26 项 pytest（schema 11 负例 + 往返 + 数值门禁 + 端到端） |
| `artifacts/xval_yml_xml_pipeline/report.json` | 全部原始数字（K、K_ref、迭代、残差、剖面、sensitivity、convergence） |
| `artifacts/xval_yml_xml_pipeline/cases/*.{xml,log}` | 渲染出的求解器文档 + 固定标记日志 |
| `artifacts/xval_yml_xml_pipeline/figure.png` | 左：剖面 got vs 解析（两分辨率）；右：渗透率误差条形图 + 15 % 判据线 |
| `artifacts/xval_yml_xml_pipeline/run.log` | 驱动 stdout（本报告数字来源） |

**repro_cmd**

```bash
python scripts/xval_yml_xml_pipeline/run_all.py          # ~85 s，4 算例，末行 all_pass=True
python -m pytest tests/xval/test_yml_xml_pipeline.py -q  # 26 passed
```

---

## 局限与未覆盖（诚实边界）

1. **OSS 源码未直读**：本节点 `~/oss-cache` 只有 `pylbm` / `Maarten-vd-Sande_lbm` / `hschwane/GraSPH`，
   MPLBM-UT **未缓存**。方法谱系取自 `oss_solutions` 与 TensorPlatform `docs/oss-inventory/MPLBM-UT.md`
   的 `file:line` 引用（二手，但可追溯）；因此本报告对 OSS 的「方法」判断是**结构性**的，
   未逐行复核 `create_palabos_input_file.py` 的模板细节。
2. **未对标 MPLBM-UT 的数值输出**：其求解器是 Palabos（3D D3Q19 Shan-Chen 两相），与本轮
   单相 2D 验证不在同一物理域；本轮只对标**解析解**，未与其算例数值对齐（未运行 Palabos）。
3. **两相 / 图像几何未接**：管线 schema 目前是单相 + 行式固体；MPLBM-UT 的 `geometry/per/fluid{1,2}`
   双周期与 3D 图像标签（`pore_utils.py:52-55`）未实现（`NOT_SUPPORTED` 明确记录）。
4. **`mplbm_utils` 的 `subprocess`/`mpirun` 提交语义未复刻**：`render_runner_script` 只产出
   bash 骨架（`bash -n` 语法校验通过），不真正提交作业。
5. 数值面仅 Poiseuille 一型（无 Couette/后台阶），判据覆盖的是「管线正确性」而非「求解器全域精度」。