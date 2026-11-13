# XVAL · TensorLBM 碰撞核 vs lbmpy 2.0 点对点参考算子（pointwise reference regression）

锚点：`XVAL-LBM-POINTWISE-REFERENCE-REGRESSION`（第 1 轮·开源对标）｜仓：TensorLBM｜分支 `agent/m18192`
对标对象：**lssfau/lbmpy** 的 pointwise-reference-regression 做法｜判据：owner 精度标准（物理量 <15%，目标 <3%）+ 本单预登记代码等价判据

---

## ① 问题 + ref 出处

**问题**：TensorLBM 的 D2Q9/D3Q19 碰撞核（SRT/BGK、TRT、MRT、正则化 BGK）是否与成熟开源实现
lbmpy 计算**同一个算子**？能否把 lbmpy 的"点对点参考回归"纪律移植成本仓可机器裁的回归门？

**OSS 做法出处（从源码读得，非手抄；运行时记录 sha256）**：

| 出处 | 关键做法 |
|---|---|
| `lbmpy/methods/creationfunctions.py :: create_srt / create_trt` | 碰撞算子按其**矩空间定义**符号化构造（平衡态 + 松弛率 → 每个方向的解析更新式），与任何 kernel 代码路径无关 |
| `lbmpy/methods/abstractlbmethod.py :: LbmCollisionRule.lambdify(symbols, fixed_symbols)` | 把同一批符号方程**逐点**在 numpy 场上求值 → 天然的"参考算子" |
| `lbmpy/moments.py :: moment_matrix(moments, stencil)` | 矩基是一等对象：任意 M 矩阵可被重新实例化为 lbmpy 方法 |
| `lbmpy/tests/test_zero_centering.py` | `assert_allclose(reference, kernel_out, rtol=1e-5)`：本仓的回归风格＝参考数组与核输出**逐点**比对 |

**移植判据的标定**（预登记，未事后调整）：代码等价臂用 fp64 舍入判据 `eps_inf < 1e-10`（比 15% 严 9 个量级——
15% 在"同算子比对"语境下是空判据）；物理臂（剪切波粘度）用 owner 15%/3%。

---

## ② 方法与实现

**参考算子链（全自动、零手抄）**：

1. **方向置换由速度元组反推**（`direction_permutation`）：不写死置换表。
   TensorLBM D2Q9 = rest,E,N,W,S,NE,NW,SW,SE；lbmpy D2Q9 = rest,N,S,W,E,NW,NE,SW,SE。
   置换 `perm=[0,4,1,3,2,6,5,7,8]` 由两套速度集匹配得出（D3Q19 同理，两库排列完全不同）。
2. **参考算子构造**：
   - SRT：`create_srt(stencil, ω=1/τ, compressible=True)`（可压缩平衡态 u=j/ρ，与仓内一致）
   - TRT：`create_trt(stencil, 1/τ₊, 1/τ₋)`，`τ₋=½+Λ/(τ₊−½)`，Λ=3/16 → 独立验证仓内"magic parameter"约定
   - MRT：**从仓内 `solver._M_D2Q9_DATA` 反推 lbmpy 矩多项式**（9 个单项式基上精确有理求解），
     再造 `MomentBasedLbMethod`；重建基与仓内 M 矩阵 **maxdiff = 0.0（逐位相同）**
   - 正则化 BGK：lbmpy 无对应实现 → 用**定义式参考**（Latt & Chopard 二阶 Hermite 投影）＋不变量检验
3. **逐点求值**：`LbmCollisionRule.lambdify` 只固定 `c_s=1/√3`，其余自由符号必须是 `f_i`（否则报错，防止静默漏参）。
4. **同一输入场**：两侧拿到**位级相同**的 PDF 场（离散二阶 Maxwellian + 1e-3 非平衡噪声，保证非流体力学矩被激励）。

**臂（预登记）**：A1 SRT｜A2 SRT-matmul（仓内另一条代码路径）｜A3 TRT｜A4 MRT 退化（全部率=1/τ，测 M/M⁻¹ 机器）｜
A5 MRT 原生率（s_e=1.64, s_ε=1.54, s_q=1.7, s_ν=1/τ）｜A6 D3Q19（lbmpy 默认连续平衡态）｜
A6b D3Q19（lbmpy 离散平衡态）｜A7 RLBM（定义式参考）｜C1–C3 阳性对照｜物理臂（周期剪切波 200 步）。

**产物**：`xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/`（`predeclared.json` / `lbmpy_pointwise_ref.py` /
`run_xval.py` / `make_figures.py` / `summary.json` / `table.csv` / `runs/raw_points.json` / `figures/`），
回归门：`tests/test_xval_pointwise_reference_regression.py`（17 例，缺 lbmpy 自动 skip）。

---

## ③ 参数表（无量纲数完整）

| 项 | 值 |
|---|---|
| 格子 | D2Q9（主）、D3Q19（广度臂）；c_s²=1/3，cs=0.5774 |
| 域 | D2Q9 8×8 周期；D3Q19 4×4×4 周期；单步碰撞（无流动） |
| τ（无量纲） | 0.51, 0.55, 0.7, 1.0, 3.0 → ν*=(τ−½)/3 = 0.0033, 0.0167, 0.0667, 0.1667, 0.8333 |
| 速度幅值（rms/分量） | 1e-5, 1e-3, 0.05, 0.15 → Ma=max\|u\|/cs ≈ 5.2e-5, 5.2e-3, 0.26, 0.78 |
| 非平衡噪声 | 1e-3（PDF 单位，ρ≈1） |
| 密度扰动 | ±1%（σ=0.01） |
| TRT Λ（magic） | 3/16 → τ₋=½+Λ/(τ₊−½) |
| MRT 率 | s_e=1.64, s_ε=1.54, s_q=1.7, s_ν=1/τ；守恒矩 s=1（lbmpy）/0（仓内，等价） |
| 物理臂 | ny=64, nx=8, 200 步, u₀=0.01, k=2π/64, ν_th=(τ−½)/3 |
| 重复 | 5 种子（gate 取最差种子）｜点数 540｜wall 19.9 s |
| 环境 | Python 3.11, torch 2.7.1, numpy 2.4.6, **lbmpy 2.0**, pystencils 2.0（sha256 见 `summary.json.provenance`） |

---

## ④ 对照表（ref vs got + 误差打勾/叉）

`eps_inf = max|f_kern − f_ref| / max|f_ref|`（逐点，全 Q 方向 × 全节点；gate 1e-10，D3Q19 同）

| 臂 | 核 | τ | eps_inf（最差种子/幅值） | 判定 |
|---|---|---|---|---|
| A1 | `solver.collide_bgk` | 0.51 / 0.55 / 0.7 / 1.0 / 3.0 | 3.7e-16 … 1.3e-15 | ✅ PASS ×5 |
| A2 | `solver.collide_bgk_matmul` | 同上 | 3.7e-16 … 1.4e-15 | ✅ PASS ×5 |
| A3 | `solver.collide_trt(τ₊, Λ=3/16)` | 同上 | 6.2e-16 … 1.4e-15 | ✅ PASS ×5 |
| A4 | `solver.collide_mrt`（全率=1/τ，退化→BGK） | 同上 | 5.0e-16 … 1.7e-15 | ✅ PASS ×5 |
| A5 | `solver.collide_mrt`（s_e/s_ε/s_q 原生） | 同上 | 1.11e-15（τ 无关） | ✅ PASS ×5 |
| A6 | `collide_bgk_d3q19` vs lbmpy **连续**平衡态 | 0.7 / 1.0 | 2.17e-01 / 1.52e-01 | ❌ FAIL（约定差异，见⑤） |
| A6b | 同上 vs lbmpy **离散**平衡态 | 0.7 / 1.0 | 1.33e-15 / 8.3e-16 | ✅ PASS ×2 |
| A7 | `solver.collide_rlbm`（定义式参考） | 0.51/0.7/1.0/3.0 | 8.6e-11 / 3.9e-11 / 0.0 / 6.0e-11 | ✅ PASS ×8 |
| MRT 基 | 重建基 vs 仓内 M | — | **0.0**（逐位相同） | ✅ |
| C1 | 去掉方向置换 | 0.7 | 9.02e-02 | ✅ 检出（置换承重） |
| C2/C3 | 参考用 τ(1+1e-3) | 0.7 | 1.574e-05 | ✅ 检出 |
| C2/C3 | 参考用 τ(1+1e-6) | 0.7 | 1.575e-08 | ⚠️ 见"偏差申报" |

**物理臂（owner  caliber）**

| 侧 | τ | ν_meas | ν_th=(τ−½)/3 | 误差 | <3%? |
|---|---|---|---|---|---|
| TensorLBM `collide_bgk` | 0.55 | 1.709291e-02 | 1.666667e-02 | **2.56%** | ✅ |
| TensorLBM `collide_bgk` | 0.7 | 6.706193e-02 | 6.666667e-02 | **0.59%** | ✅ |
| TensorLBM `collide_bgk` | 1.0 | 1.666666e-01 | 1.666667e-01 | **0.0003%** | ✅ |
| lbmpy 参考算子（同一 marching loop） | 0.7 | 6.706193e-02 | 6.666667e-02 | **0.59%** | ✅ |

**误差判定（显式回答"是否 <3%"）**：
- 代码等价臂：误差 **1e-16 ~ 1e-15**（fp64 舍入），远优于 3% 与 15% —— 判"是"；
- 物理臂：**0.59% / 2.56% / 0.0003%**，全部 < 3% —— 判"是"（τ→0.5 端 2.56% 为离散误差，随 τ 增大单调减小）；
- A6（连续平衡态约定）15–22%：**不是精度问题**，是"比错了约定"——换成约定对齐的 A6b 后 1e-15。

**偏差申报（honest）**：`control_detection_min=1e-6` 在预登记时是针对"被注入的扰动幅度"设的，实际增益为
∂ε/∂lnτ ≈ 1.6e-2，故 rel=1e-6 的 τ 扰动只产生 1.57e-8（仍比舍入底噪高 7 个量级）。
**未事后下调判据**：保留原 1e-6 判据、标注"未检出"，另**追加**（而非替换）rel=1e-3 对照臂使灵敏度被显式证明。

---

## ⑤ 收敛性 / 参数敏感性（≥2 点）

1. **松弛时间敏感性（τ = 0.51→3.0，跨 6 倍）**：`eps_inf` 在 3.7e-16~1.7e-15 间无趋势（图 1），
   即两实现的差**与 τ 无关**＝纯舍入；若存在算子代数错误，误差应随 τ 放大（∝|f−f_eq|/τ）。
2. **幅值/马赫数敏感性（rms 1e-5→0.15，Ma 5e-5→0.78）**：`eps_inf` 平台（图 2），
   说明平衡态与非线性项（u²、cu²）在两侧逐项一致，不是"低速线性区偶然重合"。
3. **数据类型**：fp32 舍入后同一判据呈 fp32 底噪（`summary.json.fp32_sensitivity`），
   即 fp64 下的 1e-15 一致不是浮点巧合。
4. **唯一真实敏感性：平衡态约定（A6 vs A6b）**——D3Q19 上 lbmpy 默认**连续**二阶 Maxwellian 与仓内
   **离散（保矩）**平衡态在 PDF 层差 15–22%（Ma 0.26 时；u² 量级项分布到各方向）。D2Q9 上两者恰好重合。
   物理后果：守恒量（ρ, j）与二阶矩不受影响（`moment_err_rho ≈ 0` 已逐点核）——差异只在高阶非流体力学矩，
   但足以让"逐点比对"这类强判据翻红，故必须在参考侧显式指定约定。

---

## ⑥ 失败模式 + 工件 + 生态摩擦

**失败模式（本轮真实踩到，均已定位而非绕过）**

| 现象 | 根因 | 暴露方式 |
|---|---|---|
| 全部臂 eps≈0.66 假 FAIL | `unpermute` 写成了与 `permute` 相同的操作（置换非对合） | 与 8×8×小场手工对照发现"看似随机打乱" |
| MRT 臂 eps 1e-2 假 FAIL | 参考侧 τ 未随 `ref_spec` 传递，核用 τ=1.0、参考用 τ=0.51…3.0 | 聚合表只出现 τ=1.0 一行 → 键空间坍缩 |
| MRT/degenerate 报错 | `np.tensordot(vel, f, axes=(1,0))` 轴错（应为 `(0,0)`） | 形状不匹配 |
| RLBM 参考报错 | `einsum` 写法误用 tensordot | — |
| C1 对照"未检出" | 对照初版实际是 A1 的复制（同样带置换） | 其值与 A1 同量级 → 判为空对照 |
| lbmpy 参考 `c_s` 未固定 → object dtype | `fixed_symbols` 键须用规则里的**真实符号对象** | 数值静默变 sympy 对象 |

**工件**

```
xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/
├── predeclared.json          # 决策/假设/判据/停时规则（先于运行）
├── lbmpy_pointwise_ref.py    # 参考算子适配层（置换反推 + 矩基重建 + lambdify + 指标）
├── run_xval.py               # 臂驱动 → summary.json / table.csv / runs/raw_points.json
├── make_figures.py           # 3 张图
├── summary.json              # 聚合 + provenance(sha256) + 对照 + 物理臂
├── table.csv                 # 540 点逐点指标
├── runs/raw_points.json      # 逐点原始记录（零手抄：报告数字均由此生成）
└── figures/{fig1_eps_vs_tau, fig2_eps_vs_amplitude, fig3_d3q19_and_physics}.png
tests/test_xval_pointwise_reference_regression.py   # 回归门 17 例全绿
```

**本仓 vs OSS 差异清单（做法 / 参数 / 边界条件）与移植取舍**

| 维度 | TensorLBM | lbmpy 2.0 | 移植取舍 |
|---|---|---|---|
| 执行形态 | torch 张量算子（(Q,·) 布局前置） | pystencils 生成 C kernel（SIMD） | **不搬** codegen（重依赖）；只搬"符号参考算子 + 逐点比对"纪律 |
| 方向序 | rest,E,N,W,S,NE,NW,SW,SE | rest,N,S,W,E,NW,NE,SW,SE | 置换由速度元组**自动反推**，不维护手工表 |
| 平衡态约定 | D2Q9/D3Q19 均为离散（保矩）构造 | 默认 `continuous_equilibrium=True` | 参考侧必须显式指定；D3Q19 上二者差 15–22%（本轮最实质发现） |
| 松弛参数接口 | τ（+Λ magic, τ₋ 内部推导） | ω=1/τ；TRT 直接给 τ₋ | 两仓"magic parameter"约定**逐位等价**（A3 1e-15） |
| MRT | 自带 Lallemand–Luo M 矩阵（9×9 数值表） | `create_mrt_raw` 用固定默认矩集，无 by-matrix 入口 | 用单多项式基精确反解仓内 M → lbmpy `MomentBasedLbMethod`（M 逐位相同） |
| 精度常量 | D3Q19 契约上用 float32 格点常量 | 全程双精度 | 判据分流：D3Q19 门 1e-6（实测 1e-15，因离散平衡态臂走 f64） |
| 边界条件 | 独立 bounce-back / Zou-He 模块 | kernel 内边界（`boundaries_in_kernel`） | 本轮只覆盖碰撞算子（无边界臂），边界 XVAL 留第 2 轮 |
| 正则化 BGK | 有（Hermite 二阶投影） | 无 D2Q9 对应 | OSS 对标标 **NOT_SUPPORTED**，改用定义式参考 + 三阶非平衡矩为零的不变量（1e-13） |

**生态摩擦（如实记）**

1. `GET /api/v1/knowledge_base/search`、`/api/v1/oss_solutions/search` → **404**（端点已迁移）；现行端点
   `/api/v1/knowledge/search?q=lbmpy|pointwise|tensorlbm` 均返回 `total:0` → 知识库无 LBM 交叉对标沉积物，
   本单的结论只能落在本仓工件里（建议：XVAL 结论按域回灌 knowledge sediment）。
2. `pip download lbmpy` 得到的 wheel **不含 tests/**（测试只在 sdist 里）→ 读取其回归做法需 `--no-binary :all:` 取 sdist；
   建议入库清单记录"sdist 才有 test 语义"这一事实，避免下轮重复摸索。
3. shell 侧 GitHub raw 直连被安全策略拦截 → 对标源只能走 PyPI 分发物（版本 2.0 与 lssfau/lbmpy fork 的测试做法一致，
   但 fork 特有改动本轮**未**核对，登记为 WITHHELD）。

**结论**：D2Q9 全部四个碰撞核 + D3Q19 BGK（约定对齐后）与 lbmpy 2.0 点对点参考算子**逐位一致到 fp64 舍入**，
回归门已固化进 `tests/`；物理臂粘度误差 0.6%（τ=0.7）~2.6%（τ=0.55）满足 owner <3%；
D3Q19 平衡态约定差异、边界条件、lssfau fork 特有改动留第 2 轮。