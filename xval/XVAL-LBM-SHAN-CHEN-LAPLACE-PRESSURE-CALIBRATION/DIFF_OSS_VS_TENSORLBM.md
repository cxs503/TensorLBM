# Diff list: TensorLBM vs `lbm-principles-practice/code`（Shan–Chen Young–Laplace 标定）

第 1 轮 XVAL，锚点 `XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION`。
叙述与数字见 `REPORT_XVAL_SHAN_CHEN_LAPLACE_PRESSURE.md`；端口代码 `oss_port.py`；
驱动 `run_xval.py`；产物 `artifacts/xval_shan_chen_laplace/`。
OSS 侧事实全部来自盘点工件
`TensorPlatform/docs/oss-inventory/REPORT_lbm-principles-practice-code.md` §4.1/§4.3/§九（带行号）。

## 一、做法差异

| 方面 | OSS `chapter9/shanchen.cpp` | TensorLBM `src/tensorlbm/sc_surface_tension.py` | 影响（本 XVAL 实测） |
|---|---|---|---|
| 赝势 ψ | **只有一种** ψ=ρ₀[1−exp(−ρ/ρ₀)]，ρ₀ 写死 1（`:74-76,:37`） | ψ_exp 之外另有 Carnahan–Starling、幂律（`multiphase.py`）、`psi_masked`/润湿壁 ψ_wall | 本单只用 ψ_exp ⇒ 逐位同 |
| 互作用力 | 8 近邻、排除静息方向、wᵢ 权重（`:79-109,:90`） | 各向同性模板族（d2q9 / shell4 / iso4）+ 谱梯度；∇ψ 以 a1 归一 ⇒ D2Q9 下**代数恒等** | 5 步 max|Δρ|=**8.9e−16**；4000 步 1.2e−13 |
| 耦合符号 | g_A=−4.7（物理约定） | `G_eff=−4.7`；但 `multiphase.py` 用 `G_lib=+4.7`（**反号**，见其 CONVENTION NOTE） | 移植/对标时**最大陷阱**：符号选错 = 相分离反演 |
| 力项格式 | Guo 二阶（1−0.5ω）（`:261-263`）+ BGK ω=1/τ | 同（`collide_guo`），另备 velocity_shift 与 EDM（`multiphase.py`） | 本单 Guo↔Guo ⇒ 同；EDM 会改界面伪力 |
| 速度 | 重心速度 + 0.5F 修正（`:142-161`） | 同 | 同 |
| **压力读数** | **只有 closure** p=ρ/3+gψ²/6（`:126-139`）——*就是模型假设本身*，圆形闭合 | closure（`pressure_eos`）+ **动量通量**（`pressure_flux`）+ 压力张量/Korteweg σ 锚 | flux vs closure 差 **0.02–0.68%**（体相带）⇒ closure 可用但非独立证据 |
| 半径 | 中线线性插值，`:358` **rad 未初始化**（UB） | 同算法 + 护栏（失败即抛异常）；另有面积等效 R_eq、`radial_profile`、角向分箱 Δp(θ) | 本文端口把 UB 换成显式失败：R=24 记 FAILED 而非 −4.86 |
| 初始化 | 陡峭界面、ρ_l=2.1（**非共存**）⇒ 液滴吸质量长大；无质量校验 | `droplet_rho` 用 tanh 扩散界面；另有 mass ledger/closure 模块 | 为可比性两支都用**陡峭且非共存**初值 |
| 伪速度 | **完全无量化** | u_max 报告（本单 2.7e−3，Ma≈0.005） | 新增能力（单侧） |
| 边界 | 仅全周期、无壁（`:271-272`） | 周期 + bounce-back + 润湿 ψ_wall + 接触角反演 | 单侧能力 |
| 维数/后端 | 2D D2Q9 C++ 单文件、double | 2D D2Q9 torch（CPU/GPU, float64），多相 3D D3Q19/D3Q27 另存 | 单侧能力 |
| 数值精度细节 | C++ double 全程 | 本单 float64；但 `collide_guo`/`pressure_kinetic` 把格子常数 `.float()` 降为 float32 ⇒ ~1e−7 相对（≈3e−5 的 Δp，因 Δp/⟨p⟩≈3e−2） | 可忽略，但**建议改 `.to(dtype)`**（一行，未在本轮改） |
| 验证面 | 打印数值、**无阈值/无验收** | 预注册阈值 + 机器可判 KPI + pytest | 本单把 OSS 的“打印”升级为“门禁” |

## 二、参数/边界条件差异

* 网格/步数：OSS 64²、4000 步（本单沿用），R_init∈{8,12,16,20,24}（上游仅报 4 点 + 1 个 UB 点）。
* 盒平均窗口：4×4（沿用）；本单测 8×8 敏感性 ⇒ γ 变 0.008%（不是混淆项）。
* 半径约定：OSS 单向中线扫描；本单主口径同（单向），并测双向平均 **+2.9%**、面积等效 **+2.6%**
  ⇒ 这是与 ref 系统性 −2.8% 的**同量级来源**。
* 压力读数位置：液滴中心盒 vs 角点汽相盒（沿用 OSS）；因 Δp 是 O(0.05) 量级压力之差
  （Δp/⟨p⟩=0.021–0.052），盒位置/窗口是误差预算主项，已单列。
* 上游未给任何参考 σ、未给验收阈值 ⇒ 本单的“ref”是**上游实测值**（盘点战役复跑所得），
  不是文献/实验值；因此本单结论是**交叉实现一致性**，不是物理精度。

## 三、移植取舍（建议）

1. **采用（已采用）**：OSS 标定器的两个约定 —— 4×4 盒平均 + 中线线性插值半径 —— 固化在
   `oss_port.py`/`run_xval.py`，作为与 OSS 对话的**公共口径**。
2. **采用并加固**：把上游 `:358` 未初始化半径改为**异常护栏**（不产生未定义 γ）。
   这是本单唯一语义改动；对已分辨液滴**从不触发**（R=12/16/20 正常）。
3. **不采用**：OSS 的 closure-only 压力证据链。理由：closure = 模型假设（圆形闭合），
   在 Δp/⟨p⟩~3e−2 的抵消结构下无法自证；实测 flux 与 closure 差 0.02–0.68% 说明
   closure 可作**廉价交叉校验**，但不能作**唯一证据**（决策 D2）。
4. **不采用**：单一 ψ、全周期无壁、无伪速度度量、无阈值打印。
5. **建议的后续小改（未在本轮做，避免越界）**：
   (a) `sc_surface_tension.collide_guo/pressure_kinetic` 的 `.float()` → `.to(f.dtype)`；
   (b) 把 `RadiusEstimatorFailed` 语义提升为模块级（现只在 XVAL 端口内）；
   (c) 用 `shell4`/`iso4` 模板重跑同一标定，检验 κ₃→0 时 γ 是否按 eq.(6) 归零（本轮 σ_K 锚已 FAIL，值得追）。
6. **无代码移植需求**：ARM_TL 与 OSS 端口在 γ 上等价到 **2.4e−15**（4000 步），
   故不引入上游实现，只吸收其**做法与护栏**。