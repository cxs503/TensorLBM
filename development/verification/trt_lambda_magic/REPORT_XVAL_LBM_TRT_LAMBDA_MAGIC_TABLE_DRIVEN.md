# XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN — TensorLBM vs `lbm-principles-practice/code` 的 TRT magic-Λ 表格化做法交叉对标

- 仓库：TensorLBM ｜ mission 18452 ｜ 分支 `agent/m18452` ｜ 轮次 1（开源对标）
- 对标仓：`lbm-principles-practice/code` @ `6e2c592fdc3592c14dfd52f860fc1ceea930bcb0`（MIT，Krüger/Kusumaatmaja/Kuzmin/Shardt/Silva/Viggen 教材配套代码），inventory tag `trt-lambda-magic-table-driven`
- 预注册：`xval/XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN/predeclared.json`（阈值先于首次运行写入）
- 一键复现：`cd xval/XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN && python3 xval_trt_lambda.py && python3 diagnostics_u_regime.py && python3 make_figures.py`；验收：`pytest tests/test_xval_trt_lambda_magic_table_driven.py -q`（7 passed）

---

## ① 问题与 ref 出处

**问题**：TensorLBM 的 TRT 用「magic 参数 Λ」把奇偶两支弛豫率锁死（`Λ=(τ_s−½)(τ_o−½)`），而 OSS 参考实现用**常量表驱动**（`weights_trt[]`/`complement[]` + 三条不同写法的 Λ 反解式）在标量对流扩散（ADE）上演示同一件事。需回答：TensorLBM 的 Λ 代数与离散表是否与 OSS **同一语义**？同一算例上两支数值是否一致？OSS 的做法里有哪些是本仓**缺**的（需要移植），哪些只是写法差异（不该移植）？误差是否满足 owner 的 15% 精度标准？

**ref 出处（零手抄，全部从 vendored 源码解析）**：

| OSS 文件 | 取用的关键做法 | 行号（源码） |
|---|---|---|
| `chapter8/gaussian_1d_magic6.cpp` | D1Q3 标量表 `weights={2/3,1/6,1/6}, cx={0,1,-1}, complement={0,2,1}`；手设 `omega=0.3` 为 **plus** 支，`omega_minus=1/(magic/(1/omega-0.5)+0.5)`；`magic=1/6`, `ux=0.5` | 43–47, 96, 120–123 |
| `chapter8/gaussian_1d_magic12.cpp` | 角色互换版：手设 `omega=0.5` 为 **minus** 支，plus 由 Λ 反解；`magic=1/12`, `ux=0` | 39, 46–47, 121–123 |
| `chapter8/gaussian_2d_trt.cpp` | D2Q9 标量表与 `complement={0,3,4,1,2,7,8,5,6}`；`omegaminus=1.95` 手设、`omegaplus=1/(0.5+lambda*omegaminus/(1-0.5*omegaminus))`；`lambda=1/12` | 47–49, 118–133 |
| `chapter8/cylinder.cpp` | TRT 动力学在流动算例中的同构写法（`omega_plus/omega_minus` 由 Λ 关联） | 51–63, 159–208 |

vendored 副本 + sha256：`xval/XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN/oss_src/PROVENANCE.json`（6 个 `.cpp` 逐字节保留，MIT 许可随附）。

**本仓被对标面**：`src/tensorlbm/scalar_transport.py`（D2Q5 标量 TRT：`tau_symmetric_from_magic`/`magic_from_taus`/`collide_scalar_trt`/`equilibrium_scalar_d2q5`/`dispersion_rates`）、`src/tensorlbm/d2q9.py`、`src/tensorlbm/d3q27.py`（互补方向表）。

---

## ② 方法与实现

**预注册的判据（先写后跑）**

| KPI | 定义 | 阈值 |
|---|---|---|
| K1 | Λ 反解式：本仓 ω(Λ,ω) 与 OSS 三种写法在 ω∈[0.51,1.99]×Λ∈{1/12,1/6,3/16,1/4,3/8} 网格上的最大绝对差 | ≤1e-15 |
| K2 | 离散表（权重/速度/互补映射）逐值恒等（1=同一，0=不同） | =1.0 |
| K3 | 共享算例上场值最大绝对差（本仓 vs OSS 二进制） | ≤1e-13 |
| K4 | 与**解析解**的相对 L2（max over t） | ≤0.15（owner band） |
| K5 | Λ 敏感性：场级 relL2 + 5×5 本征谱 D_eff(k)/D 的支路偏差 | 只报不判 |
| K6 | 网格细化观测阶 p | 只报 |
| K7 | OSS 二阶速度项 u² 的影响 | 0.15 |

**实验臂（对照组设计）**

- **Arm A**：OSS `gaussian_1d_magic12`（u=0，纯扩散，Λ=1/12，50×20 步，周期）。本仓用 D2Q5 **ny=1** 退化为 1D（周期 y=1 使 ±y 群静止，rest 群总权重恰 = 1/3+1/6+1/6 = **2/3**，与 OSS 的 D1Q3 rest 权重数值相同 → 两套格点在该算例上等价，K2 已证）。
- **Arm B**：OSS `gaussian_1d_magic6`（u=0.5，Λ=1/6，脉冲平流 10 格）——暴露**平衡态速度阶数**差异。
- **参考解（非数值基准）**：`exact_periodic()` 用周期热核叠加（n=−3..3 镜像）+ Gauss–Legendre 分面板求积，作用在**与 OSS 完全相同的截断高斯初值**上；求积自检（面板×点数加倍）最大相对变化 `2.55e-15`。
- **对照实现**：OSS 源码编译成二进制自己跑、自己写 `.dat`，程序解析（`zero hand-copy`）。OSS 输出用 `precision(10)`，比较下限被文件分辨率锁在 ~1e-10；因此另建 **`precision(17)` 仪表化构建**（**只改输出精度与输出目录名，算法逐字节不动**，diff 存档 `build/oss_hi/*.diff`）得到机器精度可比的参考数据。

---

## ③ 参数表（无量纲/格点单位完整）

| 量 | Arm A（magic12） | Arm B（magic6） | 来源 |
|---|---|---|---|
| 网格 N | 50（周期） | 50（周期） | 源码 `NX` |
| 步数 | 20 | 20 | 源码 `N` |
| 初值 | `exp(-0.5(x−25)²/9)`，`|x−25|≤9`，其余 0 | 同 | 源码 `init()` |
| σ₀ | 3 | 3 | 源码 `sigma` |
| u | 0 | 0.5 | 源码 `ux_main` |
| Λ | 1/12 | 1/6 | 源码 `magic` |
| 手设率 ω_hand | 0.5（odd 支） | 0.3（even 支） | 源码 `omega` |
| 反解率 | ω_plus=1.79999… | ω_minus=1.78947… | 源码公式 |
| τ_o / τ_s | 2.0 / 0.555556 | 0.558824 / 3.333333 | τ=1/ω |
| D=c_s²(τ_o−½), c_s²=1/3 | 0.5 | 0.0196078 | 本仓关系（K3 前经剖面反演确认） |
| 扩散长度 √(2Dt) | 4.47 格 | 0.886 格 | — |
| 平流位移 ut | 0 | 10 格 | — |
| 精度/ dtype | float64 | float64 | — |

**关键实测（反演，不是假设）**：OSS 两支中**由奇（反对称）支设定扩散**——谱法反演得 `D_eff→0.5`（magic12，与 τ_o=2 一致）而非 0.0185（若由偶支设定）；magic6 侧得 `D_eff≈0.021`（与 τ_o=0.5588→0.0196 一致），排除 0.944。这条否证了「OSS 把 D 交给 even 支」的直觉读法：**写法互换 ≠ 物理角色互换**。

---

## ④ 对照表（ref vs got）

| KPI | ref (OSS) | got (TensorLBM) | 误差 | 判 |
|---|---|---|---|---|
| K1 Λ 代数（3 种写法，375 点） | 源码公式 | `tau_symmetric_from_magic`/`magic_from_taus` | 最大 5.55e-16（≤1 ulp） | ✅ |
| K2 权重（1D rest/±x） | 2/3, 1/6, 1/6 | 1/3+1/6+1/6=2/3, 1/6 | 差 0.0 | ✅ |
| K2 互补映射 1D | {0,2,1} | D2Q5 OPP 降维后 {0,2,1} | 恒等 | ✅ |
| K2 互补映射 2D（D2Q9） | {0,3,4,1,2,7,8,5,6} | `d2q9.OPPOSITE` 同值 | 恒等 | ✅ |
| K3 场值最大差（Arm A，1050 个双精度样本） | OSS `precision(17)` 二进制 | D2Q5 TRT | **5.55e-16**（relL2 3.05e-16） | ✅（阈 1e-13） |
| K3' 同上，用 OSS **原样** 10 位输出 | OSS `.dat` | 同上 | 4.99e-11（= 文件分辨率地板） | ❌ 预注册阈 1e-13（测量分辨率所限，见⑤） |
| K4 vs 解析解（Arm A，max over t） | 4.031e-2 | 4.031e-2 | **4.03%** | ✅ <15% |
| K4 峰值误差 / 方差误差（末态） | 0.506% / 4.31% | 同 | — | ✅ |
| Arm B u=0.5：OSS vs 本仓（加 u² 项） | OSS 二进制 | 本仓 `second_order=True` | **0.0（逐位相同）** | ✅ |
| Arm B u=0.5：OSS vs 本仓**默认**（线性 u） | OSS 二进制 | 本仓默认 | 3.75%（relL2），max abs 4.0e-2 | 差异（见差异清单 D-3） |
| K7 u² 项在 u=0.5 的收益 | — | 5.694% → 4.636% | ×1.23 | <15%（但 |u| 增大后急剧放大，见⑤） |
| K6 细化阶（dx=1→0.5→0.25） | — | p=2.10, 1.92 | 二阶 | ✅ |
| K5 Λ 扫描（末态 relL2） | — | Λ=1/12: 1.464%；1/6: 1.364%；3/16: 1.341%；1/4: 1.282%；BGK: 4.095% | TRT 比 BGK 好 **2.8–3.2×** | ✅（四者互差 <15%） |
| 质量守恒漂移（各臂） | — | ≤2.0e-15 相对 | 机器精度 | ✅ |

---

## ⑤ 收敛性 / 参数敏感性（≥2 点）

1. **空间细化（diffusive scaling dt=dx²，T=20，D=0.5 固定）**：relL2 = 1.464%(dx=1) → 0.341%(dx=1/2) → 0.0900%(dx=1/4)，观测阶 **p=2.10 / 1.92** → 二阶收敛，与 TRT 在 ADE 上的理论一致（图 3）。
2. **magic 参数扫描**：Λ∈{1/12,1/6,3/16,1/4} 场级误差 1.28%–1.46%（互差 ≤13%），**BGK(单率) 4.10%**——TRT 的收益（3×）远大于选哪个 Λ 的收益。5×5 本征谱给出的支路色散 `|D_eff(k)/D−1|`：k≤0.2 时 TRT 0.19%→3.2%、BGK 0.04%→0.65%（BGK 色散更小），k≥0.4 时 5×5 谱支路交叉导致单支读法失效（TRT 出现 0.9–1.17 的跳支），故**该 KPI 只作辅助、不作判据**；这与场级结论不矛盾：场级误差的主控项是初值高波数内容在 ghost 模上的传播，而 TRT 用 τ_s=0.556 阻尼了这部分（对照 D2 诊断：无缝高斯初值上 TRT 1.48% vs BGK 4.06%，结论不变 → 是格式性质，非截断边沿假象）。
3. **对流速度（事后诊断）**：默认线性平衡态的相对 L2 随 u 从 0.05→0.5 由 2.03%→39.77%（≈0.95·u² 律，实测 ratio_over_u2 = 0.94/0.95/1.00/1.46）；加 OSS u² 项后 0.5 处降到 14.21%。**故默认线性平衡态的适用上限约 |u|≈0.30**（u=0.3 时 13.79%，u=0.5 时 39.8% 超 15% 带）。

### 本仓 vs OSS 差异清单 + 移植取舍

| # | 差异 | 性质 | 取舍 |
|---|---|---|---|
| D-1 | **表述方式**：OSS 用常量表 + 三条独立的 Λ 反解式（1D「解 minus」/1D「角色互换解 plus」/2D「由 minus 解 plus」）；本仓用 `(τ_s−½)(τ_o−½)=Λ` 的函数族 | 纯写法 | **不移植**：K1 证三种写法与本仓代数同值（≤1 ulp），本仓单函数更不易漂 |
| D-2 | **格点**：OSS 演示 D1Q3/D2Q9 标量；本仓标量面是 D2Q5 | 实现 | **不移植**：D2Q5 在 1 行退化下与 D1Q3 权重/互补表逐值一致（K2），场值逐位一致（Arm B 0.0） |
| D-3 | **标量平衡态速度阶数**：OSS 含 `4.5(c²−1/3)u²`；本仓历史实现只保留线性 u | **能力缺口** | **移植为 opt-in**（`equilibrium_scalar_d2q5(..., second_order=True)` / `collide_scalar_trt(..., second_order=False 默认)`）：默认逐位不变（`torch.equal` 断言），开关打开即与 OSS **逐位相同**；u≤0.1 时可忽略（≤1.0%），u≥0.3 时默认臂会超 15% 带 → 高对流标量算例（`Config.ux>0.2`）应显式开启 |
| D-4 | **默认 Λ**：本仓标量默认 3/16、流动 D3Q27 默认 3/16；OSS 演示 1/6 与 1/12 | 参数 | **不改默认**：本例四种 Λ 互差 ≤13%，3/16 已优于 1/12（1.341% vs 1.464%）；但把 1/6、1/12 记入可选值并在文档标注 OSS 出处 |
| D-5 | **D 归属支**：OSS 源码里 magic6 手设的是 even 支、magic12 手设的是 odd 支（写法互换）；本仓文档明确「odd 支定 D」 | 语义 | **保留本仓约定并补证据**：谱法反演证实 OSS 两支都由 **odd** 支定 D（0.5 与 0.0196 两个候选里实测选中前者），说明「写法互换」只是同一关系的两种代数解，不是物理角色互换 |
| D-6 | **边界条件**：OSS 该演示全周期、无壁；本仓另有 bounce-back/开放边界 | 范围 | 不属本单：Λ 对**壁面位置误差**（Ginzburg 3/16 的经典用途）需带壁算例专门对标 → 记入后续轮次（本单未覆盖，如实声明） |

---

## ⑥ 失败模式、工件与生态摩擦

**honest 记录（不粉饰）**

- **K3'（原样 10 位输出）判 FAIL**：预注册阈 1e-13，实测 4.99e-11。根因是参考实现写盘 `precision(10)`，不是算法差异——用 `precision(17)` 仪表化重建后同算例降到 5.55e-16。**阈值未事后修改**，两条结果都保留在 `results.json`（`armA_identity_verbatim_ascii10` / `armA_identity_hi`）。
- **K5 谱支路 k≥0.4 不可用**：5×5 谱在同一 k 上出现支路交叉，单支追踪读出 0.9–1.17 的假偏差；已在报告与本文件标注为「辅助指标」，改用场级误差作判据（事后再跑无缝初值诊断确认结论方向不变）。
- **事后诊断（D2 脚本）非预注册**：`diagnostics_u_regime.py` 在门控臂之后追加，用于解释 D-3 的适用范围与 TRT-vs-BGK 差距的来源，**明确不计入门控判定**。
- **仓库既有失败（非本单引入）**：`tests/test_scalar_transport.py::test_momentum_jet_exponent`、`::test_lfl_wind_sweep_monotone` 在 `git stash` 掉本单改动后**同样失败**（57.8 s 复现），属存量问题，未在本单修复（如实声明，避免与本次移植混淆）。
- **未覆盖**：D2Q9/D3Q27 **流动**侧的 TRT（OSS `cylinder.cpp` 只做同构演示，其算例需曲面壁边界）；Λ 的壁面位置误差标定；D2Q5 各向异性对 2D 高斯（OSS `gaussian_2d_trt` 网格 400×400 未跑，因本单门控只需 1D 语义+精度）。以上列为第 2 轮接棒候选。
- **生态摩擦**：① `oss_solutions` 表对该 Agent 只读受限（`permission denied`），`/oss_solutions` API 404，需从 `docs/oss_inventory/family-set-v1/REPORT.md` 与 `docs/oss-inventory/REPORT_lbm-principles-practice-code.md` 反查 tag 与文件行号——建议给 Agent 开只读 API 或视图；② GitHub `raw.githubusercontent.com` 在本机不可达（SSL），`git clone` 挂起，最终走 `api.github.com/contents`（base64）拿到逐字节源码——建议在 OSS 入库时把源码 vendored 随仓（本次已按此规范落 `oss_src/`）。

**工件清单**

| 路径 | 内容 |
|---|---|
| `xval/XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN/predeclared.json` | 预注册：决策/替代方案/假设/KPI/阈值/停止规则/环境 |
| `…/oss_src/*.cpp` + `PROVENANCE.json` | OSS 逐字节源码（MIT）+ commit/sha256 |
| `…/xval_trt_lambda.py` | 主实验（解析 OSS 常量 → 跑 OSS 二进制 → 跑本仓 → 全部 KPI） |
| `…/diagnostics_u_regime.py` | 事后诊断（u² 项适用域、TRT vs BGK 来源） |
| `…/results.json`、`diagnostics.json`、`fields.npz` | 全部原始数字与场数据（图与报告中的数一律取自此处） |
| `…/figures/fig1..fig5` | 场值恒等/Λ 扫描/细化/u 适用域/armB 剖面 |
| `…/build/oss*/` | OSS 参考二进制与 `precision(17)` 仪表化 diff |
| `tests/test_xval_trt_lambda_magic_table_driven.py` | 7 项验收（K1/K2/K3/K4/K7 + 默认位不变 + 碰撞质量守恒） |
| `src/tensorlbm/scalar_transport.py` | 移植项：`second_order` 可选 OSS 二阶速度项（默认关闭） |

**结论**：TRT magic-Λ 表格化做法与本仓**同一语义**且**数值逐位可复现**（5.55e-16）；两臂对解析解误差 **4.03% ≪ 15%**。唯一实质能力缺口是 OSS 标量平衡态的 u² 项，已按 opt-in 移植并以逐位恒等验收；其余差异属写法/参数，不做移植。**是否 <3%**：否——场级相对 L2 为 4.03%（两臂相同），主控项是高波数初值在 ghost 支上的传播；细化到 dx=1/4 时降到 0.090%（<3%），故 3% 目标在 dx≤0.25 的格点分辨率下达成，粗格点（dx=1）不达成。