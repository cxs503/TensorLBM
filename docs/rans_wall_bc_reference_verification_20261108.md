# RANS 壁函数（rans_wall_bc）解析基准验证欠账清偿（2026-11-08, m6600）

> 单：selfstart（generalist 席，m6600）· 仓：TensorLBM · 分支：agent/m6600
> 改动面：`tests/test_rans_wall_bc_reference.py`（新增，13 用例）、
> `src/tensorlbm/turbulence/rans_wall_bc.py`（仅 docstring 勘误，零行为改动）。
> 重跑：`PYTHONPATH=src python3 -m pytest tests/test_rans_wall_bc_reference.py -q`（6.3 s，CPU/float64）

## 1. 案例介绍（物理背景 / 对象 / 参数）

**物理背景。** RANS 壁函数是 LBM+壁函数生产链（`wall_model` 对数律反演 u_tau →
RANS 近壁边界值）的最后一环：`set_rans_wall_bc_k_epsilon` 把 u_tau 换算成
k-ε 模型近壁单元（第一离壁格）的 TKE 与耗散率边界值，OpenFOAM
`nutkWallFunction`/`epsilonWallFunction` 同款闭式：

| 量 | 闭式 | 常数 |
|---|---|---|
| k_wall | u_tau² / √C_μ | C_μ = 0.09（与 `rans_ke.py` 同源） |
| ε_wall | u_tau³ / (κ·y) | κ = 0.41 |

**验证对象与参数。** 3D 小张量场（4×6×5），近壁面 j=1；u_tau=0.05、y=0.5 lu、
ν=5e-4 → **y+ = y·u_tau/ν = 50**，落在壁函数有效区 30 < y+ < 600（有效性门已固化为测试）。

**选题依据（先读账）。** `benchmarks/gaps.md` G21 列 6 个「有实现未验证未导出」孤儿模块；
m6583（在飞）已裁 ddes/des_turbulence；本模块（71 行）此前**零直接测试**
（`tests/` 全库检索仅 `wall_model` 自身被覆盖），属典型验证欠账。

## 2. 怎么做的（方法 / 流程 / 工具链）

1. **闭式锚**：k_wall、ε_wall 逐格点与闭式值机器精度比对（atol 1e-15/1e-18）。
2. **物理不变量 I1（最强锚）**：对数律区湍流粘度 ν_t = κ·u_tau·y；k-ε 模型定义
   ν_t = C_μ·k²/ε。代入两个壁值必须**精确**复现对数律值：
   `C_μ·k_wall²/ε_wall ≡ κ·u_tau·y`。该不变量同时约束两个公式与两个常数——
   任一常数被静默改动（C_μ 或 κ）都会红。
3. **负控**：把对数律侧 κ 换成 0.40，I1 必须破（实测偏差 2.44% > 2% 门）——
   证明判据有鉴别力，不是恒真断言。
4. **掩码选择性**：非近壁格点位相同（`torch.equal`）；近壁格点必须变。
5. **边界行为**：u_tau=0 全字段位相同（mask 守卫）；u_tau=1e-6 触发地板
   （k→1e-8、ε→1e-12）。
6. **`compute_near_wall_from_solid`**：平板/角点/孤立固体体素三种几何的精确掩码。

## 3. 计算结果（程序化可复算，零手抄）

| # | 判据 | 实测 | 门限 | 判定 |
|---|---|---|---|---|
| 1 | I1 不变量相对偏差 | < 1e-12（float64 舍入级） | 1e-12 | PASS |
| 2 | I1 负控（κ→0.40）偏差 | 2.439% | > 2%（必须破） | PASS |
| 3 | k_wall vs u_tau²/√C_μ | 8.3333e-3（=0.05²/0.3） | 1e-15 | PASS |
| 4 | ε_wall vs u_tau³/(κy) | 6.0976e-4 | 1e-18 | PASS |
| 5 | 标度律 k∝u²、ε∝u³（3 档） | 全档机器精度 | — | PASS |
| 6 | 非近壁格点位相同 | 位相同 | — | PASS |
| 7 | u_tau=0 无操作 | 位相同 | — | PASS |
| 8 | 地板（u_tau=1e-6） | k=1e-8、ε=1e-12 | — | PASS |
| 9 | 近壁掩码 3 几何 | 精确匹配 | — | PASS |
| 10 | y+ 有效性门 | y+=50 ∈ (30,600) | — | PASS |

**13/13 PASS**（`pytest -q` 6.3 s，CPU/float64，单线程）。

## 4. 结果图

**无图，理由**：本单为纯测试/文档任务（小张量闭式比对，无求解器运行、无场量），
无云图/曲线可产；全部数字由 pytest 断言程序化裁决，重跑即复算。

## 5. 注意点 / 局限（含两个真实发现）

1. **发现 F1（docstring 缺陷，已勘误）**：原 docstring 声称
   `ω_wall = u_tau/(√C_μ·κ·y)（for k-ω SST）`，但函数**无 omega 参数、从不写 ω**；
   且该公式与 k-ω 模型不自洽——代入 ν_t = C_μ·k/ω 得 ν_t = C_μ·κ·u_tau·y，
   仅为对数律值 κ·u_tau·y 的 **C_μ = 0.09 倍**（差 11 倍）。模型自洽值应为
   ω = √C_μ·κ·u_tau/y（= C_μ·k/ν_t）。本次只删错误公式并留指针，**未实现 ω**
   （避免引入未经验证的公式进库）；`test_omega_is_documented_but_not_implemented`
   钉住当前契约，未来实现须先改该测试。
2. **发现 F2（roll 掩码周期约定）**：`compute_near_wall_from_solid` 用 `torch.roll`
   取邻域——周期域正确，但**非周期域会把对面边界格点误标近壁**（平板测试中
   j=5 行被 j=0 固体经 wrap 误标）。与 `wall_function_3d` 同约定，非本单引入；
   非周期算例使用此掩码前须自行裁剪对面行。
3. 覆盖范围：仅 k-ε 壁值闭式与掩码逻辑；未覆盖 u_tau 上游（`wall_model` 对数律
   反演，已有独立测试面）、k-ω SST 全流程、y+<30 粘性底层（壁函数本身不适用，
   模块未做 y+ 分支——使用方须保证 y+ 在有效区）。
4. 本机 load15≈130 ≥ 100，按容量纪律只做轻量活；全部测量 CPU/float64。

## 6. 误差判定（vs 基准逐项，显式回答是否 < 3%）

| 关键量 | 基准 | 误差 | < 3% ? |
|---|---|---|---|
| k_wall | u_tau²/√C_μ | < 1e-15（机器精度） | **是** |
| ε_wall | u_tau³/(κ·y) | < 1e-18（机器精度） | **是** |
| I1 不变量 | κ·u_tau·y | < 1e-12 | **是** |
| I1 负控 | κ=0.40 必须破 | 2.439%（破，鉴别力成立） | 是（按设计破） |

**无外推、无修正因子**；所有数字为闭式参考 + 真实函数调用，零手抄。

## 7. 证据指针

- 测试：`tests/test_rans_wall_bc_reference.py`（13 用例，6.3 s）
- 勘误：`src/tensorlbm/turbulence/rans_wall_bc.py` docstring（零行为改动）
- 账本锚：`benchmarks/gaps.md` G21（ddes/des 由 m6583 裁，本单裁 rans_wall_bc）
