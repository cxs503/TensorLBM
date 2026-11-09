# gci_v4 — pre-registration & protocol record
## SUBOFF 裸艇（AFF-1）锚 Re 下的分辨率收敛与 GCI（课题 lbm-suboff-tau，research_plan 第 2 项）

> **诚实时间线声明**：本文件在**数据产生之后**整理成文，因此不是严格意义的"跑前冻结"
> 预注册。属于跑前即已确定、无事后调整的部分是：(a) 判据 C1–C4 逐字来自任务书
> （不由本实验定义），(b) 锚点选择（DTRC Table 14 AFF-1 最低速点）与 Re 匹配方式
> （`nu_lb = u_lb·L/Re_anchor`），(c) 阶梯 L=96/112/128、域公式、MRT/D3Q19、力估计器
> `standard`、步数 150·L —— 这三项在 `gci_v4_run.py` 首次运行（L=96 冒烟）之前就写在
> 脚本里，脚本 sha256 见 `gci_v4.json::provenance.scripts`。属于**事后追加**的部分是
> 探针臂（`gci_v4_probe.py`，L=48 的 Re 扫描与锚 Re 400 步臂）与归档阶梯的 GCI 复用，
> 它们是在阶梯**不可执行**（见下）被发现之后才设计的，故明确标为探索性证据，
> 不参与任何达标声明。

## 1. 判据（任务书原文，冻结）

| 编号 | 判据 |
|---|---|
| C1 | 最细档 \|C_T − 锚\| ≤3% |
| C2 | 三档 C_T 单调收敛（符号一致） |
| C3 | 实测收敛阶与 GCI ≤5% |
| C4 | Re 相对锚偏差 ≤2%（否则该档数据不得用于 3% 判决） |

判决只读 `gci_v4.json`，其它文件均为原始数据/探针，不进入判据。

## 2. 锚（先锁后用）

- 数据：H.-L. Liu and T. T. Huang, *Summary of DARPA Suboff Experimental Program Data*,
  CRDKNSWC/HD-1298-11, ADA359226, **Table 14, report p.23**, DOI `10.21236/ADA359226`。
- 本仓逐字转写行（`docs/suboff_experimental_resistance_validation_2026-08-01.md`）：

  `| AFF-1 裸艇 | 5.92 / 10.00 / 11.84 / 13.92 / 16.00 / 17.99 | 87.40 / 242.2 / 332.9 / 451.5 / 576.9 / 697.0 |`

  同表亦冻结于 `src/tensorlbm/suboff_reference_data.py::SUBOFF_TOW_TANK_RESISTANCE_TABLE14`。
- 选用点：最低速锚 **V = 5.92 kn = 3.045509 m/s，R_T = 87.40 N**（该锚 Re 最低，是
  LBM 最有利的一端；选更高速只会更难）。
- 归一化：ρ = 1000 kg/m³、S = 5.998 m²（DTRC 模型湿表面积，锁在
  `benchmarks/pending/suboff_highre_resistance/refs_lock.json::S1_Table1_verbatim`；
  与仓内解析回转面求积 `suboff_analytic_wetted_area` = 0.31576·L² 相差 0.08%，互为交叉验证）。
- 由此 **C_T^anchor = 0.0031417**（脚本注入，`gci_v4.py::anchor_numbers`），
  **Re_anchor = V·L_max/ν = 1.16474e7**（ν = 1.139e-6 m²/s）。
- 形式因子自检：C_T/Cf_ITTC(Re_anchor) = 1.075（Cf_ITTC = 0.002923），潜体 1+k ≈ 1.07–1.10，量级合理。
- **溯源局限（如实）**：本环境离线，无法重新打开 ADA359226 原始 PDF 逐字核对；工件中
  的外部量取自仓内已冻结的逐字转写（含报告页码与 DOI）与锁定的 S1 Table 1 数值。

## 3. 阶梯协议（脚本冻结）

| 项 | 值 |
|---|---|
| 分辨率 | L = 96 / 112 / 128（每艇长格数） |
| 域 | padding (1,4,1,1,1,1) → 6L × 2.108L × 2.108L；胞数 23.74M / 37.75M / 56.52M |
| 格点 | D3Q19 |
| 碰撞 | MRT，**无亚网格模型**（closure 作为固定量，不参与精化轴） |
| Re 匹配 | u_lb = 0.05，ν_lb = u_lb·L/Re_anchor ⇒ Re 逐档精确等于锚 Re |
| 步数 | 150·L（= 7200…9600…，与已验收 Re=1000 阶梯同口径） |
| 力 | 每 10 步采样；摩擦估计器 `standard`（勘误 E2 校准过的仪器）；p0 = near_wall |
| 参考面积 | 裸艇解析湿面积 0.31576·L_lb²（**不用** π·D·L 圆柱代理；后者大 15.9%，会把 C_T 系统性压低 15.9%） |
| 观测窗 | 末 1000 样本（=末 10000 步）均值 |

## 4. 归档复用臂（Re=1000，同一 L 三档）

GCI 机器需要真实三档数据。锚 Re 阶梯不可执行（见 §5），故复用本仓库内两个**已存在**的
三档真实运行：

- `AFF3_sail_Re1000_standard`：`result.json::e2_standard_verdict_PASS`（已验收，勘误 E2 主口径）；
- `AFF1_bare_Re1000_mix50`：`diag_bare_L{96,112,128}/result_L{L}.json`（裸艇，旧 mix50 口径）。

两者锚均为 Blasius `Cf = 1.328/√Re`，**与 DTRC 锚无关**，因此只用于（i）演示 p/GCI 机器，
（ii）量化"这些分辨率下三档是否处于渐近区"，**不得**用于 C1 达标声明。

收敛阶用非均匀细化比 Roache 公式（r21 = 112/96、r32 = 128/112）反解 p，
并给出 `R < lim_{p→0} R` 时"非渐近区"的判定（p ≤ 0 则 GCI 无定义，如实报"无定义"而非夹值）。

## 5. 停止规则与偏差申报

- 每档墙钟上限：本单可用计算窗 ≈ 数小时；实测 L=96 单步 15.1–20.5 s（CPU torch eager，
  宿主负载均值 250+，无任何加速器），150·L 步 ⇒ **单档 61 h**（L=96）、96 h（L=112）、
  144 h（L=128）。三档合计 ≈ 300 h ⇒ 冻结协议**在本环境不可执行**，属于
  "诚实 FAIL（不可执行）"，不降档、不改判据、不缩短步数偷跑后声称达标。
- 偏差申报：探针臂在 **L=48**（非阶梯档）上运行，仅用于定位 Re 依赖与仪器可执行边界，
  不产生任何达标声明；探针步数 20（Re 扫描）与 300/400（锚 Re 臂），均如实记录。

## 6. 收据

- 脚本 sha256 与输入文件 sha256：见 `gci_v4.json::provenance`。
- 原始探针数据：`benchmarks/verified/suboff_sail_resistance/artifacts/gci_v4_probe*.json`。
- 归档阶梯：`result.json`、`diag_bare_L*/result_L*.json`（本目录，均带运行时日志）。