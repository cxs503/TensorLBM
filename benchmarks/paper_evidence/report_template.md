# 论文级基准证据包 — GAP-DIFFLBM-PAPER-EVIDENCE

> **本文档由 `benchmarks/paper_evidence/aggregate.py` 程序化生成**：正文模板在
> `benchmarks/paper_evidence/report_template.md`，第 4–6 节的所有数字由
> `runs/paper_evidence/*.json` 注入，**没有任何一个判据数字是手抄的**。
> 重新生成：`python benchmarks/paper_evidence/aggregate.py`。
> 机器可复核：`python -m pytest tests/test_paper_evidence_bundle.py -q`
> 会从同一批 JSON 重新推导每条判据的通过/失败并与本页对照。

---

## 1. 问题设定与基准出处

目标：为 TensorLBM 顶刊投稿提供**可独立复核的精度证据面**，要求
≥6 个相互独立的基准（解析 2 + 文献 3 + 交叉求解器 1），逐项给出 ref vs got 误差，
并**显式回答是否 <3%**。

证据分三类，互不依赖：

| 类别 | 独立性来源 | 基准数 |
|---|---|---|
| 解析解 analytic | 闭式解（抛物线、Boussinesq 双级数、剪切波衰减），无外部数据 | 3 |
| 文献 literature | 论文表格（Ghia 1982、Sangani–Acrivos 1982），表值随仓库存放并带 DOI | 3 |
| 交叉求解器 cross-solver | 三套**方法论不同**的离散（LBM / psi-omega FVM / Q2-Q1 FEM）解同一物理问题 | 1 |

外部锚点（external anchor）逐项出处见第 6 节每小节首部的 "锚点/出处" 行；
解析解的方程形式与文献表值的位置（文件:符号）同样列在那里，便于第三方直接
打开源码核对，而不是相信本文的转述。

## 2. 方法与实现

### 2.1 复用而非重写

证据包**不重新实现任何求解器**，只做仲裁与制表：

| 基准 | 实际执行代码（关键指针） |
|---|---|
| A1/A2/A3 解析 | `tests/test_lbm_verification.py::poiseuille_case / duct_poiseuille_case / shear_wave_case`（CI 断言的同一段代码，按路径 import，参数只改分辨率/步数） |
| L1/L2 Ghia 方腔 | `benchmarks/verified/cavity/re{100,400}/run.py`（已入库的 verified runner，子进程调用） |
| L3 圆柱阵列 | `benchmarks/verified/permeability/run.py`（Sangani–Acrivos 协议，子进程调用） |
| X1 交叉求解器 | `src/tensorlbm/cross_solver_verification/runner.py`（LBM/FVM/FEM 三方 + MMS + 能量/质量账本） |

这样做的好处：本证据包的数字与 CI 门禁、已入库 verified 案例**共用同一段代码**，
不存在"报告用一套、上线用另一套"的漂移面。

### 2.2 判据的定义

- 误差一律为**相对误差**：`|got − ref| / ref`（速度剖面类用解析峰值
  `u_max` 归一化，方腔用 `u_lid` 归一化，与 Ghia 论文的归一化口径一致），
  逐条列出，不用"平均误差"掩盖最差点。
- 门限：任务书主判据 **3%**；交叉求解器三方一致性用更严的 **2%**
  （`cross_solver_verification` 的 C2 判据），并在表中显式标注每条的门限。
- 每个基准至少两档分辨率（或两档参数点：φ=0.3/0.5），以便给出实测收敛阶。

### 2.3 运行方式

```bash
# 单项（每个 item 一个独立作业，可并行递交到调度队列）
python benchmarks/paper_evidence/run_evidence.py --list
python benchmarks/paper_evidence/run_evidence.py --item A1_poiseuille_2d
python benchmarks/paper_evidence/run_evidence.py --item-index 4      # 1-based

# 汇总：summary.json + tables.md + 图 + 本页
python benchmarks/paper_evidence/aggregate.py

# 一键全流程（含 pytest 门）
bash benchmarks/paper_evidence/repro.sh
```

## 3. 参数表（无量纲数完整）

每项的完整参数（τ、ν、Ma、Re、网格、步数、碰撞算子、边界条件）由 item JSON 的
`settings` / `per_resolution` 字段逐项记录，随工件一起入库：

| 基准 | 关键无量纲数 / 参数 |
|---|---|
| A1 | τ=1.0 → ν=1/6；H=30/62 格；`u_max` 目标 0.05 → Ma≈0.087；Re_H≈O(10)（体力驱动定常流） |
| A2 | τ=1.0 → ν=1/6；L=10/14 格（方截面）；`u_nominal`=0.05；Re 同上；另报告阻力 R=gL²/(ν·ū) 对文献 f·Re=56.91 |
| A3 | τ=1.0 → ν=(τ−½)/3=1/6；振幅 0.01；kx=2π/nx；100 步 |
| L1/L2 | Re=100/400；u_lid=0.06 → τ 由 `3·u_lid·n/Re+0.5` 定；n=128/192；10⁵ 步；MRT；Ma=u_lid·√3≈0.104 |
| L3 | φ=0.3/0.5（实际 φ 由掩码几何反算，JSON 中 `phi_actual`）；d=26/52/104 与 20/40/80；force_scale=10 → Re_cell≈0.5–0.6（Stokes 区） |
| X1 | Re=100，u_lid=0.1；n=64/128（LBM、FVM）、32/64（FEM）；MMS 检验观测阶 |

> 说明：上表的量纲/量级描述与 item JSON 中的实测值一一对应；任何"目标值"与
> "实测值"的差别都在第 6 节的 ref/got 列里，不用本文的概括替代。

<!--TABLES-->

## 7. 失败模式与工件

### 7.1 工件布局

| 路径 | 内容 |
|---|---|
| `runs/paper_evidence/<ITEM>.json` | 单项原始测量（含 `per_resolution`、`rows`、`env`、`wall_seconds`） |
| `runs/paper_evidence/*_raw.json` | 被复用 runner 的原始输出（方腔、交叉求解器） |
| `runs/paper_evidence/L3_cases/*.json` | 渗透率每档 (φ,d) 的 case JSON |
| `runs/paper_evidence/X1_fields.npz` | X1 三方速度场（云图来源） |
| `artifacts/paper_evidence/summary.json` | 汇总（机器可读，供 gate 复核） |
| `artifacts/paper_evidence/tables.md` | 全部数字表（本页第 4–6 节的来源） |
| `artifacts/paper_evidence/figures/*.png` | 图 1–7 |

### 7.2 已知失败模式与口径坑（如实披露）

1. **半路反弹的壁面位置偏移**：A1 的 `u_max` 与剖面误差在粗网格上略偏小，
   残差属"壁面位于首层流体节点外半格"的一阶滑移项，因此 A1/A2 的观测阶接近
   1（滑移主导）而不是 2；这是 half-way bounce-back 的固有行为，不是实现 bug，
   也是 A1/A2 只声明"误差 <3%"而不声明"二阶收敛"的原因。
2. **方腔的阶梯边界与格子 Mach 数**：L1/L2 用 u_lid=0.06（Ma≈0.104）换取
   更大的 τ 裕度；Ghia 表值是网格无关解，任何有限的格子 Ma 都会带来
   O(Ma²) 的可压缩性偏差，这部分误差不可通过加密网格消除。
3. **渗透率的求积口径**：L3 的 k_ref 由文献 f 表 + `k=d²/((1−φ)f)` 得到，
   其中 φ 必须用**实际掩码 φ**（`phi_actual`），而不是名义 φ；用名义 φ 会产生
   数十个百分点的伪误差（history: `err_prereg_A_pct` 字段即为该口径错误的对照记录）。
4. **方腔 Re=400 的收敛性**：两档均需 10⁵ 步；若步数不足，涡心位置比中心线速度
   更早"看似收敛"，因此本包同时记录 `last_resid` 作为噪声地板探针。
5. **未运行的部分（not_run）**：
   - Schäfer–Turek 2D-1（Re=20 圆柱绕流，Nabh 1998 谱参考 Cd=5.5795…）：
     本机无 GPU，D=40/80 各需 3×10⁵ 步，实测 ~4.2×10⁻² s/步（8 线程），
     单档 >3 h，超出本轮计算预算 → **本轮未运行，不记入证据面**；已入库的
     `benchmarks/verified/cylinder/re20_st/result.json` 是历史 GPU 运行结果，
     本页不引用其数字以避免"转载即证据"。
   - 原任务设想的"≥3 个文献基准"中若需圆柱绕流类，第二轮可用 GPU 队列补齐。
6. **文献表值的离线约束**：本环境无外网。Ghia(1982) 与 Sangani–Acrivos(1982)
   的表值取自仓库内已转录并带 DOI 的常量（`src/tensorlbm/lid_driven_cavity.py`、
   `benchmarks/verified/permeability/run.py`），本轮**没有**重新下载原文逐位核对；
   因此这三项的锚点强度是"仓库内转录 + DOI 可溯源"，不是"本轮现场核对原文"。
   若审计要求现场核对，请在有网环境重跑 `repro.sh` 后比对 DOI 表值。