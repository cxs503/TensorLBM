# JOUR-DIFFLBM · 文献锚补强 Ghia Re=400 —— 返修轮1 交付报告（m21138）

GAP：`GAP-DIFFLBM-EXT-LIT-GHIA400` · 仓库 TensorLBM · 分支 `agent/m21138`
事实基线：`git fetch origin && git log origin/main` → `6b6dbc3fc Merge #2430 (m21073)`（本地树与远端 main 一致，
`git status` 干净）。前一轮 m20981（commit `c6b088889`）只交付了文档/JSON，被判 L1。

---

## ① 问题设定 + 基准出处

* 物理：二维顶盖驱动方腔（1×1），顶盖 u=U_lid，其余三壁无滑移；Re = U_lid·H/ν = 400。
* 基准：**Ghia, U., Ghia, K.N., Shin, C.T. (1982), J. Comput. Phys. 48(3):387-411**，
  Table I（x/L=0.5 竖中线 u）、Table II（y/L=0.5 横中线 v），129×129 网格结果，p.398-399。
* 证据链（BOUND）：OCR 文本层见证
  `benchmarks/verified/cavity/re5000/reference/ghia1982_tables_I_II.ocr.txt`，
  sha256 `3637ef9ef72a76bb090f4e31765ed2870d18db67f0f99611730e61927e060f5b`
  （PDF sha256 `1d76e463…`，出处 nilot-pal/Lid-driven-cavity 公开副本）；
  → `benchmarks/anchors/ghia1982_provenance.json`（逐点转录，17+17 点/Re）；
  → `benchmarks/anchors/data/ghia1982_re{100,400}_table{I,II}.csv`（**机械生成**，sha256 逐文件钉死）。
  两级哈希链 + 数值回转校验，均由 `benchmarks/anchors/run_anchors.py` 机器执行。

## ② 方法与实现（方程/格式 + 代码指针）

| 环节 | 实现 | 指针 |
|---|---|---|
| 求解器（平台路径） | D2Q9 **MRT** + 三静止壁 pre-streaming 半程反弹（V3）+ Zou/He 动壁（`tensorlbm.solver.collide_mrt/stream`、`tensorlbm.lid_driven_cavity.zou_he_moving_lid`） | `benchmarks/verified/cavity/re400/run.py` |
| 求解器（验证模块路径，RLBM） | `tensorlbm.verification.lid_driven_cavity.CavitySolver`（regularized BGK，半程反弹，Zou-He 顶盖，速度爬升 ramp） | `TensorLBM/src/tensorlbm/verification/lid_driven_cavity/solver.py` |
| 锚数据生成 | provenance JSON → CSV（禁止手抄） | `benchmarks/anchors/make_ghia_csv.py` |
| 锚闸门 | 6 步：见证 sha256 / CSV 逐文件 sha256(+值回转) / 表结构 / 剔除清单登记 / **代码常量 vs BOUND 漂移守卫** / OCR 见证值存在性 | `benchmarks/anchors/run_anchors.py` |
| 打分 | 先过闸门再读 CSV；M1/M2/M3 三口径（阈值预声明） | `benchmarks/verified/cavity/re400/anchor_check.py` |
| 回归测试 | 9 项（闸门、篡改检测、常量漂移、度量解析解） | `tests/test_ghia_re400_anchor_lit.py` |

**禁重复实现清单（动手前既有实现）**

| 既有实现 | 位置 | 本轮处理 |
|---|---|---|
| Re=400 V3/MRT 运行器 + `result.json`(1.50%→0.83%) | `benchmarks/verified/cavity/re400/run.py` | **复用**，仅新增 profile 落盘 |
| RLBM 验证求解器（m20981 时代） | `TensorLBM/src/tensorlbm/verification/lid_driven_cavity/` | **复用**（本轮真实运行即走此路径） |
| 锚闸门 + provenance（m20981 落点） | 原 `TensorLBM/benchmarks/anchors/` | **迁移**至仓根 `benchmarks/anchors/` 并扩展（消除嵌套目录副本） |
| 方腔家族 README/结果 | `benchmarks/verified/cavity/README.md`, `re100/`, `re1000/` | 不改，作为同族基线引用 |
| `GHIA_RE*` 内联常量（3 处副本） | `src/tensorlbm/lid_driven_cavity.py`、`lid_driven_cavity_3d.py` | **修缺陷**（见 ⑤），不新建副本 |

## ③ 计算结果（机器注入，零手抄）

闸门（`python benchmarks/anchors/run_anchors.py`）：

| 检查 | 结果 |
|---|---|
| [1] 见证 sha256 | PASS |
| [2] CSV 锚 **4/4** sha256 + 值回转 | PASS |
| [3] 表结构（17+17 点、单调、归一、壁/盖边界） | PASS（Re=100/400/1000） |
| [4] 剔除清单登记（reason+action） | PASS（4 条） |
| [5] 代码常量 vs BOUND（4 个常量） | PASS |
| [6] OCR 见证值存在性 | PASS（102/102，其中 101 精确、1 仅尾数匹配，OCR 符号丢失） |

Re=400 打分（`anchor_check.py`，读 `artifacts/run_raw_n97_steps12000.json`，n=97/RLBM/12000 步）：

| 网格 | 步数 | M1 max-point 偏差(<5%) | M2 严格逐点相对(<5%) | M3 u_min 相对 | M3 v_min 相对 | 判定 |
|---|---|---|---|---|---|---|
| 97 | 12000 | **26.46%** | 268.57%（条件良好的 22 点） | 17.42% | 58.81% | **FAIL** |

单调收敛证据：该 run 下 u_min = -0.2957（Ghia -0.32726）、v_min = -0.2505（Ghia -0.44993）——
**远未收敛**（12000 步 + ramp 4000，方腔 Re=400 通常需 O(1e5) 步），故上述数字只证明「打分链打通」，
不构成精度结论。产物：`artifacts/ghia_re400_anchor.json`、`artifacts/report.md`、`artifacts/profiles.png`。

## ④ 结果图

* `benchmarks/verified/cavity/re400/artifacts/profiles.png`——Ghia 1982 BOUND 表值（黑方块）vs
  LBM 中线型线（实线），左右分别为竖中线 u、横中线 v；标题带 M1 数值。
  （图为 matplotlib 程序生成，随工件入库。）

## ⑤ 注意点（踩坑 / 局限 / 适用范围）

1. **真缺陷（本轮修）**：`src/tensorlbm/lid_driven_cavity.py` 与 `lid_driven_cavity_3d.py` 的
   `GHIA_RE400["v"]` 有 4 个点从 2026-05-23（`3d5ef2fa6`）起就与见证不符：
   `v(x=0.9063)=-0.38598`（0.8047 的值被复制，正确 -0.23827，|Δ|=0.14771 ≈ 33% 极值）、
   `v(x=0.1563)=+0.28003`（=+0.28124）、`v(x=0.0938)=+0.22980`（=+0.22965）、
   0.2266/0.2344 两点 (+0.30174/+0.30203) 互换。**旧 `result.json` 的 0.83% 是拿这份漂移表打的分。**
   已按 BOUND 修正，并新增 [5] 漂移守卫 + 回归测试，防止复发。
2. **根求解路径不稳定**：`benchmarks/verified/cavity/re400/run.py` 的 V3/MRT 配方在本地复跑
   （nx=64/96，u_lid=0.01–0.06，tau 0.51–0.54，含 2000 步 ramp）在 100–600 步内 ρ 发散到 NaN。
   说明现在仓库里的 V3 配方**不是**产出 `result.json` 时的代码状态（或依赖已变）。本轮真实运行改用
   验证模块 RLBM 求解器（稳定），该不稳定已单列，未在本轮预算内修完。
3. **度量口径**：M1 = max|ΔU|/U_lid（极差相对速度尺度，方腔家族既有口径 `max_abs_dev_pct`）；
   M2 = 严格逐点 |Δ|/|ref|，在 |ref| < 0.05·峰值处病态（1e-3·U_lid 绝对误差即 20%–100%），
   病态点全部列出而非静默丢弃；M3 = 两极值点相对误差。三口径阈值在跑之前写死在代码里。
4. 适用范围：以上为 D2Q9 低马赫不可压；Re≥1000 需 RLBM/更高网格；3D 展向周期另有 `lid_driven_cavity_3d.py`。

## ⑥ 误差判定（是否 <3% / <5%）

| 关键量 | 基准 | 本轮值 | 误差 | <5%? | <3%? |
|---|---|---|---|---|---|
| M1 max-point 偏差（n=97, 12000 步） | Ghia 1982 Re=400 | 0.2646 | 26.46% | **否** | 否 |
| u_min（n=97） | -0.32726 | -0.29572 | 9.63% | 否 | 否 |
| v_min（n=97） | -0.44993 | -0.25049 | 44.3% | 否 | 否 |

**结论：验收判据「剔除 SUSPECT_ENTRIES 后 max-point 相对误差 <5%」本轮未达成（honest-FAIL）。**
锚侧（文献锚、哈希链、剔除清单、闸门、度量、测试、工件）已全部落地可复核；差值来自求解器侧收敛不足。

**fail_stage**：`solver_convergence`
**failure_reason**：方腔 Re=400 需 O(1e5) 步收敛，本机（99 控制面）只准 <8 核 / <10 min 短测，
12000 步的 n=97 RLBM 解未收敛（u_min 9.6%、v_min 44% 偏差）；已递交集群作业
`cj3214`（n=65/n=97，各 120000 步，32 核，stage_dir=本工作树，命令含 `anchor_check.py`），
但排队 20+ 位、3.5 min 仍 `queued`，本单预算内未取回工件。
另：根 V3/MRT 运行器当前配方 NaN 发散，需先修复或改用 RLBM，才能承接 128²/192² 的正式跑分。
**第2轮建议输入**：①取回 `cj3214` 工件后直接 `anchor_check.py` 重打分（无需改码）；
②对 128²/192² 用 RLBM + ramp 6000–10000，步数 ≥2e5，两档收敛；③排查 V3 配方发散（corners 同时被
`zou_he_moving_lid` 与 pre-bounce 处理、MRT 参数与 `stream` 约定是否与 2026-08 时一致）。

## 复现命令（一键）

```bash
# 1) 锚闸门（含 CSV 4/4 sha256、代码常量漂移守卫、OCR 见证值存在性）
python benchmarks/anchors/run_anchors.py

# 2) 重新生成锚 CSV（机械生成，非手抄）
python benchmarks/anchors/make_ghia_csv.py --check

# 3) 打分（读已入库运行工件，秒级）
python benchmarks/verified/cavity/re400/anchor_check.py \
    --run-json benchmarks/verified/cavity/re400/artifacts/run_raw_n97_steps12000.json

# 4) 重跑求解（>10 min，走集群；勿在控制面跑）
PYTHONPATH=$PWD/TensorLBM/src python -c "<见 cj3214 命令>"

# 5) 回归测试
python -m pytest tests/test_ghia_re400_anchor_lit.py -q   # 9 passed
```