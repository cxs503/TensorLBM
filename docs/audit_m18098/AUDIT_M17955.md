# 交差审计 · m17955 `XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY`（TensorLBM）

审计单 m18098（board-audit）· 被审单 m17955（status=succeeded）· 审计日期 2026-11-13
被审工作树 `/root/tp-agent/work/m17955`（`TensorLBM/` 子目录，未提交，`git status` 全为 `??`）
审计工作树 `/root/tp-agent/work/m18098`（master `fc86f508`，**不含**被审交付文件）

## 0. 结论（disposition）

**level = L2**（独立功能实现，真代码、真数值、经独立复算成立）；**不是 L3/L4**：

* L3 不达 —— 全树 `grep` 只有 xval 驱动、docs 与本模块导出（`src/tensorlbm/porous/__init__.py`）三处引用，
  `src/tensorlbm/solver.py`（+其它求解器）0 命中，**零集成调用者**；
* L4 不达 —— `tensor_benchmark_cases` **无该锚点行**（机器裁判缺席），且交付自带的证据链
  **不可由随交付的驱动复现**（见 F3：两条判据在 `results.json` 里与实跑结果正好反号）。

**功能实质是好的，台账是坏的**：经独立复算，稀疏臂槽道渗透率误差 −0.4375 % / −0.1094 % / −0.0274 %
（H=8/16/32，analyt. `k=H²/12`），与 lbmpy 生成核逐点差 3.35e-6（pdf 量级 0.446），地址表 5152 条目
100 % 坐标级一致。**交付自己报出来的 −11.50 %/−5.985 %/−3.073 %、68.9 % 逾渗差、`all_pass=false`
全部是后处理脚本 `make_report.py` 的 φ 记账错误造成的假数字**（F1/F2/F3）——方向上是把好结果报成坏结果
与把通过的判据报成失败，**不是造假，但是错的**，且 ⑤ 的收敛性解释（O(1/H) 壁面离散）被我的复算证伪。

## 1. 方法与独立性

| 项 | 做法 |
|---|---|
| 独立复算 | 直接 import 交付的 `sparse_step/build_index_table/macroscopic_sparse`，**自己**按单一约定算 `k_int=ν⟨u⟩_fluid/g`（不信交付 json 的键） |
| 驱动 | `docs/audit_m18098/verify_m17955.py`（槽道 H=8/16/32 + 同几何稠密控制臂）、`verify_perc_oss.py`（逾渗 24²）、`verify_case1_oss.py`（case-1 三臂核等价，独立调 lbmpy/pystencils） |
| 证据冻结 | `docs/audit_m18098/evidence_snapshot/`（9 文件逐字节 + `SHA256SUMS.txt`），被审工作树未提交、会被回收 |
| 参照 | 解析解 `k=H²/12`；OSS 臂 = lbmpy 2.0 `SparseLbMapper.create_index_array` + `create_lb_update_rule_sparse('stream_pull_collide')` 编译成 C 在本机实跑 |
| 未做 | waLBerla C++ 侧（`python/lbmpy_walberla/sparse.py`、`Permeability.h`、`Percolation.cpp`）：本机无 C++ 树 → **WITHHELD**，不作断言；未重跑交付的 216 s 全量（`run_xval.py` 全部算例），只重跑判据算例 |

## 2. 三件套取证

| 件 | 结果 | 指针 |
|---|---|---|
| run_artifacts | **在**，但 `results.json` 被后处理脚本改写 | `xval/XVAL-LBM-…/results.json`（44140 B, sha256 `e3a907ad…`）、`kpi.csv`（1297 B, `5a6d3e0d…`）、`run.log`（1187 B, `60eea32…`）、`figures/fig1..fig5.png`；`make_report.py:462` 直接 `json.dump` 覆写 results.json |
| benchmark ref vs got | **平台无案号**：`SELECT … WHERE anchor_slug='XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY'` → **0 行**（全表 84 行 / 71 锚，`%SPARSE%|%LIST%|%permeab%` 也 0 行）→ 无机器裁判 | 表 `tensor_benchmark_cases` |
| repro_cmd | 交付驱动本身可跑（`PYTHONPATH=src python3 xval/…/run_xval.py`，`run.log` 记 216.12 s），但**复跑结果 ≠ results.json**（F3） | 审计复跑：`python3 docs/audit_m18098/verify_m17955.py`（101.4 s，步数 1213/4519 与交付逐位相同） |

## 3. 独立复算 vs 交付自报（裁判口径：analyt. `k=H²/12`，单一 Darcy 约定）

| H | 交付自报 `k_int` | 交付自报误差 | **审计复算 `k_int`** | **审计复算误差** | 同几何稠密控制臂误差 | 稀疏 vs 稠密 ⟨u⟩ |
|---|---|---|---|---|---|---|
| 8 | 4.7200 | −11.500 % | **5.31000** | **−0.4375 %** | −0.4375 % | 1.1e-14 |
| 16 | 20.0565 | −5.985 % | **21.31000** | **−0.1094 %** | −0.1094 % | 1.4e-14 |
| 32 | 82.7110 | −3.073 % | **85.30999** | **−0.0274 %** | −0.0274 % | 8.1e-16 |

* 误差比 0.4375 → 0.1094 → 0.0274（比 4.00 / 3.99）→ **二阶收敛**，与半程反弹壁面（y=0.5）理论一致；
* 交付 docs ⑤.1 把 −11.50 %→−3.07 % 解释为"半程壁面 O(1/H) 一阶离散误差"——**被证伪**：
  该序列 = `1−φ = 1/(H+1)`（H=8/16/32 → 11.11 %/5.88 %/3.03 %）叠加真实 −0.44 %，
  同几何稠密臂（同一半程壁面）误差 ≤0.44 % 且同样按 O(1/H²) 走，壁面离散本身没有一阶误差。

逾渗（seed 11, 24², φ=0.592014, 6000 步，`verify_perc_oss.py`）：

| 量 | 交付自报 | **审计复算（同约定）** |
|---|---|---|
| k_int 稀疏 | 2.48493e-4（标签错位） | **4.197412e-4** |
| k_int 稠密 | 4.19741e-4（js 里另一键） | **4.197412e-4** |
| 稀疏 vs 稠密相对差 | 68.915 %（c5 FAIL） | **8.12e-11** |
| k_super = φ·k_int | 1.47111e-4（=φ²k_int） | **2.484926e-4**（双臂一致） |

case-1 核等价（24×32 封闭箱 + 障碍，100 步，g=1e-4，独立跑 `oss_ref.OssSparseStepper`）：

| 配对 | 交付 `results.json` | **审计复跑（随交付代码）** | 判据 |
|---|---|---|---|
| 稀疏 vs **OSS 生成核**（同为碰撞后态） | 5.8957e-3 → c3 **FAIL** | **3.352207e-6**（= `run.log` 首轮值，逐位相同） | c3 <1e-4 → **实际 PASS**（相对 pdf 量级 7.5e-6） |
| 稀疏 vs 稠密 | 4.996e-16 → c1 **PASS** | **3.3287e-4** | 状态约定差（稠密臂存流式态），非物理差 |
| 地址表 | 5152/5152 | 100 % 邻居 cell + 100 % 方向（对 lbmpy `SparseLbMapper`） | c2 PASS |
| 剖面 vs 解析抛物线 | 49.4/49.9/50.0 %（kpi 标 PASS, ref=0） | **0.72 % / 0.19 % / 0.049 %**（把解析抛物线归一到同一均值后） | 交付的解析剖面峰值只有真峰值 2/3 → 那 ~50 % 是归一化假象 |

## 4. 发现（按严重度）

* **F1（blocking，记账/报告）** 交付数字不是驱动的输出：`make_report.py:111-130 fix_darcy()` 把
  `k_intrinsic` 再乘一次 φ（`:116` 注释自认 `stored = (ν⟨u⟩/g)/φ`，却又 `k_int = row["k_intrinsic"]*phi`），
  于是 H=8 报 4.72 = φ·5.310。`meta.first_pass_defects.darcy_double_phi` 声称"已修"，实际是把错项换了个方向。
* **F2（blocking，判据）** c5 的 68.9 % = `1−φ²`：`fix_darcy` 把 case3 的 `k_super_sparse` 写成 φ²k_int、
  `k_super_dense` 保持 φk_int，两把不同尺子相比。同约定复算 8.1e-11 → c5 实为 PASS。
* **F3（blocking，可复现）** `results.json` 的 case-1 两行与随交付代码对不上（3.35e-6/3.33e-4 vs 5.90e-3/5.0e-16），
  c1 与 c3 在交付台账里**正好互换**。故 `all_pass=false` 不可复现，也不是代码的真实行为。
* **F4（blocking，集成/测试）** 零集成调用者；新模块**无任何 pytest**（`grep -rln 'sparse_index_lbm|build_index_table' tests/` → 0）；
  既有 `tests/test_porous_fluid_node_table.py` 12 passed/31.45 s（不覆盖新模块，只证 `__init__` 改动未打断旧路径）。
  交付文件在其工作树内**未提交**（`git status` 四个 `??`），审计工作树 master `fc86f508` 内**不存在**该模块 → 能力尚未入库。
* **F5（major，流程）** 平台无 `tensor_benchmark_cases` 案号 → L4 无机器裁判；同时 mission `status=succeeded`
  而交付自报 `all_pass=false`，状态与判据互斥。
* **F6（minor，数值卫生）** H=32 臂 9000 步 = `max_steps` 上限，未达自设 tol 1e-10（复算 16662 步收敛后 k=85.31000，
  交付 85.29570，差 0.017 %）。
* **F7（minor，指标定义）** `profile_max_rel_err` 以 ref=0、err≈50 % 记 PASS，指标本身无定义（F1 同源的归一化问题）；
  正确归一后剖面误差 0.049 %–0.72 %。

## 5. 未跑 / 未知（不许当通过）

* waLBerla C++ 侧行为（`Permeability.h` / `Percolation.cpp` / MPI ghost 处理）：本机无 C++ 树 → **WITHHELD**。
* 3D（D3Q19）稀疏收益、内存比：交付未测，本审计也未测 → 未知。
* OSS 臂周期域：`SparseLbMapper.create_index_array` 在纯周期域抛 `IndexError: index 4294967295`（`mapping.py:131`）
  → 交付 `oss_probe.periodic_free_case=False` 属实；本审计未找到绕过办法，周期域只有 S+D 两臂（不是三臂）。

## 6. 升级路径（给返修单）

1. 删掉 `make_report.py` 的 `fix_darcy()` 改写，改为在驱动里一处显式 `k_super = φ k_int`、全表打印平均域；
   `results.json` 只由 `run_xval.py` 写、`make_report.py` 只读。
2. case-1 明写状态约定（三臂都用碰撞后态），并把 c1/c3 判据重新绑定到实测值（c3 实为 PASS）。
3. 补 `tests/test_porous_sparse_index_bm.py`（反弹/周期/表语义/与稠密等价），并给 `build_index_table` 加
   纯周期域回归（避开 OSS 的 uint32 缺陷，本仓支持周期）。
4. 接一个真调用者（`solver.py` 的 porous 路径或 `porous.PeriodicPorousStep`），再注册
   `tensor_benchmark_cases` 行（metric=channel k vs H²/12，tolerance=15 %）→ 才可能 L3/L4。