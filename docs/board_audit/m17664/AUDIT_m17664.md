# Board audit — m17603 `XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE` (TensorLBM)

Audit mission: **m17664** · auditor role: `board-audit` (independent re-derivation, no re-do of m17603)
Mission under audit: **m17603**, status `succeeded`, `finished_at = 2026-11-13 07:33:00Z`
Audited worktree: `/root/tp-agent/work/m17603` (branch `agent/m17603`, base `c05d7af3`)
Anchor / machine-referee slug: `XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE`
Verdict: **L2** (`request_changes` — the numeric work is real; three structural gates block promotion)

---

## 1 案例介绍（物理背景／几何／参数／对比对象）

**物理**：Inamuro 湿节点（wet-node）边界族——边界节点仍是*流体*节点，从域外入射的分布函数按壁面状态重构。被验证的两类闭包：

| 闭包 | 公式（D2Q9） | 上游文件 |
|---|---|---|
| 速度 Dirichlet（anti-bounce-back / 非平衡反弹） | `f_ī = f_ī^eq(ρ_w,u_w) + f_i^eq(ρ_w,u_w) − f_i` | `chapter5/couette_wetnode.m`, `poissonille_wetnode.m` |
| 标量 Dirichlet（residual-weight，质量自洽） | `f_i = w_i (c_target − Σ_{j∉U} f_j) / (Σ_{j∈U} w_j)` | `chapter8/film_inamuro.cpp` |
| 零法向梯度（出口/自由面） | `f_i = w_i (Σ_{j∉U} f_j) / (Σ_{j∈U} w_j)` | 同上 |
| 角点平均 | `f(x_c) = ½[f(x_c+e_a)+f(x_c+e_b)]` | 同上 |

**算例（film，唯一被外部对比的一臂）**：上游 `film_inamuro.cpp` 的降落薄膜浓度输运。几何 `nx=40 × ny=400`（`iX` 为跨膜方向），对流速度**冻结**为抛物线剖面 `u_y(x)=u0(1−x²/(nx−1.5)²)`，只有浓度被 LBM 输运。

| 参数 | 值 |
|---|---|
| `nx, ny` | 40, 400 |
| `u0` | 0.05 |
| `c_wall` / `c_inlet` | 1.0 / 0.5（入口行 `iY=0, iX=1..NX−2`） |
| `omega` | 1.4 |
| 步数 | 主跑 12000（上游默认 N=50000；审计另跑 50000 上游二进制） |
| 判据 | `rel L2 < 15 %`（owner 精度标准 + driver 内声明） |

**外部锚点（零手抄，真实执行）**：`https://github.com/lbm-principles-practice/code` @ `6e2c592fdc3592c14dfd52f860fc1ceea930bcb0`，`chapter8/film_inamuro.cpp`（Krüger et al., *The Lattice Boltzmann Method: Principles and Practice*, Springer 2017, MIT）。审计期把归档源码 `artifacts/m17603_oss_src/chapter8/film_inamuro.cpp` 用 `g++ -O2` **重新编译并重新运行**，不依赖 m17603 归档的任何二进制。

---

## 2 怎么做的（独立复算流程，A1–A7）

全部步骤由 `docs/board_audit/m17664/independent_audit.py` 程序化产出（`audit.json`），审计脚本**不 import 也不修改** m17603 的交付物，只以 `importlib` 加载其 driver 以获得 `run_film` / `read_oss_field` / `rel_l2` 三个纯函数。

| 探针 | 做法 | 目的 |
|---|---|---|
| A1 ref authenticity | `g++ -O2 artifacts/m17603_oss_src/chapter8/film_inamuro.cpp` → 在独立目录跑满 N=50000，对 4 个 checkpoint 的 `.dat` 做 sha256 与 m17603 归档比对 | 判参考场是否被篡改／伪造 |
| A2 documented repro | 逐字执行 README §Reproduce step 2，输出到**全新** outdir | 判 `repro_cmd` 是否真可执行 |
| A3 fresh run | 用 m17603 的生产模块 `tensorlbm.boundary.inamuro_wetnode` 重跑 film 12000 步（checkpoint 100/1000/10000/12000），对**审计自建**的参考场算 rel L2/Linf | 判 got 数值真伪 |
| A4 cross-check | 与 m17603 归档 `xval_inamuro_wetnode.json` 的 `rel_l2` / `got_mean` 逐项差 | 判归档数字与实时行为是否一致 |
| A5 control probe | 把 `inamuro_wetnode_scalar` 换成恒等映射（删掉标量 Dirichlet 闭包），重跑到 1000 步 | 判指标是否有鉴别力（1e-11 不是空过） |
| A6 coupling | `grep -rn inamuro src/`（排除模块自身）＋检查 `boundary_registry.py` / `__init__.py` / `boundary_capability_contract.py` | 判是否接入求解器（L3） |
| A7 ledger | `psql` 查 `tensor_benchmark_cases`（anchor 裁判行）、`tensor_reports`（mission 17603）、`git status` 交付物 | 判平台级登记/裁判/可复现身份 |

环境：单机 CPU，torch 2.x float64，`g++ -O2`；总墙钟 ≈ 8 min（A3 50 s、A5 4 s、上游二进制 94 s、其余秒级）。

---

## 3 计算结果（数字表，全部由 `audit.json` 程序化注入）

### 3.1 got vs 参考场（审计自跑，参考场为审计自己编译+运行的上游二进制）

| step | got_mean | ref_mean | rel L2 | rel Linf | <15 % ? | 归档 rel L2 | |Δ| vs 归档 |
|---|---|---|---|---|---|---|---|
| 100 | 0.09536901896812995 | 0.0953690189676046 | 3.1747e-11 | 4.9975e-11 | PASS | 3.1747e-11 | **0.0** |
| 1000 | 0.27623894781231934 | 0.27623894781020153 | 5.1186e-11 | 3.5167e-10 | PASS | 5.1186e-11 | **0.0** |
| 10000 | 0.7312817237171253 | 0.7312817237171937 | 4.3562e-11 | 3.8147e-10 | PASS | 4.3562e-11 | **0.0** |
| 12000 | 0.7602524632738576 | 0.7602524632739562 | 4.2212e-11 | 3.6127e-10 | PASS | 4.2212e-11 | **0.0** |

偏差在 12000 步内**不增长**（3.2e-11 → 4.2e-11，对数尺度上平坦），与"两个独立实现、同一 IEEE-754 双精度算法序"的解释一致（不是混沌放大的物理量，是稳定的线性标量输运）。

### 3.2 参考场真伪（A1，sha256）

| 文件 | 审计新编 g++ 二进制 | m17603 归档 | 结论 |
|---|---|---|---|
| `film_inamuro0000100.dat` | `cae64452…67f9f` | `cae64452…67f9f` | MATCH |
| `film_inamuro0001000.dat` | `566a7aa6…31d0e` | `566a7aa6…31d0e` | MATCH |
| `film_inamuro0010000.dat` | `b77c1446…da8ac` | `b77c1446…da8ac` | MATCH |
| `film_inamuro0012000.dat` | `1f40a309…f37350e3` | `1f40a309…f37350e3` | MATCH |

→ 参考场是**真上游代码真跑**的产物，未被人为扰动。

### 3.3 鉴别力控制（A5）

| 探针 | 结果 |
|---|---|
| 标量 Dirichlet 闭包 → 恒等（step 1000） | rel L2 = **0.99666**（判 FAIL） |
| 未破坏（step 1000） | rel L2 = 5.12e-11 |

→ 该指标对"闭包缺失"有 10 个数量级的鉴别力；1e-11 不是指标失效，是真一致。

### 3.4 单元/性质测试（审计重跑）

`PYTHONPATH=src python3 -m pytest tests/test_inamuro_wetnode.py -q` → **14 passed in 3.81 s**。用例断言的是不变式而非"跑一遍就算过"：面方向集（`unknown_directions`）、标量 Dirichlet 质量自洽、`all_but_rest` 变体残留 `f_0`、零梯度保均匀场、闭包只动本面、ABB 保持壁面自洽平衡态不变、ABB 在静场+静壁为恒等、ABB 在边界节点上施加壁速、`rho_w` 外推一致性、角点平均对齐上游块、inplace 语义、格点配对一致性。

### 3.5 台账事实（A7）

| 项 | 值 |
|---|---|
| `tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE'` | **0 行**（无机器裁判行 → 无预声明容差） |
| `tensor_reports WHERE mission_id=17603` | **0 行**（无论文级报告） |
| `tensor_dev_missions id=17603` | `succeeded`, `finished_at=2026-11-13 07:33:00Z` |
| `git status`（交付物） | `?? src/tensorlbm/boundary/inamuro_wetnode.py`、`?? tests/test_inamuro_wetnode.py`、`?? benchmarks/verified/inamuro_wetnode_xval/`、`?? artifacts/m17603/` → **全部未 commit**（`git ls-files` 为空） |
| `grep -rn inamuro src/`（排除模块自身） | **0 命中**；`boundary_registry.py` / `__init__.py` / `boundary_capability_contract.py` 均未登记 |

---

## 4 结果图

`docs/board_audit/m17664/fig_audit_m17664.png`（由 `gen_figure.py` 从 `audit_fields.npz` + `audit.json` 程序化生成）：

- 上排：step 12000 浓度场 —— 左＝审计经生产模块重跑，中＝审计重新编译的上游二进制，右＝逐点 `|Δc|`（log 色标，max 1.2e-10）。
- 下排：100/1000/10000/12000 四个 checkpoint 的 rel L2 / rel Linf（log 轴）＋ 15 % 验收线 ＋ 控制探针点（0.9967）。

图中两条曲线与 15 % 线相距 10 个数量级，直观说明"通过"不是靠宽松容差。

---

## 5 注意点（发现清单，按严重度）

| # | 严重度 | 发现 | 证据 |
|---|---|---|---|
| F1 | **blocking（可复现身份）** | 三件套（模块/测试/benchmark/artifacts）在 `agent/m17603` 上**全部 untracked**，`git diff --stat HEAD` 为空。平台的自动 push 只会推 commit，因此这批交付**不会被推到 GitHub**；任何"从仓库检出复现"的路径都不成立，工件只活在这一个临时 worktree 里（可被 GC）。 | `git status --short`（A7）；`git log -- src/tensorlbm/boundary/inamuro_wetnode.py` 空 |
| F2 | medium（`repro_cmd`） | README §Reproduce step 2 在**全新** outdir 上必然崩：`part_film()` 把参考二进制编译成 `<out>/oss_ref/film_inamuro`（文件，run.py:372-375），随后 `run_oss_reference()` 又 `mkdir` 同名路径当**目录**（run.py:204-205）→ `FileExistsError`。归档 `artifacts/m17603/oss_ref/` 是手工把二进制改名 `film_inamuro_bin` 后才成立的布局，driver 自己从不产生这种布局。 | `a2_recorded.json`（64 s 内复现，traceback 全文） |
| F3 | medium（覆盖面） | 锚点名含 **anti-bounce-back**，但 external xval 只覆盖*标量* residual-weight 闭包（`film_inamuro.cpp`）；anti-bounce-back 速度闭包一臂 `velocity.status = "not_run"`（driver 自认 "open defect: channel time loop does not constrain the wall rows across steps"），其证据只有 `tests/test_inamuro_wetnode.py` 的性质测试。锚点声称的主能力缺外部基准。 | `artifacts/m17603/xval_inamuro_wetnode.json` `velocity` 块；`run.py:446` `part_velocity()` docstring |
| F4 | medium（裁判） | 无机器裁判行（0 行）→ 判据 15 % 是 owner 标准在 driver 里自声明，不是预登记容差；平台侧无法机器复核"是否在容差内"。 | A7 查询 |
| F5 | medium（集成） | 0 个模块外调用点，未进 `boundary_registry` / `__init__` / 能力契约 → solver 侧无法通过 API 使用；交付实质是"库模块 + 对比 driver"，不是可调用的平台能力（L3 不成立）。 | A6 |
| F6 | minor（API 陷阱） | `run_film()` 的默认 `scalar_policy="all_but_rest"` 恰是**上游从不调用**的错误变体（残留 `f_0` 使节点值偏 `c_target+f_0`，实测 rel L2 ≈ 0.79）；只有 `part_film()` 显式覆盖成 `all_known`。复用者直接调 `run_film` 会拿到错结果。 | `run.py:161`；归档 `deviation_rows` rel L2 ≈ 0.788–0.795 |
| F7 | minor（口径） | driver 的碰撞/流动核心在 NumPy（torch 逐 kernel 调度 ~1 ms 太慢），只有闭包走生产 torch 模块——注释已如实说明，零拷贝共享内存下数值等价；但"xval 覆盖生产代码路径"的范围因此限于闭包本身。 | `run.py:113-124` 注释与 `film_apply_closures` |
| F8 | minor（报告） | mission 17603 无 `tensor_reports` 行（无六要素论文级报告），report 欠账。 | A7 |

**审计范围限制（不隐瞒）**：(a) 只复核了 film 一臂，velocity 一臂按交付自认 `not_run`；(b) 上游二进制只按归档源码重编，未重新 `git clone` 上游（网络受限），但 sha256 与归档一致 + 源码 diff 未见补丁痕迹可支撑其真实性；(c) 50000 步上游跑通并产出 `.dat`，但 TensorLBM 侧最长只重跑到 12000 步（归档 graded 行也是 12000），50000 步端的 TensorLBM 数值**未独立复现**；(d) 未做网格收敛研究——该类"跨实现 port 一致性"锚点无此语义，也因此**不能**当作物理精度验证。

---

## 6 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 参考 | got | 相对误差 | <3 % ? |
|---|---|---|---|---|
| 浓度场 mean @100 | 0.0953690189676046 | 0.09536901896812995 | 5.5e-12 | **是** |
| 浓度场 rel L2 @100 | — | — | 3.17e-11 | **是** |
| 浓度场 rel L2 @1000 | — | — | 5.12e-11 | **是** |
| 浓度场 rel L2 @10000 | — | — | 4.36e-11 | **是** |
| 浓度场 rel L2 @12000 | — | — | 4.22e-11 | **是** |
| 浓度场 rel Linf @12000 | — | — | 3.61e-10 | **是** |
| 速度 Dirichlet（anti-bounce-back）vs 解析 Couette/Poiseuille | 解析解 | **未跑** | **未测** | **无判定**（honest，理由见 F3） |

结论：**被外部对比的标量闭包，全部关键量误差 ≪3 %（实测 1e-11 量级）**；锚点标题所指的 anti-bounce-back 速度臂**无外部误差数字**，不得据此宣称"anti-bounce-back 已基准验证"。

---

## 7 判级与处置

**L2** —— 功能实质：新增生产代码 `src/tensorlbm/boundary/inamuro_wetnode.py`（525 行，真正的 torch 算子：输入场+面+目标值 → 输出重构后的分布函数）＋ 14 条断言不变式的性质测试（审计重跑全绿）＋ 一个**真实、可复算、参考场 sha256 核对无误**的外部交叉验证（16000 个格点 × 4 个 checkpoint，rel L2 3–5e-11，归档数字与审计重跑差 **0.0**，且有 0.9967 的控制探针证明鉴别力）。

**为什么不给 L3/L4**：L3 要求系统集成——`src/` 内 0 个模块外调用点、未进 `boundary_registry`/能力契约，solver 拿不到这个闭包（F5）。L4 要求基准验证可被"完整算例+误差分析+可复现"承载——数值层面它达到了，但三道结构性门不通：(i) 无机器裁判行（F4）；(ii) 交付未 commit，`repro_cmd` 在仓库检出的语义下不存在且 README 一步复现直接崩（F1/F2）；(iii) 锚点标题所指的 anti-bounce-back 速度臂 `not_run`，只有标量闭包被外部验证（F3）。**报告与图再漂亮也不改变级别**，因此判 L2，并把上述四点登记为返修项。

处置：**request_changes / reopened**。返修最小集（按性价比排序）：
1. 把 4 条 untracked 路径 commit 进 `agent/m17603` 分支（否则本单等于没交付）；
2. 修 `part_film()` 的二进制/目录同名冲突，让 README 一条命令在全新 outdir 上跑通，并把 `film_ref_*.dat` 归档成可离线对比的固定参考；
3. 补 `tensor_benchmark_cases` 裁判行（`case_name='INAMURO-WETNODE-FILM'`, `metric='rel_l2_field_vs_oss@12000'`, `tolerance='<15%'`），把容差从"driver 自声明"变成预登记；
4. `run_film()` 默认策略改为 `all_known`；
5. 补 velocity 臂（约束步间壁面行）拿到 Couette/Poiseuille 的误差数字，或把锚点降级为"标量残余权重闭包"；
6. 把闭包登记进 `boundary_registry` + 能力契约，给出至少一个 solver 调用点，才可谈 L3。