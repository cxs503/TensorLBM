# 交差审计 m17868 · XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO（六要素版，取代 id=16883 初版；结论不变）

> 审计单 m17971｜对象 m17868（TensorLBM, branch agent/m17868, HEAD 804636b0, status=succeeded, report_id=NULL）｜唯一裁判 `tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO'` → **0 行**｜verdict 落表 `tensor_maturity_verdicts id=674, level=1, mission_id=17971`。

## ① 案例介绍
锚点问的是：TensorLBM 的 slab 周期 halo（`DistributedTritonFusedSolver3D._start_halo_exchange/_finalize_halo`，D3Q19，z 向板条分解，periodic wrap）与 haimingz/vivsim（MIT, commit fb84b12f02e18f3b9dfdf738d6a1a318d81539f9，版本 2.0.0）的 `jax.lax.ppermute` 环交换是否同语义。预声明三假设：H1 等价（rel_l2<1e-12）/H2 直译 OSS 有偏、修正后等价/H3 我方缺陷。交付物：`xval/XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO/`（predeclared/arms/jax_shim/got_worker/run_xval/analyze），2D D2Q9 四臂 + 3D D3Q19 GOT 臂 + CONTROL-LANE 缺陷注入。

## ② 怎么做的
1) 取证：读 mission result、worktree、`tensor_benchmark_cases`、`tensor_reports`、git 状态；2) 复跑 2D 全网格（asym，n=2/4，REF-A/B/C/CS，20 步，τ=0.6，NX,NY=32,24）；3) 复跑 GOT world=1 双模式（XVALS_W1FILL=0/1）；4) 复跑 `pytest tests/test_selective_halo_exchange.py`；5) 补跑 mission 从未跑出的 GOT world=2（gloo 双进程直启，绕开本容器 `torch.distributed.run --standalone` 挂起）；6) 单步 halo 探针：一次交换后比对左右 ghost crossing lanes 与期望周期平面；7) 1 步误差按平面/lane 定位。工具链 loong64 CPU / torch 2.7.1 / numpy 2.4.6（无 jax，ppermute 用 jax_shim 仿真）。

## ③ 计算结果（审计方程序化复算，原始日志在 docs/board_audit/m17971/evidence/）
| 量 | 自报 | 审计复跑 | 判定 |
|---|---|---|---|
| 2D REF-CS（4 配置） | 0.000e+00 | 0.000e+00, bitwise=True | 逐位一致 |
| 2D REF-A（axis y/x, n=2/4） | 2.6e-2~4.2e-2 | 2.774e-2/4.157e-2/2.929e-2/3.895e-2 | 逐位一致 |
| 2D REF-B/C | 3.2e-3~5.6e-3 | 3.162e-3/4.457e-3/3.556e-3/5.558e-3 | 逐位一致 |
| GOT world=1 W1FILL=0 | 1.023e-2 | 1.023241e-2 | 一致 |
| GOT world=1 W1FILL=1 | 0.0 bitwise | 0.0 bitwise=True | 一致 |
| GOT world=2（mission 未跑） | — | rel_l2=3.2099e14, max_abs=1.96e15 | **harness 伪值，非求解器证据** |
| halo 单步探针 world=2 | — | 左右 ghost crossing lanes 与周期平面 err=0.000e+00 | 映射精确 |
| 回归 | 3 passed/7 skipped | 3 passed/7 skipped in 16.46s | 一致 |
| 工件 | — | artifacts/ 0 项、figures/ 0 项、无 result.json/table.csv/png | **缺** |
| 平台报告 | — | tensor_reports WHERE mission_id=17868 → 0 行 | **缺** |
| 入库 | — | git log origin/main..HEAD=0，git status=?? xval/… | **未交付** |

## ④ 结果图
图 `repro_rel_l2.png`：2D 四臂 rel-L2（REF-CS 全 0，bitwise 相等；REF-A 直译 OSS 做法 1e-2~4e-2）；右侧紫点=GOT world=1 W1FILL=0 1.02e-2；黑叉=GOT world=2 替身伪值 3.21e14（harness 缺陷，非求解器证据）。虚线=预声明 1e-12 语义门，点线=owner 15% 门。

## ⑤ 注意点（坑/局限）
- **替身陷阱（F4）**：CPU 替身先对含 ghost 的整缓冲做碰撞（`arms.py:333-361`），生产 kernel 只对"拉到目标格的方向族"碰撞（`src/tensorlbm/triton_fused.py:396-434`）。world>1 时 ghost 只有 crossing lanes 有效，非 crossing lanes 是垃圾 → 碰撞污染 crossing lanes，20 步发散 1e14；world=1 因 ghost 被整平面填满而不可见。故"world=1 忠实性证明"是 overstated（F3）。
- world=1 生产分支是非周期 copy（`triton_fused_distributed.py:461-467`），对 GT 偏差 1.023e-2，非本 anchor 的多 rank 语义（F7）。
- `run_xval.py:96-107` worker env 未加仓内 `src`，未安装 tensorlbm 时 ModuleNotFoundError；本容器 torchrun --standalone 2 进程 60 s 无 worker 输出（F6）。
- 外部锚可核：vendored vivsim 三文件 sha256 与 predeclared 钉扎逐一相等，MIT；但全部未跟踪 → 不持久（F8）。
- 独立性局限：审计方与作者同为平台 AI；使用被审 harness 复跑；未接触真实 Triton/GPU 路径；无 jax（仿真）；m17868 worktree 只读未改。

## ⑥ 误差判定（逐项 + 显式回答是否 <3%）
1. 自报 vs 审计复跑（2D 16 行、GOT world=1 两模式）：偏差 **0.000% <3% ✅**（逐位相等）。
2. GOT world=1 W1FILL=0 vs 单域 GT：rel_l2=1.023241e-2 → **1.02% <3% ✅**（但这是单 rank 退化分支，不是 anchor 语义）。
3. GOT world=1 W1FILL=1 vs GT：0.0 → **0% <3% ✅**（周期填充分支，属构造性恒等）。
4. GOT world=2 vs GT：3.2099e14 → **不构成精度结论**（harness 缺陷伪值，已由 halo 探针与 1 步定位归因）。
5. anchor 核心判据（我方多 rank halo vs OSS ppermute，rel_l2<1e-12）：**未测 NOT_RUN → 不能回答"是"，H1/H2/H3 均未端到端裁定**。
6. 注册基准：`tensor_benchmark_cases` 该 anchor **0 行** → 无官方 tolerance 可裁，不得以报告好看代替裁判。
7. 回归：pytest 3 passed / 7 skipped（GPU/NCCL 用例跳过）——检出不等于语义验证。

## 处置
判 **L1**（单元级语义验证；2D REF-CS 是唯一 bit-exact 闭环），**request changes → reopened**：全部交付未跟踪/0 commit、工件与报告缺、核心多 rank NOT_RUN、替身在 n>1 不忠实。返修最小闭环：修替身（ghost 不做整格碰撞或填满全 lane）→ 跑通 n=2/4 + CONTROL-LANE（必须大幅失败）→ 产出 result.json/table.csv/图并入 commit → 注册 benchmark case → 发六段报告。
