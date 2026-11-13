# XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN

**Mission** m17895 (TensorLBM) · anchor: sorush-khajepor/listLBM 的 *example-per-main*
反模式 · owner 精度标准 15%（目标 3%）· 全部数字由
`xval/XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN/run_artifacts/summary.json` 与各 arm JSON 程序化导出，
并可用一条命令重跑。

---

## ① 案例介绍（问题、几何、参数）

**问题（反模式是什么）。** `examples/` 里每个例子都是一个独立 `main`：自己的 argparse、
自己的格子构造、自己的时间推进循环、自己的后处理。上游 listLBM 就是这个做法；TensorLBM
的 `examples/` 把它放大到了 161 个文件。后果是可测量的，而不是口味问题：

* 求解器或后处理约定改一次，要改 N 遍，且必然漂移；
* 例子的"结果"= 脚本打印的字符串，机器无法裁判；
* 两次运行"同一个"例子不可比（没有 spec 身份、没有场指纹）。

**审计（反模式确实存在，先量出来）** — 新增 `src/tensorlbm/audit/example_antipattern.py`
（纯 AST，无领域硬编码）：

| 指标（repo `examples/`） | 数值 |
|---|---|
| python 文件 / 总行数 | 161 / 50 627 |
| 自带 `__main__` 的脚本 | 157（97.5%） |
| 自己解析 argv | 126 |
| 自己写时间推进/streaming 循环 | 91 |
| 落在"结构重复块"内的行 | 15 886（31.4%） |
| 结构重复的**驱动块**（含循环或 streaming 调用） | 232 个 → 2 848 行（5.6%） |
| 委托给共享运行时（thin main） | 0 |

**几何与参数（物理算例）。** D2Q9 / BGK / 全 Guo(2002) 源项 / 半程反弹壁 / float64 CPU。

| 算例 | 参数 | 闭式参照 |
|---|---|---|
| 压力梯度驱动平面槽道 | ny=16/32/48, nx=8, τ=0.8（ν=0.1）, g=1e-5, Δρ<1e-3 | `u_max = gH²/(8ν)`, `H = ny-2`（壁面在 y=1/2 与 y=ny-3/2）, `u(y)=g/(2ν)(y-1/2)(ny-3/2-y)` |
| Taylor–Green 涡 | 32×32 周期, τ=0.8, U0=0.05, 800 步 | `u ∝ exp(-2νk²t)`，其 Laplacian 为 `-2k²u` ⇒ 能量 `∝ exp(-4νk²t)`, `k=2π/nx` |

## ② 怎么做的（方法/工具链）

1. **新增共享运行时** `src/tensorlbm/example_harness.py`：`CaseSpec`（冻结 dataclass，
   `digest()` = 规范 JSON 的 sha256）→ `run_case(spec)` → 可 JSON 化的 run record
   （spec + 解析参照 + KPI + 场指纹 + provenance）。运行时**组合**库内核
   （`equilibrium` / `collide_bgk` / `stream`），因此例子不可能偏离库。
   `tests/test_example_harness.py::test_no_force_step_equals_library_kernels` 断言：
   body_force=0 时一步结果与 `collide_bgk`+`stream` **逐比特相同**。
2. **新增反模式审计器** `src/tensorlbm/audit/example_antipattern.py`：名字/字面量无关的
   AST 归一化 + 跨文件哈希聚类，度量 M1（独立 main 数）、M2（重复行占比，按**去重行**计，
   并区分"驱动块"重复）、M3（thin main 占比）。CLI：`python -m tensorlbm.audit example-antipattern`。
3. **实验设计**：`predeclared.json` 先写决策/假设/指标/阈值/停止规则，再动算（阈值不在看结果后调）。
4. **对照臂**：`reference_main_channel.py` —— 一个**不 import 运行时**的 example-per-main 风格脚本
   （自己的 BGK+Guo、`torch.roll` 推式 streaming、自己的反射掩码），用来证明共享运行时是
   *忠实合并*而不是*重定义*格式。
5. **检测器判别力对照**：`fixtures/antipattern`（3 个几乎相同的独立 main）vs
   `fixtures/consolidated`（3 个 thin main）——一个"什么都报警"的检测器在这里必然失败。
6. **thin 例子**：`examples/channel_poiseuille_thin.py`（40 行，无循环、无格子代码）。
7. 公开 API：`tensorlbm.api` 导出 `CaseSpec/get_case/run_case/available_cases/analytic_reference/record_digest`。

## ③ 计算结果（ref vs got，程序化导出）

**槽道 u_max vs 闭式 `gH²/(8ν)`**

| ny | u_max(got) | u_max(ref) | rel err | profile L2 | interior L2 | mass drift | steps |
|---|---|---|---|---|---|---|---|
| 16 | 0.00243100 | 0.00245000 | 0.7755% | 0.3633% | 0.3372% | -2.86e-13 | 4400 |
| 32 | 0.01123098 | 0.01125000 | 0.1691% | 0.0793% | 0.0766% | -7.51e-13 | 12000 |
| 48 | 0.02640609 | 0.02645000 | 0.1660% | 0.1224% | 0.1222% | -9.55e-13 | 15000 |

**Taylor–Green 能量衰减率**

| got λ | ref 4νk² | rel err |
|---|---|---|
| 1.543865e-02 | 1.542126e-02 | 0.1128% |

**与独立 example-main 对照（ny=32）**

| harness u_max | control u_max | rel diff | profile L2 diff |
|---|---|---|---|
| 0.0112309779 | 0.0112309779 | 6.124e-13 | 6.204e-13 |

**机器裁判（`summary.json.checkes`，11 项全绿，`arms_complete=true, all_pass=true`）**

| check | 值 | 阈值 | 判 |
|---|---|---|---|
| C1 u_max 在 owner 带内 | 7.76e-3（最差分辨率） | 0.15 | PASS |
| C2 u_max 在 3% 目标内 | 7.76e-3 | 0.03 | PASS |
| C3 ny=48 profile L2 | 1.22e-3 | 0.03 | PASS |
| C4 分辨率单调收敛 | [7.76e-3, 1.69e-3, 1.66e-3] | 不上升 | PASS |
| C5 质量守恒 | 9.6e-13 | 1e-10 | PASS |
| C6 TG 衰减率 | 1.13e-3 | 0.03 | PASS |
| C7 与独立对照一致 | 6.12e-13 | 1e-8 | PASS |
| C8 确定性/负对照 | 重跑逐比特同、扰动 spec 变 | — | PASS |
| C9 检测器判别力 | driver 重复率差 0.621 | 0.20 | PASS |
| C10 反模式在 repo 中真实存在 | 重复行 31.4%、main 占比 97.5% | 0.20 | PASS |
| C11 稳定性规则 | ρ∈[0.99942,1.00000] | [0.95,1.05] | PASS |

## ④ 结果图

* `figs/fig1_channel_profile_convergence.png` —— 左：3 个分辨率的速度剖面 vs 闭式抛物线；
  右：u_max 与 profile 的 L2 相对误差随 ny 的双对数收敛曲线（3% 目标线与 15% owner 带）。
* `figs/fig2_tg_decay_and_antipattern.png` —— 左：Taylor–Green 能量半对数衰减 + 拟合率 + 解析率；
  右：反模式检测器三树对照（repo examples / 植入反模式 / 植入合并版）的重复行占比与驱动行占比。

（图全部由 `docs/xval_example_per_main/make_figs.py` 从 run_artifacts JSON 读数生成，零手抄。）

## ⑤ 注意点（踩过的坑 / 局限）

1. **截断的 Guo 源项会抽真空。** 只保留 `3w_i(c_i-u)·F` 而省掉 `9w_i(c_i·u)c_i·F` 时，
   `Σ_i S_i ≠ 0`：槽道 12k 步后密度均匀掉 2.5e-3。全式后 mass drift 降到 ~1e-13。
   （这一条是实测出来的，不是抄的：见 `_guo_forcing` docstring。）
2. **`(1-1/(2τ))` 必须配"平衡态用 u+F/(2ρ)"**。两者拆开用会整体欠驱动：实测 u_max 只有
   闭式解的 0.375 倍（=1-1/(2τ)）；配套后误差 0.08%。
3. **壁面位置算错会伪装成"粘性错了"。** 半程反弹壁在 y=1/2 与 y=ny-3/2 ⇒ H=ny-2（不是 ny-1）。
   一开始把节点坐标取成 j-0.5，剖面看起来"抛物线但幅度差 2.7 倍"，实际是参照系错了 0.5 格。
4. **Taylor–Green 的参照是 4νk² 不是 2νk²**：该场的 Laplacian 是 `-2k²u`。第一次写成 2νk²，
   报出"100% 误差"，是参照错、不是求解错——audit 会把这种差异暴露出来，但必须自己判。
5. **去重度量不能按"块实例"计数**（嵌套语句重复计数，比例可 >100%）；改为按**去重源行**计数，
   并单列"驱动块重复"。这是一次在 dry-run 后做的指标细化，阈值未放宽（见 `predeclared.json.deviations`）。
6. **局限**：上游 listLBM 仓未 vendor 在本仓，反模式按**结构**刻画（独立 main），未执行其二进制；
   槽道算例是定常层流，未覆盖湍流/多相/三维；审计器是"结构重复"检测，不做语义等价判断。
   迭代量级（161 文件 / 5 万行）由 AST 扫描给出，不含人工确认。

## ⑥ 误差判定（逐项 + 是否 <3%）

| 关键量 | got | 基准 | 相对误差 | <3%？ |
|---|---|---|---|---|
| 槽道 u_max (ny=16) | 0.00243100 | 0.00245000 | 0.7755% | 是 |
| 槽道 u_max (ny=32) | 0.01123098 | 0.01125000 | 0.1691% | 是 |
| 槽道 u_max (ny=48) | 0.02640609 | 0.02645000 | 0.1660% | 是 |
| 剖面 L2 (ny=48) | — | 闭式抛物线 | 0.1224% | 是 |
| Taylor–Green 能量衰减率 | 1.543865e-2 | 1.542126e-2 | 0.1128% | 是 |
| 与独立 example-main 对照 u_max | 0.0112309779 | 0.0112309779 | 6.12e-13 | 是（远严于 1e-8 阈值） |
| 质量守恒 | — | 0 | 9.6e-13 | 是 |

**结论：全部关键量 <3%（且最差的 0.78% 出现在最粗网格 ny=16，随分辨率下降）。**
无 WITHHELD 项：两个物理基准都是闭式解，不依赖任何外部文献表。

## 复现

```bash
cd <repo>; export PYTHONPATH=src
python xval/XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN/run_xval.py --repo-root .   # 全部 8 臂 ~2 min CPU
python -m pytest tests/test_example_harness.py tests/test_example_antipattern_audit.py -q
python -m tensorlbm.audit example-antipattern --root examples --out /tmp/ap.json
python examples/channel_poiseuille_thin.py --ny 32
python docs/xval_example_per_main/make_figs.py
```

臂可分段跑（`--arms a1_16,a1_32` 等）并把结果合并进 `summary.json`，因此长跑可断点续跑。
本算例为 CPU float64 数十万格点级别，总耗时约 2 min，未占用集群（曾递交 cj1726 后因本地已完成而撤回；
集群仅用于 >10 min 的重算）。