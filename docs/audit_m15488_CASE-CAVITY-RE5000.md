# 交差审计 · m15370 CASE-CAVITY-RE5000（TensorLBM 顶盖方腔 Re=5000）

审计人：board-audit（mission 15488，仓 TensorLBM）· 审计时间 2026-11-12
对象：mission 15370（branch agent/m15370，commit `8e4636cf`，已并入 main）
性质：交差核验（取证 + 判级 + 落表），不是重做。

## 0. 结论先行

| 项 | 判定 |
|---|---|
| 实现 | **真代码**：`benchmarks/verified/cavity/re5000/run.py`（352 行）复用 live 求解器（D2Q9 + RLBM + Zou/He 顶盖） |
| run_artifacts | **缺失**：mission 未留任何算例工件；报告自列的 `smoke_128.json` 从未提交（`git log --all` 0 命中） |
| benchmark ref vs got | **无**：报告 ④ 判 WITHHELD（Ghia 1982 Table III/IV Re=5000 离线不可机器取得）；`tensor_benchmark_cases` 中该 anchor **0 行** |
| repro_cmd | **原样可执行但有致命分支**：短程崩溃、发散静默过关（本审计修，见 §3） |
| 报告六段 | **齐全**（①问题+ref出处 ②方法 ③参数表 ④对照表 ⑤收敛/敏感性 ⑥失败模式+工件），④为空是诚实申报而非粉饰 |
| **级别** | **L2**（独立功能实现：输入→计算→输出，审计已独立复现；无基准证据故不进 L3/L4） |

## 1. 取证：三件套逐项核验

### 1.1 impl_files —— 存在且是 live 树
* `benchmarks/verified/cavity/re5000/run.py:1-352`（唯一新增实现；`sys.path` 前置本树 `src/` + `tensorlbm.__file__` 断言守卫，`run.py:64-76`）。
* `TensorLBM/src/tensorlbm/verification/lid_driven_cavity/reference_data.py:121-143`（声称的"真实缺陷已修"位置）。

### 1.2 run_artifacts —— 缺失（虚标 1 处）
* 报告"工件（worktree 内）"第 2 条列 `benchmarks/verified/cavity/re5000/smoke_128.json`；
  实测：**文件不存在**，且 `git log --all -- benchmarks/verified/cavity/re5000/smoke_128.json` **0 命中**，
  即从未在任何提交中存在过。报告同时用它证明"本地链路 smoke 跑通"——该主张无工件支撑。
* `re5000/` 目录合并后只有 `run.py` 一个文件（`ls` 实测）。
* mission 的唯一"计算证据"是集群作业 cj1092 的排队（报告 ⑤/⑥ 自陈 queued 348→未取回）。

### 1.3 基准裁判 —— anchor 未登记
* `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-CAVITY-RE5000'`
  → **0 rows**（裁判表 16 个 anchor 中无此项）。裁判缺席属平台侧缺陷（见 §4 F6），
  本审计退到 case pool：`tensor_case_pool id=6 case_code=CAVITY-RE5000 tol_policy='<8%' status=dispatched`，
  与 mission goal 的 `<8%` 预声明一致 → **容差判据本身有效，但无 ref 载荷可判**。
* 故 benchmark 一项按 `WITHHELD` 记，**不判 pass，也不判 FAIL**（无可判对象）。

### 1.4 报告六段 —— 齐全
DB `tensor_reports id=14446` 与仓内 `docs/cavity-re5000-report.md` 同文（82 行），①–⑥ 段段在场，
参数表完整（Re/U_lid/τ(384)=0.52300、τ(512)=0.53070/ν/Ma/网格/步数/终止判据/预登记容差 8%）。
④ 明写 WITHHELD，未列任何伪造数字 → 符合 honest-FAIL 纪律。

## 2. 审计独立复现（只跑审计自己的命令，不采信报告数字）

| 复现 | 命令 | 结果 |
|---|---|---|
| R1 短程可执行性 | `run.py --n 24 --steps 60 --monitor-every 1000` | **崩溃**：`TypeError: unsupported format string passed to NoneType.__format__`（`run.py:339`），**result.json 未写出** |
| R2 发散处理 | `run.py --n 24 --re 5000 --steps 600 --monitor-every 100` | 解发散（`u_min=nan`），脚本仍打印 `[done]`、**exit 0**、把 NaN 与伪涡心写进 result.json |
| R3 Re 无关正例 | `--n 16 --re 100 --steps 40` | 正常出剖面（16 点中线），comparison=`WITHHELD` ✔ 无假通过 |
| R4 修复后合同 | 同上两条 | exit 0/`run_status=ok`；exit 2/`run_status=failed`+`comparison=VOID` ✔ |
| R5 集群复算 | cj1146（128² Re=1000 自检 + 384²/512² Re=5000 400k 步） | **not_run**：递交时排队 345，审计窗口内未派发（不等待） |

## 3. 本审计的修复（最小、可验证）

`benchmarks/verified/cavity/re5000/run.py`：
1. 每格 `run_case()` 返回 `"status": ok|diverged|non_finite`（非有限场不再伪装成功）；
2. `final_resid` 为 `None`（`--steps < --monitor-every`）时 nan-safe 打印，不再 TypeError；
3. 任一格发散/非有限 → `run_status="failed"`、`failed_grids`、`diverged`，`comparison` 置 `{"status":"VOID"}`
   （NaN 不得被当成数字引用），并 **exit 2**；文档补 exit-code 契约。

新增回归测试 `tests/test_cavity_re5000_runner.py`（4 例，实跑 `PYTHONPATH=src pytest -q` → **4 passed / 31.6s**）：
短程不崩且出工件、缺 ref 必 WITHHELD、发散必 exit 2 + VOID、**synthetic ref 打分路径首次被覆盖**
（错表必须 `pass=False`，`--ref-json` 解析/打分此前零覆盖）。

## 4. 缺陷清单（按严重度）

* **F1 阻断**（m15370 遗留）：`--steps < --monitor-every` 时 `TypeError` 崩溃、无工件 → repro_cmd 在缩短预算的场景下完全失效。**本审计已修**。
* **F2 阻断**（m15370 遗留）：发散解以 exit 0 + `[done]` 落盘，`u_min=nan`、涡心为垃圾值 → 未来报告可直接抄到 NaN。**本审计已修**。
* **F3 高（虚标）**：`smoke_128.json` 从未存在（§1.2）。
* **F4 高**：声称的"真实缺陷并已修 + 回归测试"落在 **非打包子树**：corners 见
  `TensorLBM/src/tensorlbm/verification/lid_driven_cavity/reference_data.py`，而 `pyproject.toml:83-86`
  `package-dir={""="src"}`，实测 `PYTHONPATH=src python -c "import tensorlbm.verification.lid_driven_cavity"`
  → **ModuleNotFoundError**；`pyproject.toml:96 testpaths=["tests"]` → CI 只收根 `tests/`，
  `TensorLBM/tests/test_lid_driven_cavity.py:94` 的回归测试**永不执行**。
  另：live 模块 `src/tensorlbm/lid_driven_cavity.py:664-671` 对未表化 Re 已是显式 `reference=None`（无静默回落），
  即该"真实缺陷"只存在于死副本中 → 修复对 shipped 库**零效果**，主张应降级为"文档/死代码整理"。
* **F5 中**：`grid_convergence()` 中 `"monotone_decrease_with_refinement": True if len(keys)==2 else None`
  是**断言**而非计算（未比较粗细网格误差单调性）。未改（不影响判级，标注为如实性瑕疵）。
* **F6 中（平台侧）**：`tensor_benchmark_cases` 无 `CASE-CAVITY-RE5000` 行 → 该 anchor 无机器裁判，
  gate 无法机器裁定容差。建议为冻结点算例统一补裁判行（本审计不改平台表）。

## 5. 判级理由（L2）

* **非 L0**：有可执行实现，不是纯文档。
* **非 L1**：不止单元验证——`run.py` 是完整的算例入口（输入 → 求解 → 中线剖面/涡心/ρ 界/残差 → JSON 工件），
  审计 R3 独立跑出 16 点剖面并写盘。
* **非 L3**：全仓 `grep -rn "re5000\|RE5000"` 无任何平台/注册表调用者（`benchmarks/` 与 `tensorlbm.cli` 未注册该入口）。
* **非 L4**：无 ref vs got 任何数字；无误差分析；复现性只有 exit-code 合同，Re=5000 定常解从未产出。
* **L2 成立但仅限"代码可达且产出工件"**；Re=5000 目标算例本身（稳定定常解 + Ghia 误差）**未建立**。

## 6. 未完成 / 局限（如实）

1. cj1146 未派发（排队 345），故 128² Re=1000 自检与 384²/512² Re=5000 **未取数** → "τ≈0.52 的 RLBM 在 384² 是否稳定"仍是开放问题；
   本审计只证明 24² 会发散（属于预期，τ→0.5）。
2. Ghia Table III/IV 离线不可机器取得 → 本审计**不补基准表**，也不手抄任何参考数字；因此无法把本算例抬到 L4。
3. 未审 GitHub CI（x86）；全部实跑为 loongarch64。
4. F4 的修法（把 `reference_data.py` 迁进 `src/` 或删除死副本）超出交差核验范围，未做，留作返修单。
---
报告已发布于平台 `tensor_reports id=14551`（scenario_key=CASE-CAVITY-RE5000，含 1 图）；判级落表 `tensor_maturity_verdicts id=328`。
