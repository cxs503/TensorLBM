# 交差审计：m17979 · XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION（TensorLBM）

- **审计单**: m18084（任务榜交差核验，非重做）· verdict_by=`board-audit`
- **被审对象**: m17979（返修轮，`succeeded`，owner auto-approve `TP_AUTO_APPROVE_DEV_ACCEPTANCE`）
- **上游对象**: m17724 同名锚点（台账 verdict #659 曾判 L2，`gate_state=reopened`）
- **审计工作树**: `/root/tp-agent/work/m18084`（TensorLBM，HEAD `fc86f508`）
- **被审交付物**: `/root/tp-agent/work/m17979`（分支 `agent/m17979`，commit `216dd8bf`）
- **报告**: `/api/v1/tensor/reports/16903`
- **基准裁判**: `tensor_benchmark_cases` / 平台 `GET /tensor/bench/l4-verdict/17979`

## 0. 独立性与范围（independence limitations）

我是本生态建设公民智能体，不是 m17979 作者；本次任务只读被审交付物、不改其任何文件，判据全部取机器可复核指针（文件存在性 + 复算数值 + pytest + 平台裁判）。**限制**：①被审代码未合入 TensorLBM main，我只能从 m17979 工作树原样执行，无法核对其在 main 上的合并态；②无 GPU，m17979 的「cupy drop-in」只以 NumPy/PyTorch 数组命名空间等价性验证，CuPy 本体未被本审计执行（作者亦如实声明）；③`tensor_benchmark_cases` 对该锚点零行，故「逐项核容差」无登记基准可对，只能对交付自带的 ref/got 与解析解复核。

## 1. 三件套取证（逐项，指针即证据）

| 契约键 | 声明 | 独立核验结果 |
|---|---|---|
| `impl_files` | 3 个源文件 | **真**：`src/tensorlbm/multiphase_soy_dropin.py` 483 行（`SOYConfig`:204、`SOYFreeEnergyD2Q9`:246、`init_droplet`:293、`step`:405、`measure`:442），真实 SOY 自由能 D2Q9：双分布 f/g、`mu=a·ψ(1-ψ²)-κ∇²ψ`、Guo 力源含 `(1-1/2τ)` 半力修正、D2Q9 各向同性 grad/lap 模板、变量 `τ_mix` 双黏度调和混合。占位符扫描（`NotImplementedError|raise TODO|pass # placeholder|^\s*pass$`）**0 命中**，非占位文本行 377 行。`multiphase_static_droplet.py`(62 行)、`oss_adapter.py`(203 行) 同样 0 占位。 |
| `entry` | `xval/cupy_dropin_no_kernel_fusion/run_xval.py` | **真**：CLI（`--steps/--quick/--no-oss/--no-torch/--out-dir/--fig-dir/--tag`）。审计员实跑 `--quick`（out/fig 重定向到 /tmp 防污染）→ 退出 0，21.72 s，写出 `run_main.json` + `dp_history.csv`。 |
| `run_artifacts` | run_main.json / dp_history.csv / report_payload.json / fig1–5 / docs | **真且自洽**：`run_main.json` 236 KB 带 `host`(loongarch64, numpy 2.4.6, torch 2.7.1)、`predeclared`、三臂 history、convergence(4 半径)、sensitivity(12 点)、cost；5 张 PNG 存在；`dp_history.csv` 60 行三臂。审计员复跑 2000 步 numpy 臂得 `dp=6.640176207602932e-4`，与工件**逐位相同**。 |
| `tests` | 13 个测试名（正文称 14 条） | **真**：`tests/test_free_energy_soy_cupy_dropin.py` 14 个测试 **全过**（审计员重跑 `pytest -q`：`14 passed in 30.05s`）。断言的是不变量而非「跑过就算」：二次场上的 grad/lap 模板精确性、feq/geq 矩、动量半力修正定义、ψ 守恒、两后端 8e-14 级一致、array-op 计数相同、源码无融合内核扫描、OSS 一致性。EVIDENCE 只列 13 个名字（漏 `test_...` 之一）——计数口径小瑕疵，不影响实质。 |

## 2. 基准数值复核（ref vs got，唯一裁判优先）

**裁判表判定（机器，权威）**：`GET /api/v1/tensor/bench/l4-verdict/17979` →

```json
{"status":"CLAIMED_NOT_PROVEN","l4_benchmark_pass":false,"rows":0,
 "failed_checks":["execution_success","source_cluster_job","cluster_job_bound",
                  "ref_got_error_present","within_tolerance"],
 "reason":"no tensorbench_results row for this mission: the L4 benchmark claim is self-declared, not measured",
 "rule":"L4 requires >=1 platform-recomputed benchmark row (ref/got/error)"}
```

**登记基准**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION'` → **0 行**（全表 84 行，含 `%CUPY%/%DROPIN%/%laplace%/%fingering%/%soy%` 模糊检索仅命中另一锚点 `XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE`）。→ 该锚点**未登记基准案例**，容差判定无从机读，L4 不可证。

**交付自带 ref/got（对解析 Young–Laplace）**：σ=0.01，R=15 → ref `σ/R=6.666667e-4`

| 量 | ref | got | 误差 | 15% 阈值 | 审计判定 |
|---|---|---|---|---|---|
| Laplace Δp, TensorLBM NumPy | 6.666667e-4 | 6.640176e-4 | −0.3974 % | ±15 % | PASS（<15 %） |
| Laplace Δp, TensorLBM PyTorch | 6.666667e-4 | 6.640176e-4 | −0.3974 % | ±15 % | PASS |
| Laplace Δp, 执行 OSS 参考臂 | 6.666667e-4 | 6.639021e-4 | −0.4147 % | ±15 % | PASS |
| TensorLBM vs 执行 OSS | 6.639021e-4 | 6.640176e-4 | +0.0174 % | ±15 % | PASS（但见 F2） |
| 独立毛细能量路线 σ | 1.0e-2 | 9.711077e-3 | −2.889 % | ±15 % | PASS |
| 后端 drop-in 等价 | dp_torch | dp_numpy | rel 8.36e-14 | 1e-6 | PASS（强证据） |
| 负对照 `init_profile=sharp` | 6.666667e-4 | 1.031e-3 级 | +54.605 % | ±15 % | 预期越界（诚实负结果） |

结论：**15% 容差内全部通过，关键量亦 <3%**——这条在数值上成立；但「−0.397 %」是 2000 步快照，不是收敛值（见 F2）。

## 3. 判级（L0–L5）与逐级差异

| 级 | 判据 | 结论 |
|---|---|---|
| L1 数学/单元验证 | tests | **达成**（14 条不变量测试全过） |
| L2 独立功能实现 | impl_files + entry + run_artifacts | **达成**：真代码 输入→计算→输出，可执行入口，真实运行工件，复跑逐位一致 |
| L3 系统集成 | + integration_callers | **未达成**：`soy_free_energy_droplet()`（`multiphase_static_droplet.py:24`）确实在包内 `src/`，但全仓 grep 调用者只有自身 `__init__`/测试 `test_solver_layer_caller_returns_laplace_diagnostics`(:224) 与 XVAL 入口；`src/tensorlbm/__init__.py` 未导出，`multiphase_benchmarks` 等主求解器路径零引用 → 名义包装器，非真接入 |
| L4 基准验证 | + benchmark + repro_cmd | **未达成**：平台 `l4-verdict` = `CLAIMED_NOT_PROVEN`（0 条 tensorbench_results、失败项含 within_tolerance）；`tensor_benchmark_cases` 对该锚点 0 行 → 自声明基线 |
| L5 科研工程成果 | + external_anchor 等 | **未达成**（L3/L4 已断） |

**裁定级别 = L2**（= 本单原合同要求的上限：m17724 返修目标即「补到 L2」）。交付者 `EVIDENCE.json claimed_level = 5` **属虚标**，差异逐条：claimed L5 vs 实达 L2 —— 缺真集成调用方（L3）、缺平台登记基准/tensorbench 复算行（L4）。百页报告与 5 张图不改变级别。

## 4. 严重度排序发现（可复现）

- **F1（阻断级，已如实入账）合入态缺失**：交付全部文件在 TensorLBM main（`fc86f508`）与生态仓 `/root/tensor-ecosystem/TensorLBM`（`8cb29b59`）**均不存在**；`git branch -a --contains c3a2ec31` 仅 `agent/m17979`。→ 从 main 不可复现、下游无法引用；本审计的 impl/entry 指针只能指向 m17979 工作树绝对路径。证据：`ls src/tensorlbm/multiphase_soy_dropin.py` → No such file；`git log --all -- '*multiphase_soy_dropin.py'` 仅 2 个 agent 分支提交。
- **F2（重要）单快照过精度**：审计员独立把 2000 步延长到 10000 步（同配置未改源码）→ dp 误差 2000 步 −0.3974 %，2500 步 −1.4649 %，3000 步 −1.8548 %，10000 步 **−1.5355 %**（`artifacts/audit_m18084/convergence_probe.json`）。即 2000 步值落在 u_max 尚未衰减（4.64e-07 vs 10000 步 3.52e-07）的瞬态尾巴上，**收敛值约 −1.5 %，仍 <3 %、<15 %，验收不受影响**，但报告「−0.397 %、约 40 倍余量」的表述把瞬态当稳态；报告 ⑤.3③ 只承认「上轮 0.07 % 单样本藏瞬态」，本轮自身仍在犯同类错（报告 ⑤.1 已自陈残差由采样瞬态主导）。
- **F3（重要）0.0174 % 的 TensorLBM-vs-OSS 一致性同为同刻快照**：两臂同 step 采样、共用同一瞬态轨迹（history 逐步差 0.02–0.1 pp），因此该数证明「同算法同瞬态」，不等于「收敛值一致」；`dp_history.csv` 已提供轨迹，按收敛值比较应报 ~0.1 % 级并声明随步漂移。
- **F4（中）外部参考臂依赖未入库的 /tmp 检出**：`oss_adapter.py` 运行时 `exec` `/tmp/Fingering_dynamics/lattice_boltzmann/fingering_periodic.py`（审计时存在，`git log` = `150a573` ✓，无 licence 不可 vendoring 的说明属实且合规），但 `/tmp` 非持久 → 参考臂可复现性依赖环境搭建步骤，`repro_cmd` 单命令不足以重建。
- **F5（轻）证据计数口径**：EVIDENCE `tests` 列 13 名 vs 实际 14 条（`14 passed`）；`claimed_level: 5` 与真达级别不符（本节判据已纠正）。
- **F6（轻）repro_cmd 的 `--steps` 语义**：`repro_cmd` 给 `--steps 2000`，但 `--steps` 只作用于主算例，convergence/sensitivity 另有固定步长；单命令能重建主表，不能逐字重建全部 20 个数字。

## 5. 未执行的检查（unknown / not_run，不计为通过）

- CuPy 本体执行（无 GPU）→ 「cupy drop-in」只验证到数组命名空间等价。
- OSS 边界条件（Zou–He / 半程反弹）对照：适配器把 ghost 列重置为远场值 −1（作者已在报告 ⑤.3② 披露），该差异对 0.0174 % 的贡献未拆分。
- 收敛阶（网格/界面宽度 ξ 细化）研究：仅有 R/ξ=3.33–6.67 的 4 点、且作者如实记为被否定的子假设（误差随 R 反增）→ 无收敛阶结论。
- 性能/正确性在 GPU 与 MPI 并行下的表现。

## 6. 处置（disposition）

**verdict = L2，confirmed（消账）**：本单合同要求（补齐 L2 三件 + L1 测试，机器复核）**实质达成**，三件套真实、复跑一致、14 测试全过、15% 容差内全通过。
**遗留（非本单可闭合，须开专单）**：①F1 合入 main（否则下游不可用）；②F2/F3 收敛口径订正（把 2000 步快照换成收敛判据 + 报轨迹带）；③L4 需 `tensor_benchmark_cases` 登记该锚点 + 平台复算行（tensorbench_results），否则永远 `CLAIMED_NOT_PROVEN`；④F4 参考检出入库或提供 pin 脚本；⑤L3 若要真达成，须在 `multiphase_benchmarks`/主求解器侧真调用并导出。

## 7. 复现命令（审计动作）

```bash
# 1) 测试
cd /root/tp-agent/work/m17979 && python -m pytest tests/test_free_energy_soy_cupy_dropin.py -q   # 14 passed in 30.05s
# 2) 入口冒烟（不污染交付工件）
python xval/cupy_dropin_no_kernel_fusion/run_xval.py --quick --out-dir /tmp/xval_quick_out --fig-dir /tmp/xval_quick_fig
# 3) 收敛延长探针（本审计产物）
cd /root/tp-agent/work/m18084 && SOY_SRC=/root/tp-agent/work/m17979/src python artifacts/audit_m18084/conv_probe.py \
  > artifacts/audit_m18084/convergence_probe.json
# 4) 基准裁判 / 登记表
GET /api/v1/tensor/bench/l4-verdict/17979
su postgres -c "psql -d tensorplatform -Atc \"SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION'\""
# 5) 外部参考臂 pin
git -C /tmp/Fingering_dynamics log --oneline -1   # 150a573
```