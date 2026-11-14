# 交差核验单 · m20141 GAP-DIFFLBM-3D-COST（3D D3Q19 可微反演的可承受包络）

**审计任务**：m20184（board-audit，非重做）｜**被审任务**：m20141（`succeeded`，2026-11-14 09:36 UTC）
**被审提交**：`agent/m20141` @ `090996f42`（"diff3d: memory-bounded reverse mode for 3D D3Q19 LBM + inversion envelope benchmark"）
**基准裁判**：`tensor_benchmark_cases` id=129, `anchor_slug='GAP-DIFFLBM-3D-COST'`（注册于 08:49 UTC，由兄弟任务 m20072/m20104 注册）
**审计机**：loongarch64 128 核 / torch 2.7.1 / 无 GPU；审计探针工件 `artifacts/m20184_audit/`

**结论（disposition）**：`request changes`（要求返修）— **L3**。
功能实质为真（L2 真实实现 + L3 求解器侧入口，测试断言不变量、梯度可 bit-identical 复现），但 **L4 基准验证未成立**：
① 交付包内缺少裁判注册的 Taylor–Green / 剪切波算例（交付用的是 ABC 锚）；② 注册裁判行的 auto-check / repro 指针指向**仓库内不存在**的文件；
③ 锚点标题所指 256³ 包络为**外推而非实测**，且反向模式把内存开销换成**未计入的磁盘开销**（256³/T=800 外推 475 GiB）；
④ 交付代码**未并入主分支**（`main`=0414e8681 无 `src/tensorlbm/diff3d.py`），生态侧无任何调用方，maturity_gate 机器复核将降为 L2。

---

## 1. 取证：三件套与裁判逐项核验

### 1.1 三件套存在性（全部在 `agent/m20141`@090996f42 提交内，实读实测）

| 件 | 交付声明 | 审计实测 | 判定 |
|---|---|---|---|
| 实现 | `src/tensorlbm/diff3d.py`（新）+ `src/tensorlbm/solver3d.py::inversion_gradient_3d` | 485 行真实实现：D3Q19 前向核（`step3d`/`collide_bgk3d`/`stream3d`）、磁盘 checkpoint 反向模式（`checkpointed_gradient`）、Richardson-FD 参考、ABC 连续锚（`analytic_decay_gradient`）、τ 反演（`recover_tau`）；solver3d.py:597 入口 + :643 导出 | REAL |
| 运行工件 | `artifacts/gap_difflbm_3d_cost/run_{gradient,two_tau,envelope,inversion}.json` + `numbers.md` + 3 png | 4 个 JSON 与 numbers.md/3 png 均存在且非空；`m20141/tensor_reports/` 目录**不存在**（报告走平台 API，id 18364） | REAL |
| 可复现命令 | 裁判行写 `python3 scripts/bench_difflbm3d_envelope.py --stage grad ...` | **该脚本不存在**（`scripts/` 下无 `bench_difflbm3d_envelope.py`，全仓 `find` 无此名）；等价入口 `benchmarks/gap_difflbm_3d_cost/run.py --case gradient --grids 16,24,32 --steps 48` 审计实跑成功 | 裁判指针 DEAD / 等价入口 REAL |
| 测试 | 裁判行写 `python3 -m pytest tests/test_diff3d_envelope.py -q (7 passed)` | **该文件不存在**；实际交付 `tests/test_diff3d_checkpoint_gradient.py`，审计实跑 **8 passed in 51.80s** | 裁判指针 DEAD / 等价套件 REAL |

### 1.2 裁判四项判据逐项核验（got vs ref）

| # | 裁判判据 | 交付 got | 审计独立核验 | 判定 |
|---|---|---|---|---|
| R1 | 梯度 `rel_err(AD vs FD_5pt) ≤ 1e-6`（fp64） | 2.40e-13 / 2.58e-13 / 1.90e-13（N=16/24/32） | 独立重跑同一入口，g_AD **逐位相同**（rel diff = 0.0e+00，17 位）；AD vs Richardson-FD 同值复现 | **PASS** |
| R2 | 物理 `rel_err(E_measured vs E_analytic) < 3%` 且 `A/A_analytic < 3%` | 交付包**未提供**该对照（只给了梯度 vs 连续锚 1.36%/1.25%/1.20%） | 审计补测（ABC 锚，τ=0.8，T=48）：E 误差 2.82%(16³)/0.92%(24³)/0.37%(32³)，rms\|u\| 误差 1.42%/0.46%/0.18%，全部 <3% 且随 N 收敛（O(k²)） | **PASS（审计补证）** |
| R3 | 显存 `peak_rss` 对 T 为 O(1)（非 O(T)） | 交付：T=16→48 时 states-equiv 36.7→40.0 | 审计自测 VmHWM（N=32，T=32/128/256）：peak 674/747/586 MiB，baseline 432/436/427 MiB ⇒ states-equiv 50.9/65.5/33.5，**无单调增长**；若为 O(T)，T=256 应比 T=32 多 4.98MB×224≈1.1 GiB。**结论 PASS**，但噪声带（±36 state）远大于模型态数（5），交付号 36.7/40.0 落在同一噪声带内，不能作为"恰好 5 态"的证据 | **PASS（机理成立，精确值不可由 RSS 分辨）** |
| R4 | auto-check `pytest tests/test_diff3d_envelope.py (7 passed)` + repro `scripts/bench_difflbm3d_envelope.py` | — | 两路径均不存在 ⇒ 裁判行按字面**不可执行**；等价套件 8 passed、等价入口逐位复现 | **FAIL（指针失效）** |

### 1.3 裁判 `ref_source` 与交付实现的一致性问题（关键）

裁判行 id=129 的 `ref_source` 明确写：**Taylor–Green 能量衰减 `E=E0e^{-4νk²t}` + 剪切波 `A=A0e^{-νk²t}`，τ=0.55（ν=0.0167），幅度 0.02/0.05**。
而 m20141 交付的锚是 **ABC（Arnold–Beltrami–Childress）流，τ=0.8（ν=0.1），幅度 0.05**，交付包内**从未跑过 TG/剪切波**。
即：**裁判注册的是 m20072/m20104 那套（`src/tensorlbm/diff3d_envelope.py` + `scripts/bench_difflbm3d_envelope.py`，仓库内已不存在）的设计，而非本单交付的设计**。同一 anchor_slug 上存在两套互不兼容的实现与参考解。

为判"交付实现是否满足裁判所註参考解"，审计在**交付核**上按裁判参数补跑（`artifacts/m20184_audit/audit_m20141_referee_anchor_n{32,48}.json`）：

| 参考解 | N | T | 实测衰减率 | 解析衰减率 | 相对误差 | 3% 门 |
|---|---|---|---|---|---|---|
| Taylor–Green E | 32 | 400 | 2.569983e-03 | 2.570209e-03 | 0.0088% | PASS |
| 剪切波 A | 32 | 400 | 6.445999e-04 | 6.425524e-04 | 0.3187% | PASS |
| Taylor–Green E | 48 | 300 | 1.142002e-03 | 1.142315e-03 | 0.0275% | PASS |
| 剪切波 A | 48 | 300 | 2.859833e-04 | 2.855788e-04 | 0.1416% | PASS |

⇒ 交付核**能够**满足裁判的物理参考解（0.01%–0.32%），但**交付包内没有这份证据**，且交付的测试/入口文件名与裁判行不一致。这是注册侧与交付侧的**双头记账**缺陷，不是实现缺陷。

---

## 2. 审计发现（按严重度）

**F1（阻断 L4，高）裁判指针失效 + 双实现并存。** 裁判行 id=129 的 `repro_cmd`/auto-check 指向 `scripts/bench_difflbm3d_envelope.py`、`tests/test_diff3d_envelope.py`（全仓不存在）；`ref_source` 的 TG/剪切波参考解在交付包中缺席。复现：`find /root/tp-agent/work/m20141 -name '*difflbm3d*' -o -name '*diff3d*'` → 只有 `src/tensorlbm/diff3d.py`、`tests/test_diff3d_checkpoint_gradient.py`（属 m20141 设计），无 m20072 设计文件。任何后续复核按裁判行字面执行都会失败。

**F2（阻断 L4，高）"可承受包络"的磁盘代价未计入。** 反向模式用磁盘 checkpoint 换取 O(1) RAM（`checkpointed_gradient` 每 `segment` 步 `torch.save` 一态）。审计实测（计数 `torch.save` 载荷）：32³/T=48/seg=4 → 12 个 checkpoint，57.0 MiB；按同 segment 外推到锚点目标 256³/T=800 → **200 × 2.55 GB = 475 GiB 磁盘、约 950 GiB 读写 I/O**。交付的 envelope 报告（`run_envelope.json`/`numbers.md`/docs ⑥）只报 RAM 与时间，**完全没有磁盘容量与 I/O 时间项**，而锚点标题正是"可承受包络"。

**F3（阻断 L4，中）256³ 包络为外推而非实测。** 交付如实标注 `WITHHELD` 且时间门 honest-FAIL（float64@16 核外推 6.86 h ≫ 2 h，T_max(2h)=233 ≪ T=800），这一点诚实；但外推依赖"16 核线性加速 4.0×"这一未测假设（实测仅 4 线程），且内存 11.88 GiB（fp64）/5.94 GiB（fp32）为 5 态模型值，无 256³ 实测锚。审计已向集群递交实测作业（`cj2560`，node330，32 核，memprobe 256³ grad/forward），**至本报告截稿仍在 running，结果未纳入判级**。

**F4（低）交付工件字段与模型自相矛盾。** `run_envelope.json` 报 `model_states = 10`（由 `run.py:185` 的 `args.segment + 6` 生成），而 `diff3d.memory_model_bytes()` 实际按 **5 态**计（`bytes_model/state_bytes = 5`）。外推用的 11.88 GiB 取的是 5 态值，故结论不受影响，但工件标签会误导复核者。

**F5（低）RSS 判别力的过度解读风险。** 交付 numbers.md 用 states-equiv 36.7 vs 40.0 表述"内存与 T 无关"，而审计在更大 T 跨度（32→256）测到 33.5–65.5 states 的噪声带。O(1) 的正确表述应是"无 O(T) 增长（T×8 未见 ×8 增长）"，而非具体态数吻合。

**F6（信息/交付路径，中）分支未并入主干。** `main`=0414e8681 无 `src/tensorlbm/diff3d.py`、无 `inversion_gradient_3d`；`agent/m20141` 是唯一含该提交的分支，生态仓 grep 无任何调用方。按 `maturity_gate.judge()` 机器规则（`callers_verified` 在 `/root/tensor-ecosystem/<repo>` grep），L3 及以上必被判缺 `integration_callers`，将降级到 L2。审计实测该降级路径成立（本地跑 `judge()` 复现：claim 4→2，claim 3→2）。

---

## 3. 评级（L0–L5）

| 级 | 判据 | 本单 |
|---|---|---|
| L1 数学/单元 | 守恒、AD vs FD、segment 不变性等断言 | ✅ 8 passed（51.8 s），断言为不变量（净动量>1e-3、梯度符号、O(k²) 收敛、`pytest.raises(TypeError)` 证明模型无 steps 维） |
| L2 独立功能实现 | 真代码 输入→计算→输出 | ✅ `diff3d.py` 485 行 + 入口 `benchmarks/gap_difflbm_3d_cost/run.py` + 运行工件 |
| L3 系统集成 | 接入主求解器/平台 | ⚠️ 分支内成立：`solver3d.py:597 inversion_gradient_3d` → `diff3d.recover_tau`，`__all__` 导出，bench`run.py:43` 为唯一调用方；**主干不成立**（未 merge） |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | ❌ 梯度/物理/内存三项判据经审计独立复现均 PASS，但交付包缺裁判注册的 TG/剪切波算例、裁判指针失效、256³ 未实测、磁盘代价未计（F1–F3） |
| L5 科研工程成果 | — | ❌ 无外部文献对照发表级验证 |

**判定：L3。** 与 m20072 那次被判 L4 的区别：那次 verdict 的 evidence 直接引用了不存在的 impl/entry/run_artifacts（被 gate 判 `占位/缺失` → L1）；本次全部指针均为实存文件且数值经独立复现，故非虚标，而是"实质到位、验证口径未闭合"。

---

## 4. 建议返修项（按优先级）

1. **消账裁判行**：要么让交付补跑裁判注册的 TG/剪切波（按 id=129 的 τ=0.55、amp=0.02/0.05；审计已证明交付核可过 0.01%–0.32%，工作量极小），要么由 board 以"设计变更"为由重新注册 ref_source（ABC 锚，τ=0.8）并说明理由——**审计未擅自修改裁判行**。
2. **补磁盘包络**：在 `case_envelope` 中报 checkpoint 字节数、峰值磁盘占用与写入/读取 I/O 时间，并把 256³ 的 475 GiB 列入"可承受性"结论。
3. **256³ 实测**：`memprobe --grid 256 --steps 16 --mode grad --segment 4 --threads 32 --chain`（集群 932 GiB 空闲内存足够；审计已递交 `cj2560`，可直接复用）。
4. **修正 F4 字段**：`model_states` 改为 `memory_model_bytes()/state_bytes` 的真实值，或删掉该字段。
5. **合入主干**：merge 后 `solver3d.inversion_gradient_3d` 才有生态侧调用方，L3 才能在机器复核中成立。

## 5. 审计未做/未完成（如实标注）

- **256³ 实测**：`cj2560`（node330）至截稿 running，结果未纳入本报告与判级。
- **GPU 路径**：本机与集群均无 GPU，`cuda_available=false`；任何 GPU 结论属未验证。
- **thread scaling**：审计仅用 6 线程，未独立验证 16 核 4.0× 线性假设（F3 已列）。
- **未审**：`src/tensorlbm/diff3d.py` 之外该分支其他改动（无）、`docs/GAP-DIFFLBM-3D-COST.md` 的六段完整性逐字复审（抽读通过）、m20072 分支是否存在远端副本（`git ls-remote` 网络不可达，超时未完成）。

## 6. 独立性说明

审计在**独立工作树**（`/root/tp-agent/work/m20184`）内进行，仅以只读方式（`PYTHONPATH=<m20141>/src`、`PYTHONDONTWRITEBYTECODE=1`、`-p no:cacheprovider`）调用被审代码，未修改被审工作树任何文件。审计补测脚本与被审代码不同源（自行实现 TG/剪切波初值、衰减率最小二乘拟合、`torch.save` 载荷计数）。局限：与被审实现共享同一 `tensorlbm.diff3d` 前向核——若前向核本身有物理错误，ABC/TG 两套解析锚会同时失配，故"锚一致"不等于"物理绝对正确"，仅表示与解析解在 0.01%–2.8% 内一致。

## 7. 工件清单（本工作树）

| 路径 | 内容 |
|---|---|
| `artifacts/m20184_audit/audit_m20141_cost.py` | 审计探针：物理保真（E/A vs ABC 解析）、RSS 平坦性（VmHWM）、磁盘 checkpoint 计量 + 256³ 外推 |
| `artifacts/m20184_audit/audit_m20141_probes.json` | 上述探针输出（机制化，无手抄） |
| `artifacts/m20184_audit/audit_m20141_referee_anchor.py` + `_n32/_n48.json` | 按裁判 ref_source 在交付核上跑 TG/剪切波衰减率对照 |
| `artifacts/m20184_audit/repro/run_gradient.json` | 独立复现（N=16/24/32, T=48, τ=0.8），g_AD 逐位相同 |
| `artifacts/m20184_audit/numbers.md` | R1–R8 全部数字（由 `make_audit_report.py` 从 JSON 生成） |
| `artifacts/m20184_audit/figures/fig{1,2,3,4}_*.png` | 物理锚对照 / 内存+磁盘包络 / 梯度复现 / 裁判参考解 |
| `artifacts/m20184_audit/verdict_m20184.sql` | 落表 SQL（`tensor_maturity_verdicts`，level 3） |