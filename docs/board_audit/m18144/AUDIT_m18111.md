# 交差核验：m18111 — XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR（TensorLBM）

- 被审单：**m18111**（返修轮1，原单 m17772），状态 `succeeded` @ 2026-11-13T11:52:25Z，repo=TensorLBM，base_ref=main
- 审计单：**m18144**（board-audit，本文件所在分支 `agent/m18144`）
- 锚点：`XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR`（对标 lssfau/lbmpy 的 `LBMConfig.kernel_type` 融合粒度选择器）
- 审计脚本：`docs/board_audit/m18144/verify_m18111.py` → 输出 `docs/board_audit/m18144/verify_output.json`
- 证据快照（防工作树回收）：`docs/board_audit/m18144/evidence_snapshot/`（含 `SHA256SUMS.txt`、`channel_integration.patch`、`m18111_git_status.txt`、原工件 + 审计重跑工件）

## 0. 裁定

| 项 | 结论 |
|---|---|
| **声明级别（审计方）** | **L2**（独立功能实现） |
| gate 预计裁定 | confirmed（L2 的三条判据 impl_files/entry/run_artifacts 均真） |
| L3 | **不主张**：主求解器调用点只存在于未提交工作树，`/root/tensor-ecosystem/TensorLBM` grep `fused_kernel_selector|collide_stream_bgk` = **0 文件**；gate 的机器判据（生态仓 grep）不可满足 |
| L4 | **不可判**：`tensor_benchmark_cases` 全表 91 行锚点中**无本锚点行**（`WHERE anchor_slug='XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR'` → 0 行），基准裁判缺席，无 ref/tolerance 可裁。交付自报的 ref/got 数值由审计独立重跑复核（见 §2），但那是**审计证据**，不是基准裁判 |

## 1. 独立性声明与范围

- **独立性限制（如实写）**：复跑使用被审单自带的 bench 入口（`xval/fused_kernel_type_selector/bench_fused_kernel_selector.py`），未重写融合内核；独立成分 = ①自写 sha256/工件核验与 SQL 查裁判表 ②按报告 §⑦ 命令在**独立输出目录**重跑并逐字段比对 ③自己读实现与测试代码判断不变量是否真断言 ④额外的通道集成实验（被审单只报了一个数字，审计自己跑了两条 kernel_type）。
- **已跑检查**：C1 工件存在＋快照一致性；C2 测试实跑；C3 复现逐字段比对；C4 基准裁判 SQL；C5 生态仓调用方 grep＋集成补丁；C6 报告六段＋图。全部见 `verify_output.json`。
- **未跑**：x86/GitHub CI 侧（本机 loong64/amd64 不等价）；GPU 路径；lbmpy 版本扫描（仅 2.0/2.0）；**性能收益**（该锚点本质是性能开关，但交付工件只有总 wall_time=40.2 s，无对照计时 → 「性能更好」不可判，报告也未主张）。

## 2. 数值核验（基准裁判缺席下的最佳可得证据）

被审单工件 `artifacts/xval_fused_kernel_type_selector/results.json`（sha256 `5dd22793…`）与审计重跑（sha256 `5c2ae7f0…`，仅 `wall_time_s` 不同）：

- **逐字段一致性**：34 行 × 140 个数值字段，**max rel diff = 0.0** → 复现是逐位级，不是「大概一致」。
- **粒度等价性（锚点本体）**：`tensorlbm fused(post_stream)` vs **lbmpy 2.0 JIT C 内核** `default_stream_collide`：1 步 2.8e-16、50 步 1.8e-15；`collide_only` vs `collide_bgk` 2.2e-16。跨库一致在机器精度。
- **物理面（ref = NS 解析解 `u₀·exp(−νk²t)`）**：n32/τ0.8 **0.522%**、n64/τ0.8 **0.071%**、n64/τ0.6 **0.087%**、n64/τ0.9 **0.044%** —— 全部 ≪15%（owner 判据），且 <3%。
- **阴性对照**：`stream_pull_only`（去碰撞）n64/τ0.6 = **24.19%**、n32/τ0.8 = **55.82%** → benchmark 有分辨力。
- **被证伪的自设 ref（诚实保留）**：离散剪切模本征公式 `(1−(1−cos k)/τ)^t` 与算例差 3 个量级（`err_discrete_pct` 355806%…2106%），工件原样保留，报告 §⑥ 明写该 ref 被证伪。

## 3. 分级核验（按达标判断方案逐键）

| 判据键 | 实际指针 | 核验 |
|---|---|---|
| `impl_files` | `evidence_snapshot/src/fused_kernel_selector.py`（192 行；原 `/root/tp-agent/work/m18111/src/tensorlbm/fused_kernel_selector.py`） | ✅ 真实现：3 种 kernel_type + 单遍 fused stream+collide（pre/post 两种矩序），非占位 |
| `entry` | `evidence_snapshot/xval/bench_fused_kernel_selector.py:249`（argparse `--out/--steps`）；模块入口 `.../fused_kernel_selector.py:step()` | ✅ 实跑通 |
| `run_artifacts` | `evidence_snapshot/artifacts_orig/results.json`（12 310 B）、`…/fused_kernel_selector.png`（1800×660，非空白） | ✅ 真运行工件；审计重跑逐位复现 |
| `tests` | `evidence_snapshot/tests/test_fused_kernel_selector.py`；审计实跑 `PYTHONPATH=src pytest -q` → **9 passed in 8.9 s**（被审单报 11.6 s） | ✅ 断言不变量（等价性/质量动量守恒/非法 kernel_type 拒绝/15% 物理门/跨库 lbmpy 一致），非「跑通即过」 |
| `integration_callers` | 仅 `evidence_snapshot/channel_integration.patch`（52 行 diff，`ChannelConfig.kernel_type` → `collide_stream_bgk`）；**生态仓 grep = 0** | ❌ L3 不成立（见 F1） |
| `benchmark` | 交付自报 ref/got 已复核（§2）；裁判表 **0 行** | ❌ L4 不可由裁判认证（见 F2） |
| `repro_cmd` | 报告 §⑦ `PYTHONPATH=src python bench_fused_kernel_selector.py --out <dir>` + pytest | ✅ 审计实跑，逐位复现 |
| `external_anchor` | lbmpy 2.0 + pystencils 2.0 实装并被 **import**（`xval/lbmpy_reference.py:44-47`），JIT 编译执行，非手抄复述 | ✅（但不足以替代 F2 的裁判行） |

## 4. 发现（按严重度）

**F1（material｜交付未入库＝能力可能凭空消失）** 五个交付文件 + 两个工件在 m18111 工作树**全部未提交**（`m18111_git_status.txt`：`M channel.py`、`M REPORT.md`、`?? fused_kernel_selector.py`、`?? tests/…`、`?? bench…`、`?? lbmpy_reference.py`、`?? artifacts/…`），任务 `result.head_commit = null`、`branch = null`；`origin/main`（7d60f928）里该目录**只有** `xval/fused_kernel_type_selector/REPORT.md` 被跟踪。后果：工作树一旦回收，实现/测试/工件即丢失；mission headline 的「补到 L3」相对仓状态是**夸大**。缓解：审计已把全部交付与原工件快照进本分支并记 sha256（`SHA256SUMS.txt`），可无损恢复；建议跟随一张交付单把快照 commit 回 `src/`、`tests/`、`xval/`、`artifacts/`。

**F2（material｜基准裁判缺席，L4 无从裁定）** `tensor_benchmark_cases` 共 91 行、91 个锚点，**无本锚点**；同族锚点 `XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN`(id 80) 有登记行且自带 `verify.py`，本锚点没有。因此「误差是否在容差内」没有机器可裁的 ref/tolerance；审计只能给出自建复核（§2）。建议：把本锚点的判据（粒度等价性 <1e-12 相对、剪切波 vs 解析解 <15%、阴性对照 >15%）登记为一行 `tensor_benchmark_cases` + `verify.py` 自动门。

**F3（minor｜图纸路径旁路，已诚实披露）** bench 内置绘图分支抛 `KeyError('variant')`（等价性行用 `quantity` 键），被 try/except 记入 `results.json.figure_error`；交付 PNG 由另一路径生成。审计重跑**同样**复现该错误字符串 → 说明披露属实、非事后解释；但 bench 自绘图仍未修。

**F4（minor｜选择性引用最坏值）** 报告 §4.2 阴性对照引用 1.97%（n64/τ0.8），而同一对照 n32/τ0.8 在工件里是 **55.82%**（更坏）。不影响正确性（对照臂本就该失败），但按「报告六要素零手抄」应引最坏值。

**F5（minor｜报告纪律）** ④ 段仅 1 图（衰减曲线＋等价性误差条形图），无场量云图且**未显式写「本锚点无云图理由」**。本锚点是内核粒度等价性 + 1D 剪切模，云图确无物理意义，补一句即可合规。

**F6（positive｜锚点本体被证实）** 「融合是性能开关、不是物理开关」在本实现上成立且被两路验证：与 `stream(collide_bgk(f))` 机器精度一致、与 lbmpy JIT 内核 1.8e-15 一致；`sediment.channel` 两条 kernel_type 的 `u_star` **逐位相同（rel=0.0）**、`profile_rel_err` 相同（审计自跑 200 步实测）。默认仍为 `collide_only`，老算例行为不变。

**F7（note｜踩坑价值）** 报告陷阱 3：`C∘S ≠ S∘C`，随机非平衡场相对差 **14.5%**，恰在 15% 判据边缘 —— 跨库对标若不对齐 `moments` 约定会把算子次序差误判成精度差。该条是本次交付最有复用价值的知识点，建议进 TK。

## 5. 处置（disposition）

**批准 L2（approve with notes）**：实现/入口/工件/测试四项独立复核为真、数值逐位复现，功能实质确为「独立功能实现」。**不批准 L3/L4** 的机器主张（F1/F2）。返修建议（不阻塞 L2 消账）：① 交付入库（快照 → 正式 commit/PR）；② 登记基准裁判行 + `verify.py`；③ 修 bench 绘图分支并补「无云图理由」一句；④ 阴性对照引最坏值。

## 6. 复跑

```bash
# 工件/测试/复现/裁判/集成/报告 六项一键核（~20 s；需 m18111 工作树在，缺则 C2=not_run）
python3 docs/board_audit/m18144/verify_m18111.py
# 仅复现数值（不覆盖原工件）
cd /root/tp-agent/work/m18111 && PYTHONPATH=src python3 \
  xval/fused_kernel_type_selector/bench_fused_kernel_selector.py --out /tmp/repro
```