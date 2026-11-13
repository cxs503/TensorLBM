# VERDICT m17716 — XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION（交差核验，board-audit m17735）

**裁定：L4（基准验证级）→ 预期 maturity_gate confirmed（消账）**
落表：`tensor_maturity_verdicts` id=635（repo=TensorLBM, scope='m17716 XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION', mission_id=17735, level=4, verdict_by=board-audit, gate_state=pending 待守护 30min 周期机器复核）
补注册：`tensor_benchmark_cases` id=64（本单前该 anchor 0 行，referee row absent → 按先例 id=47/55 以本单独立复跑数字注册，status=passed）

## 1 被审对象
- 工作树 `/root/tp-agent/work/m17716`，分支 `agent/m17716`，head=`4846dcdd`（m17716: waLBerla equal-level border stream correction transplant (D2Q9) + F-H prolongation calibration xval）
- 交付面：`src/tensorlbm/amr_equal_level_stream.py`（379 行新模块）、`src/tensorlbm/amr_cavity2d.py`（+51 行 frozen 分支）、`tests/test_amr_equal_level_stream.py`（139 行 7 测试）、`benchmarks/xval_block_amr_equal_level_stream_correction/{run_xval.py,make_report.py}`、`artifacts/xval_block_amr_equal_level_stream_correction/`（8 工件）、`docs/XVAL_BLOCK_AMR_EQUAL_LEVEL_STREAM_CORRECTION.md`

## 2 验收三件套核验（全部实测，非读报告）
| 项 | 判据 | 实测 | 结论 |
|---|---|---|---|
| run_artifacts | 文件真实存在 | 8 工件全在（benchmark.json/results.json/run.log/REPORT.md/SUMMARY.md/fig1 60336B/fig2 32076B/figures.json） | ✅ |
| benchmark ref vs got | 12 行 KPI 在容差内 | 12/12 pass，worst err 0.0984% vs 15% 线；inertness/round-trip 精确 0 | ✅ |
| repro_cmd 可执行 | 独立复跑 | `PYTHONPATH=src python benchmarks/.../run_xval.py --out /tmp/xval_repro` rc=0；重算 12 行与已提交 benchmark.json **逐行相等**，fh_scale_error=0.0009837083993351455 逐位一致 | ✅ 确定性复现 |
| 测试 | pytest | `pytest tests/test_amr_equal_level_stream.py -q` → **7 passed in 0.59s** | ✅ |
| 报告六段 | tensor_reports | id=16654（published, 4959 B）§1-§6 齐全，§6 逐项显式回答 <15% | ✅ |

## 3 基准裁判（唯一裁判）
- 本单前 `SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION'` → **0 行**（referee row absent）。
- 处置：按先例（id=47 CATHODIC-SPHERE、id=55 RIGID-LID「registered by audit ... because the referee row was absent」）以**本单独立复跑数字**补注册 id=64（非 subject 自报）。
- 注册后逐项：3 条 F-H 标定（tgv_n64_p32 0.0984% / tgv_n128_p64 0.0492% / shear_n64_p32 0.0349%）+ 9 条算子不变量（惰性/往返精确 0，质量缺陷 ≤7.15e-07）→ 全部在容差内。

## 4 判级（L0-L5）
- L1 ✅ 7 测试通过，不变量精确 0
- L2 ✅ 真代码：输入（blocked 掩码+场）→ 计算（链接表/prepare/restore/标定扫描）→ 输出（benchmark.json/图）
- L3 ✅ 系统集成：`CavityAMR2D.step()` frozen-shadow 分支（amr_cavity2d.py:260-307）在流式处调用 prepare/restore；集成断言 `test_cavity_frozen_shadow_uses_the_correction`
- L4 ✅ 完整 xval 算例（3 分辨率×流场）+ 误差分析（标定扫描/次优 scale 间隙 0.249）+ 可复现（本单复跑逐位一致）
- L5 ❌ 做法的动力学收益（frozen 整场时间推进精度）未验证——subject 自报 WITHHELD（早期草稿粗层 ~2x 额外耗散未归因，已从交付移除，honest FAIL 如实声明）
- **附百页报告不改变级别；级别=功能实质 → L4**

## 5 发现（findings）
- **F1 referee row absent**：基准表本单前 0 行 → 已补注册（见上）。
- **F2 frozen 动力学 WITHHELD**：整场时间推进精度未验证，不计入能力声明；不阻塞本轮声明范围（算子不变量 + F-H 标定）。
- **F3 仓内既有 F-H 约定错误**：实测因子=1.000（err 0.0984%），而 `amr_cavity2d._fh_rescale`=0.5833 与 `amr_population_transfer.rescale_nonequilibrium`=2.3333 均偏离且方向相反——属仓内既有代码问题，建议后续轮修正，非本 mission 缺陷。
- **F4 移植边界如实**：waLBerla 本体未运行（无网络/GPL/无构建链）→ ref 用解析场，OSS 跑数输出 WITHHELD，报告 §5 已标注。

## 6 未做检查（如实）
- waLBerla 本体运行（WITHHELD）；frozen 整场精度（不代验）；集群 loong64 复跑（单机 <10 s 无必要）。

## 7 机器复核自证
`simulate_gate.py` 直接 import 守护进程同款 `judge()`（/root/tp-agent/maturity_gate.py）：claimed=4 → gate_level=4, why=[] → PASS（evidence 格式已对齐 gate 契约：绝对路径/测试名列表/grep 可命中的 caller token）。
