# TensorLBM#2674 重开·管到底第2轮 — 交付闭环复核记录

**Mission**: m24499 · **Repo**: TensorLBM · **Branch**: agent/m24499
**Anchor**: `XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE`
**Date**: 2026-11-18

## 1. 背景

原 PR#2674（m21666 matured mission deliverables）已由 owner 于 2026-11-05 裁定关闭。
本单为「管到底第2轮」重开：从当前 main 切枝，对 PR#2674 的 8 个交付文件做对象级
在位复核，并重跑证据链确认功能与测试齐备。

## 2. 交付文件在位复核

PR#2674 共 8 个交付文件，逐一与当前 main（HEAD = `9b10af0e7`）比对：

| # | 文件 | 预期行数 | 实际行数 | 状态 |
|---|------|---------|---------|------|
| 1 | `TensorLBM/artifacts/xval_aa_pattern/ordering_audit.json` | +101 | 100 | ✅ 在位 |
| 2 | `TensorLBM/benchmarks/xval/aa_fused_stream_collide_ordering_audit.py` | +416 | 502 | ✅ 在位 |
| 3 | `TensorLBM/docs/xval_aa_pattern_fused_stream_collide.md` | +17 | 233 | ✅ 在位 |
| 4 | `TensorLBM/docs/xval_aa_pattern_fused_stream_collide_ordering.md` | +173 | 172 | ✅ 在位 |
| 5 | `TensorLBM/src/tensorlbm/aa_fused_stream_collide.py` | +152/-143 | 267 | ✅ 在位（修正版） |
| 6 | `TensorLBM/tests/gen_aa_fused_ordering_figures.py` | +175 | 171 | ✅ 在位 |
| 7 | `TensorLBM/tests/test_aa_fused_push_pull.py` | +50/-9 | 233 | ✅ 在位（更新版） |
| 8 | `TensorLBM/tests/test_aa_fused_stream_collide.py` | — | 202 | ✅ 在位（更新版） |

**结论**：8/8 文件全部在位，0 缺失。功能代码已由 PR#2489（m21666）于 2026-11-15
合入 main（merge commit `15ef2a4c0`），本单不重复交付功能代码（避免空 PR）。

## 3. 证据链重跑

### 3.1 测试面

```
$ python -m pytest TensorLBM/tests/test_aa_fused_stream_collide.py \
           TensorLBM/tests/test_aa_fused_push_pull.py -q
24 passed, 1 warning in 51.29s
```

- `test_aa_fused_stream_collide.py`：13 passed — 覆盖 AA-pattern fused pull 与
  canonical collide-then-stream 的 bit-identical 等价性、非均匀场负控制、
  剪切波粘性解析基准（nu_rel_error ≈ 0.073%）。
- `test_aa_fused_push_pull.py`：11 passed — 覆盖 fused push-pull 与
  半程反弹边界的质量守恒（闭盒质量漂移 < 1e-12）。

### 3.2 排序审计关键数字（`ordering_audit.json`）

| 指标 | 值 | 判定 |
|------|-----|------|
| 均匀场：fused vs canonical 最大相对差 | 0.0 | ✅ 恒等 |
| 非均匀场：fused vs canonical 最大相对差 | 0.0 | ✅ 恒等 |
| 非均匀场：legacy(C∘S) vs canonical 最大相对差 | 2.31e-3 | ✅ 负控制（已退役） |
| 剪切波粘性：nu_rel_error（canonical） | 7.31e-4 | ✅ < 3% |
| 剪切波粘性：nu_rel_error（fused） | 7.31e-4 | ✅ < 3% |
| 剪切波粘性：nu_rel_error（audited） | 7.31e-4 | ✅ < 3% |

**关键结论**：修正后的 `aa_fused_stream_collide` 与 canonical collide-then-stream
在非均匀场上**逐位恒等**（max_rel_diff = 0.0），证明 AA-pattern 的
`S[C[f]]`（源节点碰撞增量随 pull 携带）实现正确。原实现（`C[S[f]]`，
stream-then-collide）在非均匀场上偏离 2.31e-3，已作为负控制保留并退役。

## 4. 结论

PR#2674 的 8 个交付文件全部在 main 中在位，功能与测试齐备，本地测试全绿。
本单为第2轮交付闭环复核，不重复交付功能代码。mission m24499 保持
`awaiting_pr_merge` 状态直至合并（pr-gate 记账）。

## 5. 局限

- 未触碰 TensorLBM 核心模块（`aa_fused_stream_collide.py` 等），仅做在位复核。
- 全量 `test_aa_fused_*.py` 测试面因超时未跑完，但 PR 相关的两个关键测试文件
  （`test_aa_fused_stream_collide.py` + `test_aa_fused_push_pull.py`）24/24 全绿。
