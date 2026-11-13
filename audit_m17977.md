# m17977 交差审计报告 — XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE

**审计任务**: m18088 (board-audit)
**被审任务**: m17977 (XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE)
**仓库**: TensorLBM
**日期**: 2026-11-13

---

## 审计结论

**判定级别**: **L3**（系统集成）

| 级别 | 判定 | 理由 |
|---|---|---|
| L0 文档成果 | ✅ | 有完整文档 docs/XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE.md |
| L1 数学/单元验证 | ✅ | 解析场 22 例误差≈0%，24 项单元测试全绿 |
| L2 独立功能实现 | ✅ | 3 个源文件，真实 input→compute→output |
| L3 系统集成 | ✅ | 接入 color_gradient_two_tau_step(wall_contact_angle=...) |
| L4 基准验证 | ❌ | equivalence 超阈值（浮点噪声级），sessile honest-FAIL |
| L5 科研工程成果 | ❌ | L4 未通过 |

---

## 取证摘要

### 1. 实现文件
- `src/tensorlbm/cg_wetting_geometric.py` (26KB) — 梯度旋转法润湿 BC
- `src/tensorlbm/cg_contact_angle.py` (13KB) — 接触角测量工具
- `src/tensorlbm/color_gradient2d.py` (16KB) — 修改后的求解器（新增 wall_contact_angle 参数）

### 2. 基准测试
- `benchmarks/xval_openlbmpm_wetting/run.py` (568 行)
- 三块证据：equivalence / analytic / sessile

### 3. 运行工件
- `results/results.json` (24KB) — 结构化结果
- `results/figures/*.png` (5 张图，共 213KB)

### 4. 测试结果
- `tests/test_cg_geometrical_wetting.py` — **12 passed** / 5.72s
- `tests/test_contact_angle.py` — **12 passed, 1 skipped** / 94.51s

### 5. 复现命令
```
cd <TensorLBM> && python benchmarks/xval_openlbmpm_wetting/run.py \
  --blocks equivalence,analytic,sessile --steps 1000 --thetas 60,90,120 --out results
```

---

## 基准结果逐项核对

| 检查项 | 判据 | 实测 | 通过? |
|---|---|---|---|
| equivalence_port_vs_reference | max_abs_diff ≤ 1e-12 | 3.08e-12 | ❌ (超 3.08x，但为浮点噪声) |
| analytic_readback_within_15pct | err_pct ≤ 15% | ≈0.00% (22 例) | ✅ |
| sessile_local_estimator_within_15pct | err_pct ≤ 15% | NaN (全臂) | ❌ |
| sessile_global_shape_within_15pct | err_pct ≤ 15% | θ90: 9.68% ✅, θ60/120: ❌ | ❌ |

**honest_fail**: `["equivalence_port_vs_reference", "sessile_local_estimator_within_15pct", "sessile_global_shape_within_15pct"]`

---

## 发现与问题

### F1: equivalence 阈值过严
max_abs_diff = 3.08e-12，阈值 1e-12。差异在 float64 机器精度范围（per-angle 最大在 135°=3.08e-12），非语义错误。建议阈值放宽至 1e-10 或改用相对误差。

### F2: sessile local estimator 全 NaN
宿主 A-扰动求解器伪流过强（|u|max ~1e-2），接触线无法稳定形成。此为宿主求解器既有局限，非本 BC 实现之错。需先修复宿主求解器伪流问题。

### F3: 基准裁判缺位
`tensor_benchmark_cases` 表中无 `XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE` 注册行，无法进行独立基准比对。

### F4: 外部锚不可持久
参考实现位于 `/tmp/openLBMPM`，无 commit SHA 钉扎，复核依赖临时路径。

### F5: 报告质量良好
六段齐全，honest_fail 如实申报，5 张结果图程序化生成零手抄。

---

## Verdict SQL

见 `verdict_m17977.sql`

---

## 审计方法

1. 读取 m17977 工作树 `/root/tp-agent/work/m17977`
2. 逐项核验验收三件套（run_artifacts / benchmark ref vs got / repro_cmd）
3. 运行相关单元测试确认测试真实性
4. 检查实现文件与主求解器的集成点
5. 对照报告六要素完整性
6. 独立重建执行路径，探测高影响失败模式
