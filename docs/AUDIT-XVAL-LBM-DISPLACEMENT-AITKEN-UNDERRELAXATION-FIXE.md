# AUDIT · XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXE — 独立对抗复核 + 尺度无关守卫修复

任务 m18271 · 仓 TensorLBM · 分支 `agent/m18271` · 复核对象 = m17727（merge `5cc7dc04`，Merge #2067）
工件：`xval/aitken_underrelaxation_life/{scale_guard_probe.py, scale_guard_figures.py, out/scale_guard.json, out/fig5_scale_guard.png}`、
测试 `tests/test_fsi_aitken_guard_scale.py` · 数字全部由脚本写入 JSON，本文表格由 JSON 程序化生成（零手抄）。
判据：owner 精度标准 <15%（报告纪律要求显式回答 <3%）。

## 0. 结论摘要（disposition: 请求修改 → 已在**本单**完成修改）

m17727 的交付（位移 Aitken 不动点对标 + 把 `fsi_partitioned.AitkenRelaxation` 的裁剪下限 0.05 改为 1e-4）
**算法对标部分成立、可复现**（T1 递推恒等、T3/T4 在 β≤50 全部通过、收敛解对 monolithic Newmark 误差 ~1e-13），
但其"缺陷已修复"的结论**只在被测的 β=50 上成立**：1e-4 是**绝对**下限，线性附加质量界面上收敛要求
`0 < ω < 2/(1+β_eff)`，因此任何绝对下限都会在 `β_eff > 1/ω_min − 1` 重新发散——m17727 只是把发散阈值
从 `β>39` 挪到 `β>9999`（实测 β=2e4、1e5 发散），且在阈值内已开始退化（β=1e4：迭代 4.25 vs 3.00，解偏差 1.42%）。
同一缺陷还留在**另外两个**同族副本里（`fsi_added_mass` floor 1e-4；`fsi_partitioned_stability` floor 1e-3，实测 β≥4e3 发散），
即 m17727 的根因陈述（"两处实现不一致"）低估为两处、实为三份拷贝、且只修了一份。

**本单修复**：抽出共享守卫 `tensorlbm.fsi_partitioned.clamp_aitken_factor`（+ `AITKEN_POSITIVITY_EPS=1e-12`），
三份拷贝全部改为走它，默认 `omega_min=0`（尺度无关：只保正性，不再把因子钉在任意常数上），
显式传入的绝对下限仍然生效（向后兼容）。修复后默认臂在 β_eff ∈ [0.2, 1e5] 全收敛、2.98–3.00 子迭代/步、
ω 复现解析最优值（相对偏差 ≤ 2.2e-14）。

## 1. 独立性与范围

- 独立重建：不沿用 m17727 的运行产物（`out/results.json`、`out/summary.csv` 仅作对照），
  用本单新写的 `scale_guard_probe.py` 直接驱动 `solve_partitioned_1d` 与三个加速器副本，在**裸不动点映射**
  `r(a)=1−(1+β)a`（增益 `−(1+β)`，解析最优 `ω*=1/(1+β)`）上复测。
- 复核对象身份绑定的 commit：`5cc7dc04`（m17727 交付所在合并点）。本次审计在其 worktree 后继状态上执行，
  差异仅为 `agent/m18271` 分支（log 见 `git log --oneline -1`）。
- 限制：沙箱无外网，**未执行** joconnor22/LIFE 源码（同上轮 limitation，对标物仍是冻结方法摘要 + 独立转写）；
  被测系统只有 1-DOF 线性标量界面，多自由度/非线性结论不外推（见 §7）。

## 2. 严重度排序的发现

### F1 [High] 绝对下限把发散阈值移位而非消除（m17727 的核心主张被证伪一部分）

复现：`PYTHONPATH=src python3 xval/aitken_underrelaxation_life/scale_guard_probe.py`
（`out/scale_guard.json.driver_ladder`；列 = 平均子迭代/步 · 收敛解对 monolithic Newmark 的最大相对误差 %）

| β_eff | 默认 ω_min=0（本单） iters / err_mono% | m17727 floor 1e-4 | 原始 floor 5e-2 | clip=False |
|---|---|---|---|---|
| 0.2 | 3.00 / 1.1e-14 | 3.00 / 1.1e-14 | 3.00 / 1.1e-14 | 3.00 / 1.1e-14 |
| 0.5 | 2.98 / 1.7e-04 | 2.98 / 1.7e-04 | 2.98 / 1.7e-04 | 2.98 / 1.7e-04 |
| 2 | 3.00 / 1.1e-14 | 3.00 / 1.1e-14 | 3.00 / 1.1e-14 | 3.00 / 1.1e-14 |
| 10 | 3.00 / 1.1e-14 | 3.00 / 1.1e-14 | 3.00 / 1.1e-14 | 3.00 / 1.1e-14 |
| 25 | 3.00 / 2.6e-13 | 3.00 / 2.6e-13 | 10.33 / 3.5e-02 | 3.00 / 2.6e-13 |
| 50 | 3.00 / 3.2e-13 | 3.00 / 3.2e-13 | **发散** (2.20) | 3.00 / 3.2e-13 |
| 200 | 3.00 / 4.2e-12 | 3.00 / 4.2e-12 | **发散** (1.25) | 3.00 / 4.2e-12 |
| 1000 | 3.00 / 2.9e-11 | 3.00 / 2.9e-11 | **发散** (1.15) | 3.00 / 2.9e-11 |
| 4000 | 3.00 / 4.3e-10 | 3.00 / 4.3e-10 | **发散** (1.12) | 3.00 / 4.3e-10 |
| 9000 | 3.00 / 1.3e-09 | 3.00 / 1.3e-09 | **发散** (1.10) | 3.00 / 1.3e-09 |
| 10000 | 3.00 / 3.9e-09 | 4.25 / 1.4e+00 | **发散** (1.10) | 3.00 / 3.9e-09 |
| 20000 | 3.00 / 1.0e-08 | **发散** (14.27) | **发散** (1.10) | 3.00 / 1.0e-08 |
| 100000 | 3.00 / 1.6e-07 | **发散** (1.30) | **发散** (1.08) | 3.00 / 1.6e-07 |

判读：β≤9999 三臂等价；**β=1e4 起 m17727 默认（1e-4）与修复后默认分道**——1e-4 臂迭代 4.25（+42%）、
解偏差 1.42%（同判据下解析误差 1.42%），β=2e4 / 1e5 **不收敛**（残差发散到 ~1e236/1e295）。解析阈值
`β > 1/ω_min − 1 = 9999` 与实测一致（β=9000 收敛、β=1e4 退化、β=2e4 发散）。
`clip=False`（LIFE 原始约定）在全表 3.00 步收敛——说明损失来自守卫而非算法。

### F2 [High] 同缺陷仍在两份兄弟副本中（根因是"三份拷贝"，m17727 只修了一份）

复现：同上脚本 `module_factor` 段（列 = Aitken 激活次数 / 最终因子）

| β_eff | ω*=1/(1+β) | fsi_partitioned | fsi_added_mass | fsi_partitioned_stability |
|---|---|---|---|---|
| 0.2 | 0.833333 | 2 / 0.833333 | 2 / 0.833333 | 2 / 0.833333 |
| 0.5 | 0.666667 | 2 / 0.666667 | 2 / 0.666667 | 2 / 0.666667 |
| 2 | 0.333333 | 2 / 0.333333 | 2 / 0.333333 | 2 / 0.333333 |
| 10 | 0.0909091 | 2 / 0.0909091 | 2 / 0.0909091 | 2 / 0.0909091 |
| 25 | 0.0384615 | 2 / 0.0384615 | 2 / 0.0384615 | 2 / 0.0384615 |
| 50 | 0.0196078 | 2 / 0.0196078 | 2 / 0.0196078 | 2 / 0.0196078 |
| 200 | 0.00497512 | 2 / 0.00497512 | 2 / 0.00497512 | 2 / 0.00497512 |
| 1000 | 0.000999001 | 2 / 0.000999001 | 2 / 0.000999001 | 2 / 0.000999001 |
| 4000 | 0.000249938 | 2 / 0.000249938 | 2 / 0.000249938 | 2 / 0.000249938 |
| 9000 | 0.000111099 | 2 / 0.000111099 | 2 / 0.000111099 | 2 / 0.000111099 |
| 10000 | 9.999e-05 | 2 / 9.999e-05 | 2 / 9.999e-05 | 2 / 9.999e-05 |
| 20000 | 4.99975e-05 | 2 / 4.99975e-05 | 2 / 4.99975e-05 | 2 / 4.99975e-05 |
| 100000 | 9.9999e-06 | 3 / 9.9999e-06 | 3 / 9.9999e-06 | 3 / 9.9999e-06 |

判读：三份拷贝在 β≤1e3 完全一致；`fsi_partitioned_stability`（floor 1e-3、`abs()`、`drop=1`）在
β=1e3 已多花 3 倍更新，β≥4e3 **发散**（预测阈值 999）；`fsi_added_mass`（floor 1e-4）与主实现同阈值发散。
即：m17727 报告 §⑥「同仓 `fsi_added_mass` 一直用 1e-4，两处实现不一致是本缺陷的根因」——计数错误（三处），
且只修了被审计的那一处。

### F3 [Medium] 三份副本的 API 语义不一致，缺陷容易再次漂移

`fsi_partitioned.step` 返回 `x + ωr`（松弛后状态）；`fsi_added_mass.step` 返回 `ωr`（增量）；
`fsi_partitioned_stability` 用 `update(residual, delta)` + `abs(ω)` + `drop`。同名类三种契约，
调用方换模块时会静默算错（本单 probe 必须为三个模块写三段适配器才可比）。已在各 docstring 显式写明差异；
未做合并重构（改动面 = 21 个 FSI 调用点，超出最小改动原则）。

### F4 [Medium] 验收测试没有跨越自己的判据边界 → 该类缺陷对测试不可见

m17727 的 T3 在 β=50 断言「默认收敛 / floor 0.05 发散」，恰好落在 0.05 的边界（39）之外、1e-4 的边界（9999）之内，
因此无法暴露 1e-4 仍是绝对下限。**本单新增** `tests/test_fsi_aitken_guard_scale.py` 把边界变成显式断言：
G1 β∈{1e4,2e4,1e5} 默认收敛且 ≥1e4 时旧 floor 发散、G2 三副本同因子且旧 floor 发散（含 stability 4e3）、
G2b 旧 floor 复现、G3 守卫语义/向后兼容、G4 β=1e6 无停滞（对 monolithic 误差 <1%）。

### F5 [Low] 文档措辞与代码不完全对齐

①「两处实现」→ 实为三处（F2）；②docstring 称 `clip=True` 时因子"bounded by [ω_min, ω_max]"，实现里**负候选因子直接被忽略**
（不裁剪、保留上一个正值），行为正确但描述含糊——本单在 docstring 中明确。③§⑥ 未显式回答报告纪律要求的"是否<3%"
（本单 §6 补齐）。

### F6 [Info·非 finding] §④ 表 4.4 中迭代代价偏差 66.4%/48.4%/17.6% 超过 15%，但 `predeclared.json` 已预先声明
`iteration_cost_rule: "K3 is diagnostic, not an accuracy gate"`，且要求偏差归因到列出的约定旋钮。
核对结论：**不构成虚报**，属于合规的预声明豁免（记录以备后续复判引用）。

## 3. 本单修改（最小、可验证）

| 文件 | 修改 |
|---|---|
| `src/tensorlbm/fsi_partitioned.py` | 新增 `AITKEN_POSITIVITY_EPS=1e-12` 与 `clamp_aitken_factor(omega, omega_min, omega_max)`（`omega_min<=0` = 尺度无关，仅保正）；`AitkenRelaxation` 默认 `omega_min: 1e-4 → 0.0`，裁剪走共享守卫，docstring 改写为阈值表 |
| `src/tensorlbm/fsi_added_mass.py` | 同一守卫；默认 `omega_min: 1e-4 → 0.0`；校验放宽为 `0 <= omega_min <= omega0`；docstring 写明"返回增量"契约 |
| `src/tensorlbm/fsi_partitioned_stability.py` | 同一守卫；默认 `omega_min: 1e-3 → 0.0`；`abs()` 语义保留并注明 |
| `tests/test_fsi_aitken_guard_scale.py` | 新增 9 个断言点（G1/G2/G2b/G3/G4），含修复前基线复现 |
| `xval/aitken_underrelaxation_life/scale_guard_probe.py`, `scale_guard_figures.py` | 复现/作图管线；`out/scale_guard.json`、`out/fig5_scale_guard.png` |

向后兼容：显式传 `omega_min=0.05/1e-4/1e-3` 的三条历史臂行为不变（表 1/表 2 可证），21 个 FSI 调用点全部用默认值。

## 4. 结果图

`xval/aitken_underrelaxation_life/out/fig5_scale_guard.png`
（a）四条臂的「每步子迭代数 vs β_eff」对数图，标出 β=39 与 β=9999 两条理论阈值线，发散点用红叉；（b）三份副本
回采因子 vs 解析 `ω*=1/(1+β)`，修复后三点重合（相对偏差 0）。

## 5. 验证（跑了什么）

- `PYTHONPATH=src python3 -m pytest tests/test_fsi_aitken_guard_scale.py -q` → **9 passed**（0.54 s）
- `PYTHONPATH=src python3 -m pytest tests/test_fsi_aitken_life_xval.py tests/test_fsi_aitken_guard_scale.py tests/test_fsi_partitioned.py tests/test_fsi_partitioned_stability.py tests/test_fsi_added_mass.py -q` → **101 passed**（24.6 s，无回归）
- `PYTHONPATH=src python3 xval/aitken_underrelaxation_life/scale_guard_probe.py && ... scale_guard_figures.py` → JSON 25 KB + PNG
- 容量纪律：本单全部为标量 1-DOF 计算（全套 probe < 1 s），未起重算力进程（本机 load15 ≈ 118 ≥ 100），未占集群。

## 6. 误差判定（是否 <3%）

| 关键量 | 基准 | 最大相对偏差 | <3%? |
|---|---|---|---|
| 回采松弛因子 ω（默认臂，β_eff 0.2→1e5） | 解析 `ω*=1/(1+β_eff)` | 2.18e-14 | **是** |
| 三份副本回采因子 | 同解析值 | 0（逐位一致） | **是** |
| 收敛界面位移（默认臂） | monolithic Newmark（离散真值） | 1.68e-04 % | **是** |
| 收敛界面位移（默认臂） | 闭式解 `analytic_sdof` | 0.0845 %（时间离散误差主导） | **是** |
| 修复前默认（1e-4）在 β=1e4 的解 | monolithic Newmark | 1.42 % | 是（但迭代 +42%，且 β≥2e4 无解） |

（每步子迭代数无独立基准：解析值依赖预测器/停机约定，m17727 预声明为诊断量；本单以"是否 = 3 步（Aitken 活跃 2 次后精确收敛）"作内部一致性判据。）

## 7. 未做/未覆盖（不得读作通过）

未执行 OSS LIFE 可执行代码（无外网）；未覆盖多自由度 FEM 结构（标量 Aitken 无法消多模态误差，LIFE 真场景）；
未覆盖非线性结构（Newton-Raphson 全量重解）；未覆盖 IBM 铺力/插值误差路径；未做 IQN-ILS 对照；
未跑 loong64 集群（无重算需求）；未验证 `fsi_added_mass.HeavingDiskPartitioned` 在 β>1e4 的 LBM 端到端行为。

## 8. 处置与建议

- disposition：**approve with notes → 修改已完成**（F1/F2 已在本单修复并加测试；F3/F5 记录为已知差异；F4 由新测试关闭）。
- 总账：建议 `XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXE` 置 `verified`，验收证据追加本单
  `out/scale_guard.json` + `tests/test_fsi_aitken_guard_scale.py`（原验收 <15% 在 ω/位移两面均以 ≤2e-14 / 1.7e-4% 达成）。
- 建议新条目（若不并入上条）：`DEFECT-FSI-AITKEN-GUARD-THREE-COPIES-UNIFY`，域=TensorLBM/FSI 耦合，
  量化验收：三份 `AitkenRelaxation` 在 β_eff ∈ [0.2, 1e6] 上回采因子相对偏差 <1e-6、默认臂全收敛、
  且删除任意两份拷贝后 `tests/test_fsi_*` 全绿（本单只做了守卫统一，未合并类本体）。
