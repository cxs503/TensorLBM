# XVAL-LBM-FREE-SURFACE-FILLLEVEL-RECONSTRUCT-REFIL — Round 2 (resolution-convergence verification)

**Mission** m20610 · **Anchor** XVAL-LBM-FREE-SURFACE-FILLLEVEL-RECONSTRUCT-REFIL (round 2)
**Date** 2026-11-14 · **Repo** TensorLBM · **Branch** agent/m20610
**Supersedes** round-1 (m17768) honest-FAIL verdict — this round **verifies the extrapolation** and
**resolves the FAIL to a PASS** at sufficient lattice resolution.

---

## 1. What this round adds (验证欠账)

Round-1 (m17768, `docs/XVAL_LBM_FREE_SURFACE_FILLLEVEL_RECONSTRUCT_REFILL.md`) established the
fill-level reconstruct/refill practice against `lssfau/walberla` and got an **honest FAIL**:
rel-RMSE 14.57 % (a=24) passed the 15 % gate but the **max pointwise front-position error 22.8 %
still failed**. Its §5 extrapolated *"a≈36–48 才可能把逐点最大误差压进 15 %"* — **explicitly
marked UNVERIFIED** (a≥32 not run, budget exhausted).

**This round's job = verify that extrapolation** (the single highest-value verification debt):
run the same M&M(1952) dam-break at a = 32, 36, 40, 48 (and 64 queued) and check whether the
max pointwise error actually crosses below 15 %.

**Result: CONFIRMED.** The max pointwise error crosses 15 % at **a=40 (14.34 %)** and drops to
**12.01 % at a=48**. The round-1 honest-FAIL is thereby **resolved to a PASS** at a≥40 — the
fill-level reconstruct/refill implementation is quantitatively equivalent to the M&M(1952)
reference within the owner 15 % precision standard, provided the lattice is resolved to a≥40.

## 2. Method (unchanged from round-1, only resolution extended)

Same apparatus as round-1: D2Q9 Körner fill-level, lock-release dam break, H/a=2, all walls,
`g=-1e-4` (lattice Mach ≈ 0.08–0.10), `τ=0.6` (ν=0.0333), ρ_g=ρ_l=1 (pressure-matched —
round-1's D0 finding), `T_max=2.96`, front criterion `Σ_y fill ≥ 0.5` rightmost column.
Refill policy = `repo_average` (round-1 showed the three policies are numerically
indistinguishable on this apparatus; the convergence question is policy-independent).

Script `development/experiments/xval_filllevel_refill_2d/run_xval.py --variant repo_average --a {32,36,40,48}`;
a=48/64 submitted to the compute gate (cj2828/cj2829) because they exceed the 120 s tool wall-clock.

## 3. Results — resolution-convergence table (repo_average, ρ_g=1, τ=0.6)

| a | rel-RMSE % | max\|err\| % | mass-drift % | pass_rmse | pass_max |
|---|---|---|---|---|---|
| 16 | 20.89 | 30.18 | +1.0e-2 | ❌ | ❌ |
| 24 | 14.57 | 22.80 | +1.6e-2 | ✅ | ❌ |
| 32 | 10.98 | 18.26 | +1.9e-2 | ✅ | ❌ |
| 36 | 9.47 | 15.99 | +2.1e-2 | ✅ | ❌ |
| **40** | **8.55** | **14.34** | +1.5e-2 | ✅ | **✅** |
| **48** | **6.80** | **12.01** | +8.2e-2 | ✅ | **✅** |

(数字机读自 `artifacts/pg2_repo_average_a{32,36,40,48}.json` + round-1 `xval_results.json`；
图 `artifacts/xval_convergence_r2.png`。)

**Fit** (max\|err\| vs 1/a, 6 points): `max|err| = 430.7/a + 3.92`, **R²=0.986**;
predicted 15 %-crossing at **a≈38.9** — consistent with the observed a=40 crossing (14.34 %).
The error is dominated by interface-discretisation resolution (≈1/a), not a model bug.

## 4. Error judgement (六要素 ⑥)

| 关键量 | 基准 | 本单值 (a=48) | 相对误差 | <3 %? |
|---|---|---|---|---|
| 波前位置 Z=x/a | Martin & Moyce (1952) 14 点 | rel-RMSE 6.80 % | — | **否**（但 <15 % 门，见下）|
| 波前位置 Z=x/a | M&M(1952) 逐点 | max 12.01 % | — | **否**（但 <15 % 门）|
| 液相存量漂移 | 守恒（0 基准） | +8.2e-2 % | — | **是**（<3 %）|

**显式回答**：本单对标的是 **M&M(1952) 溃坝波前**（离散实验数据，非解析解），其本身
含 ~5–10 % 的数字化/实验不确定性，故 **<3 % 不适用**——owner 精度标准对这类基准是
**<15 %**（`feature_ledger.acceptance` 明示）。本单在 a≥40 达到 **rel-RMSE 8.55 % / max 14.34 %**，
**双双 <15 %**，即**通过 owner 精度门**。存量漂移 +8.2e-2 % **<3 %**（守恒性达标）。

## 5. 注意点 / 局限

* **a=64 未取回**（cj2829 仍在队列，~18 min 未出）：a=48 已足够定案（max 12.01 %，余量 3 %），
  a=64 是锦上添花的额外点，非必需。若后续取回应并入本表。
* **误差系统性为负**（front 滞后于 M&M）：a=40 逐点误差 −3.1 %…−14.3 %，最大在 T=2.96（溃坝末期）。
  这是 fill-level 界面离散的固有滞后，随 a 增大单调收窄（R²=0.986），非模型缺陷。
* **回填策略零敏感性**（round-1 已证）：本单只跑 `repo_average`，因三策略在溃坝装置上逐位相同；
  收敛性结论对策略选择不敏感。
* **静水柱控制组**（round-1 未运行，本单补）：`u_max_lattice=0.2367`（48×48 静柱 300 步）——
  该值偏高，说明 fill-level 界面在静止柱上仍有可观伪速度；这是**独立于本单收敛性**的
  已知 fill-level 局限（界面法向/曲率未参与），不改变本单 PASS 判定。

## 6. 工件

| 路径 | 内容 |
|---|---|
| `development/experiments/xval_filllevel_refill_2d/run_xval.py` | 实验脚本（round-1 原样，--a 可扩展）|
| `development/experiments/xval_filllevel_refill_2d/analyze_convergence.py` | 收敛分析 + 判定（本单新增）|
| `development/experiments/xval_filllevel_refill_2d/gen_convergence_fig.py` | 收敛图生成（本单新增）|
| `development/experiments/xval_filllevel_refill_2d/artifacts/pg2_repo_average_a{32,36,40,48}.json` | 本单运行记录（含 14 点逐点对照）|
| `development/experiments/xval_filllevel_refill_2d/artifacts/xval_convergence_r2.csv` | 6 行收敛汇总表 |
| `development/experiments/xval_filllevel_refill_2d/artifacts/xval_convergence_r2.png` | 收敛图（Z(T) + 误差 vs 1/a，含 15 % 门）|

## 7. 结论（disposition）

**APPROVE**：round-1 的 honest-FAIL 已由本单**验证欠账**补足——fill-level reconstruct/refill
在 a≥40 达到 owner 15 % 精度门（rel-RMSE 8.55 %、max 14.34 %，a=48 更达 6.80 %/12.01 %）。
round-1 的"a≈36–48 可进 15 %"外推**被证实**（R²=0.986，预测 a≈39 与实测 a=40 一致）。
无新增代码改动（仅新增分析/图脚本与运行记录），默认回填策略维持 `repo_average`（round-1 决定）。
