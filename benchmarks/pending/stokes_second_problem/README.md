# stokes_second_problem（pending）

**状态：案例自身判据 VERIFIED（2026-11-03 首次机器判定）——但 prereg 原文掩码口径不达标，故不自行晋级 `verified/`，留 pending 待口径审计。**

| 项 | 机器测量值（源=`result.json` / `diag_H*.json`） |
|---|---|
| 判决口径 | \|u_ana\| > **10%·U0** 掩码下剖面 max 相对误差 ≤ 3%（+ 两档加密不变差 + 周期稳态差 ≤0.5%·U0） |
| H=40（δ/Δx=10） | max rel **2.160%**（φ=π/2 最差相位）；周期稳态差 0.0299%·U0 ✓ |
| H=80（δ/Δx=20） | max rel **1.012%**；周期稳态差 0.0232%·U0 ✓；跨网格 η=y/δ 塌缩差 ≤2.44%·U0 ✓ |
| 判决 | `result.json`: `passed=true`, `status="VERIFIED"` |
| **诚实披露（不达标项）** | prereg 骨架原文写「\|u_ana\|>**5%**·U0 掩码 ≤3%」；按该原文口径实测 **5.157%**（H=80 φ=π/2），**>3%，不达标**。两口径数值同时输出于每个采样点（`max_rel_err_5pct_ref_pct`），不做取舍性隐藏 |
| 分辨率使用边界（诊断档，非 prereg） | δ/Δx=**4**（H=16）φ=π/2 **15.76%**（失效）；δ/Δx=**40**（H=160）1.452%（≈1% 地板） |

## 物理问题

Stokes 第二问题（振荡平板）：无限平板在自身平面内以 U(t)=U0·cos(ωt) 振荡，一侧半无限静止流体经充分长时间后趋于周期稳态 Stokes 层，满足 ∂u/∂t = ν∂²u/∂y²，解析解 u(y,t) = U0·e^(−y/δ)·cos(ωt − y/δ)，其中 δ = √(2ν/ω) 为 Stokes 层厚度（黏性扩散穿透深度）。本案例取 δ = H/4（H 为域高），即 ω = 32ν/H²。这是振荡边界层（Stokes 层）的经典直接可观测基准：全场剖面 u(y) 对解析解逐点比较，且自带周期稳态自检（同相位早/晚周期剖面差）。对库的价值：验证 D2Q9 上 pre-streaming half-way 移动壁反弹在时变壁速 U(t) 下的动量注入正确性——与 verified/couette_2d、pending/couette_3d（恒定壁速）同一动壁路径的时间非定常扩展。

## 数值设置（真实模拟，无外推、无修正因子）

| 项 | 值 |
|---|---|
| 格子 / 碰撞 | D2Q9 / BGK（库原语 `d2q9.equilibrium/macroscopic` + `solver.collide_bgk/stream`，x 周期内建） |
| τ / ν / U0 | 0.65 / 0.05 / 0.05（Ma=0.0866） |
| 边界 | 顶壁 pre-streaming half-way 移动壁反弹（每步 u_w=U0·cos(ωt)）；底壁静止 half-way 反弹；x 向周期 |
| 网格档 | H∈{40,80}，均 δ=H/4 ⇒ δ/Δx=10/20；诊断档 H∈{16,160} ⇒ δ/Δx=4/40 |
| 采样 | 相位 ωt∈{0,π/2,π} × 早/晚两个周期（H=40: k=6/8；H=80: k=3/4；瞬态 e^(−ωt/2) 已衰减） |
| 工具链 | `torch.compile` 走 `benchmarks/compile_route.py`（默认 default，CPU/eager 回退），fp32，无 GPU 需求 |

## 计算结果（机器测量，程序化注入，零手抄）

### T1 per-record-point

| H | δ (cells) | k | phase | t (lattice) | max rel err, 10%·U0 mask [%] | max rel err, 5%·U0 mask [%] | L2 rel | max abs/U0 [%] |
|---|---|---|---|---|---|---|---|---|
| 40 | 10 | 8 | 0.0000 | 50265 | **0.214** | 4.695 | 0.00969 | 1.070 |
| 40 | 10 | 8 | 1.5708 | 51836 | **2.160** | 4.992 | 0.03306 | 1.374 |
| 40 | 10 | 8 | 3.1416 | 53407 | **0.221** | 4.694 | 0.00970 | 1.070 |
| 80 | 20 | 4 | 0.0000 | 100531 | **0.820** | 4.565 | 0.00986 | 1.133 |
| 80 | 20 | 4 | 1.5708 | 106814 | **1.012** | 5.157 | 0.03246 | 1.381 |
| 80 | 20 | 4 | 3.1416 | 113097 | **0.805** | 4.519 | 0.00983 | 1.133 |
| 160 (diag) | 40 | 4 | 0.0000 | 402124 | **1.435** | 4.328 | 0.01084 | 1.167 |
| 160 (diag) | 40 | 4 | 1.5708 | 427257 | **0.663** | 4.862 | 0.03280 | 1.383 |
| 160 (diag) | 40 | 4 | 3.1416 | 452389 | **1.452** | 4.200 | 0.01099 | 1.168 |
| 16 (diag) | 4 | 4 | 0.0000 | 4021 | **2.900** | 5.803 | 0.01327 | 0.881 |
| 16 (diag) | 4 | 4 | 1.5708 | 4273 | **15.758** | 15.758 | 0.04606 | 1.696 |
| 16 (diag) | 4 | 4 | 3.1416 | 4524 | **2.903** | 5.733 | 0.01326 | 0.884 |

### T2 steady+cost

| H | worst same-phase cycle-to-cycle diff [% U0] (gate ≤0.5%) | mass drift [%] | steps | reported wall time [s] |
|---|---|---|---|---|
| 40 | 0.0299 (steady=True) | 6.0808e-02 | 53407 | 12.2 |
| 80 | 0.0232 (steady=True) | 1.2613e-01 | 113097 | 30.3 |
| 160 (diag) | n/a (single pair, not gated) | 5.1113e-01 | 452389 | — |

### T3 cross-grid similarity

| phase (rad) | max |Δu|/U0 [%] in η=y/δ | mean |Δu|/U0 [%] | Same-y diff (not comparable) [%] |
|---|---|---|---|
| 0.0000 | 2.437 | 0.069 | 34.379 |
| 1.5708 | 2.115 | 0.072 | 21.984 |
| 3.1416 | 2.427 | 0.075 | 34.362 |

判决汇总（由 `result.json` 程序化汇总）：

```json
{
  "passed_10pct_case_own_gate": true,
  "status": "VERIFIED",
  "tol_max_rel_pct": 3.0,
  "worst_10pct_by_H": {
    "40": 2.16,
    "80": 1.0123
  },
  "worst_5pct_by_H": {
    "40": 4.9948,
    "80": 5.1572
  },
  "grid_converged": true,
  "steady_by_H": {
    "40": true,
    "80": true
  },
  "diag_worst_10pct": {
    "16": 15.7578,
    "160": 1.4523
  }
}
```


## 注意点（坑 / 局限 / 适用范围）

* **掩码口径冲突（必须披露）**：骨架设计文本写 5%·U0 掩码，继承下来的 run.py 主判据掩码为 10%·U0 并同时输出 5% 值。5% 掩码下实测 5.157%（>3%），10% 掩码下 2.160%。掩码敏感的物理原因：Stokes 层内 cos/sin 有过零点，|u_ana|→0 处相对误差发散而绝对误差仅 ~1.4%·U0；但这只是「为什么」，不是豁免——故本案例不自行晋级，晋级决定交审计。
* **两档网格解的是不同物理问题**（δ=H/4 以格计）：任何「同 y 剖面差」都按 O(δ) 存在（实测 22–34%·U0），不是离散误差；跨网格收敛判据必须落在相对误差或相似变量 η=y/δ 上（η 塌缩差 ≤2.44%·U0）。
* **分辨率阈值（本轮新测）**：δ/Δx=4 时 φ=π/2 最大相对误差跳到 15.76%，δ/Δx≥10 后 ≤2.16%（10/20/40 格 ⇒ 2.16%/1.01%/0.66%）——半程动壁 BB 的 Stokes 层需**每层 ≥10 格**才可用。
* 收敛条款只要求「加密不变差」，不构成精度阶估计；干净阶数需要固定 δ、仅加密 Δx 的变体（**未做**）。H=160 的 1.452% 与 H=80 的 1.012% 同量级，说明误差已在 ~1% 地板附近。
* 适用范围：低 Ma（0.0866）层流周期边界层；不含壁面曲率/压缩性/热效应；动壁路径为 half-way BB，与 verified/couette_2d 同一路径，本案例是其**时间非定常**孪生。

## 本轮修掉的 3 个骨架缺陷（此前案例永远写不出判决档的根因）

1. `period_convergence()` 返回 `numpy.bool_` → `json.dumps` 抛 `TypeError`，判决档写不出来（`bool()` 修复）。
2. scan 模式打印 `pc['H']` 时 `KeyError`——`period_convergence()` 契约上不带 `H` 键（result.json 已写、随后崩在打印）。
3. `compare_grids()` 把**降序**的 y_profile 直接喂给 `np.interp`（剖面按距壁距离存储，H+0.5−i 递减）→ 跨网格差读出 97.7%·U0 这种「两档各自对解析解 0.2% 却互差 97%」的自相矛盾值；已 argsort 修正，并把指标重定义为相似变量 η=y/δ 下的塌缩差（同 y 比较因 δ 随 H 变而物理上不可比，已在 result.json 注明）。回归守卫：`tests/test_stokes_second_problem_benchmark.py`（4 passed，~15 s CPU）。

## 晋级路径

按战役 Phase-0 停批协议走完整弧，晋级 = 归档 PR 将 pending/stokes_second_problem → verified/stokes_second_problem：

1. ✅ Phase-0 机器诊断：run.py pilot + 骨架缺陷修复（上节 3 条）+ 诊断档 δ/Δx=4/40（`diag_H16.json` / `diag_H160.json`）。
2. ✅ 参考锁 = Stokes 层解析式本身（u_ana 在驱动内机器生成、逐 case JSON 存档，零手抄）。
3. ⏳ **待审计**：掩码口径裁定（10%·U0 vs prereg 原文 5%·U0）。
4. 判决 PASS → 归档 PR 挪目录；FAIL → 按诊断定向修复后重走（库侧修复走独立库 PR）。

详见 `docs/mission_summaries/M4902_STOKES_SECOND_PR676_REOPEN.md`。

## 运行方式

```bash
# 单档
python run.py single H out.json [--tau 0.65] [--U 0.05] [--cycles 6 8] [--nx 8]
# 两档扫描（写 case_H*.json + result.json，打印判决）
python run.py scan out_dir --H 40 80 --device cpu --compile eager
# 回归守卫
python -m pytest tests/test_stokes_second_problem_benchmark.py -q
```

案例目录：`benchmarks/pending/stokes_second_problem/`。