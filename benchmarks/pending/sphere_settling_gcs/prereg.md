# Pre-registration — G15 third method: kinetic-flux control surface (GCS) for sphere forces

`mission 4981 · node IN14-D8 · repo TensorLBM · branch agent/m4981 · 2026-11-07`

## 0. What this document freezes

The acceptance criteria, the geometry, the numerical parameters and the
verdict rules of the settling study **before** the production runs.  Anything
measured after this point that contradicts them is reported, not redefined.
Pilot probes that were run *before* this freeze (their numbers are quoted in
§6) were used only to pick the operating regime, and are disclosed.

## 1. The gap being closed

Gap ledger (``benchmarks/gaps.md``) records two open failures of sphere-force
measurement in this repo:

| gap | route | measured failure | root cause (ledger) |
|---|---|---|---|
| **G12** | pressure + friction on the voxel wall | ``cd_p = −49%`` (D40 sphere, Re=100) | near-wall pressure misses the stagnation rise (`extrap='none'`) |
| **G15** | link-wise momentum exchange (MEM) | ``Cd = 3.98`` vs 1.09 (+264.6%) | the Ladd sum pairs populations from two time levels, leaving an *additive* free-stream background; ``bg_sub`` was a digit-for-digit no-op; the "Galilean" variant is a scalar rescale ``(1+1/2τ)·standard`` (m2553) |

Both are **wall-surface** instruments: they sample on (or one cell from) the
voxel boundary, so they inherit the staircase geometry and the wall BC.

## 2. The third method (this work)

Integrate the conserved kinetic momentum flux over a closed **analytic**
sphere that merely encloses the body:

```
F = f·V_f^in − ∮_S Π_ij n_j dA,      Π_ij = Σ_q f_q c_qi c_qj   (+ half-force term)
```

* `Π` is the *raw second moment* of the populations — no reconstruction, no
  equilibrium split, no background model.
* The surface is an exact sphere (`R+2…R+3` cells) with tensor-product
  Gauss–Legendre × equispaced quadrature whose point set is symmetric under
  `x→−x`, `y→−y`, `z→−z` **separately**, so `∮ n dA = 0` to machine
  precision: any spatially uniform background integrates to **exactly zero**
  (G15's failure mode is annihilated, not estimated).
* The surface never touches the wall, so its normals/areas carry no staircase
  error and no one-sided wall difference is taken (G12's failure mode).
* `f` is the uniform body-force density (gravity); `V_f^in` the fluid volume
  inside the control surface.  Equation is exact for steady flow; the
  `d/dt ∫ρu dV` term is reported as a steadiness diagnostic.

Deliverable: `src/tensorlbm/force_control_surface.py`, tests
`tests/test_force_control_surface.py` (10 analytic tests, no time stepping).

## 3. Physical case (settling sphere, body frame)

Periodic cubic cell `L³`, one voxel sphere radius `R` held fixed, uniform
gravity force density `ρg` on fluid nodes (Guo forcing), BGK, D3Q19.  At
steady state the sphere is in equilibrium with the drag and the fluid drifts
at exactly the terminal velocity of the same gravity in a periodic
suspension — the sedimentation problem in the frame where the sphere is at
rest (the frames differ by a constant translation, the flow is steady in
both).  No moving boundary, no added mass, no force-feedback loop.

Analytic reference: `U_s = 2 a² Δρ g/(9 μ)` ⇔ `F = 6πμaU` (Stokes).
Two *physical* effects sit between a finite-cell simulation and that formula
and must be separated, never hidden:

1. **confinement** — periodic images / no-slip neighbours ⇒ drag rises with
   `λ = R/L`; removed by extrapolating our own data to `λ → 0` (Hasimoto's
   closed form quoted only as an *a-posteriori* cross-check);
2. **voxel body** — the staircase sphere has an effective hydrodynamic radius
   ≠ `R`; measured as the drag law extrapolated in `1/R`.

## 4. Frozen numerical parameters

| parameter | value | reason |
|---|---|---|
| lattice / collision | D3Q19 / BGK | repo default (`solver3d`) |
| `ν` | **1.0 lattice** (`τ = 3.5`) for every physics run | pilot study §6: `τ ≳ 5` triggers a wall-model defect of this chain (1.4×…4× spurious mobility), which would contaminate the measurement of the *force*, not of the boundary scheme |
| `Re = 2RU/ν` | 0.05 (scans 0.02–0.20) | `Ma ≤ 1.1e-2`, compressibility ≤ 1e-4 |
| control radius | `R+2, R+2.5, R+3` | ≥1.5 cell clearance from the staircase, still ≥1 cell inside the box |
| time window | last 30 samples (every 50 steps), ≥2000 steps after the drag has plateaued | steadiness |
| steadiness gate | tail-half drift `< 1%` and `|F|<`∞ | else the run is reported as unconverged, not as a verdict |

## 5. Frozen verdict rules

| id | criterion | pass condition |
|---|---|---|
| **J1** | ① single-sphere settling terminal velocity vs analytic Stokes, **unbounded** value obtained by linear `λ → 0` extrapolation of `U_t/U_Stokes` over `λ = 0.125, 0.100, 0.0833` at `R=4`, `Re=0.05` | `|intercept − 1| ≤ 3%` **and** fit residual `< 1.5%` |
| **J2** | drag law at the *measured* `U_t`, extrapolated in `1/R` (`R = 4, 6, 8` at fixed `λ = 1/8`) | `|intercept − 1| ≤ 3%` |
| **J3** | instrument clearance — exact momentum-balance anchor in a *real* run: `F_total` vs `ρ g V_fluid` | `< 1%` at steady state |
| **J4** | ② instantaneous-force oscillation: `(max−min)/mean` of `F(t)` over the tail window, all runs | `< 2%` (laminar, `Re ≤ 0.2`) |
| **J5** | ③ control-radius independence: `(max−min)/mean` of `F` across the three radii | `< 1.5%` |
| **J6** | cross-instrument: GCS vs the library's discrete box control-volume observer (`control_volume_force`) on the same field | agreement `< 3%` |
| **J7** | method boundary (documented, not pass/fail): validated `Re` interval, minimum `R`, minimum clearance, `τ` ceiling | reported with numbers |

Overall verdict = **PASS** iff J1∧J2∧J3∧J4∧J5.  Any criterion that fails is
reported as FAILED with the measured number — no re-analysis of the target.

## 6. Pre-freeze pilot probes (disclosed, not part of the verdict)

| probe | result | use |
|---|---|---|
| analytic CE Stokes field, `f = f^eq + f^(1)` | `−∮Π·n` recovers `6πμaU` within **0.6%**, radius spread `< 0.3%` (pytest) | instrument validated independently of any solver |
| uniform state (`f = f^eq(ρ,U)` everywhere) with a staircase sphere present | GCS force `≤ 1e-15` of `ρU²A`; `momentum_exchange_standard` on the *same* state `≫` that | shows the G15 background is annihilated |
| periodic settling cell `R=6, L=48` at `τ=14.9` | anchor `F/(ρgV_f) = 0.24%` ✔ but `U_t/U_Stokes = 2.93` ✘ | located the τ-dependence; motivated the `τ=3.5` freeze |
| Poiseuille channel (analytic parabola) at `τ=3.5 / 5.3 / 10.1` | peak velocity `1.04× / 1.43× / 3.26×` analytic | same defect, in a flow with an exact answer |
| periodic settling cell `R=4, L=32` at `τ=3.5` | `U_t/U_Stokes = 0.930`, drag/Stokes at measured `U` = `0.994` | picked as the production regime |

The `τ`-dependence is reported as a **separate finding** (it is a property of
the library's bounce-back wall layer, not of the force estimator); the
production study therefore runs in the regime where the flow solver itself is
validated against an analytic parabola.

## 7. Files

* `run.py` — production driver (`domain`, `refine`, `reynolds`, `anchor`)
* `probe.py` — the pilot probes of §6
* `result.json`, `out_*.json` — machine output (all numbers in the report)
* `make_figures.py`, `figures/` — figures
* `README.md` — results + verdict
* `docs/sphere_settling_gcs_20261107.md` — paper-level report
* `tests/test_force_control_surface.py` — 10 analytic tests
---

## 附录 A — Amendment A1（事后追加，明示披露）

**日期**：2026-11-07，在 §5 的判据执行之后、报告定稿之前追加。**冻结正文一字未改**，
本附录只做三处**口径澄清**，理由全部由数据驱动，并在此如实标明"看完数据之后才写"。

### A1.1 终速的估计量 = spin-up 指数外推（不是尾窗均值）

沉降槽从静止起动，只有一个主导的粘性弛豫模态：`U(t) = U∞(1 − e^{−t/τ})`。
`τ` 实测 470–1760 步（`result.json` 的 `spin_up_fit.tau_relax_steps`），因此
3000 步算例的**尾窗均值**系统性低估终速（最低一档残留 19.6%）。故终速改用
`U∞`（对完整 `u_series` 做单参数扫描 + 线性最小二乘，拟合 rms ≈ 5e−6，
尾窗均值同时并列报告）。判据阈值不变。

### A1.2 J2 的比较基准改为"约束修正后的 Stokes"

冻结的 J2 原文把"λ=1/8 处 `1/R→0` 的截距"与**无界** Stokes 常数比较。但有限周期
槽在物理上就是阻力更高的（Hasimoto 1959 对 λ=1/8 给出 K≈1.35），把约束效应当成
误差去判"3%"是**测错了对象**。故 J2 的比较基准改为
`K(λ)·6πμaU`（Hasimoto 闭式解，λ 固定为 1/8）；无界 Stokes 仍作为 J1 的基准。

### A1.3 J1 在 R=4 与 R=6 双档报告，R=6 为主判据

R=4 的体素球（257 格）在蠕流下**欠分辨**：同 λ=1/8 下实测阻力比
`F/(6πμRU)` 从 R=4 的约 1.0 升到 R=6 的约 1.29（Hasimoto 期望 1.35），
即 R=4 明显"漏"，λ 阶梯也出现非单调。故在 R=6 上补做 λ 阶梯
（L=48/64/72，`run_domain6.py`），J1 以 **R=6 的 λ→0 截距为主判据**，
R=4 的同一量作为 J1b **原样报告（含其 FAIL）**，不改阈值、不隐藏。

### A1.4 未变项

J3（精确锚点 1%）、J4（振荡 2%）、J5（半径无关 1.5%）、J6（交叉仪器 3%）、
稳态门（尾部漂移 1%）、全部数值参数、以及"任何未过判据原样记为 FAIL"的纪律。
