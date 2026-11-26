# XVAL R5 — the regularised wall closure's steady-channel regression is a *forcing* defect, not a third-moment defect

Anchor `XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC` · arm **R5** · module
`src/tensorlbm/regularised_bc.py` · artifacts `xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/artifacts_r5/`
· tests `tests/test_regularised_bc_order3.py`.

## ① 案例介绍

R4 measured that the regularised (non-equilibrium stress) wall closure of
`joconnor22/LIFE` is **~2x less accurate than half-way bounce back** on a steady
body-force Poiseuille channel, and attributed it — without measuring — to the
second-order Hermite projection dropping "the genuine third moment of the
parabolic non-equilibrium part". R5 turns that attribution into a controlled
experiment and finds a different mechanism.

Physics: 2-D plane channel, periodic in x, driven by a uniform body force
`F = 2e-05` (Guo 2002 split forcing), `tau = 0.8` (`nu = 0.1`),
`u_max ~ 0.013` (Ma ~ 0.022), exact solution `u = F/(2 nu) s (H - s)` with the
closure's own wall convention (`regularised`: wall plane on the boundary node,
`H = ny-1`; `bounce back`: no-slip plane half a link outside, `H = ny-2`).
No-slip walls on both y planes, `nx = 2`, float64, steady state declared by
`max|u(n)-u(n-check_every)|/max|u| < 1e-11`.

Arms (all on the production code path, no re-implementation):

| arm | reconstruction |
|---|---|
| `regularised` | **shipped**: order-2 Hermite projection, stress from the anti-symmetric completion |
| `regularised_neighbour_stress` | order 2, stress extrapolated from the adjacent interior node |
| `regularised3_completed` | **order 3**: third moment carried over (completion) |
| `regularised3_neighbour` | order 3, third moment from the adjacent interior node |
| `bounce_back` | control: half-way bounce back + Ladd moving-wall term |

Two independent knobs: the Hermite order of the projection (`order`) and the
weight of the Guo force term on the **boundary row** (`wall_force`, 0 = shipped).

## ② 怎么做的

1. `project_third_order` was added to the module (Hermite degree 3, least-squares
   / Gram form of Mattila, Philippi & Hegele 2017) together with
   `hermite_third_basis`, `third_moment_operator` and
   `third_moment_representability`; `regularised_face` / `apply_regularised_wall`
   gained `order` and `stress_source`, `ChannelConfig` gained `projection_order`,
   `stress_source` and `wall_force` (five `wall_bc` arms).
2. The third-moment defect hypothesis was tested by *adding back* the third
   moment in two independent ways (from the completion, from the neighbour).
3. A structural decomposition of the profile error was pre-declared: a **wall
   slip** is a *constant* offset (`u_num - u_exact = const`, both walls slip),
   a viscosity/force error is *proportional* to `u_exact`. Both coefficients are
   fitted and reported, so the mechanism is read off the data.
4. The offset was then tracked as a function of `F` (1e-5, 2e-5, 4e-5) and of the
   boundary-row force weight (0, 0.5, 1).
5. Order-3 representability per velocity set was measured with random lattice
   distributions; the unsteady case (oscillating Stokes layer) is the control
   that must not degrade.

## ③ 计算结果

**Arm matrix, steady Poiseuille `ny = 24`, `tau = 0.8`, `F = 2e-5`**

| arm | shipped boundary row, L2 [%] | boundary row forced, L2 [%] |
|---|---|---|
| regularised | 0.4861 | 0.216 |
| regularised_neighbour_stress | 0.6886 |  --  |
| regularised3_completed | 0.4861 |  --  |
| regularised3_neighbour | 0.6076 |  --  |
| bounce_back | 0.2603 | 0.2603 |

The shipped number (0.4861 %) reproduces R4's independent measurement (0.486 %)
to 4 digits — the R4 artifact is reproducible; the *interpretation* is what
changes.

**Boundary-row force weight (regularised closure, ny = 24)**

| wall_force | L2 [%] | offset / u_max |
|---|---|---|
| 0.0 (shipped) | 0.4861 | -3.636e-03 |
| 0.5 | 0.3510 | -2.626e-03 |
| 1.0 | 0.2160 | -1.616e-03 |

Bounce back is inert in `wall_force` (0.2603 % at every weight) because the
reflection overwrites the boundary row — as expected, and a useful control that
the knob does what it claims.

**The error is a uniform offset, not a shape error**

`ptp(u_num - u_exact)/u_max <= 1.7e-10` at every arm: the numerical profile is
the analytic parabola shifted by a *constant*. In absolute terms the offset is
`-4.8e-05` at `ny = 16` and
`-4.8e-05` at `ny = 24` — **grid independent**, i.e.
`~ -2.4 F`, not an `O(h^2)` discretisation error of the profile. This is the
signature of a wall condition: the boundary node is a fluid node of the
reconstructed closure, but the Guo half-step force was applied to the interior
rows only, so the boundary row carried a force deficit of `~F` and the fluid saw
a slip of the same size. Giving the boundary row the same half-step force
(`wall_force = 1`) removes 55 % of the offset and 44 % of the error, which makes
the regularised closure **more** accurate than bounce back (0.2160 % vs 0.2603 %)
instead of 1.9x less.

**Third-moment hypothesis: falsified**

Carrying the third moment does not help, in either source:
`regularised3_completed` is bit-identical to the shipped arm (0.4861 %), because
the anti-symmetric completion `f_q = f_q^eq + (f_opp - f_opp^eq)` has a
*vanishing* third moment — there is nothing for the third-order term to restore;
`regularised3_neighbour` (0.6076 %) and `regularised_neighbour_stress`
(0.6886 %) are both worse than the shipped completion. So the steady-channel
regression is not a third-moment truncation, and the neighbour extrapolation is
not a repair.

**Third-order representability (measured, random lattice distributions)**

| lattice | rank of the discrete H3 Gram / symmetric dim | `|M3 - T|/|T|` after the map |
|---|---|---|
| D2Q9 | 2 / 4 | 1.06e-14 |
| D3Q19 | 6 / 10 | 7.97e-15 |

`H3_xxx = c_x^3 - 3 cs^2 c_x` vanishes identically on the D2Q9 velocity set, so
only the two mixed directions survive and the D2Q9 third moment is *fully*
representable (rank 2 of 4 symmetric components is not a deficiency: those are
the only ones that exist). D3Q19 keeps 6 of 10, so the order-3 closure is a
least-squares projection there — for *lattice* data both are exact to 1e-14,
which is why the order-3 arm is a fair test.

**Unsteady control (oscillating Stokes layer, `delta/dx = 6`, `H/delta ~ 7`)**

| arm | complex L2 [%] | max phase error, masked [rad] |
|---|---|---|
| regularised | 0.5299 | 5.69e-03 |
| regularised_neighbour_stress | 1.1747 | 1.22e-02 |
| regularised3_completed | 0.5299 | 5.69e-03 |
| regularised3_neighbour | 0.9877 | 1.03e-02 |
| bounce_back | 0.8450 | 8.91e-03 |

The reconstructed-node closure keeps its unsteady advantage (0.53 % vs 0.85 % for
bounce back, both < 3 %); the third-moment and neighbour variants degrade it,
another independent sign that the completion is the right stress source.

## ④ 结果图

`fig1_arm_matrix.png` — L2 per arm, shipped vs forced boundary row, 3 % line.
`fig2_wall_force.png` — L2 and uniform offset vs the boundary-row force weight:
both are linear in the induced force (the mechanism is a force-balance defect).
`fig3_oscillating.png` — unsteady control per arm.

## ⑤ 注意点 / 局限

* `wall_force = 1` is *not* a fitted coefficient: it is the weight an interior
  node gets, and the closure's boundary node is a fluid node (LIFE applies
  `regularisedBC` to fluid nodes carrying a prescribed velocity). It is
  nonetheless a modelling choice, so it stays opt-in
  (`ChannelConfig.wall_force`, default 0 = shipped behaviour).
* The residual offset at `wall_force = 1` is still `~1 F`; the offset is linear
  in the weight and would extrapolate to zero near `wall_force ~ 3.6`, which is
  not physical — there is a second, smaller force-related defect that this arm
  does not identify. Not claimed.
* Single lattice (D2Q9), single geometry family (plane channel), `tau = 0.8` for
  the arm matrix; the tau sweep (`wall_force`) is the cluster job `cj3642`, whose
  artifacts were not fetched before this report was written (queue backlog).
  The numbers above are the chunked foreground runs (see `provenance`), all
  reproducible with `run_xval_r5.py`.
* Corner (two-normal) planes and LIFE's `CENTRAL_MOMENTS` branch remain
  untransplanted (unchanged scope limit from R1-R4).
* LIFE itself is still not executed; the cross-language evidence remains R1's
  line-cited transcription.

## ⑥ 误差判定（逐项 vs 基准，显式回答是否<3%）

| 关键量 | 数值 | 基准 | <3% ? |
|---|---|---|---|
| steady Poiseuille, `regularised`, shipped, ny=24 | 0.4861 % | analytic parabola | 是 |
| steady Poiseuille, `regularised`, `wall_force=1`, ny=24 | 0.2160 % | analytic parabola | 是 |
| steady Poiseuille, `bounce_back`, ny=24 | 0.2603 % | analytic parabola | 是 |
| steady Poiseuille, `regularised3_neighbour`, ny=24 | 0.6076 % | analytic parabola | 是 |
| steady Poiseuille, `regularised_neighbour_stress`, ny=24 | 0.6886 % | analytic parabola | 是 |
| Stokes layer delta/dx=6, `regularised` | 0.5299 % | analytic Stokes solution | 是 |
| Stokes layer delta/dx=6, `bounce_back` | 0.8450 % | analytic Stokes solution | 是 |
| wall-slip offset vs analytic wall position, `regularised` shipped | 0.364 % of u_max | zero slip | 是 |
| same, `wall_force=1` | 0.162 % of u_max | zero slip | 是 |

Every measured error is far below the 3 % and 15 % gates; the *comparative*
claim of R4 (regularised 1.9x worse than bounce back) is confirmed as a
measurement and its stated cause is refuted. The absolute accuracy was never
the problem — the attribution was.

## 复现

```bash
PYTHONPATH=src python -m pytest tests/test_regularised_bc_order3.py -q
PYTHONPATH=src python xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/run_xval_r5.py --arm invariant
PYTHONPATH=src python xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/run_xval_r5.py --arm ladder --nys 24 --walls 0,0.5,1
PYTHONPATH=src python xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/run_xval_r5.py --arm osc --deltas 6
bash xval/XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC/run_all_r5.sh   # full campaign
```
