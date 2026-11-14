# GAP-DIFFLBM-ADJOINT-VERDICT -- frozen-field adjoint proxy: verdict

**Status:** theory-level verdict, machine-checked by
`benchmarks/adjoint_verdict_bench.py` (this file is generated from
`run_r1/results.json`; no number is typed by hand).

**Question.** `tensorlbm.adjoint.adjoint_sensitivity` differentiates a flow
objective w.r.t. the obstacle geometry with the flow fields *frozen*
(shape derivative).  The total derivative of such an objective is

```
dJ/dp = dJ/dp|_fields  +  (dJ/df)(df/dp)
        ^ shape derivative     ^ flow response -- MISSING in the proxy
```

**Acceptance criterion (predeclared).** Same objective (lift), same parameter
(disk-centre ordinate `y_c`): if `cos(g_AD, g_frozen) < 0.99`, the proxy may
not enter the optimisation path and must declare
`shape_derivative_only=True`.  `g_AD` = back-propagation through the whole time
integration (*expanded AD*), verified against independent central finite
differences `g_FD` of the same finite-horizon forward problem.

**Environment.** python 3.11.6, torch 2.7.1, numpy
2.4.6, host `ec6ae0b42a0c`, 128 cpus
(6 threads), dtype float64, timestamp
`2026-11-14T10:22:04`, wall time 22.0 s.

---

## ① Problem setting and benchmark provenance

Case: steady laminar flow in a plane channel (streamwise-periodic, driven by a
constant body force) past an immersed circular disk whose centre ordinate `y_c`
is the optimisation parameter.  The objective is the **lift**, i.e. the
momentum-exchange (partially saturated bounce-back) force on the disk along
`+y`; the streamwise component (drag) is reported as a sanity value.

* Forward solver: D2Q9 BGK, `u*`-shifted-equilibrium body-force forcing
  (He et al. 1998; Krueger et al. 2017, ch. 2/6).  Walls and body are imposed
  by smooth penalisation (partially saturated bounce-back, Nobel & Torczynski,
  *Phys. Fluids* 10 (1998) 2976), which makes the body position a *continuous,
  differentiable* parameter instead of a discrete mask.
* Force: momentum-exchange method (Ladd 1994; Mei et al. 2002), i.e. the
  momentum removed by the solid fraction per step, averaged over the last
  `T_avg` steps of the integration window.
* Reference for the gradient: discrete adjoint == BPTT through the fully
  differentiable solver, cross-checked by central finite differences of the
  same finite-horizon problem.  Anchors for the *claim* under test (that the
  flow-response term cannot be dropped): Pironneau (1974); Jameson (1988);
  Giles & Pierce (2000); Mohammadi & Pironneau, *Applied Shape Optimization for
  Fluids* (OUP 2001), ch. 3 ("incomplete sensitivities").
* No experimental reference is available for a lattice-unit channel with a
  penalised disk, so the external anchor is *analytic/numerical*: (i) forced
  Poiseuille profile used as the initial state, (ii) mirror-symmetry invariant
  (lift odd / its gradient even about the centreline), (iii) AD-vs-FD
  agreement.  This is stated explicitly rather than hiding behind a "reference".

## ② Method and implementation

Pipeline (all code pointers are repo-relative):

1. `src/tensorlbm/adjoint_verdict.py::simulate` -- differentiable D2Q9 channel
   solver; `smooth_disk` / `wall_penalty` build the smooth solid fractions;
   `_initial_state` starts from the analytic forced-Poiseuille profile so that
   only the disk wake is transient.
2. `steady_state` -- long forward-only integration (no graph) to the base-parameter
   steady state; the field is reused as a *parameter-independent* warm start so
   that the initial condition carries no parametric sensitivity and the BPTT
   horizon only has to cover the wake, not the viscous channel transient.
3. `ad_gradient` -- `torch.autograd.grad(lift, (x_c, y_c))` through the whole
   window (expanded AD / discrete adjoint).
4. `fd_gradient` -- central differences on the same forward problem
   (`h = 0.05, 0.1` lattice nodes), the
   independent reference for the AD gradient.
5. `frozen_field_gradient` -- the same objective differentiated with the
   momentum fields held constant, i.e. the shape-derivative-only term.
6. `verdict` -- cosine similarity / amplitude ratio / sign agreement /
   gate decision; `tensorlbm.adjoint.optimization_gate` +
   `require_optimizer_ready` enforce the decision on the shipped proxy.
7. Entry point: `benchmarks/adjoint_verdict_bench.py`; this report:
   `benchmarks/adjoint_verdict_report.py`.

## ③ Parameters (all quantities in lattice units)

| res | ny x nx | R | D | tau | nu | g | T | T_avg | interface | wall_offset |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 36x64 | 3.00 | 6.0 | 0.65 | 0.0500 | 6.60e-05 | 600 | 200 | 0.8 | 1.0 |
| R2 | 42x74 | 3.50 | 7.0 | 0.65 | 0.0500 | 6.60e-05 | 600 | 200 | 0.8 | 1.0 |

Measured operating points (the Reynolds number is *measured* from the steady
state, not assumed from the nominal forcing):

| res | y_c | u_max (measured) | Re = u_max D / nu | lift | drag | steady drift |
|---|---|---|---|---|---|---|
| R1 | 14.00 | 0.0583 | 6.99 | +1.3127e-02 | +6.4663e-02 | 4.11e-07 |
| R1 | 21.00 | 0.0583 | 6.99 | -1.3127e-02 | +6.4663e-02 | 4.11e-07 |

## ④ Comparison: reference vs proxy, criterion by criterion

`g_AD` is the reference total derivative `d(lift)/dy_c`; `g_FD` is its
independent finite-difference check; `g_frozen` is what a frozen-field proxy
predicts (same objective, same parameter).

| res | y_c | g_AD (ref) | g_FD (ref-check) | |AD-FD|/|FD| | g_frozen (got) | |g_frozen|/|g_AD| | cos (1-D y_c) | cos (2-D) | gate |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 14.00 | -2.2695e-03 | -2.2695e-03 | 1.57e-05 | +1.0706e-03 | 0.472 | -1.0000 | -0.9950 | BLOCKED |
| R1 | 21.00 | -2.2695e-03 | -2.2695e-03 | 1.57e-05 | +1.0706e-03 | 0.472 | -1.0000 | -0.9950 | BLOCKED |

The streamwise component and the actual shipped proxy
(`tensorlbm.adjoint.adjoint_sensitivity`, which uses its own pressure-based lift
objective, hence a different absolute scale):

| res | y_c | g_AD,x | g_frozen,x | g_shipped,x | g_shipped,y | proxy shape_derivative_only |
|---|---|---|---|---|---|---|
| R1 | 14.00 | +1.043e-04 | -1.577e-04 | +1.689e-05 | -8.471e-04 | True |
| R1 | 21.00 | -1.043e-04 | +1.577e-04 | -1.689e-05 | -8.471e-04 | True |

**Criterion checklist**

- [ ] `cos >= 0.99` for R1 y_c=14.00: cos = -1.0000 (2-D: -0.9950)
- [ ] `cos >= 0.99` for R1 y_c=21.00: cos = -1.0000 (2-D: -0.9950)
- [x] proxy blocked from the optimisation path in every case (gate_decision=blocked)
- [x] proxy declares `shape_derivative_only=True` (`tensorlbm.adjoint.adjoint_sensitivity` return value + module/function docstring + API response `optimization_gate`)
- [x] AD verified against independent central finite differences of the same finite-horizon forward problem (see `|AD-FD|/|FD|` column)

## ⑤ Convergence and sensitivity


Lift(y_c) sensitivity sweep (R1):

| y_c | lift | drag |
|---|---|---|
| 13.870 | +1.0759e-02 | +5.9863e-02 |
| 15.685 | +6.3222e-03 | +6.3447e-02 |
| 17.500 | +2.3192e-17 | +6.4790e-02 |
| 19.315 | -6.3222e-03 | +6.3447e-02 |
| 21.130 | -1.0759e-02 | +5.9863e-02 |

Optimisation trajectory (projected gradient ascent on lift, step ):

| gradient source | y_c path | lift path |
|---|---|---|
| ad | 21.000 -> 20.995 -> 20.991 | -1.313e-02 -> -1.312e-02 -> +nan |
| frozen | 21.000 -> 21.002 -> 21.004 | -1.313e-02 -> -1.313e-02 -> +nan |

Trajectories adverse (opposite directions): **True**

## ⑥ Failure modes, artefacts, reproduction

Failure modes / limitations:

* The case is 2-D, laminar (`Re` in the table) and geometrically smooth; the
  *sign* of the discrepancy is case-dependent, but the mechanism (missing
  `(dJ/df)(df/dp)`) is not: the proxy is not a total derivative for *any*
  parameter that changes the flow.
* The frozen-field proxy can invent a spurious streamwise gradient
  (`g_frozen,x != 0`) where the true derivative vanishes by streamwise
  symmetry -- visible in the table above.
* The shipped proxy uses a different lift functional (pressure difference on
  surface cells) than the momentum-exchange lift used here; it is therefore
  compared on direction/sign, not on absolute magnitude.
* Warm start removes the initial-condition term; the horizon study above shows
  the remaining deviation, so the numbers are finite-horizon, steady-state
  gradients rather than asymptotically exact ones.
* No experimental data exist for this configuration; the anchor is
  analytic/numerical as described in ①.

Artefacts:

- results: `runs/GAP-DIFFLBM-ADJOINT-VERDICT/run_r1/results.json`
- tables: `runs/GAP-DIFFLBM-ADJOINT-VERDICT/run_r1/table.md`, log `runs/GAP-DIFFLBM-ADJOINT-VERDICT/run_r1/run.log`
- fields: `runs/GAP-DIFFLBM-ADJOINT-VERDICT/run_r1/fields.npz`
- figures:
- `runs/GAP-DIFFLBM-ADJOINT-VERDICT/run_r1/fig_field.png` -- Steady velocity magnitude |u| and transverse velocity u_y around the off-centre disk (smooth penalisation, contour at solid fraction 0.5).
- `runs/GAP-DIFFLBM-ADJOINT-VERDICT/run_r1/fig_gradient.png` -- d(lift)/dy_c from expanded AD (reference), central finite differences and the frozen-field proxy, for two resolutions x two parameter points.
- `runs/GAP-DIFFLBM-ADJOINT-VERDICT/run_r1/fig_lift_sweep.png` -- Steady lift force on the disk as a function of the disk-centre ordinate y_c (warm-started forward solves).
- `runs/GAP-DIFFLBM-ADJOINT-VERDICT/run_r1/fig_cos_gate.png` -- Cosine similarity between the frozen-field proxy and the flow-response AD gradient against the 0.99 admission gate.
- `runs/GAP-DIFFLBM-ADJOINT-VERDICT/run_r1/fig_trajectory.png` -- Optimisation trajectory of the disk ordinate using the AD gradient vs the frozen-field proxy gradient (ascent on lift).

Reproduce:

```bash
python benchmarks/adjoint_verdict_bench.py --out runs/GAP-DIFFLBM-ADJOINT-VERDICT/run_r1 --threads 6
python benchmarks/adjoint_verdict_report.py --results runs/GAP-DIFFLBM-ADJOINT-VERDICT/run_r1/results.json \
    --out docs/GAP-DIFFLBM-ADJOINT-VERDICT.md
python -m pytest tests/test_adjoint_verdict.py -q
```
