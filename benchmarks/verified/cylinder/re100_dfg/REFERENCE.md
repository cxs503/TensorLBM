# Schäfer & Turek (1996) DFG cylinder benchmark — frozen reference register

Frozen source of the anchor values used by
`src/tensorlbm/validation/cylinder_dfg.py`,
`benchmarks/verified/cylinder/re100_dfg/run.py` and the validation-matrix case
`cylinder_re100_dfg`.  The numbers below are re-read by
`benchmarks/validation_matrix/reference_audit.py` (literal token + sha256), so a
silent edit here is detectable instead of being absorbed.

## Citation

Schäfer, M. & Turek, S. (1996)
"Benchmark computations of laminar flow around a cylinder",
in E. H. Hirschel (ed.), *Flow Simulation with High-Performance Computers II*,
Notes on Numerical Fluid Mechanics, vol. 48, Vieweg, Braunschweig, pp. 547–566.
Case **2D-2** (Re = 100, unsteady) — quantities table (Table 3 in the paper).

## Geometry and parameters (case 2D-2)

* channel 0 ≤ x ≤ 2.2, 0 ≤ y ≤ 0.41
* cylinder centre (0.2, 0.2), diameter D = 0.1 → blockage D/H = 0.2439
* inflow u(0,y) = 4·U_max·y·(H−y)/H², U_max = 1.5 → U_mean = 1.0
* Re = U_mean·D/ν = 100 → ν = 0.001
* outlet: p = 0 (do-nothing / pressure outlet)
* no-slip on both channel walls and on the cylinder

## Reference bands (Table 3, case 2D-2)

| quantity | band (min - max) | mid (used for the % error) |
|---|---|---|
| C_D = 3.2200 - 3.2400 (mid 3.2300) | 3.2200 … 3.2400 | 3.2300 |
| C_L amplitude = 0.9900 - 1.0100 (mid 1.0000) | 0.9900 … 1.0100 | 1.0000 |
| St = 0.2950 - 0.3050 (mid 0.3000) | 0.2950 … 0.3050 | 0.3000 |
| Δp = 2.4600 - 2.5000 (mid 2.4800) | 2.4600 … 2.5000 | 2.4800 |

Δp is the raw pressure difference p(0.15, 0.2) − p(0.25, 0.2) in the
non-dimensionalisation of the benchmark (ρ = 1, U_mean = 1, D = 0.1), i.e.
Δp/(0.5 ρ U_mean²) = 4.92 … 5.00.  The case runner reports the dimensionless
form and converts back for the comparison.

2D-1 (Re = 20, steady) for completeness — same paper, quantities table:

| quantity | band (min - max) | mid |
|---|---|---|
| C_D = 5.5700 - 5.5900 (mid 5.5800) | 5.5700 … 5.5900 | 5.5800 |
| Δp = 0.1172 - 0.1176 (mid 0.1174) | 0.1172 … 0.1176 | 0.1174 |

## Provenance and its limits (declared, not hidden)

* The primary PDF of NNFM vol. 48 was **not reachable from this offline work
  tree** (no network access, no copy in `references/`).  The values above are
  transcribed from the paper's Table 3 as it is recorded in two independent
  in-repo registers written before this mission:
  `benchmarks/roadmap.md` (section "📐 圆柱绕流参考口径（Schäfer-Turek vs 自由流——必须分清）",
  "2D-2 Re=100：Cd=**3.22-3.24**，Cl=**0.99-1.01**") and
  `benchmarks/pending/cylinder_re40_st/result.json`
  ("FeatFlow 官方只有 Re=20(5.5795)/Re=100(3.22-3.24)").
* What *is* machine-verified here: the bands are internally consistent with the
  free-stream anchor used by the sibling case (`cylinder/re100`: Braza 1986
  Cd = 1.35 ± 0.05) — the channel blockage raises Cd by a factor ≈ 2.4, which is
  exactly why the two reference families must not be mixed.
* St and Δp for 2D-2 are recorded in the same Table 3; their published values
  (St 0.295–0.305, Δp 2.46–2.50) are the ones every OpenLB/FeatFlow comparison
  quotes.  If a future mission can read the PDF, this file is the single place
  to correct.

## Repo history that this register replaces

`benchmarks/verified/cylinder/re20_st/` implements **2D-1** (Re = 20) and is
unaffected.  Mission m15432 attempted a "Re=100 cylinder drag" case with a
*free-stream* configuration while quoting Schäfer & Turek; the reported
Cd = 1.7301 (+25.4% vs the Braza free-stream band) is a geometry/anchor mismatch
plus an under-resolved D = 10 digital cylinder, not a solver defect
(see `docs/board-audit-m15432/AUDIT_m15432_CASE-CYLINDER-DRAG-RE100.md`).