# Board audit — m17865 XVAL-LBM-ISO8-WALL-NORMAL-TABLE (audit m17969)

**Object**: mission 17865, repo TensorLBM, status `succeeded`, delivery commit `a43fd917`
(branch `agent/m17865`). **This is a hand-over verification, not a re-run of the mission.**

* audit verdict: **L2 — CONFIRMED** (L3/L4/L5 not granted; reasons in §5)
* independent recompute: `docs/board_audit/m17969/audit_m17865.py` →
  `artifacts/m17865_audit/audit_m17865.json`; figures `figures/audit_m17865.png`
* benchmark judge (`tensor_benchmark_cases`): **0 rows** for this anchor
  (76 rows in the table, none ISO8/wall-normal) → the machine authority has no case here

## ① What the delivery claims (physics/algorithm background)

The anchor transplants `lanl/MF-LBM`'s *iso8 wall-normal table* preprocessor
(`preprocessing/3.wall_boundary_preprocess/3d_wall_boundary_process.f90`, upstream commit
`11be324a97fe…`): the solid indicator is smoothed by 4 D3Q27 convolutions
(w=(8/27,2/27,1/54,1/216), kernel sum 1 → mass conserving), then an ISO8 isotropic gradient
(7 weights over 34 central pairs, 5×5×5 reach) produces a unit wall normal on every fluid node
touching the solid. Curvature-dependent BCs (contact angle, Laplace pressure, second-order slip)
need those normals; on a staircase the arithmetic axis average is O(1) wrong at 45°.

## ② How the audit was done (independent of the delivery code)

1. Own regex parse of the vendored Fortran (not the delivery extractor): 7 ISO8 weights,
   34 pairs per component, D3Q27 kernel, 27 lattice vectors.
2. Own parser for the delivered **binary** upstream tables
   (`oss_runs/*/output/indirect_addressing_walls_boundary.dat`, packed 96 B/36 B records), and
   own analytic reference (sphere radial / plane normal) → mean/max chordal error recomputed
   without importing `tensorlbm.boundary.wall_normal_iso8`.
3. `python3 scripts/xval_iso8_wall_normal_table.py run --force-build` (gfortran 12.3, 38.7 s)
   executed end-to-end; delivered vs re-run artifacts compared field by field.
4. `python3 -m pytest tests/test_wall_normal_iso8.py` re-run.
5. Judge query against `tensor_benchmark_cases` (owner role), plus caller/integration greps on
   the delivery tree and on the main clone `/root/tensor-ecosystem/TensorLBM` (HEAD `be78a87b`).
6. The audited tree was kept read-only: the re-run wrote into the worktree, the delivered
   artifacts were then restored byte-for-byte from a pre-run snapshot
   (`artifacts/m17865_audit/delivered_snapshot/`); `git status` is back to the mission's own
   single `?? preprocessing.mod`.

## ③ Results (auditor's numbers, not the report's)

Stencil re-parsed from upstream: ISO8 weights `0.0888889, 0.0476190, 0.0190476, 0.00992063,
0.00317460, 0.00158730, 0.000198413`; 34 pairs per component; D3Q27 kernel sums to **1.000000**
(shells 1/6/12/8); operator gain on linear fields **0.880952380952** in x, y and z
(anisotropy **1.000000**) → the delivery's K3 claim reproduces exactly.

| case | K1 max abs Δn | ≤1e-12 | K2 mean rel % | ≤15 % | K2 max rel % | ≤15 % |
|---|---|---|---|---|---|---|
| sphere_n48_R10_offcenter | 8.88e-16 | ✓ | 3.43 | ✓ | 11.73 | ✓ |
| sphere_n64_R14_offcenter | 8.88e-16 | ✓ | 3.29 | ✓ | 13.26 | ✓ |
| plane_n40_alpha26.565 | 6.66e-16 | ✓ | 2.47 | ✓ | 32.83 | ✗ |
| plane_n40_alpha45 | 7.77e-16 | ✓ | 3.33 | ✓ | 43.64 | ✗ |

* Node counts and per-node error statistics recomputed from the delivered OSS binaries match
  the delivery **digit for digit** (1815/3450/3216/3024 listed nodes; 1815/3450/2360/2200
  physical nodes; means 3.431 / 3.294 / 2.470 / 3.330 %).
* Re-run: every non-timing number of `results.json` identical (0 diffs); `accuracy_table.csv`,
  `sensitivity_table.csv`, `oss_fixture_normal_table.csv`, `oss_iso8_stencil.json` byte-identical;
  `exactness_table.csv` differs only in its two timing columns.
* `pytest tests/test_wall_normal_iso8.py` → **6 passed** (6.9 s, re-run by the auditor).
* K4 curvature (|κ| vs 2/R, the delivery's own sign-normalised definition):

| case | κ_ref = 2/R | |κ_ISO8| | |err| |
|---|---|---|---|
| sphere_n48_R10_offcenter | 0.2000 | 0.1652 | 17.4 % |
| sphere_n64_R14_offcenter | 0.1429 | 0.1190 | 16.7 % |

## ④ Figures

`figures/audit_m17865.png` — (a) K2 mean/max vs the 15 % band, (b) K1 vs the 1e-12 criterion,
(c) smoothing sweep (spheres improve 12.6→1.1 %, the 45° plane degrades 2.1→5.1 %),
(d) K4 |κ| vs analytic 2/R.

## ⑤ Findings (severity-ranked) and the level decision

* **F1 (blocking for L3) no integration caller.** `grep -rn wall_normal_iso8` in the delivery
  tree: 6 hits, all inside the module, its test and the XVAL harness → 0 solver callers. The
  module is absent from the main clone (`be78a87b`), i.e. commit `a43fd917` is still only on
  `agent/m17865`. Delivery's own D11 admits "not wired". ⇒ L3 not granted.
* **F2 (blocking for L4) benchmark authority empty + predeclared max/K4 criteria fail.**
  `tensor_benchmark_cases` has **0 rows** for this anchor, so no registered tolerance can be
  met; independently, of the delivery's own predeclared criteria K1 4/4 ✓, K2-mean 4/4 ✓,
  **K2-max 2/4 ✗** (26.565° 32.83 %, 45° 43.64 %) and **K4 ✗** (16.7–17.4 % > 15 %). The report
  reports these honestly ("honest FAIL", no trimming) but they are FAIL criteria. ⇒ L4 not granted.
* **F3 (medium, provenance) declared source sha256 is not the file digest.** The doc and
  `results.json` pin `d5ff4f9e…`, which is `sha256(file.read_text().encode())` — the CRLF→LF
  normalised text (the vendored file has 647 CRLF lines). `sha256sum` of the actual bytes is
  `01c2b27e…`. A `sha256sum`-based third-party check therefore disagrees with the doc, and the
  pin is blind to line-ending edits. Verification of the constants is unaffected (they are
  re-parsed from the file by the tests).
* **F4 (low) OSS arm provenance is a vendored copy, not a fetch under a pinned hash.** The
  upstream commit is declared and the file is vendored + re-parsed, but there is no
  network/archival proof that the vendored text equals upstream `11be324a…`; L5 would need that.
* **F5 (info) the K1 claim cannot be re-derived from shipped artifacts alone.** The fixture CSV
  is written *from the OSS arm*, so comparing fixture vs OSS binary is a round-trip (0.0 Δn),
  not an arm-independence proof; K1 is credible because `run_oss` really compiles and runs the
  Fortran (parsed binary, file-length checked) and because the auditor's end-to-end re-run
  reproduces max|Δn| = 8.88e-16. The module-level test against the pinned fixture (<1e-12) passes.
* **F6 (info) gain convention.** The auditor's half-offset gain is 0.440476; the delivery's
  0.880952 is the raw-difference convention (×2). Direction independence, the property that
  matters, holds in both. Not a defect, but the report should state the convention.
* **F7 (info) report quality is above the bar**: six sections present, difference list D1–D12,
  repro command in the text, 4 real PNG figures (34–134 kB), failure modes include the
  author's own C-order/Fortran-order voxel bug — i.e. the voluminous-report trap is not present
  here; the level limitation is functional (no caller, no benchmark row), not documentary.

**Verdict: L2.** L2 = independent functional implementation: input (voxel solid mask) → compute
(smoothing + ISO8 gradient + node classification) → output (per-node normal table, neighbour
list, la_weight) in real code (`src/tensorlbm/boundary/wall_normal_iso8.py`, 465 lines, no
placeholder), with a genuine test suite, real run artifacts, and an executed external anchor
(the compiled upstream preprocessor) reproducing to 8.9e-16. L3/L4/L5 fail as in F1/F2/F4.
Consistent with the board's precedent for the sibling transplant anchors
(m17677 COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE, m17757 FILL-GRADIENT-NORMAL,
m17502 BITFLAG-NEIGHBOUR-MASK — all confirmed at L2).

**Reopen / remediation (for a follow-up mission, not this audit):** (1) wire the table into a
3-D wetting/droplet BC so a caller exists; (2) trim the per-node corners or predeclare the max
criterion as a mean-only criterion, and fix the K4 estimator (project to the level set before
differentiating); (3) register a `tensor_benchmark_cases` row for this anchor; (4) publish the
byte sha256 next to the normalised one.

## ⑥ Error judgement vs the benchmark

| quantity | criterion | measured | verdict |
|---|---|---|---|
| K1 max abs Δn (4 cases) | ≤ 1e-12 (predeclared) | 6.7e-16 … 8.9e-16 | ✓ PASS (4/4) |
| K2 mean chordal error | ≤ 15 % (owner) | 2.47 … 3.43 % | ✓ PASS (4/4) |
| K2 max chordal error | ≤ 15 % (predeclared) | 11.73 / 13.26 / 32.83 / 43.64 % | ✗ 2/4 FAIL |
| K3 operator anisotropy | ≤ 1.000001 | 1.000000 | ✓ PASS |
| K4 |κ| vs 2/R | ≤ 15 % | 17.4 % / 16.7 % | ✗ FAIL |

Is the key quantity < 3 %? **K1 yes** (float64 round-off floor). **K2 mean no** — 2.47–3.43 %,
inside the owner 15 % band but above 3 %; per-node maxima up to 43.6 % are a staircase-corner
property shared by every gradient stencil (the central-difference arm is worse: 3.21–4.31 %
mean and degenerate nodes up to 1314). No physical acceptance (droplet/contact angle) is claimed
or possible from a preprocessing capability — that is the delivery's own stated limitation.
