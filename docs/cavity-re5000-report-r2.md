# CASE-CAVITY-RE5000 (TensorLBM) — round 2: reference admitted, scoring harness live

Rerun of the anchor `CASE-CAVITY-RE5000` (lid-driven square cavity, Re = 5000, tolerance
8 % of the lid speed vs Ghia 1982). Round 1 (`docs/cavity-re5000-report.md`) ended
`WITHHELD`: no machine-readable copy of the paper's tables and no converged solution.

## ① Case, physics and parameters

2-D lid-driven square cavity, `L = 1` in lattice units, lid speed `U_lid = 0.1`
(`Ma = U/cs = 0.173`, compressibility error `O(Ma^2) ~ 3 %`), `Re = U_lid L / nu = 5000`.

| quantity | value |
|---|---|
| lattice / collision | D2Q9 / regularised BGK (`collide_rlbm`, Latt & Chopard 2006) |
| boundary | V3: pre-streaming half-way bounce-back on 3 walls + Zou/He moving lid |
| tau (equal-Re recipe `3 U L / Re + 0.5`) | n=128 → 0.50768, n=160 → 0.50960, n=384 → 0.52300 |
| grids used here | 160 |
| reference | Ghia, Ghia & Shin, *J. Comput. Phys.* **48** (1982) 387-411, Table I (u, x/L=0.5) and Table II (v, y/L=0.5), Re = 5000 column, pp. 398-399 |
| tolerance (pre-registered) | pointwise `|ref - got| <= 8 % U_lid` on the 17 tabulated rows of each table |

## ② How it was done

1. **Reference admission (the blocking item of round 1) — now solved and auditable.**
   `benchmarks/verified/cavity/re5000/build_reference.py` takes three machine-read sources:
   * the paper's own OCR text layer (pp. 398-399), dumped mechanically with `pypdf`
     (`reference/ghia1982_tables_I_II.ocr.txt`, PDF sha256 `1d76e4636d24d0fa…`);
   * two independent digitised copies of the same tables (`copy_tableI_u_gist.txt`,
     `copy_tableII_v_pinn.txt`) plus a third low-Re MATLAB copy;
   and gates the output on:
   * **witness gate** — every one of the 210 tabulated values must appear in the paper's OCR text
     (207 exact digit matches,
     3 one-glyph OCR differences,
     0 untraceable);
   * **internal cross-check gate** — the Re = 100/400/1000 columns of the same copies against
     the library's own `GHIA_RE100/400/1000` tables (max |Δ| = 0 / 1.5e-1 / 5e-4, the two
     non-zero entries being documented errata);
   * **target-column gate** — the Re = 5000 column itself must contain no OCR ambiguity and no
     gross bracket violation.
   Output: `reference/ghia_re5000_reference.json`, sha256 `e43f617db17def03…`, 17 + 17 rows,
   carrying source URLs, the PDF sha256 and the full gate log. **Zero hand transcription.**
2. **Solver** — the same code path already validated at Re = 100/400/1000
   (`benchmarks/verified/cavity/run.py`), driven by `re5000/run.py` with the reference supplied
   through `--ref-json`, so a missing/again-unavailable reference can only ever yield
   `WITHHELD`, never a silent pass.
3. **Budget** — `run.py --n … --steps …`; every run writes `result.json` with the full centreline
   profiles so the numbers below are re-derivable without re-running.

## ③ Results (programmatically injected from the run artefacts)

| run | n | tau | steps | final resid | status | max abs err [%] | vs 8% |
|---|---|---|---|---|---|---|---|
| 160(re5000_n160.json) | 160 | 0.50960 | 10000 | n/a | diverged | n/a | n/a |



## ④ Figures

* `figures/cavity_re5000_centerline.png` — u(y) at x/L=0.5 and v(x) at y/L=0.5 against the
  Ghia symbols.
* `figures/cavity_re5000_error.png` — pointwise |error| per tabulated row with the 8 % tolerance
  line.

## ⑤ Notes, defects and scope

* **Errata found while admitting the reference** (all outside the scored Re = 5000 column,
  reported rather than silently repaired — see `provenance.errata` in the JSON):
  * Table I, Re = 3200, y = 0.4531: OCR reads `-0.86636` while the bracketing rows are
    `-0.04272 … -0.24427` — a lost decimal point; quarantined, column not scored.
  * Table I, Re = 10000, y = 0.5: OCR token `0.03 111` has lost the sign; magnitude exported,
    sign quarantined, column not scored.
  * Table II, Re = 400, x = 0.9063 — the digitised copy itself flags this cell as
    "probably wrong".
  * **Library defect (not fixed here, needs its own mission):** `GHIA_RE400` stores
    `v(0.1563) = 0.28003` where the paper says `0.28124` (the 0.28003 is the Re = 10000 value at
    x = 0.2266), swaps x = 0.2266/0.2344, and repeats `-0.38598` at x = 0.8047 and x = 0.9063.
    The cross-check in `build_reference.py` prints these discrepancies.
* **Scope**: Re = 5000 at these grids is a *stress* configuration for LBM (tau ≈ 0.51, wall
  boundary layers thinner than a lattice spacing); the numbers above are therefore a progress
  report on the harness, not a claim that the anchor is verified.
* **Cluster**: the 384² production run is submitted (`cj1412`) and still queued at the time of
  writing; its artefacts are **not** part of this report.

## ⑥ Error verdict

**< 3 %?** Cannot be answered: no run has written a scored comparison yet.

Honest status of the anchor after this round: reference admission **solved** (machine-verified,
reproducible), scoring harness **live and exercised on real numbers**, converged fine-grid
solution **still pending** — so `CASE-CAVITY-RE5000` stays **open / not verified**.
