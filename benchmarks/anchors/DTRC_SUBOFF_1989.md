# `dtrc-suboff-1989` — DTRC SUBOFF per-point anchor register (W30, task m21512)

> Why this file and not `README.md`: the W28 landing artifact
> (`papers/artifacts/w28/lbm-suboff-tau/anchor_landing.json`) pins the sha256 of
> `benchmarks/anchors/README.md` as a sidecar. Editing that README would rewrite
> a historical evidence record, so this register is documented here instead.

## Files

| anchor | builder | artifact | status |
|---|---|---|---|
| `dtrc-suboff-1989` | `dtrc_suboff_1989_anchor.py` | `dtrc-suboff-1989.json` | `TRANSCRIBED` / `acquisition_status=WITHHELD`; **all points grade C** |

The `dtrc-suboff-1989` register is the **per-point** census of the DTRC tow-tank
resistance data.  It parses *every* in-repo transcription family by regex at
build time (zero hand-copy):

| family | archive | what it is |
|---|---|---|
| `T14` | `docs/suboff_experimental_resistance_validation_2026-08-01.md` | single-row transcription of Table 14 (report p.23), 6 speeds 5.92…17.99 kn |
| `S1` | `benchmarks/pending/suboff_highre_resistance/refs_lock.md` (+ the machine lock `refs_lock.json`) | EFD column of Table 3 of a 2024 third-party paper that cites HD-1298-11 as [24], 5 speeds 3.050…8.230 m/s |

Each `(V, R_T)` pair is registered with its own verbatim quote, locator
(line / char offset / table / report page) and the sha256 of the quoted archive.
`C_F` is recomputed on two independent arithmetic paths (50-digit `Decimal` vs
`math.log10` float) and the residual against the `C_F` that *other* registers
already held is recorded per point.

## What the register found (material defect of the evidence chain)

The two families **disagree by a constant factor 1.170 ± 0.002 (~17%)** at the
same speeds, and that offset is independent of speed (a speed mismatch can only
contribute ≤0.3% through `R_T ∝ V^1.9`).  They cannot both be Table 14 AFF-1
bare-hull values as recorded.  17% is larger than the 15% drag gate, so **no
simulation-to-anchor claim better than 15% is adjudicable in this environment**
until the primary report (CRDKNSWC/HD-1298-11, Table 14, p.23, DTIC ADA359226)
is acquired.  Ranked hypotheses and the falsification test for each live in the
artifact under `cross_family_conflict.hypotheses_ranked`.

Grades stay **C** for every point: the sandbox is offline, so A (primary read)
and B (hashed source document) are *refused*, not merely unavailable — the
`S1` third-party PDF (`refs/cfdletters2024_suboff.pdf`) is not even present in
this checkout.

## Downstream census (LBM error table, grid/τ sensitivity, per-criterion verdicts)

| tool | artifact |
|---|---|
| `suboff_anchor_census.py` | `benchmarks/anchors/evidence/suboff_w30/suboff_anchor_census.json` |
| `suboff_anchor_census_figs.py` | `figures/w30_suboff_anchor_census/*.png` |
| `suboff_anchor_census_report.py` | `docs/W30_SUBOFF_ANCHOR_CENSUS.md` |
| `test_dtrc_suboff_anchor_census.py` | guard suite |

```bash
python benchmarks/anchors/dtrc_suboff_1989_anchor.py                        # rebuild register (+ --check)
python benchmarks/anchors/suboff_anchor_census.py                           # rebuild census  (+ --check)
python benchmarks/anchors/suboff_anchor_census_figs.py                      # figures
python benchmarks/anchors/suboff_anchor_census_report.py                    # report (numbers injected)
python -m pytest benchmarks/anchors/test_dtrc_suboff_anchor_census.py -q    # guards
```

The census keeps two LBM branches strictly apart: the verified laminar case
(`benchmarks/verified/suboff_sail_resistance/`, `Re_lattice = 1000`) is marked
`admissible_for_the_anchor = false` because it sits four decades below the
anchor's Reynolds number, and the only regime-matched branch
(`benchmarks/pending/suboff_highre_resistance/`, wall-model LES) is marked
admissible and fails the gate by +1296…+2963%.