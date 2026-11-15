# Re-chaining the Ghia (1982) reference to its sha256-pinned OCR witness

Anchor: **`XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE`** (lanl/MF-LBM
`aa-pattern-fused-stream-collide`), acceptance `< 15 %` against a reference.

Tool: `benchmarks/xval/ghia_witness_rechain.py` ·
gates: `tests/test_ghia_witness_rechain.py` (10 tests) ·
artefacts: `artifacts/xval_aa_pattern/ghia_witness_rechain.json`,
`artifacts/xval_aa_pattern/fig9_ghia_witness_rechain.png`

---

## 1. Why this exists (the verification debt being closed)

Every number of this anchor — `L2(u) = 2.61 / 1.33 / 0.67 %` at `L = 32 / 64 /
128` and `L2(v) = 1.05 / 0.77 / 0.38 %` — is measured against the `BOUND` Ghia
Re = 100 table in `benchmarks/anchors/ghia1982_provenance.json`.  That table is
*claimed* to be a mechanical transcription of the primary witness
`benchmarks/verified/cavity/re5000/reference/ghia1982_tables_I_II.ocr.txt`
(sha256 `3637ef9e…`), but nothing in the repository re-derived the digits **from
the witness**:

| existing tool | what it actually consumes |
|---|---|
| `benchmarks/anchors/make_ghia_csv.py` | provenance JSON → CSV (never opens the witness) |
| `scripts/w26_ghia/extract_ghia_tables.py` | the **PDF** (PyMuPDF + raster); the PDF is copyrighted and not in the repo |
| `docs/xval_aa_pattern_ghia_table_defect.md` §5.1 | records the gap as "witness is not present in this worktree" |

**Finding (F1, defect, corrected here).** The witness *is* present:
`<repo>/benchmarks/verified/cavity/re5000/reference/ghia1982_tables_I_II.ocr.txt`,
6697 bytes, sha256 `3637ef9ef72a76bb090f4e31765ed2870d18db67f0f99611730e61927e060f5b`
— identical to the provenance pin.  The earlier note resolved
`primary_witness.file` (`../benchmarks/verified/...`) relative to the
`TensorLBM/` package directory only, where it indeed does not exist; relative to
the repository root it does.  The limitation was therefore **not** "evidence
unavailable" but "path looked up in the wrong root".  §5.1 of the older document
is amended accordingly.

## 2. Method

1. **M1 hash.** sha256 of every witness candidate path vs
   `primary_witness.sha256`; byte count recorded.
2. **M2 magnitude re-derivation.** The OCR text layer is a column-major dump of
   the tables, and it mangles the printed minus sign (`-0.20581 → .2058 1`,
   `-0.13641 → AI13641`) while the two-column typesetting interleaves cells of
   adjacent blocks.  A positional parse is impossible, so each provenance
   magnitude must appear as a *token* of the witness stream, in table order
   (ordered subsequence match with an explicit out-of-order fallback for the
   page-break reordering).  Skipped tokens are counted, never patched; a cell
   that cannot be found is **reported**, not repaired.
3. **M3 sign witnesses.** Three independently sourced transcriptions that ship
   next to the witness (`copy_tableI_u_gist.txt`, `copy_tableII_v_pinn.txt`,
   `copy_tablesI_II_matlab_lowRe.m`) must reproduce the provenance value
   numerically; explicit `-` tokens of the OCR text are recorded as partial
   witnesses.
4. **M4 invariants** on the Re = 100 table the acceptance actually uses
   (wall values exact, `v > 0` on `x < 1/2`, `v < 0` on `x > 1/2`, one `u` sign
   change, monotone `u` above mid-height, `|v|` maximum inside the domain).
5. **M5 detection power.** 200 Monte-Carlo injections of a one-unit corruption
   in the last printed (5th) decimal, re-run through M2.  A gate that cannot
   see a defect is not evidence, so the sensitivity is *measured*.
6. **M6 metric re-score.** The stored converged fields
   (`artifacts/xval_aa_pattern/cavity_accept_field_L32.npz`, `…L128.npz`) are
   scored again with the witness-derived table and with the exact centre-line
   interpolation `(1/2, 1/2)` for even `L`, importing **no** reference module
   (`reference_data` is never loaded), and the numbers are compared with the
   published artefacts.

## 3. Results (program output, `ghia_witness_rechain.json`)

### 3.1 Witness identity

| item | value |
|---|---|
| file | `benchmarks/verified/cavity/re5000/reference/ghia1982_tables_I_II.ocr.txt` |
| bytes | 6 697 |
| sha256 (measured) | `3637ef9ef72a76bb090f4e31765ed2870d18db67f0f99611730e61927e060f5b` |
| sha256 (pinned) | same |
| match | **yes** |

### 3.2 Magnitude re-derivation (136 provenance cells)

| table | cells | re-derived from the OCR stream | residual (text-layer corrupted) | tokens skipped |
|---|---|---|---|---|
| I (`u`) | 68 | **67** (98.5 %) | `I/Re1000/u[11]` (OCR prints `4.38289` for `-0.38289`) | 48 |
| II (`v`) | 68 | **66** (97.1 %) | `II/Re400/v[8]`, `II/Re1000/v[5]` | 29 |
| Re = 100 only | 34 | **34 (100 %)** | — | — |

The three residuals are cells the *text layer itself* corrupts (the minus sign
is OCR'd as `4`/`AI`/`Al`, or the cell is displaced by the page break); each is
reproduced by at least one independent transcription and none is in the Re = 100
table that the acceptance scores against.

### 3.3 Sign / value corroboration (Re = 100/400/1000)

| item | value |
|---|---|
| cells audited | 102 |
| cells with ≥ 2 independent witnesses | 102 |
| sign disagreements with the provenance | **0** |
| cells whose value is reproduced by ≥ 2 witnesses | 101 / 102 |

**Finding (F2, provenance defect, reported not patched).**
`II/Re1000/v[5]` carries `-0.5155` in the provenance JSON, while the OCR witness
prints `-0.51500` and both the MATLAB copy and the PINN copy carry `-0.51500`
(the gist copy prints `-0.51500` as well).  The cell is **off the anchor's
scope** (the acceptance uses Re = 100 only) and the table is a pinned anchor
artefact consumed by other gates, so it is *flagged*, not silently edited.  Any
Re = 1000 gate that consumes this table inherits a 1e-4-level error in that cell.

### 3.4 Invariant gates on the Re = 100 table — all pass

`u(y=1)=1` (0.0), `u(y=0)=0` (0.0), `v(x=0)=0` (0.0), `v(x=1)=0` (0.0),
`v>0` on `0<x<1/2`, `v<0` on `1/2<x<1`, single `u` sign change, `u` monotone
above mid-height, `|v|` maximum inside the domain.

### 3.5 Detection power (measured, not assumed)

| corruption | trials | caught | rate |
|---|---|---|---|
| one unit in the 5th decimal of one random provenance cell | 200 | 144 | **72.0 %** |

So the re-chain is a *corroboration layer*, not a sole gate: roughly one
in four last-digit defects survives the text-layer match (it collides with
another cell's token).  It is combined with the independent transcriptions and
the invariants, which is why the layered check — not M2 alone — is the gate.

### 3.6 Acceptance re-score with the witness-derived table (no reference module)

| grid (stored field) | `L2(u)` | `L2(v)` | `< 3 %` | `< 15 %` |
|---|---|---|---|---|
| `L = 32` | 2.61 % | 1.05 % | **yes** | **yes** |
| `L = 128` | 0.67 % | 0.38 % | **yes** | **yes** |

These reproduce the published corrected numbers
(`cavity_accept_corrected.json`) to the printed digit, computed here from the
witness-derived table only — i.e. the anchor's error budget does not depend on
any hand-typed constant.

## 4. Error judgement (explicit)

| key quantity | reference | error | `< 3 %`? |
|---|---|---|---|
| provenance table vs sha256-pinned OCR witness (Re = 100, 34 cells) | OCR text layer | 0 cells unreproduced | **yes** |
| cavity `L2(u)`, L = 128 | Ghia 1982 (witness-derived) | 0.67 % | **yes** |
| cavity `L2(v)`, L = 128 | Ghia 1982 (witness-derived) | 0.38 % | **yes** |
| cavity `L2(u)` / `L2(v)`, L = 32 (not steady) | Ghia 1982 | 2.61 % / 1.05 % | **yes** |
| `II/Re1000/v[5]` provenance value | OCR witness + 3 copies | 5e-4 absolute (−0.5155 vs −0.51500) | **no** (off-anchor cell, flagged) |

## 5. Notes, limitations, scope

1. **The OCR text layer cannot fix signs.** Roughly a fifth of the surveyed cells
   are rendered with a mangled minus (`AI`, `Al`, `a.`, `4`).  Magnitudes are
   witness-backed; signs rest on three independent transcriptions plus the
   physical invariants.  A future mission with the PDF in hand should re-hash it
   (`pdf_sha256` is pinned in the provenance) and rerun
   `scripts/w26_ghia/extract_ghia_tables.py` to obtain pixel-level sign evidence.
2. **Two provenance files exist** (`benchmarks/anchors/ghia1982_provenance.json`
   and `TensorLBM/benchmarks/anchors/ghia1982_provenance.json`).  Their
   Re = 100/400/1000 *values* are identical (checked), but the metadata differs:
   the package copy lacks `csv_anchors` and one `suspect_entries` record.  This
   script reads the root copy (the one that carries the CSV pins) and records
   which file it used.  Finding F3 — metadata drift, no numerical consequence
   today.
3. **Scope**: Re = 100/400/1000 of Ghia (1982) Tables I–II as transcribed in this
   repository; nothing here validates the 3200/5000/7500/100 000 columns.
4. **Detection power 72 %** is a property of the *witness*, not of the solver; it
   is quoted so that nobody mistakes the re-chain for a bit-exact check.
5. The re-score in §3.6 uses the stored `L = 32` field, which is **not steady**
   (9 000 steps); it is quoted because it is the grid on which the historical
   kernel failed at `L2(u) ≈ 0.25`.

## 6. Reproduce

```
cd TensorLBM
python benchmarks/xval/ghia_witness_rechain.py --check     # exit 0 == all gates pass
python tests/gen_aa_fused_rechain_figures.py               # fig9
pytest tests/test_ghia_witness_rechain.py -q                # 10 passed
```