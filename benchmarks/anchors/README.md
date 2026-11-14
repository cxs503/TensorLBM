# `benchmarks/anchors/` — external anchor registers

Machine-readable **external quantitative anchors** used by the resistance /
ship-total-performance validation chain.  One file per anchor family; each file
is produced by a builder in this directory and is *append-only in spirit*: a
failing verdict stays in the file, it is never overwritten by a later wave.

## Files

| anchor | builder | artifact | status |
|---|---|---|---|
| `lbm-suboff-tau` | `lbm_suboff_tau_anchor.py` | `lbm-suboff-tau.json` | `TRANSCRIBED` / `acquisition_status=WITHHELD` |

## Gate contract (what a downstream anchor gate may rely on)

Any new anchor file in this directory must expose these top-level keys so that a
gate can read it without knowing the anchor family:

* `schema`, `status`, `acquisition_status`
* `conventions` (ρ, ν, unit factors, `C_T` / `C_F` definitions, reference area)
* `geometry` (with a validated `Cb` / `L` / `B` or equivalent gate)
* `anchor_points[*]` — each with `Re`, `C_T`, `C_F.value` **and** a `provenance`
  block (verbatim quote + locator + sha256)
* `documents[*]` — citation, role, `acquisition_status`, and the list of
  acquisition attempts when the primary document is not available offline
* `criteria` — per-criterion verdicts, including honest `WITHHELD` / `FAIL`

`status` values:

| value | meaning |
|---|---|
| `OBTAINED` | primary document stored in-repo with sha256 |
| `VERIFIED` | values re-read from the primary document and matched here |
| `TRANSCRIBED` | values are verbatim transcriptions of a table that is archived in-repo, primary document not re-read |

`acquisition_status` values: `OBTAINED` · `TRANSCRIBED` · `WITHHELD`.

`WITHHELD` is a *first-class* outcome, not a placeholder: it says the number was
not obtained in this environment, and it must come with the failed acquisition
attempts plus every operator-layer check that can still be run offline
(dimensional analysis, algebraic reconstruction, independent re-implementation
of any correlation line, cross-transcription agreement, sha256 cross-source
checks).

## Rules enforced by the builders

1. **Zero hand-copy.** Anchor numbers are parsed at build time from a frozen
   in-repo archive by regex; nothing is typed in from memory or from a
   secondary summary.
2. **Verbatim + locator + sha256.** A value without a `provenance` block is not
   evidence.
3. **Geometry before drag.** `Cb` / `L/D` / wetted area are validated first;
   a failure there invalidates the drag comparison regardless of how good the
   drag numbers look.
4. **No threshold tuning.** Thresholds live in the builder's pre-registration
   block and are not allowed to move after the numbers are seen.

## Load / verify

```bash
python benchmarks/anchors/lbm_suboff_tau_anchor.py           # rebuild + verify
python benchmarks/anchors/lbm_suboff_tau_anchor.py --check   # verify only
```

```python
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "anchor", "benchmarks/anchors/lbm_suboff_tau_anchor.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
anchor = mod.load_anchor(pathlib.Path("benchmarks/anchors/lbm-suboff-tau.json"))
```

Guard tests: `pytest -q tests/test_w27_lbm_suboff_tau_anchor_reconcile.py`.