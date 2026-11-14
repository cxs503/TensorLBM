# Delivered snapshot — provenance

Byte-identical copies of the m20070 (`GAP-DIFFLBM-BOUNDARY-GRAD`) deliverable, taken
from `/root/tp-agent/work/m20070` on 2026-11-14 for audit mission m20151.

Reason for the snapshot: the subject deliverable exists **only** as untracked files in
the subject work tree — `git cat-file -e origin/main:src/tensorlbm/diff_boundary_grad.py`
fails and no `agent/m20070` branch exists on the remote. The audit evidence pointers
therefore point at these copies; **the canonical location is `src/tensorlbm/…` in
TensorLBM and is still missing from `main`** (finding F6 in `../AUDIT.md`). This
directory is audit evidence, not the installation point.

| snapshot path | canonical path in the subject work tree |
|---|---|
| `src/tensorlbm/diff_boundary_grad.py` | `src/tensorlbm/diff_boundary_grad.py` |
| `tests/test_diff_boundary_grad.py` | `tests/test_diff_boundary_grad.py` |
| `scripts/run_gap_diff_boundary_grad.py` | `scripts/run_gap_diff_boundary_grad.py` |
| `scripts/report_gap_diff_boundary_grad.py` | `scripts/report_gap_diff_boundary_grad.py` |
| `scripts/plot_gap_diff_boundary_grad.py` | `scripts/plot_gap_diff_boundary_grad.py` |
| `artifacts/diff_boundary_grad_full.json` (+ `_quick.json`, 4 PNG) | `artifacts/GAP-DIFFLBM-BOUNDARY-GRAD/` |
| `GAP-DIFFLBM-BOUNDARY-GRAD.md` | `docs/GAP-DIFFLBM-BOUNDARY-GRAD.md` |

SHA-256 of every file as read from the subject work tree: `../evidence/sha256_source.txt`.
The auditor modified nothing; the copies are `cp` of the originals.