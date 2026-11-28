# Historical offline figure helpers

These four former loose `scripts/` files are historical report/figure helpers,
not canonical solver entry points. Their original files and output receipts are
recorded in the [migration ledger](../../../docs/architecture/migrations/offline-figure-helper-handoff-2026-11-28.json).

| Helper | Historical context |
| --- | --- |
| [gen_verification_figures.py](gen_verification_figures.py) | [P96W2-N04 thermal verification report](../../../docs/reports/report_p96w2n04.md) — reference/precomputed-value illustrations |
| [generate_von_neumann_figures.py](generate_von_neumann_figures.py) | [Von Neumann stability report](../../../docs/von_neumann_report.md) |
| [generate_wetting_figures.py](generate_wetting_figures.py) | Existing wetting-dynamics formula figure renderer; returns encoded PNG bytes |
| [gen_yield_stress_figures.py](gen_yield_stress_figures.py) | Existing small Bingham Couette/Poiseuille figure runner |

The historical output paths are relative to the working directory. Run in a
separate scratch directory and set `PYTHONPATH` to the repository's `src/`;
create an `artifacts/` directory for the von Neumann helper. For wetting,
`generate_all()` returns the encoded figures rather than writing them.
Use actual installed NumPy, Matplotlib and PyTorch dependencies. The two
published report URLs remain stable. Four before/after executions produced
14 byte-identical PNG files in total; this is relocation compatibility only,
not a new scientific verdict or recovery of original historical figures.
