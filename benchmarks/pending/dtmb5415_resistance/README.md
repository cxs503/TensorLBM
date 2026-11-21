# DTMB 5415 resistance

Status: **pending / honest-FAIL**. This is the canonical entry for the current
single-phase DTMB 5415 resistance study. The solver can reproduce CPU/SDAA
device parity, but the current geometry-valid run is still unsteady and its
friction result is outside the registered 3% in-frame band. The experimental
free-surface resistance frame remains structurally non-comparable because this
case does not model wave-making.

## Entry points

- `run.py` — maintained resistance runner and geometry/force-window gates.
- `test_dtmb5415_caliber.py` — static caliber and comparison-frame tests.
- `_geometry_probe.py` — engine-level voxel geometry probe.
- `make_diagnosis.py` — regenerates the machine diagnosis and published tables.
- `geometry/` — open historical STL, capped active STL and cap provenance.

## Evidence and report

- `evidence/sdaa_m452/diagnosis.json` — machine-readable m452 diagnosis.
- `evidence/sdaa_m452/records/` — eight CPU/SDAA run records and their index.
- `evidence/sdaa_m452/geometry_probe.json` — voxel geometry evidence.
- `evidence/sdaa_m452/figures/` — resistance, geometry and field figures.
- `../../../docs/sdaa_triage/DTMB5415_DIAGNOSIS.md` — published diagnosis.

Moving the evidence into this directory does not upgrade the benchmark status
and is not a new scientific replay.
