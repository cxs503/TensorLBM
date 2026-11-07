"""TensorLBM SUBOFF resistance tournament (IN25-T3, mission m5216).

Three-generation, 100-seat tournament attacking the SUBOFF bare-hull
resistance coefficient against physical numerical anchors (Blasius @ Re=1000
laminar; ITTC-1957 @ Re>=1e5 turbulent).  Scoring is a physical numerical
anchor, not a model review.

Modules
-------
anchors.py    physical anchors + red-team self-tests (known-solution cases)
scorer.py     version-locked scorer (score / err_pct / pass)
engine.py     lean D3Q19 LBM (numpy, CPU): BGK/MRT, Smagorinsky/WALE,
              half-way BB / log-law wall, MEM force with sampling phase
seats.py      3-generation seat registry (40/40/20), 5 attack families
runner.py     single-seat CLI (cluster shard entry point)
orchestrator.py  generation loop: dispatch, collect, score tables, winners
report.py     programmatic score tables + figures + report JSON
"""

SCHEMA = "tensorlbm.tournament-suboff/v1"
SCORER_VERSION = "suboff-anchor-scorer/1.0.0"
