"""TensorLBM validation matrix v1: nine-case lineage, machine-checked gates.

Modules
-------
``criteria``          pure acceptance rules (error gate, convergence trend, status)
``reference_audit``   zero-hand-copy provenance gate for reference anchors
``runner``            CLI: list / audit / check / run / status
``cases/``            new CI-tier case runners (static droplet, dam break)
``matrix.json``       the registry (9 slots, thresholds, artifacts, repro commands)
``references.json``   anchors with in-repo source file + literal token + sha256
"""

from __future__ import annotations

__all__ = ["criteria", "reference_audit", "runner"]