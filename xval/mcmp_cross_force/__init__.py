"""XVAL-LBM-MCMP-CROSS-FORCE-OPERATOR — cross-benchmark of the TensorLBM
two-component Shan-Chen (MCMP) cross-force operator against the live OSS
implementation (lbmpy 2.0, lssfau/lbmpy).

Anchor: XVAL-LBM-MCMP-CROSS-FORCE-OPERATOR-XVAL-SYMBOL-SP
Original mission: m18126 (round 1).  Rework: m18289 (maturity-gate L0 -> L2+).

Layout
------
operators.py   the two force operators under test (repo library kernel vs the
               OSS stencil transcribed from the lbmpy sdist source)
tl_arm.py      repo arm: periodic D2Q9 run driven by the library collision
               ``tensorlbm.multiphase.collide_sc_two_component``
oss_arm.py     OSS arm: live lbmpy 2.0 execution (pystencils codegen), sha256-pinned
compare.py     metrics (rel-L2 on the interface profile) + summary writer
run.py         entry point: ``python -m xval.mcmp_cross_force.run``
tests/         pytest suite (operator invariants + identity + smoke runs)

Provenance discipline: every OSS-side number is produced by executing the
lbmpy 2.0 sdist (sha256 c29aec635e2b29ef33ed716852d30e19fb027d2b704717e62ac0998b4120dca9);
no hand-copied reference values except the upstream paper profile REF_A, which
is the upstream test's own reference (lbm-principles-practice chapter9/shanchen.cpp).
"""

__version__ = "1.0.0"

ANCHOR = "XVAL-LBM-MCMP-CROSS-FORCE-OPERATOR-XVAL-SYMBOL-SP"
CRITERION_PCT = 15.0  # owner precision standard: >15 % vs reference is unacceptable
