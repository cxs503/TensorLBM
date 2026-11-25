# Historical D3Q19 method-selection study

The m19633 study compares D3Q19 collision operators for analytic ABC flow and
proposes a separate kernel comparison against pylbm. The [original audit](../../../docs/AUDIT_M19633_XVAL_LBM_ROUND2_METHOD_SELECTION_3D.md)
records which deliverables were missing when it was written. The four source
and protocol files now in this archive were present on main at the revision
recorded in the [migration ledger](../../../docs/architecture/migrations/method-selection-3d-archive-2026-11-26.json).

The matrix driver uses the analytics from the [Taylor–Green companion case](../../../benchmarks/case_taylor_green_3d/README.md).
With CPU PyTorch and NumPy installed, run the existing single-cell compatibility
check from the repository root.

```bash
OMP_NUM_THREADS=1 python development/verification/round2_method_selection_3d/run_matrix3d.py --smoke
```

Outputs are written to `outputs/historical_xval/round2_method_selection_3d/`.
The run covers BGK, N=16, tau=0.56, viscous-stress initialization and 200 steps.
The archived preregistration describes the study's broader scope. The original
matrix results and independent pylbm kernel evidence were unavailable in the
source tree. The kernel comparison needs the real pylbm-generated reference
JSON. Use the recorded upstream checkout and its actual dependencies to
generate it with `oss_pylbm_ref.py`.

The smoke run establishes that the relocated launcher can run the same cell.
The full matrix and independent comparison remain unevaluated.
