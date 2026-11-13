# Vendored OSS reference sources (anchor XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT)

* Upstream: `sailfish-team/sailfish`, commit `b7140c580e1f4cb406dde947c8d8d5a8a6b2d79e`
* Licence: LGPL3 (`sailfish/sym_equilibrium.py` header: `__license__ = 'LGPL3'`).
  The vendored files below are kept **for provenance/audit only**; TensorLBM does
  not link against them and does not copy their code. The transplant in
  `src/tensorlbm/phasefield/pressure_tensor_equilibrium.py` is re-derived from the
  algebraic form by `derive_oss_pressure_tensor.py` (sympy) and re-implemented on
  TensorLBM's own D2Q9 kernels.
* Files: `sailfish_src/sailfish/{lb_binary.py,sym_equilibrium.py,sym.py}`,
  `sailfish_src/sailfish/templates/finite_difference_optimized.mako`
* sha256 of each file is recorded in `provenance.json`; re-fetch with
  `python fetch_ref.py` (note: `raw.githubusercontent.com` was unreachable from
  this host, so the files were obtained with `git show` from a blobless clone —
  see the mission report's friction note).
