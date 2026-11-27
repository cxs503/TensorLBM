# Historical three-way cavity comparison

This is an archive index, not a maintained run command or a new benchmark
acceptance. The independent formal [Re=100 case](../../verified/cavity/re100/README.md)
and [Ghia reference](../../verified/cavity/re5000/reference/ghia_re5000_reference.json)
retain their original documented paths.

The 14 original tracked files from `xval/XVAL-LBM-CAVITY/` are preserved byte-for-byte
under `development/verification/cavity_three_way/`. The original reports retain historical inline paths.
Original Git objects remain retrievable from the [migration ledger](../../../docs/architecture/migrations/cavity-three-way-archive-2026-11-27.json).

The original comparison records two reference passes at Re=100, W=64 but
an **inter-tool failure** (3.1248% against a 3% limit); three other comparison
specifications were not run. The four raw runs and three figures named in the
summary are not available as tracked objects in the inspected refs.
No figures, runs, or replacements were generated, and no historical solver
or serial MPI shim was imported or executed during migration.

All original tracked assets:

- [METHOD_SURVEY.md](../../../development/verification/cavity_three_way/METHOD_SURVEY.md)
- [REPORT.md](../../../development/verification/cavity_three_way/REPORT.md)
- [artifacts/comparison.json](../../../development/verification/cavity_three_way/artifacts/comparison.json)
- [artifacts/comparison.md](../../../development/verification/cavity_three_way/artifacts/comparison.md)
- [cases.json](../../../development/verification/cavity_three_way/cases.json)
- [cavity_spec.py](../../../development/verification/cavity_three_way/cavity_spec.py)
- [make_comparison.py](../../../development/verification/cavity_three_way/make_comparison.py)
- [mpi4py_shim/mpi4py/MPI.py](../../../development/verification/cavity_three_way/mpi4py_shim/mpi4py/MPI.py)
- [mpi4py_shim/mpi4py/__init__.py](../../../development/verification/cavity_three_way/mpi4py_shim/mpi4py/__init__.py)
- [oss_solutions.json](../../../development/verification/cavity_three_way/oss_solutions.json)
- [party_a_tensorlbm.py](../../../development/verification/cavity_three_way/party_a_tensorlbm.py)
- [party_b_pylbm.py](../../../development/verification/cavity_three_way/party_b_pylbm.py)
- [run_all.py](../../../development/verification/cavity_three_way/run_all.py)
- [run_shard.py](../../../development/verification/cavity_three_way/run_shard.py)

The original [audit evidence](../../../development/audit/m21174/verdict_evidence.json) remains unchanged.
Current scientific acceptance: **not evaluated**.
