# Historical TensorLBM checkout

The [frozen snapshot](TensorLBM/) is the complete former root `TensorLBM/`
directory. Its 631 files, modes and layout match the original Git tree
`f482f80c56e861d7a5c038d1b6647394edaa1ba2`. The [migration ledger](../../docs/architecture/migrations/legacy-checkout-relocation-2026-11-21.json)
and [SHA-256 manifest](TensorLBM.sha256) preserve provenance.
Run `sha256sum -c TensorLBM.sha256` from this directory to verify the contents.

The [installable solver](../../src/tensorlbm/) supplies the supported library.
Historical replay entry points keep their original CLI paths and explicitly
load this frozen version when their old interfaces are required.

| Replay | Existing entry point |
| --- | --- |
| Collision matrix | [local](../../benchmarks/collision_operator_matrix/run_all.py), [cluster](../../benchmarks/collision_operator_matrix/run_cluster.py) |
| Model matrix | [run_matrix.py](../../xval/model_matrix_ci_plus_coverage/run_matrix.py) |
| Shan–Chen forcing | [run_xval.py](../../xval/XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT/run_xval.py) |
| Planar interface | [tl_arm.py](../../xval/XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE/tl_arm.py) |

Run campaign commands from the project checkout root. Some original scripts
inside the snapshot contain their historical workspace paths. To reproduce
those exactly, use a Git worktree at the [pre-move revision](https://github.com/cxs503/TensorLBM/tree/8f61b90530e635888fccb56a4cd655b15fde7863)
and the recorded runtime. Historical results retain their original scope.
Future P482-N05 runs write under `outputs/p482_n05/`.
