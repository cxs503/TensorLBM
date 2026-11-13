# Vendored upstream source (provenance)

Upstream: `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow`
(a public TensorFlow LBM library: `LatFlow/Domain.py`, `les_train.py`).

| file | sha256 | role here |
|---|---|---|
| `D2Q9.py` | `8acdfd9974d8fccf31912318f2e03c76fd60304182749052f8ba7335c558c0b7` | **executed as-is** behind a 3-line `tensorflow` shim (`oss_arm.load_upstream_tables`) — the WEIGHTS / LVELOC / BOUNCE / STREAM arrays used by the reference arm are these, not retyped |
| `Domain.py` | `d0c21f76df0e79ef511fc40581586b6a0a1bfff4dba1452fdbecd68525475522` | algorithm of record; every op of the port carries a `Domain.py:<line>` reference |
| `utils.py` | `bfeb6f59ff91c022907ebdf4db43f87db03cf6e82cadfa609c071d54973ecdaf` | `simple_conv` (VALID cross-correlation) / `pad_mobius` semantics |
| `les_train.py` | `435c6ebc192943e8093a9206ef0a176b13466a56bbb9574bf401fca5c40c9ab8` | shows the upstream training loop is **incomplete** (`lr`, `total_loss` are undefined) — the BPTT half of the recipe is described but not runnable upstream |

The files are unmodified copies kept for citation and for table execution. No
TensorFlow wheel exists for the evaluation host (`loongarch64`), so
`Domain.py` itself cannot be executed here — see `predeclared.json:not_run`
and `artifacts/results.json:env`.
