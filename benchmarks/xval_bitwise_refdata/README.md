# XVAL-LBM-BITWISE-REFDATA-REGRESSION-PLUS-BENCHMAR

Bit-exact reference-data regression for the D2Q9 primitives, plus an analytic
Taylor–Green benchmark that anchors the lock to physics rather than only to
itself.

| file | role |
|---|---|
| `harness.py` | library: case registry, byte digest, ulp metric, verify gate, negative controls, TGV benchmark |
| `run.py` | entry point: `gen` / `verify` / `controls` / `bench` / `check` |
| `refdata/d2q9_bitwise_v1.json` | frozen golden data: 9 cases, sha256 digest **and** full float64/float32 payload (base64) so a 1-ulp difference can be localised, not just reported |
| `result.json` | last real run: regression verdicts + negative controls + benchmark ref-vs-got + repro_cmd |
| `probe_weight_bias.py` | root-cause probe for the `collide_bgk_fused` float32-weight leak found by this lock |
| `figures/tgv_bitwise_bench.png` | LBM vs analytic decay + per-point error |

## Run it

```bash
PYTHONPATH=src python3 benchmarks/xval_bitwise_refdata/run.py check   # all gates -> result.json
PYTHONPATH=src python3 benchmarks/xval_bitwise_refdata/run.py verify  # bitwise regression only
python -m pytest tests/test_xval_bitwise_refdata.py -q                # same gates from the test suite
```

Exit code is 0 only if every gate passes. Runtime is a few seconds
(`torch.set_num_threads(1)` — for 16–64² lattices the thread pool costs ~20x
more than it saves).

## Why the comparison can be bitwise

* Initial fields come from **integer lattice algebra** with powers-of-two
  divisions (`harness.int_fields`), so the initial condition is exactly
  representable and identical on any IEEE-754 platform — no RNG, no `libm`
  `sin`/`cos` in the regression cases. A digest mismatch is therefore the
  *kernels*' doing.
* The digest is sha256 over the raw IEEE-754 payload plus dtype/shape.
* Golden data records the environment (torch, CPU arch) and the sha256 of
  `src/tensorlbm/{d2q9,solver}.py`. Bitwise equality is *required* only when
  that environment matches; otherwise the gate degrades honestly to an ulp
  budget (`ULP_DRIFT`) instead of raising a false red.
* `verify(..., runner=...)` is injectable so a simulated kernel drift can be
  tested; `verify` also refuses a golden file whose payload no longer matches
  its own digest (`GOLDEN_INCONSISTENT`).

## Verdict vocabulary

| verdict | meaning |
|---|---|
| `BITWISE_OK` | every byte identical to the golden file |
| `ULP_DRIFT` | environment differs from golden (or kernel source sha changed) and drift ≤ ulp budget → reported, not red |
| `REGRESSION` | same environment and any byte differs, or drift > budget |
| `GOLDEN_INCONSISTENT` | golden payload does not hash to the stored digest (hand-edited file) |
| `MISSING_GOLDEN` | case exists in code, absent from the golden file |

## Re-freezing (deliberate change)

Any intentional change to a covered kernel invalidates the golden file by
design. Re-freeze in the same commit and say why:

```bash
PYTHONPATH=src python3 benchmarks/xval_bitwise_refdata/run.py gen
PYTHONPATH=src python3 benchmarks/xval_bitwise_refdata/run.py check
```

`result.json["regression"]["code"]` must match the kernel sources in the tree —
`tests/test_xval_bitwise_refdata.py` enforces this.

## Known limitations

* Bitwise identity is a *same-build* property. On another torch build, CPU
  architecture or thread configuration the digests may drift by a few ulp; the
  gate then reports `ULP_DRIFT` and no longer proves byte identity.
* `collide_bgk_matmul` is intentionally **not** covered: it routes through
  BLAS matmul, whose summation order is not part of torch's determinism
  contract, so a bitwise lock there would be a lie.
* The physics benchmark uses the analytic TGV decay (no digitised external
  table), so it validates the operator against an exact solution but does not
  substitute for a wall-bounded / high-Re external benchmark.