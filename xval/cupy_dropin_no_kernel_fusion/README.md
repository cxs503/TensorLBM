# XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION

Cross-validation of TensorLBM against `ebinan92/Fingering_dynamics` ("cupy drop-in,
no kernel fusion" practice): one SOY free-energy D2Q9 implementation, driven through a
NumPy backend and a PyTorch backend, benchmarked against the executed OSS code and the
analytic Young–Laplace law.

| Path | What |
|---|---|
| `../../src/tensorlbm/multiphase_soy_dropin.py` | implementation (pluggable array backend, no fused kernels) |
| `../../src/tensorlbm/multiphase_static_droplet.py` | solver-layer caller `soy_free_energy_droplet()` |
| `oss_adapter.py` | executes the OSS `class Compute` + `stream` read-only (no vendoring, `OSS_FINGERING_ROOT`) |
| `run_xval.py` | entry point -> `artifacts/run_main.json`, `artifacts/dp_history.csv`, `figures/*.png` |
| `predeclared.json` | thresholds and design, fixed before the runs |
| `EVIDENCE.json` | machine-checkable evidence bundle (`emit_evidence.py`) |
| `make_report.py` | renders `docs/XVAL_LBM_CUPY_DROPIN_NO_KERNEL_FUSION.md` from the artefact |

```bash
python xval/cupy_dropin_no_kernel_fusion/run_xval.py            # ~4-5 min CPU, float64
python xval/cupy_dropin_no_kernel_fusion/emit_evidence.py
python xval/cupy_dropin_no_kernel_fusion/make_report.py
python -m pytest tests/test_free_energy_soy_cupy_dropin.py -q   # 14 passed
```

Measured (T=2000, R=15, sigma=0.01, xi=3): Laplace jump error −0.40 % (TensorLBM either
backend) and −0.41 % (executed OSS) vs sigma/R; NumPy↔PyTorch agreement 8e-14 relative;
TensorLBM vs OSS 0.017 %. Deliberate negative control: a non-equilibrium initial interface
width gives +54.6 % after the same 2000 steps.
