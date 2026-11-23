# YAML-to-XML Poiseuille verification

Run from a checkout with TensorLBM's Python dependencies installed.

```bash
python benchmarks/xval_yml_xml_pipeline/run_all.py --out outputs/xval_yml_xml_pipeline
```

The driver defines four input cases with two channel heights, two forcing
amplitudes and two relaxation rates. Each case round-trips its YAML-derived
specification through XML, executes the D2Q9 solver, and compares the
permeability and velocity profile with the analytic plane-Poiseuille solution.
Both relative errors must be below 15% in every case, and XML round-trip and
numeric log checks must pass. See the [reference and original report](../../docs/xval/XVAL-LBM-YML-DRIVEN-XML-TEMPLATE-PIPELINE.md)
for the validation scope, formulas and limitations.

The [2026-11-23 replay](replays/2026-11-23.json) records the run on both sides
of the path migration, including the exact model inputs, source identities,
software versions, measured outputs and applicable gates. The validation
covers single-phase, two-dimensional Poiseuille flow and the input pipeline.
