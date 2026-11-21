# Pressure-tensor droplet

Run from the repository root with the package dependencies installed:

```sh
python benchmarks/case_pool/pressure_tensor_droplet/run.py --radius 16 --steps 2000
```

The runner measures the mechanical pressure difference and uses a 15% error
criterion against the analytic Laplace pressure. See the
[cross-validation protocol](../../../xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/predeclared.json)
and [discrete pressure-tensor reference](../../../xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/ref/oss_discrete_pressure_tensor.json).
The layout migration did not evaluate the criterion.
