# TensorLBM

TensorLBM is a CPU-first PyTorch D2Q9 Lattice Boltzmann Method platform focused on **reproducible research experiments** with clear extension points.

## What TensorLBM is today

Current `main` provides:

- A small, explicit public API in `src/tensorlbm/__init__.py`
- Core D2Q9 primitives (`equilibrium`, `macroscopic`, lattice constants)
- Modular solver steps (`collide_bgk`, `stream`)
- Boundary helpers (`cylinder_mask`, `make_channel_wall_mask`, bounce-back channel boundaries)
- A first-class cylinder-flow runner with CLI, structured outputs, metadata, and diagnostics
- Automated tests and CI

## Engineering principles used in this step

These principles guide TensorLBM toward a world-class platform:

1. **Stable API first**: keep public exports intentional and backward-compatible.
2. **Composable solver core**: isolate lattice math, solver stepping, and boundary logic.
3. **Reproducible runs**: parameterized CLI + deterministic run folder layout + metadata snapshot.
4. **Fast feedback loops**: smoke tests and CI on push/PR.
5. **CPU-first defaults, GPU-ready shape**: default to CPU, but keep interfaces ready for device scaling.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

## Run the cylinder-flow example

Default run:

```bash
PYTHONPATH=src python examples/cylinder_flow.py
```

Small smoke run:

```bash
PYTHONPATH=src python examples/cylinder_flow.py \
  --nx 64 --ny 24 --radius 4 --n-steps 20 --output-interval 10 \
  --run-name smoke --overwrite
```

Useful options:

- `--nx`, `--ny`: grid size
- `--u-in`, `--re`, `--radius`: flow and geometry parameters
- `--n-steps`, `--output-interval`: runtime and output cadence
- `--output-root`, `--run-name`, `--overwrite`: output organization
- `--device`: `cpu` (default) or `cuda`

## Output organization

Each run writes into:

```text
outputs/
  cylinder_flow/
    <run-name>/
      run_metadata.json
      flow_step_000200.png
      flow_step_000400.png
      ...
```

`run_metadata.json` includes configuration, derived physical parameters (`nu`, `tau`), runtime info, and diagnostics history.

## Tests and checks

Run tests:

```bash
PYTHONPATH=src pytest -q
```

GitHub Actions runs the same test command on every push and pull request.

## Intentionally out of scope (for now)

To keep this step reviewable and robust, these are deferred:

- MRT/TRT and advanced boundary schemes (e.g., Zou/He)
- quantitative validation metrics (drag/lift coefficients, Strouhal extraction)
- large-scale performance tuning and benchmarking
- multi-case experiment orchestration

The current architecture is designed so those capabilities can be added incrementally without destabilizing the core API.
