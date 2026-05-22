# TensorLBM

TensorLBM is an early-stage, CPU-first PyTorch codebase for Lattice Boltzmann Method (LBM) research and engineering practice.

## Current scope

This repository currently provides a lightweight D2Q9/BGK foundation intended to support further cylinder-flow development:

- D2Q9 lattice constants and utility functions
- Equilibrium distribution and macroscopic recovery
- A minimal collision + streaming step
- A small pytest suite and GitHub Actions CI smoke coverage

## Installation

```bash
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
```

## Run the example

```bash
PYTHONPATH=src python examples/minimal_d2q9.py
```

## Run tests

```bash
PYTHONPATH=src pytest -q
```

(When running in CI, `pytest` also works via `pyproject.toml` `pythonpath` settings.)

## Current limitations

- The current implementation is intentionally minimal and CPU-first.
- Boundary-condition support is limited (simple periodic streaming and optional basic on-site bounce-back mask).
- Cylinder-flow-specific forcing/diagnostics (drag, lift, Strouhal) are not implemented yet.
- Advanced collision models (MRT/TRT) and validation cases are out of scope for this PR.
