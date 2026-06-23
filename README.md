# TensorLBM

[English](README.md) | [简体中文](README.zh-CN.md)

TensorLBM is a CPU-first PyTorch Lattice Boltzmann Method platform focused on **reproducible research experiments** with clear extension points.

## Documentation / 文档

- **[软件说明书 / Software Manual](docs/software_manual.md)** – 完整的船舶与海洋工程算例说明、定量 benchmark 对比和 API 参考。
  Full ship & ocean engineering benchmark documentation, quantitative comparisons, and API reference.
- **[SUBOFF Platform Manual](docs/suboff_platform_manual.md)** – 完整 SUBOFF 全附件案例的 CLI / Platform 运行步骤、精度判据与结果解读。
- **[HPC + AI: AI Turbulence Models](docs/ai_turbulence.md)** – Agent-driven 数据生成 → SQLite 入库 → AI 湍流模型训练 → AI 模型嵌入 LBM 的端到端示范 (`tensorlbm.ai`).
- **[Development Workflow](docs/development_workflow.md)** – single entrypoint for setup, checks, platform startup, and output naming conventions.
- **[Observability Notes](docs/observability.md)** – job lifecycle, output schema, and failure-triage checklist.

## What TensorLBM provides

- A small, explicit public API in `src/tensorlbm/__init__.py`
- **D2Q9** and **D3Q19** lattice primitives (`equilibrium`, `macroscopic`, lattice constants)
- **BGK** and **MRT** collision operators for both 2D and 3D
- Non-Newtonian **power-law BGK** rheology utilities (shear-rate, apparent viscosity, variable-τ collision)
- Boundary conditions: bounce-back, **Zou/He** inlet-velocity and outlet-pressure BCs
- Momentum-exchange force diagnostics (drag/lift) for the 2D cylinder
- A 2D cylinder-flow runner with CLI, Strouhal-number extraction, structured outputs, and diagnostics
- A 3D sphere-flow runner with CLI and structured outputs
- Batch Reynolds-number parameter scan (`examples/param_scan.py`)
- Automated tests and CI

## Engineering principles

1. **Stable API first**: keep public exports intentional and backward-compatible.
2. **Composable solver core**: isolate lattice math, solver stepping, and boundary logic.
3. **Reproducible runs**: parameterized CLI + deterministic run folder layout + metadata snapshot.
4. **Fast feedback loops**: smoke tests and CI on push/PR.
5. **CPU-first defaults, GPU-ready shape**: default to CPU, but keep interfaces ready for device scaling.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install -r platform/requirements.txt
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
- `--resume-checkpoint`: resume from an existing checkpoint directory
- `--device`: `cpu` (default), `cuda`, or `mps` (Apple Silicon)
- `--num-threads`: PyTorch CPU thread count for multicore runs on `cpu`

## Run the sphere-flow example (3D)

```bash
PYTHONPATH=src python examples/sphere_flow_3d.py \
  --nx 60 --ny 30 --nz 30 --radius 4 --n-steps 50 --output-interval 25 \
  --run-name smoke --overwrite
```

## Run the ship CAD-to-flow workflow

```bash
PYTHONPATH=src python examples/ship_hull_flow.py \
  --hull-type wigley \
  --nx 80 --ny 40 --nz 30 \
  --hull-length 40 --hull-beam 6 --hull-draft 8 \
  --re 200 --n-steps 2000 --output-interval 200 \
  --export-stl --run-name ship_workflow --overwrite
```

This workflow now writes CAD artefacts (`cad_preview.png`, optional `hull.stl`),
solver outputs (`run_metadata.json`, `forces.csv`, `flow_step_*.png`), and
post-processing files (`postprocess_summary.json`, `wake_profile.csv`).

## Batch parameter scan

```bash
PYTHONPATH=src python examples/param_scan.py \
  --re 20 40 80 100 --nx 160 --ny 60 --n-steps 2000 --output-interval 100
```

## Output organization

Each run writes into:

```text
outputs/
  cylinder_flow/
    <run-name>/
      run_metadata.json
      forces.csv
      flow_step_000200.png
      ...
  sphere_flow/
    <run-name>/
      run_metadata.json
      flow_step_000100.png
      ...
```

`run_metadata.json` includes configuration, derived physical parameters (`nu`, `tau`), runtime info, and diagnostics history.

### Restart / resume runs

- CLI: pass `--resume-checkpoint <run_dir>` (directory containing `checkpoint_f.pt` and `checkpoint_meta.json`).
- Platform API (`/api/solve/cylinder-flow` and `/api/solve/sphere-flow`):
  - `resume_checkpoint`: explicit checkpoint directory path, or
  - `resume_from_job_id`: resume from the latest checkpoint of an existing job.

## Tests and checks

Run tests:

```bash
PYTHONPATH=src pytest -q
```

GitHub Actions runs the same test command on every push and pull request.

## API layering

- Stable compatibility-oriented API: `tensorlbm.api`
- Fast-evolving API surface: `tensorlbm.experimental`
- Grouped domain namespaces for new code: `tensorlbm.lattice_models`, `tensorlbm.physics`, `tensorlbm.cad`

## Quantitative validation summary

| Benchmark | Parameter | TensorLBM | Reference | Error |
|-----------|-----------|-----------|-----------|-------|
| 2D cylinder flow | Strouhal St (Re=100) | ≈ 0.183 | 0.166 (Williamson 1989) | ~10% |
| Sloshing tank | Natural frequency ω₁ | Faltinsen formula | Faltinsen (1978) | < 2% |
| Near-bed pipeline | Strouhal St (Re=200, e/D=0.5) | measured | Bearman & Zdravkovich (1978) | — |
| Turbulent channel | Log-law slope κ (Re_τ=100) | ≈ 0.41 | 0.41 (von Kármán) | < 5% |
| Wigley ship workflow | Cb error + symmetry checks (Re=200) | Cb error < 25%, \|Cd\| > 0, \|Cs\|/\|Cd\| < 0.1, \|Cl\|/\|Cd\| < 0.25 | analytical Cb + symmetry | pass |
| Marine geometry library | Multi-hull CAD consistency (Wigley/Series60/KCS + SUBOFF variants) | ship Cb checks + SUBOFF volume monotonicity | analytical coefficients + topology expectation | pass |
| Lid-driven cavity | u-centreline (Re=100,400,1000) | matched | Ghia et al. (1982) | < 1% |

Run the full benchmark suite:

```bash
PYTHONPATH=src python benchmarks/bench_marine.py
PYTHONPATH=src python benchmarks/bench_multiphase.py
PYTHONPATH=src python benchmarks/bench_dam_break.py --fast
```

`bench_multiphase.py` 现已包含 2D + 3D 多相基准，覆盖静液滴、SCMP 自旋分相、Free-Energy 相场液滴松弛，以及 D3Q19 三维静液滴 / 三维自旋分相。

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, coding conventions, and how to add a new solver or benchmark.
