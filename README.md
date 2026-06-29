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
- **D2Q9**, **D3Q19**, and **D3Q27** lattice primitives (`equilibrium`, `macroscopic`, lattice constants)
- **BGK**, **MRT**, **TRT**, **RLBM** (Regularized), and **Cumulant** collision operators for 2D and 3D
- **Adaptive Mesh Refinement (AMR)**: dynamic patch management for D2Q9 and D3Q19, up to 5 refinement levels with Filippova–Hänel interface exchange, and multiple refinement indicators (non-equilibrium, vorticity, gradient, boundary-layer)
- **DG-LBM hybrid solver**: nodal Discontinuous-Galerkin LBM with P1-Lobatto DG advection, SSP-RK3 time stepping, DG↔LBM P0 interface coupling, and support for 2D cylinder / 3D sphere / SUBOFF hull flows
- **LES turbulence models**: Smagorinsky, Dynamic Smagorinsky (Germano identity), WALE, Vreman — for D2Q9, D3Q19, and D3Q27
- **RANS turbulence models**: k-ε (`KESolver`) and k-ω SST (`KOmegaSSTSolver`)
- Non-Newtonian **power-law BGK** rheology utilities (shear-rate, apparent viscosity, variable-τ collision)
- **Multiphase flow** (D2Q9 & D3Q19): Shan-Chen single/two-component, Color-Gradient, Free-Energy phase-field
- **Immersed Boundary Method (IBM)**: direct-forcing IBM in 2D and 3D with 2-point hat and 4-point cosine delta kernels
- **Thermal LBM**: double-distribution-function model (D2Q9+D2Q5 / D3Q19+D3Q7) with Boussinesq buoyancy
- **Conjugate heat transfer (CHT)**: coupled fluid–solid heat conduction with interface boundary conditions
- **Aeroacoustics**: Ffowcs Williams–Hawkings (FWH) far-field solver, SPL spectrum, and OASPL computation
- **AI turbulence models**: MLP eddy-viscosity model, Transformer-based self-supervised flow model, DNS-to-LES data pipeline, AI-embedded LBM collision
- Boundary conditions: bounce-back, **Zou/He** inlet-velocity and outlet-pressure BCs, Bouzidi interpolated bounce-back, **moving-wall** (Ladd 1994), **far-field**, **sponge/absorbing-layer** outlet BC, **rough-wall** (equivalent sand-grain), JONSWAP irregular-wave inlet
- **Turbulent inlet profiles**: log-law, power-law, parabolic, Blasius, Womersley, synthetic turbulence 2D, DFSEM, Digital Filter Method
- **Turbulence statistics**: `TurbulenceStatsAccumulator`, Reynolds stresses, turbulence intensity, turbulence length scale
- **Streamline / pathline tracing**: 2D and 3D integration, uniform/line seed points, residence-time computation
- **Surface & volume integrals**: mass flow rate, area average, surface force/moment, force/moment coefficients, pressure drop
- Momentum-exchange force diagnostics (drag/lift) for 2D and 3D obstacles
- **Multi-GPU domain decomposition**: `MultiGPUSolver2D/3D`, halo exchange, auto-decompose
- **Multi-backend dispatch**: `torch` (default), `paddle`, `mindspore` via `get_backend` / `set_backend`
- **Marine engineering**: Wigley / Series60 / KCS hull CAD, SUBOFF submarine CAD, propeller (KP-505), Airy and JONSWAP wave BCs
- **Benchmark runners**: cylinder flow, sphere flow (D3Q19/D3Q27), ship hull flow, SUBOFF resistance, ellipsoid, airfoil (NACA 4-digit), propeller, IBM propeller, actuator disk, backward-facing step, lid-driven cavity, rotating cylinder, turbulent channel, sloshing tank, pipeline flow, dam break, porous media (2D/3D), multiphase water entry
- **Post-processing**: Strouhal FFT, Q-criterion, λ₂-criterion, vorticity, VTK/HDF5/XDMF export, streamlines, force coefficients, wake profiles
- Batch Reynolds-number parameter scan (`examples/param_scan.py`)
- Automated tests and CI

## Engineering principles

1. **Stable API first**: keep public exports intentional and backward-compatible.
2. **Composable solver core**: isolate lattice math, solver stepping, and boundary logic.
3. **Reproducible runs**: parameterized CLI + deterministic run folder layout + metadata snapshot.
4. **Fast feedback loops**: smoke tests and CI on push/PR.
5. **CPU-first defaults, GPU-ready shape**: default to CPU, but keep interfaces ready for device scaling.
6. **Multi-backend**: PyTorch is the default; Paddle and MindSpore backends are selectable at runtime.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras for the deployable web platform:

```bash
cd app && pip install -r requirements.txt
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
- `--backend`: `torch` (default), `paddle`, or `mindspore`
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

This workflow writes CAD artefacts (`cad_preview.png`, optional `hull.stl`),
solver outputs (`run_metadata.json`, `forces.csv`, `flow_step_*.png`), and
post-processing files (`postprocess_summary.json`, `wake_profile.csv`).

## Run the DG-LBM hybrid solver

```bash
# 2D cylinder with DG advection bands
PYTHONPATH=src python examples/dg_lbm_cylinder_hybrid.py

# 3D sphere with DG-LBM hybrid
PYTHONPATH=src python examples/dg_lbm_sphere_hybrid.py

# SUBOFF with real DG solver (use_real_dg=True)
PYTHONPATH=src python examples/dg_suboff_highre_mrt.py
```

## Run the AI turbulence pipeline

```bash
# DNS data generation → SQLite → AI model training → AI-embedded LBM
PYTHONPATH=src python examples/ai_dns_case.py
PYTHONPATH=src python examples/ai_turbulence_pipeline.py
```

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
- AI turbulence sub-package: `tensorlbm.ai`
- Backend dispatch: `tensorlbm.backends` — `torch` (default), `paddle`, `mindspore`

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
| DG-LBM convergence | MMS spatial order (P1 elements) | O(Δx²)–O(Δx³) | manufactured solution | pass |
| Airfoil (NACA 4-digit) | C_L and C_D | within reference band | XFOIL / panel method | — |

Run the full benchmark suite:

```bash
PYTHONPATH=src python benchmarks/bench_marine.py
PYTHONPATH=src python benchmarks/bench_multiphase.py
PYTHONPATH=src python benchmarks/bench_dam_break.py --fast
PYTHONPATH=src python benchmarks/bench_mlups.py --collisions all --device cpu
```

`bench_multiphase.py` includes 2D + 3D multiphase benchmarks: static droplet, SCMP spinodal decomposition, Free-Energy phase-field droplet relaxation, and D3Q19 3D static droplet / 3D spinodal decomposition.

`bench_mlups.py` compares throughput of BGK / MRT / TRT / RLBM collision operators on D2Q9 and D3Q19 grids.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, coding conventions, and how to add a new solver or benchmark.
