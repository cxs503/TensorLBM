# TensorLBM Platform

A Browser/Server (B/S) web platform that integrates all TensorLBM simulation capabilities into a unified interactive interface. The backend can leverage GPU clusters via PyTorch device selection (`cuda:0`, `cuda:1`, …).

## Architecture

```
platform/
├── backend/           FastAPI REST + WebSocket backend
│   ├── main.py        App entry point; WebSocket broadcaster
│   ├── job_manager.py Thread-safe job queue (ThreadPoolExecutor)
│   └── routers/
│       ├── jobs.py        Job CRUD + file serving
│       ├── preprocess.py  Geometry generation & unit conversion
│       ├── solver.py      All simulation endpoints
│       ├── postprocess.py Result analysis & metadata
│       └── benchmarks.py  Validation benchmark suites
└── frontend/
    └── index.html     Single-page app (Bootstrap 5, vanilla JS)
```

## Quick Start

### 1. Install dependencies

```bash
# From repository root
pip install -e ".[dev]"
pip install -r platform/requirements.txt
```

### 2. Launch the server

```bash
cd platform
bash start.sh
# Or directly:
PYTHONPATH=../src uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open the browser

Navigate to **http://localhost:8000**

---

## Platform Modules

### Pre-processing
- **Polygon → obstacle mask** – convert polygon vertices to a 2D Boolean mask; returns PNG preview
- **Random porosity mask** – Gaussian-correlated random porous medium (2D)
- **STL voxelisation** – upload an STL file and convert to a 3D voxel mask
- **LBM Unit Converter** – convert physical (SI) quantities to LBM lattice units; reports Re, τ, dx, dt

### Solver
All simulation types are submitted as background jobs and monitored in real time via WebSocket.

| Simulation | Dim | Model | Config class |
|---|---|---|---|
| Cylinder flow | 2D | BGK | `CylinderFlowConfig` |
| Lid-driven cavity | 2D | BGK | `LidDrivenCavityConfig` |
| Backward-facing step | 2D | BGK | `BackwardFacingStepConfig` |
| Turbulent channel (LES) | 2D | Smagorinsky BGK | `TurbulentChannelConfig` |
| Near-bed pipeline flow | 2D | BGK | `PipelineFlowConfig` |
| Dam break | 2D | SC/CG/FE/SCMP | `DamBreakConfig` |
| Sloshing tank | 2D | Color-Gradient | `SloshingTankConfig` |
| Porous drainage | 2D | SC/CG | `PorousDrainageConfig` |
| Sphere flow | 3D D3Q19 | BGK | `SphereFlowConfig` |
| Ship hull – Wigley | 3D D3Q19 | Smagorinsky MRT | `ShipHullFlowConfig` |

### Post-processing
- View all PNG snapshots from a completed job (lightbox zoom)
- Download CSV / VTK / HDF5 output files
- Read `run_metadata.json` directly in the browser
- Full job log viewer

### Benchmarks
| Suite | Reference |
|---|---|
| Marine (5 cases) | Williamson (1988), Faltinsen (1978), Bearman & Zdravkovich (1978), Moser et al. (1999) |
| Multiphase (3 cases) | Young–Laplace, Shan & Chen (1993), Pan et al. (2004) |
| Lid-driven cavity – Ghia | Ghia et al. (1982) Re=100/400/1000 |
| MLUPS performance | D2Q9 BGK throughput |
| Porous media | Laplace + capillary invasion |

---

## GPU Cluster Usage

Select any `cuda:N` device in the **Device** dropdown of any simulation or benchmark form. The backend thread pool allows multiple jobs to run concurrently on different GPUs.

For large-scale cluster runs, increase the thread pool size via the `TENSORLBM_MAX_WORKERS` environment variable (default: 4):

```bash
TENSORLBM_MAX_WORKERS=16 bash start.sh
```

---

## API Reference

Interactive API docs are available at **http://localhost:8000/docs** (Swagger UI).

Key endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/status` | Platform status & GPU info |
| `GET` | `/api/jobs/` | List all jobs |
| `GET` | `/api/jobs/{id}` | Job detail |
| `DELETE` | `/api/jobs/{id}` | Remove job |
| `GET` | `/api/jobs/{id}/logs` | Log lines |
| `GET` | `/api/jobs/{id}/files` | List output files |
| `GET` | `/api/jobs/{id}/files/{path}` | Download file |
| `GET` | `/api/jobs/{id}/images` | List PNG snapshots |
| `POST` | `/api/preprocess/polygon-mask` | Polygon → mask |
| `POST` | `/api/preprocess/random-porosity-2d` | Random porosity |
| `POST` | `/api/preprocess/units` | Unit conversion |
| `POST` | `/api/solve/{type}` | Submit simulation |
| `POST` | `/api/benchmarks/{type}` | Run benchmark |
| `WS` | `/ws` | Real-time job updates |

---

## Development

```bash
# Run with auto-reload
cd platform
PYTHONPATH=../src uvicorn backend.main:app --reload --reload-dir backend

# Check formatting
cd ..
ruff check platform/backend
```
