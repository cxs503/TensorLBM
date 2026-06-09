# TensorLBM Platform

A Browser/Server (B/S) web platform that integrates all TensorLBM simulation capabilities into a unified interactive interface. The backend can leverage GPU clusters via PyTorch device selection (`cuda:0`, `cuda:1`, …).

## Architecture

```
platform/
├── backend/           FastAPI REST + WebSocket backend
│   ├── main.py        App entry point; WebSocket broadcaster
│   ├── job_manager.py Thread-safe job queue (ThreadPoolExecutor)
│   ├── schemas/       Router request/response Pydantic models
│   ├── services/      Shared service-layer helpers for routers
│   └── routers/
│       ├── jobs.py        Job CRUD + file serving
│       ├── preprocess.py  Geometry generation & unit conversion
│       ├── solver.py      All simulation endpoints
│       ├── postprocess.py Result analysis & metadata
│       ├── benchmarks.py  Validation benchmark suites
│       └── agent.py       Conversational LLM agent (chat / capabilities)
├── backend/agent_core.py  Agent tool registry + intent parser
└── frontend/
    ├── index.html     Single-page app shell (Bootstrap 5)
    └── static/js/     Domain-split frontend logic modules
```

## Quick Start

Canonical developer setup and checks are documented in `../docs/development_workflow.md`.

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

Optional runtime controls:

```bash
# Comma-separated CORS allow-list ("*" by default)
export TENSORLBM_CORS_ALLOW_ORIGINS="http://localhost:8000,https://your-domain"
# Job output root (default: /tmp/tensorlbm_platform)
export TENSORLBM_OUTPUT_ROOT=/var/lib/tensorlbm/jobs
# STL upload cap in MB (default: 50)
export TENSORLBM_MAX_UPLOAD_MB=100
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

### CAD (3D Modelling MVP)
- Parametric 3D hull generation (Wigley / Series60 / KCS)
- Browser-side interactive three.js view (orbit/zoom, wireframe, clipping)
- 3D export: glTF + STL (STEP when CadQuery/OpenCascade is available)
- Versioned CAD models via `/api/cad/3d/models/*`
- CAD→LBM bridge endpoint: `/api/cad/3d/models/{id}/lbm-mask`

### Solver
All simulation types are submitted as background jobs and monitored in real time via WebSocket.
The platform also supports batch Reynolds-number sweeps for cylinder flow via
`POST /api/solve/cylinder-flow/scan` (submit 2–20 `re_values` in one request).

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
- View all PNG step images from a completed job (lightbox zoom)
- Download CSV / VTK / HDF5 output files
- Read `run_metadata.json` directly in the browser
- Full job log viewer

### Benchmarks
| Suite | Reference |
|---|---|
| Marine (7 cases) | Williamson (1988), Faltinsen (1978), Bearman & Zdravkovich (1978), Moser et al. (1999), ITTC-1957, SUBOFF geometry checks |
| Multiphase (3 cases) | Young–Laplace, Shan & Chen (1993), Pan et al. (2004) |
| Lid-driven cavity – Ghia | Ghia et al. (1982) Re=100/400/1000 |
| MLUPS performance | D2Q9 BGK throughput |
| Porous media | Laplace + capillary invasion |

For the full SUBOFF appendage workflow and benchmark interpretation, see `../docs/suboff_platform_manual.md`.

### AI Agent (LLM-powered)
A conversational assistant that drives the full pipeline (modelling → solver
→ post-processing) from natural language, in Chinese or English. Exposed
under `/api/agent/*` and accessible via the **AI Agent** tab in the UI.

* `POST /api/agent/chat` — `{message, history}` → `{reply, actions,
  suggestions, used_llm, intent}`
* `GET  /api/agent/capabilities` — list of registered tools/scenarios
* `GET  /api/agent/info` — runtime info (LLM enabled? which model?)

Example prompts:

```
Run a cylinder flow at Re=200 with nx=200, n_steps=2000
用圆柱绕流做一个 Re=120 的算例，步数=2000
做一个方腔算例 Re=400
Analyze job <id> and extract a velocity profile
```

The agent works **offline by default** through a rule-based intent
parser that recognises every solver scenario plus job-management
intents (status, list, analyze, velocity profile). To switch to a
remote LLM for richer free-form replies, set:

```bash
export TENSORLBM_LLM_API_KEY=sk-…
export TENSORLBM_LLM_BASE_URL=https://api.openai.com/v1   # optional
export TENSORLBM_LLM_MODEL=gpt-4o-mini                    # optional
```

Any OpenAI-compatible Chat-Completions endpoint works (OpenAI,
Azure OpenAI proxy, vLLM, DeepSeek, Moonshot, etc.). The LLM only
generates the *natural-language reply* – all platform actions still go
through the typed, length-capped tool layer, so a hallucinated 10000×10000
grid cannot escape the safety clamps (`MAX_GRID_2D=1024`, `MAX_GRID_3D=256`,
`MAX_STEPS=200_000`).

### HPC Orchestration + AI Governance (new)

The platform now exposes baseline orchestration and governance APIs for the
HPC+AI demonstration upgrade:

* `GET /api/orchestration/templates` — staged experiment templates (A/B/C).
* `POST /api/orchestration/experiments/submit` — submit a template run
  (currently implemented: `cylinder_re_sweep`).
* `GET /api/orchestration/kpis` — aggregate queue wait, retries, runtime,
  throughput, resource distribution and estimated cost from platform jobs.
* `GET /api/ai/governance/registry-summary` — summarize AI model registry
  quality statistics.
* `POST /api/ai/governance/confidence-gate` — uncertainty/error-based decision
  gate (`accept_ai` vs `fallback_hpc_high_fidelity`).
* `POST /api/ai/governance/active-learning/prioritize` — rank candidate samples
  for HPC re-simulation and incremental retraining.

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
| `POST` | `/api/jobs/{id}/cancel` | Request cancellation |
| `POST` | `/api/jobs/cleanup` | Cleanup completed jobs by policy |
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
