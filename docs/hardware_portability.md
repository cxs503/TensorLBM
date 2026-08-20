# Hardware Portability Layer

TensorLBM's domestic-hardware (国产硬件) portability baseline is layered:

| Layer | Contents | Requirement |
|---|---|---|
| **Data / contract** | `data/` R2 products, catalog, quality gates; Sunway bridge | Platform-independent (numpy/stdlib contracts) |
| **Eager torch path** | `solver3d`/`solver` streaming & collision, all engineering benchmarks | Runs wherever torch runs: CUDA, NPU, MLU, SDAA, MUSA, MPS, CPU |
| **Triton fused path** | `triton_fused*`, `triton_suboff_*`, `backends/triton_backend` | CUDA-only *acceleration*; eager path is the fallback |
| **Distributed** | `multi_gpu`, `suboff_torch_distributed` | NCCL on CUDA; gloo/hccl/rccl elsewhere |
| **SWLBM C line** | Independent Sunway solver | No torch at all; joins via the data bridge (`docs/sunway_data_bridge.md`) |

The governing principle is FluidX3D's *probe → degrade with a warning*: never
hardcode a device list, probe what the host actually has, and when a
capability is missing, fall back to the eager path **and say so**.

## 1. Portability audit of `src/` CUDA-specific points

### L1 — blocked the eager path on non-CUDA backends (all fixed)

| # | Location | Problem | Fix | Verification |
|---|---|---|---|---|
| 1 | `backends/paddle_backend.py` `to_device` | `.cuda()` literal; any non-CUDA accelerator string silently dropped to `.cpu()` | `_place()` resolves the placement method from the device string (`cuda`/`gpu`→`.cuda()`, `npu`→`.npu()`, …; `.cpu()` fallback) | `tests/backends/test_backend_contracts.py`, `tests/test_backends.py` |
| 2 | `backends/paddle_backend.py` `build_eddy_viscosity_model` | `model.cuda()` when `"cuda" in device` | routed through `_place(model, device)` | same |
| 3 | `backends/paddle_backend.py` `build_flow_transformer` | same | same | same |
| 4 | `utils.py` `resolve_device` | only `cpu/sdaa/cuda/mps` known; `npu/mlu/musa` → `ValueError` even when the plugin is installed | generic branch consults `tensorlbm.hardware.probe()`; known-but-unavailable → `RuntimeError` with degradation advice | `tests/test_utils.py`, `tests/test_hardware_portability.py` |
| 5 | `benchmark_observability.py` `resolve_benchmark_device` | any device type outside `cpu/cuda/sdaa` → hard `RuntimeError("unsupported …")` | generic probe-driven branch (npu/mlu/musa resolve when the plugin reports availability) + `hardware_profile` recorded | `tests/test_hardware_portability.py`, `tests/test_fsi_observability.py` |
| 6 | `utils.py` `default_device_name` *(soft)* | ignored npu/mlu/musa hosts → CPU | consults the probe after sdaa/cuda | `tests/test_utils.py` |
| 7 | `ai/suboff_utils.py` `default_suboff_device` *(soft)* | sdaa→cuda→cpu only | + npu/mlu/musa via probe | `tests/test_ai_imports.py` |

A static gate (`tests/test_hardware_portability.py::
test_no_bare_cuda_calls_in_src`) now fails the build if any bare `.cuda()`
re-appears under `src/tensorlbm`.

### L2 — CUDA-specific by design (acceleration layer; guards verified)

| Location | Nature | Guard / fallback |
|---|---|---|
| `triton_fused.py`, `triton_fused_distributed.py`, `triton_suboff_step_distributed.py`, `backends/triton_backend.py` | Triton fused kernels (CUDA-only) | eager `solver3d` gather/roll path is the portable equivalent; `hardware.require("triton")` advice names it |
| `multi_gpu.py` | CUDA multi-GPU slab decomposition | `torch.cuda.device_count() if … else 0`; unused when single-device |
| `suboff_torch_distributed.py` | distributed SUBOFF runner | explicit `RuntimeError("a CUDA device is required")` at construction; `nccl if cuda else gloo` for collectives |
| `cuda_memory_budget.py` | pre-flight CUDA memory admission | both entry points return `None` for `device.type != "cuda"` |
| `cylinder_bfl_control_volume.py`, `sphere_bfl_control_volume.py`, `flat_plate_wall_model.py` | `torch.cuda.set_device`/`reset_peak_memory_stats`/`max_memory_allocated` for timing | inside `if device.type == "cuda":` blocks — no-ops elsewhere |
| `apps/ai4s_flagship.py` | demo loop | `cuda` requested-but-missing degrades to `cpu` |
| `benchmark_observability.py` | per-device availability metadata | cpu/cuda/sdaa branches + (new) generic probe branch |

### ai/ layer NPU/SDAA coverage audit

Already domesticated (before this wave): `ai/suboff_utils.py` (device
detection, SDAA-first), `ai/suboff_train_ddp.py` (dedicated SDAA DDP training
variant, `sdaa:{local_rank}` + `torch.sdaa.set_device` — by design for the
LoongArch host), `ai/train.py` (fully `ops`-abstracted, runs on any backend
including paddle).

Gaps found: the remaining ai modules (`fno.py`, `model.py`, `transformer.py`,
`pipeline.py`, `inference.py`, `suboff_train.py`, `suboff_inference.py`,
`dataset.py`) are device-**parameter** driven (`.to(device)`) and needed no
per-vendor code — but *default device selection* existed only in
`suboff_utils` and missed NPU/MLU/MUSA (fixed, item 7). Note: grepping `ai/`
for `npu` mostly hits the substring of "input"; the only real NPU/SDAA sites
are the ones above.

## 2. `tensorlbm.hardware` — capability probe module

```python
from tensorlbm.hardware import probe, require, HardwareCapabilityError

profile = probe()                    # cached; probe(refresh=True) to re-probe
profile.has_backend("npu")           # False on this host
profile.default_device               # "sdaa" | "cuda" | "npu" | … | "cpu"
profile.fp16_storage, profile.bf16_storage
profile.triton_available, profile.triton_version
profile.collective("nccl")           # CollectiveInfo(available=True, via=…)
profile.to_dict()                    # JSON-safe → benchmark records

require("triton")                    # raises HardwareCapabilityError with advice
```

* **Backends** — import-probed, not hardcoded: `cuda`/`mps` built into torch;
  `sdaa/npu/mlu/musa` detected via `importlib.util.find_spec` of their plugin
  packages (`torch_sdaa`, `torch_npu`, `torch_mlu`, `torch_musa`) and only
  imported when present; each probe is exception-guarded (a broken vendor
  plugin degrades to `available=False`, never crashes the run).
* **Reduced precision** — fp16/bf16 *storage* is probed by allocating and
  computing on the default device (allocation alone can succeed where kernels
  don't exist).
* **Collectives** — `nccl/gloo/mpi` via `torch.distributed.is_*_available()`;
  `hccl`/`rccl` via plugin presence (they ship inside `torch_npu`/`torch_musa`).
* **Triton** — import probe + version.
* `require(capability, profile=None)` — raises `HardwareCapabilityError`
  whose message always carries a degradation suggestion:
  `fp16_storage`/`bf16_storage` → "degrade to FP32FP32"; `triton` → "eager
  `stream3d` gather/roll path"; `nccl` → "gloo / vendor collective (hccl on
  Ascend, rccl on ROCm-class)"; backend names → "eager path on an available
  backend".

### Observability integration

`resolve_benchmark_device()` now embeds a full `hardware_profile` snapshot,
and every `run_status.json` written by `BenchmarkReporter` carries a
`hardware` field — each benchmark record is self-describing about the host it
ran on, so cross-platform (CUDA vs NPU vs CPU) comparisons are auditable.

## 3. Device-agnostic gate tests (`tests/test_hardware_portability.py`)

CI-runnable on CPU-only machines:

* eager solver path: `stream3d` (gather) ≡ `stream3d_roll` bit-for-bit;
  collide→stream steps finite and mass-conserving on CPU;
* core data chain: `save_fields_hdf5 → register_product → load_product →
  load_product_arrays` including catalog quality + lineage, CPU-only;
* static scan: zero bare `.cuda()` under `src/tensorlbm`;
* probe/require behaviour incl. synthetic-profile degradation advice;
* observability records carry the hardware profile.

## 4. LSF backend (`app/backend/services/hpc_scheduler.py`)

`TENSORLBM_HPC_MODE=lsf` adds `submit_lsf` (script staged under
`TENSORLBM_HPC_LOG_DIR` — must be shared-FS — submitted as `bsub -q -n -J -o
<shell> <script>`), `query_lsf_status` (parses `bjobs -l` `Status<…>`, stable
across stock and SWA LSF whose short-format column order differs; maps
PEND/RUN/DONE/EXIT/… → pending/running/done/exit/suspended/exiting), and
`cancel_lsf` (`bkill`). Live-validated on the psn002 SWA cluster; SWA
quirks encoded: no `-W`, no `-e`, `bash` missing on compute nodes
(`TENSORLBM_HPC_LSF_SHELL=sh`), bsub must run outside the "fs base" home.

## 5. Recommendations for Ascend and Hygon ports

**Ascend NPU (昇腾)**
* Install `torch_npu`; the probe picks `npu` up automatically; eager path
  needs no changes.
* HCCL is detected via the `torch_npu` plugin presence; multi-NPU runs should
  route `init_process_group(backend="hccl")` — keep the
  `nccl if cuda else gloo` pattern in `suboff_torch_distributed.py` in mind
  and extend it with an hccl branch before the NPU distributed port.
* `torchair`/`npu_fusion` graph-compile acceleration should slot in as a
  *Triton-equivalent acceleration layer*: probe `torchair` importability in
  `hardware.probe()` (one more registry entry), keep the eager fallback, and
  gate entry with `require()`-style advice ("torchair unavailable → eager").
* fp16 storage is broadly available on NPU; bf16 is the native training
  dtype — the probe reports both so precision selection stays data-driven.

**Hygon DCU / SDAA (海光)**
* `torch_sdaa` is already a first-class citizen (probe priority puts SDAA
  first, mirroring `utils.default_device_name`); the D27 BGK work ran through
  this path.
* RCCL ships with the ROCm-derived stack; probe it before choosing
  collectives on multi-card DCU runs (`hardware.collective("rccl")`).
* Watch the known Triton-on-DCU gaps (Q=19/32 padding, force all-reduce
  contiguity — see `docs/triton_distributed_notes.md`); on DCU the eager
  gather path remains the correctness baseline for cross-checks.

**Moore Threads MUSA / Cambricon MLU**
* Same pattern: plugin import probe (`torch_mlu`, `torch_musa`) + eager
  baseline; no source changes required once the plugin registers
  `torch.<name>`.
