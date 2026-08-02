# Why TensorLBM — Framework Comparison

This document compares TensorLBM against other open-source CFD / LBM
frameworks along four dimensions relevant to its target use case:
**PyTorch-first, research/prototype, and AI-CFD integration**.

The comparison is intended to be informative, not promotional.  Every project
listed has genuine strengths and a different primary audience.

---

## Frameworks at a glance

| Framework | Language | Primary method | Typical user |
|---|---|---|---|
| **TensorLBM** | Python / PyTorch | LBM (D2Q9/D3Q19/D3Q27) | CFD researchers, AI-CFD prototypers |
| **OpenLB** | C++ | LBM | Engineering practitioners, HPC |
| **Palabos** | C++ | LBM | Academic CFD, parallel computing |
| **PyFR** | Python / C++ / CUDA | FR/SD (Navier-Stokes) | High-order CFD on GPUs |
| **JAX-CFD** | Python / JAX | FD (Navier-Stokes) | ML researchers, differentiable physics |

---

## Dimension 1 — Ease of use

| Framework | Getting started | Scripting API | Notes |
|---|---|---|---|
| **TensorLBM** | `pip install -e .` + a handful of Python calls | Composable pure-Python functions | No C++ build step; Colab-ready |
| **OpenLB** | CMake + C++ template instantiation | XML configuration + C++ | Steep learning curve; rich documentation |
| **Palabos** | CMake + C++ headers | C++ template library | Requires C++ familiarity |
| **PyFR** | `pip install pyfr` + mesh prep (Gmsh) | Python CLI + INI config | Mesh generation is a separate workflow |
| **JAX-CFD** | `pip install jax-cfd` | Functional JAX API | Very concise; less CFD pre/post-processing |

TensorLBM targets the shortest path from "a Python script" to "a running
simulation".  The tradeoff is that it is less mature than OpenLB or Palabos
for production engineering workloads.

---

## Dimension 2 — Performance and engineering scalability

| Framework | Single-GPU | Multi-GPU | HPC cluster | Notes |
|---|---|---|---|---|
| **TensorLBM** | Via PyTorch CUDA | Built-in `MultiGPUSolver` | Not the primary target | Good for medium grids; scales to several GPUs |
| **OpenLB** | Partial (OpenCL/CUDA) | MPI domain decomp | Yes (MPI + OpenMP) | Designed for large-scale HPC |
| **Palabos** | No | MPI domain decomp | Yes (MPI) | CPU-only; strong HPC track record |
| **PyFR** | Yes (CUDA/OpenCL/HIP) | Yes | Yes | Leading high-order GPU performance |
| **JAX-CFD** | Yes (XLA) | Partial | Experimental | Optimised for TPUs/GPUs via XLA |

For production large-scale aerodynamics or naval engineering, OpenLB or
Palabos are typically more appropriate.  TensorLBM is better suited to
research-scale runs where rapid iteration matters more than raw throughput.

---

## Dimension 3 — Python integration

| Framework | Python-native | PyTorch interop | NumPy I/O | Notes |
|---|---|---|---|---|
| **TensorLBM** | Yes — pure Python package | Native (tensors as first-class objects) | Direct (`tensor.numpy()`) | The entire solver is Python / PyTorch |
| **OpenLB** | Python post-processing only | Via external wrappers | VTK / HDF5 files | Core solver is C++; Python is peripheral |
| **Palabos** | Python post-processing only | Via external wrappers | VTK / HDF5 files | Same as OpenLB |
| **PyFR** | CLI + Python API | Not natively | Via export | Python-controlled but not tensor-native |
| **JAX-CFD** | Yes — pure Python / JAX | Via `jax.dlpack` | `jnp.array` → NumPy | Closely related to PyTorch in spirit |

---

## Dimension 4 — AI friendliness

| Framework | Differentiable ops | Built-in ML models | Training data pipeline | Notes |
|---|---|---|---|---|
| **TensorLBM** | Partial (individual operators are `torch` ops) | FNO2d, MLP, Transformer, RANS-AI closure | `tensorlbm.ai` module | Designed explicitly for AI-CFD |
| **OpenLB** | No | No | Manual export | AI integration requires external scripting |
| **Palabos** | No | No | Manual export | Same |
| **PyFR** | No | No | Manual export | Research on differentiable PyFR exists but is not upstream |
| **JAX-CFD** | Yes (full end-to-end JIT) | No built-in; easy to add | Native JAX arrays | Best-in-class for end-to-end differentiable simulation |

For pure differentiable-simulation use cases, JAX-CFD is the most mature
option.  TensorLBM occupies a middle ground: it provides ready-made AI
turbulence and neural-operator components but does not (yet) support full
end-to-end backpropagation through the solver.

---

## Summary

| | Ease of use | Perf / HPC scale | Python integration | AI friendliness |
|---|:---:|:---:|:---:|:---:|
| **TensorLBM** | ★★★★☆ | ★★☆☆☆ | ★★★★★ | ★★★★☆ |
| OpenLB | ★★☆☆☆ | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ |
| Palabos | ★★☆☆☆ | ★★★★☆ | ★★☆☆☆ | ★★☆☆☆ |
| PyFR | ★★★☆☆ | ★★★★★ | ★★★☆☆ | ★★☆☆☆ |
| JAX-CFD | ★★★★☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ |

### When to choose TensorLBM

- You want to prototype and iterate quickly in Python without a C++ build step.
- You need to embed a CFD simulation inside a PyTorch training loop.
- You are exploring LBM-specific physical models (multiphase, free surface,
  thermal, DG-LBM, AMR) and want easy access to intermediate tensors.

### When to choose something else

- **Large-scale HPC**: use OpenLB or Palabos.
- **High-order accuracy on complex geometry**: use PyFR.
- **Full differentiability / gradient-based optimisation through the solver**:
  use JAX-CFD.

---

*Last updated: 2026-08.  Ratings are subjective and reflect the state of each
project at the time of writing.  All projects are under active development.*
