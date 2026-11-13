"""Legacy entry point for the T2 GPU core-path benchmark — now a thin alias.

History (why this file is not a benchmark any more)
---------------------------------------------------
m15657 created this file as a self-contained D2Q9 benchmark.  It imported
``fused_step_eager`` / ``fused_step_torch_compile`` from
:mod:`tensorlbm.gpu_core_path` — **neither symbol has ever existed** — and its
``run_benchmark`` referenced an undefined ``args`` global, so the documented
repro command

    PYTHONPATH=src python benchmarks/bench_gpu_core_path.py --backend eager

died on ``ImportError`` (and, before that import was reachable, on
``NameError: args``) and produced no artifact.  The m15902 audit
(``docs/audits/m15902_GAP-LBM-T2-GPU-COREPATH_audit.md`` §6.3, §8 item 2)
recorded exactly this and asked for the script to be made runnable.  A second
fault was structural: all three of its "cases" were hand-written D2Q9 solvers
that never called the delivered D3Q19 module, so even a green run would have
been evidence for nothing.

The benchmark itself now lives in ``benchmarks/t2_gpu_core_path/bench.py``
(single source of truth: channel Poiseuille vs the analytic parabola,
Taylor–Green vs the exact decay, backend parity, precision tiers — all driven
through the public API of :mod:`tensorlbm.gpu_core_path`).  This module keeps
the historical entry point working by delegating to it, so the command printed
in ``docs/T2_GPU_CORE_PATH_REPORT.md`` and in the audit is valid again.

Capability boundary (declared, never silently skipped)
------------------------------------------------------
The core path is a **solid-free** solver: half-way bounce-back is applied on
the two ``y`` wall planes only, and ``general_sim`` raises ``solid-free`` when a
solid mask is present (``tests/test_gpu_core_path.py::
test_general_sim_core_path_rejects_solid_domain``).  A lid-driven cavity
Re = 100 against Ghia et al. (1982) therefore **cannot** be run by this solver,
and the earlier claim of a "<5 % Ghia agreement" is withdrawn — see
:data:`NOT_SUPPORTED`.  The same applies to the GPU-only numbers of the T2
report (Triton throughput, 8xGPU weak scaling, FP16 error): this host has no
CUDA device, so they are unverified design intent, not results.

Usage
-----
::

    PYTHONPATH=src python benchmarks/bench_gpu_core_path.py            # full
    PYTHONPATH=src python benchmarks/bench_gpu_core_path.py --quick    # CI scale
    PYTHONPATH=src python benchmarks/bench_gpu_core_path.py --backend triton

``--backend`` is validated with the module's own resolver, so asking for
``triton`` on a host without CUDA raises instead of quietly measuring eager.
Artifacts (JSON + figures) are written to ``artifacts/t2_gpu_core_path/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
if str(REPO_ROOT / "src") not in sys.path:  # pragma: no cover - import hygiene
    sys.path.insert(0, str(REPO_ROOT / "src"))

#: Cases the T2 core path cannot run, with the reason and the pinning test.
NOT_SUPPORTED: dict[str, dict[str, str]] = {
    "lid_driven_cavity_re100": {
        "reference": "Ghia et al. (1982) J. Comput. Phys. 48:387",
        "status": "NOT_SUPPORTED",
        "reason": (
            "GPUCorePathSolver is solid-free (half-way bounce-back on the two y "
            "wall planes only; general_sim raises 'solid-free' for a solid mask), "
            "so a lid-driven cavity is outside the delivered capability. No Ghia "
            "error is claimed by this benchmark."
        ),
        "evidence": (
            "tests/test_gpu_core_path.py::"
            "test_general_sim_core_path_rejects_solid_domain"
        ),
    },
}

#: GPU-only performance claims of the T2 report that this host cannot verify.
WITHHELD: dict[str, str] = {
    "triton_throughput_glups": "no CUDA device on this host (torch.cuda.is_available() is False)",
    "multi_gpu_weak_scaling": "multi-GPU needs CUDA + torch.distributed; not available here",
    "fp16_error_scale": "FP32FP16 store tier is executable, but the ~2e-4 claim was never measured",
}

_CANONICAL = HERE / "t2_gpu_core_path" / "bench.py"


def _load_canonical():
    """Import ``benchmarks/t2_gpu_core_path/bench.py`` by path (not a package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("t2_gpu_core_path_bench", _CANONICAL)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {_CANONICAL}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="T2 GPU core-path benchmark (legacy entry point, delegates)",
    )
    ap.add_argument(
        "--backend", default="eager", choices=["eager", "torch_compile", "triton", "auto"],
        help="backend resolved with the module's own resolver; no silent fallback",
    )
    ap.add_argument("--quick", action="store_true", help="CI scale (shorter runs)")
    ap.add_argument(
        "--no-backend-check", action="store_true",
        help="skip the eager-vs-torch.compile parity arm (Inductor costs minutes on CPU)",
    )
    ap.add_argument("--ny", type=int, nargs="+", default=None,
                    help="ignored: kept for command-line compatibility with the "
                         "pre-m18379 script; the canonical bench fixes the sweep")
    ap.add_argument("--tau", type=float, nargs="+", default=None,
                    help="ignored (see --ny)")
    ap.add_argument("--steps", type=int, default=None, help="ignored (see --ny)")
    ap.add_argument("--out", default=None,
                    help="ignored: artifacts go to artifacts/t2_gpu_core_path/")
    args = ap.parse_args(argv)

    # Validate the backend the same way the solver does: a forced unavailable
    # backend must raise, not degrade to eager (m15657 problem).
    from tensorlbm.gpu_core_path import get_best_available_backend

    if args.backend == "eager":
        resolved = "eager"
    else:
        resolved = get_best_available_backend(args.backend)  # may raise RuntimeError
    if resolved != "eager":
        print(
            f"note: backend {args.backend!r} resolved to {resolved!r}; the canonical "
            "bench reports parity against eager rather than mixing backends mid-arm"
        )

    ignored = [name for name in ("ny", "tau", "steps", "out") if getattr(args, name) is not None]
    if ignored:
        print(f"note: legacy flags ignored ({', '.join('--' + n for n in ignored)}); "
              "the canonical bench owns the parameter sweep")

    for name, info in NOT_SUPPORTED.items():
        print(f"NOT_SUPPORTED {name}: {info['reason']}")

    canonical = _load_canonical()
    canonical_argv: list[str] = []
    if args.quick:
        canonical_argv.append("--quick")
    if args.no_backend_check:
        canonical_argv.append("--no-backend-check")
    return int(canonical.main(canonical_argv))


if __name__ == "__main__":
    raise SystemExit(main())