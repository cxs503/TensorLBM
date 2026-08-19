"""Unified ``torch.compile`` routing for the ``benchmarks/`` suite.

New benchmark standard (2026-08-19): every verified case must run its
time-stepping chain through the shared compile module
:mod:`tensorlbm.compile_utils` (``validate_compile_mode`` +
``compile_step``), and stay within its error budget (<3%).  This helper
is the single adapter between the per-case ``run.py`` entry scripts and
that module, so the routing logic is written once:

* :func:`ensure_tensorlbm_importable` — portable ``<repo>/src`` path
  bootstrap replacing the historical hardcoded
  ``/home/wxsc/cxs/TensorLBM/src`` inserts (the benchmarks predate the
  shared machine layout; the insert made every ``run.py`` unrunnable
  outside that one host).
* :func:`normalize_compile_mode` — maps the CLI spelling ``"eager"`` to
  ``None`` (the canonical eager mode of ``compile_utils``) and validates
  the result against the shared whitelist.
* :func:`route_step` — validate + wrap one *whole-step* function with
  ``compile_step``; prints one routing banner so every run log/artifact
  records which path was taken.
* :func:`add_compile_mode_arg` / :func:`compile_mode_from_args` — the
  uniform ``--compile-mode {eager,default,max-autotune-no-cudagraphs}``
  CLI knob (default ``"default"`` = compiled, per the new standard;
  ``eager`` keeps the A/B comparison one flag away).

The rules this enforces (all from the ``compile_utils`` lessons — read
that module docstring before changing anything here):

1. The wrapped function must be the **entire** per-step chain
   (collision + boundary conditions + streaming), a pure tensor
   function ``f -> f'`` with no host synchronisation.
2. The **step index and every step-dependent branch stay outside** the
   wrapped function — i.e. the every-N-step monitoring blocks
   (``.item()`` residuals, steady-state drift checks, NaN guards) that
   each benchmark loop already has must remain in the eager driver
   loop, exactly as before; only the plain per-step chain moves into
   the compiled closure.
3. Cudagraph-class modes are rejected by ``validate_compile_mode``
   (structural conflict with the LBM feedback loop) — do not try to
   bypass that here.

``compile_utils`` itself is consumed as-is; this module adds no
compilation behaviour of its own.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"

#: CLI spellings accepted for the eager path.  ``None`` is the canonical
#: mode inside ``compile_utils``; ``"eager"`` is the human-facing CLI
#: spelling used by the benchmark scripts (argparse cannot express a
#: ``None`` default nicely).
EAGER_CLI_SPELLINGS: frozenset[str | None] = frozenset({"eager", ""})


def ensure_tensorlbm_importable() -> str | None:
    """Put ``<repo>/src`` at the front of ``sys.path`` if a checkout exists.

    Portable replacement for the hardcoded ``sys.path.insert(0,
    "/home/wxsc/cxs/TensorLBM/src")`` lines the benchmark scripts were
    born with.  Returns the inserted path, or ``None`` when no source
    tree is found next to this file (in which case an already-imported
    or installed ``tensorlbm`` is assumed — a failure then surfaces at
    the import below with the normal ``ImportError``).
    """
    if not _SRC_DIR.is_dir():
        return None
    src = str(_SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


ensure_tensorlbm_importable()

from tensorlbm.compile_utils import (  # noqa: E402  (needs the path bootstrap above)
    compile_step,
    validate_compile_mode,
)

__all__ = [
    "EAGER_CLI_SPELLINGS",
    "ensure_tensorlbm_importable",
    "normalize_compile_mode",
    "route_step",
    "add_compile_mode_arg",
    "compile_mode_from_args",
]

StepFn = Callable[..., Any]


def normalize_compile_mode(mode: str | None) -> str | None:
    """Return the canonical compile mode for *mode* (``"eager"`` -> ``None``).

    Raises the shared :func:`tensorlbm.compile_utils.validate_compile_mode`
    ``ValueError`` (with its cudagraph/unknown-mode reason) for anything
    that is not a proven mode.
    """
    if isinstance(mode, str) and mode.lower() in EAGER_CLI_SPELLINGS:
        mode = None
    validate_compile_mode(mode)
    return mode


def route_step(
    step_fn: StepFn,
    mode: str | None = "default",
    *,
    name: str = "benchmark",
    warmup_hint: str | None = None,
    quiet: bool = False,
) -> StepFn:
    """Route one benchmark whole-step function through ``compile_step``.

    This is the single call every ``benchmarks/verified/*/run.py`` uses:

    .. code-block:: python

        step = route_step(_step, args.compile_mode, name="cavity_re100")
        for i in range(steps):
            f = step(f)            # whole chain: collide -> BC -> stream -> BC
            if i % K == 0:         # monitoring stays OUTSIDE (eager)
                ...

    Args:
        step_fn: the whole-step function ``f -> f'`` (plus any per-step
            tensor outputs).  Must not contain the step index or any
            host synchronisation (see module docstring).
        mode: ``None``/``"eager"`` (passthrough), ``"default"`` or
            ``"max-autotune-no-cudagraphs"``.
        name: benchmark/case name used in the routing banner and the
            default warmup hint.
        warmup_hint: forwarded to :func:`tensorlbm.compile_utils.compile_step`.
        quiet: suppress the routing banner (banner is also the audit
            trail showing the case went through the shared module).

    Returns:
        *step_fn* itself for the eager path (byte-identical behaviour),
        else the ``torch.compile`` wrapper.
    """
    canonical = normalize_compile_mode(mode)
    wrapped = compile_step(
        step_fn,
        canonical,
        warmup_hint=warmup_hint or f"benchmark {name!r}: one whole-step graph per grid shape",
    )
    if not quiet:
        routed = "eager (compile_step passthrough)" if canonical is None else f"torch.compile(mode={canonical!r})"
        print(f"[compile_route] {name}: mode={mode!r} -> {routed}", flush=True)
    return wrapped


def add_compile_mode_arg(
    parser: argparse.ArgumentParser,
    default: str = "default",
) -> None:
    """Add the uniform ``--compile-mode`` knob to *parser*.

    Default ``"default"`` = compiled, per the 2026-08-19 benchmark
    standard; ``"eager"`` keeps the pre-standard A/B path one flag away.
    """
    parser.add_argument(
        "--compile-mode",
        choices=["eager", "default", "max-autotune-no-cudagraphs"],
        default=default,
        help=(
            "LBM step routing through tensorlbm.compile_utils "
            "(new standard: 'default'; 'eager' = None passthrough for A/B)"
        ),
    )


def compile_mode_from_args(args: argparse.Namespace) -> str | None:
    """Return the canonical mode for ``args.compile_mode`` (validates it)."""
    return normalize_compile_mode(getattr(args, "compile_mode", "default"))
