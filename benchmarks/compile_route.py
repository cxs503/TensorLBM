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

Compile-time device support & the automatic eager fallback
----------------------------------------------------------
``mode='default'`` is a **real** compilation on CUDA (the production
path) and never needs the fallback below.  On Hygon/SDAA, however,
``torch.compile`` runs on ``teco_inductor`` (Torch-SDAA **3.1.1a3**,
``torch_sdaa._inductor.codegen.teco``) whose backend is still incomplete
for the whole LBM step chain.  :func:`_enable_sdaa_inductor` shims the
four known backend defects (L1 device-codegen registration, L2 stale
``Reduction`` binding, L3 ``TecoScheduling._sizes``, L4 ``load_count``
contract), but an upstream fifth one remains — L5: teco's ``EXPAND``
fallback path indexes ``reduction_shape`` out of range — and it is *not*
worth hacking further (it lives in the installed package and is a moving
target).  So SDAA cannot compile this chain today.

:func:`route_step` therefore no longer lets that abort a benchmark:
for a compiled mode it returns a :class:`_CompileWithEagerFallback`
instead of the raw ``torch.compile`` wrapper.  The **first** call tries
the compiled path; if it raises ``InductorError`` / ``AssertionError`` /
``IndexError`` / ``RuntimeError`` this prints a loud, greppable fallback
banner (exception + reason) and runs **every subsequent step eager** —
bit-for-bit the ``--compile-mode eager`` trajectory, so the in-registry
cases stay reproducible on SDAA.  The one-shot decision is exposed as
``compile_status`` (``"compiled"`` / ``"eager_fallback"`` /
``"eager"``) and ``compile_mode_effective``; run scripts must persist
them to ``result.json`` (``compile_mode_effective``) so a run that
silently fell back is distinguishable from a truly compiled one.  On
CUDA the same wrapper is a no-op — the first call succeeds and stays
compiled.

``compile_utils`` itself is consumed as-is; the fallback lives entirely
here and changes no numerics.
"""

from __future__ import annotations

import argparse
import functools
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


def _enable_sdaa_inductor() -> str | None:
    """Make ``torch.compile`` usable on Hygon/SDAA (``teco_inductor``).

    ``torch_sdaa`` ships its own inductor device codegen, but importing plain
    ``torch_sdaa`` does **not** register it: ``torch._inductor.codegen.common
    .device_codegens`` only has cpu/cuda/xpu/mps, so the first reduction in any
    compiled graph dies with

        AssertionError: assert scheduling_ctor
        (torch._inductor.codegen.common.get_backend_features)

    Importing ``torch_sdaa._inductor`` registers the ``sdaa`` device codegen and
    the ``teco_inductor`` backend.  One more binding is left stale after that
    import: ``torch._inductor.lowering`` did ``from .ir import Reduction`` at
    module-import time, so its module-global ``Reduction`` still points at the
    *original* class while ``torch._inductor.ir.Reduction`` was replaced by
    ``SDAAReduction`` (whose ``num_splits`` returns ``(DEFAULT, 1)`` and avoids
    ``DeviceProperties.create`` — the generic one raises
    ``AttributeError: 'torch_sdaa._C._SDAADeviceProperties' object has no
    attribute 'multi_processor_count'``).  Re-bind the stale reference so the
    lowering path picks up the SDAA reduction.

    This is a no-op on CUDA/CPU hosts and changes no numerics: the SDAA
    reduction only picks a non-split hint.  Returns a short tag when the shim
    was applied, else ``None``.
    """
    try:
        import torch_sdaa  # noqa: F401,PLC0415
    except Exception:
        return None
    try:
        import torch_sdaa._inductor  # noqa: F401,PLC0415
    except Exception:
        return None

    import torch._inductor.ir as _ir  # noqa: PLC0415

    if not hasattr(_ir, "Reduction"):
        return None
    for _mod_name in ("torch._inductor.lowering", "torch._inductor.decomposition"):
        try:
            mod = __import__(_mod_name, fromlist=["Reduction"])
        except Exception:
            continue
        if getattr(mod, "Reduction", None) is not None and mod.Reduction is not _ir.Reduction:
            mod.Reduction = _ir.Reduction

    # Second stale-/broken-binding: TecoScheduling.can_fuse_horizontal (teco.py)
    # crashes on fused pointwise groups ("'FusedSchedulerNode' object has no
    # attribute '_sizes'") while comparing reduction shapes.  The class keeps
    # the stock inductor implementation as ``_can_fuse_horizontal_impl`` (whose
    # call is commented out at the end of the override) — restore it.
    try:
        from torch_sdaa._inductor.codegen.teco import (
            TecoScheduling as _TecoScheduling,
        )

        if hasattr(_TecoScheduling, "_can_fuse_horizontal_impl"):
            _TecoScheduling.can_fuse_horizontal = _TecoScheduling._can_fuse_horizontal_impl
    except Exception:
        pass

    # Fourth stale-state bug: ``TecoKernel.infer_expand_dim`` (teco.py:2073)
    # asserts that the per-load key ``f"{name}_{self.load_count}"`` exists in
    # the *current node's* kernel meta.  ``TecoKernelProxy.codegen_nodes``
    # refreshes ``kernel.kernel_meta`` once per fused pointwise node
    # (``run_scheduler_node([node])`` — its keys restart at 0 for every node)
    # but never resets ``kernel.load_count``, so from the second fused node
    # onwards the running counter walks past the fresh meta's keys and the
    # assert fires with a bare ``AssertionError``:
    #
    #     File ".../torch_sdaa/_inductor/codegen/teco.py", line 2073
    #         assert meta_key in meta.expand_dims
    #     torch._inductor.exc.InductorError: AssertionError:
    #
    # Any LBM chain whose compiled graph fuses ≥2 pointwise nodes in ONE
    # teco kernel (e.g. collide→NoDynamics→BB→stream→far-field) hits this.
    # Fix without touching the installed package: re-anchor ``load_count`` to
    # the ``kernel_meta`` object that is about to be consumed — a fresh
    # ``kernel_meta`` identity marks a new node, and the counter must restart
    # at -1 exactly as ``TecoKernel.__init__`` does, so it lines up 1:1
    # again with that node's meta keys.  Numerics are unaffected (the keys
    # only select the pre-computed expand/broadcast shape hints).
    try:
        from torch_sdaa._inductor.codegen.teco import TecoKernel as _TecoKernel

        _orig_infer_expand_dim = _TecoKernel.infer_expand_dim

        @functools.wraps(_orig_infer_expand_dim)
        def _reanchor_infer_expand_dim(self, name):  # type: ignore[no-untyped-def]
            meta = getattr(self, "kernel_meta", None)
            if getattr(self, "_compile_route_meta", None) is not meta:
                self._compile_route_meta = meta
                self.load_count = -1
            return _orig_infer_expand_dim(self, name)

        _TecoKernel.infer_expand_dim = _reanchor_infer_expand_dim
    except Exception:
        pass
    return "sdaa-teco_inductor"


_SDAA_INDUCTOR_TAG = _enable_sdaa_inductor()


ensure_tensorlbm_importable()

from tensorlbm.compile_utils import (  # noqa: E402  (needs the path bootstrap above)
    compile_step,
    validate_compile_mode,
)

#: Exceptions that mean "the inductor backend could not compile/execute this
#: graph on this device" and must trigger the one-shot eager fallback.  The
#: first three are chain-internal failures (teco asserts / shape-index bugs);
#: ``RuntimeError`` covers ``torch._dynamo``/``torch._inductor`` wrappers, and
#: ``InductorError`` (when this torch exposes it) is added explicitly.
try:  # torch-version-dependent location — only used as a fallback trigger
    from torch._inductor.exc import InductorError as _InductorError  # noqa: PLC0415
except Exception:  # pragma: no cover - defensive across torch layouts
    _InductorError = None

_FALLBACK_EXC: tuple[type[BaseException], ...] = tuple(
    exc
    for exc in (AssertionError, IndexError, RuntimeError, _InductorError)
    if isinstance(exc, type)
)

__all__ = [
    "EAGER_CLI_SPELLINGS",
    "ensure_tensorlbm_importable",
    "normalize_compile_mode",
    "route_step",
    "compile_status_of",
    "add_compile_mode_arg",
    "compile_mode_from_args",
]

StepFn = Callable[..., Any]


class _CompileWithEagerFallback:
    """Callable LBM step: try ``torch.compile`` once, fall back to eager forever.

    The audited escape hatch that keeps the compiled benchmark path
    reproducible on hosts whose inductor backend is incomplete
    (Hygon/SDAA ``teco_inductor``, Torch-SDAA 3.1.1a3 — see the module
    docstring).  The object is callable **exactly** like the whole-step
    function it replaces (``f -> f'``, plus any per-step tensor outputs),
    so ``route_step`` callers need no change.

    Semantics (one-shot, auditable):

    * **First call** — try the compiled wrapper.  On success the object
      stays compiled for every later call (Dynamo's lazily-built graphs
      are reused; no extra per-step overhead).
    * **First call raises** one of :data:`_FALLBACK_EXC`
      (``InductorError`` / ``AssertionError`` / ``IndexError`` /
      ``RuntimeError``) — print a loud, greppable fallback banner
      (exception + reason) and run *this* step eagerly.  The compiled
      attempt died during lowering, i.e. **before** producing an output,
      so re-running the pure ``f -> f'`` step on the untouched input is
      exact.  Every later call is eager: bit-for-bit the
      ``--compile-mode eager`` trajectory.
    * The decision is frozen after the first call and recorded in
      :attr:`compile_status` (``"compiled"`` / ``"eager_fallback"``),
      :attr:`compile_status_reason` and :attr:`compile_mode_effective`.
      Read them **after** the stepping loop (via :func:`compile_status_of`)
      and persist to ``result.json``.
    """

    def __init__(self, compiled: StepFn, eager: StepFn, *, name: str, mode: str) -> None:
        self._compiled = compiled
        self._eager = eager
        self._name = name
        self._mode = mode
        self._attempted = False
        self._fallback = False
        #: ``"compiled"`` until/unless the first call falls back.
        self.compile_status = "compiled"
        #: ``None`` or ``"<ExcType>: <message>"`` once a fallback happened.
        self.compile_status_reason: str | None = None
        # Keep the wrapper as introspectable as the function it hides.
        self.__name__ = getattr(eager, "__name__", name)
        self.__doc__ = getattr(eager, "__doc__", type(self).__doc__)
        self.__wrapped__ = eager

    @property
    def compile_mode_effective(self) -> str:
        """Mode actually used for the run: requested mode, or ``"eager"``."""
        return "eager" if self._fallback else self._mode

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._fallback:
            return self._eager(*args, **kwargs)
        if not self._attempted:
            self._attempted = True
            try:
                out = self._compiled(*args, **kwargs)
            except _FALLBACK_EXC as exc:  # noqa: BLE001 - deliberate catch-all
                self._fallback = True
                self.compile_status = "eager_fallback"
                self.compile_status_reason = f"{type(exc).__name__}: {exc}"
                self._emit_banner(exc)
                return self._eager(*args, **kwargs)
            self.compile_status = "compiled"
            return out
        # First call compiled fine: stay on the (lazily re-compiled) path.
        return self._compiled(*args, **kwargs)

    def _emit_banner(self, exc: BaseException) -> None:
        bar = "!" * 78
        tag = f" (teco/inductor tag: {_SDAA_INDUCTOR_TAG})" if _SDAA_INDUCTOR_TAG else ""
        print(
            f"\n{bar}\n"
            f"[compile_route] !!! torch.compile FAILED on this device -> EAGER FALLBACK\n"
            f"[compile_route]   step    : {self._name}\n"
            f"[compile_route]   mode    : {self._mode!r}{tag}\n"
            f"[compile_route]   reason  : {type(exc).__name__}: {exc}\n"
            f"[compile_route]   effect  : this step + every later step run EAGER\n"
            f"[compile_route]             (bitwise-identical to --compile-mode eager)\n"
            f"[compile_route]   detail  : see compile_route.py docstring,"
            f" 'Compile-time device support & the automatic eager fallback'\n"
            f"{bar}\n",
            flush=True,
        )


def _tag_eager_step(step_fn: StepFn) -> StepFn:
    """Tag the eager passthrough step with a static ``"eager"`` status.

    ``route_step(mode=None/"eager")`` still returns *step_fn itself* (no
    wrapper, byte-identical behaviour); the tag is best-effort metadata so
    :func:`compile_status_of` reports a uniform result for eager and
    compiled runs alike.
    """
    try:
        step_fn.compile_status = "eager"  # type: ignore[attr-defined]
        step_fn.compile_mode_effective = "eager"  # type: ignore[attr-defined]
        step_fn.compile_status_reason = None  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - e.g. non-settable callable
        pass
    return step_fn


def compile_status_of(step: Any) -> dict[str, Any]:
    """Return the routing outcome of *step* as a JSON-ready dict.

    Keys: ``compile_status`` (``"eager"`` / ``"compiled"`` /
    ``"eager_fallback"``), ``compile_mode_effective`` (the mode actually
    used: the requested compile mode, or ``"eager"``), and
    ``compile_status_reason`` (``None``, or the exception string when a
    fallback happened).

    Works for both the eager passthrough (tagged by :func:`_tag_eager_step`)
    and the :class:`_CompileWithEagerFallback` wrapper.  Call it **after**
    the stepping loop so the one-shot fallback decision is final.
    """
    return {
        "compile_status": getattr(step, "compile_status", None),
        "compile_mode_effective": getattr(step, "compile_mode_effective", None),
        "compile_status_reason": getattr(step, "compile_status_reason", None),
    }


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
        else a :class:`_CompileWithEagerFallback` wrapping the
        ``torch.compile`` graph.  The returned object carries
        ``compile_status`` / ``compile_mode_effective`` /
        ``compile_status_reason`` (query after the run with
        :func:`compile_status_of`) so a device fallback is auditable.
    """
    canonical = normalize_compile_mode(mode)
    compiled = compile_step(
        step_fn,
        canonical,
        warmup_hint=warmup_hint or f"benchmark {name!r}: one whole-step graph per grid shape",
    )

    if canonical is None:
        # Eager: return the raw step (no wrapper, byte-identical), just tagged.
        routed_step: StepFn = _tag_eager_step(step_fn)
    else:
        # Compiled: guard the first call so an unsupported inductor backend
        # (SDAA teco, see module docstring) degrades to eager instead of
        # aborting the benchmark.
        routed_step = _CompileWithEagerFallback(compiled, step_fn, name=name, mode=canonical)

    if not quiet:
        routed = (
            "eager (compile_step passthrough)"
            if canonical is None
            else f"torch.compile(mode={canonical!r}) + auto eager fallback"
        )
        tagged = f"{routed} [{_SDAA_INDUCTOR_TAG}]" if _SDAA_INDUCTOR_TAG else routed
        print(f"[compile_route] {name}: mode={mode!r} -> {tagged}", flush=True)
    return routed_step


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
