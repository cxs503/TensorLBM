"""Shared ``torch.compile`` support for whole-step LBM evolution loops.

Any TensorLBM example whose *whole* time step is a pure-tensor function
``f -> f'`` (collide, stream, boundary conditions, diagnostics — no host
synchronisation) can wrap that step with :func:`compile_step` and select
the mode through one validated config knob.  This module is the single
source of truth for the mode whitelist and the wrapping semantics; it
was factored out of the two SUBOFF runners
(:mod:`tensorlbm.suboff_cmk_kbc_runner` and
:mod:`tensorlbm.suboff_torch_distributed`) which had each grown an
identical copy of the pattern validation + wrapper.

Measured lessons (RTX 5090, fp32, SUBOFF production chain; see
``/nfs/wangxi/triton_bench_20260819/torch_compile_prod_path.md``)
that this module encodes — read before touching anything here:

1. **整步编译才有效**：只编 collide+stream 只拿 1.3–3.2×（BC/测力
   eager 残留吃掉大半），整步再提 1.5×。  Compile the *entire* per-step
   chain (collision + streaming + BC + force measurement), not just the
   hot collision/stream pair: partial compilation leaves the boundary
   conditions and force reduction in eager mode, which eats most of the
   win (L1→whole-step gains another ~1.5×).
2. **step 序号必须留在编译域外**，否则逐步重编译。  The step index
   (and every Python-level branch that depends on it) must stay outside
   the compiled code object, otherwise Dynamo recompiles once per step.
   The canonical pattern: compile one variant per branch pattern (e.g.
   plain step + every-N-step mass-correction step) and select the
   variant in the eager driver loop.
3. **cudagraphs 与 LBM 反馈环结构性冲突**（输出张量被下一轮覆写 →
   RuntimeError），一律拒绝。  An LBM step consumes its own output: the
   tensor returned at step *t* is the input at step *t+1*.  Cudagraph
   replay writes results into fixed static buffers, so the input of the
   next replay is silently overwritten — a structural conflict, not a
   tuning issue.  Every mode whose name contains ``cudagraphs``
   (``"reduce-overhead"``, ``"max-autotune"``) is rejected with a
   ``ValueError`` stating this reason.
4. **逐步同步破坏图**：KBC 熵二分每步 ``.item()`` 不可编译；每 N 步
   分支（质量修正）留在编译域外无害（实测整步只 3 处 graph break，
   占 4%）。  Per-step host synchronisation (``.item()``,
   data-dependent ``if`` on tensors) breaks the graph; the KBC entropy
   bisection calls ``.item()`` every step and cannot be compiled.
   Every-N-step branches kept outside the compiled domain are harmless:
   the measured whole-step chain has only 3 data-dependent graph breaks
   (two tensor-tau validation branches + the mass-correction early-out),
   together ~4% of step time.
5. 冷编译 default ~10s / max-autotune-no-cudagraphs 35–42s，同形状
   复跑走 autotune 缓存 6–19s——首次运行变慢是预期。  Cold compile
   costs ~10 s (``default``) or 35–42 s (``max-autotune-no-cudagraphs``)
   per shape; re-runs of the same shape hit the autotune cache in 6–19 s.
   The first run being slower is *expected*, not a regression.

Public API:

* :data:`ALLOWED_COMPILE_MODES` — ``None`` (eager) plus the two proven
  ``torch.compile`` modes.
* :func:`validate_compile_mode` — the shared whitelist check.
* :func:`compile_step` — the unified wrapper used by the runners.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import Any, Callable, TypeVar

import torch

__all__ = [
    "ALLOWED_COMPILE_MODES",
    "validate_compile_mode",
    "compile_step",
]

logger = logging.getLogger(__name__)

StepFn = TypeVar("StepFn", bound=Callable[..., Any])

# ``None`` means "stay eager" and is the default everywhere; the two
# string modes are the ones proven on the SUBOFF production chain
# (see module docstring, lessons 1–5).  Modes with cudagraphs are
# *structurally* incompatible with an LBM feedback loop (lesson 3) and
# are rejected by :func:`validate_compile_mode`, not merely unlisted.
ALLOWED_COMPILE_MODES: tuple[str | None, ...] = (
    None,
    "default",
    "max-autotune-no-cudagraphs",
)

_CUDAGRAPHS_REASON = (
    "modes with cudagraphs are structurally incompatible with an LBM "
    "step: the step feeds its own output back as the next input, which "
    "cudagraph replay overwrites (RuntimeError / silent corruption)"
)

# torch.compile mode names that run on cudagraphs without saying so in
# the name (``"reduce-overhead"`` is the cudagraph backend; ``"max-autotune"``
# includes cudagraphs).  Everything else cudagraph-class spells it out.
_CUDAGRAPH_CLASS_MODES = frozenset({"reduce-overhead", "max-autotune"})


def _is_cudagraph_class(mode: object) -> bool:
    return isinstance(mode, str) and ("cudagraph" in mode.lower() or mode in _CUDAGRAPH_CLASS_MODES)


def validate_compile_mode(mode: str | None) -> None:
    """Raise ``ValueError`` unless *mode* is a proven compile mode.

    Accepted values are ``None`` (eager passthrough) and the two strings
    in :data:`ALLOWED_COMPILE_MODES`.  Anything else raises
    ``ValueError``: cudagraph-class modes — the literal ``cudagraph``
    spellings plus ``"reduce-overhead"``/``"max-autotune"`` — with the
    structural reason (module docstring, lesson 3), unknown or non-string
    values with an invalid-mode reason.
    """
    if mode in ALLOWED_COMPILE_MODES:
        return
    if _is_cudagraph_class(mode):
        reason = _CUDAGRAPHS_REASON
    else:
        reason = (
            f"unknown torch.compile mode; {mode!r} is not one of the "
            f"modes proven on the LBM step chain"
        )
    raise ValueError(
        f"compile_mode must be None, 'default', or "
        f"'max-autotune-no-cudagraphs'; got {mode!r} ({reason})"
    )


def _warn_if_item_sync(step_fn: Callable[..., Any], mode: str) -> None:
    """Best-effort warning when *step_fn*'s own source contains ``.item(``.

    This can only sniff the source of the function object handed to
    :func:`compile_step` — a per-step host sync hidden one call deeper
    (e.g. the KBC entropy bisection inside ``_collide_with_sgs``) is
    invisible here.  Hence warn-only: it cannot be statically decided in
    general (see :func:`compile_step` docstring, *Reject reasons*).
    """
    try:
        source = inspect.getsource(step_fn)
    except (OSError, TypeError):
        return  # not introspectable (exec/repl/C-builtins): stay silent
    if ".item(" in source:
        warnings.warn(
            f"compile_step(mode={mode!r}): the wrapped function's source "
            f"calls '.item()' — a per-step host sync that graph-breaks "
            f"and may prevent compilation. If this is an LBM step (e.g. "
            f"an entropy-bisection collision such as KBC), reject the "
            f"combination in the caller and stay eager.",
            stacklevel=3,
        )


def compile_step(
    step_fn: StepFn,
    mode: str | None = None,
    *,
    warmup_hint: str | None = None,
) -> StepFn:
    """Wrap *step_fn* in ``torch.compile`` according to *mode*.

    This is the unified entry point for "compile my whole LBM step":

    * ``mode=None`` (the default everywhere) returns *step_fn* **as is**
      — the eager path stays byte-identical, no wrapper object, no
      behaviour change.
    * ``mode="default"`` or ``"max-autotune-no-cudagraphs"`` returns
      ``torch.compile(step_fn, mode=mode)``.
    * Any other mode — in particular cudagraph-class modes — raises
      ``ValueError`` with the structural reason before any compilation
      happens (lesson 3: the step's output is its own next input, which
      cudagraph replay overwrites).

    Args:
        step_fn: The *whole-step* function ``f -> f'`` (plus any per-step
            tensor outputs).  Whole-step, not hot-op: lesson 1.  The step
            index and every step-dependent Python branch must live in the
            eager driver loop that calls the returned function, not
            inside *step_fn* (lesson 2): compile one variant per branch
            pattern (plain / with-mass-correction / ...) and select in
            the loop.
        mode: ``None``, ``"default"`` or ``"max-autotune-no-cudagraphs"``.
        warmup_hint: Optional free-text note describing what the first
            call of the returned wrapper will cost (e.g. ``"two guarded
            graphs: plain step + every-10-step mass correction"``).  When
            given with a non-None mode it is logged once at INFO together
            with the expected cold-compile cost from lesson 5, so that a
            slow first step is recognisable as the expected warm-up
            rather than a hang.  Purely informational; never changes
            numerics or control flow.

    Returns:
        *step_fn* itself for ``mode=None``, else the ``torch.compile``
        wrapper (call it exactly like *step_fn*).

    Reject reasons (caller's responsibility)
    ----------------------------------------
    ``.item()`` / host-sync detection is **warn-only**: the source of the
    function passed in can be sniffed (:func:`_warn_if_item_sync`) but a
    sync one call deeper cannot be seen statically, so this module can
    never *decide* that a step family is un-compilable.  Callers that
    know their step chain rejects compilation must enforce it
    themselves.  The proven rejections, kept in the two SUBOFF runners:

    * **KBC collision** — the entropy bisection calls ``.item()`` every
      step (lesson 4): the single-GPU runner raises ``ValueError`` for
      ``compile_mode + KBC``, the distributed runner already rejects KBC
      wholesale.
    * **Mutual exclusivity with other step backends** — e.g.
      ``compile_mode`` and ``use_triton_step`` in
      :class:`~tensorlbm.suboff_cmk_kbc_runner.SuboffCmkKbcConfig`.
    * **Device class** — ``torch.compile`` here is CUDA-only in the
      production runners (they raise ``RuntimeError`` on non-CUDA
      devices); this module itself stays device-agnostic.
    """
    if mode is None:
        return step_fn
    validate_compile_mode(mode)
    _warn_if_item_sync(step_fn, mode)
    compiled = torch.compile(step_fn, mode=mode)
    if warmup_hint is not None:
        logger.info(
            "compile_step(mode=%r): %s — expect the first call per shape "
            "to pay the cold-compile cost (~10s default / 35-42s "
            "max-autotune-no-cudagraphs; same-shape reruns 6-19s via the "
            "autotune cache); this is expected, not a hang.",
            mode,
            warmup_hint,
        )
    return compiled
