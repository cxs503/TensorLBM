"""Pin the compute-resource budget *before* tuning a kernel, and prove it.

Why this module exists
----------------------
Kernel tuning (which collision implementation, which ``torch.compile`` mode,
which tile/block size) is only meaningful if the *resource budget* the kernel
runs with is fixed while the tuning happens.  If the budget drifts, the winner
of a tuning sweep is a property of the budget, not of the kernel -- two entry
scripts that inherit different budgets record two different "tuned" choices,
and neither result is attributable.

The practice is taken from a mature OSS code base: ``AminZar69/CudaPhaseField``
pins ``maxregcount:64`` in the Makefile (``Makefile:61``/``Makefile:64``,
commit d6181e6) for **every** build mode and documents it as a *permanent
constraint* rather than a diagnostic switch, after its 18-population
``collision_h`` kernel overflowed the register budget and silently failed the
launch.  The register cap is therefore not a tuning knob -- it is pinned
*before* launch configuration is tuned, so that occupancy is a constant of the
experiment instead of a hidden variable.

This repo has no register file on its CPU path and no GPU in CI, so the same
discipline is expressed over the budget knobs that *do* drift here:

* ``threads``     -- the CPU parallelism budget (this repo's entry scripts set
  1 / 16 / ``OMP_NUM_THREADS`` / leave the torch default, i.e. it drifts);
* ``dtype``       -- fp32/fp64 changes both the arithmetic and the memory
  footprint, exactly like a register cap changes occupancy;
* ``compile_modes`` -- the whitelist of :mod:`tensorlbm.compile_utils` modes a
  measurement is allowed to use;
* ``maxnreg``     -- the GPU half of the practice: the ``maxnreg`` compile
  option that is the Triton/PTX spelling of ``nvcc -maxrregcount``.  It is part
  of the pin so that a tuned GPU result cannot be recorded without the register
  cap it was obtained under.

Usage
-----
>>> pin = ResourcePin(label="bench", threads=8)
>>> with pinned_resources(pin) as audit:
...     assert_pinned(pin)          # aborts if anything drifted
...     ...                          # measure / tune here
>>> audit["pin_sha256"]              # travels with every result

The audit dict is meant to be written next to the numbers (JSON), so a tuned
result without a pin hash is visibly incomplete.

Public API
----------
* :class:`ResourcePin` -- the immutable declaration (validated on construction).
* :class:`PinnedResourceViolation` -- raised when the live state drifted.
* :func:`pinned_resources` -- context manager that applies + restores the pin.
* :func:`assert_pinned` -- checks the live state against the pin.
* :func:`pin_audit` -- the provenance record (pin + observed state + sha256).
* :func:`maxnreg_launch_kwargs` -- GPU-side spelling of the pin.
* :func:`register_budget_words` / :func:`register_cap_verdict` -- the
  arithmetic that decides whether a foreign pin *value* (e.g. 64) transfers to
  one of our kernels.

This module deliberately contains no domain (flow-physics) semantics: it knows
about processes, threads, dtypes and register words only.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Iterator

import torch

__all__ = [
    "ResourcePin",
    "PinnedResourceViolation",
    "pinned_resources",
    "assert_pinned",
    "pin_audit",
    "maxnreg_launch_kwargs",
    "register_budget_words",
    "register_cap_verdict",
    "WORDS_PER_VALUE",
]

# A CUDA register is 32 bit wide.  One fp32 value occupies one register, one
# fp64 value occupies two (a double must live in an aligned register pair).
WORDS_PER_VALUE: dict[str, int] = {
    "fp32": 1,
    "float32": 1,
    "fp64": 2,
    "float64": 2,
}

_DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp64": torch.float64,
    "float64": torch.float64,
}


class PinnedResourceViolation(RuntimeError):
    """The live process state no longer matches the declared resource pin."""


@dataclass(frozen=True)
class ResourcePin:
    """An immutable declaration of the resource budget a measurement runs under.

    Parameters
    ----------
    label:
        Free-form name of the experiment the pin belongs to.
    threads:
        Budget of intra-op CPU threads.  Must be a positive int; it is applied
        with :func:`torch.set_num_threads`.
    dtype:
        Arithmetic precision the measurement must run in (``"fp32"``/``"fp64"``
        or the torch spellings).  Only used for validation and for the register
        arithmetic -- it does not cast anything for you.
    compile_modes:
        The set of :mod:`tensorlbm.compile_utils` modes the measurement is
        allowed to use.  Empty tuple means "eager only is guaranteed".
    warmup_steps, timed_steps, repeats:
        The timing protocol, pinned as part of the budget so that a recorded
        throughput always has the same definition.
    maxnreg:
        GPU register cap (``nvcc -maxrregcount`` / Triton ``maxnreg``).  ``None``
        means "not pinned", which is reported explicitly rather than hidden.
    """

    label: str
    threads: int
    dtype: str = "fp64"
    compile_modes: tuple[str | None, ...] = ()
    warmup_steps: int = 10
    timed_steps: int = 40
    repeats: int = 5
    maxnreg: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("ResourcePin.label must be a non-empty string")
        if not isinstance(self.threads, int) or isinstance(self.threads, bool) or self.threads < 1:
            raise ValueError(f"ResourcePin.threads must be a positive int, got {self.threads!r}")
        if self.dtype not in _DTYPES:
            raise ValueError(f"unknown dtype {self.dtype!r}; expected one of {sorted(_DTYPES)}")
        for name in ("warmup_steps", "timed_steps", "repeats"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"ResourcePin.{name} must be a positive int, got {value!r}")
        if self.maxnreg is not None and (not isinstance(self.maxnreg, int) or self.maxnreg < 16):
            raise ValueError(
                f"ResourcePin.maxnreg must be None or an int >= 16 (CUDA minimum), got {self.maxnreg!r}"
            )
        object.__setattr__(self, "compile_modes", tuple(self.compile_modes))

    # -- serialisation ---------------------------------------------------
    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "threads": self.threads,
            "dtype": self.dtype,
            "compile_modes": list(self.compile_modes),
            "warmup_steps": self.warmup_steps,
            "timed_steps": self.timed_steps,
            "repeats": self.repeats,
            "maxnreg": self.maxnreg,
        }

    def sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def replace(self, **kw: Any) -> "ResourcePin":
        return replace(self, **kw)


def maxnreg_launch_kwargs(pin: ResourcePin) -> dict[str, int]:
    """Return the Triton launch kwargs that realise ``pin.maxnreg`` on a GPU.

    This is the translation of ``nvcc -maxrregcount=N`` into the Triton/PTX
    world (``maxnreg`` is a per-kernel compile option).  An unpinned pin
    returns ``{}`` -- and the caller is expected to record that fact rather
    than silently tuning without a cap.
    """
    return {} if pin.maxnreg is None else {"maxnreg": int(pin.maxnreg)}


def _apply_interop_threads(n: int) -> None:
    """Best-effort intra-op/inter-op thread pinning.

    ``torch.set_num_interop_threads`` may only be called before any parallel
    work has started; the resulting ``RuntimeError`` is tolerated because the
    intra-op budget (which is what the timing depends on) is always applied.
    """
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(n)


@contextlib.contextmanager
def pinned_resources(pin: ResourcePin) -> Iterator[dict[str, Any]]:
    """Apply *pin* for the duration of the block, yield its audit, then restore.

    The audit is written to the caller before the block exits (``with ... as
    audit``), so a measurement can store it even if the block raises.
    """
    if not isinstance(pin, ResourcePin):
        raise TypeError(f"pinned_resources expects a ResourcePin, got {type(pin).__name__}")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(pin.threads)
    _apply_interop_threads(1)
    audit = pin_audit(pin)
    try:
        yield audit
    finally:
        torch.set_num_threads(previous_threads)


def assert_pinned(pin: ResourcePin) -> None:
    """Raise :class:`PinnedResourceViolation` if the live state drifted from *pin*."""
    live = torch.get_num_threads()
    if live != pin.threads:
        raise PinnedResourceViolation(
            f"resource pin {pin.label!r} declares threads={pin.threads} but the "
            f"measurement is running with torch.get_num_threads()={live}; "
            "a tuned result obtained under a drifting budget is not attributable"
        )


def pin_audit(pin: ResourcePin, **extra: Any) -> dict[str, Any]:
    """Provenance record for one measurement: the pin, the live state, a sha256.

    ``extra`` (e.g. ``step_time_s``, ``collision_impl``) is merged into the same
    record so that every number travels with the pin it was obtained under.
    """
    record: dict[str, Any] = {
        "pin": pin.to_dict(),
        "pin_sha256": pin.sha256(),
        "observed": {
            "torch_num_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "maxnreg_pinned": pin.maxnreg is not None,
        },
        "argv": list(sys.argv),
    }
    record.update(extra)
    return record


def register_budget_words(live_arrays: int, precision: str = "fp64") -> int:
    """Register words (32 bit each) needed to keep *live_arrays* values live.

    This is the arithmetic behind a register cap such as the OSS
    ``maxregcount:64``: a kernel that needs more live values than the cap allows
    spills them to local memory (or, in the OSS case, failed to launch).
    """
    if not isinstance(live_arrays, int) or isinstance(live_arrays, bool) or live_arrays < 0:
        raise ValueError(f"live_arrays must be a non-negative int, got {live_arrays!r}")
    if precision not in WORDS_PER_VALUE:
        raise ValueError(f"unknown precision {precision!r}; expected one of {sorted(WORDS_PER_VALUE)}")
    return live_arrays * WORDS_PER_VALUE[precision]


def register_cap_verdict(
    live_arrays: int,
    precision: str = "fp64",
    cap_words: int = 64,
    *,
    reserve_words: int = 16,
) -> dict[str, Any]:
    """Decide whether a foreign register cap transfers to a kernel of ours.

    Parameters
    ----------
    live_arrays:
        Number of scalar values the fused kernel must keep live simultaneously
        (e.g. populations + equilibria + source arrays).
    precision:
        ``"fp32"``/``"fp64"`` -- the precision the arrays are stored in.
    cap_words:
        The cap to test, in 32-bit register words (the OSS value is 64).
    reserve_words:
        Words a realistic kernel also needs for pointers, indices, loop
        counters and the reduction accumulator.  Pinned at 16 here (declared,
        not fitted), so the verdict is conservative for *every* kernel rather
        than tuned per kernel.

    Returns
    -------
    dict with ``required_words``, ``required_with_reserve``, ``cap_words``,
    ``fits`` and ``overflow_words`` (0 when it fits).
    """
    required = register_budget_words(live_arrays, precision)
    total = required + int(reserve_words)
    return {
        "live_arrays": live_arrays,
        "precision": precision,
        "required_words": required,
        "reserve_words": int(reserve_words),
        "required_with_reserve": total,
        "cap_words": int(cap_words),
        "fits": total <= int(cap_words),
        "overflow_words": max(0, total - int(cap_words)),
    }