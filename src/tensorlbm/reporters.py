# Reporter/callback protocol for TensorLBM step loops.
#
# Derived in part from lettuce (MIT License, Copyright (c) 2019 Andreas
# Kraemer and lettucecfd/lettuce contributors).  The ``interval`` +
# ``__call__`` reporter contract, the MLUPS throughput convention, and the
# reporter-triggered early-stop idea follow lettuce's
# ``lettuce/_simulation.py`` (``Reporter``, ``Simulation.__call__``,
# ``BreakableSimulation``) and ``lettuce/ext/_reporter/``.  Adapted for
# TensorLBM: reporters receive a lightweight :class:`StepContext` built
# around a raw population tensor instead of a ``Simulation`` object, the
# dispatcher (not the reporter) evaluates the interval, and early stop is
# an explicit ``ctx.stop`` flag instead of mutating the step counter.
#
# lettuce license (MIT):
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
"""Unified reporter/callback protocol for TensorLBM time-step loops.

Closes the "solver → diagnostics/data has no uniform hook point" gap: the
step executors (:class:`tensorlbm.lbm_step.LBMStepExecutor`,
:class:`tensorlbm.triton_fused.TritonFusedSolver3D`) accept a list of
reporters and call them from their multi-step loops at a single,
well-defined point — *after* a step completes, before the next one starts.

Design (lettuce-derived, see module header for attribution):

* A **reporter** is anything with an ``interval: int`` attribute and a
  ``__call__(ctx)`` method (:class:`Reporter` protocol, structural).
* The **dispatcher** (:func:`dispatch`) — not the reporter — evaluates the
  interval: a reporter fires at every positive multiple of its interval,
  counted in completed steps.  ``run(f, 100)`` with ``interval=25`` fires
  exactly 4 times (steps 25, 50, 75, 100).  Unlike lettuce, nothing fires
  at step 0; sample the initial state explicitly if you need it.
* The **context** (:class:`StepContext`) is a deliberately small, solver
  agnostic record built around the population tensor ``f`` — TensorLBM
  steps are tensor-in/tensor-out functions, not methods on a Simulation
  object, so the context carries the handles a reporter needs (step count,
  ``f``, per-step diagnostics, lattice name, cell count, an optional
  macroscopic helper and unit converter) instead of a live simulation.
* **Early stop** follows lettuce's ``BreakableSimulation`` idea without
  its step-counter mutation: a reporter sets ``ctx.stop = True`` and the
  host loop breaks after the dispatch returns (see
  :class:`EarlyStopReporter`).

Zero-overhead contract: host loops only enter the reporting path when at
least one reporter is registered; with ``reporters=[]`` (the default) the
original fast path runs unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path
    from typing import TextIO

    from .data.catalog import FieldDataCatalog

__all__ = [
    "Reporter",
    "ReporterBase",
    "StepContext",
    "dispatch",
    "CallbackReporter",
    "ThroughputReporter",
    "EarlyStopReporter",
    "FieldSampleReporter",
]


# ---------------------------------------------------------------------------
# Step context
# ---------------------------------------------------------------------------


class StepContext:
    """Minimal, solver-agnostic view of one completed time step.

    The host loop mutates a single instance in place (``step``, ``f``,
    ``diag``, ``stop``) so reporting adds no per-step allocations beyond
    the dispatch itself.  Reporters must not hold references to
    ``macroscopic()`` results after returning — the executor-backed helper
    writes into reused buffers.

    Attributes
    ----------
    step:
        Number of completed steps (1 after the first step of the run; the
        counter persists across ``run()`` calls of the same executor, so
        interval cadence and export step labels stay monotonic).
    f:
        Population tensor ``(Q, nz, ny, nx)`` *after* the completed step.
        It is a fresh tensor per step, so keeping a reference is safe.
    diag:
        Per-step diagnostics dict of the host executor (may contain
        ``"forces"``, ``"max_speed"``, ``"mean_rho"``; empty on hosts that
        produce none).
    lattice:
        Lattice name, ``"D3Q19"`` / ``"D3Q27"`` (informational; the
        macroscopic fallback dispatches on ``f.shape[0]`` anyway).
    num_cells:
        Lattice-site count ``nz * ny * nx`` (MLUPS denominator).
    num_steps:
        Steps requested from the current ``run()`` call (``len(diags)``
        may be smaller when a reporter stopped the run early).
    macroscopic_fn:
        Optional host hook ``f -> (rho, ux, uy, uz)`` (reuses the
        executor's pre-allocated buffers); :meth:`macroscopic` falls back
        to the standalone lattice macroscopic when absent.
    units:
        Optional unit converter (e.g.
        :class:`~tensorlbm.unit_converter.LBMUnitConverter`) for
        physical-unit reporting; reporters must tolerate ``None``.
    state:
        Free-form scratch dict shared by all reporters of one host loop
        (cross-reporter communication without globals).
    stop:
        Set to ``True`` by a reporter to stop the host loop after this
        dispatch (lettuce ``BreakableSimulation`` equivalent).  The host
        resets it to ``False`` before every dispatch.
    """

    __slots__ = (
        "step",
        "f",
        "diag",
        "lattice",
        "num_cells",
        "num_steps",
        "macroscopic_fn",
        "units",
        "state",
        "stop",
    )

    def __init__(
        self,
        *,
        step: int,
        f: torch.Tensor,
        diag: Mapping[str, Any] | None = None,
        lattice: str = "D3Q19",
        num_cells: int | None = None,
        num_steps: int = 0,
        macroscopic_fn: Callable[[torch.Tensor], tuple[Any, ...]] | None = None,
        units: object | None = None,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.step = int(step)
        self.f = f
        self.diag: Mapping[str, Any] = diag if diag is not None else {}
        self.lattice = lattice
        if num_cells is None:
            spatial = tuple(f.shape[1:])
            num_cells = 1
            for dim in spatial:
                num_cells *= int(dim)
        self.num_cells = int(num_cells)
        self.num_steps = int(num_steps)
        self.macroscopic_fn = macroscopic_fn
        self.units = units
        self.state: dict[str, Any] = state if state is not None else {}
        self.stop = False

    def macroscopic(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(rho, ux, uy, uz)`` for the current ``f``.

        Uses the host-provided hook when present (executor pre-allocated
        buffers), otherwise the standalone macroscopic of the matching
        lattice.  Result validity: only until the next call.
        """
        if self.macroscopic_fn is not None:
            return self.macroscopic_fn(self.f)
        q = int(self.f.shape[0])
        if q == 19:
            from .d3q19 import macroscopic3d

            return macroscopic3d(self.f)
        if q == 27:
            from .d3q27 import macroscopic27

            return macroscopic27(self.f)
        raise ValueError(f"cannot infer macroscopic for Q={q}; pass macroscopic_fn")

    def time_phys(self) -> float | None:
        """Physical time of the current step via ``units``, or ``None``."""
        converter = self.units
        if converter is None or not hasattr(converter, "steps_to_phys_time"):
            return None
        return float(converter.steps_to_phys_time(self.step))


# ---------------------------------------------------------------------------
# Protocol + dispatcher
# ---------------------------------------------------------------------------


@runtime_checkable
class Reporter(Protocol):
    """Structural protocol: ``interval`` attribute + ``__call__(ctx)``."""

    interval: int

    def __call__(self, ctx: StepContext) -> None: ...  # pragma: no cover


class ReporterBase(ABC):
    """Convenience base class validating the interval (lettuce-style).

    Inheriting from this is optional — any object with ``interval`` and
    ``__call__(ctx)`` satisfies :class:`Reporter` structurally.
    """

    def __init__(self, interval: int = 1) -> None:
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError(f"interval must be an int >= 1, got {interval!r}")
        self.interval = interval

    @abstractmethod
    def __call__(self, ctx: StepContext) -> None: ...  # pragma: no cover


def dispatch(ctx: StepContext, reporters: Sequence[Reporter]) -> None:
    """Call every reporter whose interval divides ``ctx.step``.

    A reporter with ``interval=25`` fires at steps 25, 50, 75, … (positive
    multiples only — step 0 never fires).  Reporters missing an
    ``interval`` attribute are treated as ``interval=1``; invalid
    intervals are clamped to 1 defensively.
    """
    step = ctx.step
    for reporter in reporters:
        interval = getattr(reporter, "interval", 1)
        if not isinstance(interval, int) or isinstance(interval, bool) or interval < 1:
            interval = 1
        if step % interval == 0:
            reporter(ctx)


# ---------------------------------------------------------------------------
# Built-in reporters
# ---------------------------------------------------------------------------


class CallbackReporter(ReporterBase):
    """Wrap any ``callable(ctx)`` as a reporter (escape hatch).

    Args:
        callback: called with the :class:`StepContext` at every firing.
            Callables that ignore their argument are also accepted.
        interval: firing cadence in completed steps (default: every step).
        name: label used in exception messages and reprs.
    """

    def __init__(
        self,
        callback: Callable[[StepContext], Any],
        interval: int = 1,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(interval)
        if not callable(callback):
            raise TypeError(f"callback must be callable, got {type(callback).__name__}")
        self.callback = callback
        self.name = name or getattr(callback, "__name__", repr(callback))
        self.calls = 0

    def __call__(self, ctx: StepContext) -> None:
        self.calls += 1
        self.callback(ctx)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CallbackReporter({self.name!r}, interval={self.interval})"


class ThroughputReporter(ReporterBase):
    """Lattice-site updates per second (MLUPS), lettuce ``Simulation.__call__`` convention.

    MLUPS = (lattice updates) / 1e6 / elapsed seconds, measured between
    consecutive firings of this reporter (each firing first synchronises
    the tensor's device, so async CUDA queues cannot fake throughput).
    The first firing only establishes the baseline clock.

    Args:
        interval: cadence; with ``interval=25`` over 100 steps the reporter
            records one MLUPS sample covering steps 26–50, 51–75, 76–100
            (plus the baseline at 25).
        num_cells: lattice-site count; inferred from the first context
            when omitted.
        out: optional writable (e.g. ``sys.stdout``) receiving one
            ``step elapsed_s mlups`` line per sample; silent by default.
        sync: device-synchronise before reading the clock (default True).

    After the run: ``records`` holds ``(step, elapsed_s, mlups)`` tuples,
    ``last_mlups`` the most recent sample, ``mean_mlups`` their mean
    (``None`` until at least two firings).
    """

    def __init__(
        self,
        interval: int = 1,
        *,
        num_cells: int | None = None,
        out: TextIO | None = None,
        sync: bool = True,
    ) -> None:
        super().__init__(interval)
        self.num_cells = num_cells
        self.out = out
        self.sync = bool(sync)
        self.records: list[tuple[int, float, float]] = []
        self._t_last: float | None = None
        self._step_last: int | None = None

    # -- timing helpers ----------------------------------------------------

    def _now(self) -> float:
        return perf_counter()

    def _synchronize(self, f: torch.Tensor) -> None:
        if not self.sync:
            return
        device_type = f.device.type
        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "sdaa" and hasattr(torch, "sdaa"):
            torch.sdaa.synchronize()

    # -- reporting ---------------------------------------------------------

    @property
    def last_mlups(self) -> float | None:
        return self.records[-1][2] if self.records else None

    @property
    def mean_mlups(self) -> float | None:
        if not self.records:
            return None
        return sum(record[2] for record in self.records) / len(self.records)

    def __call__(self, ctx: StepContext) -> None:
        self._synchronize(ctx.f)
        now = self._now()
        num_cells = self.num_cells if self.num_cells is not None else ctx.num_cells
        if self._t_last is None:
            self._t_last = now
            self._step_last = ctx.step
            return
        elapsed = now - self._t_last
        steps = ctx.step - self._step_last
        self._t_last, self._step_last = now, ctx.step
        if elapsed <= 0.0 or steps <= 0:
            return
        mlups = steps * num_cells / 1e6 / elapsed
        self.records.append((ctx.step, elapsed, mlups))
        if self.out is not None:
            print(f"{ctx.step} {elapsed:.3f} {mlups:.1f}", file=self.out)


class EarlyStopReporter(ReporterBase):
    """Steady-state detector: stop the run when a monitored quantity settles.

    The TensorLBM counterpart of lettuce's ``BreakableSimulation`` pattern
    (reporter-driven early termination) carrying the cp_measurement idea
    as a reusable protocol citizen: monitor any scalar, compare its change
    between consecutive firings against a threshold, and stop once the
    change has stayed below the threshold for ``patience`` consecutive
    firings.

    Args:
        monitor: ``callable(ctx) -> float`` or a key of ``ctx.diag``
            (e.g. ``"max_speed"`` when the host computes it).
        threshold: tolerated absolute change (``relative=True``: tolerated
            ``|Δ| / max(|previous|, eps)``).
        patience: consecutive below-threshold firings required to stop
            (1 = stop on the first quiet change).
        interval: cadence of the check (also the comparison lag).
        min_step: do not even record before this step (skip warm-up
            transients).
        relative: interpret *threshold* relatively.
        stop_on_nonfinite: stop immediately on NaN/Inf monitor values
            (default True — a diverging run should not export garbage).

    After the run: ``stopped`` / ``stopped_at`` / ``reason`` describe the
    termination, ``values`` holds ``(step, value)`` pairs.
    """

    def __init__(
        self,
        monitor: Callable[[StepContext], float] | str,
        threshold: float,
        *,
        patience: int = 1,
        interval: int = 1,
        min_step: int = 0,
        relative: bool = False,
        stop_on_nonfinite: bool = True,
    ) -> None:
        super().__init__(interval)
        if isinstance(monitor, str):
            self._diag_key = monitor
            self.monitor: Callable[[StepContext], float] = self._monitor_from_diag
        elif callable(monitor):
            self._diag_key = None
            self.monitor = monitor
        else:
            raise TypeError(f"monitor must be a callable or a diag key string, got {monitor!r}")
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold!r}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience!r}")
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.min_step = int(min_step)
        self.relative = bool(relative)
        self.stop_on_nonfinite = bool(stop_on_nonfinite)
        self.values: list[tuple[int, float]] = []
        self.streak = 0
        self.stopped = False
        self.stopped_at: int | None = None
        self.reason: str | None = None

    def _monitor_from_diag(self, ctx: StepContext) -> float:
        try:
            return float(ctx.diag[self._diag_key])  # type: ignore[index]
        except KeyError as error:
            raise KeyError(
                f"EarlyStopReporter monitor key {self._diag_key!r} not in ctx.diag "
                f"(available: {sorted(ctx.diag)}); use a callable monitor or a host "
                f"that produces this diagnostic"
            ) from error

    def __call__(self, ctx: StepContext) -> None:
        if ctx.step < self.min_step:
            return
        value = float(self.monitor(ctx))
        self.values.append((ctx.step, value))
        if value != value or value in (float("inf"), float("-inf")):  # NaN or Inf
            if self.stop_on_nonfinite:
                ctx.stop = True
                self.stopped = True
                self.stopped_at = ctx.step
                self.reason = f"non-finite monitor value {value!r} at step {ctx.step}"
            return
        if not self.values[:-1]:
            return  # baseline firing
        previous = self.values[-2][1]
        change = abs(value - previous)
        if self.relative:
            change /= max(abs(previous), 1e-30)
        self.streak = self.streak + 1 if change <= self.threshold else 0
        if self.streak >= self.patience:
            ctx.stop = True
            self.stopped = True
            self.stopped_at = ctx.step
            self.reason = (
                f"|change| {change:.3e} <= threshold {self.threshold:.3e} for "
                f"{self.streak} consecutive firing(s) at step {ctx.step}"
            )


class FieldSampleReporter(ReporterBase):
    """Sample fields and export them as catalog products in one hop.

    Bridges the solver loop straight into the #182 solver→data path: on
    every firing the macroscopic fields are computed from ``ctx.f``,
    written with :func:`tensorlbm.data.solver_export.save_fields_hdf5`
    into one HDF5 file per run (``/step_{step:06d}`` groups), and — when a
    catalog is given — registered with
    :func:`tensorlbm.data.solver_export.register_product` as a
    PASS-gated ``FieldDataProductR2``.  Registered product ids accumulate
    in :attr:`product_ids` (``"{run_id}:{step:06d}"``).

    Args:
        path: HDF5 file (one per run; snapshots append as groups).
        run_id: catalog run identity (unique per export run).
        case: case name (catalog metadata + grouping).
        code_sha: 40-char lowercase hex identifying the solver code
            (required by the registration gate).
        interval: sampling cadence.
        catalog: :class:`~tensorlbm.data.catalog.FieldDataCatalog` to
            register into; ``None`` writes HDF5 only.
        fields: subset of ``("rho", "ux", "uy", "uz")`` to export
            (registration requires at least ``ux``/``uy``).
        solid_mask: ``(nz, ny, nx)`` boolean tensor (or ``callable(ctx)``
            returning one) exported as the MASK array.
        extra_fields: optional ``callable(ctx) -> dict[str, field]`` of
            auxiliary float/int fields.
        metadata: extra per-snapshot attrs and catalog metadata rows
            (``collision``, ``re``, ``tau``, …; JSON-safe scalars).
        mass_tol: registration density-drift gate (lattice units).
        blob_root: NPY blob root override (default ``<h5 dir>/blobs``).

    Raises are deliberate (fail closed): non-finite fields, density drift,
    or duplicate product ids propagate out of the host loop.
    """

    _MACRO_NAMES = ("rho", "ux", "uy", "uz")

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        case: str,
        code_sha: str,
        interval: int,
        catalog: FieldDataCatalog | None = None,
        fields: Sequence[str] = ("rho", "ux", "uy", "uz"),
        solid_mask: torch.Tensor | Callable[[StepContext], torch.Tensor] | None = None,
        extra_fields: Callable[[StepContext], Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
        mass_tol: float = 1e-4,
        blob_root: str | Path | None = None,
    ) -> None:
        super().__init__(interval)
        self.path = path
        self.catalog = catalog
        self.run_id = run_id
        self.case = case
        self.code_sha = code_sha
        self.fields = tuple(fields)
        unknown = [name for name in self.fields if name not in self._MACRO_NAMES]
        if unknown:
            raise ValueError(f"fields must be a subset of {self._MACRO_NAMES}, got {unknown}")
        self.solid_mask = solid_mask
        self.extra_fields = extra_fields
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.mass_tol = float(mass_tol)
        self.blob_root = blob_root
        self.product_ids: list[str] = []
        self.exported_steps: list[int] = []

    def _resolve(
        self,
        value: torch.Tensor | Callable[[StepContext], torch.Tensor],
        ctx: StepContext,
    ) -> torch.Tensor:
        return value(ctx) if callable(value) else value

    def __call__(self, ctx: StepContext) -> None:
        from .data.solver_export import register_product, save_fields_hdf5

        rho, ux, uy, uz = ctx.macroscopic()
        available = {"rho": rho, "ux": ux, "uy": uy, "uz": uz}
        arrays: dict[str, Any] = {name: available[name] for name in self.fields}
        if self.solid_mask is not None:
            arrays["solid_mask"] = self._resolve(self.solid_mask, ctx)
        if self.extra_fields is not None:
            extra = dict(self.extra_fields(ctx))
            clash = sorted(set(extra) & {"rho", "ux", "uy", "uz", "solid_mask"})
            if clash:
                raise ValueError(f"extra_fields collide with canonical names: {clash}")
            arrays.update(extra)

        attrs: dict[str, Any] = dict(self.metadata)
        attrs.setdefault("run_id", self.run_id)
        attrs.setdefault("case", self.case)
        attrs.setdefault("lattice", ctx.lattice)
        attrs.setdefault("n_steps", ctx.num_steps)
        attrs["step"] = ctx.step

        save_fields_hdf5(self.path, arrays, attrs)
        self.exported_steps.append(ctx.step)
        if self.catalog is not None:
            registration = dict(attrs)
            registration["code_sha"] = self.code_sha
            self.product_ids.append(
                register_product(
                    self.catalog,
                    self.path,
                    registration,
                    blob_root=self.blob_root,
                    mass_tol=self.mass_tol,
                )
            )
