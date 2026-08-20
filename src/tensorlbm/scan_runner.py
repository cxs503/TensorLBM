"""Parameter-sweep execution chain: DoE plan -> case runs -> catalog -> dataset.

This module is the execution layer of the AI4S scale-out data-generation
loop (roadmap B1).  It composes four existing layers into one callable
chain — no new solver, IO or registry code:

* :mod:`tensorlbm.doe` — the sample plan (LHS / Sobol / factorial / CCD);
* :mod:`tensorlbm.cases` — the named case registry: every sweep point is a
  registry-instantiated case with parameter overrides, stepped by the
  case's own verified chain (``case.make_step()``, exactly what
  :func:`tensorlbm.cases.run_case` executes);
* :mod:`tensorlbm.reporters` — the per-point run drives its loop through
  ``dispatch`` / ``StepContext`` so a :class:`FieldSampleReporter` lands
  every snapshot as a PASS-gated catalog product in one hop, a
  :class:`ThroughputReporter` records honest MLUPS and an optional
  :class:`EarlyStopReporter` truncates converged points;
* :mod:`tensorlbm.data` — the sweep finishes as a leakage-safe
  :class:`~tensorlbm.data.FieldDatasetR2` with a training-input
  fingerprint, registered in the :class:`~tensorlbm.data.catalog.FieldDataCatalog`
  with ``plan -> run -> product -> dataset`` lineage edges readable with
  ``catalog.upstream(dataset_id)``.

Execution model
---------------
:class:`ScanExecutor` schedules **case-level GPU parallelism**: a fixed
card pool (e.g. ``gpus=(0, 1)``) becomes a task table — points are dealt
to cards round-robin (deterministic), and one worker *process* per card
(``multiprocessing`` spawn context — CUDA-safe) runs its points
sequentially.  Scheduling is in-process; only the physics runs in
children.  ``gpus=()`` runs everything in-process on ``serial_device``
(CPU tests, single-GPU debug).

Every point writes ``points/<point_id>/status.json`` next to its
``fields.h5``; a point counts as done only when its status file says
``completed`` *and* all its product ids are still registered (product
existence).  Re-running an executor with ``resume=True`` (default) skips
finished points and resets half-finished ones (their partial products are
archived, the point directory recreated) — sweeps survive GPU hiccups and
can be grown in place.

Example
-------
::

    from tensorlbm.scan_runner import ScanPlan, ScanExecutor, git_code_sha

    plan = ScanPlan.generate(
        scan_id="scan_cavity_lhs32_20260820",
        case="cavity",
        variables=[
            ScanVariable(name="re", low=100.0, high=2000.0),
            ScanVariable(name="u_lid", low=0.03, high=0.10),
            ScanVariable(name="resolution", low=32.0, high=64.0),
        ],
        method="latin_hypercube",
        n_points=32,
        seed=20260820,
        steps=2500,
        snapshot_every=500,
        fixed_params={},
        code_sha=git_code_sha(),
    )
    summary = ScanExecutor(plan, "/nfs/wangxi/datasets/scan_cavity_lhs32_20260820",
                           gpus=(0, 1)).run()
    print(summary["dataset"])
"""

from __future__ import annotations

import json
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from .data.catalog import FieldDataCatalog

__all__ = [
    "SCAN_PLAN_SCHEMA",
    "EarlyStopSpec",
    "PointOutcome",
    "ScanExecutor",
    "ScanPlan",
    "ScanPoint",
    "ScanVariable",
    "assign_points_to_gpus",
    "git_code_sha",
    "open_catalog",
    "run_scan_point",
    "split_points",
]

#: Schema tag persisted in ``plan.json`` (forward compatibility).
SCAN_PLAN_SCHEMA = "tensorlbm.scan-plan/v1"

_CODE_SHA_LEN = 40
_HEX = "0123456789abcdef"


# ---------------------------------------------------------------------------
# Plan model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanVariable:
    """One swept parameter: continuous ``[low, high]`` or discrete ``levels``.

    Mirrors :class:`tensorlbm.doe.DoEVariable` with a JSON-safe surface.
    """

    name: str
    low: float = 0.0
    high: float = 1.0
    levels: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("ScanVariable.name must be a non-empty string")
        if self.levels is None:
            if not self.low < self.high:
                raise ValueError(
                    f"ScanVariable {self.name!r}: low ({self.low}) must be < high ({self.high})"
                )
        else:
            levels = tuple(float(v) for v in self.levels)
            if len(levels) < 2:
                raise ValueError(f"ScanVariable {self.name!r}: need >= 2 discrete levels")
            object.__setattr__(self, "levels", levels)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.levels is not None:
            out["levels"] = list(self.levels)
        else:
            out["low"] = float(self.low)
            out["high"] = float(self.high)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScanVariable:
        if "levels" in data:
            return cls(name=str(data["name"]), levels=tuple(float(v) for v in data["levels"]))
        return cls(name=str(data["name"]), low=float(data["low"]), high=float(data["high"]))


@dataclass(frozen=True)
class ScanPoint:
    """One sweep point: DoE row ``params`` plus stable identities."""

    index: int
    point_id: str
    run_id: str
    params: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("ScanPoint.index must be >= 0")
        for key in ("point_id", "run_id"):
            if not getattr(self, key):
                raise ValueError(f"ScanPoint.{key} must be non-empty")


@dataclass(frozen=True)
class EarlyStopSpec:
    """Optional per-point steady-state truncation (see ``EarlyStopReporter``).

    Monitors the mean in-plane speed ``<|u|>``; the sweep's uniform step
    budget stays an upper bound per point.
    """

    threshold: float = 1e-4
    patience: int = 3
    interval: int = 250
    min_step: int = 0
    relative: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> EarlyStopSpec | None:
        if data is None:
            return None
        return cls(
            threshold=float(data["threshold"]),
            patience=int(data["patience"]),
            interval=int(data["interval"]),
            min_step=int(data.get("min_step", 0)),
            relative=bool(data.get("relative", True)),
        )


def _validate_code_sha(code_sha: str) -> str:
    code_sha = str(code_sha)
    if len(code_sha) != _CODE_SHA_LEN or any(c not in _HEX for c in code_sha):
        raise ValueError(
            "code_sha must be exactly 40 lowercase hex characters "
            "(git rev-parse HEAD); it is required by the product "
            "registration gate"
        )
    return code_sha


def git_code_sha(repo: str | Path | None = None) -> str:
    """Full 40-hex HEAD sha of the running tree (for ``plan.code_sha``).."""
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            ["git", "-C", str(repo) if repo else ".", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
    except Exception as error:  # pragma: no cover - environment dependent
        raise RuntimeError(f"cannot read git HEAD sha: {error}") from error
    return _validate_code_sha(out)


@dataclass(frozen=True)
class ScanPlan:
    """A serializable parameter-sweep plan (case + space + sampler + budget).

    Attributes
    ----------
    scan_id:
        Dataset identity; also the on-disk dataset directory name and the
        dataset asset id prefix.
    case:
        Registered case name (:func:`tensorlbm.cases.get_case`).
    variables:
        Swept parameters (continuous ranges or discrete levels).
    method, n_points, seed:
        DoE sampler settings forwarded to
        :func:`tensorlbm.doe.generate_doe`.  ``n_points`` is the request;
        the realised count is ``len(points)`` (factorial/CCD derive it).
    steps:
        LBM steps per point.
    snapshot_every:
        FieldSampleReporter interval (snapshots per point).
    fixed_params:
        Extra constructor kwargs held constant across the sweep
        (e.g. ``{"span": 16}``); DoE params override them.
    early_stop:
        Optional per-point truncation spec.
    points:
        The design matrix as :class:`ScanPoint` entries (deterministic
        given ``method``/``n_points``/``seed``).
    """

    scan_id: str
    case: str
    variables: tuple[ScanVariable, ...]
    method: str
    n_points: int
    seed: int | None
    steps: int
    snapshot_every: int
    code_sha: str
    fixed_params: dict[str, Any] = field(default_factory=dict)
    early_stop: EarlyStopSpec | None = None
    points: tuple[ScanPoint, ...] = field(default_factory=tuple)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.scan_id or not isinstance(self.scan_id, str):
            raise ValueError("scan_id must be a non-empty string")
        for name, value in (
            ("steps", self.steps),
            ("snapshot_every", self.snapshot_every),
            ("n_points", self.n_points),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.snapshot_every > self.steps:
            raise ValueError(
                f"snapshot_every ({self.snapshot_every}) must be <= steps ({self.steps})"
            )
        _validate_code_sha(self.code_sha)
        if self.points and self.points[-1].index != len(self.points) - 1:
            raise ValueError("points must be indexed 0..n-1 in order")

    # -- construction ------------------------------------------------------

    @classmethod
    def generate(
        cls,
        *,
        scan_id: str,
        case: str,
        variables: Sequence[ScanVariable],
        method: str = "latin_hypercube",
        n_points: int = 16,
        seed: int | None = 0,
        steps: int = 1000,
        snapshot_every: int = 200,
        code_sha: str,
        fixed_params: Mapping[str, Any] | None = None,
        early_stop: EarlyStopSpec | Mapping[str, Any] | None = None,
        created_at: str = "",
    ) -> ScanPlan:
        """Build a plan by running the DoE generator over *variables*.

        Determinism: identical ``(variables, method, n_points, seed)``
        yields the identical design matrix (LHS seeds ``random.Random``;
        Sobol is a fixed low-discrepancy sequence).
        """
        from .doe import DoEVariable, generate_doe

        if not variables:
            raise ValueError("at least one ScanVariable is required")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an int or None")
        doe = generate_doe(
            [
                DoEVariable(
                    name=var.name,
                    low=var.low,
                    high=var.high,
                    levels=list(var.levels) if var.levels is not None else None,
                )
                for var in variables
            ],
            method=method,  # type: ignore[arg-type]
            n_samples=n_points,
            seed=seed,
        )
        points = tuple(
            ScanPoint(
                index=i,
                point_id=f"p{i:04d}",
                run_id=f"{scan_id}-p{i:04d}",
                params={k: float(v) for k, v in row.items()},
            )
            for i, row in enumerate(doe.design_matrix)
        )
        if isinstance(early_stop, Mapping):
            early_stop = EarlyStopSpec.from_dict(early_stop)
        return cls(
            scan_id=scan_id,
            case=case,
            variables=tuple(variables),
            method=method,
            n_points=n_points,
            seed=seed,
            steps=steps,
            snapshot_every=snapshot_every,
            code_sha=code_sha,
            fixed_params=dict(fixed_params or {}),
            early_stop=early_stop,  # type: ignore[arg-type]
            points=points,
            created_at=created_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCAN_PLAN_SCHEMA,
            "scan_id": self.scan_id,
            "case": self.case,
            "variables": [v.to_dict() for v in self.variables],
            "method": self.method,
            "n_points": self.n_points,
            "seed": self.seed,
            "steps": self.steps,
            "snapshot_every": self.snapshot_every,
            "code_sha": self.code_sha,
            "fixed_params": dict(self.fixed_params),
            "early_stop": self.early_stop.to_dict() if self.early_stop else None,
            "points": [
                {
                    "index": p.index,
                    "point_id": p.point_id,
                    "run_id": p.run_id,
                    "params": p.params,
                }
                for p in self.points
            ],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScanPlan:
        schema = data.get("schema")
        if schema != SCAN_PLAN_SCHEMA:
            raise ValueError(f"unsupported plan schema {schema!r}")
        points = tuple(
            ScanPoint(
                index=int(p["index"]),
                point_id=str(p["point_id"]),
                run_id=str(p["run_id"]),
                params={k: float(v) for k, v in p["params"].items()},
            )
            for p in data["points"]
        )
        return cls(
            scan_id=str(data["scan_id"]),
            case=str(data["case"]),
            variables=tuple(ScanVariable.from_dict(v) for v in data["variables"]),
            method=str(data["method"]),
            n_points=int(data["n_points"]),
            seed=data["seed"],
            steps=int(data["steps"]),
            snapshot_every=int(data["snapshot_every"]),
            code_sha=_validate_code_sha(data["code_sha"]),
            fixed_params=dict(data.get("fixed_params") or {}),
            early_stop=EarlyStopSpec.from_dict(data.get("early_stop")),
            points=points,
            created_at=str(data.get("created_at", "")),
        )

    def save(self, path: str | Path) -> Path:
        """Write ``plan.json`` (callers pass ``<dataset dir>/plan.json``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ScanPlan:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def plan_digest(self) -> str:
        """Stable sha256 of the design matrix (plan-level provenance)."""
        payload = json.dumps(
            {
                "scan_id": self.scan_id,
                "case": self.case,
                "variables": [v.to_dict() for v in self.variables],
                "method": self.method,
                "seed": self.seed,
                "steps": self.steps,
                "snapshot_every": self.snapshot_every,
                "points": [p.params for p in self.points],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------


def assign_points_to_gpus(n_points: int, gpus: Sequence[int]) -> dict[int, list[int]]:
    """Deal point indices round-robin over the card pool (deterministic).

    Round-robin (not contiguous blocks) interleaves heterogeneous point
    costs, so a sweep whose grid size varies keeps both cards busy.
    """
    pool = [int(g) for g in gpus]
    if not pool:
        return {}
    if len(set(pool)) != len(pool):
        raise ValueError(f"gpus must be unique, got {pool}")
    if n_points < 0:
        raise ValueError("n_points must be >= 0")
    table: dict[int, list[int]] = {g: [] for g in pool}
    for i in range(n_points):
        table[pool[i % len(pool)]].append(i)
    return table


def split_points(
    n_points: int,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, list[int]]:
    """Deterministically split point indices into train/val/test groups.

    The split is at *point* (DoE configuration) granularity — the leakage
    discipline of the FieldDatasetR2 contract: no configuration
    contributes snapshots to two splits.
    """
    if len(ratios) != 3 or any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError("ratios must be three non-negative numbers summing > 0")
    if n_points < 1:
        raise ValueError("n_points must be >= 1")
    order = list(range(n_points))
    random.Random(seed).shuffle(order)
    total = sum(ratios)
    n_train = max(1, min(n_points, round(n_points * ratios[0] / total)))
    n_val = max(0, min(n_points - n_train, round(n_points * ratios[1] / total)))
    return {
        "train": sorted(order[:n_train]),
        "val": sorted(order[n_train : n_train + n_val]),
        "test": sorted(order[n_train + n_val :]),
    }


def open_catalog(db_path: str | Path, *, timeout: float = 120.0) -> FieldDataCatalog:
    """Open the sweep catalog WAL-mode for multi-worker writes.

    Each GPU worker holds its own connection; writes are short (product
    registration), so WAL + a generous busy timeout is sufficient — the
    physics dominates the wall clock, not the catalog.
    """
    from .data.catalog import FieldDataCatalog

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    return FieldDataCatalog(conn)


def coerce_case_params(case_name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Cast sweep floats to the case constructor's parameter types.

    DoE rows are floats; ``resolution``-style integer constructor params
    are detected from the registry's ``default_params()`` typing and cast.
    """
    from .cases import case_registry

    cls = case_registry[case_name]
    factory = getattr(cls, "default_params", None)
    defaults: dict[str, Any] = dict(factory()) if callable(factory) else {}
    out: dict[str, Any] = {}
    for key, value in dict(params).items():
        default = defaults.get(key)
        if (
            isinstance(default, int)
            and not isinstance(default, bool)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            # Integer constructor params (e.g. resolution) receive rounded
            # DoE values; the design-matrix value stays the metadata truth.
            out[key] = int(round(float(value)))
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Per-point execution
# ---------------------------------------------------------------------------


@dataclass
class PointOutcome:
    """Result of one sweep point (also the ``status.json`` payload)."""

    point_id: str
    status: str  # "completed" | "skipped" | "failed"
    product_ids: list[str] = field(default_factory=list)
    exported_steps: list[int] = field(default_factory=list)
    completed_steps: int = 0
    elapsed_s: float = 0.0
    mean_mlups: float | None = None
    early_stopped: bool = False
    early_stop_reason: str | None = None
    params: dict[str, float] = field(default_factory=dict)
    device: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PointOutcome:
        return cls(
            point_id=str(data["point_id"]),
            status=str(data["status"]),
            product_ids=list(data.get("product_ids") or []),
            exported_steps=[int(s) for s in data.get("exported_steps") or []],
            completed_steps=int(data.get("completed_steps", 0)),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
            mean_mlups=data.get("mean_mlups"),
            early_stopped=bool(data.get("early_stopped", False)),
            early_stop_reason=data.get("early_stop_reason"),
            params=dict(data.get("params") or {}),
            device=str(data.get("device", "")),
            error=data.get("error"),
        )


def _point_paths(out_dir: Path, point: ScanPoint) -> tuple[Path, Path, Path]:
    point_dir = out_dir / "points" / point.point_id
    return point_dir, point_dir / "fields.h5", point_dir / "status.json"


def _purge_products(catalog: FieldDataCatalog, product_ids: Sequence[str]) -> None:
    """Remove products from the catalog entirely (rerun of a half-done point).

    ``register_product`` fails closed on an already-registered product id,
    and archiving is not enough (the id still resolves), so a point reset
    must really delete its rows.  The catalog offers no delete API by
    design for finished assets; this is the one sanctioned purge path —
    it only ever touches products of the point being reset, which by
    definition never made it into a finalised dataset.
    """
    conn = catalog._conn
    for product_id in product_ids:
        conn.execute("DELETE FROM asset_metadata WHERE asset_id = ?", (product_id,))
        conn.execute("DELETE FROM quality_reports WHERE asset_id = ?", (product_id,))
        conn.execute(
            "DELETE FROM lineage WHERE source_id = ? OR target_id = ?", (product_id, product_id)
        )
        conn.execute("DELETE FROM assets WHERE asset_id = ?", (product_id,))
    conn.commit()


def run_scan_point(
    plan: ScanPlan,
    point: ScanPoint,
    out_dir: str | Path,
    catalog: FieldDataCatalog,
    device: str = "cpu",
    *,
    log: Callable[..., None] | None = None,
) -> PointOutcome:
    """Run one sweep point to registered catalog products.

    The step chain is the case registry's own (:meth:`CaseBase.make_step`
    — collide -> pre-BCs -> stream -> post-BCs, plus the case's periodic
    mass correction), i.e. exactly what :func:`tensorlbm.cases.run_case`
    executes, so sweep points are bit-comparable with direct registry
    runs.  Snapshot export rides on
    :class:`~tensorlbm.reporters.FieldSampleReporter` (HDF5 +
    PASS-gated registration in one hop) driven through
    :func:`~tensorlbm.reporters.dispatch`; a
    :class:`~tensorlbm.reporters.ThroughputReporter` records MLUPS; an
    optional :class:`~tensorlbm.reporters.EarlyStopReporter` truncates
    steady points.  Like ``run_case``, the final step is always exported
    even when it is not a sampling multiple.
    """
    from .cases import get_case
    from .reporters import (
        EarlyStopReporter,
        FieldSampleReporter,
        StepContext,
        ThroughputReporter,
        dispatch,
    )

    out_dir = Path(out_dir)
    point_dir, h5_path, status_path = _point_paths(out_dir, point)
    point_dir.mkdir(parents=True, exist_ok=True)
    say = log if log is not None else (lambda *a, **k: None)

    params = coerce_case_params(plan.case, {**plan.fixed_params, **point.params})
    case = get_case(plan.case, device=device, **params)

    metadata = dict(case.metadata())
    metadata.update(
        {
            "scan_id": plan.scan_id,
            "point_id": point.point_id,
            "point_index": point.index,
            "scan_steps": plan.steps,
            "snapshot_every": plan.snapshot_every,
            "device": str(device),
        }
    )
    metadata.update({k: float(v) for k, v in point.params.items()})

    field_reporter = FieldSampleReporter(
        h5_path,
        run_id=point.run_id,
        case=plan.case,
        code_sha=plan.code_sha,
        interval=plan.snapshot_every,
        catalog=catalog,
        solid_mask=case.solid_mask(),
        metadata=metadata,
    )
    throughput = ThroughputReporter(interval=max(1, plan.steps // 10))
    reporters: list[Any] = [throughput, field_reporter]
    early = None
    if plan.early_stop is not None:
        spec = plan.early_stop

        def _mean_speed(ctx: StepContext) -> float:
            _, ux, uy, uz = ctx.macroscopic()
            return float((ux.abs().mean() + uy.abs().mean() + uz.abs().mean()).item() / 3.0)

        early = EarlyStopReporter(
            monitor=_mean_speed,
            threshold=spec.threshold,
            patience=spec.patience,
            interval=spec.interval,
            min_step=spec.min_step,
            relative=spec.relative,
        )
        reporters.append(early)

    f = case.initial_f()
    initial_mass = float(f.sum().item())
    step_fn = case.make_step()
    mass_every = int(getattr(case, "mass_correction_interval", 0) or 0)
    if mass_every > 0:
        from .solver3d import correct_mass3d

    nz, ny, nx = case.resolution
    ctx = StepContext(
        step=0,
        f=f,
        lattice=case.lattice,
        num_cells=nz * ny * nx,
        num_steps=plan.steps,
        units=case.units,
    )
    say(
        f"[{point.point_id}] case={plan.case} params={params} "
        f"grid={nz}x{ny}x{nx} steps={plan.steps} device={device}"
    )
    t0 = time.perf_counter()
    completed = 0
    for step in range(1, plan.steps + 1):
        f = step_fn(f)
        if mass_every > 0 and step % mass_every == 0:
            f = correct_mass3d(f, initial_mass)
        ctx.step = step
        ctx.f = f
        dispatch(ctx, reporters)
        completed = step
        if ctx.stop:
            break
    if completed not in field_reporter.exported_steps:
        # run_case parity: the final step is always exported.
        ctx.step = completed
        ctx.f = f
        ctx.stop = False
        field_reporter(ctx)
    elapsed = time.perf_counter() - t0

    outcome = PointOutcome(
        point_id=point.point_id,
        status="completed",
        product_ids=list(field_reporter.product_ids),
        exported_steps=list(field_reporter.exported_steps),
        completed_steps=completed,
        elapsed_s=elapsed,
        mean_mlups=throughput.mean_mlups,
        early_stopped=bool(early.stopped) if early is not None else False,
        early_stop_reason=early.reason if early is not None else None,
        params=dict(point.params),
        device=str(device),
    )
    status_path.write_text(json.dumps(outcome.to_dict(), indent=2), encoding="utf-8")
    say(
        f"[{point.point_id}] {outcome.status}: {len(outcome.product_ids)} products, "
        f"{completed}/{plan.steps} steps in {elapsed:.1f}s "
        f"({throughput.mean_mlups or 0.0:.1f} MLUPS mean)"
        + (f", early stop: {early.reason}" if early is not None and early.stopped else "")
    )
    return outcome


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class ScanExecutor:
    """Run a :class:`ScanPlan` into a dataset directory, then finalise it.

    Args:
        plan: the sweep plan.
        out_dir: dataset directory (``plan.json``, ``catalog.db``,
            ``points/``, ``scan_summary.json``, ``dataset.json`` live here).
        gpus: card pool for case-level parallelism (one spawn worker per
            card).  Empty (default) runs serially on ``serial_device``.
        serial_device: device for the in-process path (and for
            finalisation queries).
        catalog_timeout: SQLite busy timeout for worker connections.
        split_ratios, split_seed: dataset finalisation split.
    """

    def __init__(
        self,
        plan: ScanPlan,
        out_dir: str | Path,
        *,
        gpus: Sequence[int] = (),
        serial_device: str = "cpu",
        catalog_timeout: float = 120.0,
        split_ratios: Sequence[float] = (0.7, 0.15, 0.15),
        split_seed: int | None = None,
    ) -> None:
        self.plan = plan
        self.out_dir = Path(out_dir)
        self.gpus = tuple(int(g) for g in gpus)
        self.serial_device = serial_device
        self.catalog_timeout = float(catalog_timeout)
        self.split_ratios = tuple(float(r) for r in split_ratios)
        self.split_seed = plan.seed if split_seed is None else int(split_seed)
        self._catalog: FieldDataCatalog | None = None
        self.dataset: dict[str, Any] | None = None

    # -- paths / catalog ---------------------------------------------------

    @property
    def catalog_db(self) -> Path:
        return self.out_dir / "catalog.db"

    def catalog(self) -> FieldDataCatalog:
        if self._catalog is None:
            self._catalog = open_catalog(self.catalog_db, timeout=self.catalog_timeout)
        return self._catalog

    def close(self) -> None:
        if self._catalog is not None:
            self._catalog.close()
            self._catalog = None

    def _status_path(self, point: ScanPoint) -> Path:
        return _point_paths(self.out_dir, point)[2]

    # -- resume ------------------------------------------------------------

    def point_outcome(self, point: ScanPoint) -> PointOutcome | None:
        path = self._status_path(point)
        if not path.exists():
            return None
        try:
            return PointOutcome.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def is_point_complete(self, point: ScanPoint) -> bool:
        """A point is done when its status says completed *and* every one
        of its products is still registered (product-existence check)."""
        outcome = self.point_outcome(point)
        if outcome is None or outcome.status != "completed" or not outcome.product_ids:
            return False
        catalog = self.catalog()
        return all(catalog.get_asset(pid) is not None for pid in outcome.product_ids)

    def _reset_point(self, point: ScanPoint) -> None:
        """Purge partial products and clear the point dir for a rerun."""
        catalog = self.catalog()
        stale = catalog.find_assets_by_metadata(
            "point_id", point.point_id, kind="field_product", limit=10_000
        )
        if stale:
            _purge_products(catalog, [asset.asset_id for asset in stale])
        point_dir = _point_paths(self.out_dir, point)[0]
        if point_dir.exists():
            shutil.rmtree(point_dir)

    # -- orchestration -----------------------------------------------------

    def run(self, resume: bool = True) -> dict[str, Any]:
        """Execute the sweep (skipping finished points when *resume*)."""
        from .cases import has_case

        if not has_case(self.plan.case):
            from .cases import list_cases

            raise KeyError(
                f"unknown case {self.plan.case!r}; registered: {[c['name'] for c in list_cases()]}"
            )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.plan.save(self.out_dir / "plan.json")
        (self.out_dir / "logs").mkdir(exist_ok=True)

        todo = [p for p in self.plan.points if not (resume and self.is_point_complete(p))]
        t0 = time.perf_counter()
        outcomes: list[PointOutcome] = []
        if todo and self.gpus:
            outcomes.extend(self._run_parallel(todo))
        elif todo:
            outcomes.extend(self._run_serial(todo))

        # Resume bookkeeping: points already complete count as skipped.
        done_before = {p.point_id for p in self.plan.points} - {p.point_id for p in todo}
        skipped = [PointOutcome(point_id=pid, status="skipped") for pid in sorted(done_before)]

        failed = [o for o in outcomes if o.status == "failed"]
        if failed:
            sys.stderr.write(
                f"scan {self.plan.scan_id}: {len(failed)} point(s) failed: "
                + "; ".join(f"{o.point_id}: {o.error}" for o in failed[:5])
                + "\n"
            )
        completed_total = sum(1 for p in self.plan.points if self.is_point_complete(p))
        if completed_total == 0:
            self.close()
            raise RuntimeError(
                f"scan {self.plan.scan_id}: no completed points; refusing to finalise"
            )

        self.dataset = self.finalize()
        summary = {
            "scan_id": self.plan.scan_id,
            "out_dir": str(self.out_dir),
            "plan_path": str(self.out_dir / "plan.json"),
            "plan_digest": self.plan.plan_digest(),
            "n_points": len(self.plan.points),
            "n_completed": completed_total,
            "n_skipped": len(skipped),
            "n_failed": len(failed),
            "outcomes": [o.to_dict() for o in outcomes],
            "dataset": self.dataset,
            "elapsed_s": time.perf_counter() - t0,
        }
        (self.out_dir / "scan_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        self.close()
        return summary

    def _run_serial(self, points: Sequence[ScanPoint]) -> list[PointOutcome]:
        catalog = self.catalog()
        outcomes: list[PointOutcome] = []
        for point in points:
            self._reset_point(point)
            try:
                outcomes.append(
                    run_scan_point(self.plan, point, self.out_dir, catalog, self.serial_device)
                )
            except Exception as error:  # noqa: BLE001 - record, keep sweeping
                outcomes.append(
                    PointOutcome(
                        point_id=point.point_id,
                        status="failed",
                        params=dict(point.params),
                        device=self.serial_device,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
        return outcomes

    def _run_parallel(self, points: Sequence[ScanPoint]) -> list[PointOutcome]:
        """One spawn worker per GPU in the pool; in-process scheduling."""
        import multiprocessing as mp

        todo = list(points)
        assignment = assign_points_to_gpus(len(todo), self.gpus)
        plan_dict = self.plan.to_dict()
        ctx = mp.get_context("spawn")
        processes: list[tuple[int, Any, Path, Path]] = []
        for gpu, indices in sorted(assignment.items()):
            if not indices:
                continue
            point_indices = [todo[i].index for i in indices]
            log_path = self.out_dir / "logs" / f"gpu{gpu}.log"
            outcome_path = self.out_dir / "logs" / f"gpu{gpu}.outcomes.json"
            outcome_path.write_text("[]", encoding="utf-8")
            proc = ctx.Process(
                target=_gpu_worker_entry,
                args=(
                    plan_dict,
                    str(self.out_dir),
                    int(gpu),
                    point_indices,
                    str(outcome_path),
                    str(log_path),
                    self.catalog_timeout,
                ),
                name=f"scan-{self.plan.scan_id}-gpu{gpu}",
            )
            proc.start()
            processes.append((gpu, proc, outcome_path, log_path))
        for _, proc, _, _ in processes:
            proc.join()
        outcomes: list[PointOutcome] = []
        for gpu, proc, outcome_path, log_path in processes:
            data: Any = []
            if outcome_path.exists():
                try:
                    data = json.loads(outcome_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    data = []
            by_id = {entry.get("point_id"): entry for entry in data}
            for slot in assignment[gpu]:
                point = todo[slot]
                entry = by_id.get(point.point_id)
                if proc.exitcode == 0 and entry is not None:
                    outcomes.append(PointOutcome.from_dict(entry))
                else:
                    outcomes.append(
                        PointOutcome(
                            point_id=point.point_id,
                            status="failed",
                            params=dict(point.params),
                            device=f"cuda:{gpu}",
                            error=(
                                f"gpu worker exited with code {proc.exitcode} "
                                f"before reporting this point (see {log_path})"
                            ),
                        )
                    )
        if len(outcomes) != len(todo):
            raise RuntimeError(f"collected {len(outcomes)} outcomes for {len(todo)} points")
        return outcomes

    # -- dataset finalisation ----------------------------------------------

    def finalize(self) -> dict[str, Any]:
        """Assemble + register the sweep dataset with full lineage.

        * one :class:`FieldDatasetR2` over every completed point's products
          (point-granularity train/val/test split — no configuration
          crosses splits), with ``training_input_fingerprint``;
        * catalog assets ``scan-<id>:plan`` (kind ``run``) and
          ``scan-<id>:dataset`` (kind ``dataset``);
        * lineage edges ``plan -> run:<run_id> -> <product> -> dataset``,
          so ``catalog.upstream("scan-<id>:dataset")`` returns the whole
          chain (plan + runs + products) in one query;
        * ``dataset.json`` beside the catalog for tooling.
        """
        from .data.catalog import AssetRecord, LineageRecord
        from .data.field_dataset_r2 import FieldDatasetR2, FieldSampleRefR2
        from .data.solver_export import load_product

        catalog = self.catalog()
        split_idx = split_points(len(self.plan.points), self.split_ratios, seed=self.split_seed)
        split_of = {idx: name for name, idxs in split_idx.items() for idx in idxs}

        plan_asset_id = f"scan-{self.plan.scan_id}:plan"
        dataset_asset_id = f"scan-{self.plan.scan_id}:dataset"
        catalog.register_asset(
            AssetRecord(
                asset_id=plan_asset_id,
                name=f"Scan plan {self.plan.scan_id}",
                kind="run",
                description=(
                    f"DoE plan ({self.plan.method}, {len(self.plan.points)} points, "
                    f"seed={self.plan.seed}) over case {self.plan.case!r}"
                ),
                tags=("scan_plan", f"case:{self.plan.case}"),
                source_run_id=self.plan.scan_id,
            )
        )
        plan_rows: dict[str, str] = {
            "scan_id": self.plan.scan_id,
            "case": self.plan.case,
            "doe_method": self.plan.method,
            "n_points": str(len(self.plan.points)),
            "seed": str(self.plan.seed),
            "steps": str(self.plan.steps),
            "snapshot_every": str(self.plan.snapshot_every),
            "code_sha": self.plan.code_sha,
            "plan_digest": self.plan.plan_digest(),
            "variables": json.dumps([v.to_dict() for v in self.plan.variables]),
            "fixed_params": json.dumps(self.plan.fixed_params),
        }
        for key, value in plan_rows.items():
            catalog.add_metadata(plan_asset_id, key, value, source="scan_runner")

        samples: list[Any] = []
        splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        products_by_point: dict[str, list[str]] = {}
        for point in self.plan.points:
            outcome = self.point_outcome(point)
            product_ids: list[str] = []
            if outcome is not None and outcome.status == "completed":
                product_ids = list(outcome.product_ids)
            else:
                product_ids = [
                    asset.asset_id
                    for asset in catalog.find_assets_by_metadata(
                        "point_id", point.point_id, kind="field_product", limit=10_000
                    )
                    if asset.status == "active"
                ]
                product_ids.sort()
            if not product_ids:
                continue  # failed point: excluded from the dataset
            split = split_of[point.index]
            for product_id in product_ids:
                if catalog.get_asset(product_id) is None:
                    raise RuntimeError(f"point {point.point_id}: product {product_id} vanished")
                product = load_product(catalog, product_id)
                samples.append(
                    FieldSampleRefR2(
                        sample_id=product_id,
                        product=product,
                        group_id=point.point_id,
                        source_case_id=f"{self.plan.case}:{point.point_id}",
                        source_trajectory_id=point.run_id,
                    )
                )
                splits[split].append(product_id)
            products_by_point[point.point_id] = product_ids
            catalog.add_lineage(
                LineageRecord(
                    source_id=plan_asset_id,
                    target_id=f"run:{point.run_id}",
                    relation_type="spawned",
                    transformation=f"tensorlbm.scan_runner point {point.point_id}",
                    resource_type="run",
                )
            )

        if not samples:
            raise RuntimeError(f"scan {self.plan.scan_id}: no products to build a dataset from")
        dataset = FieldDatasetR2(
            dataset_id=self.plan.scan_id,
            version="1.0.0",
            task_name="field_reconstruction",
            samples=tuple(samples),
            splits={name: tuple(ids) for name, ids in splits.items()},
            lineage={
                "created_by": "tensorlbm.scan_runner.ScanExecutor.finalize",
                "plan_asset_id": plan_asset_id,
                "plan_digest": self.plan.plan_digest(),
                "doe": {
                    "method": self.plan.method,
                    "n_points": len(self.plan.points),
                    "seed": self.plan.seed,
                    "variables": [v.to_dict() for v in self.plan.variables],
                },
                "case": self.plan.case,
                "code_sha": self.plan.code_sha,
            },
        )
        fingerprint = dataset.training_input_fingerprint()

        catalog.register_asset(
            AssetRecord(
                asset_id=dataset_asset_id,
                name=f"Scan dataset {self.plan.scan_id}",
                kind="dataset",
                description=(
                    f"{len(samples)} field snapshots from a {self.plan.method} sweep "
                    f"of case {self.plan.case!r} ({len(products_by_point)} points)"
                ),
                tags=("scan_dataset", f"case:{self.plan.case}"),
                source_run_id=self.plan.scan_id,
            )
        )
        for key, value in {
            "dataset_id": dataset.dataset_id,
            "version": dataset.version,
            "task_name": dataset.task_name,
            "n_samples": str(len(samples)),
            "n_train": str(len(splits["train"])),
            "n_val": str(len(splits["val"])),
            "n_test": str(len(splits["test"])),
            "training_input_fingerprint": fingerprint,
            "plan_asset_id": plan_asset_id,
            "plan_digest": self.plan.plan_digest(),
            "code_sha": self.plan.code_sha,
        }.items():
            catalog.add_metadata(dataset_asset_id, key, value, source="scan_runner")
        for product_id in dataset.splits["train"] + dataset.splits["val"] + dataset.splits["test"]:
            catalog.add_lineage(
                LineageRecord(
                    source_id=product_id,
                    target_id=dataset_asset_id,
                    relation_type="member_of",
                    transformation="tensorlbm.scan_runner dataset assembly",
                    resource_type="dataset",
                )
            )
        upstream = catalog.upstream(dataset_asset_id)

        info = {
            "dataset_id": dataset.dataset_id,
            "asset_id": dataset_asset_id,
            "plan_asset_id": plan_asset_id,
            "version": dataset.version,
            "task_name": dataset.task_name,
            "n_samples": len(samples),
            "splits": {k: len(v) for k, v in splits.items()},
            "split_points": {
                k: [self.plan.points[i].point_id for i in idxs] for k, idxs in split_idx.items()
            },
            "products_by_point": products_by_point,
            "training_input_fingerprint": fingerprint,
            "upstream_assets": upstream,
        }
        (self.out_dir / "dataset.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return info


# ---------------------------------------------------------------------------
# GPU worker entry (spawn target; must stay module-level and picklable-args)
# ---------------------------------------------------------------------------


def _gpu_worker_entry(
    plan_dict: dict[str, Any],
    out_dir: str,
    gpu: int,
    point_indices: list[int],
    outcome_path: str,
    log_path: str,
    catalog_timeout: float,
) -> None:
    """Run this card's points sequentially; report outcomes as JSON."""
    plan = ScanPlan.from_dict(plan_dict)
    root = Path(out_dir)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, Any]] = []
    with open(log_path, "a", buffering=1, encoding="utf-8") as log:
        sys.stdout = log  # type: ignore[assignment]
        sys.stderr = log  # type: ignore[assignment]
        import torch

        device = f"cuda:{gpu}"
        torch.cuda.set_device(gpu)
        print(
            f"[gpu{gpu}] worker start: {len(point_indices)} point(s), "
            f"device={torch.cuda.get_device_name(gpu)}",
            flush=True,
        )
        catalog = open_catalog(root / "catalog.db", timeout=catalog_timeout)
        try:
            for idx in point_indices:
                point = plan.points[idx]
                # In-worker resume guard (the parent filters, this protects
                # against a duplicated dispatch).
                status_file = _point_paths(root, point)[2]
                already = False
                if status_file.exists():
                    try:
                        prior = PointOutcome.from_dict(
                            json.loads(status_file.read_text(encoding="utf-8"))
                        )
                        already = (
                            prior.status == "completed"
                            and bool(prior.product_ids)
                            and all(catalog.get_asset(pid) is not None for pid in prior.product_ids)
                        )
                    except (json.JSONDecodeError, KeyError, ValueError):
                        already = False
                if already:
                    outcomes.append(
                        PointOutcome(point_id=point.point_id, status="skipped").to_dict()
                    )
                    print(f"[gpu{gpu}] {point.point_id}: already complete, skipping", flush=True)
                    continue
                # Reset any half-finished attempt before rerunning.
                stale = catalog.find_assets_by_metadata(
                    "point_id", point.point_id, kind="field_product", limit=10_000
                )
                if stale:
                    _purge_products(catalog, [asset.asset_id for asset in stale])
                point_dir = _point_paths(root, point)[0]
                if point_dir.exists():
                    shutil.rmtree(point_dir)
                try:
                    outcome = run_scan_point(plan, point, root, catalog, device, log=print)
                    outcomes.append(outcome.to_dict())
                except Exception as error:  # noqa: BLE001 - keep the card busy
                    print(
                        f"[gpu{gpu}] {point.point_id}: FAILED {type(error).__name__}: {error}",
                        flush=True,
                    )
                    outcomes.append(
                        PointOutcome(
                            point_id=point.point_id,
                            status="failed",
                            params=dict(point.params),
                            device=device,
                            error=f"{type(error).__name__}: {error}",
                        ).to_dict()
                    )
        finally:
            catalog.close()
        Path(outcome_path).write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
        print(f"[gpu{gpu}] worker done: {len(outcomes)} outcome(s)", flush=True)
