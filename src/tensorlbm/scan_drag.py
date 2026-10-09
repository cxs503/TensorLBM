"""Optional per-point drag time-history for parameter sweeps.

Composes the exact discrete-kinetic control-volume observer
(:mod:`tensorlbm.control_volume_force`) into the scan-runner step loop as
an opt-in plan field (``ScanPlan.drag_survey``):

* one box control volume per point, built once from the bounding box of
  ``case.solid_mask()`` expanded by :attr:`DragSurveySpec.margin` cells,
  clamped to the largest strictly interior window (so the one-cell shell
  around the control volume is fluid and all physical boundary
  conditions stay outside it — the invariant
  :func:`tensorlbm.control_volume_force.box_control_volume` validates);
* every :attr:`DragSurveySpec.interval` steps the observer evaluates
  ``force_on_body = streaming_momentum_import - fluid_momentum_change``
  over one complete solver step, in phase with the reporter ``dispatch``
  (post-stream, post-BCs, post mass correction);
* cases that run a global mass correction (``correct_mass3d``) inject a
  distributed artificial momentum inside any control volume, which the
  kinetic balance would otherwise attribute to the body; the runner
  reports each rescale to :meth:`DragSurveyObserver.note_mass_correction`
  tagged with its solver step, and the impulse of the sampled step is
  added back at the sample, restoring control-volume invariance
  (measured on ``suboff_n128`` at resolution 128, Re=148, 4000 steps
  with a mass correction every 10 steps and sampling every 25: nested
  CVs margin 2 vs 4 agree to ~3e-6 relative, and the balance matches
  the Ladd link momentum exchange to <5e-4 per step past the startup
  transient, ~1e-5 on tail means; without compensation the balance
  drifts +17% (margin 2) to +38% (margin 5) at resolution 24, and the
  wake momentum-deficit estimate sits ~63% low);
* the ``(step, force_x, force_y, force_z, |F|)`` history is appended to
  an in-memory list and flushed to a ``drag_history.json`` sidecar in the
  point directory after every sample, so an interrupted point resumes its
  history from the sidecar next to its checkpoint.

The observer is a measurement, not a solver input: populations and every
catalog product are bit-identical whether or not the survey is enabled
(the sampling steps decompose the case's own
``collide -> pre_boundaries -> stream -> post_boundaries`` chain, which is
exactly ``CaseBase.make_step``).

Units
-----
All values are LATTICE units: the force is the momentum transferred to
the body per lattice time step (``dx = dt = 1``, ``rho ~ 1``).  Convert
with ``F_phys = F_lattice * rho_phys * dx_phys**4 / dt_phys**2``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

    from .cases.base import CaseBase

__all__ = [
    "DRAG_HISTORY_SCHEMA",
    "DragSurveyObserver",
    "DragSurveySpec",
]

#: Schema tag persisted in ``drag_history.json`` (forward compatibility).
DRAG_HISTORY_SCHEMA = "tensorlbm.drag-history/v1"

_UNITS_NOTE = (
    "LATTICE units: force is momentum transferred to the enclosed solid "
    "per lattice time step (dx = dt = 1, rho_lattice ~ 1). Convert with "
    "F_phys = F_lattice * rho_phys * dx_phys**4 / dt_phys**2. Value is the "
    "discrete-kinetic control-volume balance "
    "force = streaming_momentum_import - fluid_momentum_change over one "
    "complete solver step (tensorlbm.control_volume_force), plus the "
    "momentum injected by any global mass correction inside the control "
    "volume during the sampled step; it does not depend on any "
    "far-field reference velocity."
)

#: Distance (cells) a control-volume bound keeps from the domain edge so
#: that the one-cell shell around the control volume is fluid and every
#: physical boundary condition stays outside it, even on periodic axes.
_SHELL_CELLS = 2


@dataclass(frozen=True)
class DragSurveySpec:
    """Optional per-point drag time-history survey (``ScanPlan.drag_survey``).

    Attributes
    ----------
    margin:
        Control-volume outward growth in cells relative to the bounding
        box of ``case.solid_mask()`` (clamped to the largest strictly
        interior window when the grid is tight).
    interval:
        Sample every *interval* steps.
    """

    margin: int = 6
    interval: int = 50

    def __post_init__(self) -> None:
        for name, value in (("margin", self.margin), ("interval", self.interval)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"DragSurveySpec.{name} must be a positive int, got {value!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> DragSurveySpec | None:
        if data is None:
            return None
        return cls(margin=int(data["margin"]), interval=int(data["interval"]))


class DragSurveyObserver:
    """One point's drag observer: box control volume + step history sidecar.

    Built from the case's solid mask once per point; :meth:`sample` is
    called by :func:`tensorlbm.scan_runner.run_scan_point` in phase with
    the reporter dispatch on every sampling step.
    """

    def __init__(
        self,
        spec: DragSurveySpec,
        *,
        case: CaseBase,
        scan_id: str,
        point_id: str,
        run_id: str,
        point_dir: str | Path,
    ) -> None:
        from .control_volume_force import box_control_volume

        self.spec = spec
        self.scan_id = str(scan_id)
        self.point_id = str(point_id)
        self.run_id = str(run_id)
        self.point_dir = Path(point_dir)
        self._path = self.point_dir / "drag_history.json"

        solid = case.solid_mask()
        if solid is None:
            raise ValueError(
                "drag_survey requires a case with a solid obstacle, but "
                f"case {type(case).__name__}.solid_mask() returned None"
            )
        nz, ny, nx = tuple(int(n) for n in solid.shape)
        cells = solid.nonzero()
        if cells.numel() == 0:
            raise ValueError("drag_survey: case.solid_mask() is empty — nothing to enclose")
        lo = cells.min(dim=0).values.tolist()
        hi = cells.max(dim=0).values.tolist()
        (z_min, y_min, x_min), (z_max, y_max, x_max) = lo, hi

        # Clamp the requested bbox + margin to the largest strictly
        # interior window.  Containment of the bbox is what keeps the
        # one-cell outer shell of the control volume fluid.
        bounds: dict[str, int] = {}
        for axis, size, mn, mx in (
            ("z", nz, z_min, z_max),
            ("y", ny, y_min, y_max),
            ("x", nx, x_min, x_max),
        ):
            low = max(_SHELL_CELLS, mn - spec.margin)
            high = min(size - _SHELL_CELLS, mx + 1 + spec.margin)
            if low > mn or high < mx + 1:
                raise ValueError(
                    f"drag_survey: solid bounding box {axis}[{mn}, {mx}] of "
                    f"{type(case).__name__} leaves no strictly interior "
                    f"control volume with a fluid shell on grid "
                    f"{nz}x{ny}x{nx}"
                )
            bounds[f"{axis}0"] = low
            bounds[f"{axis}1"] = high
        self.bounds = bounds
        self.grid = (nz, ny, nx)
        self.n_solid_cells = int(cells.shape[0])

        self._solid = solid
        self._cv = box_control_volume(
            (nz, ny, nx),
            x0=bounds["x0"],
            x1=bounds["x1"],
            y0=bounds["y0"],
            y1=bounds["y1"],
            z0=bounds["z0"],
            z1=bounds["z1"],
            device=solid.device,
        )
        periodic = getattr(case, "periodic_axes", None)
        self.periodic_axes: tuple[str, ...] = (
            tuple(axis for axis, is_periodic in dict(periodic()).items() if is_periodic)
            if callable(periodic)
            else ()
        )
        self._history: list[dict[str, Any]] = []
        self._injection = [0.0, 0.0, 0.0]
        self._injection_step: int | None = None

    # -- sampling ------------------------------------------------------------

    @property
    def interval(self) -> int:
        return self.spec.interval

    @property
    def samples(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._history)

    def note_mass_correction(self, f_before: Any, f_after: Any, *, step: int | None = None) -> None:
        """Record the artificial momentum a population rescale injected.

        Global mass corrections (``correct_mass3d``) rescale every cell,
        acting as a distributed forcing *inside* the control volume that
        the kinetic balance would otherwise attribute to the body (the
        observer's precondition is that no forcing acts inside the CV).

        Each :meth:`sample` is an independent *one-step* balance (its
        import and its momentum change both cover exactly the sampled
        step), so it must be compensated with the injection applied
        during **that step only**.  The record is therefore per-step: a
        later rescale overwrites an earlier one within the same step (a
        case corrects at most once per step), and a rescale on an
        unsampled step must not leak into a later sample — pass the
        solver *step* the rescale belongs to and :meth:`sample` applies
        the impulse only when the steps match (``step=None`` marks the
        record unconditionally current, for direct/manual use).
        """
        from .control_volume_force import fluid_momentum

        kwargs = {"solid": self._solid, "periodic_axes": self.periodic_axes}
        delta = fluid_momentum(f_after, self._cv, **kwargs) - fluid_momentum(
            f_before, self._cv, **kwargs
        )
        self._injection = delta.tolist()
        self._injection_step = step

    def sample(
        self,
        step: int,
        *,
        f_old: Any,
        f_post_collision: Any,
        f_new: Any,
    ) -> dict[str, Any]:
        """Observe one complete step and append it to the history."""
        from .control_volume_force import observe_control_volume_force

        result = observe_control_volume_force(
            f_old,
            f_new,
            f_post_collision,
            self._cv,
            solid=self._solid,
            periodic_axes=self.periodic_axes,
        )
        shift = (
            self._injection
            if self._injection_step is None or self._injection_step == step
            else (0.0, 0.0, 0.0)
        )
        fx, fy, fz = (value + s for value, s in zip(result.force_tuple, shift))
        self._injection = [0.0, 0.0, 0.0]
        self._injection_step = None
        entry = {
            "step": int(step),
            "force_x": fx,
            "force_y": fy,
            "force_z": fz,
            "force_abs": math.sqrt(fx * fx + fy * fy + fz * fz),
        }
        self._history.append(entry)
        self.write()
        return entry

    # -- sidecar IO -----------------------------------------------------------

    def _document(self) -> dict[str, Any]:
        return {
            "schema": DRAG_HISTORY_SCHEMA,
            "scan_id": self.scan_id,
            "point_id": self.point_id,
            "run_id": self.run_id,
            "units": _UNITS_NOTE,
            "lattice_units": True,
            "grid": list(self.grid),
            "n_solid_cells": self.n_solid_cells,
            "control_volume": {**self.bounds, "margin": self.spec.margin},
            "interval": self.spec.interval,
            "samples": list(self._history),
        }

    def write(self) -> Path:
        """(Re)write the ``drag_history.json`` sidecar next to the point."""
        self.point_dir.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._document(), indent=2), encoding="utf-8")
        return self._path

    def resume_from_sidecar(self, resume_step: int) -> int:
        """Reload samples at or below *resume_step* from the sidecar.

        Best-effort resume companion to the solver checkpoint: a missing,
        corrupt or foreign sidecar simply starts a fresh history (the
        survey never fails a physics run).  Samples beyond the checkpoint
        step are discarded — the resumed run re-samples those steps, so
        the history stays strictly increasing without duplicates.
        """
        try:
            document = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        if not isinstance(document, dict):
            return 0
        if (
            document.get("schema") != DRAG_HISTORY_SCHEMA
            or document.get("point_id") != self.point_id
            or document.get("run_id") != self.run_id
        ):
            return 0
        kept: list[dict[str, Any]] = []
        seen: set[int] = set()
        raw = document.get("samples")
        for entry in sorted(raw, key=lambda e: e.get("step", 0)) if isinstance(raw, list) else []:
            if not isinstance(entry, dict):
                continue
            step = entry.get("step")
            if not isinstance(step, int) or isinstance(step, bool):
                continue
            if step > resume_step or step in seen:
                continue
            seen.add(step)
            kept.append(dict(entry))
        self._history = kept
        return len(kept)

    # -- summaries ------------------------------------------------------------

    def summary(self) -> dict[str, float | None]:
        """``drag_final`` and ``drag_mean_tail`` for ``PointOutcome``."""
        if not self._history:
            return {"drag_final": None, "drag_mean_tail": None}
        tail = self._history[-max(1, len(self._history) // 4) :]
        return {
            "drag_final": float(self._history[-1]["force_x"]),
            "drag_mean_tail": float(sum(s["force_x"] for s in tail) / len(tail)),
        }
