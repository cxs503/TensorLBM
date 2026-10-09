"""Wake momentum-deficit drag survey for saved field snapshots.

Post-hoc drag estimation from the macroscopic fields a scan campaign
already exports (``fields.h5``: ``rho``/``ux`` snapshots plus
``solid_mask``) — no live populations required, so it applies to any
existing dataset directory.

Method
------
On a cross-section ``x_w`` downstream of the body,

``D(x_w) = sum_{y,z} rho * ux * (u_inf - ux)``   [lattice units]

with ``u_inf`` taken **per plane from a far-field border ring**, not the
nominal free-stream velocity.  Cases with periodic mass correction
(e.g. ``suboff_n128``) settle the actual far-field slightly above
nominal — on the 2026-08-21 SUBOFF sweep the drift was ~+0.6%, and a
fixed nominal reference flips the integral negative at high Re.  Plain
momentum deficit also neglects the static-pressure term, so planes are
surveyed at several offsets and near-invariance across them is the
internal consistency check (``DragSurvey.plane_spread_final``).

Normalisations: ``C_D = 2 D / (rho_inf u_inf^2 S_proj)`` with ``S_proj``
the frontal projection of the solid, and an equivalent skin-friction
``Cf_equiv = C_D * S_proj / S_wet`` for comparison against ITTC-1957 /
Blasius lines (both far outside their validity range at lattice Re —
use as scaling context only).

Caveat
------
``plane_drag`` is a momentum-deficit *estimator*, not an exact force
measurement: it depends on a far-field reference state, neglects the
static-pressure term, and assumes a developed far-field wake.  In
confining configurations it can be severely biased -- on the
``suboff_n128`` Re sweep (Re 50-800; see
``docs/benchmarks/suboff_cd_re_20260821.md``) it underestimated C_D by a
factor 2.9-7.9 that grows with Re and shifted the log-log slope from
-0.70 to -0.89.  For quantitative drag use the exact control-volume
observer in :mod:`tensorlbm.scan_drag`
(``ScanPlan(drag_survey=DragSurveySpec(...))``); keep this module for
post-hoc surveys of legacy snapshot-only datasets, read as qualitative.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DEFAULT_PLANES: tuple[int, ...] = (4, 8, 16, 32)
DEFAULT_RING = 8
DEFAULT_REFERENCE_OFFSET = 8


def plane_drag(
    ux: np.ndarray, rho: np.ndarray, *, ring: int = DEFAULT_RING
) -> tuple[float, float, float]:
    """Momentum-deficit drag on one (nz, ny) cross-section.

    The reference state ``(u_inf, rho_inf)`` is the mean over an
    ``ring``-cell border of the section (well outside the wake).
    Returns ``(drag, u_inf, rho_inf)`` in lattice units.
    """
    if ux.ndim != 2 or ux.shape != rho.shape:
        raise ValueError(f"expected matching 2-D planes, got {ux.shape} / {rho.shape}")
    if 2 * ring >= min(ux.shape):
        raise ValueError(f"ring ({ring}) too wide for plane {ux.shape}")
    ux = ux.astype(np.float64, copy=False)
    rho = rho.astype(np.float64, copy=False)
    ring_u = np.concatenate(
        [ux[:ring].ravel(), ux[-ring:].ravel(), ux[:, :ring].ravel(), ux[:, -ring:].ravel()]
    )
    ring_rho = np.concatenate(
        [
            rho[:ring].ravel(),
            rho[-ring:].ravel(),
            rho[:, :ring].ravel(),
            rho[:, -ring:].ravel(),
        ]
    )
    u_inf = float(ring_u.mean())
    rho_inf = float(ring_rho.mean())
    drag = float(np.sum(rho * ux * (u_inf - ux)))
    return drag, u_inf, rho_inf


def projected_area(solid_mask: np.ndarray) -> int:
    """Frontal projection: (y, z) columns containing at least one solid cell."""
    if solid_mask.ndim != 3:
        raise ValueError(f"expected a 3-D mask, got {solid_mask.shape}")
    return int((solid_mask != 0).any(axis=2).sum())


def wetted_area(solid_mask: np.ndarray) -> int:
    """Solid–fluid interface face count over the six lattice directions."""
    if solid_mask.ndim != 3:
        raise ValueError(f"expected a 3-D mask, got {solid_mask.shape}")
    m = solid_mask != 0
    pad = np.pad(m, 1)
    interior = (slice(1, -1),) * 3
    total = 0
    for axis in range(3):
        for shift in (1, -1):
            neighbour = np.roll(pad, shift, axis=axis)[interior]
            total += int((m & ~neighbour).sum())
    return total


def ittc_1957(re: float) -> float:
    """ITTC-1957 model-ship correlation line ``0.075 / (log10(Re) - 2)^2``."""
    if re <= 1.0e2:
        raise ValueError(f"ITTC-1957 needs Re > 100, got {re}")
    return 0.075 / (math.log10(re) - 2.0) ** 2


def blasius(re: float) -> float:
    """Laminar flat-plate line ``1.328 / sqrt(Re)``."""
    if re <= 0.0:
        raise ValueError(f"Re must be positive, got {re}")
    return 1.328 / math.sqrt(re)


@dataclass(frozen=True)
class PlaneDrag:
    """Drag on one surveyed plane, in lattice units."""

    offset: int
    drag: float
    u_inf: float
    rho_inf: float


@dataclass(frozen=True)
class SnapshotDrag:
    """One snapshot's plane survey and its normalised coefficient."""

    step: int
    planes: tuple[PlaneDrag, ...]
    c_d: float


@dataclass(frozen=True)
class DragSurvey:
    """Wake-survey result for one run (one ``fields.h5``)."""

    grid: tuple[int, int, int]
    s_proj: int
    s_wet: int
    snapshots: tuple[SnapshotDrag, ...]
    re: float | None = None

    def _reference(self) -> SnapshotDrag:
        if not self.snapshots:
            raise ValueError("survey has no snapshots")
        return self.snapshots[-1]

    @property
    def c_d_final(self) -> float:
        return self._reference().c_d

    @property
    def cf_equiv_final(self) -> float:
        return self.c_d_final * self.s_proj / self.s_wet

    @property
    def plane_spread_final(self) -> float:
        drags = [p.drag for p in self._reference().planes]
        return max(drags) - min(drags)

    @property
    def drift_last3(self) -> float | None:
        if len(self.snapshots) < 3:
            return None
        a, b = self.snapshots[-3].c_d, self.snapshots[-1].c_d
        return abs(b - a) / max(abs(a), 1.0e-12)

    def scaling_slope(self) -> float | None:
        """Log-log C_D(step-time) slope across snapshots — steady-state check."""
        if len(self.snapshots) < 3:
            return None
        x = np.log10([s.step for s in self.snapshots])
        y = np.log10([abs(s.c_d) for s in self.snapshots])
        return float(np.polyfit(x, y, 1)[0])


def _step_of(key: str) -> int:
    return int(key.rsplit("_", 1)[-1])


def survey_file(
    path: str | Path,
    *,
    planes: Sequence[int] = DEFAULT_PLANES,
    ring: int = DEFAULT_RING,
    reference_offset: int = DEFAULT_REFERENCE_OFFSET,
) -> DragSurvey:
    """Survey every snapshot of one ``fields.h5`` written by FieldSampleReporter."""
    import h5py  # lazy: the ``io`` extra

    with h5py.File(path, "r") as f:
        step_keys = sorted(f.keys(), key=_step_of)
        if not step_keys:
            raise ValueError(f"{path}: no step groups")
        mask = f[step_keys[0]]["solid_mask"][()] != 0
        nz, ny, nx = mask.shape
        if any(off >= nx for off in planes):
            raise ValueError(f"plane offsets {planes} exceed nx={nx}")
        snaps: list[SnapshotDrag] = []
        for key in step_keys:
            group = f[key]
            ux_full = group["ux"][()]
            rho_full = group["rho"][()]
            plane_rows = []
            ref = None
            for off in planes:
                xw = nx - 1 - off
                drag, u_inf, rho_inf = plane_drag(ux_full[:, :, xw], rho_full[:, :, xw], ring=ring)
                row = PlaneDrag(offset=int(off), drag=drag, u_inf=u_inf, rho_inf=rho_inf)
                plane_rows.append(row)
                if off == reference_offset:
                    ref = row
            if ref is None:
                raise ValueError(f"reference_offset {reference_offset} not among planes {planes}")
            c_d = 2.0 * ref.drag / (ref.rho_inf * ref.u_inf**2 * projected_area(mask))
            snaps.append(SnapshotDrag(step=_step_of(key), planes=tuple(plane_rows), c_d=c_d))
    return DragSurvey(
        grid=(nz, ny, nx),
        s_proj=projected_area(mask),
        s_wet=wetted_area(mask),
        snapshots=tuple(snaps),
    )


def survey_point(point_dir: str | Path, **kwargs: Any) -> DragSurvey:
    """Survey one scan point directory (``fields.h5`` + ``status.json``)."""
    point_dir = Path(point_dir)
    survey = survey_file(point_dir / "fields.h5", **kwargs)
    status = json.loads((point_dir / "status.json").read_text())
    params = status.get("params") or {}
    if "re" in params:
        object.__setattr__(survey, "re", float(params["re"]))
    return survey


def survey_dataset(out_dir: str | Path, **kwargs: Any) -> list[DragSurvey]:
    """Survey every point of a scan dataset directory, sorted by Re."""
    out_dir = Path(out_dir)
    point_dirs = [p for p in sorted((out_dir / "points").iterdir()) if (p / "fields.h5").is_file()]
    if not point_dirs:
        raise ValueError(f"{out_dir}: no point directories")
    surveys = [survey_point(p, **kwargs) for p in point_dirs]
    return sorted(surveys, key=lambda s: (s.re is None, s.re or 0.0))


def write_summary(out_dir: str | Path, surveys: Sequence[DragSurvey]) -> Path:
    """Write ``drag_summary.json`` for a dataset directory."""
    out_dir = Path(out_dir)
    payload = {
        "method": "wake momentum deficit, border-ring reference",
        "lattice_units": True,
        "points": [
            {
                "re": s.re,
                "grid": list(s.grid),
                "s_proj": s.s_proj,
                "s_wet": s.s_wet,
                "c_d_final": s.c_d_final,
                "cf_equiv_final": s.cf_equiv_final,
                "plane_spread_final": s.plane_spread_final,
                "drift_last3": s.drift_last3,
                "snapshots": [
                    {
                        "step": snap.step,
                        "c_d": snap.c_d,
                        "drag_planes": {str(p.offset): p.drag for p in snap.planes},
                        "u_inf_ring": snap.planes[0].u_inf if snap.planes else None,
                    }
                    for snap in s.snapshots
                ],
            }
            for s in surveys
        ],
    }
    path = out_dir / "drag_summary.json"
    path.write_text(json.dumps(payload, indent=1))
    return path
