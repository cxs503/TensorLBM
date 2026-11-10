"""One-dimensional energy spectra from the native channel kernel's plane dumps.

The OpenMP kernel :mod:`tensorlbm.turbulence.les_chan` (driven through
:mod:`tensorlbm.turbulence.les_channel`) integrates a plane channel and, when
asked with ``--spec-every``/``--spec-rows``, appends raw velocity planes at
selected wall-normal rows to ``<prefix>_spec.bin`` next to a
``<prefix>_spec.json`` metadata record.  This module turns those dumps into
streamwise and spanwise energy spectra of the streamwise fluctuation and fits
the inertial-range slope.

Why the spectra are *not* accumulated in the kernel: an FFT per plane inside the
time loop would tie the C binary to an external FFT dependency, whereas the
dumps are a few hundred megabytes and post-processing is reproducible in numpy.
The dump layout is recorded in the metadata file (plane-major ``[nz][nx]``,
``ux`` then ``uy`` then ``uz``, little-endian ``float32``) so a reader cannot
silently transpose a plane.

Conventions
-----------
``E(k)`` is defined so that ``sum_k E(k) = <u'^2>`` on the plane (Parseval),
with ``k`` the integer Fourier mode index along the homogeneous direction and
``u' = u - <u>`` the plane-mean-removed streamwise velocity.  Normalisation to
wall units is the caller's job (divide by ``u_tau**2``); the profiles and
``y_plus`` come from the kernel artefact.

The log-log slope fit uses the same *a-priori* band rule as
``tensorlbm.turbulence.les_channel3d.fit_power_law_slope`` (``k in
[k_min, max(k_min + 8, 0.25 k_max)]``) so the reported slope cannot be tuned to
``-5/3`` by moving the band.  Keeping this copy numpy-only lets the spectra be
post-processed without importing torch; ``tests/test_channel_spectra.py`` pins
the two implementations to each other whenever torch is available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "PlaneDumpMeta",
    "fit_loglog_slope",
    "load_plane_dump",
    "plane_spectra_legs",
    "plane_spectrum",
    "read_plane_meta",
]


@dataclass(frozen=True)
class PlaneDumpMeta:
    """Metadata emitted by the kernel next to the binary plane dump."""

    nx: int
    nz: int
    rows: tuple[int, ...]
    snapshots: int
    steps_between: int
    bytes_per_snapshot: int
    layout: str

    @property
    def bytes_per_row_component(self) -> int:
        return self.nx * self.nz * 4

    def validate(self) -> None:
        if self.nx < 8 or self.nz < 8:
            raise ValueError("plane dump needs nx, nz >= 8")
        if self.snapshots < 1:
            raise ValueError("plane dump has no snapshots")
        if not self.rows:
            raise ValueError("plane dump lists no rows")
        expected = len(self.rows) * 3 * self.bytes_per_row_component
        if self.bytes_per_snapshot != expected:
            raise ValueError(
                f"bytes_per_snapshot={self.bytes_per_snapshot} does not match "
                f"3 components x {len(self.rows)} rows x {self.bytes_per_row_component} B"
            )


def read_plane_meta(path: str | Path) -> PlaneDumpMeta:
    """Read and validate ``<prefix>_spec.json``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = PlaneDumpMeta(
        nx=int(payload["nx"]),
        nz=int(payload["nz"]),
        rows=tuple(int(r) for r in payload["rows"]),
        snapshots=int(payload["snapshots"]),
        steps_between=int(payload["steps_between"]),
        bytes_per_snapshot=int(payload["bytes_per_snapshot"]),
        layout=str(payload.get("layout", "")),
    )
    meta.validate()
    return meta


def load_plane_dump(
    bin_path: str | Path, meta: PlaneDumpMeta
) -> dict[int, dict[str, np.ndarray]]:
    """Load the dump into ``{row: {"ux": (S, nz, nx), "uy": ..., "uz": ...}}``."""
    raw = np.fromfile(Path(bin_path), dtype="<f4")
    expected = meta.snapshots * meta.bytes_per_snapshot // 4
    if raw.size != expected:
        raise ValueError(f"dump has {raw.size} floats, metadata implies {expected}")
    out: dict[int, dict[str, np.ndarray]] = {}
    per_snap = len(meta.rows) * 3 * meta.nx * meta.nz
    for ip, row in enumerate(meta.rows):
        planes = {"ux": [], "uy": [], "uz": []}
        for name, comp in (("ux", 0), ("uy", 1), ("uz", 2)):
            chunks = []
            for s in range(meta.snapshots):
                start = s * per_snap + (ip * 3 + comp) * meta.nx * meta.nz
                chunks.append(
                    raw[start : start + meta.nx * meta.nz].reshape(meta.nz, meta.nx)
                )
            planes[name] = np.stack(chunks, axis=0).astype(np.float64)
        out[row] = planes
    return out


def plane_spectrum(planes: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Parseval-normalised 1-D spectrum of ``u'`` averaged over snapshots.

    ``planes`` has shape ``(S, nz, nx)``; ``axis=-1`` gives the streamwise
    spectrum (averaged over ``z`` and snapshots), ``axis=-2`` the spanwise one.
    Returns ``(k, E)`` for the non-negative mode indices ``k = 0..N//2`` with
    ``sum_k E(k) = <u'^2>``.
    """
    arr = np.asarray(planes, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("planes must have shape (S, nz, nx)")
    arr = np.moveaxis(arr, axis, -1)  # (S, transverse, n)
    n = arr.shape[-1]
    acc = np.zeros(n // 2 + 1, dtype=np.float64)
    for snap in arr:
        fluc = snap - snap.reshape(-1).mean()
        spec = np.fft.rfft(fluc, axis=-1)
        power = (spec.real**2 + spec.imag**2) / (n * n)
        # fold the transverse direction(s) into the estimate
        power = power.reshape(-1, power.shape[-1]).mean(axis=0)
        # one-sided spectrum: double the interior modes so sum_k E = <u'^2>
        weight = np.full(power.shape, 2.0)
        weight[0] = 1.0
        if n % 2 == 0:
            weight[-1] = 1.0
        acc += weight * power
    return np.arange(acc.size, dtype=np.float64), acc / arr.shape[0]


def fit_loglog_slope(
    k: np.ndarray,
    energy: np.ndarray,
    *,
    k_min: float = 4.0,
    k_max: float | None = None,
    min_points: int = 4,
) -> dict:
    """Least-squares slope of ``log E`` vs ``log k`` in a fixed band.

    Mirrors ``les_channel3d.fit_power_law_slope`` (default band
    ``k_max = max(k_min + 8, 0.25 k_available)``).  ``valid`` is False when the
    band holds fewer than ``min_points`` positive modes.
    """
    k = np.asarray(k, dtype=np.float64).reshape(-1)
    e = np.asarray(energy, dtype=np.float64).reshape(-1)
    if k.size == 0 or e.size != k.size:
        return {"slope": float("nan"), "r2": float("nan"), "valid": False, "n_points": 0}
    if k_max is None:
        k_max = max(k_min + 8.0, 0.25 * float(k.max()))
    sel = (k >= k_min) & (k <= k_max) & (e > 0.0)
    kk, ee = k[sel], e[sel]
    if kk.size < min_points:
        return {
            "slope": float("nan"),
            "r2": float("nan"),
            "valid": False,
            "n_points": int(kk.size),
            "k_min": float(k_min),
            "k_max": float(k_max),
        }
    lx, ly = np.log(kk), np.log(ee)
    design = np.stack([np.ones_like(lx), lx], axis=1)
    coef, *_ = np.linalg.lstsq(design, ly, rcond=None)
    resid = ly - design @ coef
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {
        "slope": float(coef[1]),
        "intercept": float(coef[0]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        "valid": True,
        "n_points": int(kk.size),
        "k_min": float(k_min),
        "k_max": float(k_max),
    }


def plane_spectra_legs(
    dump: dict[int, dict[str, np.ndarray]],
    *,
    u_tau: float,
    nu: float,
    k_min: float = 4.0,
) -> dict:
    """Build the ``les.spectra``-compatible spectrum block for the gate layer.

    Every dumped row becomes one *leg* with ``y_plus``, the streamwise and
    spanwise spectra (in wall units, ``E / u_tau**2``) and their banded fits.
    """
    if u_tau <= 0.0 or nu <= 0.0:
        raise ValueError("u_tau and nu must be positive")
    streamwise: list[dict] = []
    spanwise: list[dict] = []
    for row in sorted(dump):
        y_plus = (float(row) - 0.5) * u_tau / nu
        ux = dump[row]["ux"]
        k_x, e_x = plane_spectrum(ux, axis=-1)
        k_z, e_z = plane_spectrum(ux, axis=-2)
        e_x = e_x / u_tau**2
        e_z = e_z / u_tau**2
        streamwise.append(
            {
                "row": int(row),
                "y_plus": y_plus,
                "k": k_x[1:].tolist(),
                "E": e_x[1:].tolist(),
                "fit": fit_loglog_slope(k_x[1:], e_x[1:], k_min=k_min),
                "tke_plus": float(np.sum(e_x[1:])),
            }
        )
        spanwise.append(
            {
                "row": int(row),
                "y_plus": y_plus,
                "k": k_z[1:].tolist(),
                "E": e_z[1:].tolist(),
                "fit": fit_loglog_slope(k_z[1:], e_z[1:], k_min=k_min),
                "tke_plus": float(np.sum(e_z[1:])),
            }
        )
    return {
        "planes_y_plus": [leg["y_plus"] for leg in streamwise],
        "samples": int(next(iter(dump.values()))["ux"].shape[0]) if dump else 0,
        "streamwise": streamwise,
        "spanwise": spanwise,
    }