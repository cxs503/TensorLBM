"""Integrity gate for channel-DNS reference tables (wall-unit profiles).

A reference table is an *input* to every error number in this benchmark, so it
has to be checked before it is trusted.  Two machine checks catch the failure
modes that actually occur in practice when DNS data are redistributed:

1. **Shape**: a fully developed channel mean velocity profile increases
   monotonically from the wall to the centreline, and its centreline value is
   ``~ (1/kappa) ln(Re_tau) + B + wake`` (17.7 for ``Re_tau = 180``,
   21.0 for ``Re_tau = 395``).  A table whose ``u+`` peaks below the centreline
   and then falls back is a different flow or a corrupted/extrapolated file.
2. **Stress**:``|<u'v'>+| <= 1`` everywhere (it is a correlation bounded by the
   wall stress) and its peak sits at ``y+ = 25...60`` with a value of
   ``0.6...0.85``.  A monotone ramp in the stored correlation is unphysical.

The gate is deliberately conservative: it rejects rather than repairs, and every
rejection carries the measured numbers that caused it.  Reference tables that
fail the gate are reported as ``WITHHELD`` downstream -- never used silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import B_DEFAULT, KAPPA_DEFAULT

__all__ = ["ReferenceGateResult", "validate_channel_dns_table"]


@dataclass(frozen=True)
class ReferenceGateResult:
    path: str
    re_tau: float
    verdict: str
    reasons: list[str]
    metrics: dict

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "re_tau": float(self.re_tau),
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "metrics": self.metrics,
        }


def validate_channel_dns_table(
    path: str | Path,
    *,
    re_tau: float,
    kappa: float = KAPPA_DEFAULT,
    b: float = B_DEFAULT,
) -> ReferenceGateResult:
    """Check a ``(y_over_h, y_plus, u_plus[, uv_plus])`` table for physical consistency."""

    csv_path = Path(path)
    reasons: list[str] = []
    metrics: dict = {}

    if not csv_path.exists():
        return ReferenceGateResult(str(csv_path), float(re_tau), "REJECTED", ["file missing"], metrics)

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    names = set(data.dtype.names or ())
    eta_name = "y_over_h" if "y_over_h" in names else ("y" if "y" in names else None)
    for required in ("y_plus", "u_plus"):
        if required not in names:
            reasons.append(f"missing column {required!r} (columns: {sorted(names)})")
    if reasons:
        return ReferenceGateResult(str(csv_path), float(re_tau), "REJECTED", reasons, metrics)

    order = np.argsort(np.asarray(data["y_plus"], dtype=float))
    y_plus = np.asarray(data["y_plus"], dtype=float)[order]
    u_plus = np.asarray(data["u_plus"], dtype=float)[order]
    eta = np.asarray(data[eta_name], dtype=float)[order] if eta_name else y_plus / float(re_tau)

    # --- 1. monotone mean profile ---------------------------------------
    du = np.diff(u_plus)
    decreasing = np.where(du < -1.0e-6)[0]
    metrics["n_points"] = int(y_plus.size)
    metrics["monotone_violations"] = int(decreasing.size)
    if decreasing.size:
        i = int(decreasing[0])
        reasons.append(
            "u+ is not monotone: first decrease at y+ = {:.3f} ({:.3f} -> {:.3f}), "
            "max backward step {:.3f}".format(
                y_plus[i + 1], u_plus[i], u_plus[i + 1], float(np.min(du))
            )
        )
    metrics["u_plus_centreline"] = float(u_plus[-1])
    metrics["y_plus_centreline"] = float(y_plus[-1])
    metrics["eta_centreline"] = float(eta[-1])

    expected = float(np.log(float(re_tau)) / kappa + b)
    metrics["u_plus_centreline_expected_log_law"] = expected
    metrics["centreline_rel_error_vs_log_law"] = abs(u_plus[-1] - expected) / expected
    if metrics["centreline_rel_error_vs_log_law"] > 0.15:
        reasons.append(
            "centreline u+ = {:.2f} vs log-law estimate {:.2f} "
            "({:.1%} > 15%: wrong Re_tau, wrong normalisation, or not a channel)".format(
                u_plus[-1], expected, metrics["centreline_rel_error_vs_log_law"]
            )
        )

    # --- 2. stored shear-stress column (when present) -------------------
    if "uv_plus" in names:
        uv = np.asarray(data["uv_plus"], dtype=float)[order]
        metrics["uv_plus_max"] = float(np.max(np.abs(uv)))
        metrics["uv_plus_peak_y_plus"] = float(y_plus[int(np.argmax(np.abs(uv)))])
        if metrics["uv_plus_max"] > 1.0:
            reasons.append(
                "stored |uv_plus| reaches {:.3f} > 1 at y+ = {:.1f}: "
                "a correlation cannot exceed the wall stress".format(
                    metrics["uv_plus_max"], metrics["uv_plus_peak_y_plus"]
                )
            )
        if not (20.0 <= metrics["uv_plus_peak_y_plus"] <= 60.0):
            reasons.append(
                "stored uv_plus peaks at y+ = {:.1f}, expected 20...60".format(
                    metrics["uv_plus_peak_y_plus"]
                )
            )

    verdict = "PASS" if not reasons else "REJECTED"
    return ReferenceGateResult(str(csv_path), float(re_tau), verdict, reasons, metrics)