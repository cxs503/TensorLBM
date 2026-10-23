"""Kinetic-hygiene diagnostics shared by every LBM backend.

The campaign's convicted failure mode is a period-2 (odd/even substep)
kinetic ghost: a mode that alternates sign between consecutive steps and
is invisible to per-step magnitude probes.  AMR interfaces are new
injection surfaces for it, so the probe must be callable from the uniform
runner, the static-block runner and the octree line with one convention.

Conventions (period-2 conviction signature unchanged since p15):

* the observable is the global x-momentum summed over the physical region,
  sampled once per substep in temporal order;
* ``mean_abs_d1`` / ``mean_abs_d2`` are the mean absolute one-step and
  two-step differences of that series;
* ``d2_over_d1`` ~ 0 indicates period-2 alternation, ~ 1 independently
  sampled (iid) white noise, ~ sqrt(2) an increment-dominated (random
  walk) series, ~ 2 a smooth trend — the healthy band lies anywhere the
  odd/even means agree, the conviction signature is the ratio collapse
  toward 0 together with a large ``odd_even_mean_difference``;
* ``odd_even_mean_difference`` is the absolute difference of the odd- and
  even-index substep means (the direct alternation amplitude).

Calibration note: the informal ``sqrt(2) = white noise`` reading belongs
to increment-dominated series; for iid samples the ratio is exactly 1.
Both regimes are pinned by unit tests
(``tests/test_kinetic_hygiene.py``).
"""

from __future__ import annotations

from typing import Sequence

import torch

from .d3q19 import C

__all__ = ["global_x_momentum", "period2_mx_probe"]

_CX_POS = [i for i in range(19) if float(C[i][0]) > 0.0]
_CX_NEG = [i for i in range(19) if float(C[i][0]) < 0.0]

PERIOD2_READING = (
    "d2_over_d1 ~ 0 -> period-2 alternation; ~ 1 -> iid white noise; "
    "~ sqrt(2) -> increment-dominated (random walk); ~ 2 -> smooth trend; "
    "odd_even_mean_difference is the direct alternation amplitude"
)


def global_x_momentum(f: torch.Tensor, physical_mask: torch.Tensor | None = None) -> float:
    """Global x-momentum over the physical (ghost-free) region.

    Args:
        f: populations (Q, nz, ny, nx).
        physical_mask: optional boolean (nz, ny, nx) restricting the sum to
            the physical region of an allocated (ghost-padded) grid.
    """
    if physical_mask is None:
        momentum = f[_CX_POS].sum() - f[_CX_NEG].sum()
    else:
        momentum = f[_CX_POS][:, physical_mask].sum() - f[_CX_NEG][:, physical_mask].sum()
    return float(momentum.item())


def period2_mx_probe(series: Sequence[float]) -> dict:
    """Period-2 hygiene statistics of a per-substep momentum series.

    Returns a JSON-friendly dict with the sample count, mean absolute
    one-/two-step differences, their ratio, the odd/even mean difference
    (absolute and normalised by the series mean magnitude) and the reading
    string.  Short series (fewer than three samples) yield ``None`` fields.
    """
    values = [float(v) for v in series]
    n = len(values)
    result: dict = {
        "samples": n,
        "mean_abs_d1": None,
        "mean_abs_d2": None,
        "d2_over_d1": None,
        "odd_even_mean_difference": None,
        "odd_even_mean_difference_over_abs_mean": None,
        "reading": PERIOD2_READING,
    }
    if n < 3:
        return result
    d1 = [abs(values[i + 1] - values[i]) for i in range(n - 1)]
    d2 = [abs(values[i + 2] - values[i]) for i in range(n - 2)]
    mean_d1 = sum(d1) / len(d1)
    mean_d2 = sum(d2) / len(d2)
    odd = values[0::2]
    even = values[1::2]
    odd_mean = sum(odd) / len(odd)
    even_mean = sum(even) / len(even)
    abs_mean = abs(sum(values) / n)
    difference = abs(odd_mean - even_mean)
    result["mean_abs_d1"] = mean_d1
    result["mean_abs_d2"] = mean_d2
    result["d2_over_d1"] = mean_d2 / mean_d1 if mean_d1 else None
    result["odd_even_mean_difference"] = difference
    result["odd_even_mean_difference_over_abs_mean"] = difference / abs_mean if abs_mean else None
    return result
