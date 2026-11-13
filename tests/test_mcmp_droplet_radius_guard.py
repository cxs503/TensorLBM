"""Guard test for the droplet-radius crossing scan (audit m18359).

Context
-------
Audit m18359 re-ran the shipped quick check of mission m18326
(``xval/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD/quick_smoke.py``, ``G_12 = -1.2``,
i.e. below the two-component Shan-Chen separation threshold) and it crashed:

    File "src/tensorlbm/mcmp_verification.py", line 409, in droplet_radius
        r_in, r_out = ray[i_in], ray[i_in + 1]
    IndexError: list index out of range

Root cause: ``inside[-1]`` is the *last* node on the liquid side of ``rho_av``.
When the field never separated, noise puts that node on the truncated end of the
half-domain ray, so ``ray[i_in + 1]`` does not exist.  The fix records a
``truncated_crossing`` guardrail and skips the ray instead of raising, which is
the documented contract of this function (a missing crossing returns
``radius = None`` plus a guardrail, never an exception).
"""

from __future__ import annotations

import math

import torch

from tensorlbm.mcmp_verification import droplet_radius


def _codes(result: dict) -> list[str]:
    return [g.code for g in result["guardrails"]]


def test_non_separated_noise_field_returns_nan_not_indexerror() -> None:
    """Regression of the m18326 quick-check crash: no interface -> no radius."""
    n, c = 64, 32
    # near-uniform mixture at rho_av with a few denser nodes: the centre and the
    # truncated end of every ray sit on the liquid side of rho_av while the rest
    # of the ray does not -> inside[-1] == len(ray) - 1 (the IndexError trigger)
    field = torch.full((n, n), 0.45, dtype=torch.float32)
    field[c, c] = field[c, 0] = field[c, -1] = 0.55
    field[0, c] = field[-1, c] = 0.55

    result = droplet_radius(field, 0.5)

    assert result["radius"] is None
    assert result["radius_spread"] is None
    assert "truncated_crossing" in _codes(result)


def test_uniform_field_single_phase_still_guarded() -> None:
    """All-liquid field: the pre-existing 'never leaves the droplet' branch."""
    result = droplet_radius(torch.full((32, 32), 0.7, dtype=torch.float32), 0.5)
    assert result["radius"] is None
    assert "no_crossing" in _codes(result)


def test_synthetic_droplet_radius_still_measured() -> None:
    """The guard must not disturb the well-resolved measurement path."""
    n, radius_true, width = 96, 16.0, 2.0
    c = n // 2
    y, x = torch.meshgrid(torch.arange(n, dtype=torch.float32),
                          torch.arange(n, dtype=torch.float32), indexing="ij")
    r = torch.sqrt((x - c) ** 2 + (y - c) ** 2)
    liq, gas = 0.95, 0.14
    field = gas + (liq - gas) * 0.5 * (1.0 - torch.tanh((r - radius_true) / width))
    rho_av = 0.5 * (liq + gas)

    result = droplet_radius(field, rho_av)

    assert result["crossings"] == 4
    assert result["radius"] is not None
    assert math.isclose(result["radius"], radius_true, rel_tol=0.02)
    assert result["radius_spread"] < 1.0