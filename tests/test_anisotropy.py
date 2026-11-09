"""Directional (anisotropy) tests for the isotropic gradient stencils (P143W2-N01).

The Shan-Chen surface tension is fed by the gradient of the pseudopotential, so
any lattice anisotropy of that operator shows up as a *directional* artefact of
the measured pressure jump.  Two things are checked here:

* the stencil moments (unit gain ``a1 = cs^2`` and an isotropic fourth moment),
* the directional error of the gradient operator, which is what the 4-shell
  stencil is supposed to reduce,

plus, when the campaign artifact is present, the azimuthal spread actually
measured on the droplets.
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.sc_surface_tension import (  # noqa: E402
    build_stencil,
    gradient_spectral,
    gradient_stencil,
    shell_moments,
)


def test_stencil_moments_are_isotropic():
    for name in ("d2q9", "shell4"):
        st = build_stencil(name)
        assert st["a1"] == pytest.approx(1.0 / 3.0, abs=1e-12)
        assert st["isotropy_residual"] < 1e-12, name  # S4 == 3 Sxy exactly
    # kappa3 (the Korteweg coefficient ratio) of the two stencils
    assert build_stencil("d2q9")["kappa3"] == pytest.approx(1.0 / 6.0)
    assert build_stencil("shell4")["kappa3"] == pytest.approx(11.0 / 48.0, abs=1e-12)
    assert build_stencil("shell4")["kappa3"] / build_stencil("d2q9")["kappa3"] == pytest.approx(1.375)


def test_isotropy_is_a_weighted_sum_property():
    """No single square-lattice shell is fourth-moment isotropic; isotropy is a
    property of the *weighted sum* of shells, which is why the D2Q9 weights
    (1/9, 1/36) work and the axis-only stencil does not."""
    assert shell_moments(1)["Sxy"] == 0.0 and shell_moments(1)["S4"] > 0.0
    for k in (2, 4, 8):
        m = shell_moments(k)
        assert m["S4"] != pytest.approx(3.0 * m["Sxy"]), k
    # ... but the sums used by both stencils satisfy S4 == 3 Sxy exactly
    for name in ("d2q9", "shell4"):
        assert build_stencil(name)["isotropy_residual"] < 1e-12


def _lattice_wave(nx: int, ny: int, L: int = 64):
    """A *periodic* lattice wave with lattice mode (nx, ny) and its exact gradient.

    The wavevector must be a reciprocal-lattice vector, otherwise the wrapped
    derivative operators are being asked to differentiate a discontinuous field.
    """
    y = torch.arange(L, dtype=torch.float32).view(L, 1)
    x = torch.arange(L, dtype=torch.float32).view(1, L)
    kx = 2.0 * math.pi * nx / L
    ky = 2.0 * math.pi * ny / L
    arg = kx * x + ky * y
    f = torch.cos(arg)
    gx = -kx * torch.sin(arg)
    gy = -ky * torch.sin(arg)
    return f, gx, gy, math.hypot(kx, ky)


def _plane_wave_errors(stencil_name: str, L: int = 64):
    st = build_stencil(stencil_name)
    modes = [(3, 0), (2, 2), (0, 3), (2, 1), (-1, 2), (3, 1), (-2, 3), (1, 3)]
    errs = []
    for nx, ny in modes:
        f, ex, ey, kmag = _lattice_wave(nx, ny, L)
        gx, gy = gradient_stencil(f, st)
        errs.append(float(torch.sqrt((gx - ex) ** 2 + (gy - ey) ** 2).max()) / kmag)
    return errs


def test_gradient_operators_are_second_order_accurate():
    for name in ("d2q9", "shell4"):
        errs = _plane_wave_errors(name)
        assert max(errs) < 0.08, (name, max(errs))


def test_directional_error_of_the_gradients():
    """The 4-shell stencil enters one order higher in the truncation, so its
    directional (angle-to-angle) error must be smaller than the D2Q9 one."""
    e9 = _plane_wave_errors("d2q9")
    e4 = _plane_wave_errors("shell4")
    spread9 = (max(e9) - min(e9)) / max(e9)
    spread4 = (max(e4) - min(e4)) / max(e4)
    assert spread4 < spread9, (spread9, spread4)


def test_spectral_gradient_is_rotation_covariant():
    """A spectrally exact gradient has no lattice anisotropy: the magnitude field
    of the gradient must be invariant under a 90-degree rotation of the input."""
    L = 64
    y = torch.arange(L, dtype=torch.float32).view(L, 1)
    x = torch.arange(L, dtype=torch.float32).view(1, L)
    r = torch.sqrt((x - 0.5 * (L - 1)) ** 2 + (y - 0.5 * (L - 1)) ** 2)
    f = torch.exp(-(r / 9.0) ** 2)
    gx, gy = gradient_spectral(f)
    gx90, gy90 = gradient_spectral(torch.rot90(f, 1).contiguous())
    mag = torch.sqrt(gx ** 2 + gy ** 2)
    mag90 = torch.sqrt(gx90 ** 2 + gy90 ** 2)
    err = float((mag90 - torch.rot90(mag, 1)).abs().max())
    assert err < 1e-5, err


def test_spectral_gradient_is_exact_on_a_lattice_wave():
    L = 64
    f, ex, ey, _ = _lattice_wave(3, 2, L)
    gx, gy = gradient_spectral(f)
    assert float((gx - ex).abs().max()) < 1e-4
    assert float((gy - ey).abs().max()) < 1e-4


def test_spectral_gradient_matches_stencil_for_smooth_fields():
    L = 64
    y = torch.arange(L, dtype=torch.float32).view(L, 1)
    x = torch.arange(L, dtype=torch.float32).view(1, L)
    r = torch.sqrt((x - 31.5) ** 2 + (y - 31.5) ** 2)
    f = torch.exp(-(r / 9.0) ** 2)
    gs = gradient_spectral(f)
    for name in ("d2q9", "shell4"):
        g = gradient_stencil(f, build_stencil(name))
        err = torch.sqrt((g[0] - gs[0]) ** 2 + (g[1] - gs[1]) ** 2)
        assert float(err.max()) < 5e-3, name


def _load_gate() -> dict | None:
    p = REPO / "artifacts" / "sc_surface_tension" / "laplace_gate.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def test_measured_azimuthal_spread_from_artifact():
    gate = _load_gate()
    if gate is None:
        pytest.skip("campaign artifact not present")
    aniso = gate["anisotropy"]
    for key, row in aniso.items():
        # a directional artefact above 5 % would mean the measurement is not
        # converged / not isotropic; the campaign value is 0.7-1.3 %
        assert row["spread_flux_pct"] < 5.0, key
    sh = [v["spread_flux_pct"] for v in aniso.values() if v["stencil"] == "shell4"]
    d2 = [v["spread_flux_pct"] for v in aniso.values() if v["stencil"] == "d2q9"]
    if sh and d2:
        assert sum(sh) / len(sh) < sum(d2) / len(d2), (sh, d2)


def test_measured_kappa3_scales_sigma_from_artifact():
    """The Korteweg anchor must be larger for the 4-shell stencil by roughly the
    predicted kappa3 ratio (a 10-20 % excess of the measured ratio over 1.375 is
    expected because of the O(d^5) truncation terms)."""
    gate = _load_gate()
    if gate is None:
        pytest.skip("campaign artifact not present")
    d2 = gate["laplace_d2q9"]["per_radius"]
    sh = gate["laplace_shell4"]["per_radius"]
    if not d2 or not sh:
        pytest.skip("no rows")
    sk9 = sum(r["sigma_K"] for r in d2) / len(d2)
    sk4 = sum(r["sigma_K"] for r in sh) / len(sh)
    ratio = sk4 / sk9
    assert 1.1 < ratio < 1.8, ratio