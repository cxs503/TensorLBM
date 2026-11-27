"""Independent closure check for the Zou/He channel reconstruction.

Anchor: ``XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION``.

The existing regression (``test_zou_he_channel_reconstruction.py``) asserts the
moments of the reconstructed column against *its own* rho target, i.e. it
executes the same algebra it checks.  This test instead solves the Zou/He
boundary value problem as a 4x4 linear system (mass, x- and y-momentum,
non-equilibrium bounce-back on the normal link, see Zou & He 1997 §II) with
``numpy.linalg.solve`` over incoming/known direction sets obtained by counting
``sign(c_x)``, and requires the repo's closed-form layers to match it to
round-off.

Reference: Zou, Q. & He, X. (1997) *Phys. Fluids* 9(6) 1592-1603, §II;
Krüger et al. (2017) *The Lattice Boltzmann Method: Principles and Practice*,
§8.2.2.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

XVAL = Path(__file__).resolve().parents[1] / "development" / "verification" / "zou_he_channel_2d"

pytest.importorskip("numpy")
pytest.importorskip("torch")

if not (XVAL / "independent_zouhe.py").is_file():  # pragma: no cover
    pytest.skip("xval module not present", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("independent_zouhe", XVAL / "independent_zouhe.py")
ind = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ind)


def test_repo_layers_equal_independent_linear_solve() -> None:
    res = ind.check()
    for name, val in res.items():
        assert val["max_abs_diff"] < 1e-12, f"{name}: {val}"


def test_independent_solve_reproduces_prescribed_moments() -> None:
    """The solve must hit (rho, ux, uy) exactly, for both closure modes."""
    import numpy as np

    rng = np.random.default_rng(19970317)
    layer = rng.random((9, 1)) + 0.1
    for side, mode, kw in (
        ("west", "velocity", dict(ux=0.05, uy=-0.01)),
        ("west", "pressure", dict(rho=1.004, uy=0.02)),
        ("east", "pressure", dict(rho=0.996, uy=-0.02)),
    ):
        out = ind.solve_layer(layer, side, mode=mode, **kw)  # asserts moments
        c = ind.C.cpu().numpy().astype(float)
        assert out.shape == (9, 1)
        assert np.isfinite(out).all()
        # known (out-going / tangential) populations survive bit-exactly
        c = ind.C.cpu().numpy().astype(float)
        sign = +1.0 if side == "west" else -1.0
        known = [i for i in range(9) if np.sign(c[i, 0]) != sign]
        np.testing.assert_array_equal(out[known, 0], layer[known, 0])
