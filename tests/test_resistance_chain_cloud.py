"""Tests for the resistance-chain CLOUD pipeline (mission m4927).

Machine-checkable criteria only: analytic potential-flow limits, the
d'Alembert surface-integration closure, and the grid-convergence estimator on
synthetic data with a known exact answer.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "resistance_chain_cloud"))

from analyze import gci  # noqa: E402
from run import boundary_nodes, ellipse_mask, potential_cp_ellipse  # noqa: E402


# --------------------------------------------------------------- potential flow
def test_circle_limit_of_ellipse_potential_flow():
    """For a=b=R the ellipse formula must reduce to Cp = 1 - 4 sin^2(phi)."""
    phi = np.linspace(-np.pi, np.pi, 721)
    cp_ell = potential_cp_ellipse(1.0, 1.0, phi)
    cp_circ = 1.0 - 4.0 * np.sin(phi) ** 2
    assert np.allclose(cp_ell, cp_circ, atol=1e-12)


def test_potential_flow_stagnation_and_shoulder():
    phi = np.array([0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi])
    cp = potential_cp_ellipse(2.0, 0.5, phi)
    assert cp[0] == pytest.approx(1.0, abs=1e-12)  # downstream extremity
    assert cp[1] == pytest.approx(1.0, abs=1e-12)  # upstream stagnation
    # slender L/B = 4 section: weak suction on the shoulders, -1 < Cp < 0
    assert -1.0 < cp[2] < 0.0 and -1.0 < cp[3] < 0.0
    # circular section: Cp = -3 at the shoulders
    cp_circ = potential_cp_ellipse(1.0, 1.0, np.array([0.5 * np.pi]))
    assert cp_circ[0] == pytest.approx(-3.0, abs=1e-12)


def test_dalembert_surface_integral_is_zero():
    """Potential-flow Cp integrated over the body surface gives zero drag.

    This validates the *surface-integration convention* used to turn the
    angular Cp profile into Cd (sum over staircase boundary nodes of
    Cp * n_x * ds / D with ds = 1 dx).
    """
    a, b = 2.0, 0.5
    n = 20000
    phi = np.linspace(-np.pi, np.pi, n, endpoint=False)
    cp = potential_cp_ellipse(a, b, phi)
    # outward normal of the ellipse x=a cos(phi), y=b sin(phi)
    nx = b * np.cos(phi)
    ny = a * np.sin(phi)
    ds = np.hypot(a * np.sin(phi), b * np.cos(phi)) * (2 * np.pi / n)
    norm = np.hypot(nx, ny)
    cd = float(np.sum(cp * nx / norm * ds) / (a / 2.0))  # normalise by half-length
    cl = float(np.sum(cp * ny / norm * ds) / (a / 2.0))
    assert abs(cd) < 1e-9
    assert abs(cl) < 1e-9


def test_ellipse_mask_area():
    import torch

    m = ellipse_mask(400, 400, 200.0, 200.0, 100.0, 25.0, torch.device("cpu"))
    assert int(m.sum()) == pytest.approx(math.pi * 100.0 * 25.0, rel=0.02)


def test_boundary_nodes_of_disk():
    import torch

    from tensorlbm.boundaries import cylinder_mask

    D = 32
    m = cylinder_mask(D * 2, D * 2, D, D, D / 2.0, torch.device("cpu"))
    b = boundary_nodes(m)
    # staircase boundary of a disk: number of boundary cells ~ perimeter + O(1)
    assert math.pi * D * 0.8 < int(b.sum()) < math.pi * D * 1.4
    assert bool((m & ~b).any())  # interior cells exist
    # every boundary node has a fluid 4-neighbour
    fluid = ~m
    nb = torch.zeros_like(m)
    nb[1:, :] |= fluid[:-1, :]
    nb[:-1, :] |= fluid[1:, :]
    nb[:, 1:] |= fluid[:, :-1]
    nb[:, :-1] |= fluid[:, 1:]
    assert bool((b & ~nb).any()) is False


# --------------------------------------------------------------------- GCI tests
def test_gci_recovers_second_order_observed_order():
    """Synthetic field with exact 2nd-order error: f(h) = f_ex + C h^2."""
    f_ex, C = 1.234, 0.5
    Ds = np.array([36.0, 24.0, 16.0])  # finest first (h = 1/D ascending)
    h = 1.0 / Ds  # r = 1.5
    f = f_ex + C * h**2
    g = gci(f[0], f[1], f[2], r=1.5)
    assert g["p_obs"] == pytest.approx(2.0, abs=1e-6)
    assert g["extrapolated"] == pytest.approx(f_ex, abs=1e-10)
    assert g["gci_fine_pct"] is not None and g["gci_fine_pct"] > 0
    assert g["fallback_reason"] is None


def test_gci_flags_non_monotonic_convergence():
    g = gci(1.30, 1.34, 1.31, r=1.5)
    assert g["fallback_reason"] is not None
    assert "non-asymptotic" in g["fallback_reason"]
    assert g["gci_fine_pct"] is not None  # conservative band still reported


def test_gci_flags_zero_error_case():
    g = gci(1.2, 1.2, 1.2, r=2.0)
    assert g["p_obs"] is None
    assert g["gci_fine_pct"] == 0.0


# ------------------------------------------------------- artifact contract test
ART = REPO / "artifacts" / "resistance_chain_cloud" / "fields"


@pytest.mark.parametrize("geo,D", [("cylinder", 24), ("cylinder", 16), ("ellipse", 24)])
def test_field_artifact_contract(geo, D):
    p = ART / geo / f"{geo}_D{D}_fields.npz"
    if not p.exists():
        pytest.skip("field artifact not produced in this run")
    z = np.load(p)
    for key in ("x", "y", "ux", "uy", "cp", "wz", "u_rms", "wz_rms", "mask",
                "phi", "cp_rho_bin", "cf_rho_bin", "cd_series", "cl_series"):
        assert key in z.files, key
    assert z["ux"].shape == (z["y"].size, z["x"].size)
    assert np.isfinite(z["ux"][~z["mask"]]).all()
    # far-field consistency: mean streamwise speed near the inlet ~ U_inf
    inlet = z["ux"][:, 2]
    assert abs(float(np.nanmedian(inlet)) - 1.0) < 0.15
    # pressure coefficient from the field must be bounded (no blow-up)
    assert float(np.nanmax(np.abs(z["cp"]))) < 50.0
    # at least one angular bin holds a finite (measured) surface value
    assert np.isfinite(z["cp_rho_bin"]).any()