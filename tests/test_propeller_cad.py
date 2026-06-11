"""Tests for the propeller_cad module (Wageningen B-series)."""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# B-series open-water values
# ---------------------------------------------------------------------------

def test_kt_positive_at_zero_advance():
    from tensorlbm.propeller_cad import wageningen_b_series

    r = wageningen_b_series(0.0, 1.0, 0.6, 4)
    assert r["KT"] > 0.0, "KT must be positive at J=0"


def test_kt_decreases_with_J():
    from tensorlbm.propeller_cad import wageningen_b_series

    kts = [wageningen_b_series(J, 1.0, 0.6, 4)["KT"] for J in [0.0, 0.3, 0.6, 0.9]]
    assert all(kts[i] > kts[i + 1] for i in range(len(kts) - 1)), "KT must decrease with J"


def test_efficiency_zero_at_J_zero():
    from tensorlbm.propeller_cad import wageningen_b_series

    r = wageningen_b_series(0.0, 1.0, 0.6, 4)
    assert r["eta_0"] == pytest.approx(0.0)


def test_efficiency_peak_reasonable():
    """Peak efficiency for B4-60 at P/D=1.0 should be in [0.55, 0.80]."""
    import numpy as np

    from tensorlbm.propeller_cad import wageningen_b_series

    etas = [
        wageningen_b_series(float(J), 1.0, 0.6, 4)["eta_0"]
        for J in np.linspace(0.01, 1.2, 50)
    ]
    peak = max(etas)
    assert 0.55 < peak < 0.80, f"Peak efficiency {peak:.3f} outside expected range"


def test_kq_positive():
    from tensorlbm.propeller_cad import wageningen_b_series

    r = wageningen_b_series(0.7, 1.0, 0.6, 4)
    assert r["KQ"] > 0.0


def test_more_blades_increases_kt():
    """More blades → higher KT at same J (for same P/D and Ae/A0)."""
    from tensorlbm.propeller_cad import wageningen_b_series

    kt3 = wageningen_b_series(0.5, 1.0, 0.6, 3)["KT"]
    kt5 = wageningen_b_series(0.5, 1.0, 0.6, 5)["KT"]
    assert kt5 > kt3, "More blades should give higher KT"


def test_higher_P_D_increases_KT():
    from tensorlbm.propeller_cad import wageningen_b_series

    kt_low = wageningen_b_series(0.5, 0.8, 0.6, 4)["KT"]
    kt_high = wageningen_b_series(0.5, 1.2, 0.6, 4)["KT"]
    assert kt_high > kt_low


def test_dimensional_output():
    from tensorlbm.propeller_cad import wageningen_b_series

    r = wageningen_b_series(0.7, 1.0, 0.6, 4, n=2.0, D=5.0, rho=1025.0)
    assert "T_N" in r
    assert "Q_Nm" in r
    assert "P_kW" in r
    assert r["T_N"] > 0.0


# ---------------------------------------------------------------------------
# optimal_advance_ratio
# ---------------------------------------------------------------------------

def test_optimal_advance_ratio():
    from tensorlbm.propeller_cad import optimal_advance_ratio

    result = optimal_advance_ratio(1.0, 0.6, 4)
    assert "J_opt" in result
    assert "eta_max" in result
    assert 0.4 < result["J_opt"] < 1.2
    assert 0.5 < result["eta_max"] < 0.85


# ---------------------------------------------------------------------------
# propeller_design
# ---------------------------------------------------------------------------

def test_propeller_design_diameter_positive():
    from tensorlbm.propeller_cad import propeller_design

    result = propeller_design(500_000, 6.0, 1.0, 0.6, 4, 2.0)
    assert result["D_m"] > 0.0


def test_propeller_design_power_positive():
    from tensorlbm.propeller_cad import propeller_design

    result = propeller_design(500_000, 6.0, 1.0, 0.6, 4, 2.0)
    assert result["P_kW"] > 0.0


# ---------------------------------------------------------------------------
# propeller_disk_mask
# ---------------------------------------------------------------------------

def test_disk_mask_shape():
    from tensorlbm.propeller_cad import propeller_disk_mask

    mask = propeller_disk_mask(60, 60, 10, diameter_lu=20.0)
    assert mask.shape == (60, 60, 10)
    assert mask.sum() > 0


# ---------------------------------------------------------------------------
# plot_b_series_curves
# ---------------------------------------------------------------------------

def test_plot_returns_figure():
    pytest.importorskip("matplotlib")
    import matplotlib.figure as mfig

    from tensorlbm.propeller_cad import plot_b_series_curves

    fig = plot_b_series_curves(1.0, 0.6, 4)
    assert isinstance(fig, mfig.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)

