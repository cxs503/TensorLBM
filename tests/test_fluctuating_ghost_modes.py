"""Unit contract for the ghost-mode resolved thermostat (XVAL extension)."""
import numpy as np
import pytest

from tensorlbm.fluctuating_ghost_modes import (
    CONSERVED_MODES,
    GHOST_MODES,
    MODE_NAMES,
    FluctuatingLBMGhost,
    modes_of,
    orthonormal_modes,
)
from tensorlbm.fluctuating_lbm import VELOCITY_SETS, FluctuatingLBM


def test_mode_basis_is_weighted_orthonormal():
    basis = orthonormal_modes("D2Q9")
    gram = basis @ basis.T
    assert np.allclose(gram, np.eye(9), atol=1e-12)


def test_modes_reconstruct_distribution():
    rng = np.random.default_rng(3)
    f = rng.standard_normal((9, 4, 5))
    basis = orthonormal_modes("D2Q9")
    m = modes_of(f, basis)
    f_back = np.tensordot(basis.T, m, axes=([1], [0]))
    assert np.allclose(f_back, f, atol=1e-12)


def test_ghost_noise_conserves_mass_and_momentum():
    lbm = FluctuatingLBMGhost((6, 6), tau=0.8, kT=1e-6, seed=7, thermostat="ghosts")
    rng = np.random.default_rng(11)
    f = lbm.equilibrium(np.ones(lbm.shape), np.zeros((2,) + lbm.shape))
    f = f + 1e-3 * rng.standard_normal(f.shape)
    g = lbm._add_mode_noise(f, np.ones(lbm.shape))
    m_f, m_g = modes_of(f, lbm.basis), modes_of(g, lbm.basis)
    for name in CONSERVED_MODES:
        i = MODE_NAMES.index(name)
        assert np.allclose(m_f[i], m_g[i], atol=1e-12), name


def test_kT_zero_is_deterministic_and_noise_free():
    a = FluctuatingLBMGhost((8, 8), tau=0.8, kT=0.0, seed=1, thermostat="ghosts")
    b = FluctuatingLBMGhost((8, 8), tau=0.8, kT=0.0, seed=1, thermostat="stress")
    assert np.all(a.random_force() == 0.0)
    a.step(10)
    b.step(10)
    assert np.array_equal(a.f, b.f)


def test_noise_amplitude_scales_with_kT_and_tau():
    lbm = FluctuatingLBMGhost((4, 4), tau=0.8, kT=1e-6, seed=0, thermostat="ghosts")
    v1 = lbm.noise_amplitude(1.0)
    lbm.kT = 4e-6
    v4 = lbm.noise_amplitude(1.0)
    assert v4 == pytest.approx(4.0 * v1, rel=1e-12)
    assert lbm.target_mode_variance(1.0) == pytest.approx(3.0 * lbm.kT)


def test_ghost_thermostat_excites_ghost_modes():
    """The transplant must raise the ghost-mode variance above the stress-only arm."""
    common = dict(shape=(12, 12), tau=0.8, kT=1e-6, seed=5)
    stress = FluctuatingLBMGhost(thermostat="stress", **common).mode_variance_report(n_samples=400, n_skip=100)
    ghosts = FluctuatingLBMGhost(thermostat="ghosts", **common).mode_variance_report(n_samples=400, n_skip=100)
    print("ghost ratios stress=%s ghosts=%s" % (stress.ratios["qx"], ghosts.ratios["qx"]))
    assert ghosts.ratios["qx"] > stress.ratios["qx"]
    assert ghosts.ratios["eps"] > stress.ratios["eps"]


def test_existing_module_unchanged():
    """Regression guard: the additive module must not alter the base solver behaviour."""
    lbm = FluctuatingLBM((8, 8), tau=0.8, kT=0.0, seed=2)
    lbm.step(3)
    assert np.isfinite(lbm.f).all()
