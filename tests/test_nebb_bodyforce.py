"""Tests for the force-corrected NEBB wall + body-force driver.

Anchors (XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S):

1. the derived wall direction specs equal the ones tabulated in the OSS
   reference (``vivsim.lbm.lattice.D2Q9.boundary_spec``) -- a transcription
   guard for the wall kernel;
2. the NEBB kernel imposes the prescribed density and *both* velocity
   components on the boundary row to machine precision, and is invariant under
   the population shift ``f -> f - w`` of the OSS storage convention;
3. with a zero non-equilibrium part the shear-reconstruction kernel collapses
   onto the classic Zou/He NEBB;
4. the driver reproduces the closed-form Poiseuille profile to round-off and
   keeps mass / wall velocity under control;
5. the wall *models* are distinguishable: forcing-corrected NEBB has its
   no-slip surface on the node plane (offset < 0.02 cell) while halfway
   bounce-back has it half a cell outside (a negative control -- an
   implementation that ignored the wall model could not show both);
6. the module works on ``torch`` tensors too (the integration path used by
   :mod:`tensorlbm.boundary.antibb_channel_benchmark`);
7. the CLI entry point runs and writes its artifact;
8. the channel benchmark harness really calls the module (integration).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm import nebb_bodyforce as nb  # noqa: E402

#: direction tables of the OSS reference (vivsim/lbm/lattice.py, read from
#: source; kept here so the *derived* specs are checked against a fixed table)
OSS_SPECS = {
    "bottom": dict(axis=1, sign=+1, in_dirs=(2, 5, 6), out_dirs=(4, 7, 8), tan=(1, 3)),
    "top": dict(axis=1, sign=-1, in_dirs=(4, 7, 8), out_dirs=(2, 5, 6), tan=(1, 3)),
    "left": dict(axis=0, sign=+1, in_dirs=(1, 5, 8), out_dirs=(3, 7, 6), tan=(2, 4)),
    "right": dict(axis=0, sign=-1, in_dirs=(3, 7, 6), out_dirs=(1, 5, 8), tan=(2, 4)),
}


def _equilibrium_row(n=6, rho=1.0, ux=0.0, uy=0.0):
    r = np.full((n,), rho)
    return nb.equilibrium(r, np.full((n,), ux), np.full((n,), uy))


# --------------------------------------------------------------------------- #
# 1. direction bookkeeping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("loc", sorted(OSS_SPECS))
def test_wall_spec_matches_oss_table(loc):
    spec = nb.wall_spec(loc)
    ref = OSS_SPECS[loc]
    assert spec.normal_axis == ref["axis"]
    assert spec.normal_sign == ref["sign"]
    assert spec.in_dirs == ref["in_dirs"]
    assert spec.out_dirs == ref["out_dirs"]
    assert spec.tan_dirs == ref["tan"]


def test_lattice_constants_match_repo_d2q9():
    torch = pytest.importorskip("torch")
    from tensorlbm import d2q9

    assert np.array_equal(nb.C, d2q9.C.numpy())
    # repo W is stored as float32 (4/9 rounds); compare at float32 resolution
    assert np.allclose(nb.W, d2q9.W.numpy(), rtol=1e-6, atol=1e-8)
    assert np.array_equal(nb.OPPOSITE, d2q9.OPPOSITE.numpy())


# --------------------------------------------------------------------------- #
# 2. the wall kernel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("loc", ["bottom", "top"])
def test_nebb_row_imposes_prescribed_moments(loc):
    rng = np.random.default_rng(7)
    wall = rng.uniform(0.5, 1.5, size=(9, 5))
    rho_w, ux_w, uy_w = 1.013, 0.021, 0.0
    uxw, uyw = nb.corrected_wall_velocity(ux_w, uy_w, 1e-3, 0.0, rho_w)
    out = nb.nebb_row(wall, nb.wall_spec(loc), rho_w, uxw, uyw)

    jx = (out * nb.C[:, 0, None]).sum(axis=0)
    jy = (out * nb.C[:, 1, None]).sum(axis=0)
    # both momentum components are enforced exactly *with the prescribed
    # density*; the row total only equals rho_w when rho_w is the recovered
    # one (see the next test)
    assert np.allclose(jx, uxw * rho_w, rtol=0, atol=1e-14)
    assert np.allclose(jy, uyw * rho_w, rtol=0, atol=1e-14)
    # the known (outgoing) populations are untouched
    spec = nb.wall_spec(loc)
    assert np.array_equal(out[list(spec.out_dirs)], wall[list(spec.out_dirs)])


@pytest.mark.parametrize("loc", ["bottom", "top"])
def test_recovered_density_makes_row_self_consistent(loc):
    rng = np.random.default_rng(11)
    wall = rng.uniform(0.5, 1.5, size=(9, 4))
    f = np.zeros((9, 4, 4))
    idx = 0 if loc == "bottom" else -1
    f[:, :, idx] = wall
    uxw = 0.004
    rho_w = nb.recovered_wall_density(f, loc, uxw, 0.0)
    out = nb.nebb_row(wall, nb.wall_spec(loc), rho_w, uxw, 0.0)
    assert np.allclose(out.sum(axis=0), rho_w, rtol=0, atol=1e-13)


def test_nebb_invariant_under_population_shift():
    """The OSS stores ``f - w``; the ported kernel must be shift-invariant."""
    rng = np.random.default_rng(3)
    wall = rng.uniform(0.5, 1.5, size=(9, 5))
    shifted = wall - nb.W[:, None]
    for loc in ("bottom", "top"):
        spec = nb.wall_spec(loc)
        full = nb.nebb_row(wall, spec, 1.0, -5e-4, 0.0)
        from_shift = nb.nebb_row(shifted, spec, 1.0, -5e-4, 0.0) + nb.W[:, None]
        assert np.allclose(full, from_shift, rtol=0, atol=1e-15)


def test_shear_reconstruction_reduces_to_zou_he_on_equilibrium_row():
    row = _equilibrium_row(n=6, rho=1.0, ux=0.03, uy=0.0)
    for loc in ("bottom", "top"):
        spec = nb.wall_spec(loc)
        shear = nb.nebb_row(row, spec, 1.0, 0.03, 0.0)
        zouhe = nb.nonequilibrium_nebb_row(row, spec, 1.0, 0.03, 0.0)
        assert np.allclose(shear, zouhe, rtol=0, atol=1e-15)


# --------------------------------------------------------------------------- #
# 3. the driver
# --------------------------------------------------------------------------- #
def test_driver_reproduces_closed_form_poiseuille():
    cfg = nb.ChannelConfig(nx=4, ny=10, nu=0.2, gx=1e-3, scheme="guo", steps=200_000)
    res = nb.BodyForceChannel(cfg).run()
    analytic = nb.analytic_poiseuille(10, 1e-3, 0.2, 0.0)
    err = nb.profile_errors(res.profile, analytic)
    assert err["max_rel_global"] < 1e-10, err
    assert abs(res.wall_velocity_corrected) < 1e-12
    assert abs(res.mass_drift) < 1e-10
    assert abs(res.effective_wall_offset() if hasattr(res, "effective_wall_offset") else
               nb.effective_wall_offset(res.profile, 1e-3, 0.2)) < 0.02
    assert res.steps_used < 200_000


def test_edm_and_guo_schemes_agree():
    a = nb.BodyForceChannel(
        nb.ChannelConfig(nx=4, ny=10, nu=0.2, gx=1e-3, scheme="edm", steps=200_000)
    ).run()
    b = nb.BodyForceChannel(
        nb.ChannelConfig(nx=4, ny=10, nu=0.2, gx=1e-3, scheme="guo", steps=200_000)
    ).run()
    assert np.abs(a.profile - b.profile).max() < 1e-12


def test_wall_models_are_distinguishable():
    """Negative control: NEBB puts the wall on the node, bounce-back half a cell out."""
    nebb = nb.BodyForceChannel(
        nb.ChannelConfig(nx=4, ny=10, nu=0.2, gx=1e-3, scheme="guo", steps=200_000)
    ).run()
    bb = nb.BodyForceChannel(
        nb.ChannelConfig(
            nx=4, ny=10, nu=0.2, gx=1e-3, scheme="guo",
            wall="halfway_bounce_back", steps=200_000,
        )
    ).run()
    o_nebb = nb.effective_wall_offset(nebb.profile, 1e-3, 0.2)
    o_bb = nb.effective_wall_offset(bb.profile, 1e-3, 0.2)
    assert abs(o_nebb) < 0.02, o_nebb
    assert abs(o_bb - 0.5) < 0.05, o_bb


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        nb.ChannelConfig(ny=3).validate()
    with pytest.raises(ValueError):
        nb.ChannelConfig(scheme="mrt").validate()
    with pytest.raises(ValueError):
        nb.ChannelConfig(wall="slip").validate()
    with pytest.raises(ValueError):
        nb.wall_spec("middle")


# --------------------------------------------------------------------------- #
# 4. backend compatibility (integration path used by the torch harness)
# --------------------------------------------------------------------------- #
def test_nebb_row_works_on_torch_tensors():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    wall_np = rng.uniform(0.5, 1.5, size=(9, 5))
    wall = torch.as_tensor(wall_np, dtype=torch.float64)
    rho = torch.as_tensor(np.full(5, 1.0))
    out = nb.nebb_row(wall, nb.wall_spec("bottom"), rho, -5e-4, 0.0)
    assert isinstance(out, torch.Tensor)
    ref = nb.nebb_row(wall_np, nb.wall_spec("bottom"), 1.0, -5e-4, 0.0)
    assert np.allclose(out.numpy(), ref, rtol=0, atol=1e-15)


# --------------------------------------------------------------------------- #
# 5. CLI entry point
# --------------------------------------------------------------------------- #
def test_cli_entry_writes_artifact(tmp_path):
    out = tmp_path / "nebb.json"
    proc = subprocess.run(
        [sys.executable, "-m", "tensorlbm.nebb_bodyforce",
         "--ny", "10", "--scheme", "guo", "--steps", "20000", "--out", str(out)],
        cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert data["errors_offset0"]["max_rel_global"] < 1e-8
    assert len(data["profile"]) == 10
    assert "steps_used" in data


# --------------------------------------------------------------------------- #
# 6. integration: the channel benchmark harness calls the module
# --------------------------------------------------------------------------- #
def test_channel_benchmark_uses_force_corrected_nebb():
    pytest.importorskip("torch")
    from tensorlbm.boundary.antibb_channel_benchmark import (
        ChannelSetup,
        l2_relative_error,
        poiseuille_profile,
        run_channel,
    )

    setup = ChannelSetup(ny=17, nx=3, tau=0.8, force=1e-4, steps=40_000,
                         wall_mode="force_corrected_nebb", steady_tol=1e-13)
    res = run_channel(setup)
    err = l2_relative_error(res.ux[:, 0], poiseuille_profile(setup))
    assert err < 5e-3, err
    assert abs(res.mass_drift) < 1e-9
    # the wall row carries the half-force-step shifted velocity, not 0
    assert abs(float(res.ux[0, 0]) + 0.5 * setup.force) < 1e-6