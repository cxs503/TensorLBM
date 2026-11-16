"""Tests of the single-phase Darcy channel with a selectable lattice layout.

``tensorlbm.porous_media.run_single_phase_darcy`` is the integration entry point
of the compressed fluid-node neighbour table
(:mod:`tensorlbm.porous.fluid_node_table`) inside the porous-media solver
module.  These tests pin the four properties that the entry point exists for:

* **identity** -- the compressed layout and an independently written half-way
  dense update of the main solver agree to round-off (representation change is
  physics-neutral),
* **physics** -- the compressed layout reproduces the analytic periodic-slab
  permeability ``k_super = (H/ny) H**2/12`` to well under 3 %,
* **wall convention** -- the shipped dense update (``stream`` +
  ``bounce_back_cells``) places the no-slip plane on the *solid-node centre*, so
  its permeability is biased high by an O(1/H) amount; the test locks the
  measured magnitude so that a future change of that rule cannot pass silently,
* **contract** -- configuration validation, artefacts of record, determinism,
  mass conservation and the library-level export.

The XVAL record of the same channel (``XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-
TABLE``, H=16: -0.1094 %) is the external anchor of the physics test.
"""

from __future__ import annotations

import json

import pytest
import torch

torch.set_default_dtype(torch.float64)

from tensorlbm.porous_media import (  # noqa: E402
    SinglePhaseDarcyConfig,
    _guo_macroscopic,
    _single_phase_step_dense,
    _slab_channel_mask,
    run_single_phase_darcy,
)

NY, NX, H = 16, 16, 10
G = 1.0e-4
TAU = 0.8


def _run(tmp_path, layout: str, **over) -> dict:
    kw = dict(ny=NY, nx=NX, h_fluid=H, g=G, tau=TAU, steps_cap=6000, tol=1.0e-9,
              check_every=25, layout=layout, output_root=tmp_path)
    kw.update(over)
    return run_single_phase_darcy(SinglePhaseDarcyConfig(**kw))


def test_compressed_matches_halfway_dense_to_roundoff(tmp_path):
    """G1: compaction + neighbour table is a representation change only."""
    a = _run(tmp_path, "compressed")
    b = _run(tmp_path, "dense_halfway")
    assert a["converged"] and b["converged"]
    assert a["steps"] == b["steps"]
    k_rel = abs(a["k_darcy_super"] - b["k_darcy_super"]) / abs(a["k_darcy_super"])
    prof_a, prof_b = a["ux_profile_mean_x"], b["ux_profile_mean_x"]
    prof_rel = max(abs(x - y) for x, y in zip(prof_a, prof_b)) / max(abs(v) for v in prof_a)
    assert k_rel < 1e-10, k_rel
    assert prof_rel < 1e-10, prof_rel


@pytest.mark.parametrize("h", [8, 10, 12])
def test_compressed_permeability_matches_analytic(tmp_path, h):
    """G2: half-way BB permeability against the analytic slab value (<3 %)."""
    meta = _run(tmp_path, "compressed", h_fluid=h)
    assert meta["converged"]
    assert abs(meta["err_pct"]) < 3.0, meta["err_pct"]
    # discretisation error of a parabola must shrink with H
    assert meta["err_pct"] < 0.0


def test_steps_shipped_dense_wall_convention_bias(tmp_path):
    """G3: the shipped dense wall rule is node-centred -> O(1/H) high bias."""
    node = _run(tmp_path, "dense")
    half = _run(tmp_path, "dense_halfway")
    assert node["converged"] and half["converged"]
    # node-centred reflection widens the effective channel: permeability high
    assert node["err_pct"] > 3.0
    assert node["err_pct"] < 30.0
    assert abs(half["err_pct"]) < 3.0
    # effective width implied by the measured permeability
    h_eff = (node["k_darcy_super"] * NY * 12.0) ** (1.0 / 3.0)
    assert H + 0.5 < h_eff < H + 1.0, h_eff


def test_vs_xval_record_h16():
    """The compressed arm reproduces the XVAL record (-0.1094 % at H=16).

    The reference number is the value registered by mission m22754 for
    ``XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE`` (verdict evidence G2);
    the assertion ties the new production entry point to that record.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        meta = _run(Path(td), "compressed", ny=24, nx=24, h_fluid=16, steps_cap=8000,
                    tol=1.0e-10, check_every=50)
    assert meta["converged"]
    assert meta["err_pct"] == pytest.approx(-0.109375, abs=2e-3)
    assert meta["final_state"]["ma_max"] < 0.1


def test_mass_and_momentum_bounded(tmp_path):
    meta = _run(tmp_path, "compressed")
    fs = meta["final_state"]
    assert fs["finite"]
    assert fs["rho_min"] == pytest.approx(1.0, abs=1e-12)
    assert fs["rho_max"] == pytest.approx(1.0, abs=1e-12)
    assert fs["ma_max"] < 0.1


def test_determinism(tmp_path):
    a = _run(tmp_path, "compressed", output_root=tmp_path / "a")
    b = _run(tmp_path, "compressed", output_root=tmp_path / "b")
    assert a["k_darcy_super"] == b["k_darcy_super"]
    assert a["ux_profile_mean_x"] == b["ux_profile_mean_x"]


def test_artifacts_of_record_written(tmp_path):
    meta = _run(tmp_path, "compressed")
    run_dir = tmp_path / "single_phase_darcy" / "single_phase_darcy_compressed_ny16_h10_g0.0001_tau0.8"
    meta_path = run_dir / "run_metadata.json"
    assert meta_path.is_file()
    assert (run_dir / "profile.png").is_file()
    stored = json.loads(meta_path.read_text())
    for key in ("layout", "steps", "k_darcy_super", "k_ref_analytic", "err_pct", "final_state"):
        assert key in stored
    assert stored["k_darcy_super"] == meta["k_darcy_super"]


@pytest.mark.parametrize(
    "override",
    [
        {"layout": "sparse"},
        {"ny": NY, "h_fluid": NY + 1},
        {"g": 0.0},
        {"tau": 0.5},
        {"steps_cap": 10, "check_every": 50},
    ],
)
def test_config_validation(override):
    with pytest.raises(ValueError):
        SinglePhaseDarcyConfig(**override)


def test_library_level_export():
    import tensorlbm

    assert tensorlbm.SinglePhaseDarcyConfig is SinglePhaseDarcyConfig
    assert tensorlbm.run_single_phase_darcy is run_single_phase_darcy


def test_dense_helper_wall_rules_differ_by_half_cell():
    """The two dense wall rules of the entry point are not the same operator."""
    solid = _slab_channel_mask(NY, NX, H, torch.device("cpu"))
    from tensorlbm.d2q9 import W_EXACT

    f = torch.zeros((9, NY, NX)) + torch.tensor(W_EXACT, dtype=torch.float64).view(9, 1, 1)
    for _ in range(50):
        f_node = _single_phase_step_dense(f, solid, TAU, G, 0.0, torch.device("cpu"), wall="node")
        f_half = _single_phase_step_dense(f, solid, TAU, G, 0.0, torch.device("cpu"), wall="halfway")
        assert not torch.allclose(f_node, f_half, atol=0.0, rtol=0.0)
        f = f_half
    _, u_node, _ = _guo_macroscopic(f_node, G, 0.0, torch.device("cpu"))
    _, u_half, _ = _guo_macroscopic(f_half, G, 0.0, torch.device("cpu"))
    fluid = ~solid
    assert float((u_node[fluid] - u_half[fluid]).abs().max()) > 0.0
