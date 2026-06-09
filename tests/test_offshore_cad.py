"""Tests for the offshore_cad module."""
from __future__ import annotations

import pytest


def test_monopile_mask_shape():
    from tensorlbm.offshore_cad import monopile_mask

    mask = monopile_mask(60, 60, 60, diameter=8.0)
    assert mask.shape == (60, 60, 60)


def test_monopile_has_solid_cells():
    from tensorlbm.offshore_cad import monopile_mask

    mask = monopile_mask(60, 60, 60, diameter=8.0)
    assert mask.sum() > 0


def test_jacket_mask_shape():
    from tensorlbm.offshore_cad import jacket_mask

    mask = jacket_mask(80, 80, 80, leg_diameter=4.0, foot_spread=30.0, head_spread=16.0)
    assert mask.shape == (80, 80, 80)
    assert mask.sum() > 0


def test_spar_mask_shape():
    from tensorlbm.offshore_cad import spar_mask

    mask = spar_mask(60, 60, 80, hull_diameter=8.0, keel_diameter=14.0, column_diameter=8.0)
    assert mask.shape == (60, 60, 80)
    assert mask.sum() > 0


def test_semi_sub_mask_shape():
    from tensorlbm.offshore_cad import semi_sub_mask

    mask = semi_sub_mask(
        80, 80, 80,
        pontoon_length=40.0,
        pontoon_width=8.0,
        pontoon_height=6.0,
        column_diameter=8.0,
        column_height=20.0,
    )
    assert mask.shape == (80, 80, 80)
    assert mask.sum() > 0


def test_build_offshore_mask_dict_keys():
    from tensorlbm.offshore_cad import build_offshore_mask

    result = build_offshore_mask("monopile", 40, 40, 40, device="cpu")
    assert "mask" in result
    assert "stats" in result
    stats = result["stats"]
    assert stats["solid_cells"] > 0
    assert stats["solid_cells"] + stats["fluid_cells"] == 40 ** 3


def test_all_structure_types():
    from tensorlbm.offshore_cad import OffshoreStructureType, build_offshore_mask

    for st in OffshoreStructureType:
        result = build_offshore_mask(st.value, 40, 40, 40, device="cpu")
        assert result["stats"]["solid_cells"] > 0, f"No solid cells for {st}"


def test_offshore_statistics():
    from tensorlbm.offshore_cad import build_offshore_mask, offshore_statistics

    result = build_offshore_mask("jacket", 60, 60, 60, device="cpu")
    mask = result["mask"]
    stats = offshore_statistics("jacket", 60, 60, 60, mask)
    assert stats["solid_fraction"] > 0.0
    assert stats["solid_fraction"] < 1.0


def test_export_stl(tmp_path):
    from tensorlbm.offshore_cad import export_offshore_stl

    out = str(tmp_path / "monopile.stl")
    export_offshore_stl("monopile", out, 30, 30, 30)
    with open(out) as fh:
        content = fh.read()
    assert "solid" in content
    assert "facet" in content


def test_generate_previews_returns_figure():
    pytest.importorskip("matplotlib")
    import matplotlib.figure as mfig

    from tensorlbm.offshore_cad import generate_offshore_previews

    fig = generate_offshore_previews("spar", nx=40, ny=40, nz=60)
    assert isinstance(fig, mfig.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)

