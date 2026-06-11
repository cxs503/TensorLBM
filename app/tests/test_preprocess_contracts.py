"""Contract checks between preprocess routes and tensorlbm core signatures."""
from __future__ import annotations

import inspect


def test_preprocess_core_signatures_are_stable():
    from tensorlbm import (
        LBMUnitConverter,
        poly_to_mask_2d,
        random_porosity_mask_2d,
        voxelize_stl_3d,
    )

    poly = inspect.signature(poly_to_mask_2d)
    assert list(poly.parameters)[:4] == ["vertices", "ny", "nx", "device"]

    poro = inspect.signature(random_porosity_mask_2d)
    for name in ("ny", "nx", "porosity", "device", "seed", "sigma"):
        assert name in poro.parameters

    vox = inspect.signature(voxelize_stl_3d)
    assert list(vox.parameters)[:5] == ["stl_path", "nx", "ny", "nz", "device"]

    conv = inspect.signature(LBMUnitConverter)
    for name in ("re", "l_phys", "u_phys", "nu_phys", "nx", "u_lb"):
        assert name in conv.parameters

