"""Tests for XDMF metadata export."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
import torch

from tensorlbm.io import save_hdf5, save_xdmf


def test_save_xdmf_3d(tmp_path) -> None:
    try:
        import h5py  # noqa: F401
    except ImportError:
        pytest.skip("h5py not installed")
    ux = torch.rand((4, 5, 6))
    uy = torch.rand((4, 5, 6))
    uz = torch.rand((4, 5, 6))
    rho = torch.rand((4, 5, 6))
    h5_path = save_hdf5(tmp_path / "out3d.h5", step=1, ux=ux, uy=uy, uz=uz, rho=rho)
    xdmf_path = save_xdmf(
        h5_path,
        tmp_path / "out3d.xdmf",
        step=1,
        ux_shape=tuple(ux.shape),
        has_uz=True,
        has_rho=True,
    )
    root = ET.fromstring(xdmf_path.read_text(encoding="utf-8"))
    assert root.tag == "Xdmf"


def test_save_xdmf_2d(tmp_path) -> None:
    try:
        import h5py  # noqa: F401
    except ImportError:
        pytest.skip("h5py not installed")
    ux = torch.rand((5, 6))
    uy = torch.rand((5, 6))
    h5_path = save_hdf5(tmp_path / "out2d.h5", step=2, ux=ux, uy=uy)
    xdmf_path = save_xdmf(h5_path, tmp_path / "out2d.xdmf", step=2, ux_shape=tuple(ux.shape))
    root = ET.fromstring(xdmf_path.read_text(encoding="utf-8"))
    assert root.tag == "Xdmf"
