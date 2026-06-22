"""Tests for tensorlbm.vtk_export – VTK file generation."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_2d_fields(ny: int = 8, nx: int = 10):
    rho = torch.ones(ny, nx)
    ux = torch.rand(ny, nx) * 0.05
    uy = torch.rand(ny, nx) * 0.05
    return rho, ux, uy


def _make_3d_fields(nz: int = 4, ny: int = 5, nx: int = 6):
    rho = torch.ones(nz, ny, nx)
    ux = torch.rand(nz, ny, nx) * 0.05
    uy = torch.rand(nz, ny, nx) * 0.02
    uz = torch.rand(nz, ny, nx) * 0.01
    return rho, ux, uy, uz


# ---------------------------------------------------------------------------
# 2-D export
# ---------------------------------------------------------------------------

def test_export_vtk_2d_creates_file():
    from tensorlbm.vtk_export import export_vtk_2d

    rho, ux, uy = _make_2d_fields()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "field2d.vtk"
        result = export_vtk_2d(rho, ux, uy, out)
        assert result.exists()
        assert result.stat().st_size > 0


def test_export_vtk_2d_header():
    from tensorlbm.vtk_export import export_vtk_2d

    rho, ux, uy = _make_2d_fields(ny=4, nx=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "field2d.vtk"
        export_vtk_2d(rho, ux, uy, out)
        content = out.read_text()
    assert "vtk DataFile Version 3.0" in content
    assert "STRUCTURED_POINTS" in content
    assert "DIMENSIONS 5 4 1" in content


def test_export_vtk_2d_all_fields():
    from tensorlbm.vtk_export import export_vtk_2d

    rho, ux, uy = _make_2d_fields()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "field2d_all.vtk"
        export_vtk_2d(rho, ux, uy, out, fields=None)
        content = out.read_text()
    for fname in ("density", "pressure", "velocity_magnitude", "vorticity"):
        assert fname in content


def test_export_vtk_2d_subset_fields():
    from tensorlbm.vtk_export import export_vtk_2d

    rho, ux, uy = _make_2d_fields()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "field2d_sub.vtk"
        export_vtk_2d(rho, ux, uy, out, fields=["density", "pressure"])
        content = out.read_text()
    assert "density" in content
    assert "pressure" in content
    assert "vorticity" not in content


def test_export_vtk_2d_spacing():
    from tensorlbm.vtk_export import export_vtk_2d

    rho, ux, uy = _make_2d_fields()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "field2d_sp.vtk"
        export_vtk_2d(rho, ux, uy, out, spacing=0.001)
        content = out.read_text()
    assert "SPACING 0.001" in content


# ---------------------------------------------------------------------------
# 3-D export
# ---------------------------------------------------------------------------

def test_export_vtk_3d_creates_file():
    from tensorlbm.vtk_export import export_vtk_3d

    rho, ux, uy, uz = _make_3d_fields()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "field3d.vtk"
        result = export_vtk_3d(rho, ux, uy, uz, out)
        assert result.exists()
        assert result.stat().st_size > 0


def test_export_vtk_3d_header():
    from tensorlbm.vtk_export import export_vtk_3d

    rho, ux, uy, uz = _make_3d_fields(nz=3, ny=4, nx=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "field3d.vtk"
        export_vtk_3d(rho, ux, uy, uz, out)
        content = out.read_text()
    assert "DIMENSIONS 5 4 3" in content


def test_export_vtk_3d_q_criterion():
    from tensorlbm.vtk_export import export_vtk_3d

    rho, ux, uy, uz = _make_3d_fields()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "field3d_q.vtk"
        export_vtk_3d(rho, ux, uy, uz, out, fields=["q_criterion", "velocity"])
        content = out.read_text()
    assert "q_criterion" in content
    assert "VECTORS velocity" in content


# ---------------------------------------------------------------------------
# Q-criterion helper
# ---------------------------------------------------------------------------

def test_q_criterion_3d_shape():
    from tensorlbm.vtk_export import _q_criterion_3d

    nz, ny, nx = 4, 5, 6
    ux = torch.rand(nz, ny, nx)
    uy = torch.rand(nz, ny, nx)
    uz = torch.rand(nz, ny, nx)
    q = _q_criterion_3d(ux, uy, uz)
    assert q.shape == (nz, ny, nx)


def test_q_criterion_solid_body_rotation_zero():
    """Q should be zero for uniform solid-body rotation (no strain)."""
    from tensorlbm.vtk_export import _q_criterion_3d

    nz, ny, nx = 4, 4, 4
    y_idx = torch.arange(ny).float().view(1, ny, 1).expand(nz, ny, nx)
    x_idx = torch.arange(nx).float().view(1, 1, nx).expand(nz, ny, nx)
    omega = 0.1
    ux = -omega * y_idx
    uy = omega * x_idx
    uz = torch.zeros_like(ux)
    q = _q_criterion_3d(ux, uy, uz)
    # Interior cells should have Q ≈ ω² (positive)
    assert float(q[1:-1, 1:-1, 1:-1].mean()) > 0.0


# ---------------------------------------------------------------------------
# export_checkpoint_vtk with mocked checkpoint
# ---------------------------------------------------------------------------

def test_export_checkpoint_vtk_2d(tmp_path):
    """Export a synthetic 2-D checkpoint directory to VTK."""
    import torch

    from tensorlbm.vtk_export import export_checkpoint_vtk

    # Create a minimal checkpoint
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    from tensorlbm.d2q9 import equilibrium
    rho0 = torch.ones(8, 10)
    ux0 = torch.zeros(8, 10)
    uy0 = torch.zeros(8, 10)
    f = equilibrium(rho0, ux0, uy0)
    torch.save(f, ckpt_dir / "checkpoint_f.pt")
    import json
    (ckpt_dir / "checkpoint_meta.json").write_text(json.dumps({"step": 0}))

    out = export_checkpoint_vtk(ckpt_dir, tmp_path / "out.vtk")
    assert out.exists()
    content = out.read_text()
    assert "STRUCTURED_POINTS" in content
