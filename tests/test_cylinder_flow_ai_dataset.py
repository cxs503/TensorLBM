from __future__ import annotations

import numpy as np
import torch

from tensorlbm.cylinder_flow import _save_flow_snapshot


def test_save_flow_snapshot_exports_numerical_fields(tmp_path):
    ny, nx = 8, 16
    ux = torch.full((ny, nx), 0.08, dtype=torch.float32)
    uy = torch.zeros((ny, nx), dtype=torch.float32)
    rho = torch.ones((ny, nx), dtype=torch.float32)
    obstacle = torch.zeros((ny, nx), dtype=torch.bool)
    obstacle[3:5, 5:7] = True
    speed = torch.sqrt(ux * ux + uy * uy)
    vort = np.zeros((ny, nx), dtype=np.float32)

    _save_flow_snapshot(
        tmp_path,
        20,
        speed,
        vort,
        obstacle,
        ux=ux,
        uy=uy,
        rho=rho,
    )

    path = tmp_path / "fields_step_000020.npz"
    assert path.is_file()
    with np.load(path, allow_pickle=False) as fields:
        assert set(fields.files) == {
            "step", "ux", "uy", "rho", "speed", "vorticity", "solid_mask"
        }
        assert int(fields["step"]) == 20
        assert fields["ux"].shape == (ny, nx)
        assert fields["uy"].shape == (ny, nx)
        assert fields["rho"].shape == (ny, nx)
        assert fields["solid_mask"].dtype == np.bool_
        assert fields["ux"].dtype == np.float32
        np.testing.assert_allclose(fields["ux"], 0.08)
        np.testing.assert_allclose(fields["rho"], 1.0)
        np.testing.assert_array_equal(fields["solid_mask"], obstacle.numpy())
    assert (tmp_path / "flow_step_000020.png").is_file()
