import sys

import torch

from examples.droplet_impact import _initial_phase_field, main


def test_initial_phase_field_places_pool_below_interface_and_droplet_above() -> None:
    y = torch.arange(64, dtype=torch.float32)
    x = torch.arange(64, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    water_level = 32.0
    radius = 8.0
    gap = 4.0
    center_x = 32.0
    center_y = water_level - radius - gap

    phi = _initial_phase_field(xx, yy, radius, center_x, center_y, water_level, interface_width=3.0)

    assert phi[24, 0] < 0.0
    assert phi[40, 0] > 0.0
    assert phi[int(center_y), int(center_x)] > 0.0
    assert phi[int(water_level - gap / 2), int(center_x)] < 0.0


def test_fast_run_writes_diagnostic_plot(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["droplet_impact.py", "--fast", "--overwrite", "--output-root", str(tmp_path)],
    )

    main()

    assert (tmp_path / "impact_120x90_s120" / "diagnostic_curves.png").is_file()
