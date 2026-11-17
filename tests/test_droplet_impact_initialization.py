import json
import sys

import pytest
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
    meta = json.loads((tmp_path / "impact_120x90_s120" / "run_metadata.json").read_text())
    assert meta["health"]["passed"]
    assert meta["health"]["completed_steps"] == 120
    assert meta["health"]["max_observed"]["relative_phase_mass_drift"] < 1e-4


@pytest.mark.parametrize(
    "failure",
    ["nonfinite_state", "max_speed", "phase_mass_drift", "phase_bounds", "nonpositive_density"],
)
def test_failed_run_writes_evidence_before_raising(tmp_path, monkeypatch, failure):
    import examples.droplet_impact as example

    real_step = example.free_energy_step

    def injected_step(f, g, **kwargs):
        f, g = real_step(f, g, **kwargs)
        if failure == "nonfinite_state":
            g[:, 10, 10] = float("nan")
        elif failure == "phase_mass_drift":
            g[0] += 0.01
        elif failure == "phase_bounds":
            g[0, 10, 10] = 5.0
        elif failure == "nonpositive_density":
            f[0, 10, 10] = -10.0
        return f, g

    monkeypatch.setattr(example, "free_energy_step", injected_step)
    monkeypatch.setattr(example, "save_contour", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "droplet_impact.py",
            "--fast",
            "--output-root",
            str(tmp_path),
            "--max-speed",
            "1e-12" if failure == "max_speed" else "0.15",
        ],
    )
    with pytest.raises(RuntimeError, match=failure):
        main()
    path = tmp_path / "impact_120x90_s120" / "run_metadata.json"
    raw = path.read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    meta = json.loads(raw)
    assert not meta["health"]["passed"]
    assert meta["health"]["completed_steps"] == 1
    assert failure in meta["health"]["failures"]


def test_no_flux_stencils_do_not_see_opposite_edge():
    from tensorlbm.phasefield.operators import central_gradient_2d, laplacian_2d

    field = torch.zeros((8, 9), dtype=torch.float64)
    field[-1] = 7.0
    grad_x, grad_y = central_gradient_2d(field, boundary="no_flux")
    lap = laplacian_2d(field, boundary="no_flux")
    assert grad_x[0].count_nonzero() == 0
    assert grad_y[0].count_nonzero() == 0
    assert lap[0].count_nonzero() == 0
    mask = torch.zeros_like(field, dtype=torch.bool)
    mask[-1] = True
    grad_x, grad_y = central_gradient_2d(field, boundary="no_flux", solid_mask=mask)
    lap = laplacian_2d(field, boundary="no_flux", solid_mask=mask)
    assert grad_y[-2].count_nonzero() == 0
    assert lap[-2].count_nonzero() == 0


def test_single_phase_hydrostatic_column_remains_at_rest():
    """p=rho/3, dp/dy=rho*g; a necessary baseline, not impact validation."""
    from examples.droplet_impact import _make_tank_wall_mask
    from tensorlbm.boundary.tank import stream_tank
    from tensorlbm.d2q9 import equilibrium, macroscopic
    from tensorlbm.multiphase import free_energy_step, init_free_energy_g

    ny, nx, gravity = 32, 12, 2e-5
    y = torch.arange(ny, dtype=torch.float64)[:, None].expand(ny, nx)
    rho_exact = torch.exp(3 * gravity * (y - 15))
    zero = torch.zeros_like(rho_exact)
    # Raw population momentum is offset from physical rest by -rho*g/2.
    f = equilibrium(rho_exact, zero, zero - gravity / 2)
    g = init_free_energy_g(-torch.ones_like(rho_exact), zero, zero)
    wall = _make_tank_wall_mask(ny, nx, torch.device("cpu"))
    mass0 = g[:, ~wall].sum()
    for _ in range(1000):
        f, g = free_energy_step(
            f,
            g,
            tau_f=0.8,
            tau_g=0.8,
            A=0.04,
            B=0.04,
            kappa=0.02,
            Gamma=0.5,
            gy=gravity,
            boundary="no_flux",
            solid_mask=wall,
        )
        f, g = stream_tank(f, g, wall)
    rho, ux, uy = macroscopic(f)
    assert float(((rho / rho_exact - 1)[~wall]).abs().max()) < 1e-9
    assert float(torch.sqrt(ux.square() + (uy + gravity / 2).square())[~wall].max()) < 1e-10
    assert float((g.sum(0)[~wall] + 1).abs().max()) < 1e-9
    assert float((g[:, ~wall].sum() / mass0 - 1).abs()) < 1e-12
