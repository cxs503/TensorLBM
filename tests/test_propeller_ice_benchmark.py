"""Tests for propeller-ice interaction benchmark."""

import pytest
import torch

from tensorlbm.propeller_ice_benchmark import (
    PropellerIceBenchmark,
    PropellerIceConfig,
)


def test_propeller_ice_config():
    """Test configuration creation."""
    config = PropellerIceConfig(
        nx=100,
        ny=50,
        nz=50,
        n_particles=5,
        n_steps=100,
    )
    
    assert config.nx == 100
    assert config.ny == 50
    assert config.nz == 50
    assert config.n_particles == 5
    assert config.n_steps == 100


def test_propeller_ice_initialization():
    """Test benchmark initialization."""
    config = PropellerIceConfig(
        nx=50,
        ny=30,
        nz=30,
        n_particles=3,
        n_steps=10,
        device="cpu",
    )
    
    benchmark = PropellerIceBenchmark(config)
    
    # Check LBM fields initialized
    assert benchmark.f.shape == (19, config.nz, config.ny, config.nx)
    
    # Check DEM initialized
    assert len(benchmark.dem_solver.system) == 3
    
    # Check coupling initialized
    assert benchmark.coupling is not None


def test_propeller_ice_step():
    """Test single time step."""
    config = PropellerIceConfig(
        nx=50,
        ny=30,
        nz=30,
        n_particles=2,
        n_steps=5,
        device="cpu",
    )
    
    benchmark = PropellerIceBenchmark(config)
    
    # Run a few steps
    for _ in range(5):
        benchmark.step()
    
    # Check that thrust/torque history recorded
    assert len(benchmark.thrust_history) == 5
    assert len(benchmark.torque_history) == 5


def test_propeller_thrust_torque_computation():
    """Test thrust and torque computation."""
    config = PropellerIceConfig(
        nx=100,
        ny=50,
        nz=50,
        propeller_diameter=30.0,
        propeller_rpm=10.0,
        n_particles=0,  # No particles for clean test
        n_steps=10,
        device="cpu",
    )
    
    benchmark = PropellerIceBenchmark(config)
    
    # Run a few steps
    for _ in range(10):
        benchmark.step()
    
    # Check that thrust and torque are computed
    thrust, torque = benchmark.compute_thrust_torque()
    
    # Should be finite numbers
    assert torch.isfinite(torch.tensor(thrust))
    assert torch.isfinite(torch.tensor(torque))


def test_propeller_ice_benchmark_short_run():
    """Test short benchmark run."""
    config = PropellerIceConfig(
        nx=60,
        ny=40,
        nz=40,
        n_particles=3,
        n_steps=20,
        output_interval=10,
        device="cpu",
    )
    
    benchmark = PropellerIceBenchmark(config)
    results = benchmark.run()
    
    # Check results structure
    assert "metadata" in results
    assert "config" in results
    assert "thrust_mean" in results
    assert "torque_mean" in results
    assert "elapsed_time_s" in results
    
    # Check values are finite
    assert torch.isfinite(torch.tensor(results["thrust_mean"]))
    assert torch.isfinite(torch.tensor(results["torque_mean"]))
    assert results["elapsed_time_s"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
