"""Regression tests for honest static-droplet pressure reporting."""

from pathlib import Path

from tensorlbm.multiphase_benchmarks import StaticDropletConfig, run_static_droplet


def test_static_droplet_does_not_claim_laplace_validation(tmp_path: Path) -> None:
    """A smoke run must label cs²ρ as a proxy, not validated mechanical pressure."""
    report = run_static_droplet(
        StaticDropletConfig(
            nx=40,
            ny=40,
            radii=(8.0,),
            n_steps=0,
            output_root=tmp_path,
            device="cpu",
        )
    )

    assert "NOT_VALIDATED" in str(report["note"])
    for model in ("scmc", "cg"):
        result = report["results"][model]
        assert result["pressure_diagnostic"] == "ideal_gas_density_proxy"
        assert result["laplace_physical_validation"] == "NOT_VALIDATED"
        assert result["per_radius"][0]["pressure_diagnostic"] == "ideal_gas_density_proxy"
        assert result["per_radius"][0]["laplace_physical_validation"] == "NOT_VALIDATED"
