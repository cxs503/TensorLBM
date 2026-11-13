"""Static-droplet driver for the SOY free-energy core (main-solver side caller).

``tensorlbm.multiphase_soy_dropin`` provides the pluggable-backend SOY free-energy
D2Q9 core (the XVAL port of ebinan92/Fingering_dynamics' "cupy drop-in, no kernel
fusion" practice).  This function is the package-level entry the multiphase
driver / benchmark code uses, so the core is reachable from the solver layer and
not only from the XVAL script: it builds a static-droplet case, advances it and
returns the Laplace diagnostics.

Example
-------
>>> out = soy_free_energy_droplet(radius=15.0, steps=200, backend="numpy")
>>> round(abs(out["err_pct"]), 3) < 15.0
True
"""

from __future__ import annotations

from typing import Any

__all__ = ["soy_free_energy_droplet"]


def soy_free_energy_droplet(
    radius: float = 15.0,
    nx: int = 60,
    ny: int = 60,
    steps: int = 500,
    backend: str = "torch",
    **config: Any,
) -> dict[str, Any]:
    """Advance a periodic SOY free-energy static droplet and measure ``dp``.

    Args:
        radius: initial droplet radius (lattice units).
        nx, ny: lattice size.
        steps: number of time steps.
        backend: ``"numpy"`` or ``"torch"`` (array-namespace drop-in switch).
        **config: forwarded to
            :class:`tensorlbm.multiphase_soy_dropin.SOYConfig`
            (``sigma``, ``xi``, ``gamma``, ``pressure_form``, ...).

    Returns:
        ``dict`` with the Laplace diagnostics from
        :meth:`SOYFreeEnergyD2Q9.measure` plus ``err_pct`` (signed relative error
        of ``dp`` against ``sigma/R``) and the effective configuration.
    """
    # absolute import on purpose: it keeps the module path greppable from the
    # solver layer (integration-caller evidence for the XVAL anchor).
    from tensorlbm.multiphase_soy_dropin import (  # noqa: PLC0415
        SOYConfig,
        SOYFreeEnergyD2Q9,
        laplace_error_pct,
    )

    cfg = SOYConfig(nx=nx, ny=ny, radius=radius, backend=backend, **config)
    solver = SOYFreeEnergyD2Q9(cfg)
    solver.init_droplet()
    solver.run(int(steps))
    out = solver.measure()
    out["err_pct"] = laplace_error_pct(out["dp"], cfg.sigma, cfg.radius)
    out["config"] = cfg.as_dict()
    return out