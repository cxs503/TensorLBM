"""Real TensorLBM solver adapters for the YAML-driven XML template pipeline.

The pipeline's built-in fallback solvers (``runner._stub_*``) are *demonstration
stubs*: they synthesise plausible metrics without running a lattice.  This module
closes that gap for the anchor ``XVAL-LBM-YML-DRIVEN-XML-TEMPLATE-PIPELINE`` by
wiring the pipeline to the **real** D2Q9 BGK channel solver
(``tensorlbm.verification.zou_he_channel.run_channel``) so that a YAML case can
drive a genuine LBM simulation and produce REAL evidence (converged field vs
exact plane-Poiseuille solution).

Selection
---------
A YAML case opts into a real solver through ``reference.solution_fn``::

    reference:
      type: analytical
      solution_fn: poiseuille_zou_he_2d

``resolve_real_solver(case)`` returns the adapter for the registered
``solution_fn`` name, or ``None`` (the runner then falls back to its legacy
stub path, which is kept for schema/registry tests only).

YAML -> ChannelCase mapping
---------------------------
=========================  ==============================================
YAML field                 ChannelCase field
=========================  ==============================================
``geometry.ny``            ``height = ny - 2`` (two solid wall rows)
``geometry.nx``            ``length``
``physics.viscosity``      ``tau = 3*nu + 1/2`` (default nu = 0.05)
``boundary.inlet.value -   ``drho`` (both ends must be ``pressure`` type)
 boundary.outlet.value``
``boundary.inlet.type``    ``arm`` (``velocity`` inlet -> velocity-pressure)
``solver.max_steps``       ``max_steps``
``solver.convergence_``    ``tol``
``threshold``
``solver.burn_in`` etc.    ``burn_in`` / ``ramp_steps`` / ``avg_window`` /
                           ``check_every`` (optional; scaled defaults)
=========================  ==============================================
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .yaml_schema import XValYAMLCase

# Lazy import: torch is a heavy dependency; only pay for it when a real
# solver is actually requested.
_DEFAULT_NU = 0.05          # lattice kinematic viscosity (tau = 0.65)
_DEFAULT_DRHO = 4.0e-3      # inlet-outlet density difference


def _channel_case_from_yaml(case: XValYAMLCase):
    """Map a validated YAML case onto a ``ChannelCase`` (lattice units).

    Raises
    ------
    ValueError
        If the geometry/boundary fields are inconsistent with the
        Zou-He channel solver contract (e.g. channel too short, missing
        pressure boundary values).
    """
    from tensorlbm.verification.zou_he_channel import ChannelCase

    g = case.geometry
    if g is None or g.type != "channel":
        raise ValueError(f"real solver requires geometry.type='channel', got {getattr(g, 'type', None)!r}")
    if g.dimensions != "2D":
        raise ValueError(f"real solver requires 2D geometry, got {g.dimensions!r}")
    height = g.ny - 2
    if height < 4:
        raise ValueError(f"channel height {height} (ny={g.ny}) too small for the Zou-He channel solver (need ny >= 6)")
    if g.nx < 16:
        raise ValueError(f"channel length nx={g.nx} too small (need >= 16)")

    p = case.physics
    if p is None or p.model != "bgk" or p.lattice != "D2Q9":
        raise ValueError(
            f"real solver requires physics.model='bgk', lattice='D2Q9', "
            f"got model={getattr(p, 'model', None)!r}, lattice={getattr(p, 'lattice', None)!r}"
        )
    nu = p.viscosity if p.viscosity else _DEFAULT_NU
    tau = 3.0 * nu + 0.5

    # Driving: inlet/outlet density difference from the boundary block.
    drho = _DEFAULT_DRHO
    arm = "pressure-pressure"
    b = case.boundary
    if b is not None and b.inlet is not None and b.outlet is not None:
        in_type = str(b.inlet.get("type", "pressure"))
        out_type = str(b.outlet.get("type", "pressure"))
        if in_type == "velocity":
            arm = "velocity-pressure"
        if in_type == "pressure" and out_type == "pressure":
            try:
                drho = float(b.inlet["value"]) - float(b.outlet["value"])
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(
                    "pressure-pressure arm needs numeric boundary.inlet.value / boundary.outlet.value"
                ) from e
            if drho <= 0.0:
                raise ValueError(f"drho must be positive, got {drho}")

    s = case.solver
    max_steps = s.max_steps
    tol = s.convergence_threshold
    burn_in = s.burn_in if s.burn_in else max(200, max_steps // 10)
    ramp_steps = s.ramp_steps if s.ramp_steps else max(200, max_steps // 10)
    avg_window = s.avg_window if s.avg_window else max(100, max_steps // 40)
    check_every = s.check_every if s.check_every else max(100, max_steps // 80)

    return ChannelCase(
        height=g.ny - 2,
        length=g.nx,
        tau=3.0 * (p.viscosity if p.viscosity else _DEFAULT_NU) + 0.5,
        drho=drho,
        arm=arm,
        init="analytic",
        tol=tol,
        max_steps=max_steps,
        check_every=check_every,
        burn_in=burn_in,
        ramp_steps=ramp_steps,
        avg_window=avg_window,
    )


def real_zou_he_channel(case: XValYAMLCase) -> Dict[str, Any]:
    """Run the **real** D2Q9 BGK + Zou-He channel solver for a YAML case.

    Returns the solver-result contract expected by ``YamlXmlPipeline``:
    ``metrics`` (eps_inf / eps_l2 / rel_umax / shape_rel_l2), ``details``
    (solver id, convergence telemetry, gradient consistency) and
    ``profiles`` (for plotting).
    """
    import torch

    from tensorlbm.verification.zou_he_channel import (
        analytic_profile,
        run_channel,
    )
    from tensorlbm.verification.zou_he_channel.solver import profile_metrics

    cc = _channel_case_from_yaml(case)
    res = run_channel(cc)

    u_sim = res["u_profile"]
    g = res["g_measured"]
    m = profile_metrics(cc, u_sim, g)

    y = torch.arange(1, cc.ny - 1, dtype=torch.float64)
    u_ref = analytic_profile(cc, g)

    metrics = {
        "eps_inf": m["rel_linf"],
        "eps_l2": m["rel_l2"],
        "rel_umax": m["rel_umax"],
        "shape_rel_l2": m["shape_rel_l2"],
    }
    return {
        "solver": "real:zou_he_channel",
        "metrics": metrics,
        "details": {
            "solver": "real:zou_he_channel",
            "arm": cc.arm,
            "tau": cc.tau,
            "nu": cc.nu,
            "drho": cc.drho,
            "g_measured": float(g),
            "g_nominal": float(cc.g_nominal),
            "g_rel_dev": abs(float(g) - cc.g_nominal) / cc.g_nominal,
            "steps": int(res["steps"]),
            "converged": bool(res["converged"]),
            "drift_last": float(res["drift_last"]),
            "mass_defect": float(res["mass_defect"]),
            "u_max_sim": m["u_max_sim"],
            "u_max_ref": m["u_max_ref"],
        },
        "profiles": {
            "y": [float(v) for v in y.tolist()],
            "u_sim": [float(v) for v in u_sim.tolist()],
            "u_ref": [float(v) for v in u_ref.tolist()],
        },
    }


# ---------------------------------------------------------------------------
# Registry: reference.solution_fn name -> real solver adapter
# ---------------------------------------------------------------------------

REAL_SOLVER_REGISTRY: Dict[str, Callable[[XValYAMLCase], Dict[str, Any]]] = {
    "poiseuille_zou_he_2d": real_zou_he_channel,
}


def resolve_real_solver(case: XValYAMLCase) -> Optional[Callable[[XValYAMLCase], Dict[str, Any]]]:
    """Return the real-solver adapter selected by ``reference.solution_fn``.

    Returns ``None`` when the case does not name a registered real solver
    (the pipeline then uses its legacy stub path).
    """
    if case.reference is None or not case.reference.solution_fn:
        return None
    return REAL_SOLVER_REGISTRY.get(case.reference.solution_fn)
