"""MCMP verification and calibration tooling (Young-Laplace / spurious currents).

Purpose
-------
The two-component Shan-Chen path of this library
(:func:`tensorlbm.multiphase.collide_sc_two_component`) had no reusable,
quantitative acceptance tooling: a solver host could not answer "what surface
tension did this run produce, is it Young-Laplace consistent, how large are
the parasitic currents at rest, how thick is the interface, and what contact
angle does the wall impose".  This module supplies exactly that, as plain
functions returning JSON-serialisable dictionaries, so numbers can be piped
straight into an artifact/report without retyping.

The tooling is written *against this library's conventions*, not against the
conventions of the upstream projects the ideas are borrowed from:

* the cross coupling is ``G_12 < 0`` for phase separation (repulsive between
  components).  The upstream inventories use the opposite sign (positive =
  repulsive), so every borrowed formula is re-derived here rather than copied;
* the MCMP force in :func:`~tensorlbm.multiphase.sc_two_component_force` uses the
  **linear** pseudopotential ``psi(rho) = rho`` (the density field is passed in
  directly), unlike single-component paths which default to ``psi_exp``;
* the forcing is a velocity shift ``u_eq = u + tau F / rho`` (no ``1 - omega/2``
  Guo factor), so the *equilibrium* (bulk) pressure, not the kinetic pressure,
  is what a pressure measurement must use.

Provenance (OSS knowledge consumption chain)
--------------------------------------------
Each borrowed idea is listed with the inventory entry it came from and the
symbol implementing it; ``artifacts/mcmp_verify_bench/oss_consumption_chain.json``
carries the same mapping machine-readably and
``tests/benchmarks/test_mcmp_verify_bench.py`` checks every symbol resolves.

* entry 21 (``scmp-young-laplace-calibrator-with-guardrails``) - box-averaged
  liquid/gas pressure + linear-interpolated interface crossing + explicit
  *validity guard*: the upstream estimator prints a surface tension computed
  from an uninitialised radius when the density scan finds no crossing (their
  own note: measured gamma = -4.86 at R_init = 24 on a 64x64 periodic box).
  Implemented by :func:`droplet_radius` (returns ``None`` plus a guardrail code
  instead of a number) and :func:`sc_mcmp_pressure`.
* entry 20 (``scmp-two-component-planar-interface``) - the two-component
  pressure relation ``p = sum_s rho_s/3 + sum_{s,ss} g_ss psi_s psi_ss/6``
  (their ``computePressure``); re-derived here for the library sign and linear
  psi, and cross-checked against the gradient form of the same force.
* entry 5 (``geometric-wetting-contact-angle``, openLBMPM) - contact angle
  measured geometrically: extrapolate the solid colour value from fluid
  neighbours, then project the near-wall colour gradient onto the wall normal.
  Implemented by :func:`geometric_contact_angle`.
* entry 281 (``two-independent-contact-angle-impls``, walberla) - warning that
  free-surface and phase-field contact-angle conventions are not
  interchangeable and that the phase-field variant needs the interface
  thickness; hence the interface thickness is an explicit output of the contact
  angle report and both angle conventions are documented.
* entry 1 (``porter-explicit-forcing-4th-order``, openLBMPM) - force-stencil
  isotropy suppresses spurious currents; the spurious-current table therefore
  records the stencil next to the measured level so a stencil change shows up.
* entry 44 (``shan-chen-mcmp-velocity-mixing``, listLBM) - warning that mixing
  forces normalised by total density with forces normalised by component
  density is inconsistent; :func:`barycentric_velocity` makes the
  normalisation explicit and the level is reported both in lattice units and
  as a spurious capillary number.

Everything here is a *measurement* tool: it does not modify the solver.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch

from .d2q9 import equilibrium, macroscopic
from .multiphase import collide_sc_two_component
from .solver import stream

CS2 = 1.0 / 3.0

__all__ = [
    "CS2",
    "Guardrail",
    "init_static_droplet",
    "init_planar_slab",
    "simulate_planar_slab",
    "sc_mcmp_pressure",
    "barycentric_velocity",
    "droplet_radius",
    "measure_laplace",
    "estimate_coexistence",
    "young_laplace_calibration",
    "simulate_static_droplet",
    "spurious_current_benchmark",
    "radial_profile",
    "fit_tanh_width",
    "interface_width_10_90",
    "interface_thickness",
    "geometric_contact_angle",
    "circle_fit_contact_angle",
    "cap_geometry_contact_angle",
]


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


class Guardrail(dict):
    """A single validity flag: ``{"code": ..., "detail": ...}``.

    Measurements outside the domain where they are meaningful return ``None``
    for the affected quantity *and* one of these, instead of a number computed
    from undefined state (the defect recorded for inventory entry 21).
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(code=code, detail=detail)

    @property
    def code(self) -> str:
        return self["code"]

    @property
    def detail(self) -> str:
        return self["detail"]


# ---------------------------------------------------------------------------
# Pressure and velocity conventions of the two-component Shan-Chen model
# ---------------------------------------------------------------------------


def sc_mcmp_pressure(
    rho1: torch.Tensor,
    rho2: torch.Tensor,
    G_12: float,
    psi_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Equilibrium (bulk EOS) pressure of the two-component Shan-Chen model.

    With the interaction force ``F_s = -G_12 psi_s grad psi_s'`` (library sign:
    ``G_12 < 0`` separates the components) the momentum balance
    ``grad p = F_1 + F_2`` integrates to

        ``p = cs^2 (rho_1 + rho_2) + G_12 cs^2 psi_1 psi_2 + const``

    which is the pair-sum form of inventory entry 20 (their ``computePressure``
    sums ``sum_s rho_s/3 + sum_{s,ss} g_ss psi_s psi_ss / 6``; with only the two
    cross terms non-zero the ordered sum equals ``2 G_12 psi_1 psi_2 / 6 =
    G_12 psi_1 psi_2 / 3 = G_12 cs^2 psi_1 psi_2``).

    The sign mapping is *not* the obvious one: upstream (and the textbook)
    write the force as ``F = -g psi grad psi`` **without** the lattice factor
    ``cs^2``, while this library's discrete kernel returns
    ``F = +G_12 cs^2 psi grad psi'`` (the neighbour gather at ``x - c_i`` plus
    the D2Q9 weights supply ``-cs^2 grad``, and ``-G_12`` supplies the sign).
    Since the balance is ``grad p = F_1 + F_2``, the same ``G_12`` appears in
    the pressure - flipping the sign "to match upstream" produces a pressure
    that is flat across a droplet with a dip at the interface (measured, see
    ``test_pressure_pair_sum_matches_gradient_form``).

    The sign was fixed numerically on a relaxed droplet: with ``G_12 = -2.5``
    this expression gives ``p_centre > p_far`` (a droplet has a higher internal
    pressure), while the opposite-sign variant produces a nearly flat profile
    with a dip at the interface - i.e. no Laplace pressure.  See
    ``tests/benchmarks/test_mcmp_verify_bench.py``.

    Args:
        rho1: Component-1 density field.
        rho2: Component-2 density field.
        G_12: Cross coupling (< 0 for phase separation in this library).
        psi_fn: Pseudopotential. ``None`` (default) uses the linear
            ``psi(rho) = rho`` prescribed by the MCMP force kernel of this
            library.  Pass another callable only if the force kernel is changed
            correspondingly, otherwise the pressure becomes inconsistent with
            the force (entry 44 records the failure mode of mixed
            normalisations).

    Returns:
        Pressure field (same shape as the inputs) in lattice units.
    """
    if rho1.shape != rho2.shape:
        raise ValueError(f"rho1 {tuple(rho1.shape)} and rho2 {tuple(rho2.shape)} differ")
    psi1 = rho1 if psi_fn is None else psi_fn(rho1)
    psi2 = rho2 if psi_fn is None else psi_fn(rho2)
    return CS2 * (rho1 + rho2) + G_12 * CS2 * psi1 * psi2


def barycentric_velocity(
    f1: torch.Tensor,
    f2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mass-averaged velocity of the two components.

    ``u = (rho_1 u_1 + rho_2 u_2) / (rho_1 + rho_2)``.  At rest in a closed
    periodic box this is the field whose maximum magnitude is the *spurious
    current* level.  ``rho_tot`` is clamped to a tiny positive floor so a
    density-depleted cell cannot divide by zero.

    Returns:
        ``(ux, uy, rho_tot)``.
    """
    rho1, ux1, uy1 = macroscopic(f1)
    rho2, ux2, uy2 = macroscopic(f2)
    rho_tot = rho1 + rho2
    safe = torch.clamp(rho_tot, min=1e-12)
    return (rho1 * ux1 + rho2 * ux2) / safe, (rho1 * uy1 + rho2 * uy2) / safe, rho_tot


# ---------------------------------------------------------------------------
# Static droplet: initialisation, running, measurement
# ---------------------------------------------------------------------------


def _grid(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(n, dtype=torch.float32, device=device)
    return torch.meshgrid(idx, idx, indexing="ij")


def init_static_droplet(
    n: int,
    radius: float,
    rho1_in: float,
    rho2_in: float,
    rho1_out: float,
    rho2_out: float,
    *,
    interface_width: float = 2.0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tanh-profiled circular droplet of component 1 in a component-2 bath.

    The profile is ``s = 0.5 (1 - tanh((r - R0)/w))`` and
    ``rho_s = rho_s^gas + s (rho_s^liq - rho_s^gas)``; populations are set to the
    local equilibrium at rest.  Nothing else is initialised: the run relaxes to
    the model's own coexistence state, which is what gets measured.

    A smooth profile (``w >= 1`` lattice unit) matters: a staircase ``r <= R``
    initialisation injects a large unbalanced force and, at the working point
    ``G_12 ~ -2.5``, diverges within a few hundred steps.
    """
    if not isinstance(n, int) or n < 4:
        raise ValueError(f"n must be an integer grid size >= 4, got {n!r}")
    dev = torch.device(device)
    yy, xx = _grid(n, dev)
    centre = (n - 1) / 2.0
    r = torch.sqrt((xx - centre) ** 2 + (yy - centre) ** 2)
    s = 0.5 * (1.0 - torch.tanh((r - radius) / max(interface_width, 1e-6)))
    rho1 = rho1_out + s * (rho1_in - rho1_out)
    rho2 = rho2_out + s * (rho2_in - rho2_out)
    zero = torch.zeros_like(rho1)
    return equilibrium(rho1, zero, zero), equilibrium(rho2, zero, zero)


def simulate_static_droplet(
    *,
    G_12: float,
    n: int,
    radius_init: float,
    rho1_in: float = 0.92,
    rho2_in: float = 0.08,
    rho1_out: float = 0.14,
    rho2_out: float = 0.86,
    steps: int = 8000,
    tau1: float = 1.0,
    tau2: float = 1.0,
    u_eq: str = "self",
    interface_width: float = 2.0,
    stop_tol: float = 1e-4,
    check_every: int = 250,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Relax a static two-component droplet and return fields plus diagnostics.

    Convergence is declared when the relative change of ``|u|_max`` over one
    ``check_every`` window drops below ``stop_tol`` for two consecutive windows.
    Watching ``|u|_max`` rather than the kinetic energy is deliberate: a steady
    parasitic vortex field is a *plateau*, not a decay to zero, and it is that
    plateau level which the spurious-current baseline reports.  A non-finite
    field aborts immediately with ``converged=False``.

    Returns a dict with ``rho1``, ``rho2``, ``ux``, ``uy`` (final fields),
    ``u_max``, ``steps_run``, ``converged``, ``history`` (list of
    ``[step, u_max]``) and the echoed configuration.
    """
    dev = torch.device(device)
    f1, f2 = init_static_droplet(
        n,
        radius_init,
        rho1_in,
        rho2_in,
        rho1_out,
        rho2_out,
        interface_width=interface_width,
        device=dev,
    )
    history: list[list[float]] = []
    u_prev = float("inf")
    stable_windows = 0
    converged = False
    steps_run = 0
    for step in range(1, steps + 1):
        f1, f2 = collide_sc_two_component(f1, f2, G_12=G_12, tau1=tau1, tau2=tau2, u_eq=u_eq)
        f1, f2 = stream(f1), stream(f2)
        steps_run = step
        if step % check_every == 0:
            ux, uy, _ = barycentric_velocity(f1, f2)
            u_max = float(torch.sqrt(ux * ux + uy * uy).max())
            if not math.isfinite(u_max):
                history.append([float(step), float("nan")])
                break
            history.append([float(step), u_max])
            if math.isfinite(u_prev) and abs(u_max - u_prev) <= stop_tol * max(u_max, 1e-12):
                stable_windows += 1
                if stable_windows >= 2:
                    converged = True
                    break
            else:
                stable_windows = 0
            u_prev = u_max
    rho1, _, _ = macroscopic(f1)
    rho2, _, _ = macroscopic(f2)
    ux, uy, _ = barycentric_velocity(f1, f2)
    return {
        "G_12": G_12,
        "n": n,
        "radius_init": radius_init,
        "tau1": tau1,
        "tau2": tau2,
        "u_eq": u_eq,
        "steps_requested": steps,
        "steps_run": steps_run,
        "converged": converged,
        "u_max": float(torch.sqrt(ux * ux + uy * uy).max()),
        "history": history,
        "rho1": rho1,
        "rho2": rho2,
        "ux": ux,
        "uy": uy,
    }


def _box_mean(field: torch.Tensor, y0: int, x0: int, size: int) -> float:
    return float(field[y0 : y0 + size, x0 : x0 + size].mean())


def droplet_radius(
    rho1: torch.Tensor,
    rho_av: float,
    *,
    n_rays: int = 4,
) -> dict[str, Any]:
    """Radius of a centred droplet from the ``rho_1 = rho_av`` crossing.

    Upstream (inventory entry 21) walks the centre row from ``x = 0`` and stops
    at the first node denser than the mean density, interpolating linearly
    between that node and its neighbour.  Two changes are deliberate:

    * the scan is done on **all four rays** (x/y, both directions) and the mean
      radius is returned together with the spread, so a droplet that is not
      circular (walls, insufficient relaxation) is visible instead of being
      hidden by the one-sided estimate;
    * **a missing crossing returns** ``radius = None`` **plus a guardrail** -
      the upstream estimator prints a surface tension from an uninitialised
      variable in exactly that situation (their measured gamma = -4.86 at
      ``R_init = 24`` on a 64x64 periodic box, i.e. where the droplet and its
      periodic images have merged).

    Returns ``{"radius", "radius_spread", "crossings", "guardrails"}``.
    """
    guardrails: list[Guardrail] = []
    n = int(rho1.shape[0])
    if rho1.shape[0] != rho1.shape[1]:
        raise ValueError("droplet_radius expects a square field")
    c = n // 2
    # torch has no negative-step slicing, hence the explicit flips.
    scans = [
        [float(v) for v in rho1[c, c:]],  # +x
        [float(v) for v in torch.flip(rho1[c, : c + 1], dims=(0,))],  # -x
        [float(v) for v in rho1[c:, c]],  # +y
        [float(v) for v in torch.flip(rho1[: c + 1, c], dims=(0,))],  # -y
    ]
    crossings = 0
    radii: list[float] = []
    for ray in scans[:n_rays]:
        if not all(math.isfinite(v) for v in ray):
            guardrails.append(
                Guardrail("non_finite_field", "the density ray contains NaN/Inf: the run diverged")
            )
            return {
                "radius": None,
                "radius_spread": None,
                "crossings": 0,
                "guardrails": guardrails,
            }
        if ray[0] <= rho_av:
            guardrails.append(
                Guardrail(
                    "centre_not_liquid",
                    f"rho1 at the centre is {ray[0]:.4f} <= rho_av {rho_av:.4f}: "
                    "no droplet to measure",
                )
            )
            break
        inside = [i for i, v in enumerate(ray) if v > rho_av]
        if len(inside) == len(ray):
            continue  # ray never leaves the droplet -> no crossing on this ray
        i_in = inside[-1]
        r_in, r_out = ray[i_in], ray[i_in + 1]
        denom = r_in - r_out
        frac = 0.0 if abs(denom) < 1e-12 else (r_in - rho_av) / denom
        radii.append(i_in + min(max(frac, 0.0), 1.0))
        crossings += 1
    if crossings < n_rays:
        guardrails.append(
            Guardrail(
                "no_crossing",
                f"only {crossings}/{n_rays} rays crossed rho_av={rho_av:.4f}; droplet "
                "likely dissolved or merged with its periodic images",
            )
        )
    if not radii:
        return {
            "radius": None,
            "radius_spread": None,
            "crossings": crossings,
            "guardrails": guardrails,
        }
    radius = sum(radii) / len(radii)
    spread = max(radii) - min(radii)
    if radius > 0.35 * n:
        guardrails.append(
            Guardrail(
                "periodic_image_risk",
                f"R={radius:.2f} exceeds 0.35*n={0.35 * n:.2f}; the droplet interacts "
                "with its periodic images and the Laplace pressure is inflated",
            )
        )
    # The interpolation between two nodes leaves O(1) lattice units of
    # x/y asymmetry even for a perfect circle, so the tolerance is absolute
    # (1.5 lu) OR relative (5%), whichever is larger.
    if spread > max(1.5, 0.05 * radius):
        guardrails.append(
            Guardrail(
                "anisotropic",
                f"ray radii spread {spread:.2f} > max(1.5, 5% of R={radius:.2f}): the "
                "interface is not circular (not relaxed, or a wall is present)",
            )
        )
    return {
        "radius": radius,
        "radius_spread": spread,
        "crossings": crossings,
        "guardrails": guardrails,
    }


def measure_laplace(run: dict[str, Any], *, box: int = 4, dims: int = 2) -> dict[str, Any]:
    """Measure ``Delta p``, ``R`` and ``sigma = Delta p R / (dims - 1)`` for one run.

    ``dims`` is the *geometric* dimension of the interface: a 2D circular
    droplet (a cylinder of unit depth) obeys ``Delta p = sigma / R``, a 3D
    sphere ``Delta p = 2 sigma / R``; hence ``sigma = Delta p R / (dims - 1)``.
    The default is the dimension of the fields this module produces (2).  Until
    2026-11-12 this function returned ``Delta p R / 2`` - the 3D sphere factor -
    for these 2D fields, i.e. sigma was reported **half** of its value (and any
    derived quantity such as ``Ca_sp = mu |u|/sigma`` **twice**); fixed here.

    Liquid pressure is averaged in a ``box x box`` window at the centre, gas
    pressure in a window at a corner (inventory entry 21), both with
    :func:`sc_mcmp_pressure`.  ``rho_av = (rho_l + rho_g)/2`` of the *same*
    windows defines the interface crossing for the radius, exactly as upstream.

    Returns a JSON-serialisable dict; unmeasurable quantities are ``None`` with
    a guardrail, never a stale or uninitialised value.
    """
    if dims not in (2, 3):
        raise ValueError(f"dims must be 2 (cylinder) or 3 (sphere), got {dims!r}")
    rho1, rho2 = run["rho1"], run["rho2"]
    n = int(rho1.shape[0])
    if not (bool(torch.isfinite(rho1).all()) and bool(torch.isfinite(rho2).all())):
        return {
            "G_12": run["G_12"],
            "n": n,
            "radius_init": run["radius_init"],
            "laplace_dims": dims,
            "steps_run": run["steps_run"],
            "converged": False,
            "u_max": run["u_max"],
            "p_liq": None,
            "p_gas": None,
            "dp": None,
            "radius": None,
            "radius_spread": None,
            "sigma": None,
            "rho1_liq": None,
            "rho1_gas": None,
            "rho2_liq": None,
            "rho2_gas": None,
            "density_ratio": None,
            "guardrails": [
                dict(
                    Guardrail(
                        "non_finite_field",
                        f"density field is not finite after {run['steps_run']} steps: the run diverged",
                    )
                )
            ],
        }
    p = sc_mcmp_pressure(rho1, rho2, run["G_12"])
    c = n // 2
    lo = c - box // 2
    p_liq = _box_mean(p, lo, lo, box)
    p_gas = _box_mean(p, 0, 0, box)
    rho1_liq = _box_mean(rho1, lo, lo, box)
    rho1_gas = _box_mean(rho1, 0, 0, box)
    rho2_liq = _box_mean(rho2, lo, lo, box)
    rho2_gas = _box_mean(rho2, 0, 0, box)
    rho_av = 0.5 * (rho1_liq + rho1_gas)
    measured = droplet_radius(rho1, rho_av)
    guardrails: list[Guardrail] = list(measured["guardrails"])
    if not run["converged"]:
        guardrails.append(
            Guardrail(
                "not_converged",
                f"stopped at step {run['steps_run']} without reaching a |u| plateau",
            )
        )
    dp = p_liq - p_gas
    radius = measured["radius"]
    sigma = None if radius is None else dp * radius / (dims - 1)
    if sigma is not None and sigma <= 0:
        guardrails.append(
            Guardrail(
                "negative_sigma",
                f"Delta p = {dp:.6f} gives sigma = {sigma:.6f} <= 0: the pressure "
                "convention does not describe a droplet in this state",
            )
        )
    return {
        "G_12": run["G_12"],
        "n": n,
        "radius_init": run["radius_init"],
        "laplace_dims": dims,
        "steps_run": run["steps_run"],
        "converged": run["converged"],
        "u_max": run["u_max"],
        "p_liq": p_liq,
        "p_gas": p_gas,
        "dp": dp,
        "radius": radius,
        "radius_spread": measured["radius_spread"],
        "sigma": sigma,
        "rho1_liq": rho1_liq,
        "rho1_gas": rho1_gas,
        "rho2_liq": rho2_liq,
        "rho2_gas": rho2_gas,
        "density_ratio": rho1_liq / max(rho1_gas, 1e-12),
        "guardrails": [dict(g) for g in guardrails],
    }


def estimate_coexistence(
    *,
    G_12: float,
    n: int = 96,
    geometry: str = "planar",
    steps: int = 8000,
    device: torch.device | str = "cpu",
    **kwargs: Any,
) -> dict[str, Any]:
    """Coexistence densities, from a flat slab (default) or a large droplet.

    ``geometry="planar"`` is the default on purpose: a droplet's vapour density
    is shifted by its curvature (Kelvin effect), so coexistence densities taken
    from a droplet do *not* put the later radius series on coexistence - the
    Laplace fit then acquires a curvature-independent offset ``Delta p_0``
    (measured: ``Delta p_0 = 2.1e-3``, i.e. 24% of the Laplace term at
    ``R = 10``, max relative radius error 23% at ``G_12 = -2.5``).  A flat slab
    removes that bias, and it doubles as a check of the pressure convention:
    for a flat interface ``p_liq - p_gas`` must vanish (measured 7.7e-7 on a
    96x96 slab, i.e. the pressure relation is consistent with the force).

    The returned dict also carries that ``flat_interface_dp`` check for the
    planar geometry.
    """
    if geometry == "planar":
        run = simulate_planar_slab(
            G_12=G_12,
            n=n,
            half_thickness=kwargs.pop("half_thickness", 0.23 * n),
            steps=steps,
            device=device,
            **kwargs,
        )
        rho1, rho2 = run["rho1"], run["rho2"]
        c = n // 2
        box = 8
        p = sc_mcmp_pressure(rho1, rho2, G_12)
        out = {
            "G_12": G_12,
            "geometry": "planar",
            "n": n,
            "converged": run["converged"],
            "steps_run": run["steps_run"],
            "u_max": run["u_max"],
            "rho1_liq": _box_mean(rho1, c - box // 2, c - box // 2, box),
            "rho2_liq": _box_mean(rho2, c - box // 2, c - box // 2, box),
            "rho1_gas": _box_mean(rho1, 2, 2, box),
            "rho2_gas": _box_mean(rho2, 2, 2, box),
            "flat_interface_dp": _box_mean(p, c - box // 2, c - box // 2, box)
            - _box_mean(p, 2, 2, box),
        }
    else:
        radius = kwargs.pop("radius_init", 20.0)
        run = simulate_static_droplet(
            G_12=G_12, n=n, radius_init=radius, steps=steps, device=device, **kwargs
        )
        rho1, rho2 = run["rho1"], run["rho2"]
        c = n // 2
        box = 4
        out = {
            "G_12": G_12,
            "geometry": "droplet",
            "n": n,
            "converged": run["converged"],
            "steps_run": run["steps_run"],
            "u_max": run["u_max"],
            "rho1_liq": _box_mean(rho1, c - box // 2, c - box // 2, box),
            "rho2_liq": _box_mean(rho2, c - box // 2, c - box // 2, box),
            "rho1_gas": _box_mean(rho1, 0, 0, box),
            "rho2_gas": _box_mean(rho2, 0, 0, box),
        }
    out["density_ratio"] = out["rho1_liq"] / max(out["rho1_gas"], 1e-12)
    return out


def init_planar_slab(
    n: int,
    rho1_in: float,
    rho2_in: float,
    rho1_out: float,
    rho2_out: float,
    *,
    half_thickness: float = 20.0,
    interface_width: float = 2.0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Planar (slab) two-phase initialisation for a curvature-free coexistence run.

    Component 1 fills a horizontal slab of half-thickness ``half_thickness``
    centred in the box, component 2 the rest; the interface is a tanh profile.
    Because the interface is flat, the bulk densities this run converges to are
    the *planar* coexistence densities - unlike a droplet, whose curvature
    shifts the vapour density (Kelvin effect) and therefore biases any later
    Laplace-pressure measurement by a curvature-independent offset.
    """
    dev = torch.device(device)
    yy, _ = _grid(n, dev)
    centre = (n - 1) / 2.0
    s = 0.5 * (
        1.0 - torch.tanh((torch.abs(yy - centre) - half_thickness) / max(interface_width, 1e-6))
    )
    rho1 = rho1_out + s * (rho1_in - rho1_out)
    rho2 = rho2_out + s * (rho2_in - rho2_out)
    zero = torch.zeros_like(rho1)
    return equilibrium(rho1, zero, zero), equilibrium(rho2, zero, zero)


def simulate_planar_slab(
    *,
    G_12: float,
    n: int,
    rho1_in: float = 0.92,
    rho2_in: float = 0.08,
    rho1_out: float = 0.14,
    rho2_out: float = 0.86,
    half_thickness: float = 20.0,
    steps: int = 8000,
    tau1: float = 1.0,
    tau2: float = 1.0,
    u_eq: str = "self",
    interface_width: float = 2.0,
    stop_tol: float = 1e-4,
    check_every: int = 250,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Relax a planar two-phase slab (same convergence rule as the droplet run)."""
    dev = torch.device(device)
    f1, f2 = init_planar_slab(
        n,
        rho1_in,
        rho2_in,
        rho1_out,
        rho2_out,
        half_thickness=half_thickness,
        interface_width=interface_width,
        device=dev,
    )
    history: list[list[float]] = []
    u_prev = float("inf")
    stable = 0
    converged = False
    steps_run = 0
    for step in range(1, steps + 1):
        f1, f2 = collide_sc_two_component(f1, f2, G_12=G_12, tau1=tau1, tau2=tau2, u_eq=u_eq)
        f1, f2 = stream(f1), stream(f2)
        steps_run = step
        if step % check_every == 0:
            ux, uy, _ = barycentric_velocity(f1, f2)
            u_max = float(torch.sqrt(ux * ux + uy * uy).max())
            if not math.isfinite(u_max):
                history.append([float(step), float("nan")])
                break
            history.append([float(step), u_max])
            if math.isfinite(u_prev) and abs(u_max - u_prev) <= stop_tol * max(u_max, 1e-12):
                stable += 1
                if stable >= 2:
                    converged = True
                    break
            else:
                stable = 0
            u_prev = u_max
    rho1, _, _ = macroscopic(f1)
    rho2, _, _ = macroscopic(f2)
    ux, uy, _ = barycentric_velocity(f1, f2)
    return {
        "G_12": G_12,
        "n": n,
        "half_thickness": half_thickness,
        "steps_requested": steps,
        "steps_run": steps_run,
        "converged": converged,
        "u_max": float(torch.sqrt(ux * ux + uy * uy).max()),
        "history": history,
        "rho1": rho1,
        "rho2": rho2,
        "ux": ux,
        "uy": uy,
    }


def young_laplace_calibration(
    G_values: list[float],
    radii: list[float],
    *,
    n: int = 64,
    steps: int = 8000,
    coexist_steps: int = 8000,
    threshold: float = 0.05,
    geometry: str = "planar",
    dims: int = 2,
    device: torch.device | str = "cpu",
    **kwargs: Any,
) -> dict[str, Any]:
    """Young-Laplace calibration: ``Delta p`` versus ``(dims - 1) sigma / R``.

    Protocol (pre-registered in the accompanying report, not tuned after the
    numbers were seen):

    1. for each ``G_12`` a large droplet fixes the coexistence densities;
    2. for each radius of ``radii`` a droplet is initialised with those
       densities and relaxed until ``|u|_max`` plateaus;
    3. ``sigma_i = Delta p_i R_i / (dims - 1)`` is measured per case, and a
       least-squares line ``Delta p = a (1/R) + b`` is fitted across the radius
       series of that ``G_12`` level; ``sigma_fit = a / (dims - 1)``;
    4. **error metric**: ``err_i = |sigma_i - sigma_fit| / sigma_fit`` - the
       Laplace law is satisfied when the measured surface tension does not
       depend on the droplet radius.  ``PASS`` requires ``max_i err_i <
       threshold`` (default 5%) *and* all guardrails clean *and* ``|b|/a < 0.2``
       (a large intercept would mean the fit is a line for the wrong reason).

    Args:
        G_values: Coupling values defining the >= 3 surface-tension levels.
        radii: Initial droplet radii (>= 3 points per level is required to test
            a law; fewer is accepted but flagged ``too_few_radii``).
        n: Box size (square, periodic).
        steps: Relaxation budget per droplet.
        coexist_steps: Relaxation budget for the coexistence run.
        threshold: Relative error threshold for PASS.
        geometry: Coexistence geometry, ``"planar"`` (default, curvature free)
            or ``"droplet"``.
        dims: Geometric dimension of the interface, ``2`` (default; circular
            droplet, ``Delta p = sigma / R``) or ``3`` (sphere).  These fields
            are 2D, so the default is the physically correct one.
        device: Torch device for the runs.
        **kwargs: Forwarded to :func:`simulate_static_droplet`
            (``tau1``, ``tau2``, ``u_eq``, ``interface_width``).

    Returns:
        ``{"levels": [...], "cases": [...], "threshold": ..., "pass": bool}``.
    """
    cases: list[dict[str, Any]] = []
    levels: list[dict[str, Any]] = []
    for G in G_values:
        coex = estimate_coexistence(
            G_12=G, n=max(n, 96), geometry=geometry, steps=coexist_steps, device=device, **kwargs
        )
        level_cases: list[dict[str, Any]] = []
        for radius in radii:
            run = simulate_static_droplet(
                G_12=G,
                n=n,
                radius_init=radius,
                rho1_in=coex["rho1_liq"],
                rho2_in=coex["rho2_liq"],
                rho1_out=coex["rho1_gas"],
                rho2_out=coex["rho2_gas"],
                steps=steps,
                device=device,
                **kwargs,
            )
            case = measure_laplace(run, dims=dims)
            case["coexistence"] = {k: v for k, v in coex.items()}
            level_cases.append(case)
            cases.append(case)
        levels.append(_fit_laplace_level(G, level_cases, threshold, dims=dims))
    ok = bool(levels) and all(level["pass"] for level in levels)
    return {
        "protocol": {
            "n": n,
            "steps": steps,
            "coexist_steps": coexist_steps,
            "radii": list(radii),
            "G_values": list(G_values),
            "threshold": threshold,
            "coexistence_geometry": geometry,
            "laplace_dims": dims,
            "law": "dp = (dims - 1) * sigma / R",
            "stress": "dP vs 1/R across radii, per G_12 level; sigma from the slope",
            "kwargs": {k: v for k, v in kwargs.items() if k != "device"},
        },
        "levels": levels,
        "cases": cases,
        "pass": ok,
    }


def _fit_laplace_level(
    G_12: float, level_cases: list[dict[str, Any]], threshold: float, dims: int = 2
) -> dict[str, Any]:
    """One-parameter Young-Laplace test plus a free-line diagnostic.

    Acceptance metric (pre-registered, see :func:`young_laplace_calibration`):
    the law ``Delta p = (dims - 1) sigma / R`` has exactly **one** unknown per
    coupling level, so ``sigma`` is fitted through the origin,

        ``sigma_0 = sum_i x_i y_i / sum_i x_i^2``  with ``x_i = (dims-1)/R_i``,
        ``y_i = Delta p_i``,

    and the per-case error is ``|y_i - sigma_0 x_i| / (sigma_0 x_i)``.  A free
    two-parameter line ``Delta p = a/R + b`` is fitted as well, but only as a
    *diagnostic*: its second parameter absorbs any curvature-independent bulk
    pressure offset (the Shan-Chen model has one whenever the bulk compositions
    are not exactly at coexistence), and reporting its slope as "the surface
    tension error" would silently relax the criterion.  ``slope_sigma`` vs
    ``sigma_origin`` therefore measures the systematic curvature trend, and a
    deviation above ``threshold`` raises the informational guardrail
    ``curvature_trend``.
    """
    usable = [c for c in level_cases if c["sigma"] is not None and c["radius"]]
    guardrails: list[Guardrail] = []
    for c in level_cases:
        for g in c["guardrails"]:
            guardrails.append(Guardrail(g["code"], f"R_init={c['radius_init']}: {g['detail']}"))
    if len(usable) < 2:
        return {
            "G_12": G_12,
            "sigma_origin": None,
            "sigma_slope": None,
            "sigma_cases": [c["sigma"] for c in usable],
            "slope": None,
            "intercept": None,
            "r2": None,
            "max_rel_err": None,
            "pass": False,
            "guardrails": [dict(g) for g in guardrails],
            "reason": "fewer than two usable radii; no fit",
        }
    if len(usable) < 3:
        guardrails.append(
            Guardrail("too_few_radii", f"only {len(usable)} usable radii (law test wants >= 3)")
        )
    xs = [(dims - 1.0) / c["radius"] for c in usable]
    ys = [c["dp"] for c in usable]
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    sigma_origin = sxy / sxx if sxx > 1e-30 else float("nan")
    errs = [abs(y - sigma_origin * x) / abs(sigma_origin * x) for x, y in zip(xs, ys)]
    max_err = max(errs)
    for c, e in zip(usable, errs):
        c["sigma_origin_level"] = sigma_origin
        c["rel_err_vs_law"] = e
    # free-line diagnostic
    m = len(xs)
    xs_inv = [1.0 / c["radius"] for c in usable]
    sx, sy = sum(xs_inv), sum(ys)
    sx2 = sum(x * x for x in xs_inv)
    sxi = sum(x * y for x, y in zip(xs_inv, ys))
    denom = m * sx2 - sx * sx
    slope = (m * sxi - sx * sy) / denom if abs(denom) > 1e-30 else float("nan")
    intercept = (sy - slope * sx) / m
    ss_tot = sum((y - sy / m) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs_inv, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")
    sigma_slope = slope / (dims - 1)
    offset = (
        max(abs(intercept) * c["radius"] / abs(slope) for c in usable) if slope else float("inf")
    )
    if sigma_slope and abs(sigma_slope - sigma_origin) / abs(sigma_origin) > threshold:
        guardrails.append(
            Guardrail(
                "curvature_trend",
                f"free-line sigma {sigma_slope:.5f} differs from the through-origin "
                f"{sigma_origin:.5f} by {abs(sigma_slope - sigma_origin) / abs(sigma_origin):.1%} "
                f"over R = {min(c['radius'] for c in usable):.1f}..{max(c['radius'] for c in usable):.1f}: "
                "the measured sigma still drifts with curvature (Tolman-type trend)",
            )
        )
    hard = {"no_crossing", "not_converged", "negative_sigma", "centre_not_liquid"}
    ok = bool(max_err < threshold and not any(g.code in hard for g in guardrails))
    return {
        "G_12": G_12,
        "sigma_origin": sigma_origin,
        "sigma_slope": sigma_slope,
        "laplace_dims": dims,
        "sigma_cases": [c["sigma"] for c in usable],
        "radii_measured": [c["radius"] for c in usable],
        "slope": slope,
        "intercept": intercept,
        "max_offset_fraction": offset,
        "r2": r2,
        "max_rel_err": max_err,
        "per_case_rel_err": errs,
        "point_errors_5pct": [bool(e < threshold) for e in errs],
        "pass": ok,
        "threshold": threshold,
        "guardrails": [dict(g) for g in guardrails],
    }


# ---------------------------------------------------------------------------
# Spurious currents
# ---------------------------------------------------------------------------

#: Force stencil used by the measured solver path (recorded next to the result
#: so a stencil change is visible in the baseline, inventory entry 1).
FORCE_STENCIL = "D2Q9 w=[4/9,1/9,1/36], neighbour gather, velocity-shift forcing"


def spurious_current_benchmark(
    configs: list[dict[str, Any]],
    *,
    steps: int = 12000,
    **kwargs: Any,
) -> dict[str, Any]:
    """``|u|_max`` of a droplet at rest over a (grid x coupling) table.

    Each row of ``configs`` is ``{"n": int, "G_12": float, "radius": float}``.
    Rows are independent so a caller can farm them out to workers; the function
    itself is sequential and side-effect free.

    Reported per row:

    * ``u_max`` - lattice units, mass-averaged velocity
      (:func:`barycentric_velocity`).  This is *the* raw number.
    * ``u_max_over_cs`` - spurious Mach number.
    * ``ca_spurious`` - ``mu u_max / sigma`` with the lattice viscosity
      ``mu = rho_bar (tau - 0.5)/3``; a capillary number is the only
      viscosity-independent way to compare against a literature level, because
      ``u_max`` itself scales with ``sigma / mu``.
    * ``sigma``, ``dp``, ``radius`` from :func:`measure_laplace`, so the
      current level is always reported together with the surface tension that
      drives it (a "small" u_max at a small sigma is not progress).
    """
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        run = simulate_static_droplet(
            G_12=cfg["G_12"],
            n=cfg["n"],
            radius_init=cfg["radius"],
            steps=steps,
            **kwargs,
        )
        case = measure_laplace(run)
        rho_bar = float((run["rho1"] + run["rho2"]).mean())
        mu = rho_bar * (run["tau1"] - 0.5) / 3.0
        u_max = case["u_max"]
        sigma = case["sigma"]
        rows.append(
            {
                "n": cfg["n"],
                "G_12": cfg["G_12"],
                "radius_init": cfg["radius"],
                "steps_run": case["steps_run"],
                "converged": case["converged"],
                "u_max": u_max,
                "u_max_over_cs": u_max / math.sqrt(CS2),
                "sigma": sigma,
                "ca_spurious": None if not sigma else mu * u_max / sigma,
                "mu_lattice": mu,
                "r2": sigma * 2.0 / max(u_max, 1e-30) if sigma else None,
                "dp": case["dp"],
                "radius": case["radius"],
                "density_ratio": case["density_ratio"],
                "guardrails": case["guardrails"],
                "stencil": FORCE_STENCIL,
            }
        )
    return {"stencil": FORCE_STENCIL, "rows": rows}


# ---------------------------------------------------------------------------
# Interface thickness
# ---------------------------------------------------------------------------


def radial_profile(
    field: torch.Tensor,
    *,
    bins: int | None = None,
    r_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Azimuthally averaged profile of a centred field.

    Returns ``(r_centres, mean_values)`` with ``bins`` annular bins (default
    one bin per lattice unit of radius).  Averaging over the azimuth is what
    makes the interface thickness of a droplet a meaningful 1-D quantity (a
    single ray is one lattice unit noisy); one-unit bins keep the radial
    resolution equal to the grid resolution.
    """
    n = int(field.shape[0])
    bins = max(8, n // 2) if bins is None else bins
    centre = (n - 1) / 2.0
    yy, xx = _grid(n, field.device)
    r = torch.sqrt((xx - centre) ** 2 + (yy - centre) ** 2)
    r_max = 0.5 * n if r_max is None else r_max
    edges = torch.linspace(0.0, r_max, bins + 1, device=field.device)
    centres = 0.5 * (edges[:-1] + edges[1:])
    means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (r >= lo) & (r < hi)
        means.append(field[mask].mean() if bool(mask.any()) else torch.tensor(float("nan")))
    return centres, torch.stack(means)


def fit_tanh_width(
    r: torch.Tensor,
    values: torch.Tensor,
    rho_liq: float,
    rho_gas: float,
    *,
    band: float = 0.95,
) -> dict[str, Any]:
    """Interface width from the tanh model, by algebraic inversion.

    A tanh profile ``rho = rho_g + drho/2 (1 - tanh((r - R)/w))`` linearises
    exactly as ``atanh(1 - 2 (rho - rho_g)/drho) = (r - R)/w``, so a straight
    least-squares line in the interior band (``|1 - 2u| < band``) gives ``w``
    as the reciprocal slope and ``R`` as the zero crossing.  No iterative fit,
    no external optimiser, and the residual ``r2`` exposes a profile that is
    not tanh-shaped (e.g. lattice staircase).
    """
    drho = rho_liq - rho_gas
    if abs(drho) < 1e-12:
        return {
            "w": None,
            "R": None,
            "r2": None,
            "n_points": 0,
            "guardrail": {
                "code": "no_density_jump",
                "detail": f"rho_liq={rho_liq:.4f} == rho_gas={rho_gas:.4f}",
            },
        }
    u = (values - rho_gas) / drho
    arg = 1.0 - 2.0 * u
    mask = arg.abs() < band
    x = r[mask]
    y = torch.atanh(arg[mask])
    if int(mask.sum()) < 2:
        return {
            "w": None,
            "R": None,
            "r2": None,
            "n_points": int(mask.sum()),
            "guardrail": {"code": "too_few_points", "detail": "band contains < 2 nodes"},
        }
    xm, ym = float(x.mean()), float(y.mean())
    sxx = float(((x - xm) ** 2).sum())
    sxy = float(((x - xm) * (y - ym)).sum())
    slope = sxy / sxx if sxx > 1e-30 else float("nan")
    intercept = ym - slope * xm
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")
    return {
        "w": 1.0 / slope if slope else None,
        "R": -intercept / slope if slope else None,
        "r2": r2,
        "n_points": int(mask.sum()),
        "guardrail": None,
    }


def interface_width_10_90(w_tanh: float) -> float:
    """10%-90% thickness of a tanh profile with length scale ``w``.

    ``2 atanh(0.8) w = 2.1972 w``; this conversion is what makes a thickness
    number comparable between conventions (inventory entry 281: contact-angle
    conventions are not interchangeable, so the thickness definition has to
    travel with the number).
    """
    return 2.0 * math.atanh(0.8) * w_tanh


def interface_thickness(
    rho1: torch.Tensor,
    *,
    bins: int | None = None,
    band: float = 0.95,
) -> dict[str, Any]:
    """Interface thickness of a centred droplet from the azimuthal profile.

    Returns ``{"w_tanh", "width_10_90", "R_fit", "r2", "rho_liq", "rho_gas",
    "guardrails"}``.  ``R_fit`` is the mid-density contour radius from the same
    fit and is an independent check of :func:`droplet_radius` (which uses a
    single ray interpolation).
    """
    guardrails: list[dict[str, Any]] = []
    r, prof = radial_profile(rho1, bins=bins)
    finite = torch.isfinite(prof)
    if int(finite.sum()) < 4:
        return {
            "w_tanh": None,
            "width_10_90": None,
            "R_fit": None,
            "r2": None,
            "rho_liq": None,
            "rho_gas": None,
            "guardrails": [{"code": "insufficient_profile", "detail": "fewer than 4 finite bins"}],
            "profile_r": r.tolist(),
            "profile_rho": prof.tolist(),
        }
    rho_liq = float(prof[0])
    rho_gas = float(prof[-1])
    fit = fit_tanh_width(r, prof, rho_liq, rho_gas, band=band)
    if fit.get("guardrail"):
        guardrails.append(fit["guardrail"])
    w = fit["w"]
    return {
        "w_tanh": w,
        "width_10_90": None if w is None else interface_width_10_90(abs(w)),
        "R_fit": fit["R"],
        "r2": fit["r2"],
        "rho_liq": rho_liq,
        "rho_gas": rho_gas,
        "n_points": fit["n_points"],
        "guardrails": guardrails,
        "profile_r": [float(v) for v in r],
        "profile_rho": [float(v) for v in prof],
    }


# ---------------------------------------------------------------------------
# Contact angle
# ---------------------------------------------------------------------------


def _flat_wall(mask: torch.Tensor) -> dict[str, Any]:
    """Classify a solid mask as a flat wall and return its inward normal.

    Only flat wall strips are supported: the measurement projects the near-wall
    density gradient onto one constant normal, so a curved wall would silently
    average over a varying projection direction.  Anything else returns
    ``normal=None`` plus a guardrail instead of a wrong number.
    """
    if mask.dtype != torch.bool:
        raise TypeError("solid mask must be boolean")
    ny, nx = mask.shape
    rows = mask.all(dim=1)
    cols = mask.all(dim=0)
    if bool(mask.any()):
        k = int(rows.to(torch.int8).sum().item())
        if k and bool(rows[:k].all()) and not bool(mask[k:].any()):
            return {"normal": (0.0, 1.0), "wall_index": k, "side": "bottom", "ok": True}
        if k and bool(rows[-k:].all()) and not bool(mask[:-k].any()):
            return {"normal": (0.0, -1.0), "wall_index": ny - k, "side": "top", "ok": True}
        k = int(cols.to(torch.int8).sum().item())
        if k and bool(cols[:k].all()) and not bool(mask[:, k:].any()):
            return {"normal": (1.0, 0.0), "wall_index": k, "side": "left", "ok": True}
        if k and bool(cols[-k:].all()) and not bool(mask[:, :-k].any()):
            return {"normal": (-1.0, 0.0), "wall_index": nx - k, "side": "right", "ok": True}
    return {
        "normal": None,
        "wall_index": None,
        "side": "unsupported",
        "ok": False,
        "detail": "solid mask is not a single flat wall strip",
    }


def _shift(a: torch.Tensor, dy: int, dx: int, fill: float | bool = 0.0) -> torch.Tensor:
    """Non-periodic shift: ``out[y, x] = a[y - dy, x - dx]`` outside -> ``fill``."""
    ny, nx = a.shape
    out = torch.full_like(a, fill)
    ys, yd = slice(max(-dy, 0), ny + min(-dy, 0)), slice(max(dy, 0), ny + min(dy, 0))
    xs, xd = slice(max(-dx, 0), nx + min(-dx, 0)), slice(max(dx, 0), nx + min(dx, 0))
    out[yd, xd] = a[ys, xs]
    return out


def _near_wall_fluid(
    mask: torch.Tensor,
    normal: tuple[float, float],
    layers: int,
    first_layer: int = 1,
) -> torch.Tensor:
    """Fluid nodes whose ``k``-th neighbour along ``-normal`` is solid, ``first<=k<first+layers``."""
    nx_dir, ny_dir = int(round(normal[0])), int(round(normal[1]))
    fluid = ~mask
    near = torch.zeros_like(mask)
    for k in range(first_layer, first_layer + layers):
        near |= fluid & _shift(mask, k * ny_dir, k * nx_dir, False)
    return near


def _fill_solid(
    rho: torch.Tensor, mask: torch.Tensor, normal: tuple[float, float], order: int
) -> torch.Tensor:
    """Fill solid nodes from fluid neighbours along the wall normal.

    ``order=1`` copies the nearest fluid node (zero-gradient fill, a mirror
    wall).  ``order=2`` linearly extrapolates through the two nearest fluid
    layers, ``rho_solid = 2 rho_(1) - rho_(2)``.  The order matters: a
    zero-gradient fill has zero normal derivative at the wall by construction,
    which biases the contact angle towards 90 degrees, so both are exposed and
    the report states which one was used.  The propagation is fully vectorised
    (one tensor shift per solid layer) so it stays O(layers) tensor ops rather
    than O(n^2) Python steps.
    """
    nx_dir, ny_dir = int(round(normal[0])), int(round(normal[1]))
    fluid = ~mask
    out = rho.clone()
    filled = fluid.clone()
    if order >= 2:
        a = _shift(rho, -ny_dir, -nx_dir, 0.0)
        b = _shift(rho, -2 * ny_dir, -2 * nx_dir, 0.0)
        f1 = _shift(fluid, -ny_dir, -nx_dir, False)
        f2 = _shift(fluid, -2 * ny_dir, -2 * nx_dir, False)
        layer1 = mask & f1
        extrap = torch.where(layer1 & f2, 2.0 * a - b, a)
        out = torch.where(layer1, extrap, out)
        filled = filled | layer1
    for _ in range(max(rho.shape)):
        need = mask & ~filled
        if not bool(need.any()):
            break
        src = _shift(out, -ny_dir, -nx_dir, 0.0)
        src_ok = _shift(filled, -ny_dir, -nx_dir, False)
        take = need & src_ok
        out = torch.where(take, src, out)
        filled = filled | take
    return out


def geometric_contact_angle(
    rho: torch.Tensor,
    solid_mask: torch.Tensor,
    *,
    order: int = 2,
    layers: int = 1,
    first_layer: int = 2,
    min_grad_frac: float = 0.2,
    band: tuple[float, float] = (0.15, 0.85),
) -> dict[str, Any]:
    """Contact angle from the near-wall density gradient (openLBMPM geometry).

    Inventory entry 5 imposes wetting geometrically - extrapolate the solid
    colour value from fluid neighbours, then use the colour gradient at
    near-wall fluid nodes.  The same construction is used here in *measurement*
    direction: after filling the solid rows, the interface normal

        ``n_out = -grad(rho) / |grad(rho)|``

    (pointing from liquid to gas, since ``rho`` is the liquid density) is
    evaluated at the first ``layers`` fluid rows, and

        ``cos(theta) = -(grad rho . n_wall) / |grad rho|``

    with ``n_wall`` the unit normal pointing from the wall into the fluid (the
    minus sign because ``n_out = -grad rho/|grad rho|`` and because the physical
    convention has the disc centre *below* the wall for a wetting drop).  For a
    circular cap this returns ``cos(theta) = -h/R`` with ``h`` the centre height
    above the wall: hemisphere ``h=0`` -> 90 degrees, wetting ``h<0`` ->
    ``theta<90``.

    Args:
        rho: Liquid density field with the interface crossing the wall.
        solid_mask: Boolean wall mask (flat wall only, see :func:`_flat_wall`).
        order: Solid fill order (1 = mirror, 2 = linear extrapolation).
        layers: Number of near-wall fluid layers to average over.
        first_layer: First fluid layer counted (1 = wall-adjacent).  The default
            is 2: the wall-adjacent layer has zero or extrapolated normal
            derivative, so its gradient direction is biased towards the wall
            normal; calibrating on synthetic caps (``test_contact_angle_*``)
            showed a +2.5 degree bias at theta = 90 for ``first_layer=1`` and
            +2.5 -> within +-3 degrees for ``first_layer=2``.
        min_grad_frac: Nodes whose gradient magnitude is below this fraction of
            the maximum near-wall gradient are excluded (they are not in the
            contact region and their direction is noise).
        band: Normalised density window ``(lo, hi)`` of the diffuse interface,
            ``u = (rho - rho_gas)/(rho_liq - rho_gas)``.  Only near-wall nodes
            *on the interface* carry the contact-line normal; nodes in the
            bulk wetting film or in the gas have a gradient that is set by the
            wall, not by the contact line, and averaging them in biases the
            angle towards 90 degrees (measured: 11 degrees for every true
            angle before this window was introduced).

    Returns:
        ``{"theta_deg", "theta_spread_deg", "cos_theta", "n_nodes",
        "wall_side", "fill_order", "guardrails"}``.
    """
    wall = _flat_wall(solid_mask)
    guardrails: list[dict[str, Any]] = []
    if not wall["ok"]:
        return {
            "theta_deg": None,
            "theta_spread_deg": None,
            "cos_theta": None,
            "n_nodes": 0,
            "wall_side": wall["side"],
            "fill_order": order,
            "guardrails": [{"code": "unsupported_wall_geometry", "detail": wall.get("detail", "")}],
        }
    filled = _fill_solid(rho, solid_mask, wall["normal"], order)
    gy = 0.5 * (torch.roll(filled, -1, dims=0) - torch.roll(filled, 1, dims=0))
    gx = 0.5 * (torch.roll(filled, -1, dims=1) - torch.roll(filled, 1, dims=1))
    gmag = torch.sqrt(gx * gx + gy * gy)
    ny, nx = rho.shape
    nx_dir, ny_dir = wall["normal"]  # stored as (normal_x, normal_y)
    near = _near_wall_fluid(solid_mask, wall["normal"], layers, first_layer)
    if bool(near.any()):
        vals = rho[near]
        lo, hi = float(vals.min()), float(vals.max())
        u = (rho - lo) / max(hi - lo, 1e-30)
        on_interface = (u > band[0]) & (u < band[1])
        sel = near & on_interface & (gmag > min_grad_frac * float(gmag[near].max()))
    else:
        sel = near
    n_nodes = int(sel.sum())
    if n_nodes < 3:
        guardrails.append(
            {
                "code": "too_few_nodes",
                "detail": f"{n_nodes} near-wall nodes above the gradient threshold",
            }
        )
        return {
            "theta_deg": None,
            "theta_spread_deg": None,
            "cos_theta": None,
            "n_nodes": n_nodes,
            "wall_side": wall["side"],
            "fill_order": order,
            "guardrails": guardrails,
        }
    w = (gmag * gmag)[sel]
    coords = -(
        (gx / torch.clamp(gmag, min=1e-30))[sel] * nx_dir
        + (gy / torch.clamp(gmag, min=1e-30))[sel] * ny_dir
    )
    cos_theta = float((w * coords).sum() / w.sum())
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.degrees(math.acos(cos_theta))
    angles = torch.rad2deg(torch.acos(torch.clamp(coords, -1.0, 1.0)))
    spread = float(angles.max() - angles.min())
    if spread > 20.0:
        guardrails.append(
            {
                "code": "angle_scatter",
                "detail": f"near-wall node angles span {spread:.1f} deg > 20 deg",
            }
        )
    return {
        "theta_deg": theta,
        "theta_spread_deg": spread,
        "cos_theta": cos_theta,
        "n_nodes": n_nodes,
        "wall_side": wall["side"],
        "fill_order": order,
        "guardrails": guardrails,
    }


def circle_fit_contact_angle(
    rho: torch.Tensor,
    solid_mask: torch.Tensor,
    *,
    level: float | None = None,
) -> dict[str, Any]:
    """Contact angle from a circle fit to the measured interface contour.

    Independent geometric route (the cross-check for
    :func:`geometric_contact_angle`): collect the mid-density contour from
    linear interpolation along grid edges, fit a circle algebraically (Kasa
    least squares), and convert the cap geometry to an angle.  With the liquid
    inside the circle, wall at ``y_w`` and the fitted centre a height
    ``h = y_c - y_w`` above it, the tangent at the contact line satisfies
    ``cos(theta) = -h/R`` in the physical convention (``h=0`` -> 90 degrees;
    ``h<0``, centre below the wall, -> wetting).

    For a *non-circular* interface (gravity, walls) the fit residual
    ``rms/R`` grows and is reported; above 5% the angle is flagged instead of
    silently averaged.
    """
    wall = _flat_wall(solid_mask)
    guardrails: list[dict[str, Any]] = []
    if not wall["ok"] or wall["side"] not in {"bottom", "top"}:
        return {
            "theta_deg": None,
            "R": None,
            "centre": None,
            "n_points": 0,
            "guardrails": [
                {"code": "unsupported_wall_geometry", "detail": wall.get("detail", wall["side"])}
            ],
        }
    fluid = ~solid_mask
    vals = rho[fluid]
    if level is None:
        level = 0.5 * (float(vals.max()) + float(vals.min()))
    ny, nx = rho.shape
    pts: list[tuple[float, float]] = []
    for y in range(ny):
        for x in range(nx - 1):
            if not (bool(fluid[y, x]) and bool(fluid[y, x + 1])):
                continue
            a, b = float(rho[y, x]) - level, float(rho[y, x + 1]) - level
            if a * b < 0.0:
                pts.append((x + a / (a - b), float(y)))
    for y in range(ny - 1):
        for x in range(nx):
            if not (bool(fluid[y, x]) and bool(fluid[y + 1, x])):
                continue
            a, b = float(rho[y, x]) - level, float(rho[y + 1, x]) - level
            if a * b < 0.0:
                pts.append((float(x), y + a / (a - b)))
    if len(pts) < 8:
        guardrails.append({"code": "too_few_contour_points", "detail": f"{len(pts)} crossings"})
        return {
            "theta_deg": None,
            "R": None,
            "centre": None,
            "n_points": len(pts),
            "guardrails": guardrails,
        }
    # Kasa algebraic circle fit: x^2 + y^2 + D x + E y + F = 0
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    syy = sum(p[1] * p[1] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    sxz = sum(p[0] * (p[0] ** 2 + p[1] ** 2) for p in pts)
    syz = sum(p[1] * (p[0] ** 2 + p[1] ** 2) for p in pts)
    sz = sum(p[0] ** 2 + p[1] ** 2 for p in pts)
    m = len(pts)
    a11, a12, a13 = sxx, sxy, sx
    a21, a22, a23 = sxy, syy, sy
    a31, a32, a33 = sx, sy, float(m)
    b1, b2, b3 = -sxz, -syz, -sz
    det = (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )
    if abs(det) < 1e-30:
        return {
            "theta_deg": None,
            "R": None,
            "centre": None,
            "n_points": m,
            "guardrails": [{"code": "degenerate_fit", "detail": "singular normal matrix"}],
        }
    d = (
        b1 * (a22 * a33 - a23 * a32) - a12 * (b2 * a33 - a23 * b3) + a13 * (b2 * a32 - a22 * b3)
    ) / det
    e = (
        a11 * (b2 * a33 - a23 * b3) - b1 * (a21 * a33 - a23 * a31) + a13 * (a21 * b3 - b2 * a31)
    ) / det
    f = (
        a11 * (a22 * b3 - b2 * a32) - a12 * (a21 * b3 - b2 * a31) + b1 * (a21 * a32 - a22 * a31)
    ) / det
    xc, yc = -0.5 * d, -0.5 * e
    r2 = xc * xc + yc * yc - f
    if r2 <= 0:
        return {
            "theta_deg": None,
            "R": None,
            "centre": None,
            "n_points": m,
            "guardrails": [{"code": "degenerate_fit", "detail": "negative squared radius"}],
        }
    R = math.sqrt(r2)
    rms = math.sqrt(sum(((p[0] - xc) ** 2 + (p[1] - yc) ** 2 - r2) ** 2 for p in pts) / m)
    wall_y = float(wall["wall_index"]) - 0.5
    h = yc - wall_y
    if abs(h) > R:
        guardrails.append(
            {"code": "centre_outside", "detail": f"|h|={abs(h):.2f} > R={R:.2f}: no contact line"}
        )
        return {
            "theta_deg": None,
            "R": R,
            "centre": (xc, yc),
            "n_points": m,
            "guardrails": guardrails,
        }
    theta = math.degrees(math.atan2(math.sqrt(max(R * R - h * h, 0.0)), -h))
    if rms / R > 0.05:
        guardrails.append(
            {"code": "non_circular", "detail": f"contour rms/R = {rms / R:.3f} > 0.05"}
        )
    return {
        "theta_deg": theta,
        "R": R,
        "centre": (xc, yc),
        "height_above_wall": h,
        "rms_over_R": rms / R,
        "n_points": m,
        "guardrails": guardrails,
    }


def cap_geometry_contact_angle(
    rho: torch.Tensor,
    solid_mask: torch.Tensor,
    *,
    level: float | None = None,
) -> dict[str, Any]:
    """Contact angle from the cap base length and height (openLBMPM recipe).

    This is the third, fully independent estimator, and the only one taken
    line-for-line from verified upstream source (``ShanChen2D/ShanChenD2Q9.py:
    864-888``, ``calMeasuredContactAngle``): count the wetted base length ``b``
    in the first fluid row and the maximum column height ``h`` of the liquid,
    reconstruct the cap radius

        ``R = (4 h^2 + b^2) / (8 h)``

    and evaluate ``theta = atan2(b, 2 (R - h))``.  It needs no gradient, no
    circle fit and no extrapolation, so its failure modes are different from the
    other two - which is exactly why it is worth having as a cross-check on
    (possibly non-circular) solver output.  The upstream threshold 0.485 is a
    hard-coded mid-density level for their ``rho_liq ~ 1`` convention; here the
    level defaults to the mid-density of the fluid field and can be passed in.

    Returns ``{"theta_deg", "base", "height", "R", "level", "guardrails"}``.
    """
    wall = _flat_wall(solid_mask)
    if not wall["ok"] or wall["side"] not in {"bottom", "top"}:
        return {
            "theta_deg": None,
            "base": None,
            "height": None,
            "R": None,
            "level": level,
            "guardrails": [
                {"code": "unsupported_wall_geometry", "detail": wall.get("detail", wall["side"])}
            ],
        }
    fluid = ~solid_mask
    vals = rho[fluid]
    lo, hi = float(vals.min()), float(vals.max())
    if level is None:
        level = 0.5 * (lo + hi)
    above = fluid & (rho >= level)
    if wall["side"] == "bottom":
        first = int(wall["wall_index"])
        base = int(above[first, :].sum())
        heights = above[first:, :].sum(dim=0)
        height = int(heights.max())
    else:
        last = int(wall["wall_index"]) - 1
        base = int(above[last, :].sum())
        heights = above[: last + 1, :].sum(dim=0)
        height = int(heights.max())
    guardrails: list[dict[str, Any]] = []
    if base < 3 or height < 2:
        guardrails.append(
            {
                "code": "cap_too_small",
                "detail": f"base={base} height={height} above level {level:.4f}",
            }
        )
        return {
            "theta_deg": None,
            "base": base,
            "height": height,
            "R": None,
            "level": level,
            "guardrails": guardrails,
        }
    h = float(height)
    b = float(base)
    radius = (4.0 * h * h + b * b) / (8.0 * h)
    theta = math.degrees(math.atan2(b, 2.0 * (radius - h)))
    # For theta > 90 the drop is taller than one radius, so R - h < 0 is the
    # *obtuse branch*, not a defect: atan2 handles it.  Only a vanishing base or
    # a cap larger than the domain is a real failure.
    if radius > 2.0 * rho.shape[0]:
        guardrails.append(
            {"code": "cap_exceeds_domain", "detail": f"reconstructed R={radius:.2f} > 2 n"}
        )
    return {
        "theta_deg": theta,
        "base": base,
        "height": height,
        "R": radius,
        "level": level,
        "guardrails": guardrails,
    }


def synthetic_cap_field(
    *,
    n: int = 96,
    radius: float = 25.0,
    theta_deg: float | None = None,
    height: float | None = None,
    wall_rows: int = 6,
    width: float = 1.5,
    rho_liq: float = 1.0,
    rho_gas: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic droplet cap on a flat wall - a calibration field with known angle.

    The liquid is the part of a disc (centre at ``(n/2, wall + height)``, radius
    ``radius``) inside the fluid region; the density is a tanh profile of the
    signed distance.  The contact angle is prescribed through the **physical**
    convention ``cos(theta) = -height/radius``: the disc centre moves *below*
    the wall for a wetting drop (``theta < 90``, thin cap) and above it for a
    non-wetting one (``theta > 90``, more than a hemisphere, ``height`` up to
    ``+radius`` at 180 degrees).  Pass ``theta_deg`` (preferred) or an explicit
    ``height``.

    Getting this sign wrong is the single easiest way to "calibrate" an angle
    routine to its own inversion - the first version of this helper used
    ``height = +R cos(theta)`` and every measured angle came back as
    ``180 - theta``.  ``test_contact_angle_convention`` pins the convention.
    """
    if theta_deg is not None:
        height = -radius * math.cos(math.radians(theta_deg))
    if height is None:
        height = 0.0
    yy, xx = _grid(n, torch.device("cpu"))
    solid = torch.zeros((n, n), dtype=torch.bool)
    solid[:wall_rows, :] = True
    cx = (n - 1) / 2.0
    cy = (wall_rows - 0.5) + height
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    s = 0.5 * (1.0 - torch.tanh((r - radius) / width))
    rho = rho_gas + s * (rho_liq - rho_gas)
    rho = torch.where(solid, torch.zeros_like(rho), rho)
    return rho, solid
