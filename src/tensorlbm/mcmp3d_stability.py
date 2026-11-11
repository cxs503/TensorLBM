"""3D MCMP (D3Q19 Shan-Chen two-component) stability window and loss-of-stability map.

Why this module exists
----------------------
The two-component Shan-Chen path of this library had no 3D stability tooling:
``collide_sc_two_component_3d`` could be called, but nothing answered "for this
(with/without self-coupling, pseudopotential, τ) combination, at which coupling
does the run lose stability, is that boundary reproducible, and what density
ratio does the stable window actually deliver".  This module supplies exactly
that as plain functions returning JSON-serialisable records (numbers are meant
to be piped straight into an artifact, never retyped), plus a CLI that runs the
whole acceptance protocol.

Protocol (predeclared; do not tune after inspecting results)
------------------------------------------------------------
Case
    Periodic 3D box, planar slab of component 1 in a matrix of component 2
    (two diffuse interfaces, no walls, no body force).  Initial profiles are
    ``tanh`` of width ``interface_width``.  Optional seed-controlled noise on
    the initial densities differentiates independently seeded repetitions.

Classification (fixed before the runs)
    The stability diagnostic is the **mixture (barycentric) velocity**
    ``u_mix = (rho1 u1 + rho2 u2)/(rho1 + rho2)``; the per-component maxima are
    recorded alongside but not used for the verdict (a dilute component carries
    a large ``F/rho`` there, which is not a stability statement).

    A case is **diverged** if at any sample any of
      * a moment is not finite,
      * ``min(rho) <= 0`` (negative density is unphysical for SC),
      * ``max(rho) > rho_blow`` (default 10, i.e. 10x the densest bulk),
      * ``u_max_mix > u_burst`` (default 0.30, Ma ~ 0.52 -- compressibility
        breakdown) after the declared ``startup_steps`` (default 100) window.
    Among non-diverged cases, a case is **unstable** (still usable as evidence
    of the boundary, but not a validated stable point) if the last-window
    ``u_max`` exceeds ``u_quiescent`` (default 0.10) or the ``u_max`` series
    grows over the last three samples.  A case that completes ``steps`` without
    any of the above is **stable**.

Boundary
    Reported as the interval ``(G_last_stable, G_first_unstable)`` on the
    scanned coupling axis, with the boundary estimate at the midpoint.  The
    acceptance criterion is *reproducibility*: two independently seeded scans
    must give the same interval, and the boundary estimate must agree to
    ``<= 1e-9`` (deterministic protocol) or to within one grid step (noise).

Density ratio
    ``Delta_rho`` of a run is the ratio of the *mixture* bulk densities in the
    component-1-rich region and the component-2-rich region; the per-component
    ratios are reported next to it.  For linear pseudopotentials ψ=ρ the
    measured ratio is structurally capped near 4 (EOS ``p = cs²ρ + (G/2)ρ²``),
    which is why :func:`~tensorlbm.multiphase.make_psi_carnahan_starling` /
    :func:`~tensorlbm.multiphase.psi_exp` are accepted by the force kernel.

Everything here is a *measurement* tool: it does not modify the solver.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

import torch

from .d3q19 import equilibrium3d, macroscopic3d
from .multiphase import psi_exp, psi_linear
from .multiphase3d import collide_sc_two_component_3d
from .solver3d import stream3d_cat

__all__ = [
    "init_mcmp_slab",
    "run_case",
    "classify",
    "scan",
    "stability_boundary",
    "DEFAULT_CRITERIA",
    "PROTOCOL_BOX",
]

# Predeclared thresholds (see module docstring).
DEFAULT_CRITERIA = {
    "u_burst": 0.30,
    "u_quiescent": 0.10,
    "rho_blow": 10.0,
    # The first ``startup_steps`` steps are exempt from the velocity test: a
    # tanh interface wider than the equilibrium one necessarily relaxes with a
    # finite acoustic transient, and finiteness/positivity are still enforced
    # there.  This window is fixed in the protocol, not chosen per case.
    "startup_steps": 100,
}

#: Default box used by the acceptance protocol: (nz, ny, nx).
PROTOCOL_BOX = (48, 12, 12)

_PSI_REGISTRY: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "linear": psi_linear,
    "exp": psi_exp,
}


def register_psi(name: str, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
    """Register a pseudopotential under ``name`` so records stay JSON-safe."""
    _PSI_REGISTRY[name] = fn


def get_psi(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resolve a pseudopotential name registered with :func:`register_psi`."""
    try:
        return _PSI_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown pseudopotential {name!r}; known: {sorted(_PSI_REGISTRY)}") from exc


def slab_profile(nz: int, half_fraction: float = 0.25, interface_width: float = 2.0) -> torch.Tensor:
    """Periodic 0/1 slab profile of shape ``(nz, 1, 1)`` (1 inside the slab)."""
    z = torch.arange(nz, dtype=torch.float32)
    a = (0.5 - half_fraction) * nz
    b = (0.5 + half_fraction) * nz
    p = 0.5 * (1.0 + torch.tanh((z - a) / interface_width)) - 0.5 * (
        1.0 + torch.tanh((z - b) / interface_width)
    )
    return p.view(nz, 1, 1)


def init_mcmp_slab(
    nz: int,
    ny: int,
    nx: int,
    *,
    rho1_in: float,
    rho1_out: float,
    rho2_in: float,
    rho2_out: float,
    interface_width: float = 2.0,
    half_fraction: float = 0.25,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equilibrium distributions of a two-component slab (periodic in all axes).

    ``rho{1,2}_{in,out}`` are the component densities inside/outside the slab.
    ``noise`` is a *relative* uniform perturbation applied to both components
    (conservation is restored afterwards by removing the cell-mean of the
    perturbation), so an independently seeded repetition starts from the same
    mean state with a different microscopic seed.
    """
    p = slab_profile(nz, half_fraction, interface_width).expand(nz, ny, nx).contiguous()
    rho1 = rho1_in * p + rho1_out * (1.0 - p)
    rho2 = rho2_in * p + rho2_out * (1.0 - p)
    if noise:
        g = torch.Generator().manual_seed(int(seed))
        for rho in (rho1, rho2):
            d = (torch.rand(rho.shape, generator=g) - 0.5) * 2.0 * noise
            d = d - d.mean()
            rho.mul_(1.0 + d)
    zero = torch.zeros_like(rho1)
    return equilibrium3d(rho1, zero, zero, zero), equilibrium3d(rho2, zero, zero, zero)


def _bulk_means(rho: torch.Tensor, nz: int, band: float = 0.125) -> tuple[float, float]:
    """Bulk means of the dense (slab-centred) and dilute (edge) bands."""
    h = max(1, int(round(nz * band)))
    lo = max(0, nz // 2 - h // 2)
    dense = rho[lo : lo + h].mean()
    dilute = rho[1 : 1 + h].mean()
    return float(dense), float(dilute)


def run_case(spec: dict[str, Any]) -> dict[str, Any]:
    """Run one MCMP stability case and return a JSON-serialisable record.

    ``spec`` keys (all optional unless marked):
        name, G_12 (required), G_self (scalar or [G11, G22]), psi1, psi2,
        tau1, tau2, steps, nz, ny, nx, guo, u_eq, noise, seed,
        rho1_in, rho1_out, rho2_in, rho2_out, interface_width,
        half_fraction, sample_every, criteria, series (bool).
    """
    crit = dict(DEFAULT_CRITERIA)
    crit.update(spec.get("criteria") or {})
    nz, ny, nx = int(spec.get("nz", PROTOCOL_BOX[0])), int(spec.get("ny", PROTOCOL_BOX[1])), int(
        spec.get("nx", PROTOCOL_BOX[2])
    )
    steps = int(spec.get("steps", 2000))
    sample_every = int(spec.get("sample_every", 50))
    psi1_name, psi2_name = str(spec.get("psi1", "linear")), str(spec.get("psi2", "linear"))
    psi1_fn, psi2_fn = get_psi(psi1_name), get_psi(psi2_name)
    rho1_in = float(spec.get("rho1_in", 0.92))
    rho1_out = float(spec.get("rho1_out", 0.14))
    rho2_in = float(spec.get("rho2_in", 0.08))
    rho2_out = float(spec.get("rho2_out", 0.86))

    f1, f2 = init_mcmp_slab(
        nz,
        ny,
        nx,
        rho1_in=rho1_in,
        rho1_out=rho1_out,
        rho2_in=rho2_in,
        rho2_out=rho2_out,
        interface_width=float(spec.get("interface_width", 2.0)),
        half_fraction=float(spec.get("half_fraction", 0.25)),
        noise=float(spec.get("noise", 0.0)),
        seed=int(spec.get("seed", 0)),
    )

    G_12 = float(spec["G_12"])
    G_self = spec.get("G_self", 0.0)
    tau1, tau2 = float(spec.get("tau1", 1.0)), float(spec.get("tau2", 1.0))
    guo = bool(spec.get("guo", True))
    u_eq = str(spec.get("u_eq", "self"))

    series: list[list[float]] = []
    diverged, div_step, div_reason = False, None, None
    u_max = 0.0
    u_max_comp = 0.0
    rho_min, rho_max = math.inf, -math.inf

    for step in range(1, steps + 1):
        f1, f2 = collide_sc_two_component_3d(
            f1,
            f2,
            G_12=G_12,
            tau1=tau1,
            tau2=tau2,
            use_guo=guo,
            u_eq=u_eq,
            G_self=G_self,
            psi1_fn=psi1_fn,
            psi2_fn=psi2_fn,
        )
        f1 = stream3d_cat(f1)
        f2 = stream3d_cat(f2)
        if step % sample_every == 0 or step == 1:
            r1, ux1, uy1, uz1 = macroscopic3d(f1)
            r2, ux2, uy2, uz2 = macroscopic3d(f2)
            rtot = torch.clamp(r1 + r2, min=1e-12)
            ux = (r1 * ux1 + r2 * ux2) / rtot
            uy = (r1 * uy1 + r2 * uy2) / rtot
            uz = (r1 * uz1 + r2 * uz2) / rtot
            u_mix = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max())
            u_comp = max(
                float(torch.sqrt(ux1 * ux1 + uy1 * uy1 + uz1 * uz1).max()),
                float(torch.sqrt(ux2 * ux2 + uy2 * uy2 + uz2 * uz2).max()),
            )
            rmin = float(torch.minimum(r1.min(), r2.min()))
            rmax = float(torch.maximum(r1.max(), r2.max()))
            if not (math.isfinite(u_mix) and math.isfinite(u_comp) and math.isfinite(rmin)
                    and math.isfinite(rmax)):
                diverged, div_step, div_reason = True, step, "non-finite moment"
            elif rmin <= 0.0:
                diverged, div_step, div_reason = True, step, "non-positive density"
            elif rmax > float(crit["rho_blow"]):
                diverged, div_step, div_reason = True, step, "density blow-up"
            elif step > int(crit["startup_steps"]) and u_mix > float(crit["u_burst"]):
                diverged, div_step, div_reason = True, step, "velocity burst"
            u_max = max(u_max, u_mix)
            u_max_comp = max(u_max_comp, u_comp)
            rho_min, rho_max = min(rho_min, rmin), max(rho_max, rmax)
            if spec.get("series", True):
                series.append([step, round(u_mix, 8), round(u_comp, 8), rmin, rmax])
            if diverged:
                break

    rec: dict[str, Any] = {
        "name": spec.get("name", f"G12={G_12}"),
        "G_12": G_12,
        "G_self": list(G_self) if isinstance(G_self, (tuple, list)) else G_self,
        "psi1": psi1_name,
        "psi2": psi2_name,
        "tau1": tau1,
        "tau2": tau2,
        "guo": guo,
        "u_eq": u_eq,
        "box": [nz, ny, nx],
        "steps_requested": steps,
        "steps_run": (div_step or steps),
        "sample_every": sample_every,
        "diverged": diverged,
        "diverged_at": div_step,
        "diverged_reason": div_reason,
        "u_max": None if diverged else round(u_max, 6),
        "u_max_peak": None if not math.isfinite(u_max) else round(u_max, 6),
        "u_max_peak_component": None if not math.isfinite(u_max_comp) else round(u_max_comp, 6),
        "rho_min": None if not math.isfinite(rho_min) else round(rho_min, 8),
        "rho_max": None if not math.isfinite(rho_max) else round(rho_max, 6),
        "criteria": crit,
        "init": {
            "rho1_in": rho1_in,
            "rho1_out": rho1_out,
            "rho2_in": rho2_in,
            "rho2_out": rho2_out,
            "interface_width": float(spec.get("interface_width", 2.0)),
            "half_fraction": float(spec.get("half_fraction", 0.25)),
            "noise": float(spec.get("noise", 0.0)),
            "seed": int(spec.get("seed", 0)),
        },
        "series": series,
    }

    if not diverged:
        r1, ux1, uy1, uz1 = macroscopic3d(f1)
        r2, ux2, uy2, uz2 = macroscopic3d(f2)
        rtot = torch.clamp(r1 + r2, min=1e-12)
        ux = (r1 * ux1 + r2 * ux2) / rtot
        uy = (r1 * uy1 + r2 * uy2) / rtot
        uz = (r1 * uz1 + r2 * uz2) / rtot
        d1, g1 = _bulk_means(r1, nz)
        d2, g2 = _bulk_means(r2, nz)
        rec["bulk"] = {
            "rho1_dense": round(d1, 6),
            "rho1_dilute": round(g1, 8),
            "rho2_dense": round(d2, 6),
            "rho2_dilute": round(g2, 6),
            "rho_mix_dense": round(d1 + d2, 6),
            "rho_mix_dilute": round(g1 + g2, 8),
            # mixture ratio is the "Delta_rho" of the case
            "delta_rho": round((d1 + d2) / max(g1 + g2, 1e-12), 3),
            "delta_rho_component1": round(d1 / max(g1, 1e-12), 3),
            "delta_rho_component2": round(g2 / max(d2, 1e-12), 3),
        }
        rec["u_max_final"] = round(float(torch.sqrt(ux * ux + uy * uy + uz * uz).max()), 6)
    rec["classification"] = classify(rec, crit)
    return rec


def classify(rec: dict[str, Any], crit: dict[str, float] | None = None) -> dict[str, Any]:
    """Classify a record: ``diverged`` / ``unstable`` / ``stable`` (predeclared)."""
    crit = dict(DEFAULT_CRITERIA if crit is None else crit)
    if rec.get("diverged"):
        return {"state": "diverged", "reason": rec.get("diverged_reason") or "diverged"}
    series = rec.get("series") or []
    u_final = rec.get("u_max_final")
    if u_final is None:
        return {"state": "diverged", "reason": "no final moment"}
    if u_final > float(crit["u_quiescent"]):
        return {"state": "unstable", "reason": f"u_max_final {u_final:.3g} > u_quiescent"}
    if len(series) >= 3:
        tail = [s[1] for s in series[-3:]]
        if tail[-1] > tail[0] * 1.5 and tail[-1] > 1e-3:
            return {"state": "unstable", "reason": "u_max grows over the last three samples"}
    return {"state": "stable", "reason": "completed, quiescent"}


def scan(specs: Sequence[dict[str, Any]], workers: int | None = None) -> list[dict[str, Any]]:
    """Run a list of case specs, optionally in parallel processes.

    ``workers > 1`` uses :mod:`multiprocessing`; every worker is limited to one
    torch thread so the scan does not oversubscribe the machine.
    """
    if workers is None or workers <= 1:
        return [run_case(s) for s in specs]

    import multiprocessing as mp

    with mp.get_context("fork").Pool(processes=int(workers)) as pool:
        return pool.map(_scan_worker, list(specs))


def _scan_worker(spec: dict[str, Any]) -> dict[str, Any]:
    """Module-level worker (picklable under the fork/spawn start methods)."""
    torch.set_num_threads(1)
    return run_case(spec)


def stability_boundary(records: Iterable[dict[str, Any]], axis: str = "G_12") -> dict[str, Any]:
    """Locate the loss-of-stability interval along ``axis`` (monotone |G| scan).

    Cases are ordered by ``|axis value|`` because phase separation strengthens
    (and the numerical margin shrinks) with the magnitude of the coupling.
    """
    rs = sorted(records, key=lambda r: abs(float(r[axis])))
    last_stable, first_unstable = None, None
    for r in rs:
        state = (r.get("classification") or classify(r))["state"]
        if state == "stable":
            if first_unstable is None:
                last_stable = float(r[axis])
        elif first_unstable is None:
            first_unstable = float(r[axis])
            break
    boundary = None
    if last_stable is not None and first_unstable is not None:
        boundary = 0.5 * (last_stable + first_unstable)
    elif last_stable is not None:
        boundary = last_stable  # window open beyond the scanned range
    return {
        "axis": axis,
        "last_stable": last_stable,
        "first_unstable": first_unstable,
        "boundary_estimate": boundary,
        "boundary_kind": "interval" if (last_stable is not None and first_unstable is not None) else "open",
        "states": {r.get("name", str(r[axis])): (r.get("classification") or classify(r))["state"] for r in rs},
    }