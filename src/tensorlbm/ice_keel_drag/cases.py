"""Case tables for the IN101-N01 under-ice drag campaign.

Two families, both on the periodic force-driven channel of
:mod:`tensorlbm.underice_drag`:

``smooth_ladder``
    Flat ice underside on three successively refined grids with a fixed
    refinement ratio.  Only used for the discretisation-error estimate (GCI)
    and for the keel-off negative control.

``keel_sweep``
    Triangular keel arrays at fixed ``h_eff``, sweeping the two morphology
    parameters that the analytic frontal-area argument contains:

    * ``lam = w / p``  -- areal keel density (base width over ridge pitch),
    * ``h_keel``       -- keel depth in lattice cells, reported also as the
                          morphological ratio ``h_keel / h_i`` through the
                          ``LEVEL_ICE_TO_HALF_CLEARANCE`` anchor of the kernel.

The keel cases are run at fixed ``h_eff`` on purpose: the object of the test
is the *scaling* of ``C_d^form`` with the frontal area, and refinement of the
geometry would change two things at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..underice_drag import (LEVEL_ICE_TO_HALF_CLEARANCE, KeelArraySpec,
                             UnderIceChannelConfig)


@dataclass(frozen=True)
class CaseSpec:
    """One campaign case: identifier, channel config and keel geometry."""

    case_id: str
    family: str
    cfg: UnderIceChannelConfig
    keel: KeelArraySpec | None = None
    tags: dict[str, Any] = field(default_factory=dict)


#: refinement ratio of the smooth ladder (all three directions)
GCI_RATIO_NOTE = "grids refined by ~4/3 in every direction"


def _cfg(nx: int, ny: int, nz: int, *, steps: int, warmup: int,
         u_tau: float = 0.002, re_tau: float = 180.0,
         init_profile: str = "parabolic", noise: float = 0.0,
         threads: int = 64, note: str = "") -> UnderIceChannelConfig:
    return UnderIceChannelConfig(
        nx=nx, ny=ny, nz=nz, re_tau=re_tau, u_tau=u_tau, steps=steps,
        warmup=warmup, sample_interval=20, threads=threads, seed=20261109,
        noise_amplitude=noise, init_profile=init_profile, note=note)


def smooth_ladder(*, steps: int = 4000, warmup: int = 1000,
                  levels: tuple[tuple[int, int, int], ...] = ((24, 24, 50), (32, 32, 66),
                                                              (48, 48, 98)),
                  init_profile: str = "parabolic", noise: float = 0.0,
                  threads: int = 64) -> list[CaseSpec]:
    """Flat-underside channel on a refinement ladder (GCI + negative control)."""
    out: list[CaseSpec] = []
    for i, (nx, ny, nz) in enumerate(levels):
        out.append(CaseSpec(
            case_id=f"smooth_L{i + 1}_{nx}x{ny}x{nz}",
            family="smooth",
            cfg=_cfg(nx, ny, nz, steps=steps, warmup=warmup,
                     init_profile=init_profile, noise=noise, threads=threads,
                     note=f"smooth ladder level {i + 1}"),
            tags={"level": i + 1, "h_keel": 0.0, "lam": 0.0,
                  "h_keel_over_hi": 0.0}))
    return out


def keel_sweep(*, nx: int = 32, ny: int = 32, nz: int = 66,
               steps: int = 8000, warmup: int = 2000,
               lams: tuple[float, ...] = (0.4, 0.6, 0.8),
               h_keels: tuple[int, ...] = (2, 3, 4, 6),
               theta_deg: float = 90.0, n_ridges: int = 2,
               height_modulation: float = 0.0, re_tau: float = 20.0,
               threads: int = 64) -> list[CaseSpec]:
    """Triangular keel array sweep over ``(lambda, h_keel)``.

    ``n_ridges = 2`` over ``ny = 32`` gives a ridge pitch of 16 cells, so the
    ridge *base* is 6-13 cells wide and the triangular depth profile
    ``d = h (1 - 2|t|/w)`` is resolved by several voxels.  A denser ridge set
    (``n_ridges = 4``) collapses the base to 1-2 cells, at which point the
    voxelised triangle degenerates and the sweep stops being a sweep -- measured
    on the first attempt, where ``h_keel = 3, 5, 7`` produced *identical* masks
    and identical drag (a finding kept in the report).

    ``re_tau = 20`` is the *stability* constraint of the kernel, not a physical
    choice: with ``s- = s+ = 1/tau`` the collision is BGK and the relaxation
    rate sits at ``1/tau = 1.996`` for ``Re_tau = 180`` (``tau = 0.5011``), where
    any rough-wall perturbation is amplified without bound (measured: every keel
    case returned NaN at ``Re_tau = 180``).  At ``Re_tau = 20`` the viscosity is
    10x larger (``tau = 0.5096``) and the same geometry is stable.  The
    consequence -- the swept ``C_d`` lives far below the turbulent observational
    band -- is reported, not hidden.

    Straight ridges (``height_modulation = 0``) so that the only thing that
    changes along the ``h_keel`` sweep is the ridge depth, and the only thing
    that changes along the ``lambda`` sweep is the ridge width -- which is what
    the analytic frontal-area argument ``S_fr/S_plan = 0.5 lambda h_keel`` is
    written in terms of.
    """
    out: list[CaseSpec] = []
    for lam in lams:
        for h_keel in h_keels:
            spec = KeelArraySpec(nx=nx, ny=ny, nz=nz, h_keel=float(h_keel),
                                 coverage=float(lam), angle_deg=theta_deg,
                                 height_modulation=height_modulation,
                                 n_ridges=n_ridges)
            cid = f"keel_lam{int(round(lam * 100)):03d}_h{h_keel:02d}"
            out.append(CaseSpec(
                case_id=cid,
                family="keel",
                cfg=_cfg(nx, ny, nz, steps=steps, warmup=warmup, re_tau=re_tau,
                         threads=threads, note=cid),
                keel=spec,
                tags={"level": 0, "lam": float(lam), "h_keel": float(h_keel),
                      "h_keel_over_hi": h_keel / LEVEL_ICE_TO_HALF_CLEARANCE}))
    return out


def smooth_reference(*, nx: int = 32, ny: int = 32, nz: int = 66,
                     steps: int = 12000, warmup: int = 4000, re_tau: float = 20.0,
                     threads: int = 64) -> CaseSpec:
    """Flat underside on the keel grid *and* at the keel operating point."""
    return CaseSpec(
        case_id=f"smooth_ref_re{re_tau:.0f}_{nx}x{ny}x{nz}",
        family="smooth_ref",
        cfg=_cfg(nx, ny, nz, steps=steps, warmup=warmup, re_tau=re_tau,
                 threads=threads, note="keel-off negative control"),
        tags={"level": 0, "h_keel": 0.0, "lam": 0.0, "h_keel_over_hi": 0.0})


def turbulent_attempt(*, nx: int = 32, ny: int = 32, nz: int = 66,
                      steps: int = 60000, warmup: int = 20000,
                      threads: int = 64) -> CaseSpec:
    """Flat wall seeded *below* the turbulent attractor with 3 % noise.

    This is the case that shows whether the kernel can reach the turbulent
    branch at all (see the ``a_turbulent`` verdict in :mod:`criteria`).
    """
    return CaseSpec(
        case_id=f"smooth_turbulent_attempt_{nx}x{ny}x{nz}",
        family="smooth_transient",
        cfg=_cfg(nx, ny, nz, steps=steps, warmup=warmup, init_profile="log",
                 noise=0.03, threads=threads,
                 note="log-law seed + noise, turbulent branch attempt"),
        tags={"level": 0, "h_keel": 0.0, "lam": 0.0, "h_keel_over_hi": 0.0})