"""Model registry: every collision model as one CI matrix key.

Why this module exists
----------------------
TensorLBM ships ~25 collision operators scattered over ``solver3d``,
``turbulence``, ``advanced_collision``, ``cumulant``, ... .  Nothing in the
repository enumerates *which models exist*, *what each one can run* and
*which verification case applies to it*.  Consequence: a change to one
operator can silently leave another one untested, and there is no way to
answer "what fraction of our models is machine-verified?".

This registry is the analogue of the upstream ``models/<class>/<lattice>_
<method>/`` convention (CFD-GO/TCLB, tag ``model-dir-as-build-and-ci-unit``):
the model name is *the* matrix key used by the build, the CI driver and the
coverage report.  Unlike upstream it does not need a directory per model
because a model here is a Python callable plus a declarative spec.

Registry entries are *adapters*, not reimplementations: every ``collide``
field points at a real function of this repository and ``source`` records
where it lives (machine-readable provenance).  The registry never contains
numerics of its own.

Design decisions (transfer from the OSS practice)
-------------------------------------------------
* ``ModelSpec.cases`` is the model's declared verification surface, so the
  coverage driver can distinguish *declared but unimplemented* (MISSING) from
  *declared inapplicable* (NOT_APPLICABLE, with ``inapplicable_reason``).
* ``ModelSpec.enforcement`` mirrors the upstream split between the CPU
  pipeline (configure -> compile -> tests -> gcov) and the GPU/HIP pipeline
  (compile guard only): ``"gate"`` models must pass their cases, ``"guard"``
  models only have to be constructible and to run one step without NaN.
* LES variants of a model are *not* eligible for the molecular-viscosity
  gate: the sub-grid viscosity shifts the measured decay rate by design.
  They are registered with ``C_s = 0`` (LES switched off, reduces to the base
  operator) and the LES-enabled operator is registered separately as a
  ``guard`` model -- the same way upstream gives GPU-only models a weaker
  guarantee instead of pretending they are covered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

# --------------------------------------------------------------------------- #
#  Lattice adapters (equilibrium / macroscopic / streaming per lattice)
# --------------------------------------------------------------------------- #


def lattice_ops(lattice: str) -> dict[str, Callable[..., Any]]:
    """Return the repo-native ``equilibrium``/``macroscopic``/``stream`` trio.

    Only lattices that already have a streaming operator in this repository
    are listed.  ``stream`` is the periodic roll/gather implementation used by
    the production stepper -- the harness must not carry its own streaming
    arithmetic.
    """
    key = lattice.upper()
    if key == "D3Q19":
        from ..d3q19 import equilibrium3d, macroscopic3d  # noqa: PLC0415
        from ..solver3d import stream3d  # noqa: PLC0415

        return {"equilibrium": equilibrium3d, "macroscopic": macroscopic3d, "stream": stream3d}
    if key == "D3Q27":
        from ..d3q27 import equilibrium27, macroscopic27, stream27  # noqa: PLC0415

        return {"equilibrium": equilibrium27, "macroscopic": macroscopic27, "stream": stream27}
    raise ValueError(f"lattice {lattice!r} has no streaming operator in this repository")


# --------------------------------------------------------------------------- #
#  Spec
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    """One collision model == one CI matrix key."""

    name: str
    lattice: str
    collide: Callable[..., torch.Tensor]
    source: str
    """``module:qualname`` of the production implementation (provenance)."""
    tau_arg: str = "tau"
    """Name the operator expects for the shear relaxation time."""
    params: dict[str, Any] = field(default_factory=dict)
    """Fixed operator parameters (e.g. LES constant) recorded in every report."""
    family: str = ""
    """Grouping used by the report (kinetic / multi-relaxation / ...)."""
    cases: tuple[str, ...] = ()
    """Declared verification surface (case names from :mod:`.cases`)."""
    enforcement: str = "gate"
    """``gate`` = must pass its declared cases; ``guard`` = constructible only."""
    inapplicable_reason: dict[str, str] = field(default_factory=dict)
    """case -> reason it is deliberately not applicable to this model."""
    les: bool = False

    def identity(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "lattice": self.lattice,
            "source": self.source,
            "tau_arg": self.tau_arg,
            "params": dict(self.params),
            "family": self.family,
            "enforcement": self.enforcement,
            "les": self.les,
        }


def _specs() -> list[ModelSpec]:
    from .. import advanced_collision as adv  # noqa: PLC0415
    from .. import solver3d as s3  # noqa: PLC0415
    from ..turbulence import turbulence as turb  # noqa: PLC0415

    core_cases = ("feq_fixed_point", "shear_decay", "tg_decay")
    return [
        ModelSpec(
            name="d3q19/bgk",
            lattice="D3Q19",
            collide=s3.collide_bgk3d,
            source="tensorlbm.solver3d:collide_bgk3d",
            family="single-relaxation-time",
            cases=core_cases,
        ),
        ModelSpec(
            name="d3q19/trt",
            lattice="D3Q19",
            collide=s3.collide_trt3d,
            source="tensorlbm.solver3d:collide_trt3d",
            tau_arg="tau_plus",
            params={"lambda_trt": 3.0 / 16.0},
            family="two-relaxation-time",
            cases=core_cases,
        ),
        ModelSpec(
            name="d3q19/rlbm",
            lattice="D3Q19",
            collide=s3.collide_rlbm3d,
            source="tensorlbm.solver3d:collide_rlbm3d",
            family="regularized",
            cases=core_cases,
        ),
        ModelSpec(
            name="d3q19/mrt",
            lattice="D3Q19",
            collide=s3.collide_mrt3d,
            source="tensorlbm.solver3d:collide_mrt3d",
            params={"s_e": 1.19, "s_eps": 1.4, "s_q": 1.2},
            family="multi-relaxation-time",
            cases=core_cases,
        ),
        ModelSpec(
            name="d3q19/mrt-smag",
            lattice="D3Q19",
            collide=turb.collide_smagorinsky_mrt3d,
            source="tensorlbm.turbulence.turbulence:collide_smagorinsky_mrt3d",
            params={"C_s": 0.1},
            family="mrt+les",
            les=True,
            # SGS viscosity shifts the effective shear relaxation time, so the
            # molecular-viscosity cases have no valid reference for this model.
            cases=("feq_fixed_point",),
            inapplicable_reason={
                "shear_decay": "LES on: SGS viscosity is a function of the local strain "
                "rate, so nu=(tau-1/2)/3 is not the operator's effective viscosity",
                "tg_decay": "same as shear_decay (LES on)",
            },
            enforcement="guard",
        ),
        ModelSpec(
            name="d3q27/kbc",
            lattice="D3Q27",
            collide=adv.collide_kbc_d3q27,
            source="tensorlbm.advanced_collision:collide_kbc_d3q27",
            params={"C_s": 0.0, "beta": 0.99},
            family="entropic-kbc",
            cases=core_cases,
        ),
        ModelSpec(
            name="d3q27/kbc-les",
            lattice="D3Q27",
            collide=adv.collide_kbc_d3q27,
            source="tensorlbm.advanced_collision:collide_kbc_d3q27",
            params={"C_s": 0.1, "beta": 0.99},
            family="entropic-kbc+les",
            les=True,
            cases=("feq_fixed_point",),
            inapplicable_reason={
                "shear_decay": "LES on: SGS viscosity replaces the molecular reference",
                "tg_decay": "same as shear_decay (LES on)",
            },
            enforcement="guard",
        ),
        ModelSpec(
            name="d3q27/cascaded",
            lattice="D3Q27",
            collide=adv.collide_cascaded_d3q27,
            source="tensorlbm.advanced_collision:collide_cascaded_d3q27",
            params={"C_s": 0.0},
            family="cascaded-central-moment",
            cases=core_cases,
        ),
        ModelSpec(
            name="d3q27/cascaded-les",
            lattice="D3Q27",
            collide=adv.collide_cascaded_d3q27,
            source="tensorlbm.advanced_collision:collide_cascaded_d3q27",
            params={"C_s": 0.1},
            family="cascaded+les",
            les=True,
            cases=("feq_fixed_point",),
            inapplicable_reason={
                "shear_decay": "LES on: SGS viscosity replaces the molecular reference",
                "tg_decay": "same as shear_decay (LES on)",
            },
            enforcement="guard",
        ),
    ]


_CACHE: dict[str, ModelSpec] | None = None


def registry() -> dict[str, ModelSpec]:
    """Return ``{model_name: ModelSpec}`` (built once, then cached)."""
    global _CACHE
    if _CACHE is None:
        specs = _specs()
        names = [s.name for s in specs]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise RuntimeError(f"duplicate model matrix keys: {sorted(dupes)}")
        _CACHE = {s.name: s for s in specs}
    return _CACHE


def model_names() -> list[str]:
    return sorted(registry())


def get_model(name: str) -> ModelSpec:
    try:
        return registry()[name]
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}; known keys: {', '.join(model_names())}"
        ) from None


# --------------------------------------------------------------------------- #
#  Collider construction (the executable form of a registry entry)
# --------------------------------------------------------------------------- #


def build_collider(name: str, tau: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``f -> f_post_collision`` for ``name`` at relaxation time ``tau``.

    This is the single place where a matrix key becomes a callable, so the CI
    driver, the tests and any solver entry point that accepts a model *name*
    all exercise exactly the same binding.
    """
    spec = get_model(name)
    if not tau > 0.0:
        raise ValueError(f"{name}: tau must be positive, got {tau}")
    kwargs = dict(spec.params)
    collide = spec.collide
    tau_arg = spec.tau_arg

    def collider(f: torch.Tensor) -> torch.Tensor:
        return collide(f, **{tau_arg: tau}, **kwargs)

    collider.model = spec.name  # type: ignore[attr-defined]
    collider.tau = tau  # type: ignore[attr-defined]
    collider.source = spec.source  # type: ignore[attr-defined]
    return collider


def nu_from_tau(tau: float) -> float:
    """BGK/TRT/MRT/regularized/KBC shear viscosity in lattice units."""
    return (tau - 0.5) / 3.0


def resolve_collide_fn(spec_or_name: "ModelSpec | str", tau: float, **kwargs: Any):
    """Solver-facing resolver: model key + tau -> ``collide_fn(f, tau=...)``.

    The production stepper (:func:`tensorlbm.lbm_step_correct.lbm_step_correct`)
    calls its collision operator as ``collide_fn(f, tau=tau, **kwargs)``.  The
    returned closure keeps that exact call signature, so a matrix key is a
    drop-in replacement for a raw collision function, and the registry becomes
    the single name -> operator binding used by the solver path.
    """
    spec = get_model(spec_or_name) if isinstance(spec_or_name, str) else spec_or_name
    bound_tau = float(tau)
    tau_arg = spec.tau_arg
    params = dict(spec.params)
    params.update({k: v for k, v in kwargs.items() if k in spec.params})
    collide = spec.collide

    def collide_fn(f: torch.Tensor, tau: float | None = bound_tau, **_ignored: Any) -> torch.Tensor:
        t = bound_tau if tau is None else float(tau)
        if not t > 0.0:
            raise ValueError(f"{spec.name}: tau must be positive, got {t}")
        return collide(f, **{tau_arg: t}, **params)

    collide_fn.model = spec.name  # type: ignore[attr-defined]
    collide_fn.params = params  # type: ignore[attr-defined]
    collide_fn.tau = bound_tau  # type: ignore[attr-defined]
    return collide_fn
