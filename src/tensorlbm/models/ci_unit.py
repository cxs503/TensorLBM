"""Model directory as build-and-CI unit (``XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT``).

Cross-validated against the CFD-GO/TCLB convention that **a model is a
directory, and that directory is the unit of both the build and the CI run**:
``tclb/models/<name>/{conf.xml,Dynamics.R,...}`` is generated/compiled on its
own, and the CI job for the model is bound to that directory.  The property
that convention buys is a hard one:

* the set of buildable models is *discovered from the filesystem*, never from
  a hand-maintained list, so adding a directory is the only way to add a model
  and deleting a directory removes it from CI;
* a directory that cannot be compiled/instantiated fails **on its own**, so
  one broken model cannot mask another;
* every unit carries its own parameter set and its own acceptance gate, so the
  CI verdict is per-model and machine-checkable;
* the build produces an identity (here a determinism fingerprint) that pins
  the artefact a CI verdict was measured on.

TensorLBM has no per-model code generation: kernels are PyTorch functions
composed at runtime, so the analogue of TCLB's "compile the model directory"
is *manifest validation + kernel instantiation + a bit-identical determinism
fingerprint*.  The analogue of TCLB's per-model regression run is a benchmark
with an independently-derived reference value and a declared tolerance.

Layout::

    models/units/<unit_id>/model.json     # one unit per directory
    models/units/<unit_id>/README.md      # optional, human notes

Manifest schema ``tensorlbm.model_unit/v1`` (strict: unknown keys are
rejected, so a typo can never silently degrade into "no gate").

Entry point::

    python -m tensorlbm.models.ci_unit list --root models/units
    python -m tensorlbm.models.ci_unit all  --root models/units --out artifacts/model_units
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from ..d2q9 import OPPOSITE, C
from ..d2q9 import equilibrium as _equilibrium
from ..d2q9 import macroscopic as _macroscopic
from ..solver import collide_bgk, collide_mrt, collide_trt, stream
from .contracts import ModelComposition

SCHEMA = "tensorlbm.model_unit/v1"
MANIFEST_NAME = "model.json"

_COLLISIONS = ("bgk", "mrt", "trt")
_LATTICES = ("D2Q9",)
_FORCING = ("edm",)
_BOUNDARIES = ("periodic", "halfway_bounce_back")

# Exact double-precision D2Q9 weights.  ``tensorlbm.d2q9.W`` is float32 (and its
# float32 table has a documented ~7.45e-9/step mass leak at large tau), so the
# unit harness keeps the exact table for its force term.
_W_F64 = (
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
)


class ManifestError(ValueError):
    """Raised when a model-unit directory does not satisfy the schema."""


class GateFailure(AssertionError):
    """Raised when a unit cannot produce the quantity its gates are about."""


# --------------------------------------------------------------------------
# manifest model
# --------------------------------------------------------------------------


def _strict_keys(payload: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ManifestError(f"{where}: unknown key(s) {unknown}; allowed={sorted(allowed)}")


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ManifestError(f"{name} must be finite, got {value!r}")
    return out


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError(f"{name} must be a positive int, got {value!r}")
    return value


@dataclass(frozen=True)
class Gate:
    """One machine-checkable acceptance criterion on a kernel metric."""

    metric: str
    max: float

    @classmethod
    def from_json(cls, payload: Mapping[str, Any], where: str) -> "Gate":
        if not isinstance(payload, Mapping):
            raise ManifestError(f"{where}: gate must be an object, got {payload!r}")
        _strict_keys(payload, {"metric", "max"}, where)
        metric = payload["metric"]
        if not isinstance(metric, str) or not metric:
            raise ManifestError(f"{where}: metric must be a non-empty string")
        return cls(metric=metric, max=_finite_float(payload["max"], f"{where}.max"))

    def to_json(self) -> dict[str, Any]:
        return {"metric": self.metric, "max": self.max}


@dataclass(frozen=True)
class BuildSpec:
    """Parameters of the per-directory build step."""

    dtype: str
    fingerprint_steps: int

    @classmethod
    def from_json(cls, payload: Mapping[str, Any], where: str) -> "BuildSpec":
        if not isinstance(payload, Mapping):
            raise ManifestError(f"{where}: build must be an object")
        _strict_keys(payload, {"dtype", "fingerprint_steps"}, where)
        dtype = payload["dtype"]
        if dtype not in ("float32", "float64"):
            raise ManifestError(f"{where}.dtype must be float32|float64, got {dtype!r}")
        return cls(
            dtype=str(dtype),
            fingerprint_steps=_positive_int(
                payload["fingerprint_steps"], f"{where}.fingerprint_steps"
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "fingerprint_steps": self.fingerprint_steps}


@dataclass(frozen=True)
class CISpec:
    """Parameters of the per-directory CI benchmark and its gates."""

    kernel: str
    params: Mapping[str, Any]
    gates: tuple[Gate, ...]

    @classmethod
    def from_json(cls, payload: Mapping[str, Any], where: str) -> "CISpec":
        if not isinstance(payload, Mapping):
            raise ManifestError(f"{where}: ci must be an object")
        _strict_keys(payload, {"kernel", "params", "gates"}, where)
        kernel = payload["kernel"]
        if kernel not in _KERNELS:
            raise ManifestError(f"{where}.kernel must be one of {sorted(_KERNELS)}, got {kernel!r}")
        params = payload["params"]
        if not isinstance(params, Mapping) or not params:
            raise ManifestError(f"{where}.params must be a non-empty object")
        gates_raw = payload["gates"]
        if not isinstance(gates_raw, list) or not gates_raw:
            raise ManifestError(f"{where}.gates must be a non-empty list")
        gates = tuple(Gate.from_json(g, f"{where}.gates[{i}]") for i, g in enumerate(gates_raw))
        seen: set[str] = set()
        for gate in gates:
            if gate.metric in seen:
                raise ManifestError(f"{where}.gates: duplicate metric {gate.metric!r}")
            seen.add(gate.metric)
        return cls(kernel=str(kernel), params=dict(params), gates=gates)

    def to_json(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "params": dict(self.params),
            "gates": [g.to_json() for g in self.gates],
        }


@dataclass(frozen=True)
class ModelUnit:
    """One ``models/units/<dir>/model.json`` directory."""

    unit_id: str
    description: str
    composition: ModelComposition
    build: BuildSpec
    ci: CISpec
    path: Path

    def to_json(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "description": self.description,
            "composition": {
                "lattice": self.composition.lattice,
                "collision": self.composition.collision,
                "turbulence": self.composition.turbulence,
                "forcing": list(self.composition.forcing),
                "boundaries": list(self.composition.boundaries),
                "physics_modules": dict(self.composition.physics_modules),
            },
            "build": self.build.to_json(),
            "ci": self.ci.to_json(),
        }

    @property
    def manifest_hash(self) -> str:
        return _sha256_json(self.to_json())


def _sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_manifest(path: str | Path) -> ModelUnit:
    """Load and strictly validate one model-unit manifest."""

    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"{path}: missing manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(payload, Mapping):
        raise ManifestError(f"{path}: manifest must be a JSON object")
    where = str(path)
    _strict_keys(
        payload,
        {"schema", "unit_id", "description", "composition", "build", "ci"},
        where,
    )
    missing = sorted(
        {"schema", "unit_id", "description", "composition", "build", "ci"} - set(payload)
    )
    if missing:
        raise ManifestError(f"{where}: missing required key(s) {missing}")
    if payload["schema"] != SCHEMA:
        raise ManifestError(f"{where}: schema must be {SCHEMA!r}, got {payload['schema']!r}")
    unit_id = payload["unit_id"]
    if not isinstance(unit_id, str) or not unit_id:
        raise ManifestError(f"{where}: unit_id must be a non-empty string")
    if unit_id != path.parent.name:
        raise ManifestError(
            f"{where}: unit_id {unit_id!r} must equal the directory name"
            f" {path.parent.name!r} (the directory *is* the unit)"
        )
    description = payload["description"]
    if not isinstance(description, str) or not description:
        raise ManifestError(f"{where}: description must be a non-empty string")

    comp_raw = payload["composition"]
    if not isinstance(comp_raw, Mapping):
        raise ManifestError(f"{where}: composition must be an object")
    comp_keys = {"lattice", "collision", "turbulence", "forcing", "boundaries", "physics_modules"}
    _strict_keys(comp_raw, comp_keys, f"{where}.composition")
    comp_missing = sorted(comp_keys - set(comp_raw))
    if comp_missing:
        raise ManifestError(f"{where}.composition: missing key(s) {comp_missing}")
    lattice = comp_raw["lattice"]
    if lattice not in _LATTICES:
        raise ManifestError(
            f"{where}: unsupported lattice {lattice!r}; supported={list(_LATTICES)}"
        )
    collision = comp_raw["collision"]
    if collision not in _COLLISIONS:
        raise ManifestError(
            f"{where}: unsupported collision {collision!r}; supported={list(_COLLISIONS)}"
        )
    forcing_raw = comp_raw["forcing"]
    if not isinstance(forcing_raw, list):
        raise ManifestError(f"{where}.composition.forcing must be a list")
    for item in forcing_raw:
        if item not in _FORCING:
            raise ManifestError(
                f"{where}: unsupported forcing {item!r}; supported={list(_FORCING)}"
            )
    boundaries_raw = comp_raw["boundaries"]
    if not isinstance(boundaries_raw, list) or not boundaries_raw:
        raise ManifestError(f"{where}.composition.boundaries must be a non-empty list")
    for item in boundaries_raw:
        if item not in _BOUNDARIES:
            raise ManifestError(
                f"{where}: unsupported boundary {item!r}; supported={list(_BOUNDARIES)}"
            )
    try:
        composition = ModelComposition(
            lattice=lattice,
            collision=collision,
            turbulence=comp_raw["turbulence"],
            forcing=tuple(forcing_raw),
            boundaries=tuple(boundaries_raw),
            physics_modules=comp_raw["physics_modules"],
        )
    except ValueError as exc:
        raise ManifestError(f"{where}.composition: {exc}") from exc

    build = BuildSpec.from_json(payload["build"], f"{where}.build")
    ci = CISpec.from_json(payload["ci"], f"{where}.ci")

    # cross-field contract: the kernel must be able to run this composition
    _KERNELS[ci.kernel].check(ci.params, composition)
    return ModelUnit(
        unit_id=unit_id,
        description=description,
        composition=composition,
        build=build,
        ci=ci,
        path=path,
    )


def discover_units(root: str | Path) -> list[ModelUnit]:
    """Discover every model unit under *root* (a directory *is* a unit).

    A sub-directory without a ``model.json`` is an error rather than being
    skipped: silently skipping is exactly how a broken model disappears from
    CI under this convention.
    """

    root = Path(root)
    if not root.is_dir():
        raise ManifestError(f"{root}: model-unit root is not a directory")
    units: list[ModelUnit] = []
    seen: dict[str, Path] = {}
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if child.name.startswith((".", "_")):
            continue
        manifest = child / MANIFEST_NAME
        if not manifest.is_file():
            raise ManifestError(
                f"{child}: directory has no {MANIFEST_NAME}; every directory under"
                f" {root} must be a buildable model unit"
            )
        unit = load_manifest(manifest)
        if unit.unit_id in seen:
            raise ManifestError(
                f"{child}: duplicate unit_id {unit.unit_id!r} also in {seen[unit.unit_id]}"
            )
        seen[unit.unit_id] = manifest
        units.append(unit)
    if not units:
        raise ManifestError(f"{root}: no model unit found")
    return units


# --------------------------------------------------------------------------
# kernels: each returns metrics + the analytic reference it is judged against
# --------------------------------------------------------------------------


@dataclass
class KernelResult:
    """Output of one kernel run."""

    metrics: dict[str, float]
    reference: dict[str, float]
    series: dict[str, list[float]] = field(default_factory=dict)
    state_hash: str = ""
    notes: dict[str, Any] = field(default_factory=dict)


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _collide(f: torch.Tensor, composition: ModelComposition, tau: float) -> torch.Tensor:
    kind = composition.collision.lower()
    if kind == "bgk":
        return collide_bgk(f, tau)
    if kind == "mrt":
        return collide_mrt(f, tau)
    if kind == "trt":
        return collide_trt(f, tau)
    raise ManifestError(f"unsupported collision {kind!r}")


def _weights(dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(_W_F64, dtype=dtype).view(9, 1, 1)


def _state_hash(f: torch.Tensor) -> str:
    payload = f.detach().to(torch.float64).contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _kernel_params(
    params: Mapping[str, Any], required: set[str], optional: set[str], where: str
) -> None:
    unknown = sorted(set(params) - required - optional)
    if unknown:
        raise ManifestError(f"{where}: unknown kernel param(s) {unknown}")
    missing = sorted(required - set(params))
    if missing:
        raise ManifestError(f"{where}: missing kernel param(s) {missing}")


def _check_positive(params: Mapping[str, Any], keys: set[str], where: str) -> None:
    for key in sorted(keys):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ManifestError(f"{where}.{key} must be a positive int, got {value!r}")


def _check_tau(params: Mapping[str, Any], where: str) -> float:
    if "tau" not in params:
        raise ManifestError(f"{where}: missing kernel param 'tau'")
    tau = _finite_float(params["tau"], f"{where}.tau")
    if tau <= 0.5:
        raise ManifestError(f"{where}.tau must be > 0.5 (nu=(tau-1/2)/3 > 0), got {tau}")
    return tau


def _shear_wave_check(params: Mapping[str, Any], composition: ModelComposition) -> None:
    where = "kernel shear_wave_decay"
    _kernel_params(
        params,
        required={"nx", "ny", "tau", "u0", "steps"},
        optional={"periods", "t_start", "t_end"},
        where=where,
    )
    _check_positive(params, {"nx", "ny", "steps"}, where)
    tau = _check_tau(params, where)
    u0 = _finite_float(params["u0"], f"{where}.u0")
    if not 0.0 < u0 < 0.1:
        raise ManifestError(f"{where}.u0 must be in (0, 0.1) to stay in the low-Mach regime")
    periods = params.get("periods", 1)
    if (
        isinstance(periods, bool)
        or not isinstance(periods, int)
        or not 1 <= periods <= int(params["ny"]) // 4
    ):
        raise ManifestError(f"{where}.periods must be an int in [1, ny//4]")
    t_start = int(params.get("t_start", 0))
    t_end = int(params.get("t_end", params["steps"]))
    if not 0 <= t_start < t_end <= int(params["steps"]):
        raise ManifestError(f"{where}: need 0 <= t_start < t_end <= steps")
    if "periodic" not in composition.boundaries:
        raise ManifestError(f"{where}: requires the 'periodic' boundary")
    if composition.forcing:
        raise ManifestError(
            f"{where}: unforced kernel, composition declares forcing={composition.forcing}"
        )
    if tau - 0.5 < 1e-3:
        raise ManifestError(f"{where}.tau too close to 0.5 for a stable decay measurement")


def kernel_shear_wave_decay(
    params: Mapping[str, Any],
    composition: ModelComposition,
    *,
    dtype: torch.dtype = torch.float64,
    steps_override: int | None = None,
) -> KernelResult:
    """Transverse shear wave in a periodic box: u_x = u0 sin(k y) exp(-nu k^2 t).

    The decay rate of the fundamental Fourier mode is compared against the
    analytic nu = (tau-1/2)/3 implied by the composition's relaxation time.  It
    is an exact property of the linearised scheme, so the residual error
    measures mode contamination rather than a physical approximation.
    """

    nx = int(params["nx"])
    ny = int(params["ny"])
    tau = float(params["tau"])
    u0 = float(params["u0"])
    periods = int(params.get("periods", 1))
    full_steps = int(params["steps"])
    steps = int(steps_override if steps_override is not None else full_steps)
    t_start = int(min(params.get("t_start", 0), max(steps - 1, 0)))
    t_end = int(min(params.get("t_end", steps), steps))

    k = 2.0 * math.pi * periods / float(ny)
    y = torch.arange(ny, dtype=dtype)
    mode = torch.sin(k * y).view(ny, 1).expand(ny, nx)
    rho = torch.ones(ny, nx, dtype=dtype)
    ux = u0 * mode
    uy = torch.zeros_like(ux)
    f = _equilibrium(rho, ux, uy)

    def amplitude(state: torch.Tensor) -> float:
        _, u, _ = _macroscopic(state)
        return float((2.0 / (ny * nx)) * (u * mode).sum())

    a0 = amplitude(f)
    mass0 = float(f.sum())
    amps: dict[int, float] = {0: a0}
    for step in range(1, steps + 1):
        f = stream(_collide(f, composition, tau))
        if step in (t_start, t_end) and step > 0:
            amps[step] = amplitude(f)
    a_start, a_end = amps[t_start], amps[t_end]
    dt = float(t_end - t_start)

    nu_ref = (tau - 0.5) / 3.0
    if a_start <= 0.0 or a_end <= 0.0:
        raise GateFailure(
            f"{composition.collision} shear wave lost its amplitude before t_end"
            f" (A({t_start})={a_start:.3e}, A({t_end})={a_end:.3e}); check stability"
        )
    nu_meas = math.log(a_start / a_end) / (k * k * dt)
    mass_end = float(f.sum())
    metrics = {
        "nu_rel_err": abs(nu_meas - nu_ref) / nu_ref,
        "amp0_rel_err": abs(a0 - u0) / u0,
        "mass_rel_drift": abs(mass_end - mass0) / mass0,
    }
    reference = {"nu": nu_ref, "amp0": u0}
    return KernelResult(
        metrics=metrics,
        reference=reference,
        series={"nu": [nu_ref, nu_meas], "amp": [a_start, a_end], "k": [k]},
        state_hash=_state_hash(f),
        notes={
            "steps": steps,
            "sample": [t_start, t_end],
            "k": k,
            "dtype": str(dtype).replace("torch.", ""),
        },
    )


def _equilibrium_shift(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    fx: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Exact-difference (EDM) body-force term for D2Q9, force along +x.

    ``f_post = collide(f) + [feq(rho, u + dv) - feq(rho, u)]`` with
    ``dv = F/rho``.  The shift only moves the equilibrium argument, so the
    relaxation operator stays whatever the composition declared (BGK/MRT/TRT);
    it is algebraically the Guo/EDM family at low Mach (Kupershtokh 2009).
    """

    dv = fx / rho
    return _equilibrium(rho, ux + dv, uy) - _equilibrium(rho, ux, uy)


def _apply_halfway_walls(
    f: torch.Tensor, fc: torch.Tensor, c_y: torch.Tensor, opp: torch.Tensor
) -> torch.Tensor:
    """Half-way bounce-back on the first/last row (walls at y=-1/2 and ny-1/2)."""

    out = f.clone()
    up = c_y > 0  # populations that wrap in from the north wall
    down = c_y < 0  # populations that wrap in from the south wall
    out[up, 0, :] = fc[opp[up], 0, :]
    out[down, -1, :] = fc[opp[down], -1, :]
    return out


def _poiseuille_check(params: Mapping[str, Any], composition: ModelComposition) -> None:
    where = "kernel poiseuille_channel"
    _kernel_params(
        params,
        required={"nx", "ny", "tau", "g", "steps", "avg_start"},
        optional=set(),
        where=where,
    )
    _check_positive(params, {"nx", "ny", "steps"}, where)
    _check_tau(params, where)
    g = _finite_float(params["g"], f"{where}.g")
    if not 0.0 < g < 1e-2:
        raise ManifestError(f"{where}.g must be in (0, 1e-2) to stay in the low-Mach regime")
    avg_start = params["avg_start"]
    if (
        isinstance(avg_start, bool)
        or not isinstance(avg_start, int)
        or not 0 < avg_start < int(params["steps"])
    ):
        raise ManifestError(f"{where}.avg_start must be an int in (0, steps)")
    if "halfway_bounce_back" not in composition.boundaries:
        raise ManifestError(f"{where}: requires the 'halfway_bounce_back' boundary")
    if "periodic" in composition.boundaries:
        raise ManifestError(f"{where}: walls and periodic boundaries are exclusive here")
    if "edm" not in composition.forcing:
        raise ManifestError(f"{where}: requires forcing=['edm'] in the composition")


def kernel_poiseuille_channel(
    params: Mapping[str, Any],
    composition: ModelComposition,
    *,
    dtype: torch.dtype = torch.float64,
    steps_override: int | None = None,
) -> KernelResult:
    """Force-driven plane Poiseuille flow against the analytic parabola.

    Walls sit half-way between the first/last fluid row and the wall plane, so
    the channel height is exactly ``ny`` lattice units and the steady solution
    is ``u(y) = g/(2 nu) (y+1/2)(ny-1/2-y)`` with ``g`` the force density per
    unit mass and ``nu = (tau-1/2)/3``.  The body force enters through the
    exact-difference (EDM) equilibrium shift, so any relaxation operator of the
    composition is usable without a force-specific branch.  The measured profile is compared as an
    L2 norm.  The bands are periodicity-free in x, so ``nx`` only sets the
    vectorisation width.
    """

    nx = int(params["nx"])
    ny = int(params["ny"])
    tau = float(params["tau"])
    g = float(params["g"])
    full_steps = int(params["steps"])
    steps = int(steps_override if steps_override is not None else full_steps)
    avg_start = int(min(params["avg_start"], max(steps - 1, 1)))

    c_y = C[:, 1]
    opp = OPPOSITE.to(torch.int64)
    rho = torch.ones(ny, nx, dtype=dtype)
    zero = torch.zeros_like(rho)
    f = _equilibrium(rho, zero, zero)

    mass0 = float(rho.sum())
    acc = torch.zeros(ny, dtype=torch.float64)
    samples = 0
    for step in range(1, steps + 1):
        rho, ux, uy = _macroscopic(f)
        fc = _collide(f, composition, tau)
        fc = fc + _equilibrium_shift(rho, ux, uy, g, dtype)
        f = stream(fc)
        f = _apply_halfway_walls(f, fc, c_y, opp)
        if step > avg_start:
            rho_n, ux_n, _ = _macroscopic(f)
            acc += (ux_n + 0.5 * g / rho_n).mean(dim=1).double()
            samples += 1

    profile = (acc / max(samples, 1)).tolist()
    nu = (tau - 0.5) / 3.0
    y = torch.arange(ny, dtype=torch.float64)
    height = float(ny)
    exact = (g / (2.0 * nu) * (y + 0.5) * (height - 0.5 - y)).tolist()
    got = torch.tensor(profile, dtype=torch.float64)
    ref = torch.tensor(exact, dtype=torch.float64)
    l2_rel = float(torch.linalg.norm(got - ref) / torch.linalg.norm(ref))
    umax_rel = abs(float(got.max()) - float(ref.max())) / float(ref.max())
    drift = abs(float(rho.sum()) - mass0) / mass0
    metrics = {
        "profile_l2_rel_err": l2_rel,
        "umax_rel_err": umax_rel,
        "mass_rel_drift": drift,
    }
    reference = {"nu": nu, "umax": float(ref.max())}
    return KernelResult(
        metrics=metrics,
        reference=reference,
        series={
            "y": [float(v) for v in y.tolist()],
            "u_lbm": profile,
            "u_exact": exact,
        },
        state_hash=_state_hash(f),
        notes={
            "steps": steps,
            "avg_window": [avg_start, steps],
            "samples": samples,
            "dtype": str(dtype).replace("torch.", ""),
        },
    )


@dataclass(frozen=True)
class _Kernel:
    name: str
    run: Callable[..., KernelResult]
    check: Callable[[Mapping[str, Any], ModelComposition], None]
    reference_source: str


_KERNELS: dict[str, _Kernel] = {
    "shear_wave_decay": _Kernel(
        name="shear_wave_decay",
        run=kernel_shear_wave_decay,
        check=_shear_wave_check,
        reference_source=("analytic: shear mode decays as exp(-nu k^2 t) with nu=(tau-1/2)/3"),
    ),
    "poiseuille_channel": _Kernel(
        name="poiseuille_channel",
        run=kernel_poiseuille_channel,
        check=_poiseuille_check,
        reference_source=(
            "analytic: u(y)=g/(2nu)(y+1/2)(ny-1/2-y) with nu=(tau-1/2)/3,"
            " walls at y=-1/2 and y=ny-1/2"
        ),
    ),
}


# --------------------------------------------------------------------------
# build + CI
# --------------------------------------------------------------------------


@dataclass
class BuildRecord:
    unit_id: str
    ok: bool
    manifest_hash: str
    fingerprint: str = ""
    fingerprint_repeat: str = ""
    determinism: bool = False
    collision: str = ""
    kernel: str = ""
    dtype: str = ""
    seconds: float = 0.0
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "ok": self.ok,
            "manifest_hash": self.manifest_hash,
            "build_id": _sha256_json([self.manifest_hash, self.fingerprint]),
            "fingerprint": self.fingerprint,
            "fingerprint_repeat": self.fingerprint_repeat,
            "deterministic": self.determinism,
            "collision": self.collision,
            "kernel": self.kernel,
            "dtype": self.dtype,
            "seconds": self.seconds,
            "error": self.error,
        }


def build_unit(unit: ModelUnit) -> BuildRecord:
    """Per-directory build step: validate, instantiate, fingerprint.

    The fingerprint is the sha256 of the distribution field after
    ``fingerprint_steps`` steps, computed twice from the same manifest.  A
    mismatch means the same model directory does not build the same thing
    twice, so any CI verdict measured on it would be untrustworthy.
    """

    started = time.perf_counter()
    record = BuildRecord(
        unit_id=unit.unit_id,
        ok=False,
        manifest_hash=unit.manifest_hash,
        collision=unit.composition.collision,
        kernel=unit.ci.kernel,
        dtype=unit.build.dtype,
    )
    try:
        dtype = _torch_dtype(unit.build.dtype)
        kernel = _KERNELS[unit.ci.kernel]
        kernel.check(unit.ci.params, unit.composition)
        first = kernel.run(
            unit.ci.params,
            unit.composition,
            dtype=dtype,
            steps_override=unit.build.fingerprint_steps,
        )
        second = kernel.run(
            unit.ci.params,
            unit.composition,
            dtype=dtype,
            steps_override=unit.build.fingerprint_steps,
        )
    except (ManifestError, GateFailure, ValueError, RuntimeError) as exc:
        record.error = f"{type(exc).__name__}: {exc}"
        record.seconds = time.perf_counter() - started
        return record
    record.fingerprint = first.state_hash
    record.fingerprint_repeat = second.state_hash
    record.determinism = first.state_hash == second.state_hash
    record.ok = record.determinism
    if not record.determinism:
        record.error = "non-deterministic build: fingerprint changed between two identical runs"
    record.seconds = time.perf_counter() - started
    return record


@dataclass
class CIRecord:
    unit_id: str
    ok: bool
    build_id: str
    kernel: str
    reference_source: str
    metrics: dict[str, float] = field(default_factory=dict)
    reference: dict[str, float] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)
    series: dict[str, list[float]] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    error: str = ""

    def to_json(self, *, with_series: bool = False) -> dict[str, Any]:
        payload = {
            "unit_id": self.unit_id,
            "ok": self.ok,
            "build_id": self.build_id,
            "kernel": self.kernel,
            "reference_source": self.reference_source,
            "metrics": self.metrics,
            "reference": self.reference,
            "gates": self.gates,
            "notes": self.notes,
            "seconds": self.seconds,
            "error": self.error,
        }
        if with_series:
            payload["series"] = self.series
        return payload


def run_ci(unit: ModelUnit, build: BuildRecord | None = None) -> CIRecord:
    """Per-directory CI run: benchmark the unit and apply its declared gates."""

    record = CIRecord(
        unit_id=unit.unit_id,
        ok=False,
        build_id=_sha256_json([unit.manifest_hash, (build.fingerprint if build else "")]),
        kernel=unit.ci.kernel,
        reference_source=_KERNELS[unit.ci.kernel].reference_source,
    )
    started = time.perf_counter()
    try:
        kernel = _KERNELS[unit.ci.kernel]
        result = kernel.run(unit.ci.params, unit.composition, dtype=_torch_dtype(unit.build.dtype))
    except (ManifestError, GateFailure, ValueError, RuntimeError) as exc:
        record.error = f"{type(exc).__name__}: {exc}"
        record.seconds = time.perf_counter() - started
        return record
    record.metrics = dict(result.metrics)
    record.reference = dict(result.reference)
    record.series = dict(result.series)
    record.notes = dict(result.notes)

    missing = [g.metric for g in unit.ci.gates if g.metric not in result.metrics]
    if missing:
        record.error = f"kernel {unit.ci.kernel!r} did not produce gated metric(s) {missing}"
    failed = 0
    for gate in unit.ci.gates:
        value = result.metrics.get(gate.metric)
        passed = value is not None and abs(value) <= gate.max
        failed += 0 if passed else 1
        record.gates.append(
            {"metric": gate.metric, "value": value, "max": gate.max, "passed": passed}
        )
    record.ok = failed == 0 and not record.error
    record.seconds = time.perf_counter() - started
    return record


@dataclass
class UnitRun:
    """Build + CI verdict for a single directory."""

    unit: ModelUnit
    build: BuildRecord
    ci: CIRecord | None

    @property
    def ok(self) -> bool:
        return bool(self.build.ok and self.ci is not None and self.ci.ok)

    def to_json(self, *, with_series: bool = False) -> dict[str, Any]:
        return {
            "unit_id": self.unit.unit_id,
            "ok": self.ok,
            "manifest": self.unit.to_json(),
            "build": self.build.to_json(),
            "ci": self.ci.to_json(with_series=with_series) if self.ci else None,
        }


def run_unit(unit: ModelUnit) -> UnitRun:
    """Build then CI one unit; a failed build skips the CI run by design."""

    build = build_unit(unit)
    ci = run_ci(unit, build) if build.ok else None
    return UnitRun(unit=unit, build=build, ci=ci)


def run_all(root: str | Path, *, only: tuple[str, ...] = ()) -> list[UnitRun]:
    """Build+CI every unit under *root*, in directory order."""

    units = discover_units(root)
    if only:
        known = {u.unit_id for u in units}
        missing = sorted(set(only) - known)
        if missing:
            raise ManifestError(f"--only: unknown unit(s) {missing}; known={sorted(known)}")
        units = [u for u in units if u.unit_id in set(only)]
    return [run_unit(unit) for unit in units]


# --------------------------------------------------------------------------
# artefacts
# --------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}e}" if abs(value) < 1e-2 else f"{value:.{digits}f}"


def cross_unit_comparison(runs: list[UnitRun]) -> list[dict[str, Any]]:
    """Compare units that share a kernel on their primary analytic metric.

    Two directories with the same benchmark but a different model composition
    are a controlled comparison: the kernel and parameters are identical, only
    the collision operator changes.  Reporting the ratio keeps that comparison
    in the CI artefact instead of in someone's head.
    """

    rows: list[dict[str, Any]] = []
    groups: dict[str, list[UnitRun]] = {}
    for run in runs:
        if run.ci is not None and run.ci.gates:
            groups.setdefault(run.unit.ci.kernel, []).append(run)
    for kernel, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        metric = members[0].unit.ci.gates[0].metric
        measured = {
            m.unit.unit_id: m.ci.metrics.get(metric)
            for m in members
            if m.ci is not None and m.ci.metrics.get(metric) is not None
        }
        if len(measured) < 2:
            continue
        best = min(measured, key=lambda k: measured[k])
        worst = max(measured, key=lambda k: measured[k])
        ratio = measured[worst] / measured[best] if measured[best] else math.inf
        rows.append(
            {
                "kernel": kernel,
                "metric": metric,
                "best": best,
                "best_value": measured[best],
                "worst": worst,
                "worst_value": measured[worst],
                "ratio": ratio,
            }
        )
    return rows


def summary_markdown(runs: list[UnitRun], *, repro_cmd: str) -> str:
    """Per-unit verdict table (one row per directory) plus the benchmark table."""

    lines = [
        "# Model-unit CI summary (model directory = build + CI unit)",
        "",
        f"Reproduce: `{repro_cmd}`",
        "",
        "| unit | collision | kernel | build | determinism | CI | failing gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        failing = "-"
        if run.ci:
            bad = [g for g in run.ci.gates if not g["passed"]]
            failing = ", ".join(f"{g['metric']}={_fmt(g['value'])}>{g['max']:g}" for g in bad) or (
                "-" if run.ci.ok else "error"
            )
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                run.unit.unit_id,
                run.unit.composition.collision,
                run.unit.ci.kernel,
                "ok" if run.build.ok else "FAIL",
                "{}".format("identical" if run.build.determinism else "mismatch"),
                ("pass" if run.ci.ok else "FAIL") if run.ci else "skipped",
                failing,
            )
        )
    lines.append("")
    lines.append("## Benchmark rows (reference vs measured)")
    lines.append("")
    lines.append(
        "| unit | quantity | reference (analytic) | measured | rel. error | gate | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for run in runs:
        if not run.ci:
            continue
        for gate in run.ci.gates:
            metric = gate["metric"]
            ref, got = _ref_got(run, metric)
            lines.append(
                "| `{}` | `{}` | {} | {} | {} | <= {:g} | {} |".format(
                    run.unit.unit_id,
                    metric,
                    _fmt(ref),
                    _fmt(got),
                    _fmt(gate["value"]),
                    gate["max"],
                    "ok" if gate["passed"] else "FAIL",
                )
            )
    comparison = cross_unit_comparison(runs)
    if comparison:
        lines.append("")
        lines.append("## Cross-unit comparison (same kernel, different composition)")
        lines.append("")
        lines.append("| kernel | metric | better unit | worse unit | ratio |")
        lines.append("|---|---|---|---|---|")
        for row in comparison:
            lines.append(
                "| {} | `{}` | `{}` ({}) | `{}` ({}) | {:.1f}x |".format(
                    row["kernel"],
                    row["metric"],
                    row["best"],
                    _fmt(row["best_value"]),
                    row["worst"],
                    _fmt(row["worst_value"]),
                    row["ratio"],
                )
            )
    return "\n".join(lines) + "\n"


def _ref_got(run: UnitRun, metric: str) -> tuple[float | None, float | None]:
    """Reference/measured pair for one gated metric (measured == its own error)."""

    ci = run.ci
    if ci is None:
        return None, None
    got = ci.metrics.get(metric)
    if metric == "nu_rel_err":
        return ci.series.get("nu", [None])[0], ci.reference.get("nu")
    if metric == "profile_l2_rel_err":
        return 0.0, got
    if metric == "umax_rel_err":
        return ci.reference.get("umax"), None if got is None else got
    return None, got


def write_artifacts(runs: list[UnitRun], outdir: str | Path, *, repro_cmd: str) -> dict[str, Any]:
    """Persist build/CI records, profiles, the summary table and an index."""

    out = Path(outdir)
    (out / "builds").mkdir(parents=True, exist_ok=True)
    (out / "ci").mkdir(parents=True, exist_ok=True)
    (out / "series").mkdir(parents=True, exist_ok=True)
    summary = {
        "units": [],
        "repro_cmd": repro_cmd,
        "schema": SCHEMA,
    }
    for run in runs:
        (out / "builds" / f"{run.unit.unit_id}.json").write_text(
            json.dumps(run.build.to_json(), indent=2) + "\n", encoding="utf-8"
        )
        if run.ci is None:
            summary["units"].append(run.to_json())
            continue
        (out / "ci" / f"{run.unit.unit_id}.json").write_text(
            json.dumps(run.ci.to_json(with_series=True), indent=2) + "\n", encoding="utf-8"
        )
        if "y" in run.ci.series:
            rows = ["y,u_lbm,u_exact"]
            for y, lbm, exact in zip(
                run.ci.series["y"], run.ci.series["u_lbm"], run.ci.series["u_exact"]
            ):
                rows.append(f"{y:.1f},{lbm:.10e},{exact:.10e}")
            (out / "series" / f"{run.unit.unit_id}_profile.csv").write_text(
                "\n".join(rows) + "\n", encoding="utf-8"
            )
        summary["units"].append(run.to_json())
    summary["all_ok"] = all(run.ok for run in runs)
    summary["cross_unit"] = cross_unit_comparison(runs)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "SUMMARY.md").write_text(summary_markdown(runs, repro_cmd=repro_cmd), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _default_root() -> Path:
    # src/tensorlbm/models/ci_unit.py -> <repo>/models/units
    return Path(__file__).resolve().parents[3] / "models" / "units"


def _print_list(units: list[ModelUnit]) -> None:
    for unit in units:
        print(
            f"{unit.unit_id:34s} {unit.composition.collision:4s} "
            f"{unit.ci.kernel:20s} {unit.build.dtype}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tensorlbm.models.ci_unit",
        description="Build and CI every model directory (a directory is a unit).",
    )
    parser.add_argument("command", choices=("list", "build", "ci", "all"), help="what to do")
    parser.add_argument("--root", default=None, help="model-unit root (default models/units)")
    parser.add_argument("--out", default="artifacts/model_units", help="artefact directory")
    parser.add_argument("--only", nargs="*", default=(), help="restrict to these unit ids")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="exit 0 even when a unit fails (artefacts still record the verdict)",
    )
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else _default_root()
    try:
        units = discover_units(root)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.only:
        known = {u.unit_id for u in units}
        missing = sorted(set(args.only) - known)
        if missing:
            print(f"error: unknown unit(s) {missing}", file=sys.stderr)
            return 2
        units = [u for u in units if u.unit_id in set(args.only)]

    if args.command == "list":
        _print_list(units)
        return 0

    if args.command == "build":
        records = [build_unit(unit) for unit in units]
        failed = [r for r in records if not r.ok]
        for record in records:
            state = "ok" if record.ok else f"FAIL ({record.error})"
            print(f"{record.unit_id:34s} build {state} in {record.seconds:.2f}s")
        return 0 if (not failed or args.allow_failures) else 1

    runs = run_all(root, only=tuple(args.only))
    repro = "python -m tensorlbm.models.ci_unit all --root {} --out {}".format(root, args.out)
    if args.command == "ci":
        for run in runs:
            status = "pass" if run.ok else "FAIL"
            details = "" if run.ci is None else f" kernel={run.ci.kernel} {run.ci.seconds:.2f}s"
            print(f"{run.unit.unit_id:34s} ci {status}{details}")
        return 0 if (all(r.ok for r in runs) or args.allow_failures) else 1

    summary = write_artifacts(runs, args.out, repro_cmd=repro)
    print(summary_markdown(runs, repro_cmd=repro))
    print(f"artefacts: {Path(args.out).resolve()}")
    print(f"all_ok: {summary['all_ok']}")
    return 0 if (summary["all_ok"] or args.allow_failures) else 1


__all__ = [
    "BuildRecord",
    "BuildSpec",
    "CIRecord",
    "CISpec",
    "Gate",
    "GateFailure",
    "KernelResult",
    "ManifestError",
    "ModelUnit",
    "UnitRun",
    "build_unit",
    "cross_unit_comparison",
    "discover_units",
    "kernel_poiseuille_channel",
    "kernel_shear_wave_decay",
    "load_manifest",
    "main",
    "run_all",
    "run_ci",
    "run_unit",
    "summary_markdown",
    "write_artifacts",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI entry
    raise SystemExit(main())
