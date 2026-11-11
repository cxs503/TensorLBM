"""Heterogeneous-backend equivalence ladder for TensorLBM ("换后端不改物理").

Why this module exists
----------------------
The same lattice-Boltzmann scheme can be executed by several tiers: NumPy,
PyTorch-CPU, PyTorch on an accelerator (CUDA / SDAA), Triton-fused kernels,
Paddle, MindSpore.  A faster tier is only usable if it reproduces the physics.
This module makes that requirement *machine-checkable* as a three-rung ladder
plus a cost/accuracy ledger, and it makes a missing or under-precision tier
**fail closed** (raise + register) instead of silently falling back to the CPU
reference or silently dropping to a lower precision.

The ladder
----------
``L0`` bitwise / ULP bound
    Small periodic D3Q19 box (default ``64**3``, 200 steps).  The same discrete
    scheme (second-order equilibrium + BGK + periodic streaming) is run by
    independent implementations and precision tiers.  Certified when either
    (a) the macroscopic trace is **bitwise** identical (SHA-256 digest chain of
    the per-step traces), or (b) the deviation stays inside an *a-priori
    rounding budget* ``A_N * N_OPS_PER_STEP * eps``, where ``A_N`` is the
    amplification of a one-ULP seeded perturbation measured on the golden
    trajectory itself.  The budget separates "different rounding" from
    "different algorithm": a scheme difference lands orders of magnitude above
    it (see ``tests/test_hetero_equivalence.py`` negative control).
``L1`` long-time statistical equivalence
    ``Re_tau = 180`` plane channel, ``10**4`` steps, driven by the production
    wall-resolved solver (:mod:`tensorlbm.turbulence.turbulent_channel3d`).
    Gated on the ``U+`` and Reynolds-stress rms profiles: normalised deviation
    against the CPU-fp64 golden ``<= 2 %``, **and** the deviation against the
    MKM-1999 DNS (Moser, Kim & Mansour 1999) must not be worse than golden's.
``L2`` physical invariants
    Conservation drift of the unforced box run: mass and momentum drift per
    step per cell ``<= 1e-12``.

Cost ledger
-----------
Every run reports cells*steps/s, seconds per lattice step, seconds per
convective time (``h/u_tau`` steps) and the speed-up against the golden tier;
:func:`pareto_front` ranks the (error, cost) points so the fp32/fp64 trade-off
is an explicit decision rather than a habit.

Fail-closed semantics
---------------------
* :func:`resolve_profile` raises :class:`BackendUnavailable` when the tier's
  device/framework is absent and appends a refusal record; it never returns the
  CPU fallback.
* :func:`require_precision` raises :class:`PrecisionInsufficient` when a caller
  demands fp64 semantics from a lower-precision tier.
* :func:`require_certified` raises :class:`CertificationError` when a tier is
  used above the equivalence level it was certified for.

Scope / limitations
-------------------
* The L0 budget is an engineering bound (local rounding budget scaled by a
  *measured* trajectory amplification), not a bit-exact forward-error proof of a
  chaotic nonlinear iteration.
* L1 is a *paired* comparison (identical initial condition, seed and grid): it
  detects implementation/precision differences; physical accuracy is anchored
  separately by the MKM-1999 DNS column.
* Accelerator rungs are probed at runtime; when absent they are certified as
  *refusals* and the ladder is instantiated on the tiers that do exist.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tensorlbm.d3q19 import C as C_TORCH
from tensorlbm.d3q19 import OPPOSITE as OPPOSITE_TORCH
from tensorlbm.d3q19 import W as W_TORCH

__all__ = [
    "SCHEMA",
    "GOLDEN_PROFILE",
    "L1_MAX_REL_DEVIATION",
    "L1_MKM_NOT_WORSE_TOL",
    "L2_DRIFT_MAX",
    "N_OPS_PER_STEP",
    "PROFILES",
    "BackendProfile",
    "BackendUnavailable",
    "CertificationError",
    "ChunkedChannelRun",
    "PrecisionInsufficient",
    "RefusalLedger",
    "aggregate_ladder",
    "certify",
    "check_l0",
    "check_l1",
    "check_l2",
    "default_ledger",
    "freeze_golden",
    "l0_run",
    "l1_run_channel",
    "measure_amplification",
    "normalised_profile_deviation",
    "pareto_front",
    "probe_profile",
    "require_certified",
    "require_precision",
    "resolve_profile",
    "ulp_distance",
    "ulp_budget",
    "exact_d3q19_weights",
    "weight_table_audit",
]

SCHEMA = "tensorlbm-hetero-equivalence-v1"

#: dtype name -> (numpy dtype, torch dtype name, machine epsilon)
_DTYPES: dict[str, tuple[Any, str, float]] = {
    "float64": (np.float64, "float64", float(np.finfo(np.float64).eps)),
    "float32": (np.float32, "float32", float(np.finfo(np.float32).eps)),
}

PRECISION_RANK: dict[str, int] = {"float16": 0, "float32": 1, "float64": 2}

#: Rounding-producing scalar operations per cell per D3Q19 step, used by the L0
#: budget: equilibrium 19 x 13 flops, collision 19 x 3, moment sums 4 x 19,
#: streaming is a pure copy (no rounding).
N_OPS_PER_STEP = 400

#: Conservation drift gate (L2): per step, per cell.
L2_DRIFT_MAX = 1.0e-12

#: L1 statistical equivalence gate (relative, normalised L2).
L1_MAX_REL_DEVIATION = 0.02

#: Extra tolerance on "MKM deviation must not get worse" (relative).
L1_MKM_NOT_WORSE_TOL = 0.02


# ---------------------------------------------------------------------------
# errors, profiles, refusal ledger
# ---------------------------------------------------------------------------
class BackendUnavailable(RuntimeError):
    """The requested backend/precision tier does not exist on this host."""


class PrecisionInsufficient(ValueError):
    """A lower-precision tier was asked to provide higher-precision semantics."""


class CertificationError(RuntimeError):
    """A tier was used above the equivalence level it is certified for."""


@dataclass(frozen=True)
class BackendProfile:
    """One execution tier of the LBM update."""

    name: str
    framework: str  # "numpy" | "torch" | "paddle" | "mindspore" | "triton"
    device: str  # "cpu" | "cuda:0" | "sdaa:0"
    dtype: str  # "float64" | "float32"
    kernel: str
    requires: tuple[str, ...] = ()
    description: str = ""

    @property
    def eps(self) -> float:
        return _DTYPES[self.dtype][2]

    @property
    def precision_rank(self) -> int:
        return PRECISION_RANK[self.dtype]


PROFILES: dict[str, BackendProfile] = {
    "cpu_fp64_serial": BackendProfile(
        name="cpu_fp64_serial",
        framework="numpy",
        device="cpu",
        dtype="float64",
        kernel="serial_direction_loop",
        description="golden tier: fp64 CPU, strictly ordered per-direction reduction",
    ),
    "cpu_fp64_torch": BackendProfile(
        name="cpu_fp64_torch",
        framework="torch",
        device="cpu",
        dtype="float64",
        kernel="vectorised_torch",
        description="PyTorch CPU fp64, same reduction order as golden",
    ),
    "cpu_fp64_torch_gemm": BackendProfile(
        name="cpu_fp64_torch_gemm",
        framework="torch",
        device="cpu",
        dtype="float64",
        kernel="gemm_moments",
        description="PyTorch CPU fp64, BLAS-GEMM moment reduction (different order)",
    ),
    "cpu_fp32_torch": BackendProfile(
        name="cpu_fp32_torch",
        framework="torch",
        device="cpu",
        dtype="float32",
        kernel="vectorised_torch",
        description="PyTorch CPU fp32 (reduced-precision tier)",
    ),
    "cpu_fp64_numpy_tree": BackendProfile(
        name="cpu_fp64_numpy_tree",
        framework="numpy",
        device="cpu",
        dtype="float64",
        kernel="tree_reduction",
        description=(
            "NumPy fp64 with pairwise (tree) reductions and Horner-form "
            "equilibrium: same algebra, different rounding order"
        ),
    ),
    "cpu_fp64_shipped_w": BackendProfile(
        name="cpu_fp64_shipped_w",
        framework="numpy",
        device="cpu",
        dtype="float64",
        kernel="shipped_weights",
        description=(
            "negative control: fp64 NumPy with the shipped float32 weight table "
            "of tensorlbm.d3q19 (must fail the L0/L2 gates)"
        ),
    ),
    "cuda_fp64_torch": BackendProfile(
        name="cuda_fp64_torch",
        framework="torch",
        device="cuda:0",
        dtype="float64",
        kernel="vectorised_torch",
        description="PyTorch CUDA fp64 (accelerator reference tier)",
    ),
    "cuda_fp32_torch": BackendProfile(
        name="cuda_fp32_torch",
        framework="torch",
        device="cuda:0",
        dtype="float32",
        kernel="vectorised_torch",
        description="PyTorch CUDA fp32 (accelerator fast tier)",
    ),
    "triton_fused_fp32": BackendProfile(
        name="triton_fused_fp32",
        framework="triton",
        device="cuda:0",
        dtype="float32",
        kernel="triton_fused",
        requires=("triton",),
        description="Triton fused collide+stream kernel (fp32, CUDA)",
    ),
    "sdaa_fp32_torch": BackendProfile(
        name="sdaa_fp32_torch",
        framework="torch",
        device="sdaa:0",
        dtype="float32",
        kernel="vectorised_torch",
        description="PyTorch on SDAA accelerator (fp32)",
    ),
    "sdaa_fp64_torch": BackendProfile(
        name="sdaa_fp64_torch",
        framework="torch",
        device="sdaa:0",
        dtype="float64",
        kernel="vectorised_torch",
        description="PyTorch on SDAA accelerator (fp64 reference tier)",
    ),
    "paddle_fp32": BackendProfile(
        name="paddle_fp32",
        framework="paddle",
        device="cpu",
        dtype="float32",
        kernel="framework_dispatch",
        requires=("paddle",),
        description="PaddlePaddle fp32 through tensorlbm.backends",
    ),
    "mindspore_fp32": BackendProfile(
        name="mindspore_fp32",
        framework="mindspore",
        device="cpu",
        dtype="float32",
        kernel="framework_dispatch",
        requires=("mindspore",),
        description="MindSpore fp32 through tensorlbm.backends",
    ),
}

GOLDEN_PROFILE = "cpu_fp64_serial"

#: Tiers whose L0 kernel lives in this module (others are refusal-only probes).
KERNEL_TIERS = (
    "cpu_fp64_serial",
    "cpu_fp64_numpy_tree",
    "cpu_fp64_torch",
    "cpu_fp64_torch_gemm",
    "cpu_fp32_torch",
    "cpu_fp64_shipped_w",
    "sdaa_fp32_torch",
    "sdaa_fp64_torch",
)


class RefusalLedger:
    """Append-only record of refused (unavailable / under-precision) tiers."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, profile: str, kind: str, reason: str, **extra: Any) -> dict[str, Any]:
        entry = {
            "profile": profile,
            "kind": kind,  # "unavailable" | "precision" | "uncertified"
            "reason": reason,
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **extra,
        }
        self.records.append(entry)
        return entry

    def as_list(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.records]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.records)


_DEFAULT_LEDGER = RefusalLedger()


def default_ledger() -> RefusalLedger:
    """Process-wide refusal ledger (dumped into the ladder report)."""
    return _DEFAULT_LEDGER


def _torch():
    import torch

    return torch


def _probe_torch_device(prof: BackendProfile) -> dict[str, Any]:
    try:
        torch = _torch()
    except Exception as exc:  # noqa: BLE001 - any import failure = unavailable
        return {"available": False, "reason": f"torch not importable: {exc}"}
    if prof.device.startswith("cuda"):
        if not torch.cuda.is_available():
            return {"available": False, "reason": "torch.cuda.is_available() is False"}
        if prof.dtype == "float64":
            major, _minor = torch.cuda.get_device_capability()
            if major < 6:
                return {
                    "available": False,
                    "reason": f"device compute capability {major}.x lacks usable fp64",
                }
        if prof.framework == "triton":
            try:
                import triton  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                return {"available": False, "reason": f"triton not importable: {exc}"}
        return {"available": True, "reason": "cuda device present"}
    if prof.device.startswith("sdaa"):
        sdaa = getattr(torch, "sdaa", None)
        if sdaa is None or not sdaa.is_available():
            return {"available": False, "reason": "torch.sdaa.is_available() is False/absent"}
        return {"available": True, "reason": "sdaa device present"}
    if prof.device == "cpu":
        return {"available": True, "reason": "cpu"}
    return {"available": False, "reason": f"unknown device {prof.device!r}"}


def probe_profile(profile: str | BackendProfile) -> dict[str, Any]:
    """Return ``{"available": bool, "reason": str}`` for a tier (never raises)."""
    if isinstance(profile, BackendProfile):
        prof = profile
    else:
        if profile not in PROFILES:
            raise KeyError(f"unknown backend profile {profile!r}; known: {sorted(PROFILES)}")
        prof = PROFILES[profile]
    if prof.framework in {"paddle", "mindspore"}:
        try:
            __import__(prof.framework)
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "reason": f"module '{prof.framework}' not importable: {exc}"}
        return {"available": True, "reason": "module present"}
    if prof.framework in {"torch", "triton"}:
        return _probe_torch_device(prof)
    if prof.framework == "numpy":
        return {"available": True, "reason": "numpy"}
    return {"available": False, "reason": f"framework {prof.framework!r} has no probe"}


def resolve_profile(
    profile: str | BackendProfile,
    *,
    ledger: RefusalLedger | None = None,
    require_dtype: str | None = None,
) -> BackendProfile:
    """Resolve a tier or **fail closed**.

    Raises :class:`KeyError` for an unknown name (typo protection),
    :class:`BackendUnavailable` when the tier is unusable on this host and
    :class:`PrecisionInsufficient` when ``require_dtype`` is not met.  The
    refusal is recorded before raising: no silent CPU fallback, no silent
    precision downgrade.
    """
    if isinstance(profile, BackendProfile):
        prof = profile
    else:
        if profile not in PROFILES:
            raise KeyError(f"unknown backend profile {profile!r}; known: {sorted(PROFILES)}")
        prof = PROFILES[profile]
    led = ledger if ledger is not None else _DEFAULT_LEDGER
    probe = probe_profile(prof)
    if not probe["available"]:
        led.record(prof.name, "unavailable", probe["reason"], device=prof.device, dtype=prof.dtype)
        raise BackendUnavailable(f"{prof.name}: {probe['reason']}")
    if require_dtype is not None:
        if require_dtype not in PRECISION_RANK:
            raise KeyError(f"unknown dtype {require_dtype!r}")
        if prof.precision_rank < PRECISION_RANK[require_dtype]:
            reason = (
                f"{prof.name} runs in {prof.dtype} but the caller requires {require_dtype} "
                "(refusing a silent precision downgrade)"
            )
            led.record(prof.name, "precision", reason, device=prof.device, dtype=prof.dtype)
            raise PrecisionInsufficient(reason)
    return prof


def require_precision(
    profile: str | BackendProfile, dtype: str, *, ledger: RefusalLedger | None = None
) -> BackendProfile:
    """Fail-closed precision assertion."""
    return resolve_profile(profile, ledger=ledger, require_dtype=dtype)


# ---------------------------------------------------------------------------
# D3Q19 periodic-box kernel: golden (NumPy, ordered) + torch tiers
# ---------------------------------------------------------------------------
_CN = np.asarray(C_TORCH.cpu().numpy(), dtype=np.float64)  # (19, 3)
_OPPN = np.asarray(OPPOSITE_TORCH.cpu().numpy(), dtype=np.int64)


def exact_d3q19_weights() -> np.ndarray:
    """D3Q19 weights as exact rationals (1/3, 1/18, 1/36) in ``C`` order.

    ``tensorlbm.d3q19.W`` stores the weights as a **float32** tensor; the
    float32-rounded rationals sum to ``1 + 1.49e-8`` instead of 1, so the
    equilibrium carries a 1.5e-8 relative *mass source per step*.  That is of no
    consequence for a 1e-2-accurate velocity field but it makes a strict
    conservation gate (``L2``, 1e-12/step) unreachable, so the equivalence kernel
    uses the exact rationals.  :func:`weight_table_audit` quantifies the leak.
    """
    w = np.empty(19, dtype=np.float64)
    for i in range(19):
        n_nonzero = int(np.count_nonzero(_CN[i]))
        w[i] = {0: 1.0 / 3.0, 1: 1.0 / 18.0, 2: 1.0 / 36.0}[n_nonzero]
    return w


_WN = exact_d3q19_weights()
#: The table shipped in ``tensorlbm/d3q19.py`` (4 decimals, float32) - used only
#: by the negative-control tier that must fail the L0/L2 gates.
_WSHIPPED = np.asarray(W_TORCH.cpu().numpy(), dtype=np.float64)


def weight_table_audit() -> dict[str, Any]:
    """Quantify the truncated-weight-table defect of ``tensorlbm.d3q19.W``."""
    shipped = np.asarray(W_TORCH.cpu().numpy(), dtype=np.float64)
    exact = _WN
    return {
        "shipped_sum": float(shipped.sum()),
        "exact_sum": float(exact.sum()),
        "excess_sum": float(shipped.sum() - 1.0),
        "max_abs_weight_error": float(np.max(np.abs(shipped - exact))),
        "relative_mass_source_per_step": float(shipped.sum() - 1.0),
        "shipped_dtype": str(W_TORCH.dtype).replace("torch.", ""),
        "note": (
            "tensorlbm/d3q19.py W is a float32 weight table whose sum is "
            "1 + 1.49e-8, i.e. ~1.5e-8 relative mass source per step; that "
            "blocks any 1e-12 conservation gate.  The equivalence kernel uses "
            "exact rationals; the production table is left untouched (changing "
            "it would shift every validated result) but flagged in the report."
        ),
    }


def taylor_green_state(n: int, u0: float, dtype: Any) -> tuple[np.ndarray, ...]:
    """Divergence-free Taylor-Green vortex on a periodic ``n^3`` box (lattice units)."""
    x = 2.0 * np.pi * (np.arange(n, dtype=np.float64) + 0.5) / n
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    ux = u0 * np.sin(xx) * np.cos(yy) * np.cos(zz)
    uy = -u0 * np.cos(xx) * np.sin(yy) * np.cos(zz)
    uz = np.zeros_like(ux)
    rho = np.ones_like(ux)
    dt = np.dtype(dtype)
    if dt == np.float32:
        ux, uy, uz, rho = (a.astype(dt) for a in (ux, uy, uz, rho))
    return rho, ux, uy, uz


def _feq_serial(rho, ux, uy, uz, dtype) -> np.ndarray:
    """Second-order equilibrium, per-direction ordered accumulation."""
    feq = np.zeros((19,) + rho.shape, dtype=dtype)
    usq = ux * ux + uy * uy + uz * uz
    for i in range(19):
        cu = _CN[i, 0] * ux + _CN[i, 1] * uy + _CN[i, 2] * uz
        feq[i] = _WN[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    return feq


def _moments_serial(f: np.ndarray):
    """Moments through a BLAS-style contraction with the lattice vectors.

    Rounding order of this route differs from :func:`_moments_tree` (pairwise
    ``add.reduce``); both are the *same* algebra.  A per-direction Python loop is
    deliberately avoided: it is ~6x slower at 64**3 and adds nothing to the
    reference value.
    """
    rho = f.sum(axis=0)
    mom = np.tensordot(_CN.T, f, axes=(1, 0))  # (3, n, n, n)
    return rho, mom[0] / rho, mom[1] / rho, mom[2] / rho


_STREAM_PLAN_CACHE: dict[int, list] = {}


def _stream_plan(n: int) -> list[list[tuple[tuple[slice, ...], tuple[slice, ...]]]]:
    """Per-direction ``(destination, source)`` slices for ``out[x] = f[x - c]``.

    Layout of the population array is ``(19, n_x, n_y, n_z)``.  Slice copies are
    ~2x faster than ``np.roll`` and launch one copy per (direction, axis) piece
    instead of a concatenate-based roll.
    """
    if n in _STREAM_PLAN_CACHE:
        return _STREAM_PLAN_CACHE[n]
    plan = []
    for i in range(19):
        axes = []
        for comp in (int(_CN[i, 0]), int(_CN[i, 1]), int(_CN[i, 2])):
            if comp == 0:
                axes.append([(slice(None), slice(None))])
            else:
                p = comp % n
                axes.append(
                    [
                        (slice(p, None), slice(None, n - p)),
                        (slice(None, p), slice(n - p, None)),
                    ]
                )
        combos = [()]
        for options in axes:
            combos = [c + (opt,) for c in combos for opt in options]
        plan.append(
            [
                (tuple(c[a][0] for a in range(3)), tuple(c[a][1] for a in range(3)))
                for c in combos
            ]
        )
    _STREAM_PLAN_CACHE[n] = plan
    return plan


def _stream_numpy(f: np.ndarray) -> np.ndarray:
    """Streaming ``f_new(x) = f_post(x - c)`` (physical upwind convention)."""
    n = f.shape[1]
    plan = _stream_plan(n)
    out = np.empty_like(f)
    for i in range(19):
        fi = f[i]
        oi = out[i]
        for dst, src in plan[i]:
            oi[dst] = fi[src]
    return out


def _moments_tree(f: np.ndarray):
    """Moments via pairwise (tree) reductions: same algebra, different order."""
    rho = np.add.reduce(f, axis=0)
    px = np.multiply(f, _CN[:, 0].reshape(19, 1, 1, 1)).sum(axis=0)
    py = np.multiply(f, _CN[:, 1].reshape(19, 1, 1, 1)).sum(axis=0)
    pz = np.multiply(f, _CN[:, 2].reshape(19, 1, 1, 1)).sum(axis=0)
    return rho, px / rho, py / rho, pz / rho


def _feq_tree(rho, ux, uy, uz, dtype, wvec=None) -> np.ndarray:
    """Vectorised equilibrium, Horner form ``1 + cu*(3 + 4.5 cu) - 1.5 usq``."""
    usq = ux * ux + uy * uy + uz * uz
    cu = (
        _CN[:, 0].reshape(19, 1, 1, 1) * ux
        + _CN[:, 1].reshape(19, 1, 1, 1) * uy
        + _CN[:, 2].reshape(19, 1, 1, 1) * uz
    )
    w = _WN.reshape(19, 1, 1, 1) if wvec is None else wvec.reshape(19, 1, 1, 1)
    return (w * rho) * (1.0 + cu * (3.0 + 4.5 * cu) - 1.5 * usq)


def _field_digest(rho, ux, uy, uz) -> str:
    h = hashlib.sha256()
    for arr in (rho, ux, uy, uz):
        h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
    return h.hexdigest()


def _trace_point(step: int, rho, ux, uy, uz) -> dict[str, Any]:
    r64 = np.asarray(rho, dtype=np.float64)
    ux64 = np.asarray(ux, dtype=np.float64)
    uy64 = np.asarray(uy, dtype=np.float64)
    uz64 = np.asarray(uz, dtype=np.float64)
    speed = np.sqrt(ux64 * ux64 + uy64 * uy64 + uz64 * uz64)
    return {
        "step": step,
        "mass": float(r64.sum()),
        "px": float((r64 * ux64).sum()),
        "py": float((r64 * uy64).sum()),
        "pz": float((r64 * uz64).sum()),
        "kinetic_energy": float((0.5 * r64 * speed * speed).sum()),
        "rho_min": float(r64.min()),
        "rho_max": float(r64.max()),
        "u_max": float(speed.max()),
        "digest": _field_digest(rho, ux, uy, uz),
    }


def l0_run(
    profile: str | BackendProfile,
    *,
    n: int = 64,
    steps: int = 200,
    u0: float = 0.05,
    nu: float = 0.003,
    seed_ulp: float = 0.0,
    ledger: RefusalLedger | None = None,
    keep_fields: bool = True,
    trace_every: int = 1,
) -> dict[str, Any]:
    """Run the L0 periodic-box experiment on one tier; return trace + timing.

    ``seed_ulp`` > 0 multiplies ``u_x`` by ``1 + seed_ulp`` before the first step
    (used to measure the trajectory amplification of a one-ULP seed).
    ``trace_every`` thins the digest chain for very long runs (default 1 = every
    step, which is what the L0 gate uses).
    """
    prof = resolve_profile(profile, ledger=ledger)
    if prof.name not in KERNEL_TIERS:
        raise BackendUnavailable(
            f"{prof.name}: no L0 kernel in this module (refusal-only probe tier)"
        )
    np_dtype, torch_dtype, _eps = _DTYPES[prof.dtype]
    dtype = np.dtype(np_dtype)

    rho, ux, uy, uz = taylor_green_state(n, u0, dtype)
    if seed_ulp:
        ux = (ux.astype(np.float64) * (1.0 + seed_ulp)).astype(dtype)

    omega = 1.0 / (0.5 + 3.0 * nu)
    trace: list[dict[str, Any]] = []
    digest_chain = hashlib.sha256()
    cells = n * n * n
    last_fields: dict[str, np.ndarray] = {}

    if prof.framework == "numpy":
        tree = prof.kernel == "tree_reduction"
        _mom = _moments_tree if tree else _moments_serial
        wvec = _WSHIPPED if prof.kernel == "shipped_weights" else _WN
        f = _feq_tree(rho, ux, uy, uz, dtype, wvec)
        t0 = time.perf_counter()
        for step in range(1, steps + 1):
            rho, ux, uy, uz = _mom(f)
            feq = _feq_tree(rho, ux, uy, uz, dtype, wvec)
            f += omega * (feq - f)
            f = _stream_numpy(f)
            if step % trace_every == 0 or step == steps:
                tp = _trace_point(step, rho, ux, uy, uz)
                trace.append(tp)
                digest_chain.update(tp["digest"].encode())
        elapsed = time.perf_counter() - t0
        last_fields = {"rho": rho, "ux": ux, "uy": uy, "uz": uz}
    else:
        torch = _torch()
        dev = torch.device(prof.device)
        td = getattr(torch, torch_dtype)
        c = torch.from_numpy(_CN).to(dev).to(td)  # (19,3)
        w = torch.from_numpy(_WN).to(dev).to(td)  # (19,)
        cvec = c.view(19, 3, 1, 1, 1)
        wvec = w.view(19, 1, 1, 1)
        f = torch.from_numpy(np.ascontiguousarray(_feq_tree(rho, ux, uy, uz, dtype))).to(dev).to(td)
        A = torch.from_numpy(np.vstack([np.ones(19), _CN.T]).astype(np.float64)).to(dev).to(td)
        gemm = prof.kernel == "gemm_moments"
        t0 = time.perf_counter()
        for step in range(1, steps + 1):
            if gemm:
                m = torch.mm(A, f.reshape(19, cells))
                rho_t, ux_t, uy_t, uz_t = m[0], m[1] / m[0], m[2] / m[0], m[3] / m[0]
            else:
                rho_t = f.sum(dim=0)
                ux_t = (f * c[:, 0].view(19, 1, 1, 1)).sum(dim=0) / rho_t
                uy_t = (f * c[:, 1].view(19, 1, 1, 1)).sum(dim=0) / rho_t
                uz_t = (f * c[:, 2].view(19, 1, 1, 1)).sum(dim=0) / rho_t
            rho_m, ux_m, uy_m, uz_m = (a.reshape(n, n, n) for a in (rho_t, ux_t, uy_t, uz_t))
            usq = ux_m * ux_m + uy_m * uy_m + uz_m * uz_m
            ustack = torch.stack([ux_m, uy_m, uz_m]).view(1, 3, n, n, n)
            cu = (
                cvec[:, 0] * ux_m.view(1, n, n, n)
                + cvec[:, 1] * uy_m.view(1, n, n, n)
                + cvec[:, 2] * uz_m.view(1, n, n, n)
            )
            ustack = None
            feq = wvec * rho_m.view(1, n, n, n) * (
                1.0 + cu * (3.0 + 4.5 * cu) - 1.5 * usq.view(1, n, n, n)
            )
            f = f + omega * (feq - f)
            plan = _stream_plan(n)
            out = torch.empty_like(f)
            for i in range(19):
                fi = f[i]
                oi = out[i]
                for dst, src in plan[i]:
                    oi[dst] = fi[src]
            f = out
            if step % trace_every == 0 or step == steps:
                tp = _trace_point(
                    step,
                    rho_m.detach().cpu().numpy(),
                    ux_m.detach().cpu().numpy(),
                    uy_m.detach().cpu().numpy(),
                    uz_m.detach().cpu().numpy(),
                )
                trace.append(tp)
                digest_chain.update(tp["digest"].encode())
        elapsed = time.perf_counter() - t0
        last_fields = {
            "rho": rho_m.detach().cpu().numpy(),
            "ux": ux_m.detach().cpu().numpy(),
            "uy": uy_m.detach().cpu().numpy(),
            "uz": uz_m.detach().cpu().numpy(),
        }

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "level": "L0",
        "profile": asdict(prof),
        "grid": [n, n, n],
        "steps": steps,
        "cells": cells,
        "u0": u0,
        "nu": nu,
        "tau": 0.5 + 3.0 * nu,
        "seed_ulp": seed_ulp,
        "trace_every": trace_every,
        "trace": trace,
        "digest_chain": digest_chain.hexdigest(),
        "elapsed_seconds": elapsed,
        "mcell_steps_per_second": cells * steps / max(elapsed, 1e-12) / 1e6,
        "seconds_per_step": elapsed / max(steps, 1),
        "final_digest": trace[-1]["digest"],
    }
    if keep_fields:
        result["final_fields_sha256"] = _field_digest(
            last_fields["rho"], last_fields["ux"], last_fields["uy"], last_fields["uz"]
        )
        result["_fields"] = last_fields  # popped by callers before JSON dumping
    return result


# ---------------------------------------------------------------------------
# L0 comparators
# ---------------------------------------------------------------------------
def ulp_distance(a, b) -> float:
    """max |a-b| expressed in units of the fp64 spacing of the reference."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    diff = np.abs(a64 - b64)
    scale = np.maximum(np.abs(a64), np.finfo(np.float64).tiny)
    return float(np.max(diff / np.spacing(scale)))


def ulp_budget(amplification: float, eps: float, n_ops: int = N_OPS_PER_STEP) -> float:
    """A-priori rounding budget in relative field units.

    ``A_N`` (measured amplification of a one-ULP seed) x per-step rounding budget
    ``n_ops * eps``.  A deviation above it cannot be explained by rounding alone.
    """
    return float(amplification * n_ops * eps)


def check_l0(
    golden: dict[str, Any],
    candidate: dict[str, Any],
    *,
    amplification: float | None = None,
    tolerance_factor: float = 1.0,
) -> dict[str, Any]:
    """Compare a candidate L0 run against the frozen golden run.

    Certified when the digest chain is bitwise identical **or** the measured
    relative deviation stays within the rounding budget.
    """
    g = np.asarray([t["u_max"] for t in golden["trace"]], dtype=np.float64)
    c = np.asarray([t["u_max"] for t in candidate["trace"]], dtype=np.float64)
    scale = float(np.max(np.abs(g))) or 1.0
    dev_trace = np.abs(c - g) / scale
    dev = float(dev_trace.max())
    bitwise_trace = golden["digest_chain"] == candidate["digest_chain"]
    bitwise_fields = golden.get("final_fields_sha256") == candidate.get("final_fields_sha256")
    bitwise = bool(bitwise_trace and bitwise_fields)

    eps = _DTYPES[candidate["profile"]["dtype"]][2]
    gold_fields = golden.get("_fields")
    cand_fields = candidate.get("_fields")
    max_rel_field = None
    ulps = None
    if gold_fields is not None and cand_fields is not None:
        rel = []
        ulp_vals = []
        for key in ("rho", "ux", "uy", "uz"):
            a = np.asarray(gold_fields[key], dtype=np.float64)
            b = np.asarray(cand_fields[key], dtype=np.float64)
            ref = float(np.max(np.abs(a))) or 1.0
            rel.append(float(np.max(np.abs(b - a)) / ref))
            ulp_vals.append(ulp_distance(a, b))
        max_rel_field = max(rel)
        ulps = max(ulp_vals)

    amp = float(amplification if amplification is not None else 1.0)
    budget = ulp_budget(amp, eps) * tolerance_factor
    within_budget = bool(max_rel_field is not None and max_rel_field <= budget)
    return {
        "level": "L0",
        "candidate": candidate["profile"]["name"],
        "golden": golden["profile"]["name"],
        "dtype": candidate["profile"]["dtype"],
        "bitwise_identical": bitwise,
        "bitwise_trace": bool(bitwise_trace),
        "bitwise_final_fields": bool(bitwise_fields),
        "trace_digest_golden": golden["digest_chain"],
        "trace_digest_candidate": candidate["digest_chain"],
        "max_rel_deviation_trace": dev,
        "max_rel_deviation_field": max_rel_field,
        "max_ulp_distance": ulps,
        "amplification": amp,
        "eps": eps,
        "n_ops_per_step": N_OPS_PER_STEP,
        "ulp_budget_relative": budget,
        "budget_utilisation": (
            None if max_rel_field is None or budget == 0 else max_rel_field / budget
        ),
        "passed": bool(bitwise or within_budget),
        "criterion": "bitwise digest identity OR max_rel_deviation_field <= A_N*N_OPS*eps",
    }


def measure_amplification(
    profile: str | BackendProfile = GOLDEN_PROFILE,
    *,
    n: int = 64,
    steps: int = 200,
    u0: float = 0.05,
    nu: float = 0.003,
    ledger: RefusalLedger | None = None,
) -> dict[str, Any]:
    """Amplification ``A_N`` of a one-ULP seed on the *same* trajectory.

    ``A_N = max|u_pert(N) - u_golden(N)| / max|u_pert(0) - u_golden(0)|`` with
    ``max|u_pert(0) - u_golden(0)| = eps * u0`` (the seed is a multiplicative
    ``1 + eps`` factor on ``u_x``).  ``A_N`` is the growth factor used by the L0
    rounding budget: rounding noise injected at every step is amplified by the
    same dynamics as an explicit seed.
    """
    prof = resolve_profile(profile, ledger=ledger)
    base = l0_run(prof.name, n=n, steps=steps, u0=u0, nu=nu, ledger=ledger)
    pert = l0_run(
        prof.name, n=n, steps=steps, u0=u0, nu=nu, seed_ulp=prof.eps, ledger=ledger
    )
    d0 = prof.eps * u0
    fields = []
    for key in ("rho", "ux", "uy", "uz"):
        a = np.asarray(base["_fields"][key], dtype=np.float64)
        b = np.asarray(pert["_fields"][key], dtype=np.float64)
        fields.append(float(np.max(np.abs(b - a))))
    dN = max(fields)
    return {
        "profile": prof.name,
        "dtype": prof.dtype,
        "eps": prof.eps,
        "seed_delta0": d0,
        "deltaN": dN,
        "amplification": dN / d0 if d0 > 0 else float("nan"),
        "steps": steps,
        "grid": [n, n, n],
        "golden_digest_chain": base["digest_chain"],
        "perturbed_digest_chain": pert["digest_chain"],
    }


# ---------------------------------------------------------------------------
# L1: production channel, statistical equivalence
# ---------------------------------------------------------------------------
def l1_run_channel(
    profile: str | BackendProfile,
    *,
    nx: int = 32,
    ny: int = 64,
    nz: int = 32,
    steps: int = 10000,
    warmup_steps: int = 3000,
    sample_interval: int = 5,
    re_tau: float = 180.0,
    u_tau: float = 0.01,
    wall_origin: float = 0.5,
    wall_scheme: str = "interpolated",
    tau_minus_max: float | None = None,
    smagorinsky_cs: float = 0.1,
    seed: int = 20261110,
    ledger: RefusalLedger | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Run the Re_tau = 180 channel on one tier (production solver)."""
    from tensorlbm.turbulence.turbulent_channel3d import (
        TurbulentChannel3D,
        TurbulentChannel3DConfig,
    )

    prof = resolve_profile(profile, ledger=ledger)
    if prof.framework != "torch":
        raise BackendUnavailable(
            f"{prof.name}: the production channel solver is torch-only "
            f"(framework={prof.framework}); no silent cross-framework fallback"
        )
    cfg = TurbulentChannel3DConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        re_tau=re_tau,
        u_tau=u_tau,
        wall_origin=wall_origin,
        steps=steps,
        warmup_steps=warmup_steps,
        sample_interval=sample_interval,
        report_interval=max(steps // 4, 1),
        seed=seed,
        device=prof.device,
        dtype=prof.dtype,
        wall_scheme=wall_scheme,
        force_scheme="trt_split",
        tau_minus_max=tau_minus_max,
        smagorinsky_cs=smagorinsky_cs,
        label=label or prof.name,
    )
    t0 = time.perf_counter()
    result = TurbulentChannel3D(cfg).run()
    elapsed = time.perf_counter() - t0
    cells = cfg.nx * cfg.ny * cfg.nz
    steps_per_turnover = cfg.half_height / cfg.u_tau
    wu = result["wall_units"]
    return {
        "schema": SCHEMA,
        "level": "L1",
        "profile": asdict(prof),
        "config": result["config"],
        "wall_units": {
            "y_plus": wu["y_plus"],
            "u_plus": wu["u_plus"],
            "uu_plus": wu["uu_plus"],
            "vv_plus": wu["vv_plus"],
            "ww_plus": wu["ww_plus"],
            "uv_plus": wu["uv_plus"],
        },
        "reynolds_stress": result["reynolds_stress"],
        "friction_velocity": result["friction_velocity"],
        "momentum_balance": result["momentum_balance"],
        "re_tau_achieved": result["re_tau_achieved"],
        "mean_velocity_bulk": result["mean_velocity_bulk"],
        "samples": result["config"]["samples"],
        "elapsed_seconds": elapsed,
        "mcell_steps_per_second": cells * steps / max(elapsed, 1e-12) / 1e6,
        "seconds_per_step": elapsed / max(steps, 1),
        "steps_per_convective_time": steps_per_turnover,
        "seconds_per_convective_time": steps_per_turnover * elapsed / max(steps, 1),
        "wall_scheme": wall_scheme,
    }


class ChunkedChannelRun:
    """Resumable ``Re_tau = 180`` channel run on one tier.

    The monolithic :meth:`turbulence.turbulent_channel3d.TurbulentChannel3D.run`
    integrates all steps in one call.  Long campaigns are frequently executed
    under a per-command wall-time cap (batch queue, CI job, agent shell), so this
    driver advances the same production solver in chunks and checkpoints the
    population field plus the running statistics.  The physics is untouched: it
    calls ``sim._step_once()`` and reproduces ``run()``'s sampling bookkeeping
    (warm-up flux reset, sample interval, wall-flux accumulation).
    """

    STATE_FILE = "state.pt"
    #: Divergence check cadence.  ``health()`` costs a full-field reduction set;
    #: doing it every step cost ~40 % of the whole integration in the first
    #: version of this driver.
    HEALTH_EVERY = 250

    def __init__(
        self,
        profile: str | BackendProfile,
        *,
        nx: int = 32,
        ny: int = 64,
        nz: int = 32,
        steps: int = 10000,
        warmup_steps: int = 3000,
        sample_interval: int = 5,
        re_tau: float = 180.0,
        u_tau: float = 0.01,
        wall_origin: float = 0.5,
        wall_scheme: str = "interpolated",
        tau_minus_max: float | None = None,
        smagorinsky_cs: float = 0.1,
        seed: int = 20261110,
        state_dir: str | Path | None = None,
        ledger: RefusalLedger | None = None,
        label: str | None = None,
    ) -> None:
        from tensorlbm.turbulence.turbulent_channel3d import (
            TurbulentChannel3D,
            TurbulentChannel3DConfig,
        )

        prof = resolve_profile(profile, ledger=ledger)
        if prof.framework != "torch":
            raise BackendUnavailable(
                f"{prof.name}: the production channel solver is torch-only "
                f"(framework={prof.framework}); no silent cross-framework fallback"
            )
        self.kwargs = dict(
            nx=nx,
            ny=ny,
            nz=nz,
            steps=steps,
            warmup_steps=warmup_steps,
            sample_interval=sample_interval,
            re_tau=re_tau,
            u_tau=u_tau,
            wall_origin=wall_origin,
            wall_scheme=wall_scheme,
            tau_minus_max=tau_minus_max,
            smagorinsky_cs=smagorinsky_cs,
            seed=seed,
        )
        self.prof = prof
        self.cfg = TurbulentChannel3DConfig(
            nx=nx,
            ny=ny,
            nz=nz,
            re_tau=re_tau,
            u_tau=u_tau,
            wall_origin=wall_origin,
            steps=steps,
            warmup_steps=warmup_steps,
            sample_interval=sample_interval,
            report_interval=max(steps // 4, 1),
            seed=seed,
            device=prof.device,
            dtype=prof.dtype,
            wall_scheme=wall_scheme,
            force_scheme="trt_split",
            tau_minus_max=tau_minus_max,
            smagorinsky_cs=smagorinsky_cs,
            label=label or prof.name,
        )
        self.sim = TurbulentChannel3D(self.cfg)
        self.state_dir = Path(state_dir) if state_dir is not None else None
        self.sums = None
        self.samples = 0
        self.wall_flux = 0.0
        self.elapsed = 0.0
        self._init_accumulators()
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self._load_state()

    # -- statistics bookkeeping (mirrors TurbulentChannel3D.run) ------------
    def _init_accumulators(self) -> None:
        import torch

        self.sums = torch.zeros(self.cfg.ny, 8, dtype=torch.float64)
        self.samples = 0
        self.wall_flux = 0.0
        self.elapsed = 0.0

    def _accumulate(self) -> None:
        import torch

        cfg = self.cfg
        nx, nz, N = cfg.nx, cfg.nz, self.sim.N
        f2 = self.sim.f.view(19, N)
        m = torch.mm(self.sim.A, f2)
        rho = m[0]
        ux, uy, uz = m[1] / rho, m[2] / rho, m[3] / rho
        wgt = 1.0 / (nx * nz)
        # the running sums are host-side float64: move the device reductions over
        # once per sample instead of mixing devices (which raises on accelerators)
        def _row(arr):
            return arr.view(nz, cfg.ny, nx).sum(dim=(0, 2)).double().cpu()

        host = [_row(rho), _row(ux), _row(uy), _row(uz),
                _row(ux * ux), _row(uy * uy), _row(uz * uz), _row(ux * uy)]
        for k, value in enumerate(host):
            self.sums[:, k] += value * wgt
        self.samples += 1
        self.wall_flux += self.sim._wall_flux_accum
        self.sim._wall_flux_accum = 0.0

    # -- advancing ---------------------------------------------------------
    def advance(self, chunk: int | None = None) -> dict[str, Any]:
        import time

        import torch

        cfg = self.cfg
        chunk = int(chunk) if chunk else cfg.steps
        chunk = min(chunk, cfg.steps - self.sim.step_count)
        if chunk <= 0:
            return self.status()
        t0 = time.perf_counter()
        for _ in range(chunk):
            self.sim._step_once()
            step = self.sim.step_count
            if step <= cfg.warmup_steps:
                self.sim._wall_flux_accum = 0.0
            if step > cfg.warmup_steps and (step - cfg.warmup_steps) % cfg.sample_interval == 0:
                self._accumulate()
            if step % self.HEALTH_EVERY == 0:
                hmax = float(self.sim.health()["rho_max"])
                if not math.isfinite(hmax):
                    raise FloatingPointError(f"divergence at step {step}")
        self.elapsed += time.perf_counter() - t0
        self.save_state()
        return self.status()

    def status(self) -> dict[str, Any]:
        cfg = self.cfg
        return {
            "profile": self.prof.name,
            "dtype": self.prof.dtype,
            "wall_scheme": cfg.wall_scheme,
            "steps_done": int(self.sim.step_count),
            "steps_total": int(cfg.steps),
            "samples": int(self.samples),
            "elapsed_seconds_this_process": self.elapsed,
            "complete": bool(self.sim.step_count >= cfg.steps),
        }

    # -- checkpointing -----------------------------------------------------
    def save_state(self) -> None:
        if self.state_dir is None:
            return
        import torch

        torch.save(
            {
                "kwargs": self.kwargs,
                "f": self.sim.f,
                "sums": self.sums,
                "samples": self.samples,
                "wall_flux": self.wall_flux,
                "elapsed": self.elapsed,
                "step_count": self.sim.step_count,
            },
            self.state_dir / self.STATE_FILE,
        )

    def _load_state(self) -> None:
        import torch

        path = self.state_dir / self.STATE_FILE
        if not path.exists():
            return
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob["kwargs"] != self.kwargs:
            raise ValueError("checkpoint was written with different run parameters")
        self.sim.f.copy_(blob["f"])
        self.sums = blob["sums"]
        self.samples = int(blob["samples"])
        self.wall_flux = float(blob["wall_flux"])
        self.elapsed = float(blob["elapsed"])
        self.sim.step_count = int(blob["step_count"])

    # -- result ------------------------------------------------------------
    def finalize(self) -> dict[str, Any]:
        cfg = self.cfg
        stats = self.sums / max(self.samples, 1)
        result = self.sim._finalize(stats, self.samples, self.wall_flux, self.elapsed, [])
        cells = cfg.nx * cfg.ny * cfg.nz
        steps_per_turnover = cfg.half_height / cfg.u_tau
        wu = result["wall_units"]
        return {
            "schema": SCHEMA,
            "level": "L1",
            "profile": asdict(self.prof),
            "config": result["config"],
            "wall_units": {
                "y_plus": wu["y_plus"],
                "u_plus": wu["u_plus"],
                "uu_plus": wu["uu_plus"],
                "vv_plus": wu["vv_plus"],
                "ww_plus": wu["ww_plus"],
                "uv_plus": wu["uv_plus"],
            },
            "reynolds_stress": result["reynolds_stress"],
            "friction_velocity": result["friction_velocity"],
            "momentum_balance": result["momentum_balance"],
            "re_tau_achieved": result["re_tau_achieved"],
            "mean_velocity_bulk": result["mean_velocity_bulk"],
            "elapsed_seconds": self.elapsed,
            "mcell_steps_per_second": cells * cfg.steps / max(self.elapsed, 1e-12) / 1e6,
            "seconds_per_step": self.elapsed / max(cfg.steps, 1),
            "steps_per_convective_time": steps_per_turnover,
            "seconds_per_convective_time": steps_per_turnover * self.elapsed / max(cfg.steps, 1),
            "wall_scheme": cfg.wall_scheme,
            "chunked": True,
            "steps_done": int(self.sim.step_count),
            "samples": int(self.samples),
        }


def normalised_profile_deviation(a: dict[str, Any], b: dict[str, Any], *, key: str = "u_plus",
                                 y_plus_min: float = 3.0) -> dict[str, Any]:
    """Normalised L2 deviation of ``key`` between two L1 runs (same grid)."""
    ya = np.asarray(a["wall_units"]["y_plus"], dtype=np.float64)
    yb = np.asarray(b["wall_units"]["y_plus"], dtype=np.float64)
    if ya.shape != yb.shape or not np.allclose(ya, yb, rtol=0, atol=1e-12):
        raise ValueError("profiles are not on the same y+ grid")
    va = np.asarray(a["wall_units"][key], dtype=np.float64)
    vb = np.asarray(b["wall_units"][key], dtype=np.float64)
    sel = ya >= y_plus_min
    if sel.sum() < 2:
        raise ValueError(f"not enough points above y+={y_plus_min}")
    scale = float(np.max(np.abs(va[sel]))) or 1.0
    l2 = float(np.sqrt(np.mean((vb[sel] - va[sel]) ** 2)))
    pointwise = float(np.max(np.abs(vb[sel] - va[sel])))
    return {
        "key": key,
        "y_plus_min": y_plus_min,
        "n_points": int(sel.sum()),
        "scale": scale,
        "normalised_l2": l2 / scale,
        "max_pointwise_relative": pointwise / scale,
    }


def check_l1(
    golden: dict[str, Any],
    candidate: dict[str, Any],
    *,
    mkm: Any | None = None,
    max_rel_deviation: float = L1_MAX_REL_DEVIATION,
    mkm_tol: float = L1_MKM_NOT_WORSE_TOL,
) -> dict[str, Any]:
    """L1 gate: statistical equivalence vs golden + MKM deviation not worse."""
    keys = ("u_plus", "uu_plus", "vv_plus", "ww_plus")
    devs = {k: normalised_profile_deviation(golden, candidate, key=k) for k in keys}
    worst = max(d["normalised_l2"] for d in devs.values())
    equivalent = bool(worst <= max_rel_deviation)

    mkm_block: dict[str, Any] = {"reference": None}
    mkm_ok = True
    if mkm is not None:
        def _mkm_dev(run):
            yp = np.asarray(run["wall_units"]["y_plus"], dtype=np.float64)
            up = np.asarray(run["wall_units"]["u_plus"], dtype=np.float64)
            prof = mkm
            y_ref = np.asarray(prof.y_plus, dtype=np.float64)
            u_ref = np.asarray(prof.u_plus, dtype=np.float64)
            sel = (y_ref >= 3.0) & (y_ref <= y_ref.max())
            interp = np.interp(y_ref[sel], yp, up)
            return float(np.sqrt(np.mean((interp - u_ref[sel]) ** 2)) / np.max(u_ref[sel]))

        dev_gold = _mkm_dev(golden)
        dev_cand = _mkm_dev(candidate)
        mkm_ok = bool(dev_cand <= dev_gold * (1.0 + mkm_tol))
        mkm_block = {
            "reference": "MKM1999",
            "re_tau": float(getattr(mkm, "re_tau", float("nan"))),
            "normalised_l2_deviation_golden": dev_gold,
            "normalised_l2_deviation_candidate": dev_cand,
            "not_worse": mkm_ok,
            "tolerance": mkm_tol,
        }
    return {
        "level": "L1",
        "candidate": candidate["profile"]["name"],
        "golden": golden["profile"]["name"],
        "dtype": candidate["profile"]["dtype"],
        "wall_scheme": candidate.get("wall_scheme"),
        "deviations": devs,
        "worst_normalised_l2": worst,
        "threshold": max_rel_deviation,
        "equivalent": equivalent,
        "mkm": mkm_block,
        "passed": bool(equivalent and mkm_ok),
        "criterion": (
            "normalised profile deviation (U+, uu, vv, ww) <= 2% AND MKM-1999 "
            "deviation not worse than golden"
        ),
    }


# ---------------------------------------------------------------------------
# L2: conservation invariants
# ---------------------------------------------------------------------------
def check_l2(run: dict[str, Any], *, max_drift: float = L2_DRIFT_MAX) -> dict[str, Any]:
    """Mass/momentum conservation drift per step per cell from an L0 trace."""
    trace = run["trace"]
    n_cells = float(run["cells"])
    n_steps = float(run["steps"])
    mass = np.asarray([t["mass"] for t in trace], dtype=np.float64)
    px = np.asarray([t["px"] for t in trace], dtype=np.float64)
    py = np.asarray([t["py"] for t in trace], dtype=np.float64)
    pz = np.asarray([t["pz"] for t in trace], dtype=np.float64)

    def _drift(series: np.ndarray) -> dict[str, float]:
        span = float(np.ptp(series))
        per_step = span / (n_steps * n_cells)
        return {
            "max_abs_deviation": span,
            "drift_per_step_per_cell": per_step,
            "relative_per_step": span / (max(abs(float(series[0])), 1e-30) * n_steps),
        }

    mass_d = _drift(mass)
    mom_span = float(
        max(np.ptp(px), np.ptp(py), np.ptp(pz))
    )
    mom_d = {
        "max_abs_deviation": mom_span,
        "drift_per_step_per_cell": mom_span / (n_steps * n_cells),
    }
    ok = bool(
        mass_d["drift_per_step_per_cell"] <= max_drift
        and mom_d["drift_per_step_per_cell"] <= max_drift
    )
    return {
        "level": "L2",
        "profile": run["profile"]["name"],
        "dtype": run["profile"]["dtype"],
        "cells": int(n_cells),
        "steps": int(n_steps),
        "mass": mass_d,
        "momentum": mom_d,
        "threshold_drift_per_step_per_cell": max_drift,
        "passed": ok,
        "criterion": "mass and momentum drift per step per cell <= 1e-12",
    }


# ---------------------------------------------------------------------------
# cost / accuracy ledger + certification matrix
# ---------------------------------------------------------------------------
def pareto_front(points: list[dict[str, Any]], *, cost_key: str = "cost", error_key: str = "error") -> list[str]:
    """Names on the (min error, min cost) Pareto frontier; sorted by cost."""
    pts = [p for p in points if p.get(error_key) is not None and p.get(cost_key)]
    pts.sort(key=lambda p: p[cost_key])
    front: list[str] = []
    best = math.inf
    for p in pts:
        if p[error_key] < best:
            front.append(p["name"])
            best = p[error_key]
    return front


def certify(checks: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Collapse per-level check dicts into ``{level: "certified"|"failed"}``."""
    out = {}
    for level, payload in checks.items():
        if payload is None:
            out[level] = "not_evaluated"
        else:
            out[level] = "certified" if payload.get("passed") else "failed"
    return out


def require_certified(
    profile: str,
    certificates: dict[str, dict[str, str]],
    level: str,
    *,
    ledger: RefusalLedger | None = None,
) -> str:
    """Fail-closed use gate: raise if ``profile`` is not certified at ``level``."""
    status = certificates.get(profile, {}).get(level, "not_evaluated")
    if status != "certified":
        led = ledger if ledger is not None else _DEFAULT_LEDGER
        reason = f"{profile} is {status} at equivalence level {level}"
        led.record(profile, "uncertified", reason, level=level)
        raise CertificationError(reason)
    return status


# ---------------------------------------------------------------------------
# golden freezing / aggregation
# ---------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def freeze_golden(run: dict[str, Any], out_dir: str | Path, *, name: str = "l0_golden") -> dict[str, Any]:
    """Freeze an L0 run: step trace to ``.npz`` + SHA-256 manifest (audit chain)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace = run["trace"]
    arrays = {
        "step": np.asarray([t["step"] for t in trace], dtype=np.int64),
        "mass": np.asarray([t["mass"] for t in trace], dtype=np.float64),
        "px": np.asarray([t["px"] for t in trace], dtype=np.float64),
        "py": np.asarray([t["py"] for t in trace], dtype=np.float64),
        "pz": np.asarray([t["pz"] for t in trace], dtype=np.float64),
        "kinetic_energy": np.asarray([t["kinetic_energy"] for t in trace], dtype=np.float64),
        "u_max": np.asarray([t["u_max"] for t in trace], dtype=np.float64),
        "rho_min": np.asarray([tr["rho_min"] for tr in trace], dtype=np.float64),
        "rho_max": np.asarray([tr["rho_max"] for tr in trace], dtype=np.float64),
    }
    npz = out / f"{name}.npz"
    np.savez(npz, **arrays)
    npz_sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    manifest = {
        "schema": SCHEMA,
        "name": name,
        "profile": run["profile"],
        "grid": run["grid"],
        "steps": run["steps"],
        "u0": run["u0"],
        "nu": run["nu"],
        "tau": run["tau"],
        "trace_digest_chain": run["digest_chain"],
        "final_fields_sha256": run.get("final_fields_sha256"),
        "final_field_digest": run["final_digest"],
        "per_step_digests": [t["digest"] for t in trace],
        "npz_path": str(npz),
        "npz_sha256": npz_sha,
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def aggregate_ladder(ledger: RefusalLedger | None = None) -> dict[str, Any]:
    """Probe every registered tier and record refusals (fail-closed inventory)."""
    led = ledger if ledger is not None else _DEFAULT_LEDGER
    inventory = []
    for name, prof in PROFILES.items():
        probe = probe_profile(prof)
        entry = {"profile": name, **asdict(prof), **probe}
        if not probe["available"]:
            led.record(name, "unavailable", probe["reason"], device=prof.device, dtype=prof.dtype)
            entry["certified_levels"] = {}
        inventory.append(entry)
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inventory": inventory,
        "refusals": led.as_list(),
    }