"""Host accelerator capability probing with degrade-and-warn semantics.

TensorLBM treats ``torch`` itself as the portability contract: wherever a
torch build runs (CUDA, Ascend NPU, Cambricon MLU, Hygon SDAA, Moore Threads
MUSA, Apple MPS, plain CPU), the eager solver path follows.  Everything on
top of that (Triton fused kernels, NCCL multi-GPU) is an *acceleration
layer*, not a requirement.

This module implements the FluidX3D-style pattern of *probe features first,
then degrade with a clear warning*: a :func:`probe` call discovers what the
host actually supports (import-probing vendor plugin modules rather than
trusting a hardcoded device list), :func:`require` turns a missing
capability into an actionable error with a degradation suggestion, and the
serialisable :class:`HardwareProfile` plugs into benchmark observability
records so every run documents the hardware it actually used.

The probe is deliberately cheap and side-effect free: unknown vendor
plugins are only ``importlib``-probed, never assumed.
"""

from __future__ import annotations

import importlib
import importlib.util
import platform
import time
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "BackendInfo",
    "CollectiveInfo",
    "HardwareCapabilityError",
    "HardwareProfile",
    "DEGRADATION_ADVICE",
    "describe_degradation",
    "probe",
    "require",
]

# ---------------------------------------------------------------------------
# Vendor plugin registry (import-probed, never hardcoded as "available")
# ---------------------------------------------------------------------------

#: torch plugin accelerators, in default-device priority order.  Each entry
#: maps a torch device type to the pip package(s) that register it.
_PLUGIN_ACCELERATORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sdaa", ("torch_sdaa",)),  # Hygon / LoongArch DCU-class
    ("npu", ("torch_npu",)),  # Huawei Ascend (HCCL ecosystem)
    ("mlu", ("torch_mlu",)),  # Cambricon
    ("musa", ("torch_musa",)),  # Moore Threads
)

#: Built-in (non-plugin) accelerator device types.
_BUILTIN_ACCELERATORS: tuple[str, ...] = ("cuda", "mps")

_KNOWN_BACKENDS = ("cpu",) + tuple(name for name, _ in _PLUGIN_ACCELERATORS) + _BUILTIN_ACCELERATORS

_KNOWN_COLLECTIVES = ("nccl", "gloo", "mpi", "hccl", "rccl")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HardwareCapabilityError(RuntimeError):
    """A requested capability is absent on this host, with a degradation hint.

    Attributes:
        capability: The capability token that was requested.
        advice: A human-readable degradation suggestion (also part of the
            message).
    """

    def __init__(self, capability: str, message: str, advice: str | None = None) -> None:
        self.capability = capability
        self.advice = advice or describe_degradation(capability)
        full = f"{message} Degradation advice: {self.advice}"
        super().__init__(full)


# ---------------------------------------------------------------------------
# Probe result datatypes (plain, serialisable data)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """One torch device backend and whether this host can actually use it."""

    name: str
    available: bool
    device_count: int
    plugin: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "device_count": self.device_count,
            "plugin": self.plugin,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class CollectiveInfo:
    """A distributed collectives library usable by torch.distributed."""

    name: str
    available: bool
    via: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "available": self.available, "via": self.via}


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Snapshot of host accelerator capabilities (JSON-safe via to_dict)."""

    backends: tuple[BackendInfo, ...]
    collectives: tuple[CollectiveInfo, ...]
    fp16_storage: bool
    bf16_storage: bool
    triton_available: bool
    triton_version: str | None
    default_device: str
    torch_version: str
    python_version: str
    hostname: str
    probed_at_unix: float = field(default_factory=time.time)

    # -- accessors ---------------------------------------------------------

    def backend(self, name: str) -> BackendInfo | None:
        for info in self.backends:
            if info.name == name:
                return info
        return None

    def has_backend(self, name: str) -> bool:
        info = self.backend(name)
        return bool(info and info.available and info.device_count > 0) or name == "cpu"

    @property
    def available_backends(self) -> tuple[str, ...]:
        return tuple(info.name for info in self.backends if info.available)

    def collective(self, name: str) -> CollectiveInfo | None:
        for info in self.collectives:
            if info.name == name:
                return info
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backends": [info.to_dict() for info in self.backends],
            "collectives": [info.to_dict() for info in self.collectives],
            "fp16_storage": self.fp16_storage,
            "bf16_storage": self.bf16_storage,
            "triton_available": self.triton_available,
            "triton_version": self.triton_version,
            "default_device": self.default_device,
            "torch_version": self.torch_version,
            "python_version": self.python_version,
            "hostname": self.hostname,
            "probed_at_unix": self.probed_at_unix,
        }


# ---------------------------------------------------------------------------
# Individual probes (each must never raise)
# ---------------------------------------------------------------------------


def _probe_plugin_backend(name: str, plugins: tuple[str, ...]) -> BackendInfo:
    plugin = None
    for candidate in plugins:
        if importlib.util.find_spec(candidate) is not None:
            plugin = candidate
            break
    if plugin is None:
        return BackendInfo(name, False, 0, None, f"no plugin module among {list(plugins)}")
    try:
        importlib.import_module(plugin)
    except Exception as error:  # pragma: no cover - environment specific
        return BackendInfo(name, False, 0, plugin, f"plugin import failed: {error}")
    return _probe_torch_attr_backend(name, plugin)


def _probe_torch_attr_backend(name: str, plugin: str | None) -> BackendInfo:
    handle = getattr(torch, name, None)
    if handle is None:
        return BackendInfo(name, False, 0, plugin, f"torch.{name} not registered")
    try:
        if not bool(handle.is_available()):
            return BackendInfo(name, False, 0, plugin, "is_available() is False")
        try:
            count = int(handle.device_count())
        except Exception:
            count = 1
        return BackendInfo(name, True, count, plugin, None)
    except Exception as error:  # pragma: no cover - vendor specific
        return BackendInfo(name, False, 0, plugin, f"availability probe failed: {error}")


def _probe_cuda() -> BackendInfo:
    return _probe_torch_attr_backend("cuda", None)


def _probe_mps() -> BackendInfo:
    try:
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and bool(mps.is_available()):
            try:
                count = 1
                if hasattr(torch, "mps") and hasattr(torch.mps, "device_count"):
                    count = int(torch.mps.device_count())
            except Exception:
                count = 1
            return BackendInfo("mps", True, count, None, None)
        return BackendInfo("mps", False, 0, None, "torch.backends.mps unavailable")
    except Exception as error:  # pragma: no cover
        return BackendInfo("mps", False, 0, None, f"probe failed: {error}")


def _probe_collectives() -> tuple[CollectiveInfo, ...]:
    results: list[CollectiveInfo] = []
    distributed = getattr(torch, "distributed", None)
    for name in ("nccl", "gloo", "mpi"):
        try:
            checker = getattr(distributed, f"is_{name}_available", None) if distributed else None
            results.append(CollectiveInfo(name, bool(checker and checker()), "torch.distributed"))
        except Exception:  # pragma: no cover
            results.append(CollectiveInfo(name, False, "torch.distributed"))
    # Vendor collectives ship inside the plugin, not torch.distributed.
    results.append(CollectiveInfo("hccl", _plugin_importable("torch_npu"), "torch_npu plugin"))
    results.append(CollectiveInfo("rccl", _plugin_importable("torch_musa"), "torch_musa plugin"))
    return tuple(results)


def _plugin_importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:  # pragma: no cover
        return False


def _dtype_storage_supported(dtype: torch.dtype, device: torch.device) -> bool:
    """Allocate + compute with *dtype* on *device*; False on any failure."""
    try:
        probe = torch.zeros(4, dtype=dtype, device=device)
        probe = probe + 1.0
        return bool((probe == 1).all().item())
    except Exception:
        return False


def _probe_triton() -> tuple[bool, str | None]:
    if importlib.util.find_spec("triton") is None:
        return False, None
    try:
        triton = importlib.import_module("triton")
        version = getattr(triton, "__version__", None)
        return True, str(version) if version is not None else None
    except Exception:  # pragma: no cover
        return False, None


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------

_CACHED_PROFILE: HardwareProfile | None = None


def probe(*, refresh: bool = False) -> HardwareProfile:
    """Discover host accelerator capabilities.

    Probing is import-based (vendor plugins are detected via
    ``importlib.util.find_spec`` and only imported when present), guarded so
    that any single vendor failure degrades to ``available=False`` instead of
    raising.  The result is cached; pass ``refresh=True`` to re-probe.

    Returns:
        A frozen :class:`HardwareProfile` describing every known backend,
        collectives, reduced-precision storage, and Triton availability.
    """
    global _CACHED_PROFILE
    if _CACHED_PROFILE is not None and not refresh:
        return _CACHED_PROFILE

    backends: list[BackendInfo] = [BackendInfo("cpu", True, 1, None, None)]
    backends.append(_probe_cuda())
    for name, plugins in _PLUGIN_ACCELERATORS:
        backends.append(_probe_plugin_backend(name, plugins))
    backends.append(_probe_mps())

    available_names = {info.name for info in backends if info.available}
    # Mirrors utils.default_device_name(): SDAA hosts (LoongArch/Hygon)
    # prefer their own accelerator even when a CUDA card coexists.
    default = next(
        (name for name in ("sdaa", "cuda", "npu", "mlu", "musa", "mps") if name in available_names),
        "cpu",
    )
    device = torch.device(default)
    profile = HardwareProfile(
        backends=tuple(backends),
        collectives=_probe_collectives(),
        fp16_storage=_dtype_storage_supported(torch.float16, device),
        bf16_storage=_dtype_storage_supported(torch.bfloat16, device),
        triton_available=(_triton := _probe_triton())[0],
        triton_version=_triton[1],
        default_device=default,
        torch_version=str(torch.__version__),
        python_version=platform.python_version(),
        hostname=platform.node(),
    )
    _CACHED_PROFILE = profile
    return profile


# ---------------------------------------------------------------------------
# require() + degradation advice
# ---------------------------------------------------------------------------

#: Capability token -> degradation suggestion used in errors and reports.
DEGRADATION_ADVICE: dict[str, str] = {
    "fp16_storage": (
        "fp16 storage is unavailable on this host/device -> degrade to the "
        "FP32FP32 configuration (float32 populations with float32 collision); "
        "numerics are unchanged, memory demand doubles"
    ),
    "bf16_storage": (
        "bf16 storage is unavailable -> degrade to FP32FP32 (float32 "
        "populations with float32 collision)"
    ),
    "triton": (
        "Triton is unavailable -> degrade to the eager torch path "
        "(solver3d.stream3d gather or stream3d_roll); identical numerics, "
        "lower throughput"
    ),
    "nccl": (
        "NCCL is unavailable -> degrade to gloo collectives on CPU/host "
        "fabrics, or the vendor collective (hccl on Ascend, rccl on "
        "ROCm/MUSA-class hardware); fused multi-GPU kernels stay disabled"
    ),
    "hccl": (
        "HCCL is unavailable (torch_npu plugin missing) -> multi-NPU "
        "collectives degrade to gloo over the host fabric"
    ),
    "rccl": (
        "RCCL is unavailable (torch_musa plugin missing) -> multi-MUSA "
        "collectives degrade to gloo over the host fabric"
    ),
    "cuda": (
        "CUDA is unavailable -> the eager torch path runs on any available "
        "backend (npu/mlu/sdaa/musa/cpu); Triton fused kernels are "
        "CUDA-only and stay disabled"
    ),
    "collectives": (
        "no collective library available -> degrade to single-device runs; "
        "distributed decompositions are disabled"
    ),
}


def describe_degradation(capability: str) -> str:
    """Return the stored degradation advice for *capability* (or a generic)."""
    if capability in DEGRADATION_ADVICE:
        return DEGRADATION_ADVICE[capability]
    if capability in _KNOWN_BACKENDS:
        available = ", ".join(probe().available_backends) or "cpu only"
        return (
            f"{capability} is unavailable on this host -> run the eager "
            f"path on an available backend ({available})"
        )
    return f"capability {capability!r} is unavailable -> fall back to the eager path"


def require(capability: str, *, profile: HardwareProfile | None = None) -> HardwareProfile:
    """Assert a capability is present, raising actionable advice otherwise.

    Args:
        capability: One of the backend names (``cuda``/``npu``/``mlu``/
            ``sdaa``/``musa``/``mps``/``cpu``), a reduced-precision token
            (``fp16_storage``/``bf16_storage``), ``triton``, or a collective
            name (``nccl``/``hccl``/``rccl``/``gloo``/``collectives``).
        profile: Use this profile instead of a fresh/cached :func:`probe`
            (tests inject synthetic profiles this way).

    Returns:
        The profile that was checked.

    Raises:
        ValueError: If the capability token is not recognised.
        HardwareCapabilityError: If the capability is missing; the message
            always carries a degradation suggestion.
    """
    resolved = profile if profile is not None else probe()

    def fail(detail: str) -> None:
        raise HardwareCapabilityError(capability, detail)

    if capability in ("fp16_storage", "bf16_storage", "triton"):
        attribute = {
            "fp16_storage": "fp16_storage",
            "bf16_storage": "bf16_storage",
            "triton": "triton_available",
        }[capability]
        if not getattr(resolved, attribute):
            fail(f"{capability} is required but not supported on this host.")
        return resolved

    if capability == "collectives":
        if not any(info.available for info in resolved.collectives):
            fail("a collectives library is required but none is usable.")
        return resolved

    if capability in _KNOWN_COLLECTIVES:
        info = resolved.collective(capability)
        if info is None or not info.available:
            fail(f"collectives backend {capability!r} is required but not usable.")
        return resolved

    if capability in _KNOWN_BACKENDS:
        if not resolved.has_backend(capability):
            fail(f"backend {capability!r} is required but not available.")
        return resolved

    raise ValueError(
        f"Unknown capability {capability!r}; expected one of "
        f"{sorted(set(_KNOWN_BACKENDS + _KNOWN_COLLECTIVES + tuple(DEGRADATION_ADVICE)))}"
    )
