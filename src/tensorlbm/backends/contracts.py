"""Framework-free contracts for the R1 cold-path backend boundary."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BackendId(str, Enum):
    """Known backend identifiers; R1 implements only Torch."""

    TORCH = "torch"
    PADDLE = "paddle"
    MINDSPORE = "mindspore"


class BackendSupport(str, Enum):
    """Whether a backend is actually implemented by this release."""

    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"


def _nonempty_string(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple of non-empty strings")
    return tuple(_nonempty_string(item, name) for item in value)


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """Explicit device and dtype request for cold-path plan construction."""

    device: str
    dtype_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _nonempty_string(self.device, "device"))
        object.__setattr__(self, "dtype_name", _nonempty_string(self.dtype_name, "dtype_name"))


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Declared, validated backend support without claiming untested targets."""

    backend_id: BackendId
    support: BackendSupport
    supported_devices: tuple[str, ...]
    supported_dtypes: tuple[str, ...]
    notes: str

    def __post_init__(self) -> None:
        if not isinstance(self.backend_id, BackendId):
            raise ValueError("backend_id must be a BackendId")
        if not isinstance(self.support, BackendSupport):
            raise ValueError("support must be a BackendSupport")
        object.__setattr__(self, "supported_devices", _string_tuple(self.supported_devices, "supported_devices"))
        object.__setattr__(self, "supported_dtypes", _string_tuple(self.supported_dtypes, "supported_dtypes"))
        object.__setattr__(self, "notes", _nonempty_string(self.notes, "notes"))
        if self.backend_id in {BackendId.PADDLE, BackendId.MINDSPORE} and self.support is BackendSupport.SUPPORTED:
            raise ValueError("only Torch may be marked SUPPORTED in R1")
        if self.support is BackendSupport.NOT_SUPPORTED and (self.supported_devices or self.supported_dtypes):
            raise ValueError("NOT_SUPPORTED backends must not declare supported devices or dtypes")
        if self.support is BackendSupport.SUPPORTED and (not self.supported_devices or not self.supported_dtypes):
            raise ValueError("SUPPORTED backends must declare devices and dtypes")


__all__ = ["BackendCapabilities", "BackendId", "BackendSupport", "DeviceSpec"]
