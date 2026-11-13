"""Numerical baseline block IO (format only, no acceptance logic).

A *baseline block* is a single ``.npz`` file holding named float arrays plus a
JSON metadata record.  It is the on-disk unit that the pytest reference
practice of ``pylbm/pylbm`` stores in ``tests/reference/*.h5``; this repository
uses ``.npz`` because ``h5py`` is an optional extra here while ``numpy`` is a
hard dependency (see :mod:`tensorlbm.verification.baseline_reference` for the
full comparison of the two mechanisms).

The module deliberately contains **no tolerance / pass-fail logic**: the
production solver writes blocks (:func:`write_block`), and the verification
suite (:mod:`tensorlbm.verification.baseline_reference`) decides whether a
candidate agrees with a reference.  Keeping the acceptance rule out of the
solver is the same separation the repository's ``verification/`` package
docstring asks for.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

#: Prefix of array datasets inside a block file (keeps metadata keys distinct).
FIELD_PREFIX = "field__"
META_KEY = "__meta__"
FORMAT_ID = "tensorlbm-baseline-block/1"


@dataclasses.dataclass(frozen=True)
class BlockHeader:
    """Metadata stored next to the arrays."""

    payload: dict[str, Any]

    @property
    def sha256(self) -> str | None:
        value = self.payload.get("sha256")
        return str(value) if value is not None else None


def to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor / array to numpy without torch's numpy bridge.

    The image used for this mission ships a torch build whose numpy bridge is
    broken (``Failed to initialize NumPy: module 'numpy._globals' has no
    attribute '_signature_descriptor'``), so ``Tensor.numpy()`` raises
    ``RuntimeError: Numpy is not available`` while ``numpy`` itself works.  The
    fallback goes through ``tolist()``, which is slower but exact for
    float32/float64.
    """
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        try:
            return np.asarray(tensor.numpy())
        except (RuntimeError, TypeError):  # broken torch<->numpy bridge
            return np.asarray(tensor.tolist())
    return np.asarray(value)


def as_field_dict(fields: Mapping[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in fields.items():
        arr = to_numpy(value)
        if arr.dtype.kind in "OUSV":
            msg = f"field {key!r} has non-numeric dtype {arr.dtype}"
            raise TypeError(msg)
        out[key] = arr
    if not out:
        msg = "a baseline block needs at least one numeric field"
        raise ValueError(msg)
    return out


def payload_digest(fields: Mapping[str, np.ndarray]) -> str:
    """sha256 over (name, dtype, shape, little-endian bytes) of every field."""
    digest = hashlib.sha256()
    for key in sorted(fields):
        arr = np.ascontiguousarray(np.asarray(fields[key]))
        digest.update(key.encode())
        digest.update(str(arr.dtype.str).encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes())
    return digest.hexdigest()


def environment_meta() -> dict[str, Any]:
    """Runtime fingerprint stamped into every block."""
    try:  # torch is a hard dependency, but keep this module importable alone
        import torch

        torch_version: str | None = torch.__version__
        threads: int | None = torch.get_num_threads()
    except Exception:  # pragma: no cover - defensive
        torch_version, threads = None, None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch_version,
        "torch_num_threads": threads,
    }


def write_block(
    path: str | Path,
    fields: Mapping[str, Any],
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``fields`` (+ metadata) to ``path`` as a self-describing ``.npz``."""
    path = Path(path)
    arrays = as_field_dict(fields)
    payload = {f"{FIELD_PREFIX}{k}": v for k, v in arrays.items()}
    stamp: dict[str, Any] = {
        "format": FORMAT_ID,
        "fields": sorted(arrays),
        "field_dtypes": {k: str(v.dtype) for k, v in arrays.items()},
        "field_shapes": {k: list(v.shape) for k, v in arrays.items()},
        "sha256": payload_digest(arrays),
        "environment": environment_meta(),
    }
    stamp.update(dict(meta or {}))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, __meta__=np.asarray(json.dumps(stamp, sort_keys=True)), **payload)
    return path


def read_block(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read arrays and metadata from a block file."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        fields = {
            key[len(FIELD_PREFIX) :]: data[key].copy()
            for key in data.files
            if key.startswith(FIELD_PREFIX)
        }
        meta = json.loads(str(data[META_KEY])) if META_KEY in data.files else {}
    if not fields:
        msg = f"{path} contains no {FIELD_PREFIX}* datasets"
        raise ValueError(msg)
    return fields, meta


def header(path: str | Path) -> BlockHeader:
    """Metadata-only view (no array loading)."""
    _, meta = read_block(path)
    return BlockHeader(meta)