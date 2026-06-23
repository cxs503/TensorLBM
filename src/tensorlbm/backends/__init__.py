"""Multi-backend dispatch layer for TensorLBM.

Supported backends
------------------
* ``"torch"``       – PyTorch (default, fully featured)
* ``"paddle"``      – PaddlePaddle 2.x
* ``"mindspore"``   – MindSpore 2.x (PyNative mode)

Runtime selection
-----------------
At startup via environment variable::

    TENSORLBM_BACKEND=paddle python my_script.py

Or programmatically::

    import tensorlbm.backends as B
    B.set_backend("paddle")

The LBM solver core (``d2q9``, ``d3q19``, ``solver``, ``boundaries`` …)
still runs on PyTorch directly.  The multi-backend layer is fully
exercised by the ``tensorlbm.ai`` sub-package (model building, training,
inference) and is designed to be the migration path for the solver core.

Backend modules
---------------
Each of :mod:`tensorlbm.backends.torch_backend`,
:mod:`tensorlbm.backends.paddle_backend` and
:mod:`tensorlbm.backends.mindspore_backend` exposes an identical set of
functions covering tensor creation, math, NN model factories, optimizers,
and training helpers.  New functions added to one backend must be added
to all three.
"""
from __future__ import annotations

import os
from typing import Literal, Any

BackendName = Literal["torch", "paddle", "mindspore"]
_VALID: frozenset[str] = frozenset({"torch", "paddle", "mindspore"})

_current_backend: BackendName = os.environ.get(  # type: ignore[assignment]
    "TENSORLBM_BACKEND", "torch"
)

# Cache of imported backend modules so we don't re-import on every call.
_backend_cache: dict[str, Any] = {}


def get_backend() -> BackendName:
    """Return the name of the currently active computation backend."""
    return _current_backend


def set_backend(name: str) -> None:
    """Switch the active backend.  Affects all subsequent AI calls.

    Args:
        name: One of ``"torch"``, ``"paddle"``, or ``"mindspore"``.

    Raises:
        ValueError: if *name* is not a recognised backend.
    """
    global _current_backend
    if name not in _VALID:
        raise ValueError(
            f"Unknown backend {name!r}. Valid choices: {sorted(_VALID)}."
        )
    _current_backend = name  # type: ignore[assignment]


def get_ops():
    """Return the tensor/NN operations module for the active backend.

    The returned module is cached after the first import so repeated
    calls are cheap.

    Returns:
        One of :mod:`~tensorlbm.backends.torch_backend`,
        :mod:`~tensorlbm.backends.paddle_backend`, or
        :mod:`~tensorlbm.backends.mindspore_backend`.
    """
    name = get_backend()
    if name in _backend_cache:
        return _backend_cache[name]

    if name == "torch":
        from . import torch_backend as _mod
    elif name == "paddle":
        from . import paddle_backend as _mod  # type: ignore[no-redef]
    elif name == "mindspore":
        from . import mindspore_backend as _mod  # type: ignore[no-redef]
    else:
        raise ValueError(f"No ops module for backend {name!r}")

    _backend_cache[name] = _mod
    return _mod


__all__ = [
    "BackendName",
    "get_backend",
    "set_backend",
    "get_ops",
]
