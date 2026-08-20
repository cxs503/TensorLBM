"""Compute/store precision policy for TensorLBM steppers.

The policy separates the dtype used for *arithmetic* inside a step
(``compute``) from the dtype used to *store* the distribution between
steps (``store``).  Five tiers are defined:

===============  ==============  =============  =================================
Tier             Compute         Store          Typical use
===============  ==============  =============  =================================
``FP64FP64``     float64         float64        reference / convergence audits
``FP64FP32``     float64         float32        accuracy-critical production
``FP64FP16``     float64         float16        (reserved; not wired to kernels)
``FP32FP32``     float32         float32        default production tier
``FP32FP16``     float32         float16        bandwidth-bound regimes
===============  ==============  =============  =================================

Cast boundary convention
------------------------
A stepper that honours a policy casts at exactly two points:

* **entry** — ``cast_to_compute(f)`` widens (or narrows) the stored
  distribution into the compute dtype before any arithmetic;
* **exit** — ``cast_to_store(f_post)`` narrows the post-step
  distribution back to the store dtype before it is written to the
  persistent buffer.

Everything between those two calls runs in the compute dtype.  For the
fused Triton kernels (:mod:`tensorlbm.triton_fused`) the same boundary
is realised *inside* the kernel: loads are widened to ``fp32``
registers and stores are narrowed to the storage dtype, so the kernel
is the cast boundary.  Only tiers whose compute dtype is ``float32``
are supported there; the ``FP64*`` tiers raise
``NotImplementedError`` from the Triton solvers and are intended for
the eager PyTorch paths.

Provenance / licence
--------------------
The five-tier compute/store split and the ``cast_to_compute`` /
``cast_to_store`` entry/exit convention are adapted from Autodesk XLB's
``xlb/precision_policy.py`` (Apache License 2.0, Copyright 2023
Autodesk Inc. — https://github.com/Autodesk/XLB).  Changes for
TensorLBM (state changes, per Apache-2.0 §4(b)):

* re-implemented as a plain ``enum.Enum`` carrying ``(compute_dtype,
  store_dtype)`` pairs instead of a class with module-level global
  state;
* dtypes are ``torch.dtype`` objects (no JAX dependency);
* added ``PrecisionPolicy.parse`` (accepts enum, exact name string, or
  ``None``) and module-level ``cast_to_compute`` / ``cast_to_store``
  functions usable where the policy is passed explicitly;
* documented the fused-Triton in-kernel cast boundary and the
  ``NotImplementedError`` behaviour for fp64-compute tiers.

The Apache-2.0 licence text is available in the upstream repository;
this file is a derivative of the *design*, not a copy of XLB source
code.
"""

from __future__ import annotations

import enum

import torch


__all__ = [
    "PrecisionPolicy",
    "cast_to_compute",
    "cast_to_store",
]


class PrecisionPolicy(enum.Enum):
    """Compute dtype x store dtype tier for an LBM stepper.

    The name is ``<COMPUTE><STORE>`` with the float width, e.g.
    ``FP32FP16`` = float32 arithmetic with float16 storage.  See the
    module docstring for the cast-boundary contract
    (:func:`cast_to_compute` at step entry, :func:`cast_to_store` at
    step exit).
    """

    FP64FP64 = (torch.float64, torch.float64)
    FP64FP32 = (torch.float64, torch.float32)
    FP64FP16 = (torch.float64, torch.float16)
    FP32FP32 = (torch.float32, torch.float32)
    FP32FP16 = (torch.float32, torch.float16)

    def __init__(self, compute_dtype: torch.dtype, store_dtype: torch.dtype) -> None:
        self._compute_dtype = compute_dtype
        self._store_dtype = store_dtype

    # -- dtypes ----------------------------------------------------------
    @property
    def compute_dtype(self) -> torch.dtype:
        """dtype of all arithmetic inside a step."""
        return self._compute_dtype

    @property
    def store_dtype(self) -> torch.dtype:
        """dtype of the distribution stored between steps."""
        return self._store_dtype

    # -- cast boundaries -------------------------------------------------
    def cast_to_compute(self, f: torch.Tensor) -> torch.Tensor:
        """Step-entry cast: stored distribution -> compute dtype."""
        return cast_to_compute(f, self)

    def cast_to_store(self, f: torch.Tensor) -> torch.Tensor:
        """Step-exit cast: post-step distribution -> store dtype."""
        return cast_to_store(f, self)

    # -- parsing ----------------------------------------------------------
    @classmethod
    def parse(cls, value: "PrecisionPolicy | str | None") -> "PrecisionPolicy":
        """Coerce ``value`` into a :class:`PrecisionPolicy`.

        Accepts an existing policy, its exact name (e.g. ``"FP32FP16"``),
        or ``None`` (-> the default ``FP32FP32``).  Anything else raises
        ``ValueError`` listing the valid names.
        """
        if value is None:
            return cls.FP32FP32
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value]
            except KeyError:
                valid = ", ".join(m.name for m in cls)
                raise ValueError(
                    f"Unknown precision policy {value!r}; valid names: {valid}"
                ) from None
        raise TypeError(
            f"precision policy must be PrecisionPolicy, str, or None, "
            f"got {type(value).__name__}"
        )


def cast_to_compute(f: torch.Tensor, policy: PrecisionPolicy | str | None) -> torch.Tensor:
    """Widen/narrow ``f`` into the policy's compute dtype (step entry).

    Returns ``f`` unchanged when the dtype already matches, so the
    boundary cast is free for the common ``FP32FP32`` tier.
    """
    policy = PrecisionPolicy.parse(policy)
    if f.dtype == policy.compute_dtype:
        return f
    return f.to(policy.compute_dtype)


def cast_to_store(f: torch.Tensor, policy: PrecisionPolicy | str | None) -> torch.Tensor:
    """Narrow ``f`` into the policy's store dtype (step exit).

    Returns ``f`` unchanged when the dtype already matches.
    """
    policy = PrecisionPolicy.parse(policy)
    if f.dtype == policy.store_dtype:
        return f
    return f.to(policy.store_dtype)
