"""Precision emulation for the D3Q19 pointwise reference gate (round 4).

Why this module exists
----------------------
Rounds 1-3 froze a pointwise D3Q19 bundle and gated the in-repo kernels at
``eps_inf <= 1e-10`` (fp64).  Two questions were *asserted* rather than
measured:

1. What does the same gate look like when the populations only carry reduced
   precision (fp32 / fp16 / bf16 storage, or a genuinely fp32 arithmetic
   path, as on the fp32 backends)?  I.e. what is the *noise floor* per dtype,
   and therefore what threshold is admissible per dtype?
2. Given that floor, what is the *smallest regression the gate can still
   see*?  A gate that cannot distinguish a 1 % relaxation-time drift from
   rounding noise is not a regression gate for that dtype.

Round 1's ``fp32_sensitivity`` row is superseded: it compared
``fp32(kernel_f64)`` against ``fp32(reference_f64)``, and because the two fp64
arrays already agree to ~7e-16 the fp32 casts are *identical by construction*,
so it reported ``eps_scaled_inf_fp32_rounding == 0.0``.  That measures the
cast, not fp32 arithmetic.  Here the reduced precision is applied **inside**
the time loop (per step), which is where LBM rounding actually accumulates.

Model (stated explicitly, because it defines the validity range)
---------------------------------------------------------------
* ``arithmetic``  -- dtype used for every operation of the step (moments,
  equilibrium, collision, streaming).  ``fp32`` is a true fp32 arithmetic
  path (numpy keeps float32 through the reduction), matching an fp32 kernel.
* ``storage``     -- dtype the populations are *written back* as after each
  step (round-to-nearest, then promoted back to the arithmetic dtype).
  ``None`` = no storage rounding.  This models a fp64-arithmetic / fp32-memory
  device, and is the conservative (under-estimating) proxy for a real fp32
  path, so both are reported.

The reference side is untouched: the frozen bundle
``d3q19/refdata/pointwise_d3q19_v1.npz`` (lbmpy-symbolic collide + numpy
definition streaming, fp64) is the fixed target of every measurement.

Nothing in this module regenerates or rewrites frozen evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XV = HERE.parent
D3 = XV / "d3q19"
REPO = XV.parents[1]  # .../xval/<anchor> -> repo root
for _p in (str(D3), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refcore3d as core  # noqa: E402

ARITH = {"fp64": np.float64, "fp32": np.float32}
#: storage dtypes addressed by :func:`quantize`; ``None`` = no write-back rounding
STORAGE = {"none": None, "fp32": np.float32, "fp16": np.float16, "bf16": "bfloat16"}


# ---------------------------------------------------------------------------
# reduced-precision write-back
# ---------------------------------------------------------------------------
def _bf16_round(x: np.ndarray) -> np.ndarray:
    """Round ``x`` to bfloat16 precision (round-half-to-even), return float32.

    bfloat16 has no numpy dtype; the conversion is done on the bit pattern so
    no torch dependency is introduced.
    """
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint32)
    lsb = (u >> 16) & np.uint32(1)
    u = u + np.uint32(0x7FFF) + lsb
    return (u & np.uint32(0xFFFF0000)).view(np.float32)


_STORE_DT = {"fp32": np.float32, "fp16": np.float16}


def storage_dtype(storage):
    """Normalise a storage spec (``None``/name/numpy dtype) to a numpy dtype."""
    if storage is None or isinstance(storage, np.dtype):
        return storage
    if isinstance(storage, str):
        if storage in ("none", ""):
            return None
        if storage == "bf16":
            return "bf16"
        return _STORE_DT[storage]
    return np.dtype(storage)


def quantize(f: np.ndarray, storage) -> np.ndarray:
    """Write-back rounding of ``f`` to ``storage`` precision, same dtype out."""
    st = storage_dtype(storage)
    if st is None:
        return f
    if st == "bf16":
        return _bf16_round(np.asarray(f, dtype=np.float32)).astype(np.asarray(f).dtype)
    return np.asarray(f).astype(st).astype(np.asarray(f).dtype)


# ---------------------------------------------------------------------------
# reference step, re-derived here only to be able to change the dtype
# ---------------------------------------------------------------------------
_W64 = core.weights()
_C64 = core.velocities()
_CI = core.velocities().astype(int)


def reference_step(f: np.ndarray, tau: float, arithmetic: str = "fp64",
                   storage=None) -> np.ndarray:
    """One BGK step ``Stream o Collide`` of the round-3 reference definition.

    Identical algebra to ``refcore3d`` (which binds ``dtype=float64``); at
    ``arithmetic='fp64', storage=None`` it reproduces ``ref_t30`` to ~8e-15,
    which the audit re-checks as a control (``control_fp64_reproduces_ref``).
    """
    dt = ARITH[arithmetic]
    f = np.ascontiguousarray(f, dtype=dt)
    spatial = f.shape[1:]
    w = _W64.astype(dt).reshape((19,) + (1,) * len(spatial))
    c = _C64.astype(dt)
    rho = f.sum(0)
    u = np.tensordot(c, f, axes=(0, 0)) / rho[None]
    u2 = np.einsum("i...,i...->...", u, u)
    cu = np.einsum("ki,i...->k...", c, u)
    feq = w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
    post = f - (f - feq) / dt(tau)
    out = np.empty_like(post)
    for i in range(19):
        cx, cy, cz = _CI[i]
        out[i] = np.roll(post[i], shift=(-cz, -cy, -cx), axis=(0, 1, 2))
    return quantize(out, storage).astype(dt, copy=False)


def step_intermediate_dtypes(f: np.ndarray, tau: float, arithmetic: str = "fp32") -> dict:
    """dtypes of the intermediates of one step -- proves the path is really fp32."""
    dt = ARITH[arithmetic]
    f = np.ascontiguousarray(f, dtype=dt)
    w = _W64.astype(dt).reshape((19,) + (1,) * (f.ndim - 1))
    c = _C64.astype(dt)
    rho = f.sum(0)
    u = np.tensordot(c, f, axes=(0, 0)) / rho[None]
    cu = np.einsum("ki,i...->k...", c, u)
    u2 = np.einsum("i...,i...->...", u, u)
    feq = w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
    post = f - (f - feq) / dt(tau)
    return {"rho": str(rho.dtype), "u": str(u.dtype), "cu": str(cu.dtype),
            "u2": str(u2.dtype), "feq": str(feq.dtype), "post": str(post.dtype)}


def run(f0: np.ndarray, tau: float, steps: int, arithmetic: str = "fp64",
        storage=None) -> list[np.ndarray]:
    """Trajectory ``f^1 .. f^steps`` of the reference step at the asked precision."""
    f = np.array(f0, dtype=ARITH[arithmetic], copy=True)
    traj = []
    for _ in range(steps):
        f = reference_step(f, tau, arithmetic, storage)
        traj.append(np.asarray(f, dtype=np.float64))
    return traj


# ---------------------------------------------------------------------------
# in-repo (torch fp64) arms under storage rounding
# ---------------------------------------------------------------------------
def run_repo_arm(arm: str, f0: np.ndarray, tau: float, steps: int,
                 storage=None) -> list[np.ndarray]:
    """Repo kernel arm (``arms3d``) with per-step storage rounding.

    ``arms3d`` pins torch float64, so what is measured here is the *storage*
    half of the reduced-precision error on the production code path -- the
    arithmetic-half is measured by :func:`run` above.
    """
    import arms3d  # noqa: PLC0415

    fn = arms3d.ARMS[arm]
    f = np.array(f0, dtype=np.float64, copy=True)
    traj = []
    for _ in range(steps):
        f = quantize(fn(f, tau), storage)
        traj.append(f)
    return traj


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def eps_curve(traj: list[np.ndarray], refs: dict[str, np.ndarray]) -> list[dict]:
    """``eps_inf`` of every step against the frozen reference of that step.

    The frozen bundle only stores ``ref_t1`` and ``ref_t30``; the un-stored
    steps are compared against the *fp64 reference trajectory itself* (which
    equals the frozen endpoints to 1e-15), so the curve is a deviation-from-
    reference curve and its endpoints are additionally pinned to the bundle.
    """
    out = []
    for n, f in enumerate(traj, start=1):
        key = f"ref_t{n}"
        if key in refs:
            target, src = refs[key], "frozen"
        else:
            target, src = refs["_traj"][n - 1], "fp64_trajectory"
        m = core.compare(f, target)
        out.append({"step": n, "eps_inf": m["eps_inf"], "eps_l2": m["eps_l2"],
                    "ref_source": src})
    return out


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    """Frozen bundle + the fp64 reference trajectory (control baseline)."""
    with np.load(path) as z:
        refs = {k: z[k] for k in z.files}
    tau = 0.7
    steps = 30
    refs["_traj"] = run(refs["f0"], tau, steps, "fp64", None)
    return refs


def detect_floor(rows: list[dict], threshold: float) -> dict:
    """Smallest relative ``tau`` perturbation whose ``eps_inf`` clears ``threshold``."""
    for r in sorted(rows, key=lambda x: x["delta_rel"]):
        if r["eps_inf"] > threshold:
            return {"threshold": threshold, "delta_rel": r["delta_rel"],
                    "eps_inf": r["eps_inf"]}
    return {"threshold": threshold, "delta_rel": None, "eps_inf": None}