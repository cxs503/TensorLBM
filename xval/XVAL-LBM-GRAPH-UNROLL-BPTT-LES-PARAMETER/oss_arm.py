"""Reference arm: the upstream OSS algorithm executed *without TensorFlow*.

Source of record
----------------
``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow`` (MIT-less public
repo), files vendored verbatim under ``oss_src/`` with sha256 recorded in
``oss_src/PROVENANCE.md``.

Why a port instead of running TF
--------------------------------
The evaluation host is ``loongarch64`` and no TensorFlow wheel exists for it
(``pip download tensorflow-cpu`` -> "no matching distribution"; see
``artifacts/env_probe.json``), so the upstream TF graph cannot be executed
here.  This module therefore executes the upstream **lattice tables**
(``oss_src/D2Q9.py`` is *executed as-is* behind a 3-line ``tensorflow`` shim,
so WEIGHTS/LVELOC/BOUNCE/STREAM are the upstream arrays, not retyped) and
transcribes the upstream collision/streaming/unroll **operations** one-for-one
in float32 NumPy, with a line reference for every op:

    Domain.py:134-140  CollideSC(graph_unroll=True) entry
    Domain.py:104-131  f_boundary = conv(F*boundary, Op); vel/Vel moments; Feq
    Domain.py:96-101   NonEq = f - Feq; Q = sqrt(2 sum NonEq^2 E); tau formula
    Domain.py:122-127  LES branch (EEk built at Domain.py:90-96, Sc at :57)
    Domain.py:131-132  f = f*(1-boundary) + f_boundary
    Domain.py:178-185  StreamSC: mobius pad + conv with St, moments (Cs=1)
    Domain.py:244-253  Unroll: chain N steps, keep state list
    utils.py:4-22      simple_conv (VALID cross-correlation), pad_mobius

The port keeps float32, the same op order, and the same padding (torus), so a
field-level comparison with :mod:`tensorlbm.les_bptt` is a fair implementation
cross-check rather than an algebraic identity.
"""

from __future__ import annotations

import hashlib
import math
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OSS_SRC = HERE / "oss_src"


# --------------------------------------------------------------------------
# Upstream tables, executed from the vendored source (no hand-copy)
# --------------------------------------------------------------------------
def _tf_shim() -> types.ModuleType:
    """Minimal ``tensorflow`` stand-in: ``tf.constant(x, dtype=1)`` -> float32."""
    mod = types.ModuleType("tensorflow")

    def constant(value, dtype=None, **kwargs):
        np_dtype = np.float32 if dtype in (1, None) else np.asarray(dtype)
        return np.asarray(value, dtype=np_dtype)

    mod.constant = constant
    return mod


def load_upstream_tables(path: Path | None = None) -> dict:
    """Execute ``oss_src/D2Q9.py`` with a stub TF and return its four tables."""
    src = Path(path or (OSS_SRC / "D2Q9.py"))
    code = src.read_text()
    shim = _tf_shim()
    saved = sys.modules.get("tensorflow")
    sys.modules["tensorflow"] = shim
    ns: dict = {"__name__": "oss_d2q9"}
    try:
        exec(compile(code, str(src), "exec"), ns)
    finally:
        if saved is None:
            sys.modules.pop("tensorflow", None)
        else:
            sys.modules["tensorflow"] = saved
    return {
        "WEIGHTS": np.asarray(ns["WEIGHTS"], dtype=np.float32),
        "LVELOC": np.asarray(ns["LVELOC"], dtype=np.float32),
        "BOUNCE": np.asarray(ns["BOUNCE"], dtype=np.float32),
        "STREAM": np.asarray(ns["STREAM"], dtype=np.float32),
        "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "path": str(src),
    }


TABLES = load_upstream_tables()
W = TABLES["WEIGHTS"]          # (9,)
C = TABLES["LVELOC"]           # (9,3)  upstream (x,y,z)
BOUNCE = TABLES["BOUNCE"]      # (9,9)  out[ch] = sum_in in[cin]*BOUNCE[cin,ch]
STREAM = TABLES["STREAM"]      # (3,3,9,9) upstream layout
SC_DEFAULT = 0.17              # Domain.py:57


# --------------------------------------------------------------------------
# Ops (op-for-op transcription, float32)
# --------------------------------------------------------------------------
def _conv_valid(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """``utils.py:4-13`` simple_conv for a 4-D input: TF cross-correlation.

    ``x`` is ``(1, H, W, Cin)`` and ``k`` is ``(kh, kw, Cin, Cout)``;
    out[i,j,co] = sum_{a,b,ci} x[i+a, j+b, ci] k[a, b, ci, co].
    """
    kh, kw, cin, cout = k.shape
    h, w = x.shape[1], x.shape[2]
    oh, ow = h - kh + 1, w - kw + 1
    out = np.zeros((x.shape[0], oh, ow, cout), dtype=np.float32)
    for a in range(kh):
        for b in range(kw):
            patch = x[:, a:a + oh, b:b + ow, :]           # (1, oh, ow, cin)
            out += np.einsum("nhwi,io->nhwo", patch, k[a, b])
    return out


def pad_mobius(f: np.ndarray) -> np.ndarray:
    """``utils.py:16-22``: wrap-pad one cell on both spatial axes."""
    g = np.concatenate([f[:, -1:], f, f[:, :1]], axis=1)
    g = np.concatenate([g[:, :, -1:], g, g[:, :, :1]], axis=2)
    return g


_EEK = np.zeros(9, dtype=np.float32)
for n in range(3):
    for m in range(3):
        # Domain.py:90-96 (EEk = sum_{n,m} |C_n C_m|, per direction)
        _EEK = _EEK + np.abs(C[:, n] * C[:, m])


def collide_sc(f: np.ndarray, tau0: float, sc: float, boundary: np.ndarray,
               les: bool = True) -> np.ndarray:
    """``Domain.py:96-132`` CollideSC with ``graph_unroll=True`` semantics."""
    f = np.asarray(f, dtype=np.float32)
    boundary = np.asarray(boundary, dtype=np.float32)
    # bounce-back piece from the pre-collision population (Domain.py:98-100):
    # f_boundary = conv(F * boundary, Op) with Op[cin, cout] = BOUNCE[cin, cout]
    f_boundary = _conv_valid(f * boundary, BOUNCE.reshape(1, 1, 9, 9))
    rho = f.sum(axis=-1, keepdims=True)                                  # :108
    vel = _conv_valid(f, C.reshape(1, 1, 9, 3)) / rho                    # :109 (Cs=1)
    vel_dot_vel = (vel * vel).sum(axis=-1, keepdims=True)                # :105
    vel_dot_c = np.einsum("nhwi,qi->nhwq", vel, C)                       # :111
    feq = W.reshape(1, 1, 9) * rho * (                                   # :112
        1.0 + 3.0 * vel_dot_c + 4.5 * vel_dot_c ** 2 - 1.5 * vel_dot_vel)
    noneq = f - feq                                                      # :115
    if les:
        q = (noneq * noneq * _EEK.reshape(1, 1, 9)).sum(axis=-1, keepdims=True)
        q = np.sqrt(2.0 * q)                                             # :117-118
        tau = 0.5 * (tau0 + np.sqrt(tau0 * tau0 + 6.0 * q * sc / rho))   # :119
    else:
        tau = np.float32(tau0)
    out = f - noneq / tau                                                # :121
    f_no_boundary = out * (1.0 - boundary)                               # :124
    return f_no_boundary + f_boundary                                    # :125


def stream_sc(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``Domain.py:178-196`` StreamSC: conv with the upstream STREAM table."""
    f_pad = _conv_valid(pad_mobius(f), STREAM)
    rho = f_pad.sum(axis=-1, keepdims=True)
    vel = _conv_valid(f_pad, np.broadcast_to(C, (1, 1, 9, 3)).copy()) / rho
    return f_pad, rho, vel


def equilibrium_np(rho: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Local equilibrium used by the upstream initialisation callbacks."""
    vel_dot_vel = (vel * vel).sum(axis=-1, keepdims=True)
    vel_dot_c = np.einsum("nhwi,qi->nhwq", vel, C)
    return W.reshape(1, 1, 9) * rho * (1.0 + 3.0 * vel_dot_c + 4.5 * vel_dot_c ** 2
                                       - 1.5 * vel_dot_vel)


def unroll_sc(f0: np.ndarray, n_steps: int, tau0: float, sc: float,
              boundary: np.ndarray, les: bool = True) -> list[dict]:
    """``Domain.py:244-253`` Unroll: chain collide->stream, keep the history."""
    f = np.asarray(f0, dtype=np.float32)
    hist = []
    for step in range(n_steps):
        f = collide_sc(f, tau0, sc, boundary, les=les)
        f, rho, vel = stream_sc(f)
        hist.append({"step": step, "f": f, "rho": rho, "vel": vel})
    return hist


def shear_wave_ic(n: int = 48, u0: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Same periodic shear layer as ``tensorlbm.les_bptt.shear_wave_state``."""
    j = np.arange(n, dtype=np.float32)
    ux = (u0 * np.sin(2.0 * math.pi * j / n)).reshape(1, n, 1, 1) * np.ones((1, 1, n, 1), np.float32)
    vel = np.concatenate([ux, np.zeros_like(ux), np.zeros_like(ux)],
                         axis=-1).astype(np.float32)  # (x, y, z) as upstream
    rho = np.ones((1, n, n, 1), dtype=np.float32)
    boundary = np.zeros((1, n, n, 1), dtype=np.float32)
    return equilibrium_np(rho, vel), boundary


def obstacle_mask(n: int = 64, radius: int = 8) -> np.ndarray:
    """Solid disc in a periodic box: exercises the bounce-back blend path."""
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = n // 2
    inside = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2
    return inside.astype(np.float32).reshape(1, n, n, 1)


if __name__ == "__main__":  # tiny self-check: table integrity + one step
    print("tables sha256", TABLES["sha256"])
    print("sum(W)", W.sum(), "EEK", _EEK)
    f0, b = shear_wave_ic(16)
    h = unroll_sc(f0, 3, 0.56, SC_DEFAULT, b)
    print("vel[2] absmax", float(np.abs(h[2]["vel"]).max()))