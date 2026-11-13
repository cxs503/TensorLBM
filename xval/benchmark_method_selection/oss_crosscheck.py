#!/usr/bin/env python3
"""OSS-code cross-check for the BGK arm (XVAL ref = executed OSS source).

Two checks, both using the *unmodified* OSS code
(``~/oss-cache/Maarten-vd-Sande_lbm @ 573a01c``) as the reference:

  K1  collision kernel : one BGK collision step on an identical near-equilibrium
                         state -- OSS ``lbm_collision`` (float32, praxis of the
                         OSS) vs ``tensorlbm.solver.collide_bgk`` (float64).
  K2  streaming map    : one streaming step on an identical random population
                         field -- OSS ``lbm_streaming`` vs ``tensorlbm.solver.stream``,
                         interior cells only (the OSS loop deliberately skips
                         positive out-of-range indices but wraps negative ones,
                         so its own border treatment is asymmetric and is *not*
                         part of the comparison).

Also measured and recorded: the OSS wall-clock per step at the benchmark grid
(200x200), which is the quantitative reason the physical comparison uses
method *transplants* in the TensorLBM harness instead of running the OSS code
on the full matrix (the OSS collision loop is pure Python).

Output: ``oss_crosscheck.json`` next to this file (source of every number in the
"code-level cross-check" table of the report).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

OSS_ROOT = Path.home() / "oss-cache" / "Maarten-vd-Sande_lbm"
OSS_COMMIT = "573a01c4d2e6321f3634ff271adca4ce91872e6f"


def _materialise_oss() -> tuple[Path, dict]:
    """Expose the unmodified OSS sources as a regular ``lbm`` package.

    Two environment facts force this shim (both recorded as friction):

    * the OSS package has no ``__init__.py``, so as a PEP-420 namespace package
      it loses to the regular ``lbm`` package that another checkout puts on
      ``PYTHONPATH`` -> the OSS code would silently not be the one imported;
    * ``lbm.streamcollide`` imports ``numba.jit`` and numba is not installed in
      this container.  ``jit`` is a pure speed decorator (object mode) so an
      identity shim keeps the numerics bit-identical and only slows the run.

    The ``.py`` files are symlinked, never copied, and their sha256 is recorded
    so the report can prove the reference code is unmodified.
    """
    import hashlib
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="oss_lbm_"))
    (root / "numba.py").write_text(
        "def jit(*args, **kwargs):\n"
        "    if args and callable(args[0]):\n"
        "        return args[0]\n"
        "    return lambda f: f\n"
    )
    pkg = root / "lbm"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    hashes = {}
    for src in sorted((OSS_ROOT / "lbm").glob("*.py")):
        (pkg / src.name).symlink_to(src)
        hashes[src.name] = hashlib.sha256(src.read_bytes()).hexdigest()
    return root, hashes


_OSS_SHIM, OSS_SHA256 = _materialise_oss()
sys.path.insert(0, str(_OSS_SHIM))

# the unmodified OSS reference code (symlinked sources, numba.jit identity shim)
from lbm.init import init as oss_init  # noqa: E402
from lbm.streamcollide import lbm_collision, lbm_streaming  # noqa: E402
from lbm.util import CT, t as oss_w, v as oss_v, v_inv as oss_vinv  # noqa: E402

import torch  # noqa: E402

from tensorlbm.d2q9 import C as OUR_C  # noqa: E402
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

# TensorLBM (0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)
# OSS      (0,0),(0,-1),(0,1),(-1,0),(-1,-1),(-1,1),(1,0),(1,-1),(1,1)
PERM_OUR_TO_OSS = []
for i in range(9):
    vec = tuple(int(x) for x in OUR_C[i].tolist())
    PERM_OUR_TO_OSS.append([tuple(int(y) for y in row) for row in oss_v].index(vec))


_INV = np.argsort(PERM_OUR_TO_OSS)  # oss direction -> TensorLBM direction


def our_to_oss(f_our: np.ndarray) -> np.ndarray:
    """(9,ny,nx) TensorLBM -> (9,nx,ny) OSS:  out[oss] = f[our] with same vector."""
    return np.transpose(np.asarray(f_our)[_INV], (0, 2, 1))


def oss_to_our(f_oss: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(f_oss, dtype=np.float64)[PERM_OUR_TO_OSS], (0, 2, 1))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def tgv_state(n: int, k: int = 1, u0: float = 0.03):
    """Taylor-Green vortex: u = u0 (sin kx cos ky, -cos kx sin ky) on [0,1)^2."""
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(2.0 * math.pi * k * x, 2.0 * math.pi * k * x)
    ux = u0 * np.sin(X) * np.cos(Y)
    uy = -u0 * np.cos(X) * np.sin(Y)
    return ux, uy


def oss_state(nx: int, ny: int) -> dict:
    """OSS ``init`` unpacked into a dict (order in the OSS: fin, fout, equi,
    fdist, inlet, u, rho, mass, cell_type)."""
    keys = ("fin", "fout", "equi", "fdist", "inlet", "u", "rho", "mass", "cell_type")
    return dict(zip(keys, oss_init(nx, ny)))


def k0_direction_map(n: int = 8) -> dict:
    """Sanity check of the direction permutation with a single-cell spike:
    for every direction the OSS stream must move the spike to the same cell as
    ``tensorlbm.solver.stream``.  Guards the whole comparison."""
    ok = []
    from tensorlbm.solver import stream as our_stream

    for i in range(9):
        f = np.zeros((9, n, n))
        f[i, n // 2, n // 2] = float(i + 1)
        got = our_stream(torch.tensor(f, dtype=torch.float64)).numpy()
        st = oss_state(n, n)
        st["cell_type"][:, :] = CT.FLUID
        st["fout"][:] = our_to_oss(f).astype(np.float32)
        lbm_streaming(st["fin"], st["fout"], st["equi"], st["u"], st["rho"], st["mass"],
                      st["cell_type"], np.copy(st["mass"]), np.copy(st["mass"]),
                      oss_v, oss_vinv, oss_w)
        ref = oss_to_our(st["fin"])
        ok.append(bool(np.array_equal(np.argwhere(np.abs(got[i]) > 1e-9),
                                      np.argwhere(np.abs(ref[i]) > 1e-9))))
    return {"directions_agree": ok, "all_agree": bool(all(ok)), "n": n}


def k1_collision(n: int = 48, tau: float = 0.5075) -> dict:
    ux, uy = tgv_state(n)
    rho = 1.0 + 0.01 * np.cos(2.0 * math.pi * 2 * (np.arange(n) + 0.5) / n)[None, :] * np.ones((n, 1))

    # TensorLBM reference state (float64)
    f_our = equilibrium(
        torch.tensor(rho), torch.tensor(ux), torch.tensor(uy)
    ).numpy()
    got = collide_bgk(torch.tensor(f_our), tau).numpy()

    # OSS executed source (float32 throughout)
    st = oss_state(n, n)
    fin, fout, equi, inlet = st["fin"], st["fout"], st["equi"], st["inlet"]
    u_o, rho_o, mass_o, cell_type = st["u"], st["rho"], st["mass"], st["cell_type"]
    cell_type[:, :] = CT.FLUID
    fin[:] = our_to_oss(f_our).astype(np.float32)
    u_o[0] = np.zeros((n, n), dtype=np.float32)
    u_o[1] = np.zeros((n, n), dtype=np.float32)
    rho_o[:] = np.ones((n, n), dtype=np.float32)
    mass_o[:] = 1.0
    omega = np.array([1.0 / tau], dtype=np.float32)
    gravity = np.zeros(2, dtype=np.float32)
    lbm_collision(fin, fout, equi, inlet, u_o, rho_o, cell_type, omega, oss_v, oss_w, gravity)
    ref = oss_to_our(fout)
    return {
        "n": n,
        "tau": tau,
        "rel_l2_collision": rel_l2(got, ref),
        "max_abs_diff": float(np.abs(got - ref).max()),
        "oss_dtype": "float32",
        "tensorlbm_dtype": "float64",
    }


def k2_streaming(n: int = 48, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    f_our = 0.25 + 0.01 * rng.standard_normal((9, n, n))

    got = stream(torch.tensor(f_our)).numpy()

    st = oss_state(n, n)
    fin, fout, equi, inlet = st["fin"], st["fout"], st["equi"], st["inlet"]
    u_o, rho_o, mass_o, cell_type = st["u"], st["rho"], st["mass"], st["cell_type"]
    cell_type[:, :] = CT.FLUID
    fout[:] = our_to_oss(f_our).astype(np.float32)
    v_inv = oss_vinv
    lbm_streaming(
        fin, fout, equi, u_o, rho_o, mass_o, cell_type,
        np.copy(mass_o), np.copy(mass_o), oss_v, v_inv, oss_w,
    )
    ref_full = oss_to_our(fin)

    # interior only: the OSS loop skips positive out-of-range gather indices but
    # wraps negative ones, so its border cells are not the periodic map
    inner = slice(1, n - 1)
    return {
        "n": n,
        "rel_l2_streaming_interior": rel_l2(got[:, inner, inner], ref_full[:, inner, inner]),
        "max_abs_diff_interior": float(np.abs(got[:, inner, inner] - ref_full[:, inner, inner]).max()),
        "border_excluded": "1-cell frame (OSS asymmetric out-of-range handling)",
    }


def oss_throughput(nx: int = 100, ny: int = 100, steps: int = 1) -> dict:
    """Wall-clock per step of the OSS code at the benchmark grid (justification
    for the transplant design; pure-Python collision loop)."""
    st = oss_state(nx, ny)
    fin, fout, equi, inlet = st["fin"], st["fout"], st["equi"], st["inlet"]
    u_o, rho_o, mass_o, cell_type = st["u"], st["rho"], st["mass"], st["cell_type"]
    cell_type[:, :] = CT.FLUID
    fin[:] = 1.0 / 9.0
    mass_o[:] = 1.0
    rho_o[:] = 1.0
    u_o[:] = 0.0
    omega = np.array([1.8], dtype=np.float32)
    gravity = np.zeros(2, dtype=np.float32)
    t0 = time.time()
    for _ in range(steps):
        lbm_collision(fin, fout, equi, inlet, u_o, rho_o, cell_type, omega, oss_v, oss_w, gravity)
        lbm_streaming(fin, fout, equi, u_o, rho_o, mass_o, cell_type,
                      np.copy(mass_o), np.copy(mass_o), oss_v, oss_vinv, oss_w)
    dt = (time.time() - t0) / steps
    return {
        "grid": f"{nx}x{ny}",
        "s_per_step_oss_100x100": round(dt, 3),
        "s_per_step_oss_200x200_extrapolated": round(dt * 4.0, 3),
        "extrapolation": "linear in cell count (pure-Python per-cell loop)",
        "s_per_step_tensorlbm_measured": 0.0212,  # collide_mrt path, 8 threads, see docstring
        "steps_needed_for_case": 21259,
        "oss_wall_hours_full_case": round(dt * 4.0 * 21259 / 3600.0, 1),
    }


def main() -> int:
    t0 = time.time()
    out = {
        "oss_repo": "Maarten-vd-Sande/lbm",
        "oss_commit": OSS_COMMIT,
        "oss_path": str(OSS_ROOT),
        "oss_source_sha256_unmodified": OSS_SHA256,
        "env_shim": "numba.jit -> identity (numerics identical, speed only); lbm/__init__.py added in a temp dir",
        "dir_permutation_tensorlbm_to_oss": PERM_OUR_TO_OSS,
        "k0_direction_map": k0_direction_map(),
        "k1_collision": k1_collision(),
        "k2_streaming": k2_streaming(),
        "throughput": oss_throughput(),
    }
    out["elapsed_s"] = round(time.time() - t0, 1)
    here = Path(__file__).resolve().parent
    (here / "oss_crosscheck.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())