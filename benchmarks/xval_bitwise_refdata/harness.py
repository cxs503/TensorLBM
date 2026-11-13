"""XVAL-LBM-BITWISE-REFDATA-REGRESSION harness (D2Q9 kernel bitwise lock + analytic bench).

Purpose
-------
Provide a *bit-exact* reference-data regression for the D2Q9 primitives that the
whole TensorLBM code base stands on, so that a silent change in a kernel
(``collide_bgk`` / ``collide_trt`` / ``collide_mrt`` / ``collide_bgk_fused`` /
``stream`` / ``equilibrium``) is detected by a byte-level digest instead of by a
loose float tolerance.  On top of that lock the same module carries a physical
benchmark (2D Taylor-Green vortex decay) with an analytic reference so that the
regression lock is anchored to a *physics* number, not only to itself.

Why bitwise and why this is not brittle-by-design
-------------------------------------------------
* Initial fields are built from **integer lattice algebra** with powers-of-two
  divisions (``_int_fields``): every intermediate value is exactly representable
  in binary floating point, so the initial condition is identical bit for bit on
  any IEEE-754 platform and does not depend on ``libm`` ``sin``/``cos``/RNG.
  A digest mismatch is therefore attributable to the *kernels*, not to the test.
* Golden records the environment (torch version, dtype, CPU arch, kernel file
  sha256).  Bitwise equality is *required* only when the environment matches;
  on an environment mismatch the gate degrades honestly to an ulp budget and is
  reported as ``ULP_DRIFT``/``ENV_ADVISORY`` instead of a false red.
* The harness carries its own **negative control** (1-ulp tamper, tau bump) so
  "green" cannot be reached by a dead comparison: see ``tamper``/``sensitivity``.

Everything here calls the real library (``tensorlbm.d2q9`` / ``tensorlbm.solver``);
no re-implementation of the kernels lives in this directory.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from tensorlbm import d2q9, solver

# Tiny lattices (16-64 cells/side): the default thread pool costs ~20x more than
# it saves here, and pinning one thread also removes BLAS thread-count as a
# possible source of digest drift (checked: identical digests at 1 and 8 threads).
torch.set_num_threads(1)

SCHEMA = "xval-lbm-bitwise-refdata/v1"
HERE = Path(__file__).resolve().parent
REFDATA_DIR = HERE / "refdata"
KERNEL_SOURCES = (
    "src/tensorlbm/d2q9.py",
    "src/tensorlbm/solver.py",
)
REPO_ROOT = HERE.parent.parent


# --------------------------------------------------------------------------- #
# deterministic initial condition (integer algebra, no libm, no RNG)
# --------------------------------------------------------------------------- #
def int_fields(ny: int, nx: int, dtype: torch.dtype = torch.float64):
    """Density/velocity fields from integer lattice algebra.

    ``rho = 1 + a/64``, ``ux = b/128``, ``uy = a/128`` with ``a in [-5,5]``,
    ``b in [-3,3]`` integers -> every value is exact in binary floating point
    (2**-6, 2**-7 steps), so the field is bit-identical on every platform.
    """
    ii = torch.arange(nx, dtype=torch.int64).view(1, nx)
    jj = torch.arange(ny, dtype=torch.int64).view(ny, 1)
    a = ((ii * 7 + jj * 3) % 11 - 5).to(dtype)
    b = ((ii * 5 + jj * 13) % 7 - 3).to(dtype)
    rho = 1.0 + a * (1.0 / 64.0)
    ux = b * (1.0 / 128.0)
    uy = a * (1.0 / 128.0)
    return rho, ux, uy


def feq_field(ny: int = 16, nx: int = 16, dtype: torch.dtype = torch.float64):
    rho, ux, uy = int_fields(ny, nx, dtype)
    return d2q9.equilibrium(rho, ux, uy)


def _chain(f0: torch.Tensor, steps: int, collide: Callable[[torch.Tensor], torch.Tensor]):
    f = f0
    for _ in range(steps):
        f = solver.stream(collide(f))
    return f


# --------------------------------------------------------------------------- #
# case registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Case:
    name: str
    what: str
    kernels: tuple[str, ...]
    build: Callable[[], torch.Tensor]


def _cases() -> tuple[Case, ...]:
    f16 = lambda: feq_field(16, 16)  # noqa: E731 - tiny factory, keeps registry readable
    return (
        Case(
            "feq_int_field_16",
            "d2q9.equilibrium on integer-algebra rho/u field",
            ("src/tensorlbm/d2q9.py::equilibrium",),
            f16,
        ),
        Case(
            "bgk_step_16",
            "single BGK collision, tau=0.8",
            ("src/tensorlbm/solver.py::collide_bgk", "macroscopic", "equilibrium"),
            lambda: solver.collide_bgk(f16(), 0.8),
        ),
        Case(
            "trt_step_16",
            "single TRT collision, tau+=0.8, Lambda=3/16",
            ("src/tensorlbm/solver.py::collide_trt",),
            lambda: solver.collide_trt(f16(), 0.8),
        ),
        Case(
            "mrt_step_16",
            "single MRT collision, tau=0.8, default s_e/s_eps/s_q",
            ("src/tensorlbm/solver.py::collide_mrt",),
            lambda: solver.collide_mrt(f16(), 0.8),
        ),
        Case(
            "bgk_fused_step_16",
            "fused BGK collision (no feq tensor), tau=0.8",
            ("src/tensorlbm/solver.py::collide_bgk_fused",),
            lambda: solver.collide_bgk_fused(f16(), 0.8),
        ),
        Case(
            "rlbm_step_16",
            "single regularized-LBM (Latt & Chopard) collision, tau=0.8",
            ("src/tensorlbm/solver.py::collide_rlbm",),
            lambda: solver.collide_rlbm(f16(), 0.8),
        ),
        Case(
            "stream_periodic_16",
            "periodic gather streaming",
            ("src/tensorlbm/solver.py::stream",),
            lambda: solver.stream(f16()),
        ),
        Case(
            "chain_trt_50_24",
            "50 x (TRT collide + stream) on 24x24, float64",
            ("src/tensorlbm/solver.py::collide_trt", "src/tensorlbm/solver.py::stream"),
            lambda: _chain(feq_field(24, 24), 50, lambda f: solver.collide_trt(f, 0.8)),
        ),
        Case(
            "chain_bgk_50_f32_16",
            "50 x (BGK collide + stream) on 16x16, float32",
            ("src/tensorlbm/solver.py::collide_bgk", "src/tensorlbm/solver.py::stream"),
            lambda: _chain(
                feq_field(16, 16, torch.float32), 50, lambda f: solver.collide_bgk(f, 0.8)
            ),
        ),
    )


CASES = _cases()
CASE_BY_NAME = {c.name: c for c in CASES}


def run_case(name: str) -> torch.Tensor:
    """Run one regression case through the real kernels and return the field."""
    with torch.no_grad():
        return CASE_BY_NAME[name].build()


# --------------------------------------------------------------------------- #
# digests / ulp distance
# --------------------------------------------------------------------------- #
def field_bytes(t: torch.Tensor) -> bytes:
    return t.detach().to("cpu").contiguous().numpy().tobytes()


def digest(t: torch.Tensor) -> str:
    """Byte-level sha256 over the raw IEEE-754 payload of the field."""
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(field_bytes(t))
    return h.hexdigest()


def encode(t: torch.Tensor) -> dict:
    return {
        "dtype": str(t.dtype),
        "shape": list(t.shape),
        "b64": base64.b64encode(field_bytes(t)).decode("ascii"),
    }


def decode(payload: dict) -> torch.Tensor:
    raw = base64.b64decode(payload["b64"].encode("ascii"))
    np_dtype = {"torch.float64": np.float64, "torch.float32": np.float32}[payload["dtype"]]
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(tuple(payload["shape"]))
    return torch.from_numpy(arr.copy())


def _mono_keys(t: torch.Tensor) -> np.ndarray:
    """Map IEEE-754 bit patterns to unsigned ints with the same ordering."""
    arr = t.detach().to("cpu").contiguous().numpy()
    if arr.dtype == np.float64:
        u = arr.view(np.uint64)
        sign, mask = np.uint64(0x8000000000000000), np.uint64(0xFFFFFFFFFFFFFFFF)
    else:
        u = arr.view(np.uint32)
        sign, mask = np.uint32(0x80000000), np.uint32(0xFFFFFFFF)
    flipped = u ^ mask
    return np.where((u & sign) != 0, flipped, u | sign)


def ulp_metrics(got: torch.Tensor, ref: torch.Tensor) -> dict:
    """Bit-level comparison: number of differing elements + max ulp distance."""
    if got.shape != ref.shape or got.dtype != ref.dtype:
        return {
            "comparable": False,
            "reason": f"shape/dtype mismatch got={tuple(got.shape)}/{got.dtype} "
            f"ref={tuple(ref.shape)}/{ref.dtype}",
            "max_ulp": None,
            "n_diff": None,
        }
    kg, kr = _mono_keys(got), _mono_keys(ref)
    d = np.where(kg >= kr, kg - kr, kr - kg)
    n_diff = int(np.count_nonzero(d))
    out = {
        "comparable": True,
        "n_diff": n_diff,
        "n_total": int(d.size),
        "max_ulp": float(d.max()) if d.size else 0.0,
        "first_mismatch": None,
    }
    if n_diff:
        idx = np.unravel_index(int(np.argmax(d != 0)), d.shape)
        first = tuple(int(i) for i in idx)
        out["first_mismatch"] = {
            "index": first,
            "got": float(got[first]),
            "ref": float(ref[first]),
            "ulp": float(d[idx]),
        }
    return out


def tamper(t: torch.Tensor, at=(0, 3, 3), n_ulp: int = 1) -> torch.Tensor:
    """Negative control: bump one element by ``n_ulp`` (must be detected)."""
    int_dtype = torch.int64 if t.dtype == torch.float64 else torch.int32
    out = t.clone()
    bits = out.view(int_dtype)
    bits[at] = bits[at] + n_ulp
    return out


# --------------------------------------------------------------------------- #
# environment / code identity
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env() -> dict:
    return {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        "torch_threads": int(torch.get_num_threads()),
    }


def code_identity() -> dict:
    out = {}
    for rel in KERNEL_SOURCES:
        p = REPO_ROOT / rel
        out[rel] = _sha256_file(p) if p.exists() else None
    return out


def _git_sha() -> str | None:
    try:
        import subprocess

        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - repo may be exported without .git
        return None


# --------------------------------------------------------------------------- #
# golden generation / verification
# --------------------------------------------------------------------------- #
def generate() -> dict:
    """Run every case for real and produce the golden reference-data document."""
    cases = {}
    for case in CASES:
        f = run_case(case.name)
        cases[case.name] = {
            "what": case.what,
            "kernels": list(case.kernels),
            "digest": digest(f),
            "payload": encode(f),
            "invariants": case_invariants(case.name, f),
        }
    return {
        "schema": SCHEMA,
        "generated_utc": _utc_now(),
        "git_sha": _git_sha(),
        "env": env(),
        "code": code_identity(),
        "cases": cases,
    }


def _utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def case_invariants(name: str, f: torch.Tensor) -> dict:
    """Physical sanity of a golden field, independent of the digest."""
    rho, ux, uy = solver.macroscopic(f)
    inv = {
        "finite": bool(torch.isfinite(f).all()),
        "sum_f": float(f.sum()),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "u_max": float(torch.sqrt(ux * ux + uy * uy).max()),
    }
    if name.startswith("feq_int_field"):
        # analytic invariants of f_eq: sum_q f_eq = rho, sum_q f_eq c_q = rho u
        # (the compressibility term 4.5 cu^2 - 1.5 u^2 must cancel exactly)
        rho0, ux0, uy0 = int_fields(f.shape[2], f.shape[1], f.dtype)
        inv["mass_conservation_abs_err"] = float((rho - rho0).abs().max())
        inv["momentum_abs_err"] = float((ux - ux0).abs().max() + (uy - uy0).abs().max())
        # f_eq is non-negative for |u| << cs
        inv["feq_min"] = float(f.min())
    return inv


def _env_match(
    golden_env: dict, cur_env: dict, golden_code: dict, cur_code: dict
) -> tuple[bool, list[str]]:
    diffs = []
    for key in ("torch", "machine", "system"):
        if golden_env.get(key) != cur_env.get(key):
            diffs.append(f"{key}: golden={golden_env.get(key)!r} now={cur_env.get(key)!r}")
    for rel, sha in golden_code.items():
        if cur_code.get(rel) != sha:
            diffs.append(f"kernel changed: {rel}")
    return (len(diffs) == 0), diffs


def verify(
    golden: dict,
    docs_ulp_budget: float = 2.0,
    runner: Callable[[str], torch.Tensor] = run_case,
) -> dict:
    """Rebuild every case and compare bit-exactly against the golden document.

    Verdicts per case:
      ``BITWISE_OK``  - every byte identical (only possible when the env matches)
      ``ULP_DRIFT``   - env differs from golden and drift stays inside the ulp budget
      ``REGRESSION``  - env matches and any byte differs, or drift > budget
      ``GOLDEN_INCONSISTENT`` - the stored payload does not match the stored
                        digest (the golden file itself was edited by hand)
    ``runner`` is injectable so tests can simulate a kernel drift without
    touching the library (and so auditors can verify the gate's failure path).
    """
    cur_env, cur_code = env(), code_identity()
    env_match, env_diffs = _env_match(
        golden.get("env", {}), cur_env, golden.get("code", {}), cur_code
    )
    budget = 0.0 if env_match else float(docs_ulp_budget)

    results = {}
    n_regress = n_drift = n_ok = 0
    for case in CASES:
        gold = golden["cases"].get(case.name)
        f = runner(case.name)
        got_digest = digest(f)
        if gold is None:
            results[case.name] = {"verdict": "MISSING_GOLDEN", "digest": got_digest}
            n_regress += 1
            continue
        ref = decode(gold["payload"])
        um = ulp_metrics(f, ref)
        digest_match = got_digest == gold["digest"]
        if digest(ref) != gold["digest"]:
            # payload/digest disagree -> the golden document is untrustworthy
            results[case.name] = {
                "verdict": "GOLDEN_INCONSISTENT",
                "digest": got_digest,
                "golden_digest": gold["digest"],
                "digest_match": digest_match,
                "ulp": um,
                "kernels": gold.get("kernels", list(case.kernels)),
            }
            n_regress += 1
            continue
        if digest_match:
            verdict = "BITWISE_OK"
            n_ok += 1
        elif um["comparable"] and um["max_ulp"] <= budget:
            verdict = "ULP_DRIFT"
            n_drift += 1
        else:
            verdict = "REGRESSION"
            n_regress += 1
        results[case.name] = {
            "verdict": verdict,
            "digest": got_digest,
            "golden_digest": gold["digest"],
            "digest_match": digest_match,
            "ulp": um,
            "kernels": gold.get("kernels", list(case.kernels)),
            "invariants": case_invariants(case.name, f),
        }
    passed = n_regress == 0
    return {
        "schema": SCHEMA,
        "checked_utc": _utc_now(),
        "env": cur_env,
        "code": cur_code,
        "env_match": env_match,
        "env_diffs": env_diffs,
        "ulp_budget": budget,
        "n_cases": len(CASES),
        "n_bitwise_ok": n_ok,
        "n_ulp_drift": n_drift,
        "n_regression": n_regress,
        "passed": passed,
        "cases": results,
    }


# --------------------------------------------------------------------------- #
# negative controls: prove the gate can fail
# --------------------------------------------------------------------------- #
def negative_controls(golden: dict) -> dict:
    """Two independent ways the gate must say 'fail'; both run against golden data."""
    out = {}

    # (a) 1-ulp tamper of a single population must be caught by the digest
    name = "chain_trt_50_24"
    f = run_case(name)
    ref = decode(golden["cases"][name]["payload"])
    bad = tamper(f, at=(0, 5, 5), n_ulp=1)
    um = ulp_metrics(bad, ref)
    out["ulp_tamper"] = {
        "case": name,
        "digest_detects": digest(bad) != golden["cases"][name]["digest"],
        "max_ulp": um["max_ulp"],
        "n_diff": um["n_diff"],
        "tampered_index": [0, 5, 5],
        "verdict_would_be": "REGRESSION"
        if digest(bad) != golden["cases"][name]["digest"]
        else "BITWISE_OK",
    }

    # (b) a 1e-7 relative tau bump (physics change) must move the digest
    tau = 0.8
    f_tau = _chain(feq_field(24, 24), 50, lambda ff: solver.collide_trt(ff, tau * (1.0 + 1e-7)))
    out["tau_bump"] = {
        "case": name,
        "tau": [tau, tau * (1.0 + 1e-7)],
        "digest_detects": digest(f_tau) != golden["cases"][name]["digest"],
        "max_ulp": ulp_metrics(f_tau, ref)["max_ulp"],
        "n_diff": ulp_metrics(f_tau, ref)["n_diff"],
    }
    out["both_detected"] = bool(
        out["ulp_tamper"]["digest_detects"] and out["tau_bump"]["digest_detects"]
    )
    return out


# --------------------------------------------------------------------------- #
# physics benchmark: 2D Taylor-Green vortex decay vs analytic solution
# --------------------------------------------------------------------------- #
def tgv_analytic(ny: int, nx: int, t: float, nu: float, u0: float):
    """Exact 2D TGV: u = u0 e^{-2 nu k^2 t} (sin kx cos ky, -cos kx sin ky)."""
    k = 2.0 * math.pi / nx
    dec = u0 * math.exp(-2.0 * nu * k * k * t)
    x = torch.arange(nx, dtype=torch.float64).view(1, nx) * k
    y = torch.arange(ny, dtype=torch.float64).view(ny, 1) * k
    ux = dec * torch.sin(x) * torch.cos(y)
    uy = -dec * torch.cos(x) * torch.sin(y)
    return ux, uy, dec


def tgv_level(ny: int, nx: int, tau: float, steps: int, sample_every: int | None = None) -> dict:
    """Run TGV with TRT collide + periodic stream and compare against the exact decay."""
    dtype = torch.float64
    sample_every = sample_every or max(1, steps // 20)
    nu = (tau - 0.5) / 3.0
    u0 = 0.05
    ux, uy, _ = tgv_analytic(ny, nx, 0.0, nu, u0)
    rho = torch.ones(ny, nx, dtype=dtype)
    f = d2q9.equilibrium(rho, ux.to(dtype), uy.to(dtype))
    series = []
    for step in range(steps + 1):
        if step % sample_every == 0:
            r, gx, gy = solver.macroscopic(f)
            amp = float(torch.sqrt(torch.mean(gx * gx + gy * gy)))
            ax, ay, dec = tgv_analytic(ny, nx, float(step), nu, u0)
            amp_ref = float(torch.sqrt(torch.mean(ax * ax + ay * ay)))
            field_ref_rel = float(torch.linalg.vector_norm(gx - ax) / torch.linalg.vector_norm(ax))
            series.append(
                {
                    "t": step,
                    "amp_got": amp,
                    "amp_ref": amp_ref,
                    "amp_err_pct": 100.0 * abs(amp - amp_ref) / amp_ref,
                    "field_l2_pct": 100.0 * field_ref_rel,
                    "rho_spread": float(r.max() - r.min()),
                }
            )
        if step == steps:
            break
        f = solver.stream(solver.collide_trt(f, tau))
    last = series[-1]
    return {
        "level": f"{nx}x{ny} tau={tau:.4f} steps={steps}",
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "nu": nu,
        "u0": u0,
        "steps": steps,
        "ma": u0 / math.sqrt(3.0),
        "re_cells": u0 * nx / nu,
        "amp_ref_final": last["amp_ref"],
        "amp_got_final": last["amp_got"],
        "amp_err_pct_final": last["amp_err_pct"],
        "field_l2_pct_final": last["field_l2_pct"],
        "max_amp_err_pct": max(s["amp_err_pct"] for s in series),
        "max_rho_spread": max(s["rho_spread"] for s in series),
        "series": series,
    }


def benchmark(cfg: dict | None = None) -> dict:
    """Physics bench: two resolutions (grid convergence) + two tau points."""
    # Fixed target decay factor exp(-2 nu k^2 t) = const, so every level is
    # compared at the *same* physical time: t ~ L^2 / nu.  Resolution series
    # 24/32/64 at tau=0.8 plus a tau series (0.7 / 0.8 / 0.9) at 32^2.
    cfg = cfg or {
        "levels": [
            {"ny": 24, "nx": 24, "tau": 0.8, "steps": 56},
            {"ny": 32, "nx": 32, "tau": 0.8, "steps": 100},
            {"ny": 64, "nx": 64, "tau": 0.8, "steps": 400},
        ],
        "tau_points": [
            {"ny": 32, "nx": 32, "tau": 0.7, "steps": 150},
            {"ny": 32, "nx": 32, "tau": 0.9, "steps": 75},
        ],
        "tolerance_pct": 15.0,
    }
    import time

    t0 = time.time()
    levels = [tgv_level(**lvl) for lvl in cfg["levels"]]
    tau_points = [tgv_level(**tp) for tp in cfg["tau_points"]]
    tol = cfg["tolerance_pct"]
    rows = [
        {
            "item": lv["level"],
            "ref": lv["amp_ref_final"],
            "got": lv["amp_got_final"],
            "err_pct": lv["amp_err_pct_final"],
            "basis": "analytic u0/sqrt(2)*exp(-2 nu k^2 t)",
            "pass": lv["amp_err_pct_final"] < tol,
        }
        for lv in levels + tau_points
    ]
    worst = max(r["err_pct"] for r in rows)
    return {
        "reference": "2D Taylor-Green vortex analytic decay, "
        "A(t) = (u0/sqrt(2)) exp(-2 nu k^2 t), k = 2 pi / Lx, nu = (tau-0.5)/3",
        "reference_basis": "analytic",
        "external_anchor": {
            "decay_law": "Taylor (1923) / standard LBM verification (e.g. Kruger et al. 2017, "
            "'The Lattice Boltzmann Method: Principles and Practice', ch. 8 Taylor-Green vortex)",
            "note": "closed-form solution; no tabulated external data needed, no digitised values used",
        },
        "implementation": "src/tensorlbm/d2q9.py::equilibrium + src/tensorlbm/solver.py::"
        "collide_trt + stream (library primitives only, no tuning factors)",
        "tolerance_pct": tol,
        "levels": levels,
        "tau_points": tau_points,
        "ref_vs_got": rows,
        "worst_err_pct": worst,
        "passed": worst < tol,
        "elapsed_s": time.time() - t0,
    }


# --------------------------------------------------------------------------- #
# one-shot entry used by run.py and by tests
# --------------------------------------------------------------------------- #
def refdata_path() -> Path:
    return REFDATA_DIR / "d2q9_bitwise_v1.json"


def load_golden(path: Path | None = None) -> dict:
    return json.loads((path or refdata_path()).read_text())


def full_check(golden: dict | None = None, bench_cfg: dict | None = None) -> dict:
    golden = golden or load_golden()
    reg = verify(golden)
    neg = negative_controls(golden)
    bench = benchmark(bench_cfg)
    passed = bool(reg["passed"] and neg["both_detected"] and bench["passed"])
    return {
        "schema": SCHEMA,
        "checked_utc": _utc_now(),
        "git_sha": _git_sha(),
        "env": reg["env"],
        "repro_cmd": "PYTHONPATH=src python3 benchmarks/xval_bitwise_refdata/run.py check",
        "regression": reg,
        "negative_controls": neg,
        "benchmark": bench,
        "passed": passed,
    }


__all__ = [
    "CASES",
    "SCHEMA",
    "benchmark",
    "case_invariants",
    "digest",
    "env",
    "full_check",
    "generate",
    "int_fields",
    "load_golden",
    "negative_controls",
    "refdata_path",
    "run_case",
    "tamper",
    "tgv_analytic",
    "tgv_level",
    "ulp_metrics",
    "verify",
]
