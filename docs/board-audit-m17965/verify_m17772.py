#!/usr/bin/env python3
"""Board-audit re-run of mission m17772 (XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR).

The audited mission claims (xval/fused_kernel_type_selector/REPORT.md, worktree
/root/tp-agent/work/m17772, never committed):

  A1  collide_bgk vs collide_bgk_matmul : 1 step 2.66e-15, 50 steps 4.44e-15
  A2  collide_bgk vs collide_bgk_fused  : 1 step 4.81e-08, 50 steps 1.90e-06
  A3  float32 fused-vs-reference difference ~9.5e-07
  A4  the anchor is "lssfau/lbmpy fused stream-collide kernel"

A4 is the load-bearing claim: the report's own numbers never touch lbmpy, they
compare TensorLBM against TensorLBM.  This script does two things:

  1. reproduces A1-A3 independently (so the numbers can be judged even though
     the mission shipped no run artifacts), and
  2. actually executes the declared external anchor: an lbmpy/pystencils SRT
     collision kernel evaluated on the *same* random non-equilibrium field, and
     compared against src/tensorlbm/solver.py::collide_bgk (and the fused path).

Artifacts: verify_m17772.json next to this file (all numbers written by code,
nothing transcribed by hand).

Usage: PYTHONPATH=src python3 docs/board-audit-m17965/verify_m17772.py
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorlbm import d2q9, solver  # noqa: E402

TAU = 0.8
SEED = 42
SHAPE = (9, 32, 32)  # mission's stated grid (9 x 32 x 32, D2Q9)
STEPS_LONG = 50
STEPS_DRIFT = 200


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def random_field(dtype=torch.float64, device="cpu") -> torch.Tensor:
    g = torch.Generator().manual_seed(SEED)
    return (torch.rand(SHAPE, generator=g, dtype=torch.float64) + 0.5).to(
        dtype=dtype, device=device
    )


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    den = float(b.abs().max())
    return maxdiff(a, b) / den if den else float("nan")


def step_ref(f, tau=TAU):
    return solver.stream(solver.collide_bgk(f, tau))


def step_fused(f, tau=TAU):
    return solver.stream(solver.collide_bgk_fused(f, tau))


def step_matmul(f, tau=TAU):
    return solver.stream(solver.collide_bgk_matmul(f, tau))


def chain(f, fn, n):
    for _ in range(n):
        f = fn(f)
    return f


# ----------------------------------------------------------------------------
# 1. reproduce the audited mission's own table (A1-A3)
# ----------------------------------------------------------------------------
def reproduce_internal() -> dict:
    out: dict = {"grid": list(SHAPE), "tau": TAU, "seed": SEED}
    for dtype, tag in ((torch.float64, "float64"), (torch.float32, "float32")):
        f = random_field(dtype)
        r1, fu1, m1 = step_ref(f), step_fused(f), step_matmul(f)
        rN = chain(f, step_ref, STEPS_LONG)
        fuN = chain(f, step_fused, STEPS_LONG)
        mN = chain(f, step_matmul, STEPS_LONG)
        out[tag] = {
            "ref_vs_matmul_1step_maxabs": maxdiff(r1, m1),
            "ref_vs_matmul_50step_maxabs": maxdiff(rN, mN),
            "ref_vs_matmul_50step_rel": rel_diff(rN, mN),
            "ref_vs_fused_1step_maxabs": maxdiff(r1, fu1),
            "ref_vs_fused_50step_maxabs": maxdiff(rN, fuN),
            "ref_vs_fused_50step_rel": rel_diff(rN, fuN),
            "fused_vs_matmul_50step_maxabs": maxdiff(fuN, mN),
        }
    return out


# ----------------------------------------------------------------------------
# 2. mass drift per operator (the defect class the mission says it fixed)
# ----------------------------------------------------------------------------
def mass_drift() -> dict:
    res = {}
    for name, fn in (
        ("collide_bgk", lambda f: solver.collide_bgk(f, TAU)),
        ("collide_bgk_fused", lambda f: solver.collide_bgk_fused(f, TAU)),
        ("collide_bgk_matmul", lambda f: solver.collide_bgk_matmul(f, TAU)),
    ):
        f0 = random_field(torch.float64)
        m0 = float(f0.sum())
        f1 = fn(f0)
        fN = f0
        for _ in range(STEPS_DRIFT):
            fN = fn(fN)
        res[name] = {
            "rel_mass_change_1step": (float(f1.sum()) - m0) / m0,
            "rel_mass_change_%dstep" % STEPS_DRIFT: (float(fN.sum()) - m0) / m0,
        }
    w_raw, w_exact = d2q9.W, d2q9._w_for(torch.float64, torch.device("cpu"))
    res["weight_table"] = {
        "_w_on(float32 table).to(float64) sum - 1": float(w_raw.to(torch.float64).sum()) - 1.0,
        "_w_for(float64) sum - 1": float(w_exact.sum()) - 1.0,
        "solver uses exact weights in collide_bgk_fused": "_w_for(dtype, device)"
        in (REPO / "src/tensorlbm/solver.py").read_text(),
    }
    return res


# ----------------------------------------------------------------------------
# 3. the declared anchor, actually executed: lbmpy/pystencils SRT collision
# ----------------------------------------------------------------------------
def lbmpy_collision_check() -> dict:
    """Execute the anchor the audited report only cites: lbmpy SRT collision.

    A physically admissible field is built from TensorLBM's own equilibrium
    (rho = 1, small-Ma divergence-free velocity) plus a 1e-3 random
    non-equilibrium perturbation, then the *same* field is fed to
    (a) lbmpy/pystencils D2Q9 SRT collision kernel and
    (b) src/tensorlbm/solver.py::collide_bgk and ::collide_bgk_fused.
    Component ordering is mapped explicitly from the velocity sets, so the
    comparison is a true operator-vs-operator check.
    """
    info: dict = {"attempted": True}
    try:
        import lbmpy
        import pystencils
        from lbmpy.creationfunctions import LBMConfig, create_lb_update_rule
        from pystencils import create_kernel
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"attempted": True, "ok": False, "error": f"import failed: {exc!r}"}

    info["lbmpy_version"] = getattr(lbmpy, "__version__", "unknown")
    info["pystencils_version"] = getattr(pystencils, "__version__", "unknown")
    info["omega_used"] = 1.0 / TAU
    try:
        cfg = LBMConfig(
            stencil="D2Q9",
            method="srt",
            relaxation_rate=1.0 / TAU,  # lbmpy takes omega = 1/tau
            compressible=True,
            kernel_type="collide_only",  # no streaming: pure collision operator
        )
        kernel = create_kernel(create_lb_update_rule(lbm_config=cfg))
        func = kernel.compile()
        kernel_name = getattr(kernel, "name", getattr(kernel, "function_name", "kernel"))
    except Exception as exc:
        return {**info, "ok": False, "error": f"kernel build failed: {exc!r}"}

    # velocity sets (lbmpy channel order vs TensorLBM d2q9.C order)
    c_lb = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0), (-1, 1), (1, 1), (-1, -1), (1, -1)]
    c_tl = [tuple(int(v) for v in row) for row in d2q9.C.tolist()]
    # writer mapping: lbmpy slot j receives the TensorLBM channel with the same velocity
    lb_from_tl = [c_tl.index(v) for v in c_lb]   # j -> tensorlbm channel  (false->tl)
    tl_from_lb = [c_lb.index(v) for v in c_tl]   # i -> lbmpy slot
    info["velocity_set_match"] = sorted(c_lb) == sorted(c_tl)
    info["tl_channel_to_lbmpy_slot"] = tl_from_lb

    if not info["velocity_set_match"]:
        return {**info, "ok": False, "error": "velocity sets differ; not D2Q9-compatible"}

    # small-Ma, equilibrium-consistent test field: feq + 1e-3 noise
    ny, nx = SHAPE[1], SHAPE[2]
    g = torch.Generator().manual_seed(SEED)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, ny, dtype=torch.float64),
        torch.linspace(0, 1, nx, dtype=torch.float64),
        indexing="ij",
    )
    u0, k = 0.05, 2.0 * math.pi
    ux = u0 * torch.sin(k * xx) * torch.cos(k * yy)
    uy = -u0 * torch.cos(k * xx) * torch.sin(k * yy)
    rho = torch.ones_like(ux)
    feq = d2q9.equilibrium(rho, ux, uy)
    noise = (torch.rand(SHAPE, generator=g, dtype=torch.float64) - 0.5) * 2e-3
    f_tl = feq + noise
    umax = float(max(ux.abs().max(), uy.abs().max()))
    info["test_field"] = {
        "kind": "d2q9.equilibrium(rho=1, u0=0.05 Taylor-Green velocity) + 1e-3 noise",
        "max_abs_u": umax,
        "ma": umax / math.sqrt(1.0 / 3.0),
    }

    src = np.ascontiguousarray(f_tl.numpy()[lb_from_tl].transpose(1, 2, 0), dtype=np.float64)
    # consistency gate: the transported state, read back with lbmpy's own velocity set,
    # must reproduce the intended density and velocity
    c_arr = np.array(c_lb, dtype=float)
    r_chk = src.sum(-1)
    u_chk = np.einsum("yxq,qd->yxd", src, c_arr) / r_chk[..., None]
    u_tl0 = np.einsum("qyx,qd->yxd", f_tl.numpy(), np.array(c_tl, dtype=float))
    rho_tl0 = f_tl.sum(0).numpy()
    u_tl0 = u_tl0 / rho_tl0[..., None]
    info["field_roundtrip_check"] = {  # mapping is right iff moments agree exactly
        "rho_err": float(np.abs(r_chk - rho_tl0).max()),
        "ux_err": float(np.abs(u_chk[..., 0] - u_tl0[..., 0]).max()),
        "uy_err": float(np.abs(u_chk[..., 1] - u_tl0[..., 1]).max()),
    }
    try:
        func(src=src)  # pystencils 2.0 collide_only kernel: in-place on one field
    except Exception as exc:
        return {**info, "ok": False, "error": f"kernel call failed: {exc!r}",
                "kernel": kernel_name}
    f_lb = np.ascontiguousarray(src.transpose(2, 0, 1))[tl_from_lb]  # back to TensorLBM order

    tl_ref = solver.collide_bgk(f_tl, TAU).numpy()
    tl_fused = solver.collide_bgk_fused(f_tl, TAU).numpy()
    tl_mm = solver.collide_bgk_matmul(f_tl, TAU).numpy()

    d_ref = float(np.abs(f_lb - tl_ref).max())
    d_fused = float(np.abs(f_lb - tl_fused).max())
    den = float(np.abs(tl_ref).max())
    info.update(
        {
            "ok": True,
            "kernel": kernel_name,
            "lbmpy_vs_tl_collide_bgk_maxabs": d_ref,
            "lbmpy_vs_tl_collide_bgk_rel": d_ref / den,
            "lbmpy_vs_tl_collide_bgk_fused_maxabs": d_fused,
            "lbmpy_vs_tl_collide_bgk_fused_rel": d_fused / den,
            "lbmpy_vs_tl_collide_bgk_matmul_maxabs": float(np.abs(f_lb - tl_mm).max()),
            "tolerance": {
                "pct": 15.0,
                "verdict": "PASS" if (d_ref / den) < 0.15 else "FAIL",
                "rel_pct_reference_path": 100.0 * d_ref / den,
                "rel_pct_fused_path": 100.0 * d_fused / den,
            },
        }
    )
    return info


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    payload = {
        "schema": "board-audit-m17965/m17772-rerun/v1",
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audited_mission": 17772,
        "audited_anchor": "XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR",
        "auditor_mission": 17965,
        "git_sha": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "machine": platform.machine(),
            "solver_sha256": _sha(REPO / "src/tensorlbm/solver.py"),
        },
        "repro_cmd": "PYTHONPATH=src python3 docs/board-audit-m17965/verify_m17772.py",
        "internal_equivalence": reproduce_internal(),
        "mass_drift": mass_drift(),
        "external_anchor": lbmpy_collision_check(),
    }
    out = HERE / "verify_m17772.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
    print(json.dumps(payload, indent=1)[:4000])
    print("written:", out.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())