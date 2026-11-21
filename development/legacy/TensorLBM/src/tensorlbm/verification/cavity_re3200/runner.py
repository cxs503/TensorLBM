"""Grid runner for the Re = 3200 lid-driven cavity (anchor ``REPRO-LBM-HOU1995``).

This is the missing *executable* half of the Re = 3200 reproduction: the earlier
round shipped only the scorer, so every grid was ``WITHHELD``.  The runner drives
the library's own D2Q9 solver
(:class:`tensorlbm.verification.lid_driven_cavity.solver.CavitySolver`) and emits
a machine-readable record per grid -- the same schema the analyser consumes.

Physical set-up (Hou, Zou, Chen & Doolen 1995, *J. Comput. Phys.* **118** 329-347,
the ``Re = 3200`` steady cavity they report against Ghia et al. 1982)::

    L = n - 1 lattice units,  U_lid = 0.1,  Re = U_lid * L / nu = 3200
    => nu = 0.1 * (n-1) / 3200,  tau = 3 nu + 1/2   (n=256 -> tau = 0.5239)

Collision is regularised BGK (RLBM, Latt & Chopard 2006) because ``tau`` is close
to 1/2; walls are half-way bounce-back (the lid is a moving wall through the
momentum-transfer factor), which fixes the physical coordinate of the wall at
``(j - 0.5)/(n - 2)`` -- the ``wall`` convention of :mod:`.scoring`.

Contract (pinned by ``tests/test_cavity_re3200_runner.py``)
-----------------------------------------------------------
* one call returns one dict; it never raises for a *physical* outcome, and it
  never reports a pass it did not compute;
* ``diverged`` / non-finite fields => ``run_status`` ``"diverged"`` /
  ``"non_finite"`` and no invented centreline;
* an invalid configuration (``n < 17``, ``tau <= 0.5``) => ``run_status``
  ``"config_error"`` with the exception text, still a well-formed record;
* ``resid_tol`` unset => ``converged`` is ``False`` and the run must not be
  quoted as a steady solution (the step count would be an arbitrary truncation);
* ``ref`` given => every grid additionally carries ``comparison`` with the same
  pointwise rows the analyser produces (single source of truth: :mod:`.scoring`).

Grid sequencing ("``--init-from``") is supported because a fresh start at
``n = 256`` needs an O(10^5)-step transient: a converged coarse run's velocity
field can be prolonged bilinearly onto the finer lattice and used as the initial
condition.  The record says so (``init_from``), so the reader can tell a
sequenced run from a cold start.
"""

from __future__ import annotations

import json
import os
import platform as _platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

CASE = "cavity_re3200_ghia"
ANCHOR = "REPRO-LBM-HOU1995"
RE = 3200.0
U_LID = 0.1
TOL_PCT = 5.0

DESCRIPTION = (
    "2-D lid-driven square cavity, Re = 3200, compared pointwise with the "
    "Re = 3200 column of Ghia, Ghia & Shin (1982) Table I/II"
)


def _import_torch():
    try:
        import torch  # noqa: WPS433 (deliberate: keeps import errors reportable)
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        raise RuntimeError(f"PyTorch is required for the cavity runner: {exc}") from exc
    return torch


def centerline(ux, uy, n: int, center_average: bool = True):
    """Sample ``u`` on ``x/L = 0.5`` and ``v`` on ``y/L = 0.5``.

    ``ux``/``uy`` are ``(n, n)`` arrays with the first axis along ``y`` (row
    ``n-1`` is the moving lid).  For an even ``n`` the geometric centre falls
    between two lattice planes, so the two central planes are averaged (as in the
    Re = 5000 sibling benchmark); for an odd ``n`` the middle row is exact.
    """
    import torch

    if n % 2 == 1:
        j = n // 2
        u = ux[:, j]
        v = uy[j, :]
    elif center_average:
        j0, j1 = n // 2 - 1, n // 2
        u = 0.5 * (ux[:, j0] + ux[:, j1])
        v = 0.5 * (uy[j0, :] + uy[j1, :])
    else:
        j = n // 2
        u = ux[:, j]
        v = uy[j, :]
    return u.detach().cpu().numpy().astype(float), v.detach().cpu().numpy().astype(float)


def _load_fields(path: str | Path, n: int):
    """Read a coarse velocity field and prolongate it onto the ``n x n`` lattice."""
    import torch

    data = np.load(Path(path))
    ux0 = torch.as_tensor(data["ux"], dtype=torch.float64)[None, None, :, :]
    uy0 = torch.as_tensor(data["uy"], dtype=torch.float64)[None, None, :, :]
    if ux0.shape[-1] != n:
        ux0 = torch.nn.functional.interpolate(ux0, size=(n, n), mode="bilinear",
                                              align_corners=True)
        uy0 = torch.nn.functional.interpolate(uy0, size=(n, n), mode="bilinear",
                                              align_corners=True)
    return ux0[0, 0], uy0[0, 0]


def _environment(torch_mod, threads: Optional[int]) -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch_mod.__version__,
        "torch_threads": int(torch_mod.get_num_threads()) if threads is None else threads,
        "host": _platform.node(),
        "machine": _platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def run_grid(n: int, re: float = RE, u_lid: float = U_LID, *, max_steps: int = 5000,
             monitor_every: int = 1000, resid_tol: Optional[float] = None,
             use_rlbm: bool = True, ramp_steps: int = 1000, threads: Optional[int] = None,
             center_average: bool = True, coord_convention: str = "wall",
             ref: Optional[Dict[str, Any]] = None, init_from: Optional[str] = None,
             save_fields: Optional[str] = None, verbose: bool = False,
             tol_pct: float = TOL_PCT) -> Dict[str, Any]:
    """Run one ``n x n`` grid and return a self-describing record."""
    torch = _import_torch()
    if threads is not None:
        torch.set_num_threads(int(threads))

    from tensorlbm.verification.lid_driven_cavity.solver import CavityConfig, CavitySolver
    from tensorlbm.verification.cavity_re3200 import scoring

    rec: Dict[str, Any] = {
        "case": CASE,
        "anchor": ANCHOR,
        "description": DESCRIPTION,
        "lattice": "D2Q9",
        "boundary": "half-way bounce-back walls + moving lid (momentum transfer)",
        "tolerance_pct": float(tol_pct),
        "re": float(re),
        "u_lid": float(u_lid),
        "n": int(n),
        "center_average": bool(center_average),
        "coord_convention": coord_convention,
        "init_from": str(init_from) if init_from else None,
        "environment": _environment(torch, threads),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if ref is not None:
        rec["ref_file"] = ref.get("_path")
        rec["ref_sha256"] = ref.get("_sha256_file")
        rec["ref_content_sha256"] = ref.get("_sha256_content")

    try:
        cfg = CavityConfig(n=int(n), re=float(re), u_lid=float(u_lid),
                           use_rlbm=bool(use_rlbm), ramp_steps=int(ramp_steps))
        solver = CavitySolver(cfg)
    except Exception as exc:
        rec.update(run_status="config_error", error=f"{type(exc).__name__}: {exc}",
                   steps=0, elapsed_s=0.0, resid_history=[], final_resid=None,
                   final_resid_rel=None, diverged=False, converged=False,
                   u_centerline=None, v_centerline=None, u_min=None, v_max=None,
                   comparison={"status": "VOID", "reason": "config_error"})
        if verbose:
            print(f"  n={n} CONFIG_ERROR {exc}", flush=True)
        return rec

    rec.update(tau=float(cfg.tau), nu_lattice=float(cfg.nu),
               collision="rlbm (regularised BGK, Latt & Chopard 2006)"
               if use_rlbm else "bgk",
               ramp_steps=int(ramp_steps))

    if init_from:
        ux0, uy0 = _load_fields(init_from, solver.n)
        rho0 = torch.ones((solver.n, solver.n), dtype=solver.dtype, device=solver.device)
        ux0 = ux0.to(solver.dtype).to(solver.device)
        uy0 = uy0.to(solver.dtype).to(solver.device)
        # walls stay at rest; the lid is imposed by the boundary treatment
        ux0[solver.solid] = 0.0
        uy0[solver.solid] = 0.0
        solver.f = solver._equilibrium(rho0, ux0, uy0)
        solver._t = 0

    t0 = time.time()
    history = []
    converged = False
    diverged = False
    step = 0
    _, ux_prev, uy_prev = solver._macroscopic(solver.f)
    for step in range(1, int(max_steps) + 1):
        solver.step()
        if monitor_every and step % int(monitor_every) == 0:
            _, ux, uy = solver._macroscopic(solver.f)
            with torch.no_grad():
                du = float(torch.max(torch.abs(ux[solver.fluid] - ux_prev[solver.fluid])))
                dv = float(torch.max(torch.abs(uy[solver.fluid] - uy_prev[solver.fluid])))
                finite = bool(torch.all(torch.isfinite(ux[solver.fluid]))) and \
                    bool(torch.all(torch.isfinite(uy[solver.fluid])))
            resid = max(du, dv)
            ux_prev, uy_prev = ux.clone(), uy.clone()
            if not finite or not np.isfinite(resid):
                diverged = True
                history.append([step, None, None, None])
                break
            u_now, v_now = centerline(ux, uy, n, center_average)
            u_now, v_now = u_now / u_lid, v_now / u_lid
            history.append([step, resid, resid / u_lid,
                            float(max(np.abs(u_now).max(), np.abs(v_now).max()))])
            if verbose:
                print(f"  n={n} step={step} max|du|={resid:.3e} "
                      f"(rel {resid / u_lid:.2e})", flush=True)
            if resid_tol is not None and resid < float(resid_tol):
                converged = True
                break

    elapsed = time.time() - t0
    _, ux, uy = solver._macroscopic(solver.f)
    uc, vc = centerline(ux, uy, n, center_average)
    uc, vc = uc / u_lid, vc / u_lid
    finite = bool(np.all(np.isfinite(uc)) and np.all(np.isfinite(vc)))
    if diverged or not finite:
        status = "diverged" if diverged else "non_finite"
    elif converged:
        status = "ok"
    elif resid_tol is not None:
        status = "not_converged"
    else:
        status = "ok"

    y_node = scoring.coords(n, "node")
    y_wall = scoring.coords(n, "wall")
    if finite:
        rec.update(u_min=float(uc.min()), u_min_y=float(y_wall[int(np.argmin(uc))]),
                   v_max=float(vc.max()), v_max_x=float(y_wall[int(np.argmax(vc))]))
    else:
        rec.update(u_min=None, u_min_y=None, v_max=None, v_max_x=None)
    rec.update(
        run_status=status,
        converged=bool(converged),
        diverged=bool(diverged or not finite),
        steps=int(step),
        elapsed_s=round(elapsed, 1),
        monitor_every=int(monitor_every or 0),
        resid_tol=None if resid_tol is None else float(resid_tol),
        resid_history=history,
        final_resid=(history[-1][1] if history else None),
        final_resid_rel=(history[-1][2] if history else None),
        u_centerline=uc.tolist() if finite else None,
        v_centerline=vc.tolist() if finite else None,
        y_centerline_node=y_node.tolist(),
        x_centerline_node=y_node.tolist(),
        y_centerline_wall=y_wall.tolist(),
        x_centerline_wall=y_wall.tolist(),
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )

    if save_fields and finite:
        np.savez_compressed(save_fields,
                            ux=ux.detach().cpu().numpy().astype(np.float32),
                            uy=uy.detach().cpu().numpy().astype(np.float32))
        rec["fields_file"] = str(save_fields)

    if ref is not None:
        if finite:
            rec["comparison"] = {
                "tol_pct": float(tol_pct),
                **{conv: scoring.score(rec, ref, conv, tol_pct)
                   for conv in scoring.CONVENTIONS},
                "quarantine_sensitivity": scoring.quarantine_sensitivity(
                    rec, ref, coord_convention),
            }
        else:
            rec["comparison"] = {"status": "VOID", "reason": status}
    else:
        rec["comparison"] = {"status": "WITHHELD", "reason": "no reference supplied"}

    if verbose:
        cmp_ = rec["comparison"]
        if "tol_pct" in cmp_:
            print(f"  n={n} {status} steps={step} err({coord_convention})="
                  f"{cmp_[coord_convention]['max_abs_err_pct']:.2f}%", flush=True)
        else:
            print(f"  n={n} {status} steps={step} comparison {cmp_.get('status')}",
                  flush=True)
    return rec


def write_record(rec: Dict[str, Any], out_dir: str | Path) -> Path:
    """Write ``re3200_n<n>.json`` (the name the analyser reads) plus a hash sidecar."""
    import hashlib

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"re3200_n{rec['n']}.json"
    payload = json.dumps(rec, indent=1, sort_keys=False)
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (out / f"re3200_n{rec['n']}.sha256").write_text(f"{digest}  {path.name}\n",
                                                    encoding="utf-8")
    return path