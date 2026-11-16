#!/usr/bin/env python
"""m23154 / anchor LBM-P0-KBC: Taylor-Green decay audit of the KBC and Cumulant kernels.

Purpose
-------
Decide, with numbers rather than prose, whether the KBC (entropic) and Cumulant
collision families that ``advanced_collision_contract`` advertises as AVAILABLE
actually carry the shear viscosity they claim (``nu = c_s^2 (tau - 1/2)``).

Protocol (pre-registered in ``docs/m23154/PREREG.json``)
--------------------------------------------------------
* 3-D Taylor-Green vortex, periodic box N^3:
  ``u = (U0 sin kx cos ky cos kz, -U0 cos kx sin ky cos kz, 0)``, ``k = 2 pi/N``.
  Sum of the 8 plane waves with wave vector (+-k, +-k, +-k) => ``|kappa|^2 = 3k^2``,
  hence the kinetic-energy decay rate is ``gamma_E = 2 nu |kappa|^2 = 6 nu k^2``
  (identical for D3Q19 and D3Q27: both have ``c_s^2 = 1/3``).
* Single-relaxation-step order ``f <- collide(stream(f))``, float64 throughout
  (the fp32 protocol of ``benchmarks/pending/collision_kernels_3d_tg`` is
  re-run separately with ``--dtype float32`` for reconciliation).
* Sweep the relaxation time at fixed grid (tier A: N=32, tau in
  {0.6, 0.8, 1.0, 1.5}) and once on a finer grid (tier B: N=48, tau=0.7, Re=24).
  The tau sweep is the decisive test for "is the viscosity controlled by tau";
  the grid tier isolates discretisation from kernel error.

Gates (all must hold for a kernel to be reported VISCOSITY_CONTROLLED)
---------------------------------------------------------------------
G1 rate      |gamma_sim/gamma_theory - 1| <= 3%   for every (tau, N)
G2 exponence R^2 of the ln E(t) fit >= 0.999
G3 mass      |d rho / rho| <= 1e-12 (float64)
G4 positivity min f >= -1e-14 over the run   (entropic/KBC kernels only)

Everything that touches the physics is a library call
(``tensorlbm.<lattice>.equilibrium/macroscopic/streaming`` and the collision
operator under test); this script only builds the analytic initial field,
advances the loop and fits the decay.

Usage
-----
    python experiments/m23154_kbc_cumulant_tg.py --kernel kbc_d3q19 --n 32 --tau 0.8
    python experiments/m23154_kbc_cumulant_tg.py --kernel cumulant_d3q27 --n 32 --tau 0.6 \
        --out docs/m23154/runs/cumulant_d3q27_N32_tau0.6.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tensorlbm.cascaded_collision import collide_cascaded_d3q27  # noqa: E402
from tensorlbm.cumulant import (  # noqa: E402
    collide_cumulant_d3q27,
    collide_cumulant_geier_d3q27,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.d3q27 import (  # noqa: E402
    collide_mrt27,
    collide_rlbm27,
    collide_trt27,
    equilibrium27,
    macroscopic27,
    stream27_roll,
)
from tensorlbm.entropic_kbc import collide_kbc_d3q19, collide_kbc_d3q27  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402

U0 = 0.05

#: kernel registry -- library entry point under test, one per row.
KERNELS: dict[str, dict] = {
    "bgk_d3q19_ctl": dict(
        lattice="d3q19",
        op=lambda f, tau: collide_bgk3d(f, tau),
        entry="tensorlbm.solver3d.collide_bgk3d",
        family="BGK",
        control=True,
    ),
    "kbc_d3q19": dict(
        lattice="d3q19",
        op=lambda f, tau: collide_kbc_d3q19(f, tau),
        entry="tensorlbm.entropic_kbc.collide_kbc_d3q19",
        family="KBC",
        entropy=True,
    ),
    "kbc_natural_d3q19": dict(
        lattice="d3q19",
        op=None,  # filled in below (natural-KBC is imported lazily)
        entry="tensorlbm.entropic_kbc.collide_natural_kbc_d3q19",
        family="KBC",
        entropy=True,
    ),
    "kbc_d3q27": dict(
        lattice="d3q27",
        op=lambda f, tau: collide_kbc_d3q27(f, tau),
        entry="tensorlbm.entropic_kbc.collide_kbc_d3q27",
        family="KBC",
        entropy=True,
    ),
    "cumulant_d3q27": dict(
        lattice="d3q27",
        op=lambda f, tau: collide_cumulant_d3q27(f, tau),
        entry="tensorlbm.cumulant.collide_cumulant_d3q27",
        family="CUMULANT",
    ),
    "cumulant_geier_d3q27": dict(
        lattice="d3q27",
        op=lambda f, tau: collide_cumulant_geier_d3q27(f, tau),
        entry="tensorlbm.cumulant.collide_cumulant_geier_d3q27",
        family="CUMULANT",
    ),
    "mrt_d3q27_ctl": dict(
        lattice="d3q27",
        op=lambda f, tau: collide_mrt27(f, tau),
        entry="tensorlbm.d3q27.collide_mrt27",
        family="MRT",
        control=True,
    ),
    "trt_d3q27_ctl": dict(
        lattice="d3q27",
        op=lambda f, tau: collide_trt27(f, tau),
        entry="tensorlbm.d3q27.collide_trt27",
        family="TRT",
        control=True,
    ),
    "rlbm_d3q27_ctl": dict(
        lattice="d3q27",
        op=lambda f, tau: collide_rlbm27(f, tau),
        entry="tensorlbm.d3q27.collide_rlbm27",
        family="RLBM",
        control=True,
    ),
    "cascaded_d3q27_ctl": dict(
        lattice="d3q27",
        op=lambda f, tau: collide_cascaded_d3q27(f, tau),
        entry="tensorlbm.cascaded_collision.collide_cascaded_d3q27",
        family="CM",
        control=True,
    ),
}

LATTICE = {
    "d3q19": dict(equilibrium=equilibrium3d, macroscopic=macroscopic3d, streaming=stream3d),
    "d3q27": dict(
        equilibrium=equilibrium27,
        macroscopic=macroscopic27,
        streaming=stream27_roll,
    ),
}


def _natural_d3q19(f: torch.Tensor, tau: float) -> torch.Tensor:  # pragma: no cover - imported lazily
    from tensorlbm.entropic_kbc import collide_natural_kbc_d3q19

    return collide_natural_kbc_d3q19(f, tau)


KERNELS["kbc_natural_d3q19"]["op"] = _natural_d3q19


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_field(n: int, u0: float, dtype: torch.dtype, device: str):
    """Analytic TG initial velocity field -> library equilibrium."""
    idx = torch.arange(n, dtype=dtype, device=device)
    z, y, x = torch.meshgrid(idx, idx, idx, indexing="ij")
    k = 2.0 * math.pi / n
    ux = u0 * torch.sin(k * x) * torch.cos(k * y) * torch.cos(k * z)
    uy = -u0 * torch.cos(k * x) * torch.sin(k * y) * torch.cos(k * z)
    uz = torch.zeros_like(ux)
    rho = torch.ones_like(ux)
    return rho, ux, uy, uz


def run_case(
    kernel: str,
    n: int,
    tau: float,
    *,
    u0: float = U0,
    dtype: str = "float64",
    efolds: float = 10.0,
    record_every: int = 10,
    min_steps: int = 100,
    max_steps: int = 6000,
    device: str = "cpu",
    threads: int = 0,
) -> dict:
    if threads:
        torch.set_num_threads(threads)
    info = KERNELS[kernel]
    lat = LATTICE[info["lattice"]]
    dt = torch.float64 if dtype == "float64" else torch.float32

    nu = (tau - 0.5) / 3.0
    k = 2.0 * math.pi / n
    gamma_theory = 6.0 * nu * k * k

    steps = int(math.ceil(efolds / gamma_theory)) if gamma_theory > 0 else max_steps
    steps = min(max_steps, max(min_steps, (steps // record_every + 1) * record_every))

    rho, ux, uy, uz = build_field(n, u0, dt, device)
    f = lat["equilibrium"](rho, ux, uy, uz)
    mass0 = float(f.sum().item())
    e0 = float((0.5 * (ux * ux + uy * uy + uz * uz)).mean().item())

    op = info["op"]
    t_list: list[int] = [0]
    e_list: list[float] = [e0]
    fmin = float(f.min().item())
    wall0 = time.time()
    for step in range(1, steps + 1):
        f = op(lat["streaming"](f), tau)
        if info.get("entropy"):
            fmin = min(fmin, float(f.min().item()))
        if step % record_every == 0:
            _, ax, ay, az = lat["macroscopic"](f)
            e_list.append(float((0.5 * (ax * ax + ay * ay + az * az)).mean().item()))
            t_list.append(step)
    wall = time.time() - wall0
    mass_end = float(f.sum().item())

    # fit window: drop the first two recorded samples (startup) and everything
    # below E/E0 = 1e-8 (noise floor guard).
    t = np.asarray(t_list, dtype=np.float64)
    e = np.asarray(e_list, dtype=np.float64)
    mask = (t > 0) & (e > 0)
    if e0 > 0:
        mask &= e / max(abs(e0), 1e-300) > 1e-8
    idx = np.nonzero(mask)[0]
    idx = idx[2:] if idx.size > 4 else idx
    t_fit, e_fit = t[idx], e[idx]
    ln_e = np.log(e_fit)
    slope, intercept = np.polyfit(t_fit, ln_e, 1)
    gamma_sim = -float(slope)
    resid = ln_e - (intercept + slope * t_fit)
    ss_tot = float(np.sum((ln_e - ln_e.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2) / ss_tot) if ss_tot > 0 else float("nan")

    nu_sim = gamma_sim / (6.0 * k * k)
    return dict(
        kernel=kernel,
        family=info["family"],
        entry=info["entry"],
        lattice=info["lattice"],
        n=n,
        tau=tau,
        nu_theory=nu,
        k=k,
        u0=u0,
        dtype=dtype,
        device=device,
        steps=steps,
        record_every=record_every,
        gamma_theory=gamma_theory,
        gamma_sim=gamma_sim,
        err_pct=(gamma_sim - gamma_theory) / gamma_theory * 100.0,
        nu_sim=nu_sim,
        nu_err_pct=(nu_sim - nu) / nu * 100.0,
        r2=r2,
        n_fit=int(t_fit.size),
        finite=bool(np.isfinite(e).all()),
        e0=e0,
        e_end=float(e[-1]),
        mass_drift_rel=(mass_end - mass0) / mass0,
        f_min=fmin,
        wall_sec=wall,
        series=dict(t=t_list, e=e_list),
        meta=dict(
            command=" ".join(sys.argv),
            git_head=_git_head(),
            torch=torch.__version__,
            numpy=np.__version__,
            python=platform.python_version(),
            threads=torch.get_num_threads(),
        ),
    )


def gates(rec: dict) -> dict:
    """Pre-registered gates G1-G4 for one run."""
    entropy = KERNELS[rec["kernel"]].get("entropy", False)
    g = {
        "G1_rate_le_3pct": abs(rec["err_pct"]) <= 3.0,
        "G2_r2_ge_0.999": rec["r2"] >= 0.999,
        "G3_mass_le_1e-12": abs(rec["mass_drift_rel"]) <= 1e-12,
        "G4_positivity": (rec["f_min"] >= -1e-14) if entropy else True,
    }
    g["all_pass"] = all(g.values())
    return g


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernel", required=True, choices=sorted(KERNELS))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--u0", type=float, default=U0)
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rec = run_case(
        args.kernel,
        args.n,
        args.tau,
        u0=args.u0,
        dtype=args.dtype,
        device=args.device,
        threads=args.threads,
    )
    rec["gates"] = gates(rec)
    printable = {k: v for k, v in rec.items() if k != "series"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        print("wrote", args.out)
    return 0 if rec["gates"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())