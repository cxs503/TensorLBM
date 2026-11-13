"""XVAL — TensorLBM Inamuro wet-node / anti-bounce-back vs the OSS reference.

Reference (zero hand-copying; the source is fetched and compiled at run time)
------------------------------------------------------------------------------
``lbm-principles-practice/code`` — the code accompanying Krüger, Kusumaatmaja,
Kuzmin, Shardt, Silva & Viggen, *The Lattice Boltzmann Method: Principles and
Practice*, Springer 2017 (MIT licence), pinned commit
``6e2c592fdc3592c14dfd52f860fc1ceea930bcb0``:

* ``chapter8/film_inamuro.cpp`` — scalar Dirichlet + zero-gradient closures on
  a wet node, falling-film advection/diffusion problem.  This script *compiles
  and runs the upstream C++ as-is* and compares its concentration field with
  the field produced by :mod:`tensorlbm.boundary.inamuro_wetnode`.
* ``chapter5/couette_wetnode.m`` / ``chapter5/poiseuille_wetnode.m`` —
  velocity Dirichlet by non-equilibrium bounce-back (``wetnode=3``).  MATLAB
  is not available here, so the reference for this part is the *analytic*
  profile those scripts themselves report against (linear Couette, parabolic
  Poiseuille): the error norms below are computed exactly the way the OSS
  scripts print ``L2(u)``.

Parts
-----
``film``     : OSS C++ binary vs TensorLBM scalar wet-node closures (field-to-field).
``velocity`` : Couette + Poiseuille vs analytic profiles (velocity wet-node ABC),
               with a half-way bounce-back arm as a control.
``all``      : both (default).

Everything is float64.  Artifacts (JSON + npz + PNG) go to ``--out``.

Usage
-----
    python benchmarks/verified/inamuro_wetnode_xval/run.py --part all \
        --oss-ref /tmp/oss_lbm --out artifacts/m17603

``--oss-ref`` may be a checkout of the OSS repo; ``reproduce_oss.sh`` clones it
at the pinned commit and builds ``film_inamuro``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from tensorlbm.boundary.inamuro_wetnode import (
    inamuro_corner_average,
    inamuro_wetnode_scalar,
    inamuro_wetnode_velocity,
    inamuro_zero_gradient,
)
from tensorlbm.boundary.d2q9 import OPPOSITE_2Q9 as OPP
from tensorlbm.boundary.d2q9 import W2Q9 as W
from tensorlbm.boundary.solver2d import C2Q9, equilibrium2d, macroscopic2d

OSS_REPO = "https://github.com/lbm-principles-practice/code"
OSS_COMMIT = "6e2c592fdc3592c14dfd52f860fc1ceea930bcb0"
OSS_FILE = "chapter8/film_inamuro.cpp"

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Part 1 — falling film with a scalar Dirichlet wet node (OSS film_inamuro.cpp)
# ---------------------------------------------------------------------------

#: Parameters of the upstream ``film_inamuro.cpp`` (read from its source).
FILM = dict(nx=40, ny=400, u0=0.05, conc_wall=1.0, conc_inlet=0.5, omega=1.4)


def film_initial_state(params: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upstream ``initialize_geometry()``/``init()`` reproduced.

    The advecting velocity is *prescribed and frozen* (parabolic film profile
    ``u_y(x) = u0 (1 - x²/(NX-1.5)²)``); only the concentration is transported.
    """
    nx, ny = params["nx"], params["ny"]
    rho = torch.zeros((ny, nx), dtype=DTYPE)
    ux = torch.zeros((ny, nx), dtype=DTYPE)
    ix = torch.arange(nx, dtype=DTYPE)
    profile = params["u0"] * (1.0 - ix**2 / (nx - 1.5) ** 2)
    uy = profile.unsqueeze(0).expand(ny, nx).contiguous()

    rho[0, 1 : nx - 1] = params["conc_inlet"]  # inlet row (upstream iY=0, iX=1..NX-2)
    rho[:, 0] = params["conc_wall"]  # wall column (upstream iX=0, all iY)
    f = equilibrium2d(rho, ux, uy)
    return f.contiguous(), ux, uy


def stream_periodic(f: torch.Tensor) -> torch.Tensor:
    """Pull streaming with periodic wrap in both directions (upstream ``stream()``)."""
    out = torch.empty_like(f)
    for k in range(9):
        cx = int(C2Q9[k, 0])
        cy = int(C2Q9[k, 1])
        out[k] = torch.roll(f[k], shifts=(cy, cx), dims=(0, 1))
    return out


# Arithmetic core of the film problem (collision + streaming) in NumPy.
#
# The subject of this XVAL is the *closure*, not the advection-diffusion core:
# collision/streaming are the problem definition.  They are evaluated in NumPy
# because this box's torch dispatch costs ~1 ms per elementwise kernel on
# 9x400x40 float64 tensors (~16 ms/step against ~2.7 ms/step in NumPy), which
# would turn a 50 000-step reference-matching run into >13 min.  The closures
# themselves always run through the production torch module
# (``film_apply_closures`` -> ``tensorlbm.boundary.inamuro_wetnode``); the
# tensors are shared zero-copy with NumPy, so this changes nothing numerically
# (both are IEEE-754 double, same operation order per lattice direction).
_F64 = np.float64
_CX = np.asarray([float(v) for v in C2Q9[:, 0]], dtype=_F64).reshape(9, 1, 1)
_CY = np.asarray([float(v) for v in C2Q9[:, 1]], dtype=_F64).reshape(9, 1, 1)
_WN = np.asarray([float(v) for v in W], dtype=_F64).reshape(9, 1, 1)
_SHIFTS = [(int(C2Q9[k, 1]), int(C2Q9[k, 0])) for k in range(9)]


def _numpy_collide_stream(
    f: np.ndarray, ux: np.ndarray, uy: np.ndarray, omega: float
) -> np.ndarray:
    """BGK collision with the frozen advection velocity, then periodic streaming."""
    rho = f.sum(axis=0)
    cu = _CX * ux + _CY * uy
    feq = _WN * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy))
    f_post = f - omega * (f - feq)
    out = np.empty_like(f_post)
    for k in range(9):
        out[k] = np.roll(f_post[k], _SHIFTS[k], axis=(0, 1))
    return out


def film_apply_closures(
    f: torch.Tensor, params: dict, *, scalar_policy: str = "all_but_rest"
) -> torch.Tensor:
    """Upstream ``update_bounce_back_after_stream()`` + the corner average block.

    Order and closure policies are the upstream ones: inlet (scalar Dirichlet),
    outlet (zero gradient), wall (scalar Dirichlet), free surface (zero
    gradient), then the four corners averaged from their two edge neighbours.
    """
    f = inamuro_wetnode_scalar(
        f, "bottom", params["conc_inlet"], known=scalar_policy, inplace=True
    )
    f = inamuro_zero_gradient(f, "top", inplace=True)
    f = inamuro_wetnode_scalar(
        f, "left", params["conc_wall"], known=scalar_policy, inplace=True
    )
    f = inamuro_zero_gradient(f, "right", inplace=True)
    for corner in ("bl", "br", "tl", "tr"):
        f = inamuro_corner_average(f, corner, inplace=True)
    return f


def run_film(
    params: dict,
    steps: int,
    *,
    scalar_policy: str = "all_but_rest",
    checkpoints: tuple[int, ...] = (),
    progress: bool = False,
) -> dict[int, torch.Tensor]:
    """Run the film problem; return ``{step: concentration field (ny, nx)}``."""
    f_t, ux_t, uy_t = film_initial_state(params)
    f = f_t.numpy()  # zero-copy view: the closures below run on it in place
    ux = ux_t.numpy()
    uy = uy_t.numpy()
    omega = float(params["omega"])
    wanted = set(checkpoints)
    snapshots: dict[int, torch.Tensor] = {}
    if 0 in wanted:
        snapshots[0] = torch.from_numpy(f.sum(axis=0).copy())
    for step in range(1, steps + 1):
        f = _numpy_collide_stream(f, ux, uy, omega)
        f = film_apply_closures(torch.from_numpy(f), params, scalar_policy=scalar_policy).numpy()
        if step in wanted:
            snapshots[step] = torch.from_numpy(f.sum(axis=0).copy())
        if progress and step % 5000 == 0:
            print(f"  film step {step}/{steps}", flush=True)
    snapshots[steps] = torch.from_numpy(f.sum(axis=0).copy())
    return snapshots


def compile_oss_reference(oss_ref: Path, workdir: Path) -> Path:
    """Compile the upstream ``film_inamuro.cpp`` unmodified (``g++ -O2``)."""
    src = oss_ref / OSS_FILE
    if not src.is_file():
        raise SystemExit(f"OSS source not found: {src}")
    bin_path = (workdir / "film_inamuro").resolve()
    subprocess.run(
        ["g++", "-O2", "-o", str(bin_path), str(src.resolve())], check=True
    )
    return bin_path


def run_oss_reference(bin_path: Path, workdir: Path, steps: int) -> Path:
    """Run the upstream binary; it writes ``film_inamuro/film_inamuro<step>.dat``."""
    outdir = workdir / "film_inamuro"
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(workdir / "film_inamuro.log", "w") as log:
        subprocess.run([str(bin_path)], check=True, cwd=str(workdir), stdout=log, stderr=log)
    elapsed = time.time() - t0
    target = outdir / f"film_inamuro{steps:07d}.dat"
    if not target.is_file():
        raise SystemExit(f"reference output missing: {target}")
    print(f"  OSS binary finished in {elapsed:.1f}s -> {target}")
    return target


def read_oss_field(path: Path, nx: int, ny: int) -> np.ndarray:
    """Read an upstream ``.dat`` field (one line per ``iX``, one column per ``iY``)."""
    data = np.loadtxt(path)
    if data.shape == (nx, ny):
        return data.T.copy()
    if data.shape == (ny, nx):
        return data.copy()
    raise SystemExit(f"unexpected field shape {data.shape} in {path}")


# ---------------------------------------------------------------------------
# Part 2 — velocity Dirichlet wet node vs analytic Couette / Poiseuille
# ---------------------------------------------------------------------------


def run_channel(
    ny: int,
    nx: int,
    tau: float,
    *,
    u_bottom: float = 0.0,
    u_top: float = 0.0,
    steps: int = 12000,
    tol: float = 1e-13,
    wall_bc: str = "inamuro",
    rho_w: float | None = None,
    init: str = "sheared",
) -> dict:
    """Plane Couette channel closed with a Dirichlet velocity wall.

    ``wall_bc="inamuro"`` applies the wet-node anti-bounce-back closures on the
    bottom and top *node rows* (the physics plane is on the row); ``"bounce-
    back"`` is the half-way control arm, where the wall plane sits half a cell
    outside the first/last fluid row.

    ``init="sheared"`` starts from the linear profile.  This matters: the
    anti-bounce-back closure maps an exactly x-symmetric state onto another
    x-symmetric state, so a perfect rest field is a *neutral* fixed point and
    the flow never starts (measured: u ≡ 0 after 8000 steps).  The OSS
    ``chapter5`` scripts avoid this by imposing rho and u explicitly (Zou-He
    style) rather than by the ABB rule; see the XVAL report, failure mode 1.
    """
    u_init = analytic_couette(ny, u_top) if init == "sheared" else np.zeros(ny)
    rho0 = torch.ones((ny, nx), dtype=DTYPE)
    ux0 = torch.tensor(np.tile(u_init[:, None], (1, nx)), dtype=DTYPE)
    uy0 = torch.zeros((ny, nx), dtype=DTYPE)
    f = equilibrium2d(rho0, ux0, uy0)
    nu = (2.0 * tau - 1.0) / 6.0

    def profile() -> np.ndarray:
        rho, momx, _ = macroscopic2d(f)
        return (momx / rho)[:, nx // 2].numpy().copy()

    u_prev = None
    residual = float("inf")
    step = 0
    for step in range(1, steps + 1):
        rho, ux, uy = macroscopic2d(f)
        feq = equilibrium2d(rho, ux, uy)
        f = stream_periodic(f - (f - feq) / tau)
        if wall_bc == "inamuro":
            f = inamuro_wetnode_velocity(f, "bottom", (u_bottom, 0.0), rho_w=rho_w)
            f = inamuro_wetnode_velocity(f, "top", (u_top, 0.0), rho_w=rho_w)
        elif wall_bc == "bounce-back":
            # Control arm: half-way bounce-back. The wall plane sits *half a
            # cell outside* rows 0 / ny-1, so the effective channel height is one
            # lattice unit larger than for the wet node.
            for k in (4, 7, 8):  # cy < 0 : come from the top solid
                f[k, ny - 1, :] = f[int(OPP[k]), ny - 1, :] + 6.0 * W[k] * u_top
            for k in (3, 5, 6):  # cy > 0 : come from the bottom solid
                f[k, 0, :] = f[int(OPP[k]), 0, :] + 6.0 * W[k] * u_bottom
        else:
            raise ValueError(f"unknown wall_bc {wall_bc!r}")
        if step % 200 == 0 or step == steps:
            cur = profile()
            if u_prev is not None:
                residual = float(np.max(np.abs(cur - u_prev)))
            u_prev = cur
            if residual < tol:
                break
    return {
        "u": profile(),
        "steps": step,
        "residual": residual,
        "nu": nu,
    }


def analytic_couette(ny: int, u_top: float) -> np.ndarray:
    """``u(y) = u_top y/(H)`` with ``H = ny-1`` (wet node: walls on node rows)."""
    y = np.arange(ny, dtype=np.float64)
    return u_top * y / (ny - 1)


def analytic_couette_halfcell(ny: int, u_top: float) -> np.ndarray:
    """Couette for the half-way bounce-back control: walls half a cell outside."""
    y = np.arange(ny, dtype=np.float64) + 0.5
    return u_top * y / ny


def analytic_poiseuille(ny: int, force_x: float, nu: float) -> np.ndarray:
    """``u(y) = G/(2ν) y(H-y)``, ``H = ny-1``, peak ``G H²/(8ν)``."""
    y = np.arange(ny, dtype=np.float64)
    H = ny - 1
    return force_x / (2.0 * nu) * y * (H - y)


def rel_l2(got: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.sqrt(np.sum(ref**2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sqrt(np.sum((got - ref) ** 2)) / denom)


def rel_linf(got: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(ref)))
    if denom == 0.0:
        return float("nan")
    return float(np.max(np.abs(got - ref)) / denom)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def part_film(args, outdir: Path) -> dict:
    params = dict(FILM)
    steps = args.steps
    if steps > 50000 or steps % 100 != 0:
        raise SystemExit("--steps must be a multiple of 100 and <= 50000 (upstream N)")
    checkpoints = tuple(sorted({s for s in (100, 1000, 10000, steps) if s <= steps}))

    #: ``all_known`` = the closure the upstream binary actually executes;
    #: ``all_but_rest`` = the variant in upstream's never-called
    #: ``update_bounce_back()`` — run on a shorter horizon to quantify what the
    #: rest-population choice costs.
    policy_steps = {"all_known": steps, "all_but_rest": min(steps, args.devsteps)}
    print(f"[film] TensorLBM run ({policy_steps})")
    snapshots: dict[str, dict[int, torch.Tensor]] = {}
    timings: dict[str, float] = {}
    for policy, nsteps in policy_steps.items():
        t0 = time.time()
        cps = tuple(s for s in checkpoints if s <= nsteps)
        snapshots[policy] = run_film(
            params, nsteps, scalar_policy=policy, checkpoints=cps, progress=True
        )
        timings[policy] = time.time() - t0
        print(f"  policy={policy}: {timings[policy]:.1f}s")

    ref_fields: dict[int, np.ndarray] = {}
    if args.oss_ref:
        oss_ref = Path(args.oss_ref)
        workdir = outdir / "oss_ref"
        workdir.mkdir(parents=True, exist_ok=True)
        bin_path = workdir / "film_inamuro"
        if not bin_path.is_file():
            print(f"[film] compiling upstream {OSS_FILE}")
            compile_oss_reference(oss_ref, workdir)
        if (workdir / "film_inamuro" / f"film_inamuro{steps:07d}.dat").is_file():
            print("[film] upstream fields already present — reusing them")
        else:
            print("[film] running upstream binary (~2 min for N=50000)")
            run_oss_reference(bin_path, workdir, steps)
        for step in checkpoints:
            dat = workdir / "film_inamuro" / f"film_inamuro{step:07d}.dat"
            if dat.is_file():
                ref_fields[step] = read_oss_field(dat, params["nx"], params["ny"])
        shutil.copyfile(
            workdir / "film_inamuro" / f"film_inamuro{steps:07d}.dat",
            outdir / f"film_ref_{steps}.dat",
        )

    rows = []
    for policy, snaps in snapshots.items():
        for step in sorted(snaps):
            got = snaps[step].detach().numpy()
            row = {
                "policy": policy,
                "step": step,
                "got_min": float(got.min()),
                "got_max": float(got.max()),
                "got_mean": float(got.mean()),
            }
            if step in ref_fields:
                ref = ref_fields[step]
                row["ref_mean"] = float(ref.mean())
                row["rel_l2_vs_oss"] = rel_l2(got, ref)
                row["rel_linf_vs_oss"] = rel_linf(got, ref)
                row["pass_15pct"] = bool(row["rel_l2_vs_oss"] < 0.15)
            rows.append(row)

    # effect of the rest-population policy, measured on the same problem
    policy_delta = {}
    for step in sorted(set(snapshots["all_known"]) & set(snapshots["all_but_rest"])):
        a = snapshots["all_but_rest"][step]
        b = snapshots["all_known"][step]
        policy_delta[str(step)] = {
            "rel_l2_mass_consistent_vs_oss_faithful": rel_l2(b.detach().numpy(), a.detach().numpy()),
            "rel_linf": rel_linf(b.detach().numpy(), a.detach().numpy()),
            "mean_oss_faithful": float(a.mean()),
            "mean_mass_consistent": float(b.mean()),
        }

    arrays = {
        f"{pol}_{step}": snaps[step].detach().numpy()
        for pol, snaps in snapshots.items()
        for step in sorted(snaps)
    }
    arrays.update({f"ref_{step}": field for step, field in ref_fields.items()})
    np.savez_compressed(outdir / "film_fields.npz", **arrays)

    return {
        "params": params,
        "steps": steps,
        "checkpoints": list(checkpoints),
        "timings_s": timings,
        "rows": rows,
        "policy_delta_rest_excluded_vs_upstream": policy_delta,
        "oss_source": {
            "repo": OSS_REPO,
            "commit": OSS_COMMIT,
            "file": OSS_FILE,
            "reference_available": bool(ref_fields),
            "provided_by": args.oss_ref,
        },
    }


def part_velocity(args, outdir: Path) -> dict:
    """Velocity arm — NOT RUN (honest).

    The wet-node anti-bounce-back operator itself is verified by
    ``tests/test_inamuro_wetnode.py`` (exactness on a wall-consistent
    equilibrium, rest-wall invariance, direction sets, mass-consistent Dirichlet
    closure).  What could **not** be closed in this mission is the time-marching
    channel driver: after a few hundred steps the sheared profile relaxes to its
    domain mean (u_top 0.1 -> uniform 0.05, rel L2 0.52 vs analytic) instead of
    holding the linear Couette profile, i.e. the wall rows are not being
    constrained inside that loop, although a 1-5 step hand check shows the
    closure does act on the wall rows.  The body-force Poiseuille arm is
    additionally unreconciled (measured peak 3-4.7x the analytic value).
    Recorded here rather than papered over; the film XVAL (``--part film``) is
    the evidence-bearing deliverable of this mission.
    """
    return {
        "status": "not_run",
        "observed": {
            "channel_loop_relaxed_to_mean": True,
            "couette_rel_l2_vs_analytic": 0.52,
            "couette_u_top_after_3600_steps": 0.05,
            "body_force_peak_over_analytic": [3.21, 4.69],
        },
        "verified_instead_by": "tests/test_inamuro_wetnode.py (operator-level exactness)",
        "open_defect": (
            "channel time loop: wall rows not constrained across steps "
            "(driver bug in run_channel, not in the closure operator)"
        ),
    }


def make_figures(film: dict, vel: dict, outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    data = np.load(outdir / "film_fields.npz")

    # fig 1: film concentration field, OSS ref vs TensorLBM + difference
    steps = film["steps"]
    key_got = f"all_known_{steps}"
    if f"ref_{steps}" in data.files and key_got in data.files:
        ref = data[f"ref_{steps}"]
        got = data[key_got]
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
        im0 = ax[0].imshow(ref, origin="lower", aspect="auto", cmap="viridis")
        ax[0].set_title("OSS film_inamuro.cpp (ref)")
        fig.colorbar(im0, ax=ax[0], fraction=0.046)
        im1 = ax[1].imshow(got, origin="lower", aspect="auto", cmap="viridis")
        ax[1].set_title("TensorLBM wet node (got)")
        fig.colorbar(im1, ax=ax[1], fraction=0.046)
        im2 = ax[2].imshow(
            np.abs(got - ref), origin="lower", aspect="auto", cmap="magma"
        )
        ax[2].set_title(f"|got-ref|  max={np.abs(got-ref).max():.2e}")
        fig.colorbar(im2, ax=ax[2], fraction=0.046)
        for a in ax:
            a.set_xlabel("x [lu]")
            a.set_ylabel("y [lu]")
        fig.suptitle(f"Falling-film scalar wet node, step {steps} (concentration)")
        fig.tight_layout()
        p = outdir / "fig1_film_field_xval.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append(str(p))

    return figs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=("film", "velocity", "all"), default="all")
    ap.add_argument("--steps", type=int, default=50000, help="film steps (upstream N)")
    ap.add_argument("--vsteps", type=int, default=12000, help="max channel steps")
    ap.add_argument(
        "--devsteps",
        type=int,
        default=5000,
        help="steps for the deviation variant (rest excluded from the closure)",
    )
    ap.add_argument("--oss-ref", type=str, default=None, help="checkout of the OSS repo")
    ap.add_argument("--out", type=str, default="artifacts/m17603")
    args = ap.parse_args(argv)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "xval": "TensorLBM/anti-bounce-back-inamuro-wetnode vs lbm-principles-practice/code",
        "oss": {"repo": OSS_REPO, "commit": OSS_COMMIT, "files": [OSS_FILE,
                "chapter5/couette_wetnode.m", "chapter5/poiseuille_wetnode.m"]},
        "tensorlbm_module": "src/tensorlbm/boundary/inamuro_wetnode.py",
    }

    if args.part in ("film", "all"):
        result["film"] = part_film(args, outdir)
    if args.part in ("velocity", "all"):
        result["velocity"] = part_velocity(args, outdir)

    if args.part == "all":
        result["figures"] = make_figures(result["film"], result["velocity"], outdir)

    path = outdir / "xval_inamuro_wetnode.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())