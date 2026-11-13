"""XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S -- cross-validation harness.

Anchor
------
Cross-validate TensorLBM's *force-corrected NEBB + body-force driver* against the
mature open-source implementation **haimingz/vivsim**, and publish the ref/got
table that the campaign gate re-checks.

Reference (OSS) implementation
------------------------------
``haimingz/vivsim`` (JAX, Apache-2.0), upstream tree used for the transcription:
``/tmp/vivsim_src`` (recorded with ``sha256`` in ``REPORT_XVAL_NEBB_BODYFORCE.md``
and in the ``oss_provenance`` block of ``results.json``).  Kernels used:

* ``vivsim/lbm/basic.py`` -- ``get_macroscopic`` / ``get_equilibrium`` /
  ``get_velocity_correction`` (shifted populations ``f_i = f_i^full - w_i``);
* ``vivsim/lbm/forcing/edm.py`` + ``guo.py`` -- ``forcing_edm`` /
  ``forcing_guo_bgk`` / ``get_guo_forcing_term``;
* ``vivsim/lbm/boundary/nebb.py`` + ``_helpers.py`` -- shear-reconstruction
  NEBB and the half-force wall-velocity correction;
* ``examples/2d/poiseuille_channel.py`` -- step order, ``rho_wall = 1`` default,
  post-processing correction.

**Limitation (disclosed, not papered over):** the upstream package needs JAX,
which is not installable in this environment (``pip install jax`` timed out;
``friction`` entry in ``results.json``).  Arm A is therefore a *line-by-line
NumPy transcription* of the upstream kernels in the upstream's own shifted
convention, not the upstream binary.  What that buys and what it does not:

* it does cross-check TensorLBM's implementation (arm B, full-population
  convention, :mod:`tensorlbm.nebb_bodyforce`) against an independent second
  implementation of the same published formulation, including the
  population-shift convention, the wall-density policy and the vectorisation;
* it does **not** prove that the JAX code path of vivsim behaves identically
  (JIT/dtype differences are outside this harness).  Flagged as
  ``oss_binary_executed: false``.

Decision, alternatives, pre-declared thresholds (experimental-design skill)
--------------------------------------------------------------------------
* Decision: is TensorLBM's force-corrected NEBB + body-force driver a faithful
  port of the OSS formulation, and does it reproduce the closed-form Poiseuille
  solution within the owner accuracy standard?
* Alternatives being compared: (A) OSS-transcribed arm; (B) TensorLBM module;
  (C) wall model variants inside one arm (shear-reconstruction NEBB vs classic
  Zou/He NEBB vs halfway bounce-back) -- these distinguish "the formulation is
  right" from "this particular wall kernel is right".
* Metrics: ``max|x_armB - x_armA| / max|x_armA|`` (cross-arm), and
  ``max|x - x_analytic| / max|x_analytic|`` (accuracy).
* Pre-declared thresholds (fixed before the runs, owner standard):
  ``PASS_OWNER = 0.15``; ``PASS_STRETCH = 0.03`` (ecosystem stretch);
  ``PASS_CROSS_ARM = 1e-9`` (formulation-level port: anything above this means
  a real implementation difference, not round-off).
* Sensitivity (more than the two required): forcing scheme EDM vs Guo-BGK;
  wall model; grid ``ny = 10, 20, 40``; body force ``gx = 1e-3`` vs ``1e-2``
  (Mach-number sensitivity); viscosity ``nu = 0.2`` vs ``0.05`` (tau).
* Environment: CPU, NumPy float64, ``nx = 8`` (x is periodic and the solution is
  x-independent, so ``nx`` is a pure cost knob), single process, no GPU.

Outputs (``--out``, default ``runs/XVAL-NEBB-BODYFORCE``): ``results.json``,
``run.log``, and the figures in ``FIGURES``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:  # allow `python benchmarks/xval/...`
    sys.path.insert(0, str(REPO / "src"))

from tensorlbm import nebb_bodyforce as nb  # noqa: E402
from tensorlbm.nebb_bodyforce import (  # noqa: E402
    C,
    OPPOSITE,
    W,
    BodyForceChannel,
    ChannelConfig,
    analytic_poiseuille,
    effective_wall_offset,
    profile_errors,
    wall_spec,
)

ANCHOR = "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S"
OSS_REPO = "haimingz/vivsim"
OSS_TREE = "/tmp/vivsim_src"

PASS_OWNER = 0.15
PASS_STRETCH = 0.03
PASS_CROSS_ARM = 1e-9

FIGURES = {
    "fig_profile.png": "steady u_x(y) for both arms and the closed-form parabola",
    "fig_cross_arm.png": "arm-A vs arm-B difference and wall-model comparison",
    "fig_convergence.png": "grid-refinement error of both arms (log-log)",
    "fig_field.png": "u_x(x, y) field of the TensorLBM arm",
}


# --------------------------------------------------------------------------- #
# Arm A: line-by-line NumPy transcription of the OSS kernels
# --------------------------------------------------------------------------- #
class OssPort:
    """vivsim's Poiseuille recipe in NumPy, in the upstream shifted convention.

    The transcription keeps the upstream signatures and the upstream
    ``f_i = f_i^full - w_i`` storage so that the comparison with arm B also
    exercises the population-shift convention (module docstring of
    :mod:`tensorlbm.nebb_bodyforce` claims the NEBB algebra is invariant).
    """

    def __init__(self, nx: int, ny: int, nu: float, gx: float, scheme: str = "edm"):
        if scheme not in ("edm", "guo"):
            raise ValueError(scheme)
        self.nx, self.ny, self.nu, self.gx, self.scheme = nx, ny, nu, gx, scheme
        self.omega = 1.0 / (3.0 * nu + 0.5)
        self.rho = np.ones((nx, ny))
        self.u = np.zeros((2, nx, ny))
        self.g = np.zeros((2, nx, ny))
        self.g[0] = gx
        self.f = self.get_equilibrium(self.rho, self.u - self.g / 2.0)  # upstream init

    # -- basic.py ------------------------------------------------------- #
    def get_equilibrium(self, rho, u):
        uc = u[0][None] * C[:, 0, None, None] + u[1][None] * C[:, 1, None, None]
        return W[:, None, None] * (
            rho[None] * (1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * (u * u).sum(axis=0)[None]) - 1.0
        )

    def get_macroscopic(self, f):
        rho = 1.0 + f.sum(axis=0)
        right, left = [1, 5, 8], [3, 7, 6]
        up, down = [2, 5, 6], [4, 7, 8]
        u = np.zeros((2, *rho.shape))
        u[0] = (f[right].sum(axis=0) - f[left].sum(axis=0)) / rho
        u[1] = (f[up].sum(axis=0) - f[down].sum(axis=0)) / rho
        return rho, u

    def get_velocity_correction(self, g, rho=1.0):
        return g * 0.5 / rho

    # -- forcing/guo.py, forcing/edm.py --------------------------------- #
    def get_guo_forcing_term(self, g, u):
        uc = u[0][None] * C[:, 0, None, None] + u[1][None] * C[:, 1, None, None]
        return W[:, None, None] * (
            g[0][None] * (3.0 * (C[:, 0, None, None] - u[0][None]) + 9.0 * uc * C[:, 0, None, None])
            + g[1][None]
            * (3.0 * (C[:, 1, None, None] - u[1][None]) + 9.0 * uc * C[:, 1, None, None])
        )

    def forcing_edm(self, f, g, u):
        return f + self.get_guo_forcing_term(g, u)

    def forcing_guo_bgk(self, f, g, u, omega):
        return f + self.get_guo_forcing_term(g, u) * (1.0 - 0.5 * omega)

    # -- boundary/nebb.py + _helpers.py --------------------------------- #
    def boundary_nebb(self, f, loc, rho_wall=1, ux_wall=0, uy_wall=0):
        spec = wall_spec(loc)
        idx = 0 if loc in ("bottom", "left") else -1
        wall = f[..., idx] if spec.normal_axis == 1 else f[:, idx, :]

        s = spec.normal_sign
        un = s * (ux_wall if spec.normal_axis == 0 else uy_wall)
        ut = s * (uy_wall if spec.normal_axis == 0 else ux_wall)
        i0, i1, i2 = spec.in_dirs
        o0, o1, o2 = spec.out_dirs
        t0, t1 = spec.tan_dirs

        shear = 0.5 * (wall[t0] - wall[t1]) * s
        normal_term = un * rho_wall / 6.0
        tangential_term = ut * rho_wall / 2.0
        new = np.stack(
            (
                wall[o0] + 2.0 / 3.0 * un * rho_wall,
                wall[o1] - shear + normal_term + tangential_term,
                wall[o2] + shear + normal_term - tangential_term,
            )
        )
        out = f.copy()
        if spec.normal_axis == 1:
            out[:, :, idx][list(spec.in_dirs)] = new
        else:
            out[:, idx, :][list(spec.in_dirs)] = new
        # the two lines above index a view; write through explicitly
        if spec.normal_axis == 1:
            out[list(spec.in_dirs), :, idx] = new
        else:
            out[list(spec.in_dirs), idx, :] = new
        return out

    def boundary_force_corrected_nebb(self, f, loc, gx_wall=0.0, gy_wall=0.0, rho_wall=1):
        """``wrap_force_corrected``: corrected wall velocity, ``rho_wall=1`` default."""
        ux = 0.0 - self.get_velocity_correction(gx_wall, rho_wall)
        uy = 0.0 - self.get_velocity_correction(gy_wall, rho_wall)
        return self.boundary_nebb(f, loc, rho_wall=rho_wall, ux_wall=ux, uy_wall=uy)

    def streaming(self, f):
        out = f.copy()
        for i in range(1, 9):
            out[i] = np.roll(f[i], (int(C[i, 0]), int(C[i, 1])), axis=(0, 1))
        return out

    # -- examples/2d/poiseuille_channel.py ------------------------------ #
    def step(self):
        rho, u = self.get_macroscopic(self.f)
        if self.scheme == "guo":
            u = u + self.get_velocity_correction(self.g, rho)
        feq = self.get_equilibrium(rho, u)
        f = (1.0 - self.omega) * self.f + self.omega * feq
        if self.scheme == "edm":
            f = self.forcing_edm(f, self.g, u)
        else:
            f = self.forcing_guo_bgk(f, self.g, u, self.omega)
        f = self.streaming(f)
        f = self.boundary_force_corrected_nebb(f, loc="top", gx_wall=self.gx)
        f = self.boundary_force_corrected_nebb(f, loc="bottom", gx_wall=self.gx)
        self.f = f

    def run(self, steps: int, steady_tol: float = 1e-13, check_every: int = 200):
        t0 = time.perf_counter()
        history = {"step": [], "residual": [], "u_mid": []}
        prev = None
        steps_used = steps
        residual = float("nan")
        for k in range(steps):
            self.step()
            if k % check_every == 0 or k == steps - 1:
                rho, u = self.get_macroscopic(self.f)
                ux = u[0] + self.get_velocity_correction(self.g, rho)[0]
                prof = ux.mean(axis=0)
                history["step"].append(k + 1)
                history["u_mid"].append(float(prof[self.ny // 2]))
                if prev is not None:
                    res = float(np.abs(prof - prev).max())
                    history["residual"].append(res)
                    residual = res
                    if res < steady_tol:
                        steps_used = k + 1
                        break
                prev = prof
        rho, u = self.get_macroscopic(self.f)
        ux = u[0] + self.get_velocity_correction(self.g, rho)[0]
        uy = u[1] + self.get_velocity_correction(self.g, rho)[1]
        return {
            "profile": ux.mean(axis=0),
            "ux": ux,
            "uy": uy,
            "rho": rho,
            "steps_used": steps_used,
            "steady_residual": residual,
            "history": history,
            "wall_velocity_corrected": float(ux[:, 0].mean()),
            "runtime_s": time.perf_counter() - t0,
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def oss_provenance() -> dict:
    """sha256 of the upstream files the transcription was read from."""
    tree = Path(OSS_TREE)
    files = [
        "vivsim/lbm/basic.py",
        "vivsim/lbm/lattice.py",
        "vivsim/lbm/forcing/edm.py",
        "vivsim/lbm/forcing/guo.py",
        "vivsim/lbm/boundary/nebb.py",
        "vivsim/lbm/boundary/_helpers.py",
        "examples/2d/poiseuille_channel.py",
    ]
    out = {"tree": OSS_TREE, "oss_binary_executed": False, "files": {}}
    for rel in files:
        p = tree / rel
        out["files"][rel] = sha256(p) if p.is_file() else None
    return out


def row(label: str, got: np.ndarray, ref: np.ndarray, cfg: dict, extra: dict | None = None) -> dict:
    err = profile_errors(got, ref)
    r = {
        "label": label,
        "params": cfg,
        "ref_kind": "analytic_poiseuille",
        "got": [float(v) for v in got],
        "ref": [float(v) for v in ref],
        "max_abs": err["max_abs"],
        "rel_err": err["max_rel_global"],
        "umax_got": err["umax_got"],
        "umax_ref": err["umax_ref"],
        "tolerance": PASS_OWNER,
        "pass_owner_15pct": bool(err["max_rel_global"] < PASS_OWNER),
        "pass_stretch_3pct": bool(err["max_rel_global"] < PASS_STRETCH),
        "wall_offset_fitted": effective_wall_offset(got, cfg["gx"], cfg["nu"]),
    }
    # second reference: the same parabola shifted by the wall offset the profile
    # actually has.  It separates "wrong wall position" (lattice convention) from
    # "wrong solution" -- halfway bounce-back, for instance, has offset 0.5.
    ref_fit = analytic_poiseuille(cfg["ny"], cfg["gx"], cfg["nu"], r["wall_offset_fitted"])
    r["rel_err_offset_fit"] = profile_errors(got, ref_fit)["max_rel_global"]
    r["ref_offset_fit"] = [float(v) for v in ref_fit]
    if extra:
        r.update(extra)
    return r


def cross_arm(label: str, got: np.ndarray, ref: np.ndarray, params: dict) -> dict:
    d = float(np.abs(got - ref).max())
    scale = max(float(np.abs(ref).max()), 1e-300)
    return {
        "label": label,
        "params": params,
        "max_abs_diff": d,
        "rel_diff": d / scale,
        "tolerance": PASS_CROSS_ARM,
        "pass": bool(d / scale < PASS_CROSS_ARM),
        "got": [float(v) for v in got],
        "ref": [float(v) for v in ref],
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(res: dict, out_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made: dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, style in (("armA_edm", "o-"), ("armB_guo", "s--"), ("armB_edm", "^:")):
        if key in res["runs"]:
            ax.plot(res["runs"][key]["got"], label=key, **{"marker": style[0], "ls": style[1:]})
    ax.plot(res["analytic"], "k-", lw=1, label="analytic (walls on nodes)")
    ax.set_xlabel("y (node index)")
    ax.set_ylabel("u_x")
    ax.set_title("Poiseuille channel, ny=10: OSS-transcribed vs TensorLBM")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_profile.png", dpi=130)
    plt.close(fig)
    made["fig_profile.png"] = FIGURES["fig_profile.png"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("cross_arm_edm", "cross_arm_guo"):
        if key in res["cross_arm"]:
            ax.semilogy(
                np.abs(np.array(res["cross_arm"][key]["got"]) - np.array(res["cross_arm"][key]["ref"]))
                + 1e-18,
                "o-",
                label=key,
            )
    for key in ("armB_wall_nonequilibrium_nebb", "armB_wall_halfway_bounce_back"):
        if key in res["runs"]:
            ax.semilogy(
                np.abs(np.array(res["runs"][key]["got"]) - np.array(res["runs"][key]["ref"])) + 1e-18,
                "s--",
                label=key,
            )
    ax.set_xlabel("y (node index)")
    ax.set_ylabel("|u_x - reference|")
    ax.set_title("implementation / wall-model differences (log scale)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cross_arm.png", dpi=130)
    plt.close(fig)
    made["fig_cross_arm.png"] = FIGURES["fig_cross_arm.png"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for arm, style in (
        ("armA_guo", "k-"),
        ("armB_guo", "o--"),
        ("armB_guo_recovered", "^--"),
        ("armB_guo_zouhe", "s-"),
        ("armB_guo_halfway", "d-"),
    ):
        rows = [r for r in res["convergence"] if r["arm"] == arm]
        if not rows:
            continue
        xs = [r["params"]["ny"] for r in rows]
        ax.loglog(xs, [max(r["rel_err"], 1e-16) for r in rows], style, label=f"{arm} (walls on nodes)")
    half = [r for r in res["convergence"] if r["arm"] == "armB_guo_halfway"]
    if half:
        ax.loglog([r["params"]["ny"] for r in half],
                  [max(r["rel_err_offset_fit"], 1e-16) for r in half],
                  "d:", label="armB_guo_halfway (ref shifted by 0.5 cell)")
    ny = np.array(sorted({r["params"]["ny"] for r in res["convergence"]}), dtype=float)
    ax.loglog(ny, 2.0 / ny**2, "k:", lw=0.8, label="1/ny^2 guide")
    ax.set_xlabel("ny")
    ax.set_ylabel("max relative error vs closed-form Poiseuille")
    ax.set_title("grid refinement: forcing/wall-model combinations")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.3, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_convergence.png", dpi=130)
    plt.close(fig)
    made["fig_convergence.png"] = FIGURES["fig_convergence.png"]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    field = np.array(res["fields"]["armB_guo_ux"])
    im = ax.imshow(field.T, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("x (periodic)")
    ax.set_ylabel("y")
    ax.set_title("TensorLBM arm: u_x(x, y)")
    fig.colorbar(im, ax=ax, label="u_x")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_field.png", dpi=130)
    plt.close(fig)
    made["fig_field.png"] = FIGURES["fig_field.png"]
    return made


# --------------------------------------------------------------------------- #
# main study
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "runs" / "XVAL-NEBB-BODYFORCE"))
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--nx", type=int, default=8)
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_fh = log_path.open("w")

    def log(msg: str) -> None:
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    t_start = time.perf_counter()
    log(f"# {ANCHOR}")
    log(f"OSS reference: {OSS_REPO} (transcribed tree {OSS_TREE}; binary not executed)")

    res: dict = {
        "anchor": ANCHOR,
        "oss_repo": OSS_REPO,
        "oss_provenance": oss_provenance(),
        "problem": "2-D D2Q9 plane channel, periodic x, no-slip top/bottom rows, "
                   "constant body force gx in x; steady solution u_x(y) = gx/(2 nu) y (H-y)",
        "thresholds": {
            "owner": PASS_OWNER,
            "stretch": PASS_STRETCH,
            "cross_arm": PASS_CROSS_ARM,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "dtype": "float64",
            "device": "cpu",
        },
        "runs": {},
        "cross_arm": {},
        "sensitivity": {},
        "convergence": [],
        "fields": {},
        "friction": [
            {
                "issue": "upstream vivsim requires JAX; `pip install jax` timed out in this "
                         "environment (no usable wheel/network) so the OSS arm is a "
                         "source-transcribed NumPy port, not the upstream binary",
                "impact": "cross-check validates the formulation + our implementation, not "
                          "the JAX code path of the upstream package",
                "mitigation": "line-by-line mapping table in src/tensorlbm/nebb_bodyforce.py, "
                              "sha256 of every transcribed upstream file recorded here, "
                              "shift-convention invariance tested in "
                              "tests/test_nebb_bodyforce.py",
            }
        ],
    }

    NX = args.nx
    PRIMARY = dict(nx=NX, ny=10, nu=0.2, gx=1e-3, steps=args.steps)

    # ---- primary runs (OSS example parameters) ------------------------- #
    analytic10 = analytic_poiseuille(10, PRIMARY["gx"], PRIMARY["nu"], 0.0)
    res["analytic"] = analytic10.tolist()

    arm_results: dict[str, np.ndarray] = {}
    for scheme in ("edm", "guo"):
        t0 = time.perf_counter()
        oss = OssPort(NX, 10, PRIMARY["nu"], PRIMARY["gx"], scheme=scheme)
        o = oss.run(PRIMARY["steps"])
        res["runs"][f"armA_{scheme}"] = row(
            f"armA_{scheme}_vivsim_transcription",
            o["profile"],
            analytic10,
            {"ny": 10, "nu": PRIMARY["nu"], "gx": PRIMARY["gx"], "scheme": scheme},
            extra={
                "arm": "A",
                "implementation": "numpy transcription of haimingz/vivsim (shifted populations, rho_wall=1)",
                "steps_used": o["steps_used"],
                "steady_residual": o["steady_residual"],
                "u_wall_corrected": o["wall_velocity_corrected"],
                "runtime_s": o["runtime_s"],
                "history": o["history"],
            },
        )
        arm_results[f"A_{scheme}"] = o["profile"]
        log(
            f"armA/{scheme}: rel_err={res['runs'][f'armA_{scheme}']['rel_err']:.3e} "
            f"steps={o['steps_used']} u_wall={o['wall_velocity_corrected']:.2e} "
            f"({time.perf_counter()-t0:.1f}s)"
        )

        cfg = ChannelConfig(nx=NX, ny=10, nu=PRIMARY["nu"], gx=PRIMARY["gx"],
                            scheme=scheme, steps=PRIMARY["steps"])
        b = BodyForceChannel(cfg).run()
        res["runs"][f"armB_{scheme}"] = row(
            f"armB_{scheme}_tensorlbm_module",
            b.profile,
            analytic10,
            {"ny": 10, "nu": PRIMARY["nu"], "gx": PRIMARY["gx"], "scheme": scheme},
            extra={
                "arm": "B",
                "implementation": "tensorlbm.nebb_bodyforce.BodyForceChannel (full populations, rho_wall from field)",
                "steps_used": b.steps_used,
                "steady_residual": b.steady_residual,
                "mass_drift": b.mass_drift,
                "u_wall_raw": b.wall_velocity_raw,
                "u_wall_corrected": b.wall_velocity_corrected,
                "history": b.history,
            },
        )
        arm_results[f"B_{scheme}"] = b.profile
        log(
            f"armB/{scheme}: rel_err={res['runs'][f'armB_{scheme}']['rel_err']:.3e} "
            f"steps={b.steps_used} u_wall={b.wall_velocity_corrected:.2e} "
            f"mass_drift={b.mass_drift:.1e}"
        )
        if scheme == "guo":
            res["fields"]["armB_guo_ux"] = b.ux_corrected.tolist()
            res["density_range_armB"] = [float(b.rho.min()), float(b.rho.max())]
            res["fields"]["armA_guo_ux"] = o["ux"].tolist()

    for scheme in ("edm", "guo"):
        res["cross_arm"][f"cross_arm_{scheme}"] = cross_arm(
            f"armB - armA ({scheme})",
            arm_results[f"B_{scheme}"],
            arm_results[f"A_{scheme}"],
            {"ny": 10, "nu": PRIMARY["nu"], "gx": PRIMARY["gx"], "scheme": scheme},
        )
        log(
            f"cross-arm/{scheme}: rel_diff={res['cross_arm'][f'cross_arm_{scheme}']['rel_diff']:.3e} "
            f"(threshold {PASS_CROSS_ARM:.0e})"
        )

    # ---- sensitivity 1: wall model inside arm B ----------------------- #
    for wall in ("force_corrected_nebb_recovered", "nonequilibrium_nebb", "halfway_bounce_back"):
        cfg = ChannelConfig(nx=NX, ny=10, nu=PRIMARY["nu"], gx=PRIMARY["gx"],
                            scheme="guo", wall=wall, steps=PRIMARY["steps"])
        b = BodyForceChannel(cfg).run()
        res["runs"][f"armB_wall_{wall}"] = row(
            f"armB_guo_wall_{wall}",
            b.profile,
            analytic10,
            {"ny": 10, "nu": PRIMARY["nu"], "gx": PRIMARY["gx"], "scheme": "guo", "wall": wall},
            extra={"arm": "B(wall variant)", "steps_used": b.steps_used,
                   "u_wall_corrected": b.wall_velocity_corrected},
        )
        log(f"wall={wall}: rel_err={res['runs'][f'armB_wall_{wall}']['rel_err']:.3e} "
            f"offset={res['runs'][f'armB_wall_{wall}']['wall_offset_fitted']:.4f}")

    # ---- sensitivity 2: body-force magnitude (Mach number) ------------ #
    for gx in (1e-2, 1e-1):
        ana = analytic_poiseuille(10, gx, PRIMARY["nu"], 0.0)
        cfg = ChannelConfig(nx=NX, ny=10, nu=PRIMARY["nu"], gx=gx, scheme="guo", steps=PRIMARY["steps"])
        b = BodyForceChannel(cfg).run()
        res["sensitivity"][f"gx_{gx:g}"] = row(
            f"armB_guo_gx{gx:g}", b.profile, ana,
            {"ny": 10, "nu": PRIMARY["nu"], "gx": gx},
            extra={"umax_over_cs": float(ana.max() / (1.0 / np.sqrt(3.0)))},
        )
        log(f"gx={gx:g}: rel_err={res['sensitivity'][f'gx_{gx:g}']['rel_err']:.3e} "
            f"Ma~{res['sensitivity'][f'gx_{gx:g}']['umax_over_cs']:.4f}")

    # ---- sensitivity 3: viscosity (tau) ------------------------------- #
    for nu in (0.05, 0.5):
        ana = analytic_poiseuille(10, PRIMARY["gx"], nu, 0.0)
        cfg = ChannelConfig(nx=NX, ny=10, nu=nu, gx=PRIMARY["gx"], scheme="guo", steps=args.steps)
        b = BodyForceChannel(cfg).run()
        res["sensitivity"][f"nu_{nu:g}"] = row(
            f"armB_guo_nu{nu:g}", b.profile, ana, {"ny": 10, "nu": nu, "gx": PRIMARY["gx"]}
        )
        log(f"nu={nu:g}: rel_err={res['sensitivity'][f'nu_{nu:g}']['rel_err']:.3e} "
            f"steps={b.steps_used}")

    # ---- convergence: grid refinement, arms + wall models -------------- #
    for ny in (10, 20, 40):
        ana = analytic_poiseuille(ny, PRIMARY["gx"], PRIMARY["nu"], 0.0)
        for arm, wall in (
            ("armB_guo", "force_corrected_nebb"),
            ("armB_guo_recovered", "force_corrected_nebb_recovered"),
            ("armB_guo_zouhe", "nonequilibrium_nebb"),
            ("armB_guo_halfway", "halfway_bounce_back"),
        ):
            cfg = ChannelConfig(nx=NX, ny=ny, nu=PRIMARY["nu"], gx=PRIMARY["gx"],
                                scheme="guo", wall=wall, steps=args.steps)
            b = BodyForceChannel(cfg).run()
            off = effective_wall_offset(b.profile, PRIMARY["gx"], PRIMARY["nu"])
            ref_fit = analytic_poiseuille(ny, PRIMARY["gx"], PRIMARY["nu"], off)
            res["convergence"].append(
                {
                    "arm": arm,
                    "params": {"ny": ny, "wall": wall},
                    "rel_err": profile_errors(b.profile, ana)["max_rel_global"],
                    "rel_err_offset_fit": profile_errors(b.profile, ref_fit)["max_rel_global"],
                    "wall_offset_fitted": off,
                }
            )
            log(f"ny={ny} {arm}: rel_err={res['convergence'][-1]['rel_err']:.3e} "
                f"offset={off:.4f}")
        oss = OssPort(NX, ny, PRIMARY["nu"], PRIMARY["gx"], scheme="guo")
        o = oss.run(args.steps)
        res["convergence"].append({"arm": "armA_guo", "params": {"ny": ny},
                                   "rel_err": profile_errors(o["profile"], ana)["max_rel_global"],
                                   "rel_err_offset_fit": profile_errors(o["profile"], ana)["max_rel_global"],
                                   "wall_offset_fitted": 0.0})
        log(f"ny={ny} armA_guo: rel_err={res['convergence'][-1]['rel_err']:.3e}")

    # ---- verdict ------------------------------------------------------ #
    rows = [r for r in res["runs"].values() if r["label"].startswith(("armA_", "armB_"))]
    primary_rows = [res["runs"][f"armA_{s}"] for s in ("edm", "guo")] + [
        res["runs"][f"armB_{s}"] for s in ("edm", "guo")
    ]
    res["verdict"] = {
        "primary_max_rel_err_armB": max(
            res["runs"]["armB_edm"]["rel_err"], res["runs"]["armB_guo"]["rel_err"]
        ),
        "primary_max_rel_err_armA": max(
            res["runs"]["armA_edm"]["rel_err"], res["runs"]["armA_guo"]["rel_err"]
        ),
        "max_cross_arm_rel_diff": max(v["rel_diff"] for v in res["cross_arm"].values()),
        "pass_owner_15pct": all(r["pass_owner_15pct"] for r in primary_rows),
        "pass_stretch_3pct": all(r["pass_stretch_3pct"] for r in primary_rows),
        "pass_cross_arm_1e-9": all(v["pass"] for v in res["cross_arm"].values()),
        "wall_model_variants": {
            k: res["runs"][k]["rel_err"] for k in res["runs"] if k.startswith("armB_wall_")
        },
        "n_rows": len(rows),
    }
    res["runtime_s"] = time.perf_counter() - t_start
    res["figures"] = make_figures(res, out_dir)
    res["repro_cmd"] = (
        f"python benchmarks/xval/xval_vivsim_nebb_bodyforce.py --out {out_dir.relative_to(REPO)}"
    )

    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    log(json.dumps(res["verdict"], indent=2))
    log(f"runtime {res['runtime_s']:.1f}s -> {out_dir/'results.json'}")
    log_fh.close()
    return 0 if res["verdict"]["pass_owner_15pct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())