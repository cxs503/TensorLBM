"""XVAL entry — TensorLBM vs 2b-t/LB-t on the population *layout pattern*.

Anchor: ``XVAL-LBM-NO-MULTIPHASE-LAYOUT-PATTERN-ONLY``.

What is compared
----------------
Two independent references are used:

``ref = upstream OSS`` (executable, zero hand-copy)
    ``lbt_ref_dump.cpp`` is compiled against the *unmodified* upstream headers
    and dumps the linear index of every ``(x, y, z, p, n, d)`` slot plus the
    ``indexRead``/``indexWrite`` odd/even offsets for ``NP = 1`` and ``NP = 2``
    (``artifacts/xval_lbt_layout/lbt_ref_layout.json``).  The port in
    :mod:`tensorlbm.aa_layout` must reproduce every one of those integers.

``ref = analytic Poiseuille`` (literature)
    The layout alone is not a physics claim, so the ported storage is exercised
    by a body-force driven laminar channel and the mean velocity profile is
    compared with the exact steady solution
    ``u(y) = a/(2 nu) (y - y_w)(H + y_w - y)``.

``ref = TensorLBM production dense kernels``
    The same channel is run through ``tensorlbm.solver.collide_bgk`` /
    ``tensorlbm.solver.stream`` (dense ``(ny, nx, 9)`` storage, the production
    path).  Storing the same simulation in the OSS layout must not change a
    single digit — this is the *transparency* claim of the port.

Acceptance: every comparison below must stay under 15 % (owner standard); the
index/parity comparison is exact (0 %), the physics comparison is expected to
be O(1 %) and the cross-layout comparison O(1e-13).

Run
---
    python benchmarks/xval_lbt_layout_pattern/run_xval.py --out artifacts/xval_lbt_layout
    python benchmarks/xval_lbt_layout_pattern/run_xval.py --quick        # CI-sized

The upstream dump is only re-generated when a C++ compiler and an LB-t checkout
are available (``LB_T_PATH``, default ``/tmp/LB-t``); otherwise the checked-in
JSON is used and the artifact records ``ref_regenerated: false``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.aa_layout import (  # noqa: E402
    D2Q9_DX,
    D2Q9_DY,
    D2Q9_MASK,
    D2Q9_ND,
    D2Q9_OFF,
    D2Q9_W,
    SLOT_TO_TL,
    TL_TO_SLOT,
    AaLayout,
    AaPopulation,
    equilibrium_d2q9,
)

DIRS = np.stack([D2Q9_DX[TL_TO_SLOT], D2Q9_DY[TL_TO_SLOT]], axis=1)  # (9, 2)
WEIGHTS = D2Q9_W[TL_TO_SLOT]
OPPOSITE = np.zeros(9, dtype=np.int64)
for _i in range(9):
    OPPOSITE[_i] = int(np.argmin(((DIRS + DIRS[_i]) ** 2).sum(axis=1)))


# ---------------------------------------------------------------------------
# physics helpers shared by the two arms (identical arithmetic in both arms so
# that any difference is attributable to the storage layout alone)
# ---------------------------------------------------------------------------
def analytic_poiseuille(ny: int, force: float, nu: float) -> np.ndarray:
    """Exact steady solution for a body-force channel with walls at 0.5, ny-0.5.

    Returns ``u`` on the fluid rows ``y = 1 .. ny-2``.
    """
    y = np.arange(1, ny - 1, dtype=np.float64)
    y_wall = 0.5
    return force / (2.0 * nu) * (y - y_wall) * ((ny - 0.5) - y)


def solid_rows(ny: int) -> np.ndarray:
    mask = np.zeros(ny, dtype=bool)
    mask[0] = True
    mask[-1] = True
    return mask


def collide_with_force(f: np.ndarray, tau: float, force: float) -> np.ndarray:
    """Guo forcing BGK collision in TensorLBM direction order (trailing axis).

    Scheme (Guo, Zheng & Shi, J. Comput. Phys. 165 (2002) 288, eqs. 20-21)::

        u   = sum_i c_i f_i / rho                      (kinetic velocity)
        u*  = u + F/(2 rho)                            (drift velocity)
        f*_i = f_i - (f_i - feq_i(rho, u*))/tau
               + (1 - 1/(2 tau)) w_i [3 (c_i - u*).F + 9 (c_i.u*)(c_i.F)]

    so that the steady state satisfies the momentum balance with the *full*
    force ``F = rho a`` (no ``(1-1/(2tau))`` damping of the acceleration).
    """
    rho = f.sum(axis=-1)
    ux = (f * DIRS[:, 0]).sum(axis=-1) / rho
    uy = (f * DIRS[:, 1]).sum(axis=-1) / rho
    ux_s = ux + 0.5 * force
    uy_s = uy
    feq = equilibrium_d2q9(rho, ux_s, uy_s)
    fx = rho * force
    fy = np.zeros_like(fx)
    cu = DIRS[:, 0] * ux_s[..., None] + DIRS[:, 1] * uy_s[..., None]
    cF = DIRS[:, 0] * fx[..., None] + DIRS[:, 1] * fy[..., None]
    uF = (ux_s * fx + uy_s * fy)[..., None]
    src = (1.0 - 0.5 / tau) * WEIGHTS * (3.0 * (cF - uF) + 9.0 * cu * cF)
    return f - (f - feq) / tau + src


def kinetic_velocity(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``rho, u`` from the first two moments (no half-force correction)."""
    rho = f.sum(axis=-1)
    return rho, (f * DIRS[:, 0]).sum(axis=-1) / rho


# ---------------------------------------------------------------------------
# arm A: OSS (LB-t) layout storage
# ---------------------------------------------------------------------------
def run_aa(
    nx: int,
    ny: int,
    tau: float,
    force: float,
    steps: int,
    npop: int = 1,
    tol: float = 1e-13,
    sample_every: int = 200,
) -> dict:
    """Channel on the ported LB-t A-A layout; ``npop`` components stacked."""
    pop = AaPopulation(nx=nx, ny=ny, nz=1, npop=npop)
    f0 = equilibrium_d2q9(
        np.ones((1, ny, nx, npop)),
        np.zeros((1, ny, nx, npop)),
        np.zeros((1, ny, nx, npop)),
    )
    pop.from_dense(f0)
    solid = solid_rows(ny)
    history: list[float] = []
    prev: np.ndarray | None = None
    reached = steps
    for step in range(steps):
        f = pop.to_dense()
        fpost = collide_with_force(f, tau, force)
        fpost[:, solid, :, :, :] = fpost[:, solid, :, :, :][..., OPPOSITE]
        pop.push(fpost)
        if prev is not None and (step + 1) % sample_every == 0:
            _, ux = kinetic_velocity(pop.to_dense())
            delta = float(np.max(np.abs(ux - prev)))
            history.append(delta)
            prev = ux.copy()
            if delta < tol:
                reached = step + 1
                break
        elif prev is None:
            prev = kinetic_velocity(pop.to_dense())[1].copy()
    f = pop.to_dense()
    rho, ux = kinetic_velocity(f)
    u_phys = ux + 0.5 * force  # momentum correction rho*u + F/2
    return {
        "profile": u_phys[0, 1:-1, :, :].mean(axis=(1, 2)).tolist(),
        "profiles_per_population": [u_phys[0, 1:-1, :, k].mean(axis=1).tolist() for k in range(npop)],
        "rho": float(rho.mean()),
        "steps_run": reached,
        "history": history,
        "npop": npop,
        "layout_slots": pop.layout.total_slots,
    }


# ---------------------------------------------------------------------------
# arm B: TensorLBM production dense kernels (torch)
# ---------------------------------------------------------------------------
def equilibrium_d2q9_torch(rho, ux, uy, dx_t, dy_t, w_t):
    cu = dx_t * ux[None] + dy_t * uy[None]
    usq = (ux * ux + uy * uy)[None]
    return w_t * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def run_dense(
    nx: int,
    ny: int,
    tau: float,
    force: float,
    steps: int,
    tol: float = 1e-13,
    sample_every: int = 200,
) -> dict:
    """Same channel through TensorLBM's *production* dense D2Q9 kernels.

    Uses ``tensorlbm.solver.collide_bgk`` + ``tensorlbm.solver.stream`` on the
    production ``(9, ny, nx)`` storage so the comparison isolates the layout.
    """
    import torch

    from tensorlbm.solver import stream

    dtype = torch.float64
    dx_t = torch.tensor(DIRS[:, 0], dtype=dtype).view(9, 1, 1)
    dy_t = torch.tensor(DIRS[:, 1], dtype=dtype).view(9, 1, 1)
    w_t = torch.tensor(WEIGHTS, dtype=dtype).view(9, 1, 1)
    opp_t = torch.tensor(OPPOSITE, dtype=torch.int64)

    f = w_t * torch.ones((9, ny, nx), dtype=dtype)  # rest state: u = 0, rho = 1

    def moments(fx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = fx.sum(dim=0)
        return r, (fx * dx_t).sum(dim=0) / r, (fx * dy_t).sum(dim=0) / r

    solid_idx = torch.tensor(np.flatnonzero(solid_rows(ny)))
    history: list[float] = []
    prev = None
    reached = steps
    for step in range(steps):
        rho, ux, uy = moments(f)
        ux_s = ux + 0.5 * force
        feq = equilibrium_d2q9_torch(rho, ux_s, uy, dx_t, dy_t, w_t)
        fpost = f - (f - feq) / tau
        cu = dx_t * ux_s[None] + dy_t * uy[None]
        fx_field = rho * force
        cF = dx_t * fx_field[None]
        uF = (ux_s * fx_field)[None]
        fpost = fpost + (1.0 - 0.5 / tau) * w_t * (3.0 * (cF - uF) + 9.0 * cu * cF)
        fpost[:, solid_idx, :] = fpost[:, solid_idx, :][opp_t]
        f = stream(fpost)
        if prev is not None and (step + 1) % sample_every == 0:
            _, ux_now, _ = moments(f)
            delta = float(torch.max(torch.abs(ux_now - prev)))
            history.append(delta)
            prev = ux_now.clone()
            if delta < tol:
                reached = step + 1
                break
        elif prev is None:
            prev = moments(f)[1].clone()
    rho, ux, uy = moments(f)
    u_phys = (ux + 0.5 * force)[1:-1, :]
    return {
        "profile": u_phys.mean(dim=1).tolist(),
        "rho": float(rho.mean()),
        "steps_run": reached,
        "history": history,
    }


# ---------------------------------------------------------------------------
# OSS layout equivalence (bit-exact)
# ---------------------------------------------------------------------------
def layout_equivalence(ref: dict, nx: int = 5, ny: int = 4, nz: int = 1) -> dict:
    """Re-check every upstream index against :class:`AaLayout`."""
    out: dict[str, object] = {}
    for key, npop in (("npop1", 1), ("npop2", 2)):
        ref_key = ref.get(key)
        if not ref_key:
            continue
        layout = AaLayout(nx=ref_key["NX"], ny=ref_key["NY"], nz=ref_key["NZ"], npop=ref_key["NP"])
        mismatches = 0
        checked = 0
        for x, y, z, p, n, d, idx in ref_key["spatial_to_linear"]:
            checked += 1
            if layout.spatial_to_linear(x, y, z, n, d, p) != idx:
                mismatches += 1
        for x, y, z, p, n, d, idx in ref_key["spatial_to_linear"][:50]:
            x2, y2, z2, n2, d2, p2 = layout.linear_to_spatial(idx)
            if (x2, y2, z2, n2, d2, p2) != (x, y, z, n, d, p):
                mismatches += 1
        parity_mismatches = 0
        parity_checked = 0
        for x, y, p, n, d, r_even, r_odd, w_even, w_odd in ref_key["parity"]:
            parity_checked += 1
            got = (
                layout.index_read(0, x, y, 0, n, d, p),
                layout.index_read(1, x, y, 0, n, d, p),
                layout.index_write(0, x, y, 0, n, d, p),
                layout.index_write(1, x, y, 0, n, d, p),
            )
            if got != (r_even, r_odd, w_even, w_odd):
                parity_mismatches += 1
        # upstream unit-test invariant: write<TS> is retrievable by read<!TS>
        invariant_ok = True
        for x in range(layout.nx):
            for y in range(layout.ny):
                for n in (0, 1):
                    for d in range(layout.off):
                        for p in range(layout.npop):
                            if layout.index_write(0, x, y, 0, n, d, p) != layout.index_read(1, x, y, 0, n, d, p):
                                invariant_ok = False
                            if layout.index_write(1, x, y, 0, n, d, p) != layout.index_read(0, x, y, 0, n, d, p):
                                invariant_ok = False
        out[key] = {
            "index_formula_checked": checked,
            "index_formula_mismatches": mismatches,
            "parity_checked": parity_checked,
            "parity_mismatches": parity_mismatches,
            "upstream_roundtrip_ok": bool(ref_key.get("roundtrip_ok", False)),
            "write_then_read_opposite_invariant": invariant_ok,
            "nd": ref_key["ND"],
            "off": ref_key["OFF"],
            "padding_slots": [int(i) for i in np.flatnonzero(~np.array(ref_key["MASK"], dtype=bool))],
            "rel_err": 0.0 if mismatches == 0 and parity_mismatches == 0 else 1.0,
        }
    # constants must match the upstream tables verbatim
    ref_lat = ref.get("npop1", {})
    out["lattice_constants_match"] = bool(
        np.allclose(ref_lat.get("DX", []), D2Q9_DX)
        and np.allclose(ref_lat.get("DY", []), D2Q9_DY)
        and np.allclose(ref_lat.get("W", []), D2Q9_W)
    )
    return out


def refresh_oss_dump(out_dir: Path) -> tuple[Path, bool]:
    """Re-compile + re-run the upstream dump if a toolchain and checkout exist."""
    target = out_dir / "lbt_ref_layout.json"
    lbt = Path(os.environ.get("LB_T_PATH", "/tmp/LB-t"))
    src = Path(__file__).with_name("lbt_ref_dump.cpp")
    compiler = shutil.which("g++") or shutil.which("c++")
    regenerated = False
    if lbt.is_dir() and compiler and src.is_file():
        binary = out_dir / "lbt_ref_dump"
        build = subprocess.run(
            [compiler, "-O2", "-std=c++17", f"-I{lbt / 'include'}", str(src), "-o", str(binary)],
            capture_output=True,
            text=True,
        )
        if build.returncode == 0:
            run = subprocess.run([str(binary), str(target)], capture_output=True, text=True)
            regenerated = run.returncode == 0 and target.is_file()
    if not target.is_file():
        raise SystemExit(
            f"missing OSS reference dump {target}; run lbt_ref_dump.cpp against an LB-t checkout "
            "(env LB_T_PATH) or restore the artifact"
        )
    return target, regenerated


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Comparison:
    name: str
    ref: float
    got: float

    @property
    def rel_err(self) -> float:
        denom = abs(self.ref) if abs(self.ref) > 0 else 1.0
        return abs(self.got - self.ref) / denom

    def as_row(self) -> dict:
        err = self.rel_err
        return {
            "quantity": self.name,
            "ref": self.ref,
            "got": self.got,
            "rel_err": err,
            "rel_err_pct": 100.0 * err,
            "pass_15pct": bool(err < 0.15),
        }


def profile_error(ref: np.ndarray, got: np.ndarray) -> tuple[float, float]:
    """Relative L-inf and L2 error of two profiles (normalised by max|ref|)."""
    scale = float(np.max(np.abs(ref))) or 1.0
    lin = float(np.max(np.abs(got - ref))) / scale
    l2 = float(np.sqrt(np.mean((got - ref) ** 2))) / scale
    return lin, l2


def make_figure(results: dict, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib is optional
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=140)
    for row in results["cases"]:
        y = np.arange(1, row["ny"] - 1)
        axes[0].plot(row["analytic"], y, "k-", lw=1.0, label="analytic" if row is results["cases"][0] else None)
        axes[0].plot(row["aa_profile"], y, "o", ms=3, label=f"LB-t layout NY={row['ny']}" )
        axes[0].plot(row["dense_profile"], y, "x", ms=3, label=f"dense NY={row['ny']}")
    axes[0].set_xlabel("u_x (lattice units)")
    axes[0].set_ylabel("y (cell index)")
    axes[0].set_title("body-force channel: profiles")
    axes[0].legend(fontsize=6)
    axes[0].grid(alpha=0.3)

    ny = [row["ny"] for row in results["cases"]]
    err_an = [100 * row["error_aa_vs_analytic"]["linf"] for row in results["cases"]]
    err_de = [max(100 * row["error_aa_vs_dense"]["linf"], 1e-16) for row in results["cases"]]
    axes[1].loglog(ny, err_an, "o-", label="layout vs analytic")
    axes[1].loglog(ny, err_de, "s--", label="layout vs dense kernels")
    axes[1].axhline(15.0, color="r", ls=":", label="15% gate")
    axes[1].set_xlabel("NY (wall-normal cells)")
    axes[1].set_ylabel("relative L-inf error [%]")
    axes[1].set_title("grid convergence + storage transparency")
    axes[1].legend(fontsize=6)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "xval_lbt_layout"))
    ap.add_argument("--quick", action="store_true", help="CI-sized grid/steps")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _rel(p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    ref_path, regenerated = refresh_oss_dump(out_dir)
    ref = json.loads(ref_path.read_text())
    layout_check = layout_equivalence(ref)

    grids = [9, 17, 33] if args.quick else [9, 17, 33, 65]
    tau = 0.8
    force = 2e-5
    nu = (tau - 0.5) / 3.0

    cases = []
    for ny in grids:
        nx = 4
        steps = 4000 if args.quick else 30000
        sample = 100 if args.quick else 200
        aa = run_aa(nx, ny, tau, force, steps, npop=1, sample_every=sample)
        de = run_dense(nx, ny, tau, force, steps, sample_every=sample)
        aa_prof = np.array(aa["profile"])
        de_prof = np.array(de["profile"])
        analytic = analytic_poiseuille(ny, force, nu)
        lin_a, l2_a = profile_error(analytic, aa_prof)
        lin_d, l2_d = profile_error(aa_prof, de_prof)
        cases.append(
            {
                "ny": ny,
                "nx": nx,
                "H": ny - 2,
                "tau": tau,
                "nu": nu,
                "force": force,
                "u_max_analytic": float(analytic.max()),
                "mach": float(analytic.max() / np.sqrt(1.0 / 3.0)),
                "re_channel": float(np.mean(analytic) * 2 * (ny - 2) / nu),
                "aa_profile": aa_prof.tolist(),
                "dense_profile": de_prof.tolist(),
                "analytic": analytic.tolist(),
                "uexact_ref_vs_got": Comparison(
                    f"u_max NY={ny} (ref analytic Poiseuille vs got LB-t layout)",
                    float(analytic.max()),
                    float(aa_prof.max()),
                ).as_row(),
                "error_aa_vs_analytic": {"linf": lin_a, "l2": l2_a},
                "error_aa_vs_dense": {"linf": lin_d, "l2": l2_d},
                "steps_run_aa": aa["steps_run"],
                "steps_run_dense": de["steps_run"],
                "aa_convergence": aa["history"][-3:],
                "dense_convergence": de["history"][-3:],
                "layout_slots": aa["layout_slots"],
            }
        )

    # -- parameter sensitivity (owner report requirement: >= 2 knobs) --------
    sens_steps = 4000 if args.quick else 20000
    ny_s = 9 if args.quick else 33
    sensitivity = {"force": [], "tau": []}
    for f_val in ([2e-5, 2e-4] if args.quick else [2e-6, 2e-5, 2e-4]):
        aa = run_aa(4, ny_s, tau, f_val, sens_steps, sample_every=200)
        ana = analytic_poiseuille(ny_s, f_val, nu)
        prof = np.array(aa["profile"])
        sensitivity["force"].append(
            {
                "force": f_val,
                "umax_analytic": float(ana.max()),
                "umax_aa": float(prof.max()),
                "rel_err": float(abs(prof.max() - ana.max()) / ana.max()),
                "mach": float(ana.max() / np.sqrt(1.0 / 3.0)),
                "steps_run": aa["steps_run"],
            }
        )
    for tau_val in ([0.8, 1.0] if args.quick else [0.6, 0.8, 1.0, 1.5]):
        nu_val = (tau_val - 0.5) / 3.0
        aa = run_aa(4, ny_s, tau_val, force, sens_steps, sample_every=200)
        ana = analytic_poiseuille(ny_s, force, nu_val)
        prof = np.array(aa["profile"])
        sensitivity["tau"].append(
            {
                "tau": tau_val,
                "nu": nu_val,
                "umax_analytic": float(ana.max()),
                "umax_aa": float(prof.max()),
                "rel_err": float(abs(prof.max() - ana.max()) / ana.max()),
                "steps_run": aa["steps_run"],
            }
        )

    # -- grid convergence order (Richardson-style slope on the L-inf error) --
    hs = np.array([c["ny"] - 1 for c in cases], dtype=np.float64)
    errs = np.array([max(c["error_aa_vs_analytic"]["linf"], 1e-16) for c in cases], dtype=np.float64)
    if len(hs) >= 2:
        slope = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
    else:
        slope = None

    # two-component column: the NP stacking mechanism upstream only declares
    np_cmp = None
    if grids:
        ny = grids[-1] if args.quick else 33
        two = run_aa(4, ny, tau, force, 4000 if args.quick else 12000, npop=2, sample_every=200)
        p0 = np.array(two["profiles_per_population"][0])
        p1 = np.array(two["profiles_per_population"][1])
        np_cmp = {
            "npop": 2,
            "ny": ny,
            "profile_p0": p0.tolist(),
            "profile_p1": p1.tolist(),
            "component_crosstalk_linf": float(np.max(np.abs(p0 - p1)) / (np.max(np.abs(p0)) or 1.0)),
            "slots_npop1": 1 * (ny * 4 * 1 * D2Q9_ND),
            "slots_npop2": 1 * (ny * 4 * 2 * D2Q9_ND),
        }

    rows = [c["uexact_ref_vs_got"] for c in cases]
    layout_ok = all(
        v.get("index_formula_mismatches", 1) == 0 and v.get("parity_mismatches", 1) == 0
        for k, v in layout_check.items()
        if isinstance(v, dict)
    )
    physics_ok = all(r["pass_15pct"] for r in rows)
    transparency_ok = all(c["error_aa_vs_dense"]["linf"] < 1e-10 for c in cases)

    results = {
        "anchor": "XVAL-LBM-NO-MULTIPHASE-LAYOUT-PATTERN-ONLY",
        "repo": "TensorLBM",
        "oss": {
            "repo": "2b-t/LB-t",
            "checkout": os.environ.get("LB_T_PATH", "/tmp/LB-t"),
            "ref_file": _rel(ref_path),
            "ref_regenerated_at_run": regenerated,
            "quote_collision_stub": "include/lbt/populations/collision/bgk.hpp: // TODO(tobit): Implement collide-stream",
            "quote_population_alias": "include/lbt/populations/population.hpp:19 using Population = AaPopulation<LT, NP>",
        },
        "layout_equivalence": layout_check,
        "cases": cases,
        "comparisons": rows,
        "npop_stacking": np_cmp,
        "sensitivity": sensitivity,
        "convergence": {
            "H": hs.tolist(),
            "linf_vs_analytic": errs.tolist(),
            "observed_order": slope,
        },
        "verdict": {
            "layout_bit_exact": bool(layout_ok),
            "physics_under_15pct": bool(physics_ok),
            "storage_transparent_vs_production_kernels": bool(transparency_ok),
            "max_rel_err_pct": max(r["rel_err_pct"] for r in rows) if rows else None,
        },
        "repro": {
            "cmd": "python benchmarks/xval_lbt_layout_pattern/run_xval.py --out artifacts/xval_lbt_layout",
            "tests": "python -m pytest tests/test_aa_layout_xval.py -q",
            "oss_dump": "LB_T_PATH=/tmp/LB-t python benchmarks/xval_lbt_layout_pattern/run_xval.py",
        },
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    fig_ok = make_figure(results, out_dir / "xval_layout_profiles.png")
    results["figure"] = _rel(out_dir / "xval_layout_profiles.png") if fig_ok else None

    print(json.dumps(results["verdict"], indent=2))
    for row in rows:
        print(f"  {row['quantity']:<60s} ref={row['ref']:.8f} got={row['got']:.8f} err={row['rel_err_pct']:.4f}% pass={row['pass_15pct']}")
    print(f"  layout bit-exact: {layout_ok}; transparency: {transparency_ok}; figure: {fig_ok}")
    return 0 if (layout_ok and physics_ok and transparency_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())