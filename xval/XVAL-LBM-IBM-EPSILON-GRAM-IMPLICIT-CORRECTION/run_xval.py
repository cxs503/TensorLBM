"""XVAL runner: LIFE epsilon-Gram implicit correction vs the uncorrected IBM.

Anchor: XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION
OSS reference: joconnor22/LIFE @ 2af2428 (src/Objects.cpp:235-320 computeEpsilon,
src/IBMNode.cpp:73-74 spread, inc/Utils.h:220-231 diracDelta).

Arms
----
A1  LIFE 3-point delta + epsilon-Gram correction (transplant), spread weight
    ``w = epsilon * ds * diracVal``                               (under test)
A2  same LIFE 3-point delta and supports, ``epsilon = 1``          (control,
    isolates the correction as the single changed variable)
A3  TensorLBM in-repo ``IBMStencil`` kernel="4pt" baseline         (in-repo
    reference for the difference list; physics arm only)

Part A  force-transfer round trip / deposited-force bookkeeping on a marker
        ring (no time integration).
Part B  2-D fixed cylinder in uniform flow, force-based IBM, drag coefficient.

Every number written to ``results.json``/``benchmark.json`` is produced by this
run; provenance (git sha, dtype, torch version, host, elapsed) is recorded.
Thresholds live in ``predeclared.json`` and are not tuned after the fact.

Usage
-----
    PYTHONPATH=src python xval/XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION/run_xval.py
    ... --quick                       # N=16 arms + 200-step physics smoke
    ... --steps 1200 --transient 800  # longer cylinder average
    ... --ny 96 --nx 192              # lower-blockage box (sensitivity point)
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

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorlbm.ibm_epsilon_gram import (  # noqa: E402
    build_spread_matrix_2d,
    build_support_2d,
    compute_ds_2d,
    compute_epsilon_gram,
    interpolate_force_2d,
    interpolate_force_2d_vec,
    life_correction,
    make_life_ibm_stencil,
    spread_force_2d,
    spread_force_2d_vec,
    spread_with_correction,
)

DTYPE = torch.float64
LIT_CD_BAND = (1.30, 1.40)  # unbounded 2-D cylinder, Re=100 (Braza 1986 / Norberg 1993 range)


def provenance() -> dict:
    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "script": os.path.relpath(os.path.abspath(__file__), ROOT),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "dtype": "float64",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# --------------------------------------------------------------------------- #
# Part A: round trip + deposited-force bookkeeping
# --------------------------------------------------------------------------- #
def circle_markers(n: int, cx: float, cy: float, r: float) -> tuple[torch.Tensor, torch.Tensor]:
    th = torch.arange(n, dtype=DTYPE) * (2.0 * math.pi / n)
    return cx + r * torch.cos(th), cy + r * torch.sin(th)


def run_roundtrip(n_markers: int, force_field: str, arm: str,
                  ny: int = 64, nx: int = 64, r: float = 10.0,
                  cx: float = 32.0, cy: float = 32.0) -> dict:
    mx, my = circle_markers(n_markers, cx, cy, r)
    supports = build_support_2d(mx, my, ny, nx)
    ds = compute_ds_2d(mx, my)
    eps = compute_epsilon_gram(mx, my, supports, ds) if arm == "A1" else torch.ones_like(ds)

    th = torch.arctan2(my - cy, mx - cx)
    if force_field == "constant":
        fx_t = torch.ones(n_markers, dtype=DTYPE)
        fy_t = torch.zeros(n_markers, dtype=DTYPE)
    elif force_field == "radial":
        fx_t = torch.cos(th)
        fy_t = torch.sin(th)
    else:  # pragma: no cover - defensive
        raise ValueError(force_field)

    fxg, fyg = spread_force_2d(fx_t, fy_t, supports, eps, ds, ny, nx)
    fxr, fyr = interpolate_force_2d(fxg, fyg, supports)
    rec = torch.cat([fxr, fyr])
    tgt = torch.cat([fx_t, fy_t])
    l2 = float((rec - tgt).norm().item() / tgt.norm().item())
    mx_abs = float((rec - tgt).abs().max().item() / tgt.abs().max().item())

    res = {
        "n_markers": n_markers,
        "force_field": force_field,
        "arm": arm,
        "roundtrip_l2_rel": l2,
        "roundtrip_maxabs_rel": mx_abs,
        "epsilon_min": float(eps.min().item()),
        "epsilon_max": float(eps.max().item()),
        "epsilon_mean": float(eps.mean().item()),
        "ds_mean": float(ds.mean().item()),
        "target_total_x": float((fx_t * ds).sum().item()),
        "got_total_x": float(fxg.sum().item()),
    }
    tgt_tot = (torch.stack([fx_t, fy_t], dim=1) * ds.unsqueeze(1)).sum(dim=0)
    got_tot = torch.stack([fxg.sum(), fyg.sum()])
    # normalise by the total absolute marker flux (well defined also for the
    # radial field, whose net target is zero)
    flux = float((tgt.abs() * ds.repeat(2)).sum().item())
    res["deposit_dev_rel"] = float(((got_tot - tgt_tot).abs().sum().item()) / flux)
    res["deposit_abs_x"] = float(got_tot[0].abs().item())
    res["deposit_abs_y"] = float(got_tot[1].abs().item())
    return res


def selftest_vectorisation() -> dict:
    """Prove the vectorised ops are semantics-identical to the loop form."""
    torch.manual_seed(7)
    n = 9
    th = torch.arange(n, dtype=DTYPE) * (2.0 * math.pi / n)
    mx = 20.0 + 6.0 * torch.cos(th)
    my = 32.0 + 6.0 * torch.sin(th)
    ny, nx = 64, 64
    supports = build_support_2d(mx, my, ny, nx)
    ds = compute_ds_2d(mx, my)
    eps = compute_epsilon_gram(mx, my, supports, ds)
    fx = torch.randn(n, dtype=DTYPE)
    fy = torch.randn(n, dtype=DTYPE)
    S, _ = build_spread_matrix_2d(mx, my, ny, nx)
    fxg1, fyg1 = spread_force_2d(fx, fy, supports, eps, ds, ny, nx)
    fxg2, fyg2 = spread_force_2d_vec(fx, fy, S, eps, ds, ny, nx)
    fxr1, fyr1 = interpolate_force_2d(fxg1, fyg1, supports)
    fxr2, fyr2 = interpolate_force_2d_vec(fxg1, fyg1, S)
    st = make_life_ibm_stencil(torch.stack([mx, my], dim=1), (ny, nx))
    sx = spread_with_correction(st, fx, fy, eps, ds)
    return {
        "spread_maxdiff_loop_vs_vec": float((fxg1 - fxg2).abs().max().item()),
        "interp_maxdiff_loop_vs_vec": float((fxr1 - fxr2).abs().max().item()),
        "spread_maxdiff_stencil_vs_vec": float(max((fxg2 - sx[0]).abs().max().item(),
                                                   (fyg2 - sx[1]).abs().max().item())),
        "pass": bool((fxg1 - fxg2).abs().max().item() < 1e-14
                     and (fxr1 - fxr2).abs().max().item() < 1e-14
                     and (fxg2 - sx[0]).abs().max().item() < 1e-14),
    }


# --------------------------------------------------------------------------- #
# Part B: 2D fixed cylinder drag (force-based IBM on the main solver loop)
# --------------------------------------------------------------------------- #
def cylinder_drag(n_steps: int, arm: str, nx: int = 128, ny: int = 64,
                  diameter: float = 8.0, u_target: float = 0.1,
                  tau: float = 0.524, n_transient: int = 500) -> dict:
    """Fixed cylinder in uniform flow; force-based IBM.

    A1/A2 ride on the repo's ``IBMStencil`` (LIFE supports, corrected/plain
    spread); A3 uses the in-repo ``build_stencil(kernel="4pt")`` path.
    """
    from tensorlbm.d2q9 import equilibrium, macroscopic
    from tensorlbm.solver import collide_bgk, stream
    from tensorlbm.ibm_force_transfer import build_cylinder_markers_2d, build_stencil
    from tensorlbm.ibm_force_transfer import body_source_2d, interior_mask_force
    from tensorlbm.ibm_validation_cases import disc_mask_2d

    nu = (tau - 0.5) / 3.0
    re = u_target * diameter / nu
    c = (0.35 * nx, 0.5 * ny)
    markers = build_cylinder_markers_2d(c, 0.5 * diameter, spacing=0.8)
    pts = markers
    shape = (ny, nx)

    t_eps = time.time()
    if arm in ("A1", "A2"):
        eps, ds = life_correction(pts, shape)
        if arm == "A2":  # control: identical supports and ds, epsilon = 1
            eps = torch.ones_like(ds)
        stencil = make_life_ibm_stencil(pts, shape)
    else:  # A3: in-repo baseline, 4-point kernel, no correction
        eps = torch.ones(pts.shape[0], dtype=DTYPE)
        ds = torch.ones(pts.shape[0], dtype=DTYPE)
        stencil = build_stencil(pts, shape, kernel="4pt")
    t_eps = time.time() - t_eps

    mask = disc_mask_2d(shape, c, max(0.5 * diameter - 0.5, 0.5))
    rho = torch.ones(shape, dtype=DTYPE)
    ux = u_target * torch.ones_like(rho)
    uy = torch.zeros_like(rho)
    f = equilibrium(rho, ux, uy)

    drag_hist: list[float] = []
    t0 = time.time()
    for step in range(n_steps):
        _, uxf, uyf = macroscopic(f)
        u_grid = torch.stack([uxf, uyf])
        u_int = stencil.interpolate(u_grid).t()  # (M, 2)
        rho_m = torch.maximum(stencil.interpolate(rho), torch.tensor(1e-12, dtype=DTYPE))
        # direct forcing with the two Guo corrector passes (repo direct_forcing)
        f_m = u_int.clone()
        for _ in range(2):
            u_phys = u_int + 0.5 * f_m / rho_m.unsqueeze(1)
            f_m = f_m + (torch.zeros_like(u_int) - u_phys)
        if arm == "A3":
            fx_grid, fy_grid = stencil.spread(f_m)  # IBMStencil vector layout (M, C)
        else:
            fx_grid, fy_grid = spread_with_correction(stencil, f_m[:, 0], f_m[:, 1], eps, ds)
        f_int = interior_mask_force(u_grid, mask, torch.zeros_like(u_grid))
        fx = fx_grid + f_int[0]
        fy = fy_grid + f_int[1]
        f = collide_bgk(f, tau)
        f = body_source_2d(f, fx, fy)
        f = stream(f)
        if step >= n_transient:
            drag_hist.append(float(-(fx_grid + f_int[0]).sum().item()))
    elapsed = time.time() - t0

    cd_inst = [d / (0.5 * 1.0 * u_target ** 2 * diameter) for d in drag_hist]
    half = len(cd_inst) // 2
    cd_mean = sum(cd_inst) / len(cd_inst)
    cd_tail = sum(cd_inst[half:]) / max(len(cd_inst) - half, 1)
    return {
        "arm": arm,
        "re": re,
        "u": u_target,
        "diameter": diameter,
        "nu": nu,
        "tau": tau,
        "nx": nx,
        "ny": ny,
        "blockage": diameter / ny,
        "n_markers": int(pts.shape[0]),
        "marker_spacing": 0.8,
        "n_steps": n_steps,
        "n_transient": n_transient,
        "n_avg": len(cd_inst),
        "cd_mean": cd_mean,
        "cd_tail_half_mean": cd_tail,
        "cd_std": float(torch.tensor(cd_inst).std().item()) if len(cd_inst) > 1 else 0.0,
        "epsilon_mean": float(eps.mean().item()),
        "eps_and_stencil_setup_s": t_eps,
        "integration_s": elapsed,
        "drag_history": cd_inst,
    }


# --------------------------------------------------------------------------- #
# Benchmark table (ref vs got)
# --------------------------------------------------------------------------- #
def build_benchmark(res: dict) -> dict:
    rows: list[dict] = []

    def add(kpi: str, ref: float, got: float, tol: float, src: str, note: str = "") -> None:
        err = abs(got - ref)
        rel = err / abs(ref) if ref != 0 else err
        rows.append({
            "kpi": kpi, "ref": ref, "got": got, "abs_err": err,
            "rel_err": rel, "tol": tol, "pass": bool(rel <= tol), "ref_src": src, "note": note,
        })

    # --- exact analytic references -------------------------------------
    st = res["module_selftest"]
    add("dirac_partition_unity_1d_maxerr", 0.0, st["partition_unity_1d_maxerr"], 1e-12,
        "analytic: sum_i phi(|r-i|) = 1 for the LIFE 3-point delta (exact, all offsets)")
    add("dirac_sum_2d_lattice_aligned_maxerr", 0.0, st["dirac_sum_2d_maxerr"], 1e-12,
        "analytic: separable 2-D support sum = 1 (exact at lattice-aligned markers)")
    add("gram_roundtrip_A1_N16_l2", 0.0, st["roundtrip_l2_A1"], 1e-12,
        "analytic: non-overlapping supports => A^-1 Gram solve makes spread->interp exact")
    add("gram_roundtrip_A2_N16_l2", 0.0, st["roundtrip_l2_A2"], 1e-3,
        "control arm (epsilon=1) - expected NON-zero, quantifies the discretisation defect",
        "control, not a requirement")
    add("stencil_spread_vs_vectorised_maxdiff", 0.0, st["stencil_spread_maxdiff"], 1e-14,
        "bit-level: IBMStencil.spread(f*eps*ds) == spread_force_2d_vec")
    add("stencil_interp_vs_vectorised_maxdiff", 0.0, st["stencil_interp_maxdiff"], 1e-14,
        "bit-level: IBMStencil.interpolate == interpolate_force_2d_vec")

    # --- round-trip / deposit over marker layouts -----------------------
    for key, a in res["arms"].items():
        n = a["n_markers"]
        arm = a["arm"]
        if arm == "A1":
            add(f"roundtrip_l2[{key}]", 0.0, a["roundtrip_l2_rel"], 1e-2,
                "analytic identity (spread->interpolate returns the input marker force)",
                "correction is a normalisation fix; 1e-2 is 15x inside the owner's 15% band")
        else:
            add(f"roundtrip_l2[{key}]", 0.0, a["roundtrip_l2_rel"], 1e9,
                "control arm, no gate (quantifies what the correction removes)",
                "control")
        add(f"deposit_dev[{key}]", 0.0, a["deposit_dev_rel"], 1e-1,
            "exact bookkeeping: sum(F_grid) = sum(f_marker*ds) when epsilon=1",
            "the correction trades this by O(epsilon-1); reported, not hidden")

    # --- physics arm -----------------------------------------------------
    phys = res["physics"]
    for arm, p in phys.items():
        add(f"cd_re100[{arm}]", 0.5 * sum(LIT_CD_BAND), p["cd_mean"], 0.15,
            f"literature band {LIT_CD_BAND} for the unbounded 2-D cylinder at Re=100 "
            "(Braza 1986 / Norberg 1993)",
            f"periodic box ny={p['ny']}, blockage D/ny={p['blockage']:.3f} raises Cd above the "
            "unbounded value; range/blockage check, see REPORT §5")
    return {
        "case": "XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION",
        "generated_utc": res["provenance"]["started_utc"],
        "git_sha": res["provenance"]["git_sha"],
        "tolerance_policy": "ref/got/abs_err/rel_err; pass = rel_err <= tol",
        "rows": rows,
        "n_pass": sum(1 for r in rows if r["pass"]),
        "n_rows": len(rows),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="N=16 arms only + 200-step physics smoke")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--transient", type=int, default=500)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--arms", default="A1,A2,A3", help="comma separated physics arms")
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args(argv)

    t_start = time.time()
    out: dict = {"provenance": provenance(), "config": vars(args),
                 "module_selftest": __import__("tensorlbm.ibm_epsilon_gram", fromlist=["selftest"]).selftest(verbose=False),
                 "selftest": selftest_vectorisation(), "arms": {}, "physics": {}}

    ns = (16,) if args.quick else (16, 32)
    for n in ns:
        for ff in ("constant", "radial"):
            for arm in ("A1", "A2"):
                key = f"N{n}_{ff}_{arm}"
                out["arms"][key] = run_roundtrip(n, ff, arm)
                a = out["arms"][key]
                print(f"[A] {key}: rt_l2={a['roundtrip_l2_rel']:.3e} "
                      f"deposit_dev={a['deposit_dev_rel']:.3e} "
                      f"eps=[{a['epsilon_min']:.4f},{a['epsilon_max']:.4f}]", flush=True)

    steps = 200 if args.quick else args.steps
    transient = 100 if args.quick else args.transient
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        p = cylinder_drag(n_steps=steps, arm=arm, nx=args.nx, ny=args.ny, n_transient=transient)
        hist = p.pop("drag_history")
        out["physics"][arm] = p
        out.setdefault("drag_history", {})[arm] = hist
        print(f"[B] {arm}: Re={p['re']:.1f} Cd={p['cd_mean']:.4f} "
              f"(tail {p['cd_tail_half_mean']:.4f}, n={p['n_avg']}, "
              f"{p['integration_s']:.1f}s, eps_mean={p['epsilon_mean']:.4f})", flush=True)

    out["physics_reference"] = {
        "case": "2-D fixed cylinder, force-based IBM, periodic box, uniform inflow",
        "re": 100.0,
        "cd_literature_band_unbounded": LIT_CD_BAND,
        "citation": "Braza, Chassaing & Ha Minh 1986; Norberg 1993 (Re=100, unbounded)",
        "blockage_note": "D/ny is finite: periodic images + confinement raise Cd; the "
                         "band is used as a range check, not as a <15% acceptance gate",
    }
    out["elapsed_s"] = time.time() - t_start
    bench = build_benchmark(out)
    out["benchmark"] = bench

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    bench_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "benchmark.json")
    with open(bench_path, "w") as fh:
        json.dump(bench, fh, indent=2)
    print(f"wrote {args.out} and {bench_path} "
          f"({bench['n_pass']}/{bench['n_rows']} benchmark rows pass, {out['elapsed_s']:.1f}s)")
    print("selftest:", out["selftest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())