#!/usr/bin/env python3
"""XVAL driver: pylbm pytest-baseline-reference practice vs TensorLBM.

Arms
----
A. ``ghia``   : TensorLBM production cavity run (``run_lid_driven_cavity``)
                scored against the Ghia et al. (1982) Re=100 tables at two grids.
B. ``pylbm``  : TensorLBM run at *pylbm's own demo parameters* (nx=64,
                u_lid=0.05, tau=0.5+3*mu/dx with mu=5e-6, t=0.5 -> 32 steps)
                compared field-by-field with the checked-in OSS reference
                ``test2D_lid_driven_cavity.h5`` (zero hand-copying: the arrays
                are read from the OSS file).
C. ``baseline``: register the cavity block as the git-tracked regression
                baseline and re-run the comparison (upstream semantics).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorlbm import GHIA_RE100, LidDrivenCavityConfig, equilibrium, macroscopic, run_lid_driven_cavity  # noqa: E402
import tensorlbm.lid_driven_cavity as _cavity_mod  # noqa: E402
from tensorlbm.baseline_io import read_block, to_numpy  # noqa: E402

# the CI image's torch<->numpy bridge is broken (see baseline_io.to_numpy); the
# production snapshot writer calls Tensor.numpy(), so it is disabled here and
# the figures below are built from the reference block instead.
_cavity_mod._save_cavity_snapshot = lambda *a, **k: None

# The production Ghia scoring block segfaults on this image for nx=65 (see
# REPORT.md "friction"); the XVAL driver scores the centre lines itself from the
# reference block (independent recompute), so the production hook is stubbed to
# NaN and the block is rewritten with the driver's numbers afterwards.
from tensorlbm.lid_driven_cavity import GHIA_METRIC_CONVENTIONS  # noqa: E402
_NAN_SCORES = {conv: {"rmse_u": float("nan"), "rmse_v": float("nan")} for conv in GHIA_METRIC_CONVENTIONS}
_cavity_mod.ghia_metric_report = lambda *a, **k: {
    "scores": _NAN_SCORES, "boundary_recipe": None, "wall_planes": None, "analytic_floor": None}
from tensorlbm.boundary.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.solver import collide_bgk, collide_mrt, stream  # noqa: E402
from tensorlbm.lid_driven_cavity import make_cavity_wall_mask, zou_he_moving_lid  # noqa: E402
from tensorlbm.verification.baseline_reference import (  # noqa: E402
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    BaselineStore,
    compare_fields,
)

PYLBM_H5 = Path("/root/oss-cache/pylbm/tests/reference/test2D_lid_driven_cavity.h5")
ART = HERE / "artifacts"
FIG = ART / "figures"
BASELINE_NAME = "cavity_re100_nx33_u0.1_s300.npz"


def _key(ref: dict, *candidates: str) -> str:
    for cand in candidates:
        if cand in ref:
            return cand
    raise KeyError(f"none of {candidates} in {sorted(ref)}")


YK, UK = _key(GHIA_RE100, "y", "y/H"), _key(GHIA_RE100, "u", "u/u_lid")
XK, VK = _key(GHIA_RE100, "x", "x/H"), _key(GHIA_RE100, "v", "v/u_lid")


def run_case(nx: int, re: float, u_lid: float, n_steps: int, tag: str) -> dict:
    cfg = LidDrivenCavityConfig(
        nx=nx, re=re, u_lid=u_lid, n_steps=n_steps,
        output_interval=n_steps, output_root=HERE / "runs", run_name=tag, overwrite=True,
    )
    t0 = time.time()
    run_dir = run_lid_driven_cavity(cfg)
    fields, meta = read_block(run_dir / "reference_block.npz")
    return {
        "tag": tag, "nx": nx, "re": re, "u_lid": u_lid, "n_steps": n_steps,
        "tau": cfg.tau, "nu": cfg.nu, "walltime_s": round(time.time() - t0, 2),
        "run_dir": str(run_dir), "fields": fields, "meta": meta,
    }


def ghia_rows(fields: dict, nx: int) -> dict:
    """Per-row ref vs got at every Ghia tabulation point (node centres)."""
    coords = (np.arange(nx) + 0.5) / nx
    u_got = np.interp(GHIA_RE100[YK], coords, fields["ux_centerline_normalised"])
    v_got = np.interp(GHIA_RE100[XK], coords, fields["uy_centerline_normalised"])
    u_ref = np.asarray(GHIA_RE100[UK], dtype=float)
    v_ref = np.asarray(GHIA_RE100[VK], dtype=float)
    u_err = u_got - u_ref
    v_err = v_got - v_ref
    return {
        "u": [
            {"y": float(y), "ref": float(r), "got": float(g), "err": float(e)}
            for y, r, g, e in zip(GHIA_RE100[YK], u_ref, u_got, u_err)
        ],
        "v": [
            {"x": float(x), "ref": float(r), "got": float(g), "err": float(e)}
            for x, r, g, e in zip(GHIA_RE100[XK], v_ref, v_got, v_err)
        ],
        "u_rmse": float(np.sqrt(np.mean(u_err**2))),
        "v_rmse": float(np.sqrt(np.mean(v_err**2))),
        "u_max_abs": float(np.max(np.abs(u_err))),
        "v_max_abs": float(np.max(np.abs(v_err))),
    }


def pylbm_run_fields(nx: int, re: float, u_lid: float, n_steps: int) -> tuple[dict, dict]:
    """TensorLBM kernel loop at pylbm's parameters (same recipe as production)."""
    cfg = LidDrivenCavityConfig(nx=nx, re=re, u_lid=u_lid, n_steps=n_steps)
    cfg.validate()
    f = equilibrium(torch.ones((nx, nx)), torch.zeros((nx, nx)), torch.zeros((nx, nx)))
    wall = make_cavity_wall_mask(nx, nx, torch.device("cpu"), include_top=False)
    collide = collide_mrt if cfg.tau < 0.60 else collide_bgk
    for _ in range(n_steps):
        f = collide(f, tau=cfg.tau)
        f = stream(f)
        f = bounce_back_cells(f, wall)
        f = zou_he_moving_lid(f, u_lid)
    rho, ux, uy = macroscopic(f)
    fields = {
        "rho": to_numpy(rho).astype(np.float64),
        "qx": to_numpy(rho * ux).astype(np.float64),
        "qy": to_numpy(rho * uy).astype(np.float64),
    }
    return fields, {"nx": nx, "tau": cfg.tau, "nu": cfg.nu, "n_steps": n_steps, "u_lid": u_lid}


def pylbm_arm(n_steps: int) -> dict:
    if not PYLBM_H5.exists():
        return {"status": "not_run", "reason": f"missing OSS reference {PYLBM_H5}"}
    import h5py

    with h5py.File(PYLBM_H5, "r") as h5:
        ref = {k: np.asarray(h5[k][...], dtype=np.float64) for k in ("rho", "qx", "qy")}
        grid = {k: np.asarray(h5[f"x_{k}"][...], dtype=np.float64) for k in (0, 1)}
    nu = 3.0 * 5.0e-6 / (1.0 / 64) / 3.0  # pylbm: sigma2 = 3*mu/(la*rho*dx); nu = sigma2/3
    re = 0.05 * 64 / nu
    got, params = pylbm_run_fields(64, re, 0.05, n_steps)
    diffs = compare_fields(ref, got, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL)
    rel_l2 = {
        k: float(np.linalg.norm(got[k] - ref[k]) / np.linalg.norm(ref[k])) for k in ref
    }
    return {
        "status": "ok",
        "oss_reference": str(PYLBM_H5),
        "oss_reference_grid_points": int(grid[0].size),
        "oss_ranges": {k: [float(ref[k].min()), float(ref[k].max())] for k in ref},
        "params": {**params, "re": re, "sigma_2": 3.0 * 5.0e-6 / (1.0 / 64)},
        "per_field": {k: v.as_dict() for k, v in diffs.items()},
        "rel_l2": rel_l2,
        "upstream_tolerance_verdict": {k: v.ok for k, v in diffs.items()},
        "upstream_tolerance": {"atol": DEFAULT_ATOL, "rtol": DEFAULT_RTOL},
    }


def figure_ghia(cases: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for case in cases:
        f = case["fields"]
        n = case["nx"]
        coords = (np.arange(n) + 0.5) / n
        axes[0].plot(f["ux_centerline_normalised"], coords, label=f"TensorLBM nx={n}")
        axes[1].plot(coords, f["uy_centerline_normalised"], label=f"TensorLBM nx={n}")
    axes[0].plot(GHIA_RE100[UK], GHIA_RE100[YK], "ko", ms=3, label="Ghia 1982")
    axes[1].plot(GHIA_RE100[XK], GHIA_RE100[VK], "ko", ms=3, label="Ghia 1982")
    axes[0].set_xlabel("u/u_lid @ x=0.5"), axes[0].set_ylabel("y/H")
    axes[1].set_xlabel("x/H"), axes[1].set_ylabel("v/u_lid @ y=0.5")
    for ax in axes:
        ax.grid(alpha=0.3), ax.legend(fontsize=7)
    fig.suptitle("Lid-driven cavity Re=100: TensorLBM vs Ghia (1982)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def figure_pylbm(arm: dict, got: dict, out: Path) -> None:
    import h5py

    with h5py.File(PYLBM_H5, "r") as h5:
        ref_qx = np.asarray(h5["qx"][...], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    vmax = max(np.abs(ref_qx).max(), np.abs(got["qx"]).max())
    for ax, data, title in (
        (axes[0], ref_qx, "pylbm h5 reference qx"),
        (axes[1], got["qx"], "TensorLBM qx"),
        (axes[2], got["qx"] - ref_qx, "difference"),
    ):
        im = ax.imshow(data, origin="lower", cmap="RdBu_r",
                       vmin=-vmax if title != "difference" else None,
                       vmax=vmax if title != "difference" else None)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"same-parameter comparison (nx=64, u=0.05, {arm['params']['n_steps']} steps)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def figure_summary(ghia: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    labels = [f"nx={c['nx']}" for c in ghia]
    axes[0].bar([f"{l}\nu" for l in labels], [c["rows"]["u_rmse"] for c in ghia], alpha=0.7, label="RMSE u")
    axes[0].bar([f"{l}\nv" for l in labels], [c["rows"]["v_rmse"] for c in ghia], alpha=0.4, label="RMSE v")
    axes[0].axhline(0.15, color="r", ls="--", lw=0.8, label="owner 15% band")
    axes[0].axhline(0.03, color="g", ls=":", lw=0.8, label="3% target")
    axes[0].set_ylabel("RMSE / u_lid"), axes[0].legend(fontsize=7), axes[0].grid(alpha=0.3)
    for c in ghia:
        axes[1].plot(np.arange(c["nx"]), c["fields"]["rho_field"].mean(axis=1) - 1.0,
                     label=f"nx={c['nx']} (drift={float(c['fields']['mass_drift_relative']):+.2e})")
    axes[1].set_xlabel("y index"), axes[1].set_ylabel("mean(rho)-1")
    axes[1].legend(fontsize=7), axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx-coarse", type=int, default=33)
    ap.add_argument("--nx-fine", type=int, default=65)
    ap.add_argument("--steps-coarse", type=int, default=4000)
    ap.add_argument("--steps-fine", type=int, default=8000)
    ap.add_argument("--pylbm-steps", type=int, default=32)
    ap.add_argument("--register-baseline", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    ART.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    summary: dict = {"anchor": "XVAL-LBM-PYTEST-BASELINE-REFERENCE", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ")}

    # --- arm A: Ghia literature anchor ----------------------------------
    ghia = []
    for nx, steps in ((args.nx_coarse, args.steps_coarse), (args.nx_fine, args.steps_fine)):
        case = run_case(nx, 100.0, 0.1, steps, tag=f"ghia_nx{nx}_re100_s{steps}")
        rows = ghia_rows(case["fields"], nx)
        case["rows"] = rows
        from tensorlbm.baseline_io import write_block
        write_block(Path(case["run_dir"]) / "reference_block.npz",
                    {**case["fields"], "ghia_rmse_u": rows["u_rmse"], "ghia_rmse_v": rows["v_rmse"]},
                    case["meta"])
        ghia.append(case)
        (ART / f"ghia_nx{nx}.json").write_text(json.dumps(
            {k: v for k, v in case.items() if k != "fields"} | {
                "mass_drift_relative": float(case["fields"]["mass_drift_relative"]),
                "max_speed": float(case["fields"]["max_speed"]),
            }, indent=2))
        print(f"[A] nx={nx} steps={steps} u_rmse={rows['u_rmse']:.4f} v_rmse={rows['v_rmse']:.4f} "
              f"mass_drift={float(case['fields']['mass_drift_relative']):+.3e} ({case['walltime_s']}s)")
    summary["arm_a_ghia"] = [
        {"nx": c["nx"], "n_steps": c["n_steps"], "tau": c["tau"], "walltime_s": c["walltime_s"],
         "u_rmse": c["rows"]["u_rmse"], "v_rmse": c["rows"]["v_rmse"],
         "u_max_abs": c["rows"]["u_max_abs"], "v_max_abs": c["rows"]["v_max_abs"],
         "mass_drift_relative": float(c["fields"]["mass_drift_relative"]),
         "max_speed": float(c["fields"]["max_speed"])}
        for c in ghia
    ]

    # --- arm B: pylbm checked-in reference (same parameters) ------------
    pylbm_main = pylbm_arm(args.pylbm_steps)
    pylbm_steps_sens = pylbm_arm(2 * args.pylbm_steps)
    got32, _ = pylbm_run_fields(64, pylbm_main["params"]["re"], 0.05, args.pylbm_steps) if pylbm_main["status"] == "ok" else ({}, {})
    (ART / "pylbm_ref_vs_got.json").write_text(json.dumps(
        {"main": pylbm_main, "step_sensitivity": pylbm_steps_sens}, indent=2))
    if pylbm_main["status"] == "ok":
        print("[B] pylbm h5 arm rel_l2:", {k: round(v, 4) for k, v in pylbm_main["rel_l2"].items()},
              "| 2x steps:", {k: round(v, 4) for k, v in pylbm_steps_sens["rel_l2"].items()})
        summary["arm_b_pylbm"] = {
            "params": pylbm_main["params"],
            "rel_l2_main": pylbm_main["rel_l2"],
            "rel_l2_2x_steps": pylbm_steps_sens["rel_l2"],
            "max_abs_main": {k: v["max_abs"] for k, v in pylbm_main["per_field"].items()},
            "upstream_tolerance_verdict": pylbm_main["upstream_tolerance_verdict"],
        }
    else:
        print("[B] pylbm arm not run:", pylbm_main.get("reason"))
        summary["arm_b_pylbm"] = pylbm_main

    # --- arm C: baseline regression semantics ---------------------------
    base_case = run_case(33, 100.0, 0.1, 300, tag="baseline_nx33_s300")
    store = BaselineStore()
    if args.register_baseline:
        target = store.register(BASELINE_NAME, Path(base_case["run_dir"]) / "reference_block.npz")
        print(f"[C] registered baseline -> {target}")
    try:
        check = store.compare(BASELINE_NAME, Path(base_case["run_dir"]) / "reference_block.npz")
        check_payload = check.as_dict()
    except Exception as exc:  # honest: not_run rather than a fake pass
        check_payload = {"ok": None, "status": "not_run", "error": str(exc)}
    (ART / "baseline_regression.json").write_text(json.dumps(check_payload, indent=2))
    summary["arm_c_baseline"] = {
        "name": BASELINE_NAME,
        "reference_dir": str(store.reference_dir),
        "ok": check_payload.get("ok"),
        "atol": DEFAULT_ATOL, "rtol": DEFAULT_RTOL,
        "max_abs": {k: v["max_abs"] for k, v in check_payload.get("fields", {}).items()},
    }
    print(f"[C] baseline check ok={check_payload.get('ok')}")

    # --- figures --------------------------------------------------------
    figure_ghia(ghia, FIG / "fig1_ghia_centerlines.png")
    if pylbm_main["status"] == "ok":
        figure_pylbm(pylbm_main, got32, FIG / "fig2_pylbm_field_diff.png")
    figure_summary(ghia, FIG / "fig3_error_and_mass.png")
    summary["figures"] = sorted(str(p.relative_to(HERE)) for p in FIG.glob("*.png"))
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    (ART / "xval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
