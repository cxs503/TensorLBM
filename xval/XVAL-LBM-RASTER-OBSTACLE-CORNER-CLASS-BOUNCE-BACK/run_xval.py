"""Driver for XVAL-LBM-RASTER-OBSTACLE-CORNER-CLASS-BOUNCE-BACK.

Usage (one case, used by the cluster shards)::

    python run_xval.py run --config C1 --arm oss  --outdir runs/C1_oss
    python run_xval.py run --config C1 --arm mask --outdir runs/C1_mask

Usage (aggregate every finished case + analytic anchor + figures)::

    python run_xval.py aggregate --root runs --out results.json --figdir figures

All thresholds used by the aggregation are read from ``predeclared.json``
(written before the ladder was submitted).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from raster_case import (  # noqa: E402
    Arm,
    Case,
    analytic_channel,
    link_difference,
    oss_rect_corner_list,
    raster_solid_mask,
    save_run,
)

CONFIGS: dict[str, dict] = {
    # analytic anchor: no obstacles, exact discrete Poiseuille (pipeline check).
    # g is small here so the anchor stays at Ma << 1 (the empty channel has a
    # large "permeability" ny^2/12, hence a large u_bar for the same g).
    "C0": dict(blocks=False, b=12, nx=132, ny=144, tau=2.0, g=1e-6,
               steps=60000, check_every=2000, u_rel_tol=1e-4),
    # baseline (u_bar ~ g*K/nu = 5e-4*40.5/0.5 ~ 0.04, Ma ~ 0.07)
    "C1": dict(blocks=True, b=12, nx=132, ny=144, tau=2.0, g=5e-4,
               steps=40000, check_every=1000, u_rel_tol=1e-7),
    # viscosity / pore Reynolds sensitivity
    "C2": dict(blocks=True, b=12, nx=132, ny=144, tau=1.5, g=5e-4,
               steps=40000, check_every=1000, u_rel_tol=1e-7),
    # forcing (Mach) sensitivity, larger
    "C3": dict(blocks=True, b=12, nx=132, ny=144, tau=2.0, g=1e-3,
               steps=40000, check_every=1000, u_rel_tol=1e-7),
    # forcing (Mach) sensitivity, smaller
    "C4": dict(blocks=True, b=12, nx=132, ny=144, tau=2.0, g=2.5e-4,
               steps=40000, check_every=1000, u_rel_tol=1e-7),
    # 2x lattice resolution at the same porosity / block count; g scaled by 1/4
    # so that u_bar (and Ma) match C1 while K grows as b^2
    "C5": dict(blocks=True, b=24, nx=264, ny=288, tau=2.0, g=1.25e-4,
               steps=40000, check_every=1000, u_rel_tol=1e-7),
}

ARMS = ("oss", "mask")


def run_case(cfg_name: str, arm: str, outdir: Path) -> dict:
    case = Case(**CONFIGS[cfg_name])
    a = Arm(case, arm)
    res = a.run()
    save_run(outdir, res)
    meta = {k: v for k, v in res.items() if k not in ("ux", "rho", "mask", "solid")}
    meta["config"] = cfg_name
    (outdir / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"[{cfg_name}/{arm}] steps={meta['stopped_at']} u_bar={meta['history'][-1]['u_bar']:.6e} "
          f"K={meta['history'][-1]['k_lattice']:.6f}")
    return meta


def load_case(root: Path, cfg: str, arm: str) -> dict | None:
    d = root / f"{cfg}_{arm}"
    mf = d / "metrics.json"
    ff = d / "fields.npz"
    if not (mf.exists() and ff.exists()):
        return None
    meta = json.loads(mf.read_text())
    with np.load(ff) as z:
        meta["ux"] = z["ux"]
        meta["mask"] = z["mask"]
        meta["solid"] = z["solid"]
    return meta


def aggregate(root: Path, out: Path, figdir: Path) -> dict:
    pre = json.loads((HERE / "predeclared.json").read_text())
    accept = pre["predeclared_thresholds"]["accept_pct"]
    anchor_accept = pre["predeclared_thresholds"]["analytic_anchor_accept_pct"]

    rows = []
    for cfg in CONFIGS:
        got = {}
        for arm in ARMS:
            m = load_case(root, cfg, arm)
            if m is not None:
                got[arm] = m
        if len(got) != 2:
            rows.append(dict(config=cfg, status="missing", arms_present=list(got)))
            continue
        case = Case(**CONFIGS[cfg])
        b = case.b
        k_o = got["oss"]["history"][-1]["k_lattice"]
        k_m = got["mask"]["history"][-1]["k_lattice"]
        fx_o = got["oss"]["history"][-1]["fx_lattice"]
        fx_m = got["mask"]["history"][-1]["fx_lattice"]
        flow_m = got["mask"]["solid"] == False  # noqa: E712
        du = got["mask"]["ux"][flow_m] - got["oss"]["ux"][flow_m]
        ref = got["oss"]["ux"][flow_m]
        rows.append(
            dict(
                config=cfg,
                status="ok",
                blocks=got["oss"]["n_blocks"],
                solid_fraction=got["oss"]["solid_fraction"],
                tau=case.tau,
                g=case.g,
                b=b,
                steps_oss=got["oss"]["stopped_at"],
                steps_mask=got["mask"]["stopped_at"],
                converged_oss=got["oss"]["converged_at"],
                converged_mask=got["mask"]["converged_at"],
                u_bar_oss=got["oss"]["history"][-1]["u_bar"],
                u_bar_mask=got["mask"]["history"][-1]["u_bar"],
                k_oss=k_o,
                k_mask=k_m,
                k_oss_norm=k_o / b ** 2,
                k_mask_norm=k_m / b ** 2,
                err_k_pct=abs(k_m - k_o) / abs(k_o) * 100.0,
                fx_oss=fx_o,
                fx_mask=fx_m,
                err_fx_pct=abs(fx_m - fx_o) / abs(fx_o) * 100.0 if fx_o != 0 else None,
                field_rel_l2=float(np.linalg.norm(du) / np.linalg.norm(ref)) * 100.0,
                field_max_rel=float(np.abs(du).max() / np.abs(ref).max()) * 100.0,
                re_block=got["mask"]["history"][-1]["u_bar"] * b / case.nu,
                mach=got["mask"]["history"][-1]["u_bar"] / (1.0 / np.sqrt(3.0)),
            )
        )

    # analytic anchor (C0): both arms vs the discrete Poiseuille solution
    anchor = {}
    case0 = Case(**CONFIGS["C0"])
    ana = analytic_channel(case0)
    for arm in ARMS:
        m = load_case(root, "C0", arm)
        if m is None:
            anchor[arm] = None
            continue
        u_sim = m["ux"][:, :].mean(axis=0)  # x-independent
        k_sim = m["history"][-1]["k_lattice"]
        anchor[arm] = dict(
            k_sim=k_sim,
            k_analytic=ana["k_lattice"],
            err_k_pct=abs(k_sim - ana["k_lattice"]) / ana["k_lattice"] * 100.0,
            u_profile_sim=u_sim.tolist(),
            u_profile_analytic=ana["u_profile"].tolist(),
            err_profile_max_pct=float(
                np.abs(u_sim - ana["u_profile"]).max() / np.abs(ana["u_profile"]).max() * 100.0
            ),
            steps=m["stopped_at"],
        )
    anchor["accept_pct"] = anchor_accept

    # link-level operator difference (geometry only, no time stepping)
    geom = []
    for cfg in ("C1", "C5"):
        case = Case(**CONFIGS[cfg])
        rects = oss_rect_corner_list(case.nx, case.ny, case.b)
        solid = raster_solid_mask(case.nx, case.ny, rects)
        d = link_difference(solid, rects)
        sp = []
        for x in range(case.nx):
            for y in range(case.ny):
                v = int(d["spurious_mask"][x, y])
                if v:
                    sp.append(dict(x=int(x), y=int(y), dirs=[i for i in range(1, 9) if v >> (i - 1) & 1]))
        geom.append(
            dict(
                config=cfg,
                blocks=len(rects),
                exact_links=d["exact_links"],
                table_links=d["table_links"],
                spurious_links=d["spurious_links"],
                missed_links=d["missed_links"],
                spurious_per_block=d["spurious_links"] / max(len(rects), 1),
                spurious_sites=len(sp),
                spurious_detail=sp[:64],
            )
        )

    decisions = []
    for r in rows:
        if r["status"] != "ok":
            decisions.append(dict(config=r["config"], verdict="MISSING"))
            continue
        decisions.append(
            dict(
                config=r["config"],
                err_k_pct=r["err_k_pct"],
                err_fx_pct=r["err_fx_pct"],
                accept=bool(r["err_k_pct"] < accept),
            )
        )
    payload = dict(
        thresholds=pre["predeclared_thresholds"],
        rows=rows,
        anchor=anchor,
        geometry=geom,
        decisions=decisions,
        all_accept=all(d.get("accept", False) for d in decisions if d.get("verdict") != "MISSING"),
    )
    out.write_text(json.dumps(payload, indent=2, default=float))
    (root / "summary.csv").write_text(
        "config,blocks,solid_fraction,b,tau,g,u_bar_oss,u_bar_mask,K_oss,K_mask,"
        "K_oss_norm,K_mask_norm,err_K_pct,fx_oss,fx_mask,err_fx_pct,field_rel_l2_pct,"
        "field_max_rel_pct,Re_block,Ma,steps_oss,steps_mask\n"
        + "".join(
            ",".join(
                f"{r[k]:.6g}" if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)
                else str(r.get(k))
                for k in (
                    "config", "blocks", "solid_fraction", "b", "tau", "g", "u_bar_oss",
                    "u_bar_mask", "k_oss", "k_mask", "k_oss_norm", "k_mask_norm", "err_k_pct",
                    "fx_oss", "fx_mask", "err_fx_pct", "field_rel_l2", "field_max_rel",
                    "re_block", "mach", "steps_oss", "steps_mask",
                )
            )
            + "\n"
            for r in rows
            if r["status"] == "ok"
        )
    )
    print(f"[aggregate] rows={len(rows)} all_accept={payload['all_accept']} -> {out}")
    for r in rows:
        if r["status"] == "ok":
            fx_err = "n/a" if r["err_fx_pct"] is None else f"{r['err_fx_pct']:.2f}%"
            print(f"  {r['config']}: K_oss={r['k_oss']:.6f} K_mask={r['k_mask']:.6f} "
                  f"err={r['err_k_pct']:.2f}% fx_err={fx_err} "
                  f"field_rel_l2={r['field_rel_l2']:.2f}%")
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--arm", required=True, choices=ARMS)
    r.add_argument("--outdir", required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--root", default=str(HERE / "runs"))
    a.add_argument("--out", default=str(HERE / "results.json"))
    a.add_argument("--figdir", default=str(HERE / "figures"))
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run_case(args.config, args.arm, Path(args.outdir))
        return 0
    payload = aggregate(Path(args.root), Path(args.out), Path(args.figdir))
    fig = HERE / "make_figures.py"
    if fig.exists():
        subprocess.run([sys.executable, str(fig), "--results", str(args.out),
                        "--root", args.root, "--figdir", args.figdir], check=False)
    return 0 if payload.get("rows") else 1


if __name__ == "__main__":
    raise SystemExit(main())