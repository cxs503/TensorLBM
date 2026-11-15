"""R3 (verification debt): does the fractional mask put the wall where the analytic
solution says it is?

Round 1/2 of this benchmark verified the binary mask blend (wall on a cell face).
The library then gained :func:`tensorlbm.step_hooks.partial_cell_layer`, the
Noble-Torczynski (1998) partially saturated cell, whose wall may sit *inside* a
cell.  The code and its unit tests were merged, and the sweep harness
:func:`run_xval.run_fractional_sweep` was written, but no verification number was
ever published.  This script produces exactly that number set, with the design
frozen in ``predeclared_r3_fractional.json`` beforehand.

It reuses the round-1/2 harness (``run_xval``) unchanged -- same geometry
convention, same analytic references, same stopping rule -- so the only new
variable is the expression of the sub-cell wall:

``naive_eps``  float solid fraction fed to the binary bounce layer (upstream's
               ``f*(1-m) + bounce*f*m`` transplanted verbatim)
``nt_pre``     Noble-Torczynski weight ``B(eps_s, tau)``, pre-collision populations
``nt_post``    the published NT operator: ``B`` weight, post-collision populations

Usage::

    python benchmarks/xval_per_step_hook_mask_blend/run_r3_fractional.py [--quick]

Outputs ``artifacts/xval_frac_mask_wall_position/``::

    result.json        every KPI + the predeclared verdicts
    table.csv          one row per arm
    profiles.npz       measured and analytic profiles for the figure set
    figures/*.png      four figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_xval as R  # noqa: E402

OUT = HERE.parents[1] / "artifacts" / "xval_frac_mask_wall_position"
PREDECL = json.loads((HERE / "predeclared_r3_fractional.json").read_text())

OWNER_PCT = PREDECL["predeclared_thresholds"]["owner_acceptance_pct"]
TARGET_PCT = PREDECL["predeclared_thresholds"]["internal_target_pct"]
TOL_DX = PREDECL["predeclared_thresholds"]["wall_position_tolerance_dx"]
COLLAPSE_REL = PREDECL["predeclared_thresholds"]["collapse_rel"]


# ------------------------------------------------------------------ verdicts
def _pick(rows, **kw):
    out = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    return out


def judge(rows: list[dict], limits: dict) -> dict:
    """P1..P5 exactly as predeclared (no threshold moves after the fact)."""
    couette = [r for r in rows if r["case"] == "couette" and r["tau"] == 0.8 and r["ny"] == 16]
    pois = [r for r in rows if r["case"] == "poiseuille" and r["ny"] == 16]
    v: dict = {}

    # P1: nt_post places the wall (offset <= 0.05 dx, L2 < 3%) for eps in {0.25,0.5,0.75}
    p1_rows = [r for r in couette + pois if r["arm"] == "nt_post" and r["eps_s"] < 1.0]
    v["P1_nt_post_places_the_wall"] = {
        "rows": [
            {
                "case": r["case"],
                "eps_s": r["eps_s"],
                "offset_err_dx": r["offset_err_dx"],
                "l2_pct": r["profile_l2_pct"],
                "pass": bool(abs(r["offset_err_dx"]) <= TOL_DX and r["profile_l2_pct"] < TARGET_PCT),
            }
            for r in p1_rows
        ],
        "max_abs_offset_dx": max((abs(r["offset_err_dx"]) for r in p1_rows), default=float("nan")),
        "max_l2_pct": max((r["profile_l2_pct"] for r in p1_rows), default=float("nan")),
    }
    v["P1_nt_post_places_the_wall"]["holds"] = all(
        row["pass"] for row in v["P1_nt_post_places_the_wall"]["rows"]
    )

    # P2: naive_eps bias >= 0.25 dx at eps=0.5 and monotone in (1 - eps)
    naive = sorted(
        [r for r in couette if r["arm"] == "naive_eps"], key=lambda r: r["eps_s"]
    )
    bias = [r["offset_err_dx"] for r in naive]
    mono = all(b <= a + 1e-9 for a, b in zip(bias, bias[1:]))  # bias grows as eps falls
    at_half = [r for r in naive if abs(r["eps_s"] - 0.5) < 1e-12]
    v["P2_naive_is_biased"] = {
        "offset_err_dx_by_eps": {str(r["eps_s"]): r["offset_err_dx"] for r in naive},
        "bias_at_eps_0.5_dx": at_half[0]["offset_err_dx"] if at_half else None,
        "monotone_in_1_minus_eps": bool(mono),
        "holds": bool(
            at_half and abs(at_half[0]["offset_err_dx"]) >= 0.25 and mono
        ),
    }

    # P3: nt_post offset is tau independent (<= 0.05 dx) at eps=0.5
    tau_rows = sorted(
        [r for r in rows if r["case"] == "couette" and abs(r["eps_s"] - 0.5) < 1e-12],
        key=lambda r: r["tau"],
    )
    nt_post_tau = [r for r in tau_rows if r["arm"] == "nt_post"]
    v["P3_tau_robustness"] = {
        "nt_post_offset_err_dx_by_tau": {f"{r['tau']:.2f}": r["offset_err_dx"] for r in nt_post_tau},
        "max_abs_offset_dx": max((abs(r["offset_err_dx"]) for r in nt_post_tau), default=float("nan")),
        "holds": bool(nt_post_tau and max(abs(r["offset_err_dx"]) for r in nt_post_tau) <= TOL_DX),
    }

    # P4: nt_pre is worse than nt_post at eps in {0.25,0.5,0.75}
    cmp_rows = []
    for eps in (0.25, 0.5, 0.75):
        pre = _pick(couette, arm="nt_pre", eps_s=eps)
        post = _pick(couette, arm="nt_post", eps_s=eps)
        if pre and post:
            cmp_rows.append(
                {
                    "eps_s": eps,
                    "offset_pre_dx": pre[0]["offset_err_dx"],
                    "offset_post_dx": post[0]["offset_err_dx"],
                    "pre_worse": bool(abs(pre[0]["offset_err_dx"]) > abs(post[0]["offset_err_dx"])),
                }
            )
    v["P4_reflect_choice_matters"] = {
        "rows": cmp_rows,
        "holds": bool(cmp_rows and all(r["pre_worse"] for r in cmp_rows)),
    }

    # P5: at eps = 1 the three modes collapse (profile L2 and peak error agree)
    one = [r for r in couette if abs(r["eps_s"] - 1.0) < 1e-12]
    collapse = {}
    if one:
        ref = next((r for r in one if r["arm"] == "naive_eps"), one[0])
        for r in one:
            denom_l2 = max(abs(ref["profile_l2_pct"]), 1e-300)
            denom_pk = max(abs(ref["max_err_pct"]), 1e-300)
            collapse[r["arm"]] = {
                "l2_rel": abs(r["profile_l2_pct"] - ref["profile_l2_pct"]) / denom_l2,
                "peak_rel": abs(r["max_err_pct"] - ref["max_err_pct"]) / denom_pk,
            }
    v["P5_eps_one_collapse"] = {
        "rel_differences": collapse,
        "holds": bool(
            collapse
            and all(
                d["l2_rel"] <= COLLAPSE_REL and d["peak_rel"] <= COLLAPSE_REL
                for d in collapse.values()
            )
        ),
    }
    v["P_limits"] = {
        "eps1_vs_bounce_max_rel": limits["eps1_vs_bounce_max_rel"],
        "eps1_bit_identical": limits["bit_identical_eps1"],
        "eps0_vs_identity_max_rel": limits["eps0_vs_identity_max_rel"],
        "eps0_bit_identical": limits["bit_identical_eps0"],
        "partial_blob_mass_rel": limits["partial_blob_mass_rel"],
        "weights": limits["weights"],
        "holds": bool(
            limits["bit_identical_eps1"]
            and limits["bit_identical_eps0"]
            and limits["partial_blob_mass_rel"] < 1e-12
        ),
    }
    v["owner_gate_15pct"] = {
        "all_arms_under_15pct": bool(all(r["profile_l2_pct"] < OWNER_PCT for r in rows)),
        "worst_arm": max(rows, key=lambda r: r["profile_l2_pct"])["arm"],
        "worst_l2_pct": max(r["profile_l2_pct"] for r in rows),
    }
    return v


# -------------------------------------------------------------------- tables
def write_csv(rows: list[dict]) -> None:
    cols = [
        "case", "ny", "nx", "tau", "nu", "eps_s", "arm", "steps",
        "y_bottom_geom", "d_eff_dx", "offset_err_dx", "offset_err_pct_H",
        "profile_l2_pct", "max_err_pct", "mass_drift_rel", "finite", "f_sha256",
    ]
    with (OUT / "table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def md_tables(rows: list[dict]) -> str:
    def block(title, subset, cols=("eps_s", "arm", "d_eff_dx", "offset_err_dx", "profile_l2_pct", "max_err_pct")):
        head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        body = ""
        for r in subset:
            cells = []
            for c in cols:
                v = r[c]
                if c == "eps_s":
                    cells.append(f"{v:.2f}")
                elif c == "arm":
                    cells.append(f"`{v}`")
                elif c in ("profile_l2_pct", "max_err_pct"):
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(f"{v:+.4f}")
            body += "| " + " | ".join(cells) + " |\n"
        return f"### {title}\n\n{head}{body}\n"

    out = ""
    for ny in sorted({r["ny"] for r in rows if r["case"] == "couette"}):
        sub = [r for r in rows if r["case"] == "couette" and r["ny"] == ny and r["tau"] == 0.8]
        out += block(f"Plane Couette, Ny={ny}, tau=0.80 (interface at y_b = 0.5 + eps_s)", sub)
    sub = [r for r in rows if r["case"] == "poiseuille"]
    out += block("Plane Poiseuille, Ny=16, tau=0.80", sub)
    sub = [
        r for r in rows
        if r["case"] == "couette" and abs(r["eps_s"] - 0.5) < 1e-12 and r["ny"] == 16 and r["tau"] != 0.8
    ]
    out += block(
        "Tau sweep, plane Couette, Ny=16, eps_s=0.50",
        sub,
        cols=("tau", "arm", "d_eff_dx", "offset_err_dx", "profile_l2_pct", "max_err_pct"),
    )
    return out


# ------------------------------------------------------------------- figures
def write_figures(rows: list[dict]) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    made = []

    couette = [r for r in rows if r["case"] == "couette" and r["ny"] == 16 and r["tau"] == 0.8]
    eps_axis = np.array([0.25, 0.5, 0.75, 1.0])

    # fig1: effective interface position vs solid fraction
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(eps_axis, 0.5 + eps_axis, "k--", lw=1.5, label="exact interface $y_b=0.5+\\epsilon_s$")
    style = {"naive_eps": ("o", "tab:red", "naive $\\epsilon_s$ weight (upstream transplant)"),
             "nt_pre": ("s", "tab:orange", "NT weight, pre-collision populations"),
             "nt_post": ("^", "tab:blue", "NT weight, post-collision populations")}
    for arm, (mk, col, lab) in style.items():
        sub = sorted([r for r in couette if r["arm"] == arm], key=lambda r: r["eps_s"])
        ax.plot([r["eps_s"] for r in sub], [r["d_eff_dx"] for r in sub], mk + "-", color=col, label=lab)
    ax.set_xlabel("solid fraction $\\epsilon_s$ of the partial row")
    ax.set_ylabel("measured interface position $d_eff$ [dx]")
    ax.set_title("Plane Couette, Ny=16, $\\tau$=0.80: where does the wall sit?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = figdir / "fig1_wall_position_vs_eps.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append({"name": p.name, "caption": "Measured no-slip interface position against the exact $y_b=0.5+\\epsilon_s$ line for the three ways of expressing a sub-cell wall."})

    # fig2: profile error vs solid fraction
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for arm, (mk, col, lab) in style.items():
        sub = sorted([r for r in couette if r["arm"] == arm], key=lambda r: r["eps_s"])
        ax.plot([r["eps_s"] for r in sub], [r["profile_l2_pct"] for r in sub], mk + "-", color=col, label=lab)
    ax.axhline(TARGET_PCT, color="0.4", ls=":", label="3 % internal target")
    ax.axhline(OWNER_PCT, color="k", ls=":", label="15 % owner gate")
    ax.set_xlabel("solid fraction $\\epsilon_s$ of the partial row")
    ax.set_ylabel("profile $L_2$ error vs analytic [%]")
    ax.set_title("Couette profile error: the bias of the raw-$\\epsilon_s$ weight")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = figdir / "fig2_profile_error_vs_eps.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append({"name": p.name, "caption": "Profile $L_2$ error against the sub-cell analytic solution; the raw solid fraction never reaches the 3 % target while the NT weights do."})

    # fig3: tau dependence of the interface position at eps=0.5
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sub_all = [r for r in rows if r["case"] == "couette" and abs(r["eps_s"] - 0.5) < 1e-12 and r["ny"] == 16]
    for arm, (mk, col, lab) in style.items():
        sub = sorted([r for r in sub_all if r["arm"] == arm], key=lambda r: r["tau"])
        ax.plot([r["tau"] for r in sub], [r["d_eff_dx"] for r in sub], mk + "-", color=col, label=lab)
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="exact $y_b$ = 1.0 dx")
    ax.set_xlabel("relaxation time $\\tau$")
    ax.set_ylabel("measured interface position $d_eff$ [dx]")
    ax.set_title("Is the sub-cell wall position $\\tau$-independent? ($\\epsilon_s$=0.50)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = figdir / "fig3_wall_position_vs_tau.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append({"name": p.name, "caption": "Interface position versus relaxation time at $\\epsilon_s$=0.50: the NT weight depends on $\\tau$, the wall position must not."})

    # fig4: profiles at eps=0.5
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for r in couette:
        if abs(r["eps_s"] - 0.5) > 1e-12:
            continue
        prof = np.asarray(r["_profile"], dtype=float)
        y = np.arange(2, r["ny"] - 1, dtype=float)
        ax.plot(prof, y, "o-", ms=3, label=f"`{r['arm']}` (L2 {r['profile_l2_pct']:.3f} %)")
        ax.plot(np.asarray(r["_exact"], dtype=float), y, "k--", lw=1.0)
    ax.plot([], [], "k--", label="analytic, $y_b$=1.0 dx")
    ax.set_xlabel("$u_x$ [lattice units]")
    ax.set_ylabel("y [dx]")
    ax.set_title("Couette profile at $\\epsilon_s$=0.50, $\\tau$=0.80")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = figdir / "fig4_profiles_eps_0p5.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append({"name": p.name, "caption": "Measured profiles against the analytic line whose wall sits inside the cell: the raw-$\\epsilon_s$ arm has the wrong slope, i.e. the wrong channel width."})
    return made



# ------------------------------------------------------------- partitioning
def enumerate_configs(quick: bool = False) -> list[dict]:
    """The full arm list of the predeclared design, as data (shardable)."""
    cfgs: list[dict] = []
    eps_couette = (0.5,) if quick else (0.25, 0.5, 0.75, 1.0)
    ny_couette = (16,) if quick else (16, 24)
    modes = ("naive_eps", "nt_pre") if quick else R.FRACTIONAL_MODES
    for ny in ny_couette:
        for eps in eps_couette:
            for mode in modes:
                cfgs.append({"case": "couette", "ny": ny, "nx": 8, "tau": 0.8, "eps_s": eps, "mode": mode})
    poi_modes = ("nt_pre",) if quick else R.FRACTIONAL_MODES
    for eps in eps_couette:
        for mode in poi_modes:
            cfgs.append({"case": "poiseuille", "ny": 16, "nx": 8, "tau": 0.8, "eps_s": eps, "mode": mode})
    if not quick:
        for tau in (0.6, 0.7, 0.9):
            for mode in R.FRACTIONAL_MODES:
                cfgs.append({"case": "couette", "ny": 16, "nx": 8, "tau": tau, "eps_s": 0.5, "mode": mode})
    return cfgs


def run_configs(cfgs: list[dict]) -> list[dict]:
    rows = []
    for c in cfgs:
        r = R.run_fractional_wall(c["case"], c["ny"], c["nx"], c["tau"], c["eps_s"], c["mode"])
        rows.append(r)
        print(
            f"[r3] {r['case']:10s} ny={r['ny']:2d} tau={r['tau']:.2f} eps={r['eps_s']:.2f} "
            f"{r['arm']:9s} d_eff={r['d_eff_dx']:.4f} off={r['offset_err_dx']:+.4f} "
            f"L2={r['profile_l2_pct']:.4f}%",
            flush=True,
        )
    return rows


def combine_parts(shard_dir: Path, expected: int) -> list[dict]:
    """Load every shard part, restore the profiles, and fail closed on gaps."""
    parts = sorted(shard_dir.glob("part_*.json"))
    if not parts:
        raise SystemExit(f"no shard parts under {shard_dir}")
    rows: list[dict] = []
    seen: set[tuple] = set()
    for p in parts:
        data = json.loads(p.read_text())
        for row, cfg in zip(data["rows"], data["configs"]):
            key = (cfg["case"], cfg["ny"], cfg["tau"], cfg["eps_s"], cfg["mode"])
            if key in seen:
                raise SystemExit(f"duplicate arm {key} across shards")
            seen.add(key)
            rows.append(row)
    if len(rows) != expected:
        raise SystemExit(f"shard coverage {len(rows)} != expected {expected} arms")
    # profiles travel next to the shard parts so the figures can be rebuilt
    data: dict = {}
    for npz_path in sorted(shard_dir.glob("profiles_*.npz")):
        with np.load(npz_path) as z:
            data.update({k: z[k] for k in z.files})
    missing = []
    for row in rows:
        stem = f"{row['case']}_ny{row['ny']}_tau{row['tau']:.2f}_eps{row['eps_s']:.2f}_{row['arm']}"
        if f"{stem}_profile" in data:
            row["_profile"] = data[f"{stem}_profile"]
            row["_exact"] = data[f"{stem}_exact"]
        else:
            missing.append(stem)
    if missing:
        raise SystemExit(f"missing profiles for {len(missing)} arms, e.g. {missing[:3]}")
    return rows


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="CI-sized subset (2 arms, eps=0.5)")
    ap.add_argument("--shard", type=int, default=0, help="shard index for a partitioned sweep")
    ap.add_argument("--shards", type=int, default=1, help="number of shards (1 = run everything)")
    ap.add_argument("--combine", action="store_true", help="merge shards/part_*.json and write the report artefacts")
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(R.THREADS)
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    configs = enumerate_configs(quick=args.quick)
    if args.combine:
        rows = combine_parts(OUT / "shards", expected=len(configs))
        limits = R.run_fractional_limits()
        verdicts = judge(rows, limits)
    else:
        idx = list(range(len(configs)))
        if args.shards > 1:
            idx = [i for i in idx if i % args.shards == args.shard]
        rows = run_configs([configs[i] for i in idx])
        part = OUT / "shards" / f"part_{args.shard}.json"
        part.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            OUT / "shards" / f"profiles_{args.shard}.npz",
            **{
                f"{r['case']}_ny{r['ny']}_tau{r['tau']:.2f}_eps{r['eps_s']:.2f}_{r['arm']}_{what}": np.asarray(r[key])
                for r in rows
                for what, key in (("profile", "_profile"), ("exact", "_exact"))
            },
        )
        part.write_text(
            json.dumps(
                {"shard": args.shard, "shards": args.shards,
                 "configs": [configs[i] for i in idx],
                 "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]},
                indent=1, default=str,
            )
        )
        print(f"[r3] shard {args.shard}/{args.shards}: {len(rows)} arms -> {part}")
        return 0

    np.savez_compressed(
        OUT / "profiles.npz",
        **{
            f"{r['case']}_ny{r['ny']}_tau{r['tau']:.2f}_eps{r['eps_s']:.2f}_{r['arm']}_profile": np.asarray(r["_profile"])
            for r in rows
        },
        **{
            f"{r['case']}_ny{r['ny']}_tau{r['tau']:.2f}_eps{r['eps_s']:.2f}_{r['arm']}_exact": np.asarray(r["_exact"])
            for r in rows
        },
    )
    write_csv(rows)
    figures = write_figures(rows)

    result = {
        "predeclared": PREDECL,
        "verdicts": verdicts,
        "limits": limits,
        "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        "figures": figures,
        "wall_s": time.perf_counter() - t0,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "threads": R.THREADS,
            "dtype": str(R.DT),
            "quick": args.quick,
        },
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=str))
    (OUT / "tables.md").write_text(md_tables(rows))

    print(json.dumps(verdicts, indent=1, default=str))
    print(f"[r3] {len(rows)} arms in {result['wall_s']:.1f} s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())