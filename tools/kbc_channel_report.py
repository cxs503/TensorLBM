#!/usr/bin/env python3
"""Production driver / reporter for the KBC coarse-grid channel campaign.

Two subcommands, deliberately split so that the *expensive* half needs nothing
but NumPy (the cluster nodes are lean loong64 boxes) and the *pretty* half
(matplotlib figures, gate tables) runs anywhere afterwards on the saved CSVs.

``run``
    One case: call the OpenMP kernel, evaluate the gates against MKM-1999 DNS,
    write ``profiles.csv`` + ``gates.json`` + ``meta.json`` (all small: a few kB,
    no binary blobs -- a previous mission of this node died on a full disk).

``figures``
    Read every ``*/gates.json`` under ``--root`` and emit the report figures.

Example
-------
    PYTHONPATH=src python tools/kbc_channel_report.py run \
        --label L180_n33 --re-tau 180 --ny 33 --nx 96 --nz 48 \
        --steps 36000 --warmup 12000 --out artifacts/kbc/p495w4-n03
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.kbc_channel import evaluate_gates, wall_units  # noqa: E402
from tensorlbm.kbc_channel_native import native_channel_run  # noqa: E402

FIELDS = ("y", "y_plus", "u_plus", "uu_plus", "vv_plus", "ww_plus", "uv_plus", "stress", "y_over_h")


def _grid(re_tau: float, ny: int) -> tuple[int, int, float]:
    """Standard channel box: ``Lx = 2 pi h``, ``Lz = pi h`` (rounded to ints)."""
    half = (ny - 2) / 2.0
    return int(round(2.0 * math.pi * half)), int(round(math.pi * half)), half


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=True)
    nx = args.nx or _grid(args.re_tau, args.ny)[0]
    nz = args.nz or _grid(args.re_tau, args.ny)[1]
    half = (args.ny - 2) / 2.0
    force = args.u_tau**2 / half
    nu = args.u_tau * half / args.re_tau
    t0 = time.time()
    res = native_channel_run(
        nx=nx, ny=args.ny, nz=nz, steps=args.steps, warmup=args.warmup,
        sample_every=args.sample_every, n_iter=args.n_iter, re_tau=args.re_tau,
        u_tau=args.u_tau, alpha_cap=args.alpha_cap, alpha_min=args.alpha_min,
        omega_g=args.omega_g,
        tau_bulk=args.tau_bulk, trigger=args.trigger,
        enforce_entropy=bool(args.enforce_entropy), noise_amp=args.noise_amp,
        seed=args.seed, log_every=args.log_every,
    )
    wall_clock = time.time() - t0
    wu = wall_units(res.profiles, nu=nu, force=force, half_height=half, re_tau=args.re_tau)
    gates = evaluate_gates(
        res.profiles, nu=nu, force=force, half_height=half, re_tau=args.re_tau,
        nominal_re_tau=args.nominal_re_tau or args.re_tau, scalars=res.scalars,
    )
    head = ",".join(FIELDS)
    table = np.column_stack([getattr(wu, f) for f in FIELDS])
    np.savetxt(out / "profiles.csv", table, delimiter=",", header=head, comments="")
    np.savetxt(
        out / "history.csv", res.history, delimiter=",",
        header="step,max_abs_u,mean_rho,min_pop,alpha_mean,alpha_min,alpha_max,"
               "gamma_mean,gamma_limited_fraction,ghost_fraction,dh_global,dh_max_cell",
        comments="",
    )
    meta = dict(
        label=args.label, nx=nx, ny=args.ny, nz=nz, half_height=half,
        re_tau=args.re_tau, u_tau=args.u_tau, nu=nu, tau=0.5 + 3.0 * nu,
        force=force, delta_plus=args.re_tau / half, steps=args.steps,
        warmup=args.warmup, sample_every=args.sample_every, n_iter=args.n_iter,
        alpha_cap=args.alpha_cap, alpha_min=args.alpha_min, omega_g=args.omega_g,
        tau_bulk=args.tau_bulk, trigger=args.trigger,
        enforce_entropy=bool(args.enforce_entropy), noise_amp=args.noise_amp,
        seed=args.seed, wall_clock_s=wall_clock,
        mcell_per_s=nx * args.ny * nz * args.steps / wall_clock / 1e6,
        cells=nx * args.ny * nz,
    )
    # kernel diagnostics land under their own names; update() keeps the
    # configuration keys from clashing with the KBC_SCALARS names.
    meta.update({k: float(v) for k, v in res.scalars.items()})
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    (out / "gates.json").write_text(json.dumps(gates, indent=1, default=float))
    print(json.dumps({k: v for k, v in meta.items() if k in
                      ("label", "re_tau", "ny", "delta_plus", "steps", "wall_clock_s",
                       "mcell_per_s", "alpha_mean", "gamma_limited_fraction",
                       "newton_iterations", "min_population", "max_abs_u")}, indent=1, default=float))
    print(json.dumps({"cf": gates["cf"], "uplus": gates["uplus"],
                      "reynolds_stress": gates["reynolds_stress"],
                      "wall": gates["wall"]}, indent=1, default=float))
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tensorlbm.validation import mkm1999

    root = Path(args.root)
    cases = sorted(p for p in root.iterdir() if (p / "profiles.csv").exists())
    if not cases:
        print(f"no cases under {root}", file=sys.stderr)
        return 1
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. u+ versus y+ ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for case in cases:
        d = np.genfromtxt(case / "profiles.csv", delimiter=",", names=True)
        meta = json.loads((case / "meta.json").read_text())
        axes[0].semilogx(d["y_plus"], d["u_plus"], "-", lw=1.3,
                         label=f"{case.name} (Ny={meta['ny']})")
        axes[1].semilogx(d["y_plus"], -d["uv_plus"], "--", lw=1.1, label=case.name)
    for re_tau in (180, 395, 590):
        m = mkm1999.load_profile(re_tau)
        axes[0].semilogx(m.y_plus, m.u_plus, "k.", ms=2.0, alpha=0.6,
                         label=f"MKM {re_tau} DNS" if re_tau == 180 else None)
        axes[1].semilogx(m.y_plus, -m.uv_plus, "k.", ms=2.0, alpha=0.5,
                         label=f"MKM {re_tau} DNS" if re_tau == 180 else None)
    for ax in axes:
        ax.set_xlabel(r"$y^+$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$u^+$")
    axes[0].set_title("mean profile")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].set_ylabel(r"$-\langle u'v'\rangle^+$")
    axes[1].set_title("Reynolds shear stress")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(fig_dir / "kbc_uplus.png", dpi=150)
    plt.close(fig)

    # ---- 2. Cf error bar chart ------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels, errs, ok = [], [], []
    for case in cases:
        g = json.loads((case / "gates.json").read_text())
        labels.append(case.name)
        errs.append(100.0 * g["cf"]["rel_err"])
        ok.append(g["cf"]["pass"])
    x = np.arange(len(labels))
    ax.bar(x, errs, color=["#2a9d4a" if o else "#c1432e" for o in ok])
    ax.axhline(3.0, color="k", ls="--", lw=1.0, label="3% gate")
    ax.axhline(5.0, color="0.5", ls=":", lw=1.0, label="5% gate (Re_tau=180)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel(r"$|C_f - C_{f,DNS}|/C_{f,DNS}$ [%]")
    ax.set_title("friction coefficient error vs MKM-1999")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "kbc_cf.png", dpi=150)
    plt.close(fig)

    # ---- 3. total stress / momentum balance ------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for case in cases:
        d = np.genfromtxt(case / "profiles.csv", delimiter=",", names=True)
        ax.plot(d["y_over_h"], d["stress"], "-", lw=1.2, label=case.name)
    ax.set_xlabel(r"$y/h$")
    ax.set_ylabel(r"$\langle \Pi_{xy}\rangle / u_\tau^2$")
    ax.set_title("total stress carried by the populations (must be linear)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "kbc_stress.png", dpi=150)
    plt.close(fig)
    print(f"figures -> {fig_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one channel case")
    r.add_argument("--label", required=True)
    r.add_argument("--re-tau", type=float, required=True)
    r.add_argument("--nominal-re-tau", type=int, default=0)
    r.add_argument("--ny", type=int, required=True)
    r.add_argument("--nx", type=int, default=0)
    r.add_argument("--nz", type=int, default=0)
    r.add_argument("--steps", type=int, required=True)
    r.add_argument("--warmup", type=int, required=True)
    r.add_argument("--sample-every", type=int, default=20)
    r.add_argument("--n-iter", type=int, default=6)
    r.add_argument("--u-tau", type=float, default=0.01)
    r.add_argument("--alpha-cap", type=float, default=2.0)
    r.add_argument("--alpha-min", type=float, default=1.0)
    r.add_argument("--omega-g", type=float, default=1.0)
    r.add_argument("--tau-bulk", type=float, default=-1.0)
    r.add_argument("--trigger", type=float, default=1e-9)
    r.add_argument("--enforce-entropy", type=int, default=0)
    r.add_argument("--noise-amp", type=float, default=0.1)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--log-every", type=int, default=2000)
    r.add_argument("--out", default="artifacts/kbc/p495w4-n03")
    r.set_defaults(func=cmd_run)

    f = sub.add_parser("figures", help="build report figures from saved cases")
    f.add_argument("--root", default="artifacts/kbc/p495w4-n03")
    f.add_argument("--fig-dir", default="figures/kbc_p495w4_n03")
    f.set_defaults(func=cmd_figures)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())