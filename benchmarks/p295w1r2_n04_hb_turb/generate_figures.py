#!/usr/bin/env python3
"""Figures for campaign node P295W1R2-N04 (all panels from artifacts only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from analyze import load_cases  # noqa: E402

from tensorlbm.hb_turbulent_channel3d import dodge_metzner_friction  # noqa: E402

FIG_DIR = REPO / "benchmarks" / "p295w1r2_n04_hb_turb" / "figures"


def wall_coords(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    cfg = rec["config"]
    h = cfg["half_height"]
    nu = cfg["u_tau"] * h / cfg["re_tau"]
    y = np.arange(cfg["ny"], dtype=float) - 0.5
    y = np.clip(y, 0.0, 2.0 * h)
    return y, y * cfg["u_tau"] / nu


def fig_laminar(cases: dict, path: Path) -> None:
    lam = {k: v for k, v in cases.items() if v.get("mode") == "laminar"}
    if not lam:
        return
    ns = sorted({v["config"]["n"] for v in lam.values()})
    fig, axes = plt.subplots(1, len(ns), figsize=(4.4 * len(ns), 4.2), squeeze=False)
    for ax, n in zip(axes[0], ns):
        for rec in sorted(
            (v for v in lam.values() if v["config"]["n"] == n),
            key=lambda r: r["derived"]["yield_number"],
        ):
            cfg = rec["config"]
            y = np.arange(cfg["ny"], dtype=float) - 0.5
            h = cfg["half_height"]
            u = np.asarray(rec["profiles"]["u_mean"], dtype=float)
            m = (y > 0) & (y < 2 * h)
            ax.plot(
                y[m] / h,
                u[m] / cfg["u_tau"],
                "o",
                ms=2.2,
                label=f"LBM Y={rec['derived']['yield_number']:.2g}",
            )
            # analytical profile sampled on the same wall-normal coordinates
            from tensorlbm.hb_channel_analytical import HBChannelParams, hb_velocity_profile

            p = HBChannelParams(
                tau_y=cfg["tau_y"], K=cfg["K"], n=cfg["n"], G=cfg["body_force"], h=h
            )
            ya, ua = hb_velocity_profile(p, n_points=1000)
            ax.plot(
                ya / h,
                ua / cfg["u_tau"],
                "-",
                lw=1.2,
                label=f"HB exact Y={rec['derived']['yield_number']:.2g}",
            )
        ax.set_xlabel("y / h")
        ax.set_ylabel(r"$u/u_\tau$")
        ax.set_title(f"n = {n:g}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("B1: laminar HB channel - LBM vs analytical solution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_friction(cases: dict, crit: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    turb = {k: v for k, v in cases.items() if k.startswith("turb")}
    ns = sorted({v["config"]["n"] for v in turb.values()}) or [1.0]
    markers = ["o", "s", "^", "D"]
    re_axis = np.logspace(2.5, 4.6, 200)
    ax.plot(re_axis, 24.0 / re_axis, "k--", lw=1.0, label="laminar 24/Re_MR")
    for n, mk in zip(ns, markers):
        f_dm = [dodge_metzner_friction(float(r), n) for r in re_axis]
        ax.plot(re_axis, f_dm, "-", lw=1.0, alpha=0.8, label=f"Dodge-Metzner n={n:g}")
    for n, mk in zip(ns, markers):
        pts = sorted(
            (v["derived"]["re_mr"], v["derived"]["f_darcy"], v["derived"]["yield_number"])
            for v in turb.values()
            if v["config"]["n"] == n
        )
        if not pts:
            continue
        ax.plot(
            [p[0] for p in pts], [p[1] for p in pts], mk, ms=5, mfc="none", label=f"LBM n={n:g}"
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$Re_{MR}$")
    ax.set_ylabel(r"Darcy friction factor $f$")
    ax.set_title("B3: friction-factor collapse in generalised coordinates")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    b3 = crit.get("B3_friction_collapse", {})
    txt = f"max spread = {b3.get('max_spread', float('nan')) * 100:.1f} %"
    ax.text(0.03, 0.05, txt, transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_yield(cases: dict, crit: dict, path: Path) -> None:
    b4 = crit.get("B4_yield_modulation", {})
    rows = b4.get("rows", [])
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for row in rows:
        Y = [p["Y"] for p in row["series"]]
        ax = axes[0]
        ax.plot(
            Y,
            [p["tke_ratio"] for p in row["series"]],
            "o-",
            label=f"Re_tau={row['re_tau']:g}, n={row['n']:g}",
        )
        axes[1].plot(
            Y,
            [p["u_bulk_ratio"] for p in row["series"]],
            "o-",
            label=f"Re_tau={row['re_tau']:g}, n={row['n']:g}",
        )
    for ax in axes:
        ax.axhline(0.1, color="grey", ls=":", lw=1.0)
        ax.set_xlabel(r"yield number $Y=\tau_y/(\rho u_\tau^2)$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel(r"$TKE(Y)/TKE(0)$")
    axes[0].set_title("TKE modulation by yield stress")
    axes[1].set_ylabel(r"$U_b(Y)/U_b(0)$")
    axes[1].set_title("bulk-velocity response (shut-off at Y->1)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_profiles(cases: dict, path: Path) -> None:
    turb = {k: v for k, v in cases.items() if k.startswith("turb")}
    if not turb:
        return
    groups: dict[tuple[float, float], list[dict]] = {}
    for v in turb.values():
        groups.setdefault((v["config"]["re_tau"], v["config"]["n"]), []).append(v)
    key = max(groups, key=lambda k: len(groups[k]))
    recs = sorted(groups[key], key=lambda r: r["derived"]["yield_number"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for rec in recs:
        _, yp = wall_coords(rec)
        cfg = rec["config"]
        h = cfg["half_height"]
        y = np.arange(cfg["ny"], dtype=float) - 0.5
        m = (y > 0) & (y < 2 * h)
        u = np.asarray(rec["profiles"]["u_mean"], dtype=float)
        tke = np.asarray(rec["profiles"]["tke"], dtype=float)
        uv = np.asarray(rec["profiles"]["uv_prime"], dtype=float)
        axes[0].semilogx(
            yp[m], u[m] / cfg["u_tau"], "-", label=f"Y={rec['derived']['yield_number']:.2f}"
        )
        axes[1].semilogx(
            yp[m], -uv[m] / cfg["u_tau"] ** 2, "-", label=f"Y={rec['derived']['yield_number']:.2f}"
        )
    axes[0].set_xlabel(r"$y^+$")
    axes[0].set_ylabel(r"$u^+$")
    axes[0].set_title(f"mean velocity, Re_tau={key[0]:g}, n={key[1]:g}")
    axes[1].set_xlabel(r"$y^+$")
    axes[1].set_ylabel(r"$-\overline{u'v'}^+$")
    axes[1].set_title("Reynolds shear stress")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_tke_profiles(cases: dict, path: Path) -> None:
    turb = {k: v for k, v in cases.items() if k.startswith("turb")}
    if not turb:
        return
    groups: dict[tuple[float, float], list[dict]] = {}
    for v in turb.values():
        groups.setdefault((v["config"]["re_tau"], v["config"]["n"]), []).append(v)
    key = max(groups, key=lambda k: len(groups[k]))
    recs = sorted(groups[key], key=lambda r: r["derived"]["yield_number"])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    base = next((r for r in recs if r["derived"]["yield_number"] < 1e-12), None)
    for rec in recs:
        _, yp = wall_coords(rec)
        cfg = rec["config"]
        h = cfg["half_height"]
        y = np.arange(cfg["ny"], dtype=float) - 0.5
        m = (y > 0) & (y < h)
        tke = np.asarray(rec["profiles"]["tke"], dtype=float)
        if base is not None:
            ref = np.asarray(base["profiles"]["tke"], dtype=float)
            tke = np.divide(tke, ref, out=np.zeros_like(tke), where=ref > 1e-12)
            ylabel = r"$TKE(Y)/TKE(0)$"
        else:
            ylabel = r"$TKE/u_\tau^2$"
        ax.semilogx(yp[m], tke[m], "-", label=f"Y={rec['derived']['yield_number']:.2f}")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"TKE modulation (Re_tau={key[0]:g}, n={key[1]:g})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_gci(cases: dict, crit: dict, path: Path) -> None:
    b5 = crit.get("B5_lattice_gci", {})
    g = b5.get("gci")
    if not g:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ny = g["ny"]
    f = g["f"]
    ax.plot(1.0 / np.array(ny, dtype=float), f, "o-", label="D3Q19 (3 grids)")
    ax.axhline(
        g["richardson_extrapolated"], color="k", ls="--", lw=1.0, label="Richardson extrapolation"
    )
    pair = b5.get("lattice_pair")
    if pair:
        ax.axhline(pair["f_d3q27"], color="tab:red", ls=":", lw=1.2, label="D3Q27 (finest)")
    ax.set_xlabel("1 / n_y")
    ax.set_ylabel("Darcy friction factor f")
    ax.set_title(f"B5: grid convergence (GCI={g['gci_fine_pct']:.2f} %)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", default="artifacts/p295w1r2_n04")
    ap.add_argument("--out-dir", default=str(FIG_DIR))
    args = ap.parse_args()
    art_dir = Path(args.art_dir)
    if not art_dir.is_absolute():
        art_dir = REPO / art_dir
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)

    cases = load_cases(art_dir)
    crit_path = art_dir / "criteria.json"
    crit = json.loads(crit_path.read_text()) if crit_path.exists() else {}

    made = []
    for name, fn in (
        ("fig1_laminar_profiles.png", lambda p: fig_laminar(cases, p)),
        ("fig2_friction_collapse.png", lambda p: fig_friction(cases, crit, p)),
        ("fig3_yield_modulation.png", lambda p: fig_yield(cases, crit, p)),
        ("fig4_velocity_reynolds.png", lambda p: fig_profiles(cases, p)),
        ("fig5_tke_profiles.png", lambda p: fig_tke_profiles(cases, p)),
        ("fig6_grid_gci.png", lambda p: fig_gci(cases, crit, p)),
    ):
        p = out / name
        fn(p)
        if p.exists():
            made.append(str(p.relative_to(REPO)) if str(p).startswith(str(REPO)) else str(p))
            print("wrote", p)
    (out / "figures.json").write_text(json.dumps(made, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
