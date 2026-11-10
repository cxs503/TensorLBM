#!/usr/bin/env python3
"""P223W1-N01 figures: mean profile, Reynolds stresses, spectra, friction law.

Reads only ``gates.json`` plus the per-case native artefacts, so every plotted
curve is the same number that appears in the report table (no hand-typed data).

    python3 benchmarks/p223w1_n01_channel_dns/plot.py \
        --run-dir runs/p223w1_n01_channel_dns --out-dir figures/p223w1_n01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tensorlbm.turbulence.channel_dns import load_dns_case  # noqa: E402
from tensorlbm.validation import mkm1999  # noqa: E402
from tensorlbm.validation.wall_law import dean_channel_cf  # noqa: E402

_PALETTE = ("#1f77b4", "#8c564b", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e")


def _color(name: str) -> str:
    return _PALETTE[hash(name) % len(_PALETTE)]


def _cases(run_dir: Path) -> dict[str, int]:
    """Case name -> MKM anchor, taken from the gate payload (single source)."""
    path = run_dir / "gates.json"
    if not path.exists():
        raise SystemExit("run gates.py first: gates.json is the case registry")
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors = {180: 180, 395: 395, 590: 590}
    out = {}
    for name, row in payload["metrics"].items():
        if row.get("re_tau") is None:
            continue
        nominal = min(anchors, key=lambda a: abs(a - float(row["re_tau"])))
        out[name] = anchors[nominal]
    return out


def _load(run_dir: Path) -> dict[str, dict]:
    out = {}
    for name, anchor in _cases(run_dir).items():
        case = load_dns_case(run_dir, name, re_tau_anchor=anchor)
        if case and not case.get("diverged"):
            out[name] = case
    return out


def fig_mean_profile(cases: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    for name, case in cases.items():
        prof = case["profile"]
        y_plus = np.asarray(prof["y_plus"])
        u_plus = np.asarray(prof["u_plus"])
        ax.semilogx(y_plus, u_plus, label=f"{name} ({case['resolution']['grade']})",
                    color=_color(name), lw=1.4)
    mkm = mkm1999.load_profile(180)
    ax.semilogx(mkm.y_plus, mkm.u_plus, "k--", lw=1.0, label="MKM-1999 Re_tau=178")
    mkm = mkm1999.load_profile(590)
    ax.semilogx(mkm.y_plus, mkm.u_plus, "k:", lw=1.0, label="MKM-1999 Re_tau=590")
    y = np.array([1.0, 10.0])
    ax.semilogx(y, y, "0.6", lw=0.8)
    yl = np.logspace(np.log10(30), np.log10(200), 40)
    ax.semilogx(yl, np.log(yl) / 0.41 + 5.2, "0.4", lw=0.8, ls="-.",
                label="$\\ln(y^+)/0.41+5.2$")
    ax.set_xlabel("$y^+$")
    ax.set_ylabel("$U^+$")
    ax.set_title("mean velocity profile (wall units)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    ax = axes[1]
    for name, case in cases.items():
        prof = case["profile"]
        y_plus = np.asarray(prof["y_plus"])
        u_plus = np.asarray(prof["u_plus"])
        dns = mkm1999.load_profile(case["anchor"]).u_plus_at(y_plus)
        ax.semilogx(y_plus, 100.0 * (u_plus - dns) / dns, color=_color(name),
                    lw=1.2, label=name)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(3.0, color="r", lw=0.8, ls="--")
    ax.axhline(-3.0, color="r", lw=0.8, ls="--", label="±3 %")
    ax.set_xlabel("$y^+$")
    ax.set_ylabel("$U^+$ error vs MKM [%]")
    ax.set_title("deviation from the DNS mean profile")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_reynolds_stress(cases: dict, path: Path) -> None:
    keys = ("u_rms_plus", "v_rms_plus", "w_rms_plus")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True)
    for ax, key in zip(axes[:3], keys):
        for name, case in cases.items():
            prof = case["profile"]
            ax.plot(np.asarray(prof["y_plus"]), np.asarray(prof[key]),
                    color=_color(name), lw=1.3, label=name)
        for name in cases:
            m = mkm1999.load_profile(case["anchor"])
            ax.plot(m.y_plus, getattr(m, key), "k--", lw=0.8)
        ax.set_title(key.replace("_plus", "$^+$"))
        ax.set_xlabel("$y^+$")
        ax.grid(alpha=0.3)
    for name, case in cases.items():
        axes[3].plot(np.asarray(case["profile"]["y_plus"]),
                     -np.asarray(case["profile"]["uv_plus"]),
                     color=_color(name), lw=1.3, label=name)
    for name, case in cases.items():
        m = mkm1999.load_profile(case["anchor"])
        axes[3].plot(m.y_plus, -m.uv_plus, "k--", lw=0.8)
    axes[3].set_title("$-\\langle u'v'\\rangle^+$")
    axes[3].set_xlabel("$y^+$")
    axes[3].grid(alpha=0.3)
    axes[0].set_ylabel("stress [$u_\\tau^2$]")
    axes[0].legend(fontsize=7)
    fig.suptitle("Reynolds stresses vs MKM-1999 DNS (dashed = DNS)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_spectra(cases: dict, path: Path) -> None:
    have = {n: c for n, c in cases.items() if c.get("spectra")}
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(4.2 * len(have), 3.9), squeeze=False)
    for ax, (name, case) in zip(axes[0], have.items()):
        for leg in case["spectra"]["streamwise"]:
            k = np.asarray(leg["k"], dtype=float)
            e = np.asarray(leg["E"], dtype=float)
            ax.loglog(k, e, lw=1.1, label=f"$y^+$={leg['y_plus']:.0f}")
            fit = leg["fit"]
            if fit.get("valid"):
                kk = np.array([fit["k_min"], fit["k_max"]])
                ax.loglog(kk, np.exp(fit["intercept"]) * kk ** fit["slope"], "k--",
                          lw=0.8)
        k = np.logspace(np.log10(4), np.log10(30), 10)
        ax.loglog(k, 0.02 * k ** (-5.0 / 3.0), "0.5", ls=":", lw=1.0,
                  label="$k^{-5/3}$ guide")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("$k_x$ (mode)")
        ax.set_ylabel("$E_{uu}(k_x)/u_\\tau^2$")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6)
    fig.suptitle("streamwise energy spectra on log-layer planes", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_friction(cases: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    re_b = np.logspace(np.log10(3e3), np.log10(1.2e5), 100)
    ax.loglog(re_b, [dean_channel_cf(r) for r in re_b], "k-", lw=1.2,
              label="Dean (1978) $C_f=0.073 Re_b^{-1/4}$")
    for name, case in cases.items():
        cf = case["cf"]
        ax.loglog([cf["re_bulk"]], [cf["cf"]], "o", color=_color(name), ms=6,
                  label=f"{name} ($\\Delta y^+$={case['resolution']['dy_plus']:.1f})")
        if "cf_mkm" in cf:
            ax.loglog([cf["re_bulk_mkm"]], [cf["cf_mkm"]], "x", color=_color(name),
                      ms=8, mew=1.6)
    ax.set_xlabel("$Re_b = U_b\\,2h/\\nu$")
    ax.set_ylabel("$C_f$")
    ax.set_title("friction coefficient: sim (o) vs MKM DNS (x) vs Dean")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_resolution(run_dir: Path, path: Path) -> None:
    """Cells a dy+ = 1 DNS box needs vs the grids this node actually ran."""
    from tensorlbm.turbulence.channel_dns import dns_cells_required

    payload = json.loads((run_dir / "gates.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    re_values = sorted(
        {float(m["re_tau"]) for m in metrics.values() if m.get("re_tau") is not None}
    )
    if re_values:
        proj = [dns_cells_required(r, dy_plus=1.0)["cells"] for r in re_values]
        ax.loglog(re_values, proj, "ks-",
                  label="DNS grid ($\\Delta y^+=1$, $2\\pi h\\times2h\\times\\pi h$)")
    for name, m in metrics.items():
        if m.get("grid") is None or m.get("re_tau") is None:
            continue
        cells = m["grid"][0] * m["grid"][1] * m["grid"][2]
        mark = "o" if m["status"] != "DIVERGED" else "x"
        ax.loglog([m["re_tau"]], [cells], mark, color=_color(name), ms=7,
                  label=f"{name} ($\\Delta y^+$={m['dy_plus']:.1f}, {m['status'].lower()})")
    ax.set_xlabel("$Re_\\tau$")
    ax.set_ylabel("grid cells")
    ax.set_title("what a DNS box asks for vs the grids actually affordable")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=str(_REPO / "runs" / "p223w1_n01_channel_dns"))
    ap.add_argument("--out-dir", default=str(_REPO / "figures" / "p223w1_n01"))
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = _load(run_dir)
    if not cases:
        raise SystemExit("no usable case artefacts found")
    fig_mean_profile(cases, out_dir / "mean_profile.png")
    fig_reynolds_stress(cases, out_dir / "reynolds_stress.png")
    fig_spectra(cases, out_dir / "spectra.png")
    fig_friction(cases, out_dir / "friction.png")
    fig_resolution(run_dir, out_dir / "resolution_budget.png")
    print(f"cases={list(cases)} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())