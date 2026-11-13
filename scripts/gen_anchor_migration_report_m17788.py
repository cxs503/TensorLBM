#!/usr/bin/env python
"""Generate the m17788 anchor-migration report figures + markdown from the
JSON artifacts (zero hand-copying: every number is read from the artifact that
computed it).

Usage::

    PYTHONPATH=src python scripts/gen_anchor_migration_report_m17788.py \
        --n65 artifacts/anchor_migration_m17788/anchor_n65.json \
        --n129 artifacts/anchor_migration_m17788/anchor_n129.json \
        --outdir artifacts/anchor_migration_m17788
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(path: str | None) -> dict[str, Any] | None:
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def fig_mse_curves(n65: dict | None, n129: dict | None, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=140)
    for payload, label, color in ((n65, "n=65 (H=64)", "tab:blue"),
                                  (n129, "n=129 (H=128)", "tab:red")):
        if payload is None:
            continue
        curve = payload["sweep"]["curve"]
        re = [c["re"] for c in curve]
        rmse = [c["rmse_vs_ghia"] for c in curve]
        ax.plot(re, rmse, "o-", color=color, label=label, lw=1.6, ms=4)
        go = payload["sweep"]["grid_optimum"]
        if go:
            ax.axvline(go["re_at_min"], color=color, ls=":", lw=1.0, alpha=0.7)
            ax.annotate(f"min @ Re={go['re_at_min']:.0f}",
                        xy=(go["re_at_min"], go["rmse_vs_ghia_at_min"]),
                        xytext=(8, 8), textcoords="offset points", fontsize=8, color=color)
    ax.axvline(100.0, color="k", ls="--", lw=1.0)
    ax.annotate("true Re=100", xy=(100.0, ax.get_ylim()[1] * 0.9),
                xytext=(6, 0), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Re (sweep parameter)")
    ax.set_ylabel("RMSE vs Ghia (1982) centreline")
    ax.set_title("External-anchor objective: Ghia-MSE(Re) -- where does it minimise?")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def fig_re_error_ladder(n65: dict | None, n129: dict | None, out_png: Path) -> None:
    # m17147 n=33 external-anchor result (from docs/AUDIT_m17147...md, Re err 52.6%)
    rows = [("n=33\n(m17147)", 52.6, "tab:gray")]
    for payload, label, color in ((n65, "n=65", "tab:blue"), (n129, "n=129", "tab:red")):
        if payload is None:
            continue
        s = payload["summary"]
        rows.append((label, 100.0 * s["rel_err_re_recovered"], color))
    labels = [r[0] for r in rows]
    errs = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    bars = ax.bar(labels, errs, color=colors, width=0.55)
    ax.axhline(15.0, color="k", ls="--", lw=1.0)
    ax.annotate("15 % acceptance", xy=(len(rows) - 0.4, 15.0), fontsize=8)
    for b, e in zip(bars, errs):
        ax.annotate(f"{e:.1f}%", xy=(b.get_x() + b.get_width() / 2, e),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_ylabel("Re recovery error (%)")
    ax.set_title("External-anchor (Ghia) tau-inversion: Re error vs resolution")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n65", type=str, default=None)
    p.add_argument("--n129", type=str, default=None)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    n65 = _load(args.n65)
    n129 = _load(args.n129)
    fig_mse_curves(n65, n129, args.outdir / "ghia_mse_vs_re.png")
    fig_re_error_ladder(n65, n129, args.outdir / "re_error_ladder.png")

    def _b64(name: str) -> str:
        return base64.b64encode((args.outdir / name).read_bytes()).decode()

    payload = {
        "figures": [
            {"name": "ghia_mse_vs_re.png", "caption": "Ghia-MSE(Re) 曲线",
             "png_base64": _b64("ghia_mse_vs_re.png")},
            {"name": "re_error_ladder.png", "caption": "Re 恢复误差阶梯",
             "png_base64": _b64("re_error_ladder.png")},
        ],
        "n65": n65, "n129": n129,
    }
    (args.outdir / "report_payload.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.outdir}/report_payload.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
