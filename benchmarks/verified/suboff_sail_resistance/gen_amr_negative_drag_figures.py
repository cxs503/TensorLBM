#!/usr/bin/env python3
"""Figures for the AMR negative-drag reproduction artifact (paper-exp-lbm-suboff-tau-4).

Reads the machine-written artifact ``amr_negative_drag_repro.json`` and the
per-arm anchor samples, and writes three PNGs under
``runs/amr_negative_drag_repro/figures/``:

1. ``dqw_vs_tau.png``      D_QW vs tau: historical record vs replay protocol vs
                           Re-matched arms (the attribution plot).
2. ``drag_vs_re.png``      background-subtracted drag impulse vs Reynolds number
                           for every arm, sign-coded.
3. ``anchor_series.png``   D_QW time series per arm with the sampled window
                           shaded (drift disclosure).

No number is hard-coded here: everything is read back out of the artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARTIFACT = HERE / "amr_negative_drag_repro.json"
FIG_DIR = REPO / "runs" / "amr_negative_drag_repro" / "figures"


def main() -> int:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = payload["results"]
    hist = payload["historical_record"]["ladder"]
    ev = payload["evaluation"]

    # ---------------------------------------------------------------- figure 1
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(0.0, color="0.4", lw=1.0, ls="--", label="sign boundary (D_QW = 0)")
    ax.plot(
        [h["tau"] for h in hist],
        [h["D_QW"] for h in hist],
        "kx-",
        ms=9,
        lw=1.4,
        label="historical record (tau_axis_20261015, #278)",
    )
    for protocol, style in (("replay", ("o", "C0")), ("matched", ("s", "C3"))):
        xs, ys, labels = [], [], []
        for arm in payload["arm_design"]["arms"]:
            if arm["protocol"] != protocol:
                continue
            res = results.get(arm["arm"])
            if not res or not res.get("window", {}).get("mean"):
                continue
            xs.append(arm["tau_target"])
            ys.append(res["window"]["mean"]["D_QW"])
            labels.append(arm["arm"])
        if xs:
            ax.plot(
                xs,
                ys,
                style[0],
                color=style[1],
                ms=7,
                ls="none",
                label=f"{protocol} arms (this run)",
            )
            for x, y, lab in zip(xs, ys, labels):
                ax.annotate(lab.replace("_", " "), (x, y), textcoords="offset points", xytext=(4, 6), fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel(r"relaxation parameter $\tau$ (log scale)")
    ax.set_ylabel(r"$D_{QW} = \rho_{fwd} - \rho_{aft}$")
    ax.set_title("D_QW vs tau: historical record vs Re-matched re-run")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dqw_vs_tau.png", dpi=160)
    plt.close(fig)

    # ---------------------------------------------------------------- figure 2
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(0.0, color="0.4", lw=1.0, ls="--")
    for arm in payload["arm_design"]["arms"]:
        res = results.get(arm["arm"])
        if not res or not res.get("window", {}).get("mean"):
            continue
        m = res["window"]["mean"]
        marker = "o" if arm["protocol"] == "replay" else "s"
        color = "C0" if m["mem_bgsub_fx"] > 0 else "C3"
        ax.plot(res["reynolds_coarse"], m["mem_bgsub_fx"], marker, color=color, ms=8)
        ax.annotate(
            f"{arm['arm']}\nCt={m['ct_bgsub']:.3g}",
            (res["reynolds_coarse"], m["mem_bgsub_fx"]),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=6,
        )
    ax.axvspan(
        min(
            [results[a["arm"]]["reynolds_coarse"] for a in payload["arm_design"]["arms"] if a["protocol"] == "matched" and a["arm"] in results]
            or [1.0]
        )
        * 0.95,
        max(
            [results[a["arm"]]["reynolds_coarse"] for a in payload["arm_design"]["arms"] if a["protocol"] == "matched" and a["arm"] in results]
            or [2.0]
        )
        * 1.05,
        color="C3",
        alpha=0.08,
        label="Re-matched band",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Reynolds number Re = u L / nu (coarse level)")
    ax.set_ylabel("background-subtracted drag impulse  mem_bgsub_fx")
    ax.set_title("Drag sign vs Reynolds number (all frozen arms)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "drag_vs_re.png", dpi=160)
    plt.close(fig)

    # ---------------------------------------------------------------- figure 3
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for arm in payload["arm_design"]["arms"]:
        res = results.get(arm["arm"])
        if not res:
            continue
        path = REPO / res["samples_path"]
        if not path.exists():
            continue
        steps, dqw = [], []
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            steps.append(row["step"])
            dqw.append(row["D_QW"])
        if not steps:
            continue
        ax.plot(steps, dqw, lw=1.0, label=arm["arm"])
        ax.axvspan(steps[-1] * 0.75, steps[-1], color="0.6", alpha=0.12)
    ax.axhline(0.0, color="0.4", lw=1.0, ls="--")
    ax.set_xlabel("lattice step")
    ax.set_ylabel(r"$D_{QW}$")
    ax.set_title("D_QW time series (shaded: sampled window = last 25%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "anchor_series.png", dpi=160)
    plt.close(fig)

    print(
        json.dumps(
            {
                "figures": [str(p.relative_to(REPO)) for p in sorted(FIG_DIR.glob("*.png"))],
                "verdict": ev["overall"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())