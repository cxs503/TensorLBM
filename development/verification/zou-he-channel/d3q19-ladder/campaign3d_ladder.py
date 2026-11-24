"""Resolution / thickness / march-budget ladder for the Zou/He D3Q19 3D channel.

Anchor: ``XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI``.

``campaign3d.py`` published a two-point comparison (H = 8, 16) with no artifact
tree, so the 3-D half of this anchor rested on a *single* pre-registered number
per arm and on the in-suite assertion of ``tests/test_zou_he_3d_inlet_outlet.py``.
This script turns that into evidence a reviewer can re-derive from numbers:

* **ladder** — ``H = 8, 16, 32, 48`` at fixed ``nx``/``tau``/``drho``/steps.
  Two points cannot distinguish "converging" from "accidentally small at the
  tested width"; four points give an observed order.
* **thickness** — ``nz = 2`` vs ``nz = 8`` at ``H = 16``.  Every published 3-D
  number so far came from a *two-cell* periodic slab; if the closure had a
  spanwise defect the slab could hide it.  This arm states the size of that
  assumption instead of assuming it is benign.
* **march** — ``H = 16``, ``nx = 64``, 24 000 steps (4x the campaign budget) with
  a 2 000-step averaging block, i.e. a direct check that the numbers are not
  measured on a still-relaxing flow (the failure mode found in the 2-D round-2
  review of this same anchor).

Everything printed or plotted is read back from
``outputs/verification/zou-he-channel/d3q19-ladder/ladder.json``; nothing is re-typed.  The raw
streamwise profiles are stored in the artifact so the error can be recomputed
without re-running the solver.

Usage::

    PYTHONPATH=src python development/verification/zou-he-channel/d3q19-ladder/campaign3d_ladder.py --arms ladder
    PYTHONPATH=src python .../campaign3d_ladder.py --arms thickness,march --figures
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from tensorlbm.verification.zou_he_channel3d import simulate_channel_3d

OUT = pathlib.Path("outputs/verification/zou-he-channel/d3q19-ladder")
FIG = OUT / "figures"
JSON_PATH = OUT / "ladder.json"

# tag -> kwargs for simulate_channel_3d
ARMS: dict[str, dict] = {
    # resolution ladder: H = ny - 2
    "L08": dict(nx=48, ny=10, nz=2, steps=6000, block=800),
    "L16": dict(nx=48, ny=18, nz=2, steps=6000, block=800),
    "L32": dict(nx=48, ny=34, nz=2, steps=6000, block=800),
    "L48": dict(nx=48, ny=50, nz=2, steps=6000, block=800),
    # spanwise thickness at fixed H
    "T02": dict(nx=48, ny=18, nz=2, steps=6000, block=800),
    "T08": dict(nx=48, ny=18, nz=8, steps=6000, block=800),
    # 4x march budget / longer domain
    "M16": dict(nx=64, ny=18, nz=2, steps=24000, block=2000),
}

GROUPS = {
    "ladder": ["L08", "L16", "L32", "L48"],
    "thickness": ["T02", "T08"],
    "march": ["M16"],
}


def _load() -> dict:
    if JSON_PATH.exists():
        return json.loads(JSON_PATH.read_text())
    return {"arms": {}, "wall_time_s": {}}


def _save(doc: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(doc, indent=2, sort_keys=True))


def run(tags: list[str]) -> None:
    doc = _load()
    for tag in tags:
        t0 = time.time()
        res = simulate_channel_3d(**ARMS[tag])
        dt = time.time() - t0
        arm = res.as_dict()
        # recompute the headline error from the stored raw profile, so the
        # artifact is not a bag of self-reported scalars
        rec = float(
            np.abs(res.u_profile - res.u_exact).max() / res.u_exact.max()
        )
        arm["linf_rel_err_recomputed"] = rec
        arm["trace"] = res.trace
        doc["arms"][tag] = arm
        doc["wall_time_s"][tag] = round(dt, 1)
        _save(doc)
        print(
            f"{tag}: H={res.H:g} nz={res.nz} steps={res.steps} "
            f"linf={arm['metrics']['linf_rel_err']:.4%} amp={arm['metrics']['amp_rel_err']:.4%} "
            f"drift={arm['metrics']['drift_rel']:.2e} ({dt:.0f}s)"
        )


def order(tag_fine: str, tag_coarse: str) -> float:
    """Observed order from ``Linf ~ H^-p`` between two ladder arms."""
    doc = _load()
    a, b = doc["arms"][tag_fine], doc["arms"][tag_coarse]
    ea, eb = a["metrics"]["linf_rel_err"], b["metrics"]["linf_rel_err"]
    return float(np.log(eb / ea) / np.log(a["H"] / b["H"]))


def figures() -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    doc = _load()
    arms = doc["arms"]
    made: list[str] = []

    ladder = [t for t in GROUPS["ladder"] if t in arms]
    if len(ladder) >= 2:
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for tag in ladder:
            a = arms[tag]
            yt = np.arange(1, int(a["H"]) + 1) - 0.5
            ax[0].plot(a["u_profile"], yt, "o-", ms=3, label=f"H={a['H']:g} ({tag})")
            if tag == ladder[0]:
                ax[0].plot(a["u_exact"], yt, "k--", lw=1, label="analytic")
        ax[0].set_xlabel("$u_x$ [lattice]")
        ax[0].set_ylabel("$y'$ [lattice]")
        ax[0].set_title("Zou/He-driven D3Q19 channel vs plane Poiseuille")
        ax[0].legend(fontsize=7)

        hs = [arms[t]["H"] for t in ladder]
        es = [arms[t]["metrics"]["linf_rel_err"] for t in ladder]
        ax[1].loglog(hs, es, "s-", label=r"measured $L_\infty$")
        ref = es[0] * (hs[0] / np.array(hs)) ** 2
        ax[1].loglog(hs, ref, "k:", lw=1, label=r"2nd order from $H=%g$" % hs[0])
        ax[1].axhline(0.15, color="r", lw=1, label="owner gate 15%")
        ax[1].axhline(0.03, color="g", lw=1, label="stretch gate 3%")
        for h, e, t in zip(hs, es, ladder):
            ax[1].annotate(f"{e:.3%}", (h, e), textcoords="offset points",
                           xytext=(4, -10), fontsize=7)
        ax[1].set_xlabel("channel width $H$ [lattice]")
        ax[1].set_ylabel(r"$\|u-u_{ex}\|_\infty/u_{ex,\max}$")
        ax[1].set_title("resolution ladder (all arms $\\ll$ 15%)")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        p = FIG / "fig7_ladder3d.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(str(p))

    thick = [t for t in GROUPS["thickness"] if t in arms]
    if len(thick) == 2:
        a, b = arms[thick[0]], arms[thick[1]]
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
        yt = np.arange(1, int(a["H"]) + 1) - 0.5
        ax[0].plot(a["u_profile"], yt, "o-", ms=3, label=f"nz={a['nz']}")
        ax[0].plot(b["u_profile"], yt, "x-", ms=5, label=f"nz={b['nz']}")
        ax[0].plot(a["u_exact"], yt, "k--", lw=1, label="analytic")
        ax[0].set_xlabel("$u_x$ [lattice]")
        ax[0].set_ylabel("$y'$ [lattice]")
        ax[0].set_title("spanwise thickness arm (H=16)")
        ax[0].legend(fontsize=7)
        keys = ["linf_rel_err", "amp_rel_err", "shape_l2_rel", "drift_rel"]
        x = np.arange(len(keys))
        ax[1].bar(x - 0.2, [a["metrics"][k] for k in keys], 0.4, label=f"nz={a['nz']}")
        ax[1].bar(x + 0.2, [b["metrics"][k] for k in keys], 0.4, label=f"nz={b['nz']}")
        ax[1].set_yscale("log")
        ax[1].set_xticks(x, keys, rotation=15, fontsize=7)
        ax[1].axhline(0.03, color="g", lw=1)
        ax[1].set_title("thickness-sensitivity of the metrics")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        p = FIG / "fig8_thickness3d.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(str(p))

    if "M16" in arms:
        a = arms["M16"]
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
        ax[0].plot(a["trace"]["step"], a["trace"]["u_peak"], "o-", ms=4)
        ax[0].set_xlabel("step")
        ax[0].set_ylabel("mid-channel $u_{x,\\max}$")
        ax[0].set_title(f"march arm: {a['steps']} steps, block {a['metrics'].get('block', 2000)}")
        yt = np.arange(1, int(a["H"]) + 1) - 0.5
        ax[1].plot(a["u_profile"], yt, "o-", ms=3, label="march (24k steps)")
        ax[1].plot(a["u_exact"], yt, "k--", lw=1, label="analytic")
        ax[1].set_xlabel("$u_x$ [lattice]")
        ax[1].set_ylabel("$y'$ [lattice]")
        ax[1].set_title(f"$L_\\infty$={a['metrics']['linf_rel_err']:.3%}, drift={a['metrics']['drift_rel']:.1e}")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        p = FIG / "fig9_march3d.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(str(p))

    manifest = FIG / "figures_manifest_3d.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact": str(JSON_PATH),
                "pngs": [pathlib.Path(m).name for m in made],
            },
            indent=2,
        )
    )
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="ladder", help="comma list of groups or tags")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    tags: list[str] = []
    for item in args.arms.split(","):
        item = item.strip()
        if item in GROUPS:
            tags.extend(GROUPS[item])
        elif item in ARMS:
            tags.append(item)
        else:
            raise SystemExit(f"unknown arm/group {item!r}")

    run(tags)
    doc = _load()
    if {"L08", "L16"} <= set(doc["arms"]):
        pairs = [
            (f, c) for f, c in (("L16", "L08"), ("L32", "L16"), ("L48", "L32"))
            if f in doc["arms"] and c in doc["arms"]
        ]
        doc["observed_order"] = {f"{f}/{c}": round(order(f, c), 3) for f, c in pairs}
        _save(doc)
        print("observed order:", doc["observed_order"])
    if args.figures:
        print("figures:", figures())


if __name__ == "__main__":
    main()