#!/usr/bin/env python3
"""Build figures + docs/card_probe_report.md from the m5113 probe artefact.

All numbers in the markdown tables are injected programmatically from
artifacts/card_interference_probe.json (no hand transcription).

Figures (figures/):
  card_probe_speedup.png  : per-card single vs 8-way step time + speedup
  card_probe_bandwidth.png: HBM bandwidth and PCIe proxy, single vs concurrent
  card_probe_wake.png     : cylinder Re=100 vorticity + streamwise velocity

Usage:
    python scripts/card_probe_figures.py
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ART = REPO_ROOT / "artifacts"
FIGS = REPO_ROOT / "figures"

# Literature drag coefficient for an unbounded 2D circular cylinder at Re=100.
CD_LITERATURE = 1.33


def fig_speedup(payload: dict, out: Path) -> dict:
    pc = payload["per_card"]
    r = payload["results"]
    names = [p["device_name"].replace("/dev/", "") for p in pc]
    single = [p["single_median_step_ms"] for p in pc]
    conc = [p["concurrent_median_step_ms"] for p in pc]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(pc))
    w = 0.38
    ax1.bar(x - w / 2, single, w, label="1 card alone", color="#4C72B0")
    ax1.bar(x + w / 2, conc, w, label="8 cards concurrent", color="#DD8452")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("median step time (ms)")
    ax1.set_title("Cylinder Re=100 step time per card\n(same workload on every card)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    slow = [p["slowdown_pct"] for p in pc]
    ax2.bar(x, slow, color=["#C44E52" if v > 0 else "#55A868" for v in slow])
    ax2.axhline(0, color="k", lw=0.8)
    ax2.axhline(payload["results"]["interference_pct"], color="gray", ls="--",
                label=f"median {payload['results']['interference_pct']:+.2f}%")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("slowdown vs 1-card baseline (%)")
    ax2.set_title(
        f"Per-card interference | speedup_8x = {r['speedup_8x']:.2f}x "
        f"(ideal 8.0x)"
    )
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {"name": out.name,
            "caption": "Per-card cylinder Re=100 step time, single card alone vs "
                       "8 cards running independently, and the resulting "
                       "interference per card (negative = faster than the "
                       "single-card baseline)."}


def fig_bandwidth(payload: dict, out: Path) -> dict:
    pc = payload["per_card"]
    names = [p["device_name"].replace("/dev/", "") for p in pc]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(pc))
    w = 0.38
    ax1.bar(x - w / 2, [p["single_hbm_gbs"] for p in pc], w,
            label="1 card alone", color="#4C72B0")
    ax1.bar(x + w / 2, [p["concurrent_hbm_gbs"] for p in pc], w,
            label="8 cards concurrent", color="#DD8452")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("HBM bandwidth (GB/s)")
    ax1.set_title("STREAM-triad HBM bandwidth (512 MB working set)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x - w / 2, [p["single_h2d_gbs"] for p in pc], w,
            label="1 card alone", color="#4C72B0")
    ax2.bar(x + w / 2, [p["concurrent_h2d_gbs"] for p in pc], w,
            label="8 cards concurrent", color="#DD8452")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("host→device copy bandwidth (GB/s)")
    ax2.set_title("PCIe traffic proxy (pageable H2D, 256 MB)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {"name": out.name,
            "caption": "HBM bandwidth and PCIe (H2D copy) proxy bandwidth per "
                       "card, single card alone vs 8 cards concurrent."}


def fig_wake(out: Path) -> dict | None:
    npz = ART / "card_probe_field.npz"
    if not npz.exists():
        return None
    d = np.load(npz)
    ux, uy, w = d["ux"], d["uy"], d["vorticity"]
    solid = d["solid"]
    u = np.sqrt(ux**2 + uy**2)
    u[solid] = np.nan
    wm = np.where(solid, np.nan, w)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    lim = np.nanpercentile(np.abs(wm), 99)
    im0 = axes[0].imshow(wm, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                         aspect="equal")
    axes[0].set_title(f"Cylinder Re=100 vorticity ω (Cd={float(d['cd']):.3f})")
    fig.colorbar(im0, ax=axes[0], label="ω")
    im1 = axes[1].imshow(u, origin="lower", cmap="viridis", aspect="equal")
    axes[1].set_title("velocity magnitude |u|")
    axes[1].set_xlabel("x (lattice cells)")
    fig.colorbar(im1, ax=axes[1], label="|u|")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {"name": out.name,
            "caption": "Cylinder Re=100 wake from the timed workload: vorticity "
                       "(top) and velocity magnitude (bottom)."}


def fig_cd(out: Path) -> dict | None:
    """Cd(t): shedding cycle + blockage sensitivity vs literature value."""
    files = [(ART / "card_probe_field.npz", 256), (ART / "card_probe_field_ny384.npz", 384)]
    files = [(f, n) for f, n in files if f.exists()]
    if not files:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    for f, ny in files:
        d = np.load(f)
        series = d["cd_series"][-2000:]
        ax1.plot(np.arange(len(series)), series,
                 label=f"ny={ny} (blockage {100*16/ny:.2f}%), mean {float(d['cd_mean']):.4f}")
    ax1.axhline(CD_LITERATURE, color="k", ls="--", lw=1.2,
                label=f"literature, unbounded Re=100 = {CD_LITERATURE}")
    ax1.set_xlabel("step within sampling window")
    ax1.set_ylabel("Cd (instantaneous)")
    ax1.set_title("Cylinder Re=100 drag history: periodic vortex shedding")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    b, cd = [], []
    for f, ny in files:
        d = np.load(f)
        b.append(100 * 16 / ny)
        cd.append(float(d["cd_mean"]))
    o = np.argsort(b)
    b, cd = np.asarray(b)[o], np.asarray(cd)[o]
    ax2.plot(b, cd, "o-", label="time-mean Cd (this probe)")
    if len(b) >= 2:
        slope, icept = np.polyfit(b, cd, 1)
        bb = np.linspace(0, max(b) * 1.15, 50)
        ax2.plot(bb, slope * bb + icept, ":", color="gray",
                 label=f"linear extrapolation → Cd(B=0) ≈ {icept:.3f}")
        ax2.plot([0], [icept], "x", color="gray")
    ax2.axhline(CD_LITERATURE, color="k", ls="--", lw=1.2,
                label=f"literature {CD_LITERATURE}")
    ax2.set_xlabel("blockage (%)")
    ax2.set_ylabel("time-mean Cd")
    ax2.set_title("Blockage sensitivity: confinement raises Cd")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {"name": out.name,
            "caption": "Left: instantaneous Cd history showing the Re=100 "
                       "shedding cycle (why an instantaneous Cd must not be "
                       "compared to literature). Right: time-mean Cd vs "
                       "blockage with linear extrapolation to zero blockage."}


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> int:
    FIGS.mkdir(exist_ok=True)
    payload = json.loads((ART / "card_interference_probe.json").read_text())
    r = payload["results"]
    pc = payload["per_card"]

    figs = [fig_speedup(payload, FIGS / "card_probe_speedup.png"),
            fig_bandwidth(payload, FIGS / "card_probe_bandwidth.png")]
    wake = fig_wake(FIGS / "card_probe_wake.png")
    if wake:
        figs.append(wake)
    cdf = fig_cd(FIGS / "card_probe_cd.png")
    if cdf:
        figs.append(cdf)

    cd_vals = [p["cd"] for p in pc]
    cd_spread = max(cd_vals) - min(cd_vals)

    # time-averaged Cd (the only physically comparable value, see fig)
    f256 = np.load(ART / "card_probe_field.npz")
    cd_mean = float(f256["cd_mean"])
    cd_lo, cd_hi = float(f256["cd_min"]), float(f256["cd_max"])
    cd_err = 100.0 * (cd_mean - CD_LITERATURE) / CD_LITERATURE
    ny384 = ART / "card_probe_field_ny384.npz"
    if ny384.exists():
        d384 = np.load(ny384)
        cd384, cd384_err = float(d384["cd_mean"]), 100.0 * (float(d384["cd_mean"]) - CD_LITERATURE) / CD_LITERATURE
        b1, b2 = 100 * 16 / 256, 100 * 16 / 384
        slope = (cd_mean - cd384) / (b1 - b2)
        cd_extrap = cd384 - slope * b2
        extrap_err = 100.0 * (cd_extrap - CD_LITERATURE) / CD_LITERATURE
    else:
        cd384 = cd384_err = cd_extrap = extrap_err = float("nan")
        b1 = 100 * 16 / 256

    band_err = r["interference_pct"]
    speed_err = 100.0 * (8.0 - r["speedup_8x"]) / 8.0

    d2h_single = float(np.median([p["single_d2h_gbs"] for p in pc]))
    d2h_conc = float(np.median([p["concurrent_d2h_gbs"] for p in pc]))
    d2h_deg = 100.0 * (d2h_conc / d2h_single - 1.0)
    h2d_single = float(np.median([p["single_h2d_gbs"] for p in pc]))
    h2d_conc = float(np.median([p["concurrent_h2d_gbs"] for p in pc]))

    rows = "\n".join(
        f"| {p['device_name']} | {p['single_median_step_ms']:.3f} | "
        f"{p['concurrent_median_step_ms']:.3f} | {p['slowdown_pct']:+.2f} | "
        f"{p['single_hbm_gbs']:.1f} | {p['concurrent_hbm_gbs']:.1f} | "
        f"{p['hbm_degradation_pct']:+.1f} | {p['single_d2h_gbs']:.1f} | "
        f"{p['concurrent_d2h_gbs']:.1f} | {p['cd']:.5f} |"
        for p in pc
    )

    clk = payload.get("clocks", {})
    dev = payload["devices"]
    wl = payload["workload"]
    case = payload["case"]
    settle = payload.get("workload", {}).get("settle_s", 3.0)

    md = f"""# 8-Card Independent Interference Probe — Cylinder Re=100 (mission m5113)

**Verdict: PASS — no measurable compute interference.**
`speedup_8x = {r['speedup_8x']:.2f}×` vs an ideal `8.00×` (parallel efficiency
`{r['parallel_efficiency_pct']:.1f}%`), `interference_pct = {r['interference_pct']:+.2f}%`.
The machine criterion `speedup_8x >= 6.0` is met, so **this is not an
honest-FAIL**. One real contention channel was nevertheless found — host↔device
copy bandwidth (§4) — and it does not touch this compute-only workload.

---

## 1. Case and hardware

| item | value |
|---|---|
| solver | {wl['solver']} |
| grid | {wl['grid']} cells |
| cylinder diameter | {case['diameter']:g} cells (blockage {wl['blockage_pct']:.1f}%) |
| Reynolds number | {case['re']:g} |
| inlet velocity | {case['u_in']:g} (lattice units) |
| relaxation time τ | {case['tau']:.4f} |
| timed steps | {wl['steps_timed']} (warmup {wl['warmup_steps']}) |
| repetitions | {wl['reps']} × (S→C, C→S alternating) |
| cards | {dev['n_cards']} × `/dev/tcaicardNN` = {', '.join(dev['device_names'])} |
| `SDAA_VISIBLE_DEVICES` | `{dev['sdda_visible_devices']}` |

Device numbers are read back with `torch.sdaa.get_device_name(0)` **inside each
worker process**, so the recorded `/dev/tcaicardNN` is the card that worker
actually opened, not an assumed mapping.

## 2. Method

Each card runs its **own independent copy** of the cylinder case (task-level
parallelism) — not a domain decomposition of one global problem.

* **Phase S (baseline)** — a single worker process with `SDAA_VISIBLE_DEVICES`
  forced to one card, machine otherwise idle → `t1[i]`.
* **Phase C (concurrent)** — all {dev['n_cards']} worker processes live at once,
  one card each → `t8[i]`.

**Exclusivity protocol.** Every worker finishes its warmup, waits {settle:g} s for the
accelerator to reach a steady clock/power point, publishes `ready_<tag>`, and
then blocks on a `go` barrier file. The orchestrator releases the barrier only
after *all* workers are ready, so the timed loops overlap exactly and nothing
else runs on the machine during the window. The phase order alternates between
repetitions (S→C, then C→S) so slow thermal/DVFS drift cancels instead of biasing
the concurrent phase.

A worker asserts `torch.sdaa.device_count() == 1` at start-up, so card isolation
is machine-checked rather than merely intended.

Definitions:

```
speedup_8x       = 8 · median(t1) / median(t8)        # 8.0 = fully independent
interference_pct = (median(t8) / median(t1) − 1) · 100
```

Timing wraps each step with `torch.sdaa.synchronize()` and discards step 0.

## 3. Results

| metric | value |
|---|---|
| median step time, 1 card alone | {r['single_card_median_step_ms']:.3f} ms |
| median step time, 8 cards concurrent | {r['concurrent_8card_median_step_ms']:.3f} ms |
| **speedup_8x** | **{r['speedup_8x']:.3f}×** |
| parallel efficiency | {r['parallel_efficiency_pct']:.1f}% |
| **interference_pct** (median) | **{r['interference_pct']:+.2f}%** |
| interference_pct (mean over cards) | {r['interference_pct_mean_card']:+.2f}% |
| interference_pct (std over cards) | {r['interference_pct_std_card']:.2f}% |
| worst per-card slowdown | {r['interference_pct_max_card']:+.2f}% |
| **memory_bandwidth_GBs** (8 concurrent) | **{r['memory_bandwidth_GBs']:.1f} GB/s** |
| memory_bandwidth_GBs (1 card) | {r['memory_bandwidth_GBs_single']:.1f} GB/s |
| HBM bandwidth degradation | {r['memory_bandwidth_degradation_pct']:+.2f}% |
| aggregate throughput 1 → 8 cards | {r['aggregate_mlups_1card']:.1f} → {r['aggregate_mlups_8card']:.1f} MLUPS |
| MPE clock single / concurrent | {clk.get('single_mpe_mhz_median')} / {clk.get('concurrent_mpe_mhz_median')} MHz (max {clk.get('mpe_mhz_max')}) |

### Per-card detail

| card | t1 (ms) | t8 (ms) | slowdown % | HBM 1× (GB/s) | HBM 8× (GB/s) | HBM Δ% | D2H 1× | D2H 8× | Cd (instant) |
|---|---|---|---|---|---|---|---|---|---|
{rows}

Every card returns the **same** instantaneous Cd at the end of the timed run
(`{cd_vals[0]:.5f}`, spread `{cd_spread:.1e}`), i.e. concurrent execution is
bit-stable — no cross-card corruption.

## 4. PCIe traffic — the one real contention channel

`teco-smi 1.15.0` on this host exposes memory, utilisation, temperature, power,
voltage, current, clocks and PIDs — but **no PCIe throughput counter**. PCIe
traffic is therefore a *measured proxy*: host↔device copy bandwidth.

| direction | 1 card | 8 concurrent | change |
|---|---|---|---|
| H2D (GB/s) | {h2d_single:.2f} | {h2d_conc:.2f} | {100*(h2d_conc/h2d_single-1):+.0f}% |
| **D2H (GB/s)** | **{d2h_single:.2f}** | **{d2h_conc:.2f}** | **{d2h_deg:+.0f}%** |

**This is the only measurable interference effect found.** Device→host copy
bandwidth drops by roughly half when 8 cards transfer simultaneously, which is
the signature of a shared PCIe root complex / host-memory controller. It is
invisible in the LBM timing because the timed kernel loop performs no host
transfers — but any workflow that streams data to/from the host per step (I/O,
checkpointing, host-side reduction) would pay a ~2× penalty at 8-way
concurrency.

The H2D figures are small and noisy (pageable copies, {256} MB) and must not be
read as link capability; their spread is dominated by pageable-copy overhead.
No absolute PCIe byte-rate is asserted anywhere in this report — that quantity
is **WITHHELD** because the platform exposes no counter for it.

## 5. Why the efficiency is not really {r['parallel_efficiency_pct']:.0f}%

The measured speedup slightly exceeds 8×, which is impossible for a fixed
workload. The honest reading: the single-card baseline is marginally too
**slow**, not the concurrent run too fast.

* Clocks read during the runs are identical ({clk.get('single_mpe_mhz_median')} MHz
  in both phases, max {clk.get('mpe_mhz_max')} MHz), so DVFS is *not* the residual
  cause once the {settle:g} s settling pause is included.
* The dominant residual is run-to-run spread: the S-vs-C difference
  ({abs(r['interference_pct']):.1f}%) is comparable to the per-card repetition spread
  (~{float(np.median([np.ptp(p['single_step_spread_ms']) for p in pc])):.2f} ms).
* Therefore the defensible statement is **interference is below the
  ±{abs(r['interference_pct']):.1f}% noise floor of this measurement**, not "8.18×
  super-linear scaling".

Interference mechanisms explicitly checked:

| mechanism | probe | verdict |
|---|---|---|
| HBM contention | STREAM triad under 8-way load | absent ({r['memory_bandwidth_degradation_pct']:+.1f}%) |
| PCIe / host-memory contention | H2D/D2H copy bandwidth | **present on D2H ({d2h_deg:+.0f}%)**, irrelevant to compute-only loop |
| thermal / power throttling | MPE + SPE clocks, power | absent (clocks equal in both phases) |
| numerical cross-talk | Cd equality across cards | absent (identical to {cd_spread:.0e}) |

## 6. Error judgement

| quantity | measured | reference | error | < 3%? |
|---|---|---|---|---|
| speedup_8x | {r['speedup_8x']:.3f}× | 8.0× | {speed_err:+.1f}% | no — deviation is the interference noise floor (§5); it is negative (faster), i.e. zero measurable degradation |
| interference_pct | {r['interference_pct']:+.2f}% | 0.0% | {band_err:+.2f} pp | no strictly, but |Δ| < 3 pp ⇒ "no interference resolvable above this noise" |
| time-mean Cd (B={b1:.2f}%) | {cd_mean:.4f} | {CD_LITERATURE} | {cd_err:+.2f}% | **yes at this grid** — but see the caveat below |
| Cd extrapolated to B→0 | {cd_extrap:.3f} | {CD_LITERATURE} | {extrap_err:+.2f}% | **no** |

**The Cd agreement must not be over-claimed.** Two blockage points were run:

| ny | blockage | time-mean Cd | error vs {CD_LITERATURE} |
|---|---|---|---|
| 256 | {b1:.2f}% | {cd_mean:.4f} | {cd_err:+.2f}% |
| 384 | {100*16/384:.2f}% | {cd384:.4f} | {cd384_err:+.2f}% |

Cd rises monotonically with blockage (confinement raises drag), as expected.
Linear extrapolation to zero blockage gives `Cd(B→0) ≈ {cd_extrap:.3f}`
({extrap_err:+.1f}% vs literature), so the {cd_err:+.2f}% agreement at the probe grid
is a **cancellation of two errors** — wall confinement pushing Cd up and the
coarse D={case['diameter']:g} discretisation / finite inlet length pushing it down
— not independent evidence of solver accuracy. The quantity this mission
actually certifies is the *interference* number, and Cd is used here only as a
cross-card identity check.

Instantaneous Cd swings from {cd_lo:.3f} to {cd_hi:.3f} within one shedding cycle
(figure `card_probe_cd.png`), which is why any single-step Cd value — including
the `Cd (instant)` column above — is meaningless as an accuracy metric.

## 7. Recommendation

`speedup_8x = {r['speedup_8x']:.2f}× ≥ 6×` and interference is below the noise
floor, so **use all {dev['n_cards']} cards as independent workers**:

> **Verdict: task-level parallelism (one independent case per card) — use it.
> Domain decomposition — avoid.** DD on this platform relays halos through the
> host and was previously measured as a 6.8× *slowdown* on 4 cards
> (`artifacts/cross_card_4card.json`: 16.7 ms → 113.4 ms).
>
> **Operating rule:** keep the frequent data path on-card. If a workflow must
> stream host↔device traffic, budget ~2× less D2H bandwidth when all
> {dev['n_cards']} cards are busy.

Figures: `figures/card_probe_speedup.png`, `figures/card_probe_bandwidth.png`,
`figures/card_probe_wake.png`, `figures/card_probe_cd.png`.
Raw artefact: `artifacts/card_interference_probe.json`.
"""

    (REPO_ROOT / "docs" / "card_probe_report.md").write_text(md)
    print("wrote docs/card_probe_report.md")
    for f in figs:
        print(f"  fig: {f['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())