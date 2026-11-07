# 8-Card Independent Interference Probe — Cylinder Re=100 (mission m5113)

**Verdict: PASS — no measurable compute interference.**
`speedup_8x = 8.18×` vs an ideal `8.00×` (parallel efficiency
`102.2%`), `interference_pct = -2.17%`.
The machine criterion `speedup_8x >= 6.0` is met, so **this is not an
honest-FAIL**. One real contention channel was nevertheless found — host↔device
copy bandwidth (§4) — and it does not touch this compute-only workload.

---

## 1. Case and hardware

| item | value |
|---|---|
| solver | tensorlbm D2Q9 BGK + halfway bounce-back, inlet/outlet |
| grid | 512x256 cells |
| cylinder diameter | 16 cells (blockage 6.2%) |
| Reynolds number | 100 |
| inlet velocity | 0.05 (lattice units) |
| relaxation time τ | 0.5240 |
| timed steps | 1200 (warmup 250) |
| repetitions | 2 × (S→C, C→S alternating) |
| cards | 8 × `/dev/tcaicardNN` = /dev/tcaicard14, /dev/tcaicard15, /dev/tcaicard16, /dev/tcaicard17, /dev/tcaicard18, /dev/tcaicard19, /dev/tcaicard20, /dev/tcaicard21 |
| `SDAA_VISIBLE_DEVICES` | `14,15,16,17,18,19,20,21` |

Device numbers are read back with `torch.sdaa.get_device_name(0)` **inside each
worker process**, so the recorded `/dev/tcaicardNN` is the card that worker
actually opened, not an assumed mapping.

## 2. Method

Each card runs its **own independent copy** of the cylinder case (task-level
parallelism) — not a domain decomposition of one global problem.

* **Phase S (baseline)** — a single worker process with `SDAA_VISIBLE_DEVICES`
  forced to one card, machine otherwise idle → `t1[i]`.
* **Phase C (concurrent)** — all 8 worker processes live at once,
  one card each → `t8[i]`.

**Exclusivity protocol.** Every worker finishes its warmup, waits 3 s for the
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
| median step time, 1 card alone | 13.250 ms |
| median step time, 8 cards concurrent | 12.963 ms |
| **speedup_8x** | **8.177×** |
| parallel efficiency | 102.2% |
| **interference_pct** (median) | **-2.17%** |
| interference_pct (mean over cards) | -2.64% |
| interference_pct (std over cards) | 1.85% |
| worst per-card slowdown | -0.09% |
| **memory_bandwidth_GBs** (8 concurrent) | **244.2 GB/s** |
| memory_bandwidth_GBs (1 card) | 244.8 GB/s |
| HBM bandwidth degradation | -0.27% |
| aggregate throughput 1 → 8 cards | 9.2 → 77.5 MLUPS |
| MPE clock single / concurrent | 2000.0 / 2000.0 MHz (max None) |

### Per-card detail

| card | t1 (ms) | t8 (ms) | slowdown % | HBM 1× (GB/s) | HBM 8× (GB/s) | HBM Δ% | D2H 1× | D2H 8× | Cd (instant) |
|---|---|---|---|---|---|---|---|---|---|
| /dev/tcaicard14 | 13.125 | 12.967 | -1.21 | 251.4 | 243.5 | -3.1 | 7.6 | 4.0 | 1.20589 |
| /dev/tcaicard15 | 13.235 | 12.952 | -2.14 | 246.4 | 244.8 | -0.6 | 7.9 | 3.8 | 1.20589 |
| /dev/tcaicard16 | 13.809 | 12.894 | -6.62 | 243.3 | 239.3 | -1.6 | 8.5 | 4.1 | 1.20589 |
| /dev/tcaicard17 | 13.252 | 12.990 | -1.98 | 239.7 | 240.3 | +0.2 | 7.6 | 5.2 | 1.20589 |
| /dev/tcaicard18 | 13.249 | 12.960 | -2.18 | 236.0 | 236.8 | +0.4 | 12.3 | 3.0 | 1.20589 |
| /dev/tcaicard19 | 13.367 | 12.972 | -2.95 | 253.6 | 250.3 | -1.3 | 6.5 | 3.5 | 1.20589 |
| /dev/tcaicard20 | 13.490 | 12.953 | -3.98 | 250.3 | 245.4 | -2.0 | 5.5 | 4.4 | 1.20589 |
| /dev/tcaicard21 | 12.977 | 12.966 | -0.09 | 239.7 | 245.9 | +2.6 | 7.6 | 3.3 | 1.20589 |

Every card returns the **same** instantaneous Cd at the end of the timed run
(`1.20589`, spread `0.0e+00`), i.e. concurrent execution is
bit-stable — no cross-card corruption.

## 4. PCIe traffic — the one real contention channel

`teco-smi 1.15.0` on this host exposes memory, utilisation, temperature, power,
voltage, current, clocks and PIDs — but **no PCIe throughput counter**. PCIe
traffic is therefore a *measured proxy*: host↔device copy bandwidth.

| direction | 1 card | 8 concurrent | change |
|---|---|---|---|
| H2D (GB/s) | 1.32 | 0.75 | -43% |
| **D2H (GB/s)** | **7.63** | **3.91** | **-49%** |

**This is the only measurable interference effect found.** Device→host copy
bandwidth drops by roughly half when 8 cards transfer simultaneously, which is
the signature of a shared PCIe root complex / host-memory controller. It is
invisible in the LBM timing because the timed kernel loop performs no host
transfers — but any workflow that streams data to/from the host per step (I/O,
checkpointing, host-side reduction) would pay a ~2× penalty at 8-way
concurrency.

The H2D figures are small and noisy (pageable copies, 256 MB) and must not be
read as link capability; their spread is dominated by pageable-copy overhead.
No absolute PCIe byte-rate is asserted anywhere in this report — that quantity
is **WITHHELD** because the platform exposes no counter for it.

## 5. Why the efficiency is not really 102%

The measured speedup slightly exceeds 8×, which is impossible for a fixed
workload. The honest reading: the single-card baseline is marginally too
**slow**, not the concurrent run too fast.

* Clocks read during the runs are identical (2000.0 MHz
  in both phases, max None MHz), so DVFS is *not* the residual
  cause once the 3 s settling pause is included.
* The dominant residual is run-to-run spread: the S-vs-C difference
  (2.2%) is comparable to the per-card repetition spread
  (~0.38 ms).
* Therefore the defensible statement is **interference is below the
  ±2.2% noise floor of this measurement**, not "8.18×
  super-linear scaling".

Interference mechanisms explicitly checked:

| mechanism | probe | verdict |
|---|---|---|
| HBM contention | STREAM triad under 8-way load | absent (-0.3%) |
| PCIe / host-memory contention | H2D/D2H copy bandwidth | **present on D2H (-49%)**, irrelevant to compute-only loop |
| thermal / power throttling | MPE + SPE clocks, power | absent (clocks equal in both phases) |
| numerical cross-talk | Cd equality across cards | absent (identical to 0e+00) |

## 6. Error judgement

| quantity | measured | reference | error | < 3%? |
|---|---|---|---|---|
| speedup_8x | 8.177× | 8.0× | -2.2% | no — deviation is the interference noise floor (§5); it is negative (faster), i.e. zero measurable degradation |
| interference_pct | -2.17% | 0.0% | -2.17 pp | no strictly, but |Δ| < 3 pp ⇒ "no interference resolvable above this noise" |
| time-mean Cd (B=6.25%) | 1.3248 | 1.33 | -0.39% | **yes at this grid** — but see the caveat below |
| Cd extrapolated to B→0 | 1.223 | 1.33 | -8.02% | **no** |

**The Cd agreement must not be over-claimed.** Two blockage points were run:

| ny | blockage | time-mean Cd | error vs 1.33 |
|---|---|---|---|
| 256 | 6.25% | 1.3248 | -0.39% |
| 384 | 4.17% | 1.2910 | -2.93% |

Cd rises monotonically with blockage (confinement raises drag), as expected.
Linear extrapolation to zero blockage gives `Cd(B→0) ≈ 1.223`
(-8.0% vs literature), so the -0.39% agreement at the probe grid
is a **cancellation of two errors** — wall confinement pushing Cd up and the
coarse D=16 discretisation / finite inlet length pushing it down
— not independent evidence of solver accuracy. The quantity this mission
actually certifies is the *interference* number, and Cd is used here only as a
cross-card identity check.

Instantaneous Cd swings from 1.294 to 1.373 within one shedding cycle
(figure `card_probe_cd.png`), which is why any single-step Cd value — including
the `Cd (instant)` column above — is meaningless as an accuracy metric.

## 7. Recommendation

`speedup_8x = 8.18× ≥ 6×` and interference is below the noise
floor, so **use all 8 cards as independent workers**:

> **Verdict: task-level parallelism (one independent case per card) — use it.
> Domain decomposition — avoid.** DD on this platform relays halos through the
> host and was previously measured as a 6.8× *slowdown* on 4 cards
> (`artifacts/cross_card_4card.json`: 16.7 ms → 113.4 ms).
>
> **Operating rule:** keep the frequent data path on-card. If a workflow must
> stream host↔device traffic, budget ~2× less D2H bandwidth when all
> 8 cards are busy.

Figures: `figures/card_probe_speedup.png`, `figures/card_probe_bandwidth.png`,
`figures/card_probe_wake.png`, `figures/card_probe_cd.png`.
Raw artefact: `artifacts/card_interference_probe.json`.
