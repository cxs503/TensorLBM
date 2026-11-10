# P223W1-N01 — DNS-arm turbulent channel verification face (Re_tau = 180/395/590)

This benchmark turns the native OpenMP channel kernel
(`tensorlbm.turbulence.les_chan`, D3Q19 TRT, Guo forcing, half-way bounce-back)
into a **DNS arm** (`cs = 0`, i.e. the Smagorinsky closure switched off
bit-for-bit) and grades the result against the MKM-1999 DNS statistics and
Dean's friction correlation through the fail-closed gate set `D0..D6`
(`tensorlbm.turbulence.channel_dns`).

It is deliberately *not* another LES campaign: the sibling benchmark
`benchmarks/in99_n03_les_channel` grades the LES closure (`C1..C6`); this one
asks whether the solver itself resolves the wall layer, and reports the
wall-unit resolution it actually achieved.

## Files

| file | role |
|---|---|
| `run_case.py` | case ladder (grid, `Re_tau`, averaging budget, spectrum rows) + native runner |
| `gates.py` | recomputes every number from the artefacts and writes `gates.json` (fail-closed) |
| `plot.py` | mean profile, Reynolds stresses, spectra, friction law, resolution budget figures |

Kernel additions used here (`src/tensorlbm/turbulence/les_chan.c`):
`--spec-every` / `--spec-rows` append raw velocity planes (float32, plane-major)
to `<prefix>_spec.bin` with `<prefix>_spec.json` metadata; the post-processing is
`tensorlbm.turbulence.channel_spectra` (Parseval-normalised 1-D spectra with an
*a-priori* log-log fit band, so the reported slope cannot be tuned to `-5/3`).

## Solver finding that the ladder depends on

The regularised (Hermite-projected) collision used by the LES campaigns is
**slowly unstable** at `tau0 - 0.5 ~ 4e-3` when the SGS term is off. Measured on
`188x122x94`, `u_tau = 0.0125`, `Re_tau = 180`:

```
--reg 1 : umax ~ 1e6, rho_min = nan        (diverged)
--reg 0 : umax ~ 0.25, rho_min = 0.999     (stable)
```

so the DNS arm runs `--reg 0` (plain TRT, magic parameter `Lambda = 3/16`).
This is a solver-level defect of the regularised path and is reported as such;
it is not a modelling choice.

## Run

```bash
python3 benchmarks/p223w1_n01_channel_dns/run_case.py --list
python3 benchmarks/p223w1_n01_channel_dns/run_case.py --case R180_h60 --threads 32
python3 benchmarks/p223w1_n01_channel_dns/gates.py
python3 benchmarks/p223w1_n01_channel_dns/plot.py
```

## Ladder

| case | `Re_tau` | grid (`nx x ny x nz`) | `h` | `dy+` | `Lx/h`, `Lz/h` | steps / warmup | sampled turnovers |
|---|---|---|---|---|---|---|---|
| `R180_h60` | 178.12 | 126x122x63 | 60 | 2.97 | 2.10, 1.05 | 40000 / 12000 | 3.75 |
| `R180_h30` | 178.12 | 63x62x32 | 30 | 5.94 | 2.10, 1.07 | 30000 / 9000 | 4.38 |
| `R395_h60` | 395 | 126x122x63 | 60 | 6.58 | 2.10, 1.05 | 40000 / 12000 | 3.75 |
| `R590_h60` | 590 | 126x122x63 | 60 | 9.83 | 2.10, 1.05 | 40000 / 12000 | 3.75 |

Two deliberate trade-offs, both forced by the compute actually available (the
dispatcher queue did not schedule any job during the campaign window, so every
leg ran on the shared 128-core control host at 12-40 threads):

* **box**: `2.1h x 2h x 1.05h` instead of the textbook `2pi h x 2h x pi h`. A
  minimal-ish box truncates the very-large-scale motions; low-order statistics
  (mean profile, `C_f`, near-wall stresses) are the least sensitive to it, and
  that is what the gate set grades.
* **wall-normal resolution vs Reynolds number**: the cost `~ h^3` per turnover
  is what makes `dy+ <= 1.5` unaffordable above `Re_tau ~ 200` here; keeping one
  cost unit per leg means `dy+` grows with `Re_tau`.

### Measured stability map (every configuration attempted)

`DIVERGED` means the run was stopped once the running plane dump went
non-finite (the kernel keeps integrating NaN and never yields a statistic);
`LAMINARISED` means the run was stable but the fluctuation peaks stayed below
half of the MKM-1999 level (`channel_dns.turbulence_level`), i.e. the initial
perturbation decayed before/inside the averaging window; `RAN` means the run
completed and was graded by `D0..D7`.

| case | `Re_tau` | `u_tau` | `C_s` | reg | `dy+` | steps | outcome |
|---|---|---|---|---|---|---|---|
| `R180_h60` | 178.12 | 0.0125 | 0.0 | 0 | 2.97 | 40000 | DIVERGED |
| `R180_h60_cs03` | 178.12 | 0.0125 | 0.03 | 0 | 2.97 | 40000 | DIVERGED |
| `R180_h60_les_cs010` | 178.12 | 0.0125 | 0.1 | 0 | 2.97 | 40000 | DIVERGED |
| `R180_h60_ut006_cs0` | 178.12 | 0.006 | 0.0 | 0 | 2.97 | 20000 | LAMINARISED |
| `R180_h60_dns_ut006` | 178.12 | 0.006 | 0.0 | 0 | 2.97 | 65000 | LAMINARISED |
| `R180_h30_les_cs010` | 178.12 | 0.0125 | 0.1 | 0 | 5.94 | 30000 | DIVERGED |
| `M1_ut0.006_cs0.0_reg1` | 178.12 | 0.006 | 0.0 | 1 | 5.94 | 30000 | DIVERGED |
| `M2_ut0.006_cs0.1_reg1` | 178.12 | 0.006 | 0.1 | 1 | 5.94 | 30000 | RAN |
| `M3_ut0.0125_cs0.1_reg1` | 178.12 | 0.0125 | 0.1 | 1 | 5.94 | 30000 | DIVERGED |
| `M4_ut0.006_cs0.1_reg1_long` | 178.12 | 0.006 | 0.1 | 1 | 5.94 | 70000 | RAN |
| `R395_h60` | 395.00 | 0.0125 | 0.0 | 0 | 6.58 | 40000 | DIVERGED |
| `R590_h60` | 590.00 | 0.0125 | 0.0 | 0 | 9.83 | 40000 | DIVERGED |
| `R395_h60_dns_ut006` | 395.00 | 0.006 | 0.0 | 0 | 6.58 | 40000 | DIVERGED |
| `R590_h60_dns_ut006` | 590.00 | 0.006 | 0.0 | 0 | 9.83 | 40000 | DIVERGED |

Read-off: **`u_tau = 0.0125` (centre-line `Ma ~ 0.45`) diverges in every
configuration tested**, with or without an SGS term; `u_tau = 0.006`
(`Ma ~ 0.22`) is stable, but the wall-resolved `cs = 0` arms relaminarise inside
the affordable window while the `cs = 0.1` arm at `dy+ ~ 6` is turbulent and
stable (that is the only run that produced full statistics, `M4`).  A
resolvable DNS arm therefore needs both a low-Mach formulation/forcing and
enough averaging time, neither of which fits the compute that was actually
available (`channel_dns.dns_cells_required`: 2.2e8 cells at `Re_tau = 180`,
8.1e9 at 590).

## Honest scope

`D0` (resolution attestation) is *expected to fail* for the `Re_tau = 395/590`
legs: at `Re_tau = 590` a `dy+ = 1` DNS needs ~8.1e9 cells (see
`channel_dns.dns_cells_required`), i.e. three orders of magnitude beyond a
7200 s job on the two available 128-core nodes, and ~30x that for the sampling
window. The report states the achieved `dy+` per leg and grades the physics gates
that are still meaningful (`D1..D5`), rather than relabelling a wall-resolved LES
as a DNS.