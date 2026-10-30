# SDAA sweep — `sphere_re100_d3q27`

Generated 2026-10-29T14:00:55Z by `scripts/sdaa_sweep_sphere_re100_d3q27.py` (branch `agent/m42`, commit `0d417663`). Every number below is emitted by that script; no judgment value is hand-copied.

## Case & frozen gate protocol

- Case: `benchmarks/verified/sphere_re100_d3q27` — D3Q27 twin of the verified D3Q19 sphere at Re=100, BGK27 collision, BFL sparse wall links (exact ray-sphere q), Cd taken over the last-20 % sampled window.
- Reference: `cd_ref` machine-read from the sha256-pinned archive `benchmarks/verified/sphere_re100_d3q27/diag/remote_pull/verified__sphere_re100/result.json` → **1.0917310910949** (Schiller–Naumann 24/Re·(1+0.15·Re^0.687)); archived tiers scored T1_D40 err 2.7917 %, T2_D60 err 2.5692 %.
- Gates, unchanged from `verify.py` / `prereg.md`: **G1** |err| ≤ 3 % per tier; **G2** err strictly decreasing with resolution; **G3** window drift < 0.3 %, CV-closure ≤ 1e-2, |mass drift| ≤ 100 ppm, finite; **G4** cross-stencil / collision rows are disclosures, not gates.
- Cd window: mean of sampled `cd` for `0.8*steps < step ≤ steps`, drift against `0.6*steps < step ≤ 0.8*steps` (`verify.window_cd`).

## Device evidence

```
torch.sdaa.get_device_name(0) = '/dev/tcaicard27'
torch 2.7.1 · torch_sdaa 3.1.1a3+gite3971b9 · device_count 1 · tensor device sdaa:0
sdaa probe sum 84.0 == cpu 84.0 -> True ; torch.cuda.is_available() = False
```

Live in-process smoke run on the same device: D6 (domain [42, 42, 48], tau 0.509000, Re 100.000000), 40 steps, finite=True, peak SDAA allocation 0.0452 GiB.

torch.cuda.is_available() is False on this host, so no CUDA run can be mistaken for the SDAA run; the case driver allocates every field on device='sdaa:0' (device=torch.device(DEV) with DEV=W27_DEV).

## Configuration

- `u_lb` = 0.05, Re target = 100.0, padding lat/up/down = 3.0/3.0/4.0, force sample interval = 50 steps.
- `tau = 0.5 + 3*u_lb*D/re_target (archived tiers: D40->0.56, D60->0.59)` — rule self-check against the archived tiers: D40 → 0.560000 vs archived 0.56 (ok), D60 → 0.590000 vs archived 0.59 (ok).

Deviations from the archived tiers (disclosed; no threshold relaxed):

- reduced lattice resolution D (archived D40/D60 take ~24 min / ~2.6 h loop time per tier on a 32 GB CUDA card; the wall-clock budget here is <= 20 min)
- tau recomputed per D to hold Re=100 (the same rule reproduces the archived tau values 0.56/0.59 exactly -- see tau_rule_check)
- force-sample interval (50 steps) and Cd window definition unchanged

## Results (machine values from the tier JSONs)

| tier | domain (cells) | Mcells | tau | steps | conv. times | Cd window | err vs cd_ref | drift % | CV closure | mass ppm | loop wall s | peak SDAA GiB (live proof in the device block) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D8 | [56, 56, 64] | 0.201 | 0.512000 | 1200 | 0.94 | 1.2415132401 | +13.7197 % | 0.8116 | 2.588e-02 | -0.078 | 420 | n/a |
| D16 | [112, 112, 128] | 1.606 | 0.524000 | 1400 | 0.55 | 1.1761841883 | +7.7357 % | 1.7274 | 6.033e-02 | +0.078 | 1608 | n/a |

## Transient trace (Cd at fixed fractions of each run)

| tier | 10 % | 20 % | 30 % | 50 % | 70 % | 90 % | 100 % |
|---|---|---|---|---|---|---|---|
| D8 | 1.2392@100 | 1.3767@250 | 1.3129@350 | 1.2666@600 | 1.2544@850 | 1.2408@1100 | 1.2382@1200 |
| D16 | 1.4226@150 | 1.2445@300 | 1.2621@400 | 1.2195@700 | 1.1915@1000 | 1.1745@1250 | 1.1726@1400 |

The 90 %/100 % values are still moving, i.e. the drag plateau has not been reached inside the affordable step count — that is the machine explanation of the G1/G3 FAIL rows above (no threshold was changed to hide it).

## Gate-by-gate verdict

| gate | threshold | value | verdict | source |
|---|---|---|---|---|
| `T.D8.re_eff_lock` | |re_eff-100| <= 1e-6 | 100 | **PASS** | tier meta re_eff |
| `T.D16.re_eff_lock` | |re_eff-100| <= 1e-6 | 100 | **PASS** | tier meta re_eff |
| `G1.accuracy.D8` | |err_pct| <= 3.0 | 13.7196925393 | **FAIL** | verify.tier_verdict |
| `G1.accuracy.D16` | |err_pct| <= 3.0 | 7.73570505407 | **FAIL** | verify.tier_verdict |
| `G2.monotone` | err(T2) < err(T1) (strict) | [13.7197, 7.73571] | **PASS** | verify.py G2 |
| `G3.D8.drift` | drift_pct < 0.3 | 0.811639836141 | **FAIL** | verify.tier_verdict |
| `G3.D8.cv_closure` | cv_closure <= 1e-2 | 0.0258830447642 | **FAIL** | verify.tier_verdict |
| `G3.D8.mass_drift` | |mass_drift_ppm| <= 100 | -0.0778509827869 | **PASS** | verify.tier_verdict |
| `G3.D8.finite` | no NaN/Inf through the loop | True | **PASS** | verify.tier_verdict |
| `G3.D8.archive_twin_param_guard` | archived-tier twin guard armed | None | **NOT_APPLICABLE** | run.py check_against_archive |
| `G3.D16.drift` | drift_pct < 0.3 | 1.72738957754 | **FAIL** | verify.tier_verdict |
| `G3.D16.cv_closure` | cv_closure <= 1e-2 | 0.0603277071922 | **FAIL** | verify.tier_verdict |
| `G3.D16.mass_drift` | |mass_drift_ppm| <= 100 | 0.0778511343066 | **PASS** | verify.tier_verdict |
| `G3.D16.finite` | no NaN/Inf through the loop | True | **PASS** | verify.tier_verdict |
| `G3.D16.archive_twin_param_guard` | archived-tier twin guard armed | None | **NOT_APPLICABLE** | run.py check_against_archive |
| `ref.archive_sha256_pin` | all declared pins match | {"result.json": true, "t1_d40big_12k.json": true, "t2_d60big_24k.json": true, "manifest.json": true} | **PASS** | gate/verdict27.json ref_sha256 |
| `env.device_is_sdaa` | fields computed on 'sdaa:0' | torch.sdaa.get_device_name(0) = '/dev/tcaicard27' | **PASS** | torch.sdaa |
| `env.sdaa_live_compute` | peak SDAA allocation > 0 and finite | 0.0451889038086 | **PASS** | torch.sdaa (this process) |

NOT_APPLICABLE rows and why:

- `G3.D8.archive_twin_param_guard`: run.py check_against_archive is keyed to D in (40,60); a reduced tier has no archived twin, so this sub-gate cannot be armed -- substituted by the ref.archive_sha256_pin row (no threshold changed)
- `G3.D16.archive_twin_param_guard`: run.py check_against_archive is keyed to D in (40,60); a reduced tier has no archived twin, so this sub-gate cannot be armed -- substituted by the ref.archive_sha256_pin row (no threshold changed)

**Overall: FAIL** — every applicable row must PASS.

## Wall clock (budget vs. measured)

- Budget: **1120 s**; measured harness wall clock: **2039 s** → over budget by **919 s** (OVER BUDGET).
- Per-tier loop time: {"D8": 419.7725999355316, "D16": 1608.0994911193848} s; per-step cost {"D8": 349.8, "D16": 1148.6} ms.
- Pre-run probe per-step cost {"D12": 325.1, "D16": 459.4} ms; slowdown observed during the run {"D16": 2.5}×.
- Cause of the overrun: the SDAA device is shared with other tenants (several tensorlbm sweep processes are resident on /dev/tcaicard27); per-step cost measured in the pre-run probe was reproduced for the first tier but degraded for the second, which is the machine explanation of the budget overrun below.
- The verdict does not depend on wall clock; nothing was trimmed to fit it.

## Bypass record

- Triton: not required: no triton kernel on the run.py import/exec path (d3q27 stream27_roll is pure torch.roll); TRITON_INTERPRET set to 1 defensively but never exercised. Runtime present: 3.4.0.
- Entry chain scanned for triton references: ['benchmarks/verified/sphere_re100_d3q27/run.py', 'src/tensorlbm/bfl_common.py', 'src/tensorlbm/d3q27.py', 'src/tensorlbm/general_sim.py', 'src/tensorlbm/boundaries3d.py']; hits: none.
- Device: `W27_DEV=sdaa:0`; the case driver's `torch.cuda.*` bookkeeping calls are either guarded by `dev.type == 'cuda'` or are no-ops without CUDA (`torch.cuda.empty_cache`), so no CUDA path executed.
- Untouched: no file under `benchmarks/verified/sphere_re100_d3q27/` or `src/` was modified by this sweep; only `scripts/`, `artifacts/`, `docs/sdaa_sweep/` gained files.

## Honest notes

- FAIL rows: `G1.accuracy.D8`, `G1.accuracy.D16`, `G3.D8.drift`, `G3.D8.cv_closure`, `G3.D16.drift`, `G3.D16.cv_closure` — see the table above for the machine value against the frozen threshold; the causes are recorded in the note/source columns.
- Window statistics are thin: only D8 n=5, D16 n=6 samples in the Cd window (force sample interval 50 steps, window = 20 % of the steps).
- Steps cover < 1 convective time per tier (see conv. times column), so the 0.8-1.0 window still sits in the transient: this is what fails the drift and CV-closure rows, and it inflates |err| (the drag is still relaxing onto its plateau). The gate thresholds were NOT relaxed to absorb it.
- The reduced-resolution tiers sit far from the archived 25.1M / 84.7M-cell domains, so accuracy/drift/CV rows measure the SDAA port at small scale, not the archived tier quality; the archived verdict (`gate/verdict27.json`) is unchanged and remains the case's headline result.

- Cross-check performed once by hand during this sweep: re-running the case's own `verify.py` in this environment re-derives `gate/verdict27.json` with identical gate booleans (G1/G2/G3 true, verdict PASS) but last-digit float differences in the window sums (summation-order nondeterminism); the file was restored with `git checkout`, so no case/business file is modified by this sweep.
- Caveat on scope: this sweep proves the SDAA port runs the real D3Q27/BFL route and records where the frozen gates stand at affordable resolution. It does NOT claim the archived PASS on SDAA — reaching the archived tiers (25.1M/84.7M cells, 12k/24k steps) is out of reach of the 20-minute budget on this shared card.

Artifacts: `artifacts/sdaa_sphere_re100_d3q27.json`, tier JSONs under `artifacts/sdaa_sphere_re100_d3q27_tiers`, this report `docs/sdaa_sweep/sphere_re100_d3q27.md`.
