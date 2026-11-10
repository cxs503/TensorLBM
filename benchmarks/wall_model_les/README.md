# Wall-modelled LES benchmarks

Machine-checked anchors for `tensorlbm.wall_model_les`. Every script writes a
JSON artifact under `artifacts/wall_model_les/`; `acceptance.py` turns them into
the criterion table of the mission node (`PASS` / `FAIL` / `WITHHELD`).

| script | what it runs | artifact |
|---|---|---|
| `apriori.py` | wall models against the MKM-180 channel DNS shipped in the repo (anchor recovery, log-layer profile, stress peak, non-equilibrium sensitivity) | `apriori.json` |
| `flat_plate.py` | ZPG flat plate `Re_theta = 1e4 ... 5e4` vs ITTC-1957 (local form), Prandtl-Schlichting, `0.024 Re_theta^-1/4`, plus a `Pi`/`kappa` sensitivity sweep | `flat_plate.json`, `flat_plate_profiles.npz` |
| `channel_wmles.py` | D3Q19 wall-modelled channel (`--case eq|neq|eq_log`), wall-unit profiles, stress identity, budget | `channel_<case>.json`, `profiles_channel_<case>.npz` |
| `kvlcc2.py` | KVLCC2 `C_T = C_F (1+k) + C_R` at `Re = 1.3e7` vs the ship platform's traceable ITTC-78 chain | `kvlcc2.json` |
| `acceptance.py` | criterion table + figures | `acceptance.json`, `fig_*.png` |
| `run_shard.sh` | cluster shard dispatcher (`shard 1/3` = channel eq, `2/3` = channel non-eq, `3/3` = flat plate) | — |

Reproduce the whole suite locally:

```bash
export PYTHONPATH=src
python benchmarks/wall_model_les/apriori.py    --out artifacts/wall_model_les
python benchmarks/wall_model_les/flat_plate.py --out artifacts/wall_model_les
python benchmarks/wall_model_les/kvlcc2.py     --out artifacts/wall_model_les   # needs the ship API on :8001
python benchmarks/wall_model_les/channel_wmles.py --case eq --steps 240 --nx 32 --nz 32 --cs 0.2 \
       --out artifacts/wall_model_les
python benchmarks/wall_model_les/acceptance.py --out artifacts/wall_model_les
```

Reading the record honestly:

* criterion (b) is `WITHHELD`: the coarse-grid channel run does not reach a
  statistically stationary state, so the resolved stress peak is not a
  measurement. See `docs/wall_model_les_report.md` §5.
* criterion (c) is `FAIL` at 11.4 % (threshold 5 %); the `Pi` sweep in
  `flat_plate.json` shows that the Coles wake strength controls that number.
* criterion (d) passes against the platform's ITTC-78 chain (`C_F` is
  independent, `C_R` is borrowed) and is `WITHHELD` against an experiment.

## Mission m7797 additions (wall-model tier: defects found and fixed)

| script | what it runs | artifact |
|---|---|---|
| `wall_origin.py` | laminar plane-Poiseuille calibration of the effective bounce-back wall origin (wall model off) | `wall_origin.json` |
| `log_law_fit.py` | `(kappa, B)` implied by the reference channel DNS, and the profile error over the criterion band | `log_law_fit.json` |
| `m7797_report.py` | assembles and publishes the paper-grade report from the artifacts above | `docs/wall_model_les_m7797_report.md` |

Two defects were found and fixed in this pass (both pinned by
`tests/test_wall_model_les_m7797.py`):

1. `flat_plate.coles_profile` multiplied the Coles wake function by
   `4 Pi/kappa` instead of `2 Pi/kappa`.  The boundary-layer edge velocity was
   ~10 % high, so `c_f = 2/Ue+^2` was ~10 % low and criterion (c) failed at
   11.4 %; after the fix the error over `Re_theta = 1e4 ... 5e4` is 3.6 %
   (`PASS` vs the 5 % threshold, and consistent with the independent
   Prandtl-Schlichting line at 3.6 %).
2. The wall-modelled channel treated the bounce-back wall row as a fluid node
   half a cell from the wall (`wall_origin_offset = 0.5`) and sampled the wall
   model one cell too close to the wall.  The recovered friction velocity
   collapsed to 0.60-0.71 of the imposed force balance.  With the measured
   origin (`wall_origin_offset = 1.0`, exchange node at 1.5 cells) the recovery
   is 1.01.  `wall_origin_offset = 0.5` is kept to reproduce the before/after.

New channel options: `--coupling {assimilate,stress,none}`, `--wall-origin-offset`,
`--re-tau`, `--u-tau`.  `stress` imposes the modelled wall stress as a momentum
flux (fixed point `tau_num = rho u_tau^2`); `none` is the no-wall-model control.

Honest verdicts of the m7797 pass: (a) FAIL (6.3 % vs MKM-180 DNS, a
log-law-constant issue -- the DNS implies kappa = 0.36), (b) FAIL (coarse-grid
quantisation of the peak abscissa plus an unclosed near-wall momentum budget),
(c) PASS (3.6 %), (d) `WITHHELD` vs experiment (`PASS` only against the
platform's ITTC-78 chain, with `C_R` borrowed).  The `Re_tau = 590` tier of the
channel solver still diverges to NaN at `nu = 8.1e-4`; that boundary is recorded,
not hidden.

## Mission m8872 additions (closure tier + reference integrity + offline chain)

| script | what it runs | artifact |
|---|---|---|
| `high_re.py` | the wall-model closure solved over the whole channel at `Re_tau = 180 ... 1e4`: `u+`, `-<u'v'>+`, production, log-layer fit vs the closure's own `(kappa, B)`, DNS validation, reference-integrity gate | `high_re.json`, `fig_high_re_*.png`, `fig_reference_gate.png` |
| `stability_map.py` | maps the `(Re_tau, Cs)` range in which the D3Q19 wall-modelled channel converges (`--summarise` merges the cases) | `stability_*.json`, `stability_map.json` |
| `m8872_report.py` | renders `docs/wall_model_les_m8872_report.md` from the artifacts and optionally publishes it (`--publish`) | report |

Three defects were found and fixed in this pass:

1. `budget.load_dns_channel_profile` hard-coded `Re_tau = 180`; the shipped tables
   stop at the last cell centre (`y+ = 178.12`), so the value cannot be inferred
   from the table end. The loader now takes the case identification from
   `DNS_REFERENCES` and reports the table end separately.
2. `apriori.py` reported the *boundary* maximum of a profile truncated at the
   last DNS sample (`1.009` at `y+ = 178.1`) as the modelled Reynolds-stress
   peak. The closure is now solved over the whole channel half-height, where the
   peak is an interior maximum (`y+ = 29.28`, 8.4 % from the DNS `y+ = 31.98`).
3. `apriori.py` reported `max(-uv_plus)` as the DNS peak value, i.e. the negative
   of the tabulated correlation; fixed to `max(uv_plus)` (0.724).

Honest verdicts of the m8872 pass: (a) FAIL (3.93 % best predictive limb vs the
3 % threshold; the 1.72 % of the m7797 record was the *anchored shape* limb and
is not a profile prediction), (b) FAIL (8.43 %, and the modelled peak abscissa
drifts with `Re_tau` under an eddy-viscosity closure while the DNS one does not),
(c) PASS (3.63 %), (d) `WITHHELD` versus a measurement -- no KVLCC2 towing-tank
`C_T` exists in this environment and the ship API was unreachable -- while the
offline friction limb passes at 4.44 % against the ITTC-1957 line.

The wall-modelled channel itself does **not** reach the criterion band: it
diverges for `Re_tau >= 590` at `Cs = 0.05 / 0.1` (see `stability_map.json`), so
the resolved `y+ = 30...300` measurement is `WITHHELD` and the closure tier
carries criterion (a).

Reference integrity: `benchmarks/in99_n03_channel_turb/dns_ref/mkm395.csv` is
**rejected** by `reference_gate` (non-monotone `u+`, centreline `u+ = 15.96` vs a
log-law estimate of 19.78, stored correlation reaching 6.0). It is not a
`Re_tau = 395` channel profile and must not be used as a reference.
