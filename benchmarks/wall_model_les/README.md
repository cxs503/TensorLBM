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
