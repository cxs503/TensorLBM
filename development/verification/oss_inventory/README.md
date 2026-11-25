# Historical OSS inventory studies

The two studies recorded the contents of the OSS cache available at the time.
The original anchors identify those snapshots. Their reports and verdicts
describe their original inputs, references, and acceptance criteria.

## BEM

- [Board audit](../../../docs/board_audit/m18158/AUDIT_m18106.md)
- [Preregistration](bem_cavity_convection/predeclared.json)
- [Cavity driver](bem_cavity_convection/run_xval.py)
- [Maintained cavity benchmark](../../../benchmarks/verified/cavity_natural_convection/README.md)

Run a portable CPU path smoke from the checkout root:

```sh
python development/verification/oss_inventory/bem_cavity_convection/run_xval.py \
  --nx 16 --ra 1000 --steps 6 --report-every 2 --threads 1 \
  --out outputs/historical_xval/oss_inventory/bem-smoke.json
```

The historical analysis and report generation require the original cache,
full-grid runs and derived files. The board audit retains the old path as
historical provenance. `finalize.py` is the original publication utility.

## DEM

- [Final report](../../../docs/REPORT_XVAL_LBM_NO_DEM_REPO_FINAL.md)
- [OSS survey](dem_literature_anchors/survey_oss.md)
- [Contact and settling evidence](dem_literature_anchors/results.json)
- [Round 3 fluid-side evidence](dem_literature_anchors/round3_lbm_cd.json)
- [Maintained sphere-drag benchmark](../../../benchmarks/pending/sphere_drag_closure/README.md)

Pass an output path when replaying the driver to preserve the archived result:

```sh
mkdir -p outputs/historical_xval/oss_inventory
OMP_NUM_THREADS=1 python development/verification/oss_inventory/dem_literature_anchors/run_xval.py \
  --quick --out outputs/historical_xval/oss_inventory/dem-smoke.json
```

Use a separate checkout for the historical round 3 driver since it writes
beside its source. It reads the tracked sphere-drag results from the maintained
case. `publish_report.py` is the original publication utility.
