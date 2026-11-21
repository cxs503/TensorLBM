-- m22809: compressed fluid-node neighbour table wired into the porous-media solver
DELETE FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE';
INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at)
VALUES ('XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE',
        'single_phase_slab_darcy_channel_production_backend',
        'analytic periodic-slab k_super=(H/ny)*H^2/12 (half-way walls); m22754 XVAL record -0.1094% at H=16',
        'compressed layout |err| = 0.4375% (H=8), 0.1094% (H=16); identity vs dense_halfway 6.9e-15/2.6e-15',
        '0.03', 'TensorLBM', 'passed', now());

-- reference vs got, one row per arm (k_got / k_ref / err_pct)
WITH b(case_name, k_ref, k_got, err_pct) AS (VALUES
    ('compressed_ny12_h8', 3.5555555555555554, 3.5399999997434226, -0.43750000721623356),
    ('dense_halfway_ny12_h8', 3.5555555555555554, 3.539999999743447, -0.4375000072155466),
    ('dense_ny12_h8', 3.5555555555555554, 4.662222221752547, 31.12499998679039),
    ('compressed_ny24_h16', 14.222222222222221, 14.206666661051756, -0.10937503947983651),
    ('dense_halfway_ny24_h16', 14.222222222222221, 14.206666661051718, -0.1093750394800988),
    ('dense_ny24_h16', 14.222222222222221, 16.417324255626927, 15.434311172376834),
)
SELECT * FROM b;

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM','m22809 XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE production backend', 22809, 3,
 $J${
 "anchor": "XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE",
 "mission": 22809,
 "level": 3,
 "impl_files": [
  "src/tensorlbm/porous_media.py",
  "src/tensorlbm/porous/fluid_node_table.py",
  "src/tensorlbm/porous/__init__.py",
  "src/tensorlbm/__init__.py"
 ],
 "tests": [
  "tests/test_porous_darcy_layouts.py",
  "tests/test_porous_fluid_node_table.py"
 ],
 "entry": {
  "file": "benchmarks/xval_compressed_node_table_prod/run_bench.py",
  "function": "src/tensorlbm/porous_media.py::run_single_phase_darcy",
  "library": "tensorlbm.run_single_phase_darcy",
  "cmd": "PYTHONPATH=src python3 benchmarks/xval_compressed_node_table_prod/run_bench.py --cases 12x8,24x16"
 },
 "integration_callers": [
  "src/tensorlbm/porous_media.py::run_single_phase_darcy -> tensorlbm.porous.fluid_node_table.build_fluid_node_table",
  "src/tensorlbm/porous_media.py::run_single_phase_darcy -> tensorlbm.porous.fluid_node_table.compressed_step",
  "src/tensorlbm/__init__.py:_LAZY_ATTRS['run_single_phase_darcy'] -> porous_media"
 ],
 "run_artifacts": [
  "benchmarks/xval_compressed_node_table_prod/results.json",
  "benchmarks/xval_compressed_node_table_prod/figures/profiles.png",
  "benchmarks/xval_compressed_node_table_prod/figures/wall_convention.png",
  "benchmarks/xval_compressed_node_table_prod/runs"
 ],
 "report": "docs/xval_compressed_node_table_prod_integration.md",
 "benchmark": {
  "ref_source": "analytic periodic-slab permeability k_super = (H/ny) H^2 / 12 (half-way no-slip planes); XVAL record m22754 at H=16",
  "rows": [
   {
    "case": "compressed_ny12_h8",
    "ref": 3.5555555555555554,
    "got": 3.5399999997434226,
    "err_pct": -0.43750000721623356
   },
   {
    "case": "dense_halfway_ny12_h8",
    "ref": 3.5555555555555554,
    "got": 3.539999999743447,
    "err_pct": -0.4375000072155466
   },
   {
    "case": "dense_ny12_h8",
    "ref": 3.5555555555555554,
    "got": 4.662222221752547,
    "err_pct": 31.12499998679039
   },
   {
    "case": "compressed_ny24_h16",
    "ref": 14.222222222222221,
    "got": 14.206666661051756,
    "err_pct": -0.10937503947983651
   },
   {
    "case": "dense_halfway_ny24_h16",
    "ref": 14.222222222222221,
    "got": 14.206666661051718,
    "err_pct": -0.1093750394800988
   },
   {
    "case": "dense_ny24_h16",
    "ref": 14.222222222222221,
    "got": 16.417324255626927,
    "err_pct": 15.434311172376834
   }
  ],
  "gates": {
   "G1_identity_lt_1e-8": {
    "H8": {
     "k_rel_diff": 6.899691113989759e-15,
     "profile_max_rel": 9.10674296036035e-15,
     "steps_equal": true
    },
    "H16": {
     "k_rel_diff": 2.625773836847636e-15,
     "profile_max_rel": 5.453390369989586e-15,
     "steps_equal": true
    }
   },
   "G2_physics_lt_3pct": {
    "compressed_ny12_h8": {
     "err_pct": -0.43750000721623356,
     "pass_3pct": true
    },
    "dense_halfway_ny12_h8": {
     "err_pct": -0.4375000072155466,
     "pass_3pct": true
    },
    "dense_ny12_h8": {
     "err_pct": 31.12499998679039,
     "pass_3pct": false
    },
    "compressed_ny24_h16": {
     "err_pct": -0.10937503947983651,
     "pass_3pct": true
    },
    "dense_halfway_ny24_h16": {
     "err_pct": -0.1093750394800988,
     "pass_3pct": true
    },
    "dense_ny24_h16": {
     "err_pct": 15.434311172376834,
     "pass_3pct": false
    }
   },
   "G2_all_compressed_pass": true,
   "G3_wall_convention": {
    "detail": {
     "H8": {
      "err_pct": 31.12499998679039,
      "k_eff_width": 8.756256749196336,
      "predicted_width_h_plus_1": 9,
      "k_at_h_plus_1": 5.0625
     },
     "H16": {
      "err_pct": 15.434311172376834,
      "k_eff_width": 16.784108869781416,
      "predicted_width_h_plus_1": 17,
      "k_at_h_plus_1": 17.05902777777778
     }
    },
    "err_pct_dense": {
     "H8": 31.12499998679039,
     "H16": 15.434311172376834
    },
    "ratio_lowH_over_highH": 2.016610889800872,
    "expected_ratio_for_half_cell_offset": 2.0
   }
  }
 },
 "repro_cmd": "PYTHONPATH=src python3 benchmarks/xval_compressed_node_table_prod/run_bench.py --cases 12x8,24x16"
}$J$, 'agent');
