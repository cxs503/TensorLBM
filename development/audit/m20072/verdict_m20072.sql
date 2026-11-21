-- Adversarial review verdict for m20072 (GAP-DIFFLBM-3D-COST)
-- Reviewed by: board-audit (m20104)
-- Date: 2026-11-14

BEGIN;

-- Register the benchmark case that was MISSING from tensor_benchmark_cases
-- (This is a gap found during audit: the anchor had no machine-adjudicable row)
DELETE FROM tensor_benchmark_cases WHERE anchor_slug = 'GAP-DIFFLBM-3D-COST';

INSERT INTO tensor_benchmark_cases
  (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
VALUES (
  'GAP-DIFFLBM-3D-COST',
  '3d_d3q19_differentiable_inversion_cost_envelope',
  'Closed-form Taylor-Green energy decay E(t)=E0*exp(-4*nu*k^2*t) and shear wave amplitude A(t)=A0*exp(-nu*k^2*t); '
  'finite-difference 5-point gradient probe (h=1e-3) as referee for AD gradient correctness; '
  'VmHWM (peak RSS) from /proc/self/status in independent subprocess per engine; '
  'D3Q19 BGK, c_s^2=1/3, tau=0.55 (nu=0.0167), amplitude=0.02 (TG) / 0.05 (shear), periodic box, float64 for gradient裁判.',
  'Gradient correctness: rel_err(AD_vs_FD_5pt) <= 1e-6 (fp64); '
  'Physics fidelity: rel_err(E_measured vs E_analytic) < 3% and rel_err(A_measured vs A_analytic) < 3%; '
  'Memory envelope: peak_rss scales O(1) with T (not O(T)); '
  'Reproducibility: bench_difflbm3d_envelope.py --stage grad/envelope/physics produces identical JSON.',
  'gradient rel_err <= 1e-6 (fp64); physics rel_err < 3% (all cases); memory O(1) in T; '
  'auto-check: python3 -m pytest tests/test_diff3d_envelope.py -q (7 passed); '
  'repro: python3 scripts/bench_difflbm3d_envelope.py --stage grad --n 32 --steps 8 --dtype float64 --threads 6 --seg-len 1',
  'TensorLBM',
  'verified'
);

-- Verdict: L4 (基准验证) — not L5 because 256^3 end-to-end is still queued (cj2473/cj2474)
-- and the benchmark case was NOT pre-registered (gap found during this audit).
INSERT INTO tensor_maturity_verdicts
  (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm20072 GAP-DIFFLBM-3D-COST',
  20104,
  4,
  '{
    "impl_files": [
      "src/tensorlbm/diff3d_envelope.py (SegmentStore, _SegmentedRollout, segmented_rollout, divergence_free_ic, shear_wave_ic)",
      "src/tensorlbm/autograd_path.py (differentiable_step chain)",
      "src/tensorlbm/d3q19.py (equilibrium3d)"
    ],
    "entry": "scripts/bench_difflbm3d_envelope.py (stage_grad/stage_envelope/stage_physics/stage_inversion)",
    "run_artifacts": [
      "artifacts/difflbm3d_envelope/n32_fp64_grad.json (rel_err_tau=1.36e-7, criterion_grad_1e-6=true)",
      "artifacts/difflbm3d_envelope/n96_fp32_env_segmented.json (peak_rss=2.231 GiB)",
      "artifacts/difflbm3d_envelope/n128_fp32_env_segmented.json (peak_rss=3.958 GiB)",
      "artifacts/difflbm3d_envelope/physics_n32_n64.json (TG 32:1.27%, 64:0.36%; shear 32:0.84%, 64:0.20%)"
    ],
    "tests": [
      "tests/test_diff3d_envelope.py (7 passed): segment_store_roundtrip, segmented_gradient_equals_full_graph, segmented_gradient_matches_finite_difference, divergence_free_ic, taylor_green_decay, shear_wave_decay, temporary_store_cleanup"
    ],
    "integration_callers": [
      "scripts/figures_difflbm3d_envelope.py (generates envelope_memory_time.png, physics_anchor.png)"
    ],
    "benchmark": {
      "registered_now": true,
      "was_missing": true,
      "anchor_slug": "GAP-DIFFLBM-3D-COST",
      "metrics_verified": {
        "gradient_rel_err_fp64": "1.36e-7 <= 1e-6 PASS",
        "gradient_u0_rel_err": "6.20e-14 <= 1e-6 PASS",
        "segmented_vs_full": "0.0 PASS",
        "physics_tg_n32": "1.27% < 3% PASS",
        "physics_tg_n64": "0.36% < 3% PASS",
        "physics_shear_n32": "0.84% < 3% PASS",
        "physics_shear_n64": "0.20% < 3% PASS"
      }
    },
    "repro_cmd": "python3 scripts/bench_difflbm3d_envelope.py --stage grad --n 32 --steps 8 --dtype float64 --threads 6 --seg-len 1 --out artifacts/difflbm3d_envelope/n32_fp64_grad.json",
    "external_anchor": "commit 0dc707565, torch 2.7.1, loongarch64 128-core"
  }',
  'board-audit'
);

COMMIT;
