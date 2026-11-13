#!/usr/bin/env python3
"""Emit the machine-checkable evidence bundle for the maturity gate.

The platform's ``maturity_gate.py`` re-checks each ``tensor_maturity_verdicts.evidence``
key by machine (files must exist **repo-relative**, impl must be non-placeholder,
callers must grep-hit inside the repo, benchmark must carry ref/got numbers).
This script derives that bundle directly from the run artefact, so the evidence can
never drift from the numbers:

    python xval/cupy_dropin_no_kernel_fusion/emit_evidence.py

Writes ``xval/cupy_dropin_no_kernel_fusion/EVIDENCE.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REL = lambda p: str(Path(p).relative_to(REPO))  # noqa: E731


def main() -> int:
    run = json.loads((HERE / "artifacts" / "run_main.json").read_text(encoding="utf-8"))
    arms = run["main"]["arms"]
    nb, tb, ob = arms["tensorlbm_numpy"], arms["tensorlbm_torch"], arms["oss_exec"]
    agr = run["main"]["backend_agreement"]
    oss_agr = run["main"]["oss_agreement"]
    sharp = next(s for s in run["sensitivity"]
                 if s["param"] == "init_profile" and s["value"] == "sharp")

    evidence = {
        "anchor": "XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION",
        "scope": "m17979 XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION",
        "repo": "TensorLBM",
        "claimed_level": 5,
        "tests": [
            "tests/test_free_energy_soy_cupy_dropin.py::test_static_droplet_obeys_young_laplace_law_within_15_percent",
            "tests/test_free_energy_soy_cupy_dropin.py::test_backends_agree_to_machine_precision_after_200_steps",
            "tests/test_free_energy_soy_cupy_dropin.py::test_backends_report_identical_array_op_counts",
            "tests/test_free_energy_soy_cupy_dropin.py::test_isotropic_gradient_is_exact_for_quadratic_field",
            "tests/test_free_energy_soy_cupy_dropin.py::test_isotropic_laplacian_is_exact_for_quadratic_field",
            "tests/test_free_energy_soy_cupy_dropin.py::test_equilibrium_populations_carry_the_expected_moments",
            "tests/test_free_energy_soy_cupy_dropin.py::test_momentum_definition_matches_half_force_correction",
            "tests/test_free_energy_soy_cupy_dropin.py::test_order_parameter_is_conserved_by_the_periodic_update",
            "tests/test_free_energy_soy_cupy_dropin.py::test_sharp_initial_profile_is_a_slow_transient_not_an_equilibrium",
            "tests/test_free_energy_soy_cupy_dropin.py::test_pressure_form_switch_changes_the_result_but_stays_acceptable",
            "tests/test_free_energy_soy_cupy_dropin.py::test_implementation_contains_no_fused_or_custom_kernels",
            "tests/test_free_energy_soy_cupy_dropin.py::test_solver_layer_caller_returns_laplace_diagnostics",
            "tests/test_free_energy_soy_cupy_dropin.py::test_oss_reference_provenance_and_agreement",
        ],
        "tests_note": "14 tests in one module; run `python -m pytest tests/test_free_energy_soy_cupy_dropin.py -q`",
        "impl_files": [
            "src/tensorlbm/multiphase_soy_dropin.py",
            "src/tensorlbm/multiphase_static_droplet.py",
            "xval/cupy_dropin_no_kernel_fusion/oss_adapter.py",
        ],
        "entry": "xval/cupy_dropin_no_kernel_fusion/run_xval.py",
        "entry_cli": "python xval/cupy_dropin_no_kernel_fusion/run_xval.py --steps 2000",
        "run_artifacts": [
            "xval/cupy_dropin_no_kernel_fusion/artifacts/run_main.json",
            "xval/cupy_dropin_no_kernel_fusion/artifacts/dp_history.csv",
            "xval/cupy_dropin_no_kernel_fusion/artifacts/report_payload.json",
            "xval/cupy_dropin_no_kernel_fusion/figures/fig1_laplace_convergence.png",
            "xval/cupy_dropin_no_kernel_fusion/figures/fig2_fields.png",
            "xval/cupy_dropin_no_kernel_fusion/figures/fig3_error_vs_radius.png",
            "xval/cupy_dropin_no_kernel_fusion/figures/fig4_sensitivity.png",
            "xval/cupy_dropin_no_kernel_fusion/figures/fig5_cost.png",
            "docs/XVAL_LBM_CUPY_DROPIN_NO_KERNEL_FUSION.md",
        ],
        "integration_callers": [
            "tensorlbm.multiphase_soy_dropin",
            "tensorlbm.multiphase_static_droplet:soy_free_energy_droplet",
            "tests/test_free_energy_soy_cupy_dropin.py:test_solver_layer_caller_returns_laplace_diagnostics",
        ],
        "integration_note": (
            "solver-layer caller: src/tensorlbm/multiphase_static_droplet.py "
            "`soy_free_energy_droplet()` imports tensorlbm.multiphase_soy_dropin absolutely "
            "and is exercised by tests::test_solver_layer_caller_returns_laplace_diagnostics; "
            "the XVAL entry also drives the same core."
        ),
        "benchmark": {
            "reference": "Young-Laplace dp = sigma/R (sigma = 0.01, R = 15 -> 6.666667e-04), "
                         "plus the executed OSS implementation (ebinan92/Fingering_dynamics@150a573)",
            "got_vs_ref": {
                "laplace_dp_tensorlbm_numpy": {
                    "got": nb["dp"], "ref": run["main"]["reference_dp"],
                    "err_pct": nb["err_pct"], "tol_pct": 15.0,
                    "pass": abs(nb["err_pct"]) < 15.0},
                "laplace_dp_tensorlbm_torch": {
                    "got": tb["dp"], "ref": run["main"]["reference_dp"],
                    "err_pct": tb["err_pct"], "tol_pct": 15.0,
                    "pass": abs(tb["err_pct"]) < 15.0},
                "laplace_dp_oss_exec": {
                    "got": ob["dp"], "ref": run["main"]["reference_dp"],
                    "err_pct": ob["err_pct"], "tol_pct": 15.0,
                    "pass": abs(ob["err_pct"]) < 15.0},
                "tensorlbm_vs_executed_oss": {
                    "got": oss_agr["dp_tensorlbm"], "ref": oss_agr["dp_oss"],
                    "err_pct": oss_agr["rel_diff_pct"], "tol_pct": 15.0,
                    "pass": oss_agr["rel_diff_pct"] < 15.0},
                "backend_dropin_equivalence": {
                    "got": agr["dp_torch"], "ref": agr["dp_numpy"],
                    "rel_diff": agr["rel_diff"], "tol_rel": 1e-06, "pass": agr["pass"]},
                "sigma_capillary_energy_route": {
                    "got": nb["sigma_from_capillary_energy"], "ref": 0.01,
                    "err_pct": nb["sigma_rel_err_pct"], "tol_pct": 15.0,
                    "pass": abs(nb["sigma_rel_err_pct"]) < 15.0},
            },
            "known_failure_mode": {
                "case": "init_profile=sharp (tanh(2(r-R)/xi), non-equilibrium initial width)",
                "err_pct": sharp["err_pct"], "tol_pct": 15.0, "pass": False,
                "note": "deliberate negative control; documented in report section 5.2"},
            "convergence_vs_radius": [
                {"R": c["radius"], "err_pct": c["err_pct"], "err_pct_oss": c.get("err_pct_oss")}
                for c in run["convergence"]],
            "transient_band_last_third": {
                "err_pct_min": min(h["err_pct"] for h in nb["history"] if h["step"] >= run["steps"] * 2 // 3),
                "err_pct_max": max(h["err_pct"] for h in nb["history"] if h["step"] >= run["steps"] * 2 // 3),
            },
            "host": run["host"],
        },
        "repro_cmd": (
            "python xval/cupy_dropin_no_kernel_fusion/run_xval.py --steps 2000 "
            "&& python xval/cupy_dropin_no_kernel_fusion/make_report.py "
            "&& python -m pytest tests/test_free_energy_soy_cupy_dropin.py -q"
        ),
        # the gate checks that every listed path *exists*: point at the generated
        # report, which carries the citations and the recorded OSS provenance
        "external_anchor": ["docs/XVAL_LBM_CUPY_DROPIN_NO_KERNEL_FUSION.md"],
        "external_anchor_note": (
            "OSS ebinan92/Fingering_dynamics @150a573 lattice_boltzmann/fingering_periodic.py "
            "(class Compute + stream executed read-only at run time, sha256 in run_main.json) "
            "+ Swift, Osborn & Yeomans, 'Lattice Boltzmann simulation of nonideal fluids', "
            "Phys. Rev. Lett. 75 (1995) 830 (SOY free-energy model) "
            "+ Young-Laplace law dp = sigma/R"
        ),
        "not_run": [
            "GPU / CuPy execution (host has no CUDA device; the practice is validated via array-namespace equivalence)",
            "OSS Zou-He inlet/outlet and half-way bounce-back walls (periodic domain only)",
            "fingering / viscous-fingering dynamics at the OSS operating point (380x380, MAX_T=1000): not needed for the Laplace cross-validation",
        ],
        "friction": [
            "host `import tensorlbm` resolves to an editable install of another worktree; tests purge and re-import this worktree's src",
            "OSS checkout lives in /tmp (ephemeral): the adapter degrades to NOT_RUN with a reason when it is missing",
            "round-0 artefacts were tracked but the numbers were not reproducible from a single canonical entry (fixed: run_main.json is the one artefact)",
        ],
    }
    out = HERE / "EVIDENCE.json"
    out.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"wrote {REL(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())