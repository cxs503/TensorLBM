"""Exercise the public registry command against tracked repository assets."""
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkRegistryTests(unittest.TestCase):
    def registry(self, *args):
        result = subprocess.check_output(
            [sys.executable, "tools/benchmark_registry.py", *args], cwd=ROOT,
        )
        return json.loads(result)["cases"]

    def test_published_manuals_and_archive_verdicts(self):
        records = self.registry("--json")
        self.assertEqual(len({row["id"] for row in records}), len(records))
        self.assertEqual(sum(bool(row["tutorial"]) for row in records), 35)
        self.assertEqual(sum("pdf" in row["tutorial"] for row in records), 33)
        self.assertTrue(all(row["current_acceptance"] == "not_evaluated" for row in records))

    def test_sparse_and_raster_archives_have_searchable_evidence(self):
        for name, result, figure in (
            ("sparse_list_aos", "round2_d3q19/results.json", "poiseuille_profile.png"),
            ("raster_corner_class", "results.json", "fig1_fields.png"),
        ):
            with self.subTest(case=name):
                (record,) = self.registry("--case", f"xval:{name}")
                variant = record["variants"][0]
                archive = f"development/verification/{name}"
                self.assertIn(f"{archive}/{result}", variant["results"])
                self.assertIn(f"{archive}/figures/{figure}", variant["evidence"])
                self.assertIn(f"{archive}/README.md", variant["documentation"])
                self.assertEqual(record["current_acceptance"], "not_evaluated")

    def test_relocated_kovasznay_runners_and_norberg_reference_are_searchable(self):
        (kovasznay,) = self.registry("--case", "benchmarks/verified/kovasznay_2d")
        directory = "benchmarks/verified/kovasznay_2d"
        variant = next(item for item in kovasznay["variants"] if item["path"] == directory)
        self.assertIn(f"{directory}/run.py", variant["source"])
        self.assertIn(f"{directory}/reproduction/run.py", variant["source"])
        self.assertIn(f"{directory}/reproduction/RESULTS.md", variant["documentation"])

        (norberg,) = self.registry("--case", "benchmarks/pending/square_cylinder_norberg")
        self.assertIn(
            "benchmarks/pending/square_cylinder_norberg/refs/norberg1993.json",
            norberg["variants"][0]["reference_files"],
        )
        self.assertEqual(norberg["current_acceptance"], "not_evaluated")

    def test_mcmp_archives_keep_distinct_protocols_and_results(self):
        (force,) = self.registry("--case", "xval:mcmp_cross_force")
        self.assertEqual(force["current_acceptance"], "not_evaluated")
        evidence = force["variants"][0]
        for path in (
            "development/verification/mcmp_cross_force_operator/artifacts/xval_results.json",
            "development/verification/mcmp_two_component_cross_force/artifacts/summary.json",
        ):
            self.assertIn(path, evidence["results"])
        self.assertIn(
            "development/audit/xval_mcmp_cross_force_m18289/refs/operators.py",
            evidence["reference_files"],
        )
        (velocity,) = self.registry("--case", "xval:mcmp_velocity_mixing")
        self.assertEqual(velocity["current_acceptance"], "not_evaluated")
        self.assertIn(
            "benchmarks/verified/mcmp_stratified_poiseuille/analytic.py",
            velocity["variants"][0]["reference_files"],
        )

    def test_fused_streaming_archive_retains_case_ownership(self):
        case_path = "benchmarks/verified/fused_push_pull_aa_pattern"
        (record,) = self.registry("--case", case_path)
        evidence = next(variant["evidence"] for variant in record["variants"]
                        if variant["path"] == case_path)
        archive = f"{case_path}/evidence/m17839"
        for item in ("result.json", "ref/PROVENANCE.json", "run.py", "figures/index.json"):
            self.assertIn(f"{archive}/{item}", evidence)
        self.assertEqual(record["current_acceptance"], "not_evaluated")

    def test_zou_he_and_dam_slot_archives_have_searchable_evidence(self):
        for name, result, image in (
            ("zou_he_channel_2d", "artifacts/r3_steady_gate.json", "figures/fig5_drift_gate.png"),
            ("dam_break_slot_muscl2", "artifacts/result.json", "artifacts/figures/xval_dam_break_slot_profiles.png"),
        ):
            with self.subTest(case=name):
                (record,) = self.registry("--case", f"xval:{name}")
                archive = f"development/verification/{name}"
                variant = record["variants"][0]
                self.assertIn(f"{archive}/{result}", variant["results"])
                self.assertIn(f"{archive}/{image}", variant["evidence"])
                self.assertEqual(record["current_acceptance"], "not_evaluated")

    def test_free_surface_case_links_replay_and_historical_campaigns(self):
        case_path = "benchmarks/xval_oracle_minimal_d2q9_fs"
        (record,) = self.registry("--case", case_path)
        variant = record["variants"][0]
        self.assertIn(f"{case_path}/reference_study/resolution_study.py", variant["source"])
        self.assertIn(f"{case_path}/replays/2026-11-26-ci-tier.json", variant["results"])
        self.assertIn(
            "development/verification/thuerey_vof_mass_exchange/artifacts/operator_audit.json",
            variant["evidence"],
        )
        self.assertIn(
            "development/verification/mass_exchange_selector/run_xval.py",
            variant["evidence"],
        )
        self.assertEqual(record["current_acceptance"], "not_evaluated")

    def test_periodic_channel_benchmark_retains_historical_replays(self):
        directory = "benchmarks/cross_validation/periodic_channel_dpdx"
        (record,) = self.registry("--case", "xval:periodic_channel_dpdx")
        variant = record["variants"][0]
        self.assertIn(f"{directory}/run_xval_r4.py", variant["source"])
        self.assertIn(f"{directory}/out/xval_life_periodic_dpdx_r2.json", variant["results"])
        self.assertIn(f"{directory}/out/xval_life_periodic_dpdx_r4.json", variant["results"])
        self.assertIn("docs/reports/figures/xval_periodic_dpdx_r4_profiles.png", variant["evidence"])
        self.assertEqual(record["current_acceptance"], "not_evaluated")

    def test_asset_index_covers_every_tracked_benchmark_path(self):
        data = subprocess.check_output([sys.executable, "tools/benchmark_registry.py", "--asset-index"], cwd=ROOT)
        inventory = json.loads(data)
        expected = {}
        listings = subprocess.check_output(["git", "ls-files", "-s", "-z", "benchmarks"], cwd=ROOT)
        for entry in listings.split(b"\0"):
            if entry:
                metadata, path = entry.decode().split("\t", 1)
                expected[path] = metadata.split()[:2]
        assets = {row["path"]: row for row in inventory["assets"]}
        self.assertEqual(inventory["tracked_count"], len(expected))
        self.assertEqual({p: [a["git_mode"], a["git_oid"]] for p, a in assets.items()}, expected)
        unresolved = sum(not row["declared_case_ids"] and not row["directory_case_ids"]
                         for row in assets.values())
        self.assertEqual(inventory["unassigned_count"], unresolved)
        self.assertEqual(sum(inventory["unassigned_by_directory"].values()), unresolved)
        readme = assets["benchmarks/README.md"]
        self.assertFalse(readme["declared_case_ids"] or readme["directory_case_ids"])
        key = "benchmarks/p161w2_n05_oscillating_foil/artifacts/results.json"
        self.assertEqual(assets[key]["declared_case_ids"], ["campaign:p161w2_n05_oscillating_foil"])

    def test_trt_and_beam_archive_cases_preserve_reference_assets(self):
        for name, ref, result in (
            ("trt_lambda_magic", "fields.npz", "results_wall_lambda.json"),
            ("corotational_beam_newmark", "elastica_reference.py", "d3order_summary.json"),
        ):
            with self.subTest(case=name):
                (record,) = self.registry("--case", f"xval:{name}")
                variant = record["variants"][0]
                archive = f"development/verification/{name}"
                self.assertIn(f"{archive}/{ref}", variant["reference_files"])
                self.assertIn(f"{archive}/{result}", variant["results"])
                self.assertIn(f"{archive}/README.md", variant["documentation"])
                self.assertEqual(record["current_acceptance"], "not_evaluated")
                if name == "trt_lambda_magic":
                    reference_fields = [
                        item for item in variant["reference_files"] if item.endswith(".dat")
                    ]
                    self.assertEqual(len(reference_fields), 74)

    def test_yaml_xml_pipeline_links_its_analytic_replay(self):
        directory = "benchmarks/xval_yml_xml_pipeline"
        report = "docs/xval/XVAL-LBM-YML-DRIVEN-XML-TEMPLATE-PIPELINE.md"
        (case,) = self.registry("--case", "campaign:xval_yml_xml_pipeline")
        self.assertEqual(case["current_acceptance"], "not_evaluated")
        (variant,) = case["variants"]
        self.assertIn(f"{directory}/run_all.py", variant["source"])
        self.assertIn(report, variant["reference_files"])
        (replay_path,) = variant["results"]
        replay = json.loads((ROOT / replay_path).read_text(encoding="utf-8"))
        runner = replay["sources"]["relocated_runner"]
        self.assertEqual(runner["path"], f"{directory}/run_all.py")
        self.assertEqual(
            hashlib.sha256((ROOT / runner["path"]).read_bytes()).hexdigest(),
            runner["sha256"],
        )
        self.assertIn(f"](../../{directory}/README.md)", (ROOT / report).read_text())

    def test_cac_gravity_archives_keep_failed_result_distinct_from_report(self):
        (case,) = self.registry("--case", "xval:cac_gravity_historical")
        self.assertEqual(case["current_acceptance"], "not_evaluated")
        (variant,) = case["variants"]
        archive = "development/verification/cac_gravity"
        self.assertEqual(variant["results"], [f"{archive}/m23012/results.json"])
        self.assertIn(f"{archive}/m23012/PREREG.json", variant["config"])
        self.assertIn(f"{archive}/m23170/PHASE_A_REPORT.md", variant["documentation"])
        self.assertTrue(all(path.startswith("development/legacy/") for path in variant["source"]))
        result = json.loads((ROOT / variant["results"][0]).read_text())
        self.assertFalse(result["verdict"]["overall_pass"])

    def test_case_lookup_links_nested_evidence_to_the_owning_variant(self):
        path = "benchmarks/verified/cavity/re400"
        (case,) = self.registry("--case", path)
        variants = {variant["path"]: variant for variant in case["variants"]}
        evidence = f"{path}/evidence/sdaa/sdaa_cavity_evidence.json"
        plot = f"{path}/evidence/sdaa/sdaa_cavity_re400_verification.png"
        self.assertIn(evidence, variants[path]["evidence"])
        self.assertIn(plot, variants[path]["evidence"])
        self.assertNotIn(evidence, variants[path]["results"])
        for directory, variant in variants.items():
            self.assertTrue(all(asset.startswith(f"{directory}/evidence/") for asset in variant["evidence"]))
            if directory != path:
                self.assertNotIn(evidence, variant["evidence"])
        self.assertEqual(case["current_acceptance"], "not_evaluated")

    def test_eos_maxwell_retains_related_peng_robinson_archive(self):
        (record,) = self.registry("--case", "benchmarks/eos_maxwell")
        variant = record["variants"][0]
        archive = "development/verification/peng-robinson-pseudopotential"
        self.assertIn(f"{archive}/prereg.md", variant["reference_files"])
        self.assertIn(f"{archive}/artifacts/results.json", variant["evidence"])
        self.assertIn(f"{archive}/artifacts/snapshot.npz", variant["evidence"])
        self.assertIn("benchmarks/eos_maxwell/out/results.json", variant["results"])
        self.assertEqual(record["current_acceptance"], "not_evaluated")

    def test_standalone_ritter_keeps_its_gate_separate_from_matrix_case(self):
        (campaign,) = self.registry("--case", "campaign:dam_break_ritter")
        (matrix,) = self.registry("--case", "matrix:dam_break_ritter_ci")
        self.assertNotEqual(campaign["id"], matrix["id"])
        self.assertEqual(campaign["current_acceptance"], "not_evaluated")
        self.assertIn("src/tensorlbm/shallow_water.py", campaign["variants"][0]["source"])
        self.assertIn("benchmarks/dam_break_ritter/replays/2026-11-21.json", campaign["variants"][0]["results"])

    def test_shan_chen_edm_rounds_keep_separate_historical_evidence(self):
        (case,) = self.registry("--case", "xval:shan_chen_edm_lycett_luo")
        self.assertEqual(case["current_acceptance"], "not_evaluated")
        variant = case["variants"][0]
        base = "development/verification/shan_chen_edm_lycett_luo"
        self.assertIn(f"{base}/round1/artifacts/results.json", variant["results"])
        self.assertIn(f"{base}/round2/artifacts/compact.json", variant["results"])
        self.assertEqual(sum(x.endswith(".png") for x in variant["evidence"]), 6)
        self.assertIn("development/audit/m21400/AUDIT_M21378.md",
                      variant["documentation"])

    def test_single_drop_case_separates_formal_and_exploratory_histories(self):
        case_id = "benchmarks/pending/droplet_impact_3d_pool"
        (case,) = self.registry("--case", case_id)
        self.assertEqual(case["id"], case_id)
        self.assertEqual(case["current_acceptance"], "not_evaluated")
        self.assertEqual(case["tutorial"], {})
        variant, = case["variants"]
        self.assertEqual(variant["archive_bucket"], "pending")
        directory = "benchmarks/water_entry/single_drop"
        self.assertIn(f"{directory}/protocol.json", variant["config"])
        self.assertIn(f"{directory}/fig5_experiment.csv", variant["reference_files"])
        self.assertEqual(variant["results"], [
            f"{case_id}/result.json", "docs/audits/spherical_entry_20261118/histories.zip",
        ])
        archive = "development/audit/drop_entry/cmb5_hydrorr_stability"
        self.assertIn(f"{archive}/cmb5_verdict.json", variant["evidence"])
        self.assertIn(f"{archive}/arms/cmb5_parity/result.json.gz", variant["evidence"])

    def test_contact_angle_case_retains_all_archived_round_dependencies(self):
        (record,) = self.registry("--case", "xval:contact_angle_geodesic")
        variant = record["variants"][0]
        root = "development/verification/contact-angle"
        self.assertIn(f"{root}/round-2/predeclared.json", variant["reference_files"])
        self.assertIn(f"{root}/round-3/summary_r3.json", variant["reference_files"])
        self.assertIn(f"{root}/round-3/runs_r3/geometric_ghost__th070.0__baseline__xi3.npz", variant["evidence"])
        self.assertIn(f"{root}/wall-orientation/summary.json", variant["evidence"])
        self.assertIn(f"{root}/round-5/results_r5.json", variant["results"])
        self.assertEqual(record["current_acceptance"], "not_evaluated")

    def test_find_archived_assets_by_path(self):
        result = subprocess.run(
            [sys.executable, "tools/benchmark_registry.py", "--find-asset", "M17444/AKAI2018_REPRO"],
            cwd=ROOT, capture_output=True, check=True,
        )
        paths = json.loads(result.stdout)["paths"]
        self.assertIn("development/audit/m17444/akai2018_repro/analytic_verification.json", paths)
        self.assertEqual(paths, sorted(set(paths)))
        self.assertTrue(all((ROOT / path).is_file() for path in paths))

    def test_ibm_epsilon_gram_archive_keeps_original_manifest_and_derivatives(self):
        (case,) = self.registry("--case", "xval:ibm_epsilon_gram")
        self.assertEqual(case["current_acceptance"], "not_evaluated")
        variant = case["variants"][0]
        archive = "development/verification/ibm_epsilon_gram"
        self.assertIn(f"{archive}/results.json", variant["results"])
        self.assertIn(f"{archive}/manifest.json", variant["reference_files"])
        for name in ("fig1_roundtrip.png", "fig2_epsilon_cost.png",
                     "fig3_cd.png", "fig4_convergence.png"):
            self.assertIn(f"{archive}/derived_figures/{name}", variant["evidence"])

    def test_matrix_lookup_uses_exact_artifacts_and_preserves_verdict_scope(self):
        (ship,) = self.registry("--case", "matrix:ship_resistance_suboff")
        self.assertEqual(ship["id"], "suboff_re1000")
        entry, = ship["validation_matrix"]
        self.assertEqual(entry["definition"]["measure"]["artifact"], entry["archived_comparison"]["artifact"])
        self.assertTrue(entry["references"])
        self.assertEqual(ship["current_acceptance"], "not_evaluated")

        records = self.registry("--json")
        entries = [e for row in records for e in row["validation_matrix"]]
        matrix = json.loads((ROOT / "benchmarks/validation_matrix/matrix.json").read_text())
        self.assertCountEqual([e["id"] for e in entries], [c["id"] for c in matrix["cases"]])
        report = json.loads((ROOT / "benchmarks/validation_matrix/BENCHMARK_REF_VS_GOT.json").read_text())
        reported = {row["case"]: row["status"] for row in report["cases"]}
        for entry in entries:
            comparison = entry["archived_comparison"]
            self.assertEqual(None if comparison is None else comparison["status"], reported.get(entry["id"]))

    def test_phasefield_poiseuille_uses_existing_case_and_decoded_figures(self):
        (case,) = self.registry("--case", "campaign:phasefield_mcmp_poiseuille")
        self.assertEqual(case["current_acceptance"], "not_evaluated")
        base = "benchmarks/phasefield_mcmp_poiseuille"
        variant = case["variants"][0]
        self.assertIn(f"{base}/reference/round1.py", variant["source"])
        self.assertIn(f"{base}/evidence/m18190/xval_poiseuille_results.json",
                      variant["results"])
        self.assertEqual(sum(p.endswith(".png") for p in variant["evidence"]), 4)

    def test_xval_registration_preserves_archived_and_replayed_results(self):
        (case,) = self.registry("--case", "xval:ludwig_fe_stress")
        self.assertEqual(case["current_acceptance"], "not_evaluated")
        variant, = case["variants"]
        canonical = "benchmarks/cross_validation/ludwig_fe_stress"
        self.assertEqual(variant["path"], canonical)
        self.assertEqual(self.registry("--case", canonical), [case])
        self.assertEqual(self.registry("--case", "xval/ludwig_fe_stress"), [case])
        self.assertEqual(case["cross_validation"]["criteria"]["threshold"], 0.15)
        original, replay, ledger = (ROOT / path for path in variant["results"])
        evidence = json.loads(ledger.read_text())
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),
                         evidence["compatibility"]["archive_sha256"])
        self.assertEqual(hashlib.sha256(replay.read_bytes()).hexdigest(),
                         evidence["run"]["result_sha256"])
        summary = json.loads(replay.read_text())["summary"]
        self.assertTrue(summary["pass_15pct"])
        self.assertGreater(summary["worst_force_diff_rel"], 1.0)

    def test_young_laplace_and_two_pdf_archives_are_searchable(self):
        for name, result in (
            ("young_laplace_analytic", "artifacts/summary.json"),
            ("two_pdf_colorgradient", "artifacts/laplace_log.json"),
        ):
            with self.subTest(case=name):
                (record,) = self.registry("--case", f"xval:{name}")
                variant = record["variants"][0]
                archive = f"development/verification/{name}"
                self.assertIn(f"{archive}/{result}", variant["results"])
                self.assertIn(f"{archive}/REPORT.md", variant["documentation"])
                self.assertEqual(record["current_acceptance"], "not_evaluated")
                if name == "young_laplace_analytic":
                    for image in ("fig_hero.png", "fig_laplace.png", "fig_sensitivity.png"):
                        self.assertIn(f"{archive}/artifacts/{image}", variant["evidence"])

    def test_alias_retains_all_variants(self):
        path = "benchmarks/verified/poiseuille_3d_duct/duct_ar05"
        (record,) = self.registry("--case", path)
        self.assertEqual(record["id"], "poiseuille_3d_duct")
        self.assertEqual({variant["path"] for variant in record["variants"]},
                         {path, path.rsplit("/", 1)[0]})


    def test_cavity_three_way_archive_keeps_formal_case_separate(self):
        (case,) = self.registry("--case", "xval:cavity_three_way")
        self.assertEqual(case["current_acceptance"], "not_evaluated")
        archive = "development/verification/cavity_three_way"
        variant, = case["variants"]
        self.assertEqual(variant["results"], [f"{archive}/artifacts/comparison.json"])
        self.assertIn(f"{archive}/REPORT.md", variant["documentation"])
        self.assertIn(f"{archive}/mpi4py_shim/mpi4py/MPI.py", variant["evidence"])
        self.assertNotIn(f"{archive}/mpi4py_shim/mpi4py/MPI.py", variant["source"])
        original = json.loads((ROOT / variant["results"][0]).read_text())
        completed, = [spec for spec in original["specs"] if spec["W"] == 64 and spec["re"] == 100]
        self.assertFalse(completed["verdict"]["inter_tool_pass"])
        self.assertEqual(sum(spec["verdict"]["inter_tool_pass"] is None for spec in original["specs"]), 3)
        (formal,) = self.registry("--case", "benchmarks/verified/cavity/re100")
        self.assertNotEqual(formal["id"], case["id"])
        self.assertEqual(formal["current_acceptance"], "not_evaluated")


if __name__ == "__main__":
    unittest.main()
