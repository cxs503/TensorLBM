#!/usr/bin/env python3
"""XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN -- machine-refereed run.

Arms (all thresholds come from ``predeclared.json``, none are re-typed here):

  A1  channel Poiseuille at ny = 16 / 32 / 48 vs closed-form u_max + profile
  A2  independent example-main control (torch.roll streaming) vs A1
  A3  Taylor-Green energy decay rate vs 4 nu k^2
  A4  determinism (same spec twice) + negative control (perturbed spec)
  A5  anti-pattern audit of the repository ``examples/`` tree
  A6  detector discrimination control on planted fixtures

Arms can be run selectively (``--arms a1_16,a1_48``) and are merged into
``summary.json``; the machine referee (``checks``) is recomputed after every
invocation and only reports ``all_pass`` once every arm it needs is present.
That keeps a long run restartable without losing completed arms.

Usage::

    PYTHONPATH=src python3 xval/XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN/run_xval.py \
        --repo-root . --out xval/XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN/run_artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Callable

HERE = os.path.dirname(os.path.abspath(__file__))
ALL_ARMS = ("a1_16", "a1_32", "a1_48", "a2", "a3", "a4", "a5", "a6")
CHANNEL_STEPS = {16: 8000, 32: 12000, 48: 15000}
CONTROL_NY = 32


def _json_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=float)


def _load_thresholds() -> dict[str, Any]:
    with open(os.path.join(HERE, "predeclared.json"), encoding="utf-8") as fh:
        return json.load(fh)["predeclared_thresholds"]


class Runner:
    def __init__(self, repo_root: str, out: str) -> None:
        import torch

        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
        self.repo_root = os.path.abspath(repo_root)
        self.out = out
        self.thr = _load_thresholds()
        self.summary_path = os.path.join(out, "summary.json")
        if os.path.exists(self.summary_path):
            with open(self.summary_path, encoding="utf-8") as fh:
                self.summary = json.load(fh)
        else:
            self.summary = {
                "xval": "XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN",
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "thresholds": self.thr,
                "arms": {},
                "checks": {},
            }
        self.summary["thresholds"] = self.thr

    # -- artifact helpers -------------------------------------------------
    def write(self, name: str, obj: Any) -> None:
        _json_dump(obj, os.path.join(self.out, name))

    # -- arms -------------------------------------------------------------
    def arm_a1(self, ny: int) -> None:
        from tensorlbm.example_harness import get_case, record_digest, run_case

        spec = get_case("channel_poiseuille", ny=ny, steps=CHANNEL_STEPS[ny])
        rec = run_case(spec, keep_history=False)
        rec["record_digest"] = record_digest(rec)
        rec["elapsed_s"] = rec["kpis"]["elapsed_s"]
        self.write(f"arm_a1_channel_ny{ny}.json", rec)
        self.summary["arms"]["A1_channel"] = self.summary["arms"].get("A1_channel", {})
        self.summary["arms"]["A1_channel"][str(ny)] = {
            "u_max": rec["kpis"]["u_max"],
            "u_max_analytic": rec["analytic"]["u_max"],
            "u_max_rel_err": rec["kpis"]["relative_error_vs_analytic"],
            "profile_rel_l2": rec["kpis"]["profile_rel_l2"],
            "profile_interior_rel_l2": rec["kpis"]["profile_interior_rel_l2"],
            "mass_drift": rec["kpis"]["mass_drift"],
            "converged_at": rec["kpis"]["converged_at"],
            "steps_run": rec["kpis"]["steps_run"],
            "rho_min": rec["kpis"]["rho_min"],
            "rho_max": rec["kpis"]["rho_max"],
            "spec_digest": rec["provenance"]["spec_digest"],
            "record_digest": rec["record_digest"],
            "profile": rec.get("profile"),
        }

    def arm_a2(self) -> None:
        ref_path = os.path.join(HERE, "reference_main_channel.py")
        ref_json = os.path.join(self.out, f"arm_a2_reference_main_ny{CONTROL_NY}.json")
        env = dict(os.environ)
        src_dir = os.path.join(self.repo_root, "src")
        env["PYTHONPATH"] = src_dir + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")
        subprocess.run(
            [
                sys.executable,
                ref_path,
                "--ny",
                str(CONTROL_NY),
                "--steps",
                str(CHANNEL_STEPS[CONTROL_NY]),
                "--out",
                ref_json,
            ],
            check=True,
            env=env,
            stdout=subprocess.DEVNULL,
        )
        with open(ref_json, encoding="utf-8") as fh:
            control = json.load(fh)
        harness = self.summary["arms"].get("A1_channel", {}).get(str(CONTROL_NY))
        if harness is None:
            raise SystemExit(
                f"A2 needs arm a1_{CONTROL_NY} first (run --arms a1_{CONTROL_NY},a2)"
            )
        rel_umax = abs(harness["u_max"] - control["u_max"]) / abs(control["u_max"])
        prof_h = harness.get("profile")
        prof_c = control["profile"]
        rel_prof = None
        if prof_h and len(prof_h) == len(prof_c):
            num = sum((a - b) ** 2 for a, b in zip(prof_h, prof_c)) ** 0.5
            den = sum(b * b for b in prof_c) ** 0.5
            rel_prof = num / den
        self.summary["arms"]["A2_control"] = {
            "ny": CONTROL_NY,
            "control_u_max": control["u_max"],
            "control_u_max_rel_err_vs_analytic": control["u_max_rel_err_vs_analytic"],
            "control_mass_drift": control["mass_drift"],
            "control_streaming": control["streaming"],
            "harness_u_max": harness["u_max"],
            "harness_vs_control_rel": rel_umax,
            "harness_vs_control_profile_rel_l2": rel_prof,
        }

    def arm_a3(self) -> None:
        from tensorlbm.example_harness import get_case, record_digest, run_case

        rec = run_case(get_case("taylor_green_2d", ny=32, nx=32, steps=800, sample_every=50))
        rec["record_digest"] = record_digest(rec)
        self.write("arm_a3_taylor_green_32x32.json", rec)
        self.summary["arms"]["A3_taylor_green"] = {
            "decay_rate_measured": rec["kpis"]["decay_rate_measured"],
            "decay_rate_analytic": rec["analytic"]["decay_rate"],
            "velocity_decay_rate_analytic": rec["analytic"]["velocity_decay_rate"],
            "rel_err": rec["kpis"]["relative_error_vs_analytic"],
            "energy_initial": rec["history"][0]["kinetic_energy"] if rec["history"] else None,
            "energy_final": rec["history"][-1]["kinetic_energy"] if rec["history"] else None,
            "record_digest": rec["record_digest"],
            "history": rec["history"],
        }

    def arm_a4(self) -> None:
        from tensorlbm.example_harness import get_case, record_digest, run_case

        spec_a = get_case("channel_poiseuille", ny=24, steps=1500)
        rec_a1 = run_case(spec_a, keep_history=False)
        rec_a2 = run_case(spec_a, keep_history=False)
        rec_b = run_case(get_case("channel_poiseuille", ny=24, steps=1500, body_force=2e-5),
                         keep_history=False)
        d_a1, d_a2, d_b = (record_digest(r) for r in (rec_a1, rec_a2, rec_b))
        payload = {
            "digest_run1": d_a1,
            "digest_run2": d_a2,
            "digest_perturbed_spec": d_b,
            "identical_rerun": d_a1 == d_a2,
            "perturbation_changes_digest": d_a1 != d_b,
            "field_digest_run1": rec_a1["kpis"]["field_digest"],
            "field_digest_run2": rec_a2["kpis"]["field_digest"],
            "field_identical": rec_a1["kpis"]["field_digest"] == rec_a2["kpis"]["field_digest"],
            "u_max_run1": rec_a1["kpis"]["u_max"],
            "u_max_perturbed": rec_b["kpis"]["u_max"],
            "perturbed_u_max_rel_change": abs(
                rec_b["kpis"]["u_max"] - rec_a1["kpis"]["u_max"]
            )
            / rec_a1["kpis"]["u_max"],
            "spec_digest_run1": rec_a1["provenance"]["spec_digest"],
            "spec_digest_perturbed": rec_b["provenance"]["spec_digest"],
        }
        self.write("arm_a4_determinism.json", payload)
        self.summary["arms"]["A4_determinism"] = payload

    def arm_a5(self) -> None:
        from tensorlbm.audit.example_antipattern import audit_tree

        audit = audit_tree(os.path.join(self.repo_root, "examples"))
        self.write("arm_a5_examples_audit.json", audit)
        self.summary["arms"]["A5_examples_audit"] = {
            "root": audit["root"],
            "python_files": audit["python_files"],
            "total_loc": audit["total_loc"],
            "M1": audit["M1_standalone_mains"],
            "M2": audit["M2_duplication"],
            "M3": audit["M3_thin_mains"],
        }

    def arm_a6(self) -> None:
        from tensorlbm.audit.example_antipattern import audit_tree

        ap_fix = audit_tree(os.path.join(HERE, "fixtures", "antipattern"))
        co_fix = audit_tree(os.path.join(HERE, "fixtures", "consolidated"))
        self.write("arm_a6_detector_control.json", {"antipattern": ap_fix, "consolidated": co_fix})
        self.summary["arms"]["A6_detector_control"] = {
            "antipattern_dup_loc_ratio": ap_fix["M2_duplication"]["dup_loc_ratio"],
            "consolidated_dup_loc_ratio": co_fix["M2_duplication"]["dup_loc_ratio"],
            "dup_ratio_gap": ap_fix["M2_duplication"]["dup_loc_ratio"]
            - co_fix["M2_duplication"]["dup_loc_ratio"],
            "driver_loc_ratio_gap": ap_fix["M2_duplication"]["dup_driver_loc_ratio"]
            - co_fix["M2_duplication"]["dup_driver_loc_ratio"],
            "antipattern_driver_loc_ratio": ap_fix["M2_duplication"]["dup_driver_loc_ratio"],
            "consolidated_driver_loc_ratio": co_fix["M2_duplication"]["dup_driver_loc_ratio"],
            "antipattern_dup_driver_blocks": ap_fix["M2_duplication"]["n_dup_driver_blocks"],
            "consolidated_dup_driver_blocks": co_fix["M2_duplication"]["n_dup_driver_blocks"],
            "antipattern_thin_share": ap_fix["M3_thin_mains"]["share"],
            "consolidated_thin_share": co_fix["M3_thin_mains"]["share"],
            "antipattern_main_share": ap_fix["M1_standalone_mains"]["share_with_main"],
            "consolidated_main_share": co_fix["M1_standalone_mains"]["share_with_main"],
            "antipattern_files": ap_fix["python_files"],
            "consolidated_files": co_fix["python_files"],
        }

    # -- referee ----------------------------------------------------------
    def referee(self) -> None:
        arms = self.summary["arms"]
        thr = self.thr
        checks: dict[str, Any] = {}
        chan = arms.get("A1_channel", {})
        if len(chan) == 3:
            peaks = [chan[str(ny)]["u_max_rel_err"] for ny in (16, 32, 48)]
            checks["C1_channel_umax_within_owner_band"] = {
                "value": max(peaks),
                "threshold": thr["u_max_rel_err_max"],
                "pass": max(peaks) < thr["u_max_rel_err_max"],
                "note": "worst of ny=16/32/48; owner band is 15%",
            }
            checks["C2_channel_umax_within_3pct_target"] = {
                "value": max(peaks),
                "threshold": thr["u_max_rel_err_target"],
                "pass": max(peaks) < thr["u_max_rel_err_target"],
            }
            best_l2 = chan["48"]["profile_rel_l2"]
            checks["C3_profile_l2_target_at_finest"] = {
                "value": best_l2,
                "threshold": thr["profile_rel_l2_target"],
                "pass": best_l2 < thr["profile_rel_l2_target"],
                "worst_over_resolutions": max(chan[str(ny)]["profile_rel_l2"] for ny in (16, 32, 48)),
                "interior_l2_ny48": chan["48"]["profile_interior_rel_l2"],
            }
            checks["C4_resolution_monotone_umax_err"] = {
                "peaks": peaks,
                "pass": peaks[0] >= peaks[1] - 1e-3 and peaks[1] >= peaks[2] - 1e-3,
            }
            md = max(abs(chan[str(ny)]["mass_drift"]) for ny in (16, 32, 48))
            checks["C5_mass_drift"] = {
                "value": md,
                "threshold": thr["mass_drift_max"],
                "pass": md < thr["mass_drift_max"],
            }
            checks["C11_stability_rule"] = {
                "rho_range": [
                    min(chan[str(ny)]["rho_min"] for ny in (16, 32, 48)),
                    max(chan[str(ny)]["rho_max"] for ny in (16, 32, 48)),
                ],
                "pass": all(
                    chan[str(ny)]["rho_min"] > 0.95 and chan[str(ny)]["rho_max"] < 1.05
                    for ny in (16, 32, 48)
                ),
            }
        if "A3_taylor_green" in arms:
            tg = arms["A3_taylor_green"]
            checks["C6_tg_decay_target"] = {
                "value": tg["rel_err"],
                "threshold": thr["tg_decay_rel_err_target"],
                "pass": tg["rel_err"] < thr["tg_decay_rel_err_target"],
            }
        if "A2_control" in arms:
            a2 = arms["A2_control"]
            checks["C7_harness_matches_independent_control"] = {
                "value": a2["harness_vs_control_rel"],
                "threshold": thr["harness_vs_control_rel_max"],
                "profile_l2": a2.get("harness_vs_control_profile_rel_l2"),
                "pass": a2["harness_vs_control_rel"] < thr["harness_vs_control_rel_max"],
            }
        if "A4_determinism" in arms:
            a4 = arms["A4_determinism"]
            checks["C8_determinism"] = {
                "rerun_identical": a4["identical_rerun"],
                "field_identical": a4["field_identical"],
                "perturbation_changes_digest": a4["perturbation_changes_digest"],
                "perturbed_u_max_rel_change": a4["perturbed_u_max_rel_change"],
                "pass": a4["identical_rerun"]
                and a4["field_identical"]
                and a4["perturbation_changes_digest"],
            }
        if "A6_detector_control" in arms:
            a6 = arms["A6_detector_control"]
            checks["C9_detector_discriminates"] = {
                "value": a6["driver_loc_ratio_gap"],
                "threshold": thr["detector_gap_min"],
                "antipattern_dup_ratio": a6["antipattern_dup_loc_ratio"],
                "consolidated_dup_ratio": a6["consolidated_dup_loc_ratio"],
                "antipattern_driver_ratio": a6["antipattern_driver_loc_ratio"],
                "consolidated_driver_ratio": a6["consolidated_driver_loc_ratio"],
                "thin_share": [a6["antipattern_thin_share"], a6["consolidated_thin_share"]],
                "pass": a6["driver_loc_ratio_gap"] > thr["detector_gap_min"]
                and a6["antipattern_thin_share"] == 0.0
                and a6["consolidated_thin_share"] == 1.0,
            }
        if "A5_examples_audit" in arms:
            a5 = arms["A5_examples_audit"]
            checks["C10_antipattern_present_in_repo"] = {
                "main_share": a5["M1"]["share_with_main"],
                "dup_loc_ratio": a5["M2"]["dup_loc_ratio"],
                "dup_driver_loc_ratio": a5["M2"]["dup_driver_loc_ratio"],
                "n_dup_driver_blocks": a5["M2"]["n_dup_driver_blocks"],
                "own_step_loop_share": a5["M1"]["n_files_own_step_loop"] / a5["python_files"],
                "threshold": thr["dup_loc_ratio_present_min"],
                "pass": a5["M2"]["dup_loc_ratio"] > thr["dup_loc_ratio_present_min"]
                and a5["M1"]["share_with_main"] > 0.5,
            }
        self.summary["checks"] = checks
        complete = all(
            [
                len(self.summary["arms"].get("A1_channel", {})) == 3,
                "A2_control" in arms,
                "A3_taylor_green" in arms,
                "A4_determinism" in arms,
                "A5_examples_audit" in arms,
                "A6_detector_control" in arms,
            ]
        )
        self.summary["arms_complete"] = complete
        self.summary["all_pass"] = bool(complete and all(c["pass"] for c in checks.values()))
        self.summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def write_tables(self) -> None:
        arms = self.summary["arms"]
        checks = self.summary["checks"]
        lines = ["# XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN -- machine referee", ""]
        lines.append(f"arms_complete={self.summary.get('arms_complete')} all_pass={self.summary.get('all_pass')}")
        lines.append("")
        lines.append("| check | value | threshold | pass |")
        lines.append("|---|---|---|---|")
        for k, c in checks.items():
            val = c.get("value", c.get("peaks", "-"))
            lines.append(f"| {k} | {val} | {c.get('threshold', '-')} | {c['pass']} |")
        chan = arms.get("A1_channel", {})
        if chan:
            lines += [
                "",
                "## channel u_max vs closed form g H^2 / (8 nu)",
                "| ny | u_max got | u_max ref | rel err | profile L2 | interior L2 | mass drift | steps |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for ny in (16, 32, 48):
                a = chan.get(str(ny))
                if not a:
                    continue
                lines.append(
                    f"| {ny} | {a['u_max']:.8f} | {a['u_max_analytic']:.8f} | "
                    f"{a['u_max_rel_err']*100:.4f}% | {a['profile_rel_l2']*100:.4f}% | "
                    f"{a['profile_interior_rel_l2']*100:.4f}% | {a['mass_drift']:.2e} | "
                    f"{a['steps_run']} |"
                )
        if "A3_taylor_green" in arms:
            tg = arms["A3_taylor_green"]
            lines += [
                "",
                "## Taylor-Green energy decay",
                "| got | ref (4 nu k^2) | rel err |",
                "|---|---|---|",
                f"| {tg['decay_rate_measured']:.6e} | {tg['decay_rate_analytic']:.6e} | "
                f"{tg['rel_err']*100:.4f}% |",
            ]
        if "A2_control" in arms:
            a2 = arms["A2_control"]
            lines += [
                "",
                "## independent example-main control (ny=32, torch.roll streaming)",
                "| harness u_max | control u_max | rel diff | profile L2 diff |",
                "|---|---|---|---|",
                f"| {a2['harness_u_max']:.10f} | {a2['control_u_max']:.10f} | "
                f"{a2['harness_vs_control_rel']:.3e} | "
                f"{(a2.get('harness_vs_control_profile_rel_l2') or 0):.3e} |",
            ]
        with open(os.path.join(self.out, "summary.md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        self.summary["tables_md"] = "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out", default=os.path.join(HERE, "run_artifacts"))
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ns = ap.parse_args(argv)

    os.makedirs(ns.out, exist_ok=True)
    runner = Runner(ns.repo_root, ns.out)
    arms = [a.strip() for a in ns.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ALL_ARMS:
            raise SystemExit(f"unknown arm {a!r}; known: {ALL_ARMS}")
    t0 = time.time()
    for a in arms:
        ta = time.time()
        if a.startswith("a1_"):
            runner.arm_a1(int(a.split("_")[1]))
        else:
            getattr(runner, f"arm_{a}")()
        print(f"[arm {a}] {time.time()-ta:.1f}s", flush=True)
        # persist after each arm so the run is restartable
        runner.referee()
        runner.write_tables()
        _json_dump(runner.summary, runner.summary_path)
    print(f"[total] {time.time()-t0:.1f}s")
    print(runner.summary["tables_md"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())