#!/usr/bin/env python3
"""Board audit m17880 — build the machine-produced evidence bundle for mission m17762.

Every number written into artifacts/ is read from the delivered run artifact or from a
command that is executed here; nothing is transcribed by hand.

    python docs/board_audit/m17880/build_evidence.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
REFS = HERE / "refs"
ART = HERE / "artifacts"
M17762 = pathlib.Path("/root/tp-agent/work/m17762")
TOL = 0.15  # owner standard: >15% is not acceptable


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sh(cmd: str, cwd: pathlib.Path | None = None) -> dict:
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return {"cmd": cmd, "rc": r.returncode, "out": (r.stdout or r.stderr).strip()[:2000]}


def manifest() -> dict:
    files = {}
    for p in sorted(REFS.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(HERE))] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    # original (m17762 worktree) counterparts, by relative path
    orig = {
        "src/tensorlbm/multiphase3d.py": M17762 / "src/tensorlbm/multiphase3d.py",
        "scripts/xval_m17762_free_energy_ch_core.py": M17762 / "scripts/xval_m17762_free_energy_ch_core.py",
        "scripts/render_xval_m17762_report.py": M17762 / "scripts/render_xval_m17762_report.py",
        "tests/test_xval_m17762_free_energy_ch.py": M17762 / "tests/test_xval_m17762_free_energy_ch.py",
        "artifacts/xval_free_energy_ch_core.json":
            M17762 / "artifacts/xval_m17762/xval_free_energy_ch_core.json",
    }
    pairs = {}
    for rel, p in orig.items():
        snap = REFS / rel
        pairs[rel] = {
            "original_path": str(p),
            "original_exists": p.exists(),
            "original_sha256": sha256(p) if p.exists() else None,
            "snapshot_sha256": sha256(snap) if snap.exists() else None,
            "identical": bool(p.exists() and snap.exists() and sha256(p) == sha256(snap)),
        }
    return {"snapshot_files": files, "originals": pairs}


def landed_checks() -> dict:
    """F1 landing audit: is any deliverable present in the mission branch HEAD?"""
    out = {}
    for rel in ["scripts/xval_m17762_free_energy_ch_core.py",
                "scripts/render_xval_m17762_report.py",
                "tests/test_xval_m17762_free_energy_ch.py",
                "docs/XVAL_M17762_LBM_FREE_ENERGY_CAHN_HILLIARD_CORE.md"]:
        r = sh(f"git cat-file -e HEAD:{rel}", cwd=M17762)
        out[rel] = {"in_mission_branch_HEAD": r["rc"] == 0}
    r = sh("git cat-file -p HEAD:src/tensorlbm/multiphase3d.py | grep -c geq_convention", cwd=M17762)
    out["src/tensorlbm/multiphase3d.py"] = {
        "geq_convention_occurrences_in_HEAD": int(r["out"] or 0),
        "in_mission_branch_HEAD": int(r["out"] or 0) > 0,
    }
    # main-line presence in the auditor's own checkout (branch agent/m17880 off main)
    main = pathlib.Path("/root/tp-agent/work/m17880")
    out["_main_checkout"] = {
        rel: (main / rel).exists()
        for rel in ["scripts/xval_m17762_free_energy_ch_core.py",
                    "tests/test_xval_m17762_free_energy_ch.py"]
    }
    out["_git_status_mission_worktree"] = sh("git status --porcelain", cwd=M17762)["out"].splitlines()
    return out


def ref_vs_got() -> dict:
    d = json.loads((REFS / "artifacts/xval_free_energy_ch_core.json").read_text())
    c = d["parts"]["C_dispersion"]
    rows = []
    for r in c["rows"]:
        rows.append({
            "m": r["m"], "k": r["k"],
            "ref_analytic_omega": r["analytic"],
            "got_oss_omega": r["oss_omega"],
            "got_tl_lee_lin_omega": r["tl_lee_lin_omega"],
            "got_tl_swift_omega": r["tl_swift_omega"],
            "err_tl_vs_analytic_rel": r["tl_lee_lin_rel_error"],
            "err_oss_vs_analytic_rel": r["oss_rel_error"],
            "abs_dev_tl_vs_analytic": abs(r["tl_lee_lin_omega"] - r["analytic"]),
            "err_tl_vs_oss_rel": abs(r["tl_lee_lin_omega"] - r["oss_omega"]) / abs(r["oss_omega"]),
            "pass_15pct_tl_vs_analytic": abs(r["tl_lee_lin_rel_error"]) < TOL,
            "pass_15pct_tl_vs_oss": abs(abs(r["tl_lee_lin_omega"] - r["oss_omega"])
                                        / abs(r["oss_omega"])) < TOL,
        })
    summary = {
        "modes": len(rows),
        "tl_vs_analytic_pass": sum(r["pass_15pct_tl_vs_analytic"] for r in rows),
        "tl_vs_analytic_fail_modes": [r["m"] for r in rows if not r["pass_15pct_tl_vs_analytic"]],
        "max_abs_err_tl_vs_analytic": max(abs(r["err_tl_vs_analytic_rel"]) for r in rows),
        "max_rel_err_tl_vs_oss": max(r["err_tl_vs_oss_rel"] for r in rows),
        "neutral_mode_kc_analytic": (c["parameters"]["A_tensorlbm"]
                                     / c["parameters"]["kappa"]) ** 0.5,
        "highest_mode_k_ratio_to_kc": rows[-1]["k"] / (c["parameters"]["A_tensorlbm"]
                                                       / c["parameters"]["kappa"]) ** 0.5,
    }
    b = d["parts"]["B_calibration"]
    calibration = {
        "sigma_target": b["inputs"]["sigma"],
        "sigma_recovered_oss": b["oss_constants"]["sigma_recovered"],
        "sigma_recovered_tl": b["tensorlbm_parameters"]["sigma_recovered"],
        "sigma_identity_rel_error_oss": b["oss_constants"]["sigma_identity_rel_error"],
        "sigma_identity_rel_error_tl": b["tensorlbm_parameters"]["sigma_identity_rel_error"],
        "profile_measured_sigma_rel_error": b["surface_tension_from_discrete_profile"]["oss_rel_error"],
        "beta_over_abs_a": b["tensorlbm_parameters"]["beta_over_abs_a"],
        "kappa_over_oss_kappa": b["tensorlbm_parameters"]["kappa_over_oss_kappa"],
        "mu_operator_rel_l2_tl_vs_oss": b["chemical_potential_on_discrete_tanh"]["tl_central7_vs_oss_rel_l2"],
    }
    mob = [{"Gamma": r["Gamma"], "M_eff_tl": r["tl_lee_lin_M_eff"],
            "M_eff_oss": r["oss_M_eff"], "M_predicted": r["predicted_M_lee_lin"],
            "rel_err_tl_vs_predicted": abs(r["tl_lee_lin_M_eff"] - r["predicted_M_lee_lin"])
            / r["predicted_M_lee_lin"]} for r in c["mobility_calibration"]]
    swift = [{"m": r["m"], "measured_decay": r["omega_measured"],
              "model_spurious": r["omega_model_spurious"], "rel_error": r["rel_error"]}
             for r in c["swift_decay_scan"]["rows"]]
    return {
        "anchor": d["anchor"], "mission": d["mission"],
        "reference": d["reference"],
        "tolerance_bar": TOL,
        "dispersion_rows": rows,
        "dispersion_summary": summary,
        "calibration": calibration,
        "mobility_calibration": mob,
        "swift_spurious_decay_scan": swift,
        "operator_order": d["parts"]["A_operators"]["observed_order"],
        "difference_identity": d["parts"]["D_differences"],
        "resolution_sensitivity": c["resolution_sensitivity"],
        "amplitude_sensitivity": c["amplitude_sensitivity"],
    }


def repro_compare() -> dict:
    a = json.loads((REFS / "artifacts/xval_free_energy_ch_core.json").read_text())
    b = json.loads(pathlib.Path("/tmp/m17880_repro/xval_free_energy_ch_core.json").read_text())
    volatile = ("timestamp", "tensorlbm_git_head")
    a2 = {k: v for k, v in a.items() if k not in volatile}
    b2 = {k: v for k, v in b.items() if k not in volatile}
    png = {}
    for f in ["fig1_ch_dispersion.png", "fig2_mode_history.png", "fig3_operator_convergence.png"]:
        o = REFS / "artifacts" / f
        n = pathlib.Path("/tmp/m17880_repro") / f
        png[f] = {"delivered_sha256": sha256(o) if o.exists() else None,
                  "rerun_sha256": sha256(n) if n.exists() else None,
                  "identical": bool(o.exists() and n.exists() and sha256(o) == sha256(n))}
    return {
        "repro_cmd": "cd /root/tp-agent/work/m17762 && PYTHONPATH=src python3 "
                     "scripts/xval_m17762_free_energy_ch_core.py --parts ABCDEF "
                     "--oss-root /tmp/Fingering_dynamics --out /tmp/m17880_repro",
        "oss_checkout": sh("git -C /tmp/Fingering_dynamics rev-parse HEAD"),
        "oss_file_sha256": sh("sha256sum /tmp/Fingering_dynamics/lattice_boltzmann/"
                              "fingering_periodic.py /tmp/Fingering_dynamics/lattice_boltzmann/"
                              "create_block.py /tmp/Fingering_dynamics/lattice_boltzmann/fingering.py"),
        "json_payload_identical": json.dumps(a2, sort_keys=True) == json.dumps(b2, sort_keys=True),
        "volatile_keys_ignored": list(volatile),
        "timestamp_delivered": a.get("timestamp"), "timestamp_rerun": b.get("timestamp"),
        "png": png,
    }


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    bundle = {
        "manifest.json": manifest(),
        "ref_vs_got.json": ref_vs_got(),
        "repro_compare.json": repro_compare(),
        "landing_checks.json": landed_checks(),
        "pytest_rerun.json": {
            "cmd": "cd /root/tp-agent/work/m17762 && python3 -m pytest "
                   "tests/test_xval_m17762_free_energy_ch.py -q",
            "tail": pathlib.Path("/tmp/m17880_pytest.log").read_text().splitlines()[-3:],
        },
        "regression_rerun.json": {
            "cmd": "cd /root/tp-agent/work/m17762 && python3 -m pytest "
                   "tests/test_multiphase3d_d3q27_free_energy.py "
                   "tests/test_multiphase_extensions.py -q",
            "tail": pathlib.Path("/tmp/m17880_regress.log").read_text().splitlines()[-3:],
        },
        "render_script_syntax_check.json": sh(
            "python3 -m py_compile scripts/render_xval_m17762_report.py", cwd=M17762),
    }
    for name, obj in bundle.items():
        (ART / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
        print("wrote", name)


if __name__ == "__main__":
    main()