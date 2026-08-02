from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "assess_wall_resolved_channel3d_dns.py"
)
SPEC = importlib.util.spec_from_file_location("channel3d_dns_assessor", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dns_assessor_recovers_matching_symmetric_profile(tmp_path: Path) -> None:
    dns = tmp_path / "dns.means"
    dns.write_text(
        "# Re_tau = 20.0\n"
        "0.0 0.0 0.0 0 0 0 0\n"
        "0.1 2.5 2.5 0 0 0 0\n"
        "0.2 7.5 7.5 0 0 0 0\n"
        "0.3 12.5 12.5 0 0 0 0\n"
        "0.4 17.5 17.5 0 0 0 0\n",
        encoding="utf-8",
    )
    result = tmp_path / "channel.json"
    stress = tmp_path / "dns.reystress"
    stress.write_text(
        "# Re_tau = 20.0\n"
        "0.0 0.0 0 0 0 0 0 0\n"
        "0.1 2.5 0 0 0 0 0 0\n"
        "0.2 7.5 0 0 0 0 0 0\n"
        "0.3 12.5 0 0 0 0 0 0\n"
        "0.4 17.5 0 0 0 0 0 0\n",
        encoding="utf-8",
    )
    result.write_text(json.dumps({
        "schema": "tensorlbm-wall-resolved-channel3d-result-v1",
        "configuration": {"ny": 10, "u_tau": 0.01, "re_tau": 20.0},
        "derived": {"nu": 0.002},
        "statistics": {
            "mean_velocity_profile": [0.0, 0.025, 0.075, 0.125, 0.175,
                                      0.175, 0.125, 0.075, 0.025, 0.0],
            "reynolds_stress_profiles_uu_vv_ww_uv": [[0.0] * 10] * 4,
        },
        "acceptance": {"all": True},
        "physical_validation": True,
    }), encoding="utf-8")
    assessment = MODULE.assess(
        result,
        dns,
        dns_reynolds_stress_path=stress,
        minimum_y_plus=1.0,
        maximum_outer_fraction=1.0,
    )
    assert assessment["profile_error"]["u_plus_rms"] == 0.0
    assert assessment["physical_validation"] is True
    assert assessment["reynolds_stress_error"]["target_met"] is True
    assert len(assessment["sources"]["dns_reference_sha256"]) == 64
