"""Publish the P333W2-N06 report (markdown + embedded figures) to the Tensor API.

Credentials come from the environment (``TENSOR_API_BASE``, ``TENSOR_AGENT_USER``,
``TENSOR_AGENT_PASS``); nothing is stored in the repository.

    python scripts/post_perforated_plate_report.py
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys

RUN = os.environ.get("PP_RUN", "runs/perforated_plate_n6")
DOC = os.environ.get("PP_DOC", "docs/PERFORATED_PLATE_N6_VERIFICATION.md")

CAPTIONS = {
    "fig1_cd_cm_phase_vs_KC.png":
        "C_d, C_m and dp-Q phase lag versus KC (beta=0.25); dashed = Borda-Carnot sharp-edge "
        "reference, band = +-10% acceptance window",
    "fig2_cd_vs_beta.png":
        "C_d versus open-area ratio beta at KC~11 with the sharp-edge reference",
    "fig3_gci.png":
        "Grid convergence of C_d over three self-similar resolutions (r=2), Richardson extrapolation",
    "fig4_phase_orbit.png":
        "Phase-averaged dp-Q orbits (area = cycle dissipation Phi) and dp waveforms",
    "fig5_negative_controls.png":
        "Negative controls: amplitude match, relative errors vs acceptance limits, and the "
        "oscillating-channel velocity profile against Womersley",
    "fig6_energy_budget.png":
        "Cycle dissipation from momentum exchange and from pressure stations, and the closure error",
    "fig7_field_kcA04.png": "Velocity / vorticity at peak flow (KC=3.0)",
    "fig7_field_kcA08.png": "Velocity / vorticity at peak flow (KC=22.3)",
}


def main() -> int:
    api = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    pwd = os.environ["TENSOR_AGENT_PASS"]

    login = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{api}/auth/login",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"username": user, "password": pwd})],
        capture_output=True, text=True, check=True)
    tok = json.loads(login.stdout)
    token = tok.get("access_token") or tok.get("token")
    if not token:
        print("login failed:", login.stdout[:400])
        return 1

    with open(DOC) as fh:
        markdown = fh.read()

    figs = []
    figdir = os.path.join(RUN, "figures")
    for name in sorted(os.listdir(figdir)):
        if not name.endswith(".png"):
            continue
        with open(os.path.join(figdir, name), "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        figs.append({"name": name, "caption": CAPTIONS.get(name, name), "png_base64": b64})

    with open(os.path.join(RUN, "metrics.json")) as fh:
        m = json.load(fh)
    g = m["gates"]
    verdict = ", ".join(f"{k}={'PASS' if v.get('passed') else 'FAIL'}" for k, v in g.items())

    payload = {
        "title": "P333W2-N06 TensorLBM 穿孔板振荡流验证面：C_d(KC,β)、C_m(KC,β)、Φ→ζ_eff",
        "summary": ("LBM 穿孔板振荡流验证面：D2Q9+Guo 强迫+半程反弹+动量交换测力；"
                    "9 点 KC 扫描 (0.43-22.3)、5 点 β 扫描、三档 GCI、双负控、双路径能量账本。"
                    "判据：" + verdict),
        "scenario_key": "perforated_plate_oscillatory_flow",
        "mission_id": 11101,
        "tags": ["TensorLBM", "perforated_plate", "oscillatory_flow", "C_d", "added_mass",
                 "energy_dissipation", "GCI", "verification", "P333W2-N06",
                 "sloshing_damping"],
        "markdown": markdown,
        "figures": figs,
    }

    out = os.environ.get("PP_OUT", "/dev/shm/tmp/report.json")
    with open(out, "w") as fh:
        json.dump(payload, fh)
    print("payload bytes:", os.path.getsize(out), "figures:", len(figs))

    r = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{api}/tensor/reports",
         "-H", f"Authorization: Bearer {token}",
         "-H", "Content-Type: application/json",
         "-d", "@" + out],
        capture_output=True, text=True)
    print(r.returncode, r.stdout[:800], r.stderr[:300])
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())