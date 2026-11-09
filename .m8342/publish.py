"""Publish the IN101-N01 report to the tensor report API."""
import base64
import json
import os
import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, method="POST",
                                 data=json.dumps(payload).encode())
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:800])
        raise


tok = None
res = post("/auth/login", {"username": USER, "password": PASS})
tok = res.get("access_token") or res.get("token") or (res.get("data") or {}).get("access_token")
print("logged in:", bool(tok))

md = open("docs/IN101_N01_ICE_KEEL_DRAG.md").read()
figs = [
    ("fig_profiles", "Wall-unit mean profile: three-level smooth ladder and the deepest keel case vs the log law",
     "artifacts/IN101_N01/figures/fig_profiles.png"),
    ("fig_form_scaling", "Form-drag scaling: C_d^form vs measured S_fr/S_plan with the power-law fit (n=1.096, R2=0.847)",
     "artifacts/IN101_N01/figures/fig_form_scaling.png"),
    ("fig_drag_band", "Swept C_d vs the drift-inversion band 0.004-0.02 (all 12 cases above the band, Re_h~130)",
     "artifacts/IN101_N01/figures/fig_drag_band.png"),
    ("fig_geometry", "Measured vs analytic frontal-area ratio and normalised form drag along the depth sweep",
     "artifacts/IN101_N01/figures/fig_geometry.png"),
]
fig_payload = []
for name, cap, path in figs:
    with open(path, "rb") as f:
        fig_payload.append({"name": name, "caption": cap,
                            "png_base64": base64.b64encode(f.read()).decode()})

payload = {
    "title": "IN101-N01 ice-keel drag decomposition: 2/5 criteria pass, 4 kernel defects fixed",
    "summary": ("ice_keel_drag pipeline (case table -> LBM sweep -> 5 machine criteria -> figures/JSON) + 4 kernel "
                "defects fixed. Smooth cf matches exact 18/Re_tau^2 to 0.011% (GCI 0.048%); negative control exact; "
                "form-drag exponent 1.096 (9.6%, conditional). Honest FAIL: turbulent branch unreachable (91%/92% vs "
                "log law/Dean), skin+form closure median 96%, C_d 0.023-0.027 outside the 0.004-0.02 band."),
    "scenario_key": "ice_keel_drag_cd_decomposition",
    "mission_id": 8342,
    "tags": ["TensorLBM", "ice", "keel", "drag-decomposition", "C_d", "LBM",
             "IN101-N01", "honest-fail"],
    "markdown": md,
    "figures": fig_payload,
}
res = post("/tensor/reports", payload, tok)
print(json.dumps(res)[:1500])
