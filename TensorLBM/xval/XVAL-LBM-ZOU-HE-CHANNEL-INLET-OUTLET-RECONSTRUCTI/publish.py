#!/usr/bin/env python
"""Publish the Zou-He channel cross-validation report to the platform API."""

import base64
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    login = post("/auth/login", {"username": USER, "password": PASS})
    token = (login.get("data") or login).get("access_token") or login.get("access_token")
    if not token:
        raise SystemExit(f"no token in login response: {list(login)[:6]}")

    md = open(os.path.join(HERE, "REPORT.md")).read()
    figs = []
    for name, cap in [
        ("fig1_field_and_profile.png",
         "Pressure-driven Zou-He channel: ux cloud, mid profile vs exact parabola, "
         "linear density field and per-column mass flux."),
        ("fig2_profile_and_resolution.png",
         "Normalised mid-channel profiles (all arms) and the resolution study "
         "against the 3% target / 15% acceptance bounds."),
        ("fig3_inlet_closure_contrast.png",
         "Zou-He velocity inlet vs local-equilibrium inlet: profiles at two "
         "stations and the shape-only error of both closures."),
    ]:
        path = os.path.join(HERE, "figures", name)
        if os.path.exists(path):
            figs.append({
                "name": name, "caption": cap,
                "png_base64": base64.b64encode(open(path, "rb").read()).decode(),
            })

    summary = json.load(open(os.path.join(
        HERE, "..", "..", "artifacts", "xval_zouhe", "results_summary.json")))
    pp = summary.get("pp_H16")
    payload = {
        "title": "XVAL-LBM Zou-He channel inlet/outlet reconstruction: "
                 "pressure/velocity closures vs exact plane Poiseuille",
        "summary": (
            "New tensorlbm.boundary.zou_he_channel (all four walls, velocity and "
            "pressure modes) closes an open D2Q9 channel without body force or "
            "periodic wrap; pressure-driven run matches the exact parabola to "
            f"{pp['rel_umax']:.2e} in u_max and {pp['shape_rel_l2']:.2e} in shape, "
            "interior mass flux constant to "
            f"{pp['q_interior_spread']:.1e}. Diagonal-pairing defect found and fixed "
            "(density was exact while transverse momentum was wrong); "
            "local-equilibrium inlet is NOT measurably worse at mid-channel "
            "(recorded negative result)."
        ),
        "scenario_key": "XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI",
        "mission_id": 21826,
        "tags": ["LBM", "Zou-He", "boundary-condition", "Poiseuille",
                 "cross-validation", "TensorLBM", "D2Q9"],
        "markdown": md,
        "figures": figs,
    }
    with open("/tmp/report.json", "w") as fh:
        json.dump(payload, fh)
    res = post("/tensor/reports", payload, token)
    print(json.dumps(res)[:600])


if __name__ == "__main__":
    main()