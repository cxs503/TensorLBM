"""CMB7-CSTUNE2-R1: publish the paper-level progress report.

Logs in, builds the report JSON (6 elements), and POSTs to /tensor/reports.
Secrets are read from env and never printed.
"""
import base64
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
FIGDIR = "/root/tp-agent/work/m27389/development/audit/cmb7_cstune2_r1/figs"


def post(path, payload, token=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main():
    login = post("/auth/login", {"username": USER, "password": PASS})
    token = login.get("access_token") or login.get("token")
    if not token:
        raise SystemExit(f"login failed: {list(login.keys())}")

    def fig(name, caption):
        b64 = base64.b64encode(open(f"{FIGDIR}/{name}", "rb").read()).decode()
        return {"name": name, "caption": caption, "png_base64": b64}

    markdown = open("/tmp/cmb7r1_report_md.md").read()
    payload = {
        "title": "CMB7-CSTUNE2-R1: z-axis closure of the Cs-zref response surface — mechanism probe verdict, cross-stack invalidation, and z=10/12 completion arms",
        "summary": "Closed the CMB7 Cs-zref refinement: a local 40-step mechanism probe proved the hydrostatic z_ref correction IS applied to all interface cells (6/6 distinct f-field hashes), so the z_ref=18/20 bitwise-identical depth histories are a gauge-resolution artifact (uniform 3.07e-5 rho_g shift), not a flat response surface; the draft's cross-stack comparison against the CMB6 champion (different commit) is invalidated; two completion arms (z=10, z=12) show the z-axis is engaged and non-monotonic with a peak at z_ref~12 (1.6146D), still below the 3% HIT gate (1.8017D).",
        "scenario_key": "DE-X-CMB7-CSTUNE2-R1",
        "mission_id": 27389,
        "tags": ["TensorLBM", "free-surface", "water-entry", "Smagorinsky", "hydrostatic-compensation", "mechanism-probe", "honest-FAIL"],
        "markdown": markdown,
        "figures": [
            fig("fig1_depth_time.png", "Fig 1. Cavity depth vs time (cells=8, RR+SMAG+HYDRO). Blue/red: z-axis completion arms (z_ref=10/12); green/orange: CMB7 z_ref=18/20; gray: Cs sweep. Black stars: Santini et al. experimental markers (Fig.5). Dashed: 3% HIT gate (1.8017D)."),
            fig("fig2_response.png", "Fig 2. Peak cavity depth vs z_ref (Cs=0.075, blue) and Cs sweep at z_ref=18 (red squares). The z-axis is engaged and non-monotonic, peaking at z_ref~12."),
            fig("fig3_mechanism.png", "Fig 3. Mechanism probe. Left: analytic hydrostatic correction magnitude at the water surface (z=24.5) and cavity bottom (z=12) vs z_ref. Right: all 6 probe arms produce distinct 40-step f-field hashes (correction engaged)."),
        ],
    }
    out = post("/tensor/reports", payload, token)
    print("report posted:", json.dumps(out, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
