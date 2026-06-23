from __future__ import annotations


def test_cylinder_flow_integrates_closure_features(client, waiter):
    r = client.post(
        "/api/solve/cylinder-flow",
        json={
            "nx": 48,
            "ny": 20,
            "u_in": 0.05,
            "re": 50.0,
            "radius": 4.0,
            "n_steps": 10,
            "output_interval": 5,
            "physics": {
                "synthetic_inflow": {
                    "enabled": True,
                    "method": "digital_filter",
                    "length_scale": 3.0,
                },
                "sponge_layer": {
                    "enabled": True,
                    "start_fraction": 0.75,
                    "amplitude": 0.25,
                },
                "turbulence_statistics": {
                    "enabled": True,
                    "start_step": 0,
                    "sample_every": 1,
                },
            },
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    job = waiter(job_id, timeout=60)
    assert job["status"] == "completed"

    meta = client.get(f"/api/jobs/{job_id}/metadata")
    assert meta.status_code == 200, meta.text
    closure = meta.json()["metadata"]["engineering_closure"]
    assert closure["synthetic_inflow"]["enabled"] is True
    assert closure["sponge_layer"]["enabled"] is True
    assert closure["turbulence_statistics"]["enabled"] is True
    assert closure["synthetic_inflow_runtime"]["mean_u_rms"] > 0.0
    assert closure["sponge_layer_runtime"]["max_strength"] > 0.0
    assert closure["turbulence_statistics_runtime"]["n_samples"] >= 1

    live = client.get(f"/api/jobs/{job_id}/live-metrics")
    assert live.status_code == 200, live.text
    diagnostics = live.json()["diagnostics"]
    assert diagnostics
    assert "cd" in diagnostics[-1]
    assert "inlet_rms_u" in diagnostics[-1]


def test_turbulent_channel_integrates_roughness_and_stats(client, waiter):
    r = client.post(
        "/api/solve/turbulent-channel",
        json={
            "nx": 32,
            "ny": 16,
            "re_tau": 50.0,
            "u_tau": 0.005,
            "smagorinsky_cs": 0.1,
            "n_steps": 10,
            "averaging_start": 0,
            "output_interval": 5,
            "physics": {
                "rough_wall": {
                    "enabled": True,
                    "ks": 0.4,
                    "reference_u_tau": 0.005,
                },
                "turbulence_statistics": {
                    "enabled": True,
                    "start_step": 0,
                    "sample_every": 1,
                },
            },
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    job = waiter(job_id, timeout=60)
    assert job["status"] == "completed"

    meta = client.get(f"/api/jobs/{job_id}/metadata")
    assert meta.status_code == 200, meta.text
    closure = meta.json()["metadata"]["engineering_closure"]
    assert closure["rough_wall"]["enabled"] is True
    assert closure["turbulence_statistics"]["enabled"] is True
    assert closure["rough_wall_runtime"]["mean_damping"] >= 0.0
    assert closure["turbulence_statistics_runtime"]["n_samples"] >= 1

    live = client.get(f"/api/jobs/{job_id}/live-metrics")
    assert live.status_code == 200, live.text
    diagnostics = live.json()["diagnostics"]
    assert diagnostics
    assert "tke_mean" in diagnostics[-1]
