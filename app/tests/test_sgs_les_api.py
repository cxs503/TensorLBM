"""HTTP tests for the LBM-LES sub-grid-closure verification API."""
from __future__ import annotations


def test_sgs_verification_default_request(client):
    r = client.post("/api/v1/calculations/les/sgs_verification", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["checks"]) == {
        "smagorinsky", "dynamic_smagorinsky", "wale", "vreman"
    }
    levels = body["laminar_degeneracy"]["fields"]
    assert "poiseuille" in levels and "uniform" in levels
    # closures that must vanish in unidirectional laminar shear
    for model in ("wale", "vreman", "dynamic_smagorinsky"):
        assert levels["poiseuille"][model]["nut_max"] < 1e-9
        assert body["checks"][model]["pass"] is True
    # the static model is documented as non-degenerate
    assert levels["poiseuille"]["smagorinsky"]["nut_max"] > 0.0


def test_sgs_verification_subset_and_tgv(client):
    r = client.post(
        "/api/v1/calculations/les/sgs_verification",
        json={"models": ["wale"], "resolution": 16, "include_tgv": True,
              "tgv_size": 16, "tgv_steps": 200, "re": 400.0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert list(body["checks"]) == ["wale"]
    assert body["tgv"]["eps_max"] > 0.0
    assert body["tgv"]["blew_up_at_step"] is None


def test_sgs_verification_rejects_bad_input(client):
    bad = [
        {"models": []},
        {"resolution": 64},
        {"tau": 0.4},
        {"re": 0.0},
        {"amplitude": 0.5},
        {"models": ["not_a_model"]},
        {"include_tgv": True, "tgv_steps": 100000},
    ]
    for payload in bad:
        r = client.post("/api/v1/calculations/les/sgs_verification", json=payload)
        assert r.status_code == 422, f"{payload} -> {r.status_code}: {r.text}"


def test_sgs_report_endpoint(client):
    r = client.get("/api/v1/calculations/les/sgs_verification/report")
    assert r.status_code in (200, 404), r.text
    if r.status_code == 200:
        body = r.json()
        assert "acceptance" in body or "campaign" in body
