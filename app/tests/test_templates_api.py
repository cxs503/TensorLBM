"""Tests for the TensorLBM Engineering Templates API."""
from __future__ import annotations

import pytest


class TestTemplatesList:
    def test_list_all(self, client):
        r = client.get("/api/templates/")
        assert r.status_code == 200
        data = r.json()
        assert "templates" in data
        assert "categories" in data
        assert "total" in data
        assert data["total"] > 0
        assert len(data["templates"]) == data["total"]

    def test_list_category_filter(self, client):
        r = client.get("/api/templates/?category=external_flow")
        assert r.status_code == 200
        data = r.json()
        for tmpl in data["templates"]:
            assert tmpl["category"] == "external_flow"

    def test_list_empty_category(self, client):
        r = client.get("/api/templates/?category=nonexistent_cat")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["templates"] == []

    def test_list_categories(self, client):
        r = client.get("/api/templates/categories")
        assert r.status_code == 200
        data = r.json()
        assert "categories" in data
        cats = data["categories"]
        assert len(cats) > 0
        for cat in cats:
            assert "id" in cat
            assert "label" in cat
            assert "count" in cat

    def test_template_fields(self, client):
        r = client.get("/api/templates/")
        for tmpl in r.json()["templates"]:
            assert "id" in tmpl
            assert "title" in tmpl
            assert "category" in tmpl
            assert "description" in tmpl
            assert "solver_type" in tmpl
            assert "default_config" in tmpl
            assert isinstance(tmpl["default_config"], dict)


class TestTemplateGet:
    def test_get_known_template(self, client):
        r = client.get("/api/templates/ext_aero_cylinder")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == "ext_aero_cylinder"
        assert data["solver_type"] == "cylinder_flow"
        assert "re" in data["default_config"]

    def test_get_all_templates(self, client):
        """All template IDs returned by list must be retrievable individually."""
        ids = [t["id"] for t in client.get("/api/templates/").json()["templates"]]
        for tid in ids:
            r = client.get(f"/api/templates/{tid}")
            assert r.status_code == 200, f"Template {tid} not found"

    def test_get_not_found(self, client):
        r = client.get("/api/templates/not_a_real_template")
        assert r.status_code == 404


class TestTemplatesContent:
    def test_cylinder_template_has_references(self, client):
        r = client.get("/api/templates/ext_aero_cylinder")
        data = r.json()
        assert len(data.get("references", [])) > 0

    def test_chinese_title_present(self, client):
        r = client.get("/api/templates/")
        for tmpl in r.json()["templates"]:
            # Not all templates need zh title, but cylinder should have one
            if tmpl["id"] == "ext_aero_cylinder":
                assert "title_zh" in tmpl
                assert tmpl["title_zh"]

    def test_difficulty_values(self, client):
        valid = {"beginner", "intermediate", "advanced"}
        for tmpl in client.get("/api/templates/").json()["templates"]:
            assert tmpl["difficulty"] in valid, f"{tmpl['id']} has invalid difficulty"
