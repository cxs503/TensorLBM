"""Tests for platform i18n JSON dictionaries and static file serving."""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).parent
_STATIC_I18N = _HERE.parent / "frontend" / "static" / "i18n"


def _flatten(d: dict, prefix: str = "") -> set:
    keys: set = set()
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            keys |= _flatten(v, full)
        else:
            keys.add(full)
    return keys


def test_en_json_is_valid():
    """en.json must be parseable and non-empty."""
    path = _STATIC_I18N / "en.json"
    assert path.exists(), "platform/frontend/static/i18n/en.json not found"
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, dict) and len(data) > 0


def test_zh_json_is_valid():
    """zh.json must be parseable and non-empty."""
    path = _STATIC_I18N / "zh.json"
    assert path.exists(), "platform/frontend/static/i18n/zh.json not found"
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, dict) and len(data) > 0


def test_i18n_key_parity():
    """en.json and zh.json must have identical key sets – no missing or extra keys."""
    en_keys = _flatten(json.loads((_STATIC_I18N / "en.json").read_text(encoding="utf-8")))
    zh_keys = _flatten(json.loads((_STATIC_I18N / "zh.json").read_text(encoding="utf-8")))

    missing_zh = sorted(en_keys - zh_keys)
    extra_zh = sorted(zh_keys - en_keys)

    assert not missing_zh, f"Keys in en.json but missing from zh.json: {missing_zh}"
    assert not extra_zh, f"Keys in zh.json but missing from en.json: {extra_zh}"


def test_critical_keys_present_in_both():
    """A selection of critical UI keys must exist in both locale files."""
    critical_keys = [
        "title",
        "nav.dashboard",
        "nav.solve",
        "nav.postprocess",
        "ws.connected",
        "ws.disconnected",
        "sidebar.no_jobs",
        "solve.submit_btn",
        "postprocess.select_hint",
        "bench.run_btn",
        "agent.thinking",
        "common.loading",
    ]
    for lang in ("en", "zh"):
        path = _STATIC_I18N / f"{lang}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        flat = _flatten(data)
        for k in critical_keys:
            assert k in flat, f"Critical key '{k}' missing from {lang}.json"


def test_static_i18n_files_served(client):
    """The /static/i18n/en.json and zh.json endpoints must return valid JSON."""
    for lang in ("en", "zh"):
        r = client.get(f"/static/i18n/{lang}.json")
        assert r.status_code == 200, f"/static/i18n/{lang}.json returned {r.status_code}"
        data = r.json()
        assert isinstance(data, dict) and len(data) > 0


def test_i18n_js_served(client):
    """The /static/js/i18n.js file must be served successfully."""
    r = client.get("/static/js/i18n.js")
    assert r.status_code == 200
    assert b"window.t" in r.content or b"window.i18n" in r.content
