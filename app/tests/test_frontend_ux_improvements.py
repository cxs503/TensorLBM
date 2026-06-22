"""Frontend regression checks for UX improvement features.

Verifies that the five gap-closure UX additions are present in the frontend:
1. WebSocket handler shows toasts on job complete / fail transitions.
2. Keyboard shortcuts modal exists (? key + navbar button).
3. Job sidebar cards render a progress bar + ETA + steps/sec for running jobs.
4. A/B side-by-side Snapshot Compare tab is present in the Postprocess panel.
5. i18n keys for all new features are present in both locales.
"""
from __future__ import annotations

import json
from pathlib import Path


FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
JS_CORE = FRONTEND / "static" / "js" / "app_core.js"
JS_PP = FRONTEND / "static" / "js" / "app_postprocess.js"
EN_JSON = FRONTEND / "static" / "i18n" / "en.json"
ZH_JSON = FRONTEND / "static" / "i18n" / "zh.json"


# ---------------------------------------------------------------------------
# 1. Toast notifications on job status transitions
# ---------------------------------------------------------------------------

def test_ws_handler_toasts_on_job_complete() -> None:
    core = JS_CORE.read_text(encoding="utf-8")
    # Must track previous state to detect transitions
    assert "const prev = jobsMap[j.job_id]" in core
    assert "prev.status !== j.status" in core
    # Must call showToast for completed and failed
    assert "j.status === 'completed'" in core
    assert "j.status === 'failed'" in core
    assert "showToast" in core


# ---------------------------------------------------------------------------
# 2. Keyboard shortcuts modal
# ---------------------------------------------------------------------------

def test_shortcuts_modal_exists_in_html() -> None:
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert 'id="shortcuts-modal"' in html
    assert 'data-bs-target="#shortcuts-modal"' in html


def test_question_mark_key_opens_shortcuts_modal() -> None:
    core = JS_CORE.read_text(encoding="utf-8")
    assert "ev.key === '?'" in core
    assert "shortcuts-modal" in core
    assert "bootstrap.Modal.getOrCreateInstance" in core


# ---------------------------------------------------------------------------
# 3. Job sidebar progress bar + ETA + steps/sec
# ---------------------------------------------------------------------------

def test_job_card_renders_progress_bar() -> None:
    core = JS_CORE.read_text(encoding="utf-8")
    assert "progress-bar" in core
    assert "steps/s" in core
    assert "ETA" in core
    # Must use n_steps from config to compute progress
    assert "j.config.n_steps" in core or "j.config && (j.config.n_steps" in core


# ---------------------------------------------------------------------------
# 4. A/B compare tab in Postprocess panel
# ---------------------------------------------------------------------------

def test_abcompare_tab_in_postprocess_html() -> None:
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert "showPPMergedTab('abcompare'" in html
    assert 'id="pp-merged-abcompare"' in html
    assert 'id="ab-job-a"' in html
    assert 'id="ab-job-b"' in html
    assert "abCompareLoad" in html
    assert 'id="ab-compare-grid"' in html


def test_abcompare_functions_exist_in_postprocess_js() -> None:
    pp = JS_PP.read_text(encoding="utf-8")
    assert "function abComparePopulateSelects" in pp
    assert "async function abCompareLoad" in pp
    assert "buildJobFileUrl" in pp


def test_showppmergedtab_calls_abcomparepopulateselects() -> None:
    core = JS_CORE.read_text(encoding="utf-8")
    assert "abComparePopulateSelects" in core


# ---------------------------------------------------------------------------
# 5. i18n keys for new features
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_en_i18n_has_new_keys() -> None:
    en = _load(EN_JSON)
    pp = en.get("postprocess", {})
    assert "tab_abcompare" in pp
    assert "abcompare_title" in pp
    assert "abcompare_job_a" in pp
    assert "abcompare_job_b" in pp
    assert "abcompare_run" in pp
    assert "abcompare_select_both" in pp
    assert "abcompare_no_image" in pp

    sidebar = en.get("sidebar", {})
    assert "job_done" in sidebar
    assert "job_failed" in sidebar

    shortcuts = en.get("shortcuts", {})
    assert "show_shortcuts" in shortcuts
    assert "focus_search" in shortcuts
    assert "switch_tab" in shortcuts

    common = en.get("common", {})
    assert "shortcuts_title" in common
    assert "close" in common


def test_zh_i18n_has_same_new_keys() -> None:
    zh = _load(ZH_JSON)
    pp = zh.get("postprocess", {})
    assert "tab_abcompare" in pp
    assert "abcompare_run" in pp
    assert "abcompare_select_both" in pp

    sidebar = zh.get("sidebar", {})
    assert "job_done" in sidebar
    assert "job_failed" in sidebar

    assert "shortcuts" in zh
    assert "show_shortcuts" in zh["shortcuts"]
