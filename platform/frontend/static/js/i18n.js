/**
 * TensorLBM Platform – Lightweight i18n engine
 *
 * Usage:
 *   t('key.path')                   – translate a key
 *   i18n.switch('zh')               – switch language (persisted to localStorage)
 *   i18n.init()                     – call once on DOMContentLoaded
 *
 * HTML attributes:
 *   data-i18n="key"                 – replaces textContent
 *   data-i18n-html="key"            – replaces innerHTML (use for rich text)
 *   data-i18n-placeholder="key"     – replaces placeholder attribute
 *   data-i18n-title="key"           – replaces title attribute
 */
(function () {
  'use strict';

  const STORAGE_KEY = 'tensorlbm_lang';
  const SUPPORTED   = ['en', 'zh'];
  const FALLBACK    = 'en';

  let _dict = {};
  let _lang = FALLBACK;

  // ── Helpers ──────────────────────────────────────────────────────────────

  /** Detect preferred language: localStorage → browser navigator → fallback. */
  function detect() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && SUPPORTED.includes(saved)) return saved;
    const nav = (navigator.language || navigator.userLanguage || FALLBACK).toLowerCase();
    if (nav.startsWith('zh')) return 'zh';
    return FALLBACK;
  }

  /** Fetch and parse the locale JSON file. */
  async function load(lang) {
    const r = await fetch('/static/i18n/' + lang + '.json');
    if (!r.ok) throw new Error('i18n: failed to load ' + lang + '.json (' + r.status + ')');
    return r.json();
  }

  /** Apply all data-i18n* attributes in the document. */
  function applyDOM() {
    // Text content
    document.querySelectorAll('[data-i18n]').forEach(function (el) {
      const v = t(el.getAttribute('data-i18n'));
      if (v !== el.getAttribute('data-i18n')) el.textContent = v;
    });
    // Inner HTML (for keys that contain markup like <strong>, <em>, <code>)
    document.querySelectorAll('[data-i18n-html]').forEach(function (el) {
      const v = t(el.getAttribute('data-i18n-html'));
      if (v !== el.getAttribute('data-i18n-html')) el.innerHTML = v;
    });
    // Placeholder attribute
    document.querySelectorAll('[data-i18n-placeholder]').forEach(function (el) {
      const v = t(el.getAttribute('data-i18n-placeholder'));
      if (v !== el.getAttribute('data-i18n-placeholder')) el.placeholder = v;
    });
    // Title attribute
    document.querySelectorAll('[data-i18n-title]').forEach(function (el) {
      const v = t(el.getAttribute('data-i18n-title'));
      if (v !== el.getAttribute('data-i18n-title')) el.title = v;
    });
    // <html lang>
    document.documentElement.lang = _lang === 'zh' ? 'zh-CN' : 'en';
    // <title>
    const pageTitle = t('title');
    if (pageTitle !== 'title') document.title = pageTitle;
    // Highlight active switcher button
    document.querySelectorAll('.lang-btn').forEach(function (btn) {
      btn.classList.toggle('active-lang', btn.dataset.lang === _lang);
    });
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Translate a dot-notation key.
   * Returns the key itself if not found (graceful degradation).
   * @param {string} key
   * @returns {string}
   */
  window.t = function (key) {
    if (!key) return '';
    const parts = key.split('.');
    let node = _dict;
    for (const p of parts) {
      if (node && typeof node === 'object' && Object.prototype.hasOwnProperty.call(node, p)) {
        node = node[p];
      } else {
        return key; // key not found – return the key itself
      }
    }
    return typeof node === 'string' ? node : key;
  };

  window.i18n = {
    /** Return the currently active language code. */
    lang: function () { return _lang; },

    /**
     * Switch to a new language, persist the choice, and re-apply translations.
     * @param {string} lang  One of SUPPORTED locales.
     */
    switch: async function (lang) {
      if (!SUPPORTED.includes(lang)) return;
      try {
        _dict = await load(lang);
        _lang = lang;
        localStorage.setItem(STORAGE_KEY, lang);
        applyDOM();
        // Re-render any JS-built panels that depend on t()
        if (typeof onSimTypeChange === 'function') onSimTypeChange();
        if (typeof onCADHullTypeChange === 'function') onCADHullTypeChange();
        // Update WS status label to current connection state
        const wsEl = document.getElementById('ws-status');
        if (wsEl) {
          const dot = wsEl.querySelector('.dot');
          if (dot) {
            if (dot.classList.contains('dot-completed')) {
              wsEl.innerHTML = '<span class="dot dot-completed"></span> ' + t('ws.connected');
            } else if (dot.classList.contains('dot-failed')) {
              wsEl.innerHTML = '<span class="dot dot-failed"></span> ' + t('ws.disconnected');
            } else {
              wsEl.innerHTML = '<span class="dot dot-queued"></span> ' + t('ws.connecting');
            }
          }
        }
      } catch (e) {
        console.warn('i18n.switch failed:', e);
      }
    },

    /**
     * Initialise: detect language, load dictionary, apply translations.
     * Call once inside DOMContentLoaded.
     */
    init: async function () {
      _lang = detect();
      try {
        _dict = await load(_lang);
      } catch (e) {
        // Fallback: try English
        if (_lang !== FALLBACK) {
          _lang = FALLBACK;
          try { _dict = await load(FALLBACK); } catch (_) { /* silent */ }
        }
      }
      applyDOM();
    },
  };
})();
