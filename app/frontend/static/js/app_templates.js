/**
 * app_templates.js – TensorLBM Engineering Templates Panel
 *
 * Renders the catalogue of pre-configured engineering simulation templates
 * (scenario wizard for guided simulation setup).
 * Selecting a template pre-fills the Solve panel form and navigates there.
 */
"use strict";

let _pf_templates = [];
let _pf_template_categories = {};

/* =========================================================================
   Public entry point
   ========================================================================= */

/** Load and render templates. Called when the Templates tab is activated. */
async function templatesInit() {
  const containerEl = document.getElementById("pf-templates-container");
  if (!containerEl) return;
  containerEl.innerHTML = `<div class="text-muted p-3" data-i18n="templates.loading">${t("templates.loading")}</div>`;

  try {
    const resp = await fetch("/api/templates/");
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    _pf_templates = data.templates || [];
    _pf_template_categories = data.categories || {};
    _renderTemplates("all");
  } catch (e) {
    containerEl.innerHTML = `<div class="alert alert-danger">${e.message}</div>`;
  }
}

/* =========================================================================
   Render
   ========================================================================= */

function _renderTemplates(categoryFilter) {
  const containerEl = document.getElementById("pf-templates-container");
  if (!containerEl) return;

  const filtered = categoryFilter === "all"
    ? _pf_templates
    : _pf_templates.filter(t => t.category === categoryFilter);

  if (!filtered.length) {
    containerEl.innerHTML = `<div class="text-muted p-3">No templates in this category.</div>`;
    return;
  }

  containerEl.innerHTML = filtered.map(_templateCard).join("");
  i18n.apply(containerEl);
}

function _templateCard(tmpl) {
  const diffColor = { beginner: "success", intermediate: "warning", advanced: "danger" };
  const col = diffColor[tmpl.difficulty] || "secondary";
  const icon = tmpl.icon || "bi-play-circle";
  const refs = (tmpl.references || []).map(r => `<li>${_tEsc(r)}</li>`).join("");
  const configJson = JSON.stringify(tmpl.default_config || {}, null, 2);
  const catLabel = _pf_template_categories[tmpl.category] || tmpl.category;
  // Use Chinese description if i18n language is zh
  const isZh = typeof i18n !== "undefined" && i18n.currentLang && i18n.currentLang() === "zh";
  const desc = isZh && tmpl.description_zh ? tmpl.description_zh : (tmpl.description || "");
  const title = isZh && tmpl.title_zh ? tmpl.title_zh : tmpl.title;

  return `
<div class="card mb-3">
  <div class="card-header d-flex justify-content-between align-items-center">
    <span><i class="bi ${_tEsc(icon)} me-2"></i><strong>${_tEsc(title)}</strong></span>
    <span class="badge bg-${col}">${t("templates." + tmpl.difficulty) || tmpl.difficulty}</span>
  </div>
  <div class="card-body">
    <div class="text-muted small mb-2">
      <i class="bi bi-tag me-1"></i>${_tEsc(catLabel)}
      &nbsp;|&nbsp;
      <i class="bi bi-cpu me-1"></i>${t("templates.solver_type")}: <code>${_tEsc(tmpl.solver_type)}</code>
    </div>
    <p class="small mb-2">${_tEsc(desc)}</p>
    ${refs ? `<div class="small text-muted mb-2"><strong>${t("templates.references")}:</strong><ul class="mb-0">${refs}</ul></div>` : ""}
    <details class="small">
      <summary class="text-muted" style="cursor:pointer">${t("templates.default_config")}</summary>
      <pre class="mt-1" style="background:#f8f9fa;border-radius:.3rem;padding:.5rem;font-size:.75rem">${_tEsc(configJson)}</pre>
    </details>
    <div class="mt-2">
      <button class="btn btn-sm btn-primary" onclick="templatesUse('${_tEsc(tmpl.id)}')">
        <i class="bi bi-lightning-fill me-1"></i>${t("templates.use_template")}
      </button>
    </div>
  </div>
</div>`;
}

/* =========================================================================
   Category filter
   ========================================================================= */

function templatesFilterCategory(cat) {
  _renderTemplates(cat);
  // Update active button
  document.querySelectorAll(".pf-cat-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.cat === cat);
  });
}

/* =========================================================================
   Use template → navigate to Solve and pre-fill form
   ========================================================================= */

async function templatesUse(templateId) {
  try {
    const resp = await fetch(`/api/templates/${templateId}`);
    if (!resp.ok) throw new Error(await resp.text());
    const tmpl = await resp.json();
    _applyTemplateToSolvePanel(tmpl);
    // Navigate to Solve tab
    const solveLink = document.querySelector('[data-tab="solve"]');
    if (solveLink) showTab("solve", solveLink);
  } catch (e) {
    alert("Could not load template: " + e.message);
  }
}

/** Apply a template's default_config to the Solve panel inputs. */
function _applyTemplateToSolvePanel(tmpl) {
  const cfg = tmpl.default_config || {};
  const solverType = tmpl.solver_type || "";
  const resolvedSolverType = solverType === "ship_hull_flow" ? "ship_hull" : solverType;

  // Show a brief notification in the solve panel
  const isZh = typeof i18n !== "undefined" && i18n.currentLang && i18n.currentLang() === "zh";
  const title = isZh && tmpl.title_zh ? tmpl.title_zh : tmpl.title;
  loadSolveConfiguration(resolvedSolverType, cfg, {
    message: `✓ Template loaded: "${title}"`,
  });
}

/* =========================================================================
   Utility
   ========================================================================= */

function _tEsc(s) {
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}
