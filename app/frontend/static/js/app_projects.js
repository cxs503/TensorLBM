/**
 * app_projects.js – PowerFlow Project/Case Management Panel
 *
 * Provides the full project → case management workflow:
 *   - List / create / delete projects
 *   - For each project: list / create / update / delete simulation cases
 *   - Link cases to solver jobs and navigate to results
 */
"use strict";

/* =========================================================================
   State
   ========================================================================= */
let _pf_projects = [];          // loaded project list
let _pf_active_project = null;  // currently open project object
let _pf_active_case = null;     // currently viewed case object

/* =========================================================================
   Public entry points called by index.html
   ========================================================================= */

/** Refresh projects list and render. Called when Projects tab becomes active. */
async function projectsInit() {
  await projectsLoad();
}

/** Load all projects from backend and render the list. */
async function projectsLoad() {
  const listEl = document.getElementById("pf-projects-list");
  if (!listEl) return;
  listEl.innerHTML = `<div class="text-muted small p-3" data-i18n="projects.loading">Loading…</div>`;
  try {
    const resp = await fetch("/api/projects/");
    if (!resp.ok) throw new Error(await resp.text());
    _pf_projects = await resp.json();
    _renderProjectsList();
  } catch (e) {
    listEl.innerHTML = `<div class="alert alert-danger small p-2">${e.message}</div>`;
  }
}

/* =========================================================================
   Project list view
   ========================================================================= */

function _renderProjectsList() {
  // Switch to list pane, hide case pane
  document.getElementById("pf-project-list-pane").style.display = "";
  document.getElementById("pf-case-pane").style.display = "none";
  _pf_active_project = null;

  const listEl = document.getElementById("pf-projects-list");
  if (!_pf_projects.length) {
    listEl.innerHTML = `<div class="text-muted small p-3" data-i18n="projects.no_projects">${t("projects.no_projects")}</div>`;
    i18n.apply(listEl);
    return;
  }
  listEl.innerHTML = _pf_projects.map(_projectCard).join("");
  i18n.apply(listEl);
}

function _projectCard(proj) {
  const tagHtml = (proj.tags || [])
    .map(tag => `<span class="badge bg-secondary me-1">${_esc(tag)}</span>`).join("");
  const caseCount = ""; // we don't load case count in list view
  return `
<div class="card mb-2 pf-project-card" style="cursor:pointer" onclick="projectsOpenProject('${_esc(proj.id)}')">
  <div class="card-body py-2 px-3">
    <div class="d-flex justify-content-between align-items-center">
      <div>
        <strong>${_esc(proj.name)}</strong>
        ${proj.owner ? `<span class="text-muted ms-2 small">${_esc(proj.owner)}</span>` : ""}
      </div>
      <div class="d-flex gap-1">
        <button class="btn btn-sm btn-outline-danger" onclick="event.stopPropagation();projectsDeleteProject('${_esc(proj.id)}')" title="${t("projects.delete")}">
          <i class="bi bi-trash"></i>
        </button>
      </div>
    </div>
    ${proj.description ? `<div class="text-muted small mt-1">${_esc(proj.description)}</div>` : ""}
    <div class="mt-1">${tagHtml}</div>
    <div class="text-muted" style="font-size:.72rem">${t("projects.created")}: ${_shortDate(proj.created_at)}</div>
  </div>
</div>`;
}

/* =========================================================================
   Create project
   ========================================================================= */

async function projectsCreate() {
  const name = document.getElementById("pf-new-project-name").value.trim();
  const desc = document.getElementById("pf-new-project-desc").value.trim();
  const owner = document.getElementById("pf-new-project-owner").value.trim();
  const tagsRaw = document.getElementById("pf-new-project-tags").value.trim();
  if (!name) return _showPfMsg("pf-project-msg", "Project name is required.", "danger");

  const tags = tagsRaw ? tagsRaw.split(",").map(s => s.trim()).filter(Boolean) : [];
  const body = { name, description: desc, owner, tags };
  try {
    const resp = await fetch("/api/projects/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(await resp.text());
    document.getElementById("pf-new-project-name").value = "";
    document.getElementById("pf-new-project-desc").value = "";
    document.getElementById("pf-new-project-owner").value = "";
    document.getElementById("pf-new-project-tags").value = "";
    _showPfMsg("pf-project-msg", "Project created.", "success");
    await projectsLoad();
  } catch (e) {
    _showPfMsg("pf-project-msg", e.message, "danger");
  }
}

async function projectsDeleteProject(projectId) {
  if (!confirm(t("projects.delete_confirm"))) return;
  try {
    const resp = await fetch(`/api/projects/${projectId}`, { method: "DELETE" });
    if (!resp.ok && resp.status !== 204) throw new Error(await resp.text());
    await projectsLoad();
  } catch (e) {
    alert(e.message);
  }
}

/* =========================================================================
   Case list view (inside a project)
   ========================================================================= */

async function projectsOpenProject(projectId) {
  const proj = _pf_projects.find(p => p.id === projectId);
  if (!proj) return;
  _pf_active_project = proj;

  // Show case pane
  document.getElementById("pf-project-list-pane").style.display = "none";
  document.getElementById("pf-case-pane").style.display = "";
  document.getElementById("pf-case-project-title").textContent = proj.name;

  await _loadCases(projectId);
}

async function _loadCases(projectId) {
  const listEl = document.getElementById("pf-cases-list");
  if (!listEl) return;
  listEl.innerHTML = `<div class="text-muted small p-3">Loading…</div>`;
  try {
    const resp = await fetch(`/api/projects/${projectId}/cases`);
    if (!resp.ok) throw new Error(await resp.text());
    const cases = await resp.json();
    if (!cases.length) {
      listEl.innerHTML = `<div class="text-muted small p-3">${t("projects.no_cases")}</div>`;
      return;
    }
    listEl.innerHTML = cases.map(_caseCard).join("");
    i18n.apply(listEl);
  } catch (e) {
    listEl.innerHTML = `<div class="alert alert-danger small p-2">${e.message}</div>`;
  }
}

function _caseCard(c) {
  const statusColors = {
    draft: "secondary", running: "warning", completed: "success", failed: "danger"
  };
  const col = statusColors[c.status] || "secondary";
  return `
<div class="card mb-2">
  <div class="card-body py-2 px-3">
    <div class="d-flex justify-content-between align-items-center">
      <div>
        <strong>${_esc(c.name)}</strong>
        <span class="badge bg-${col} ms-1 status-badge">${_esc(c.status)}</span>
        <span class="text-muted ms-1 small">${_esc(c.scenario)}</span>
      </div>
      <div class="d-flex gap-1">
        ${c.job_id
          ? `<button class="btn btn-sm btn-outline-primary" onclick="projectsViewJob('${_esc(c.job_id)}')" title="View Job"><i class="bi bi-eye"></i></button>`
          : ""}
        <button class="btn btn-sm btn-outline-danger" onclick="projectsDeleteCase('${_esc(c.id)}')" title="${t("projects.delete")}">
          <i class="bi bi-trash"></i>
        </button>
      </div>
    </div>
    ${c.description ? `<div class="text-muted small mt-1">${_esc(c.description)}</div>` : ""}
    ${c.job_id ? `<div class="small mt-1"><i class="bi bi-link-45deg"></i> Job: <code>${_esc(c.job_id)}</code></div>` : ""}
    <div class="text-muted" style="font-size:.72rem">${t("projects.updated")}: ${_shortDate(c.updated_at)}</div>
  </div>
</div>`;
}

function projectsBack() {
  _renderProjectsList();
}

/* =========================================================================
   Create case
   ========================================================================= */

async function projectsCreateCase() {
  if (!_pf_active_project) return;
  const name = document.getElementById("pf-new-case-name").value.trim();
  const desc = document.getElementById("pf-new-case-desc").value.trim();
  const scenario = document.getElementById("pf-new-case-scenario").value || "custom";
  if (!name) return _showPfMsg("pf-case-msg", "Case name is required.", "danger");

  const body = { name, description: desc, scenario, config: {} };
  try {
    const resp = await fetch(`/api/projects/${_pf_active_project.id}/cases`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(await resp.text());
    document.getElementById("pf-new-case-name").value = "";
    document.getElementById("pf-new-case-desc").value = "";
    _showPfMsg("pf-case-msg", "Case created.", "success");
    await _loadCases(_pf_active_project.id);
  } catch (e) {
    _showPfMsg("pf-case-msg", e.message, "danger");
  }
}

async function projectsDeleteCase(caseId) {
  if (!_pf_active_project) return;
  if (!confirm(t("projects.delete_confirm"))) return;
  try {
    const resp = await fetch(`/api/projects/${_pf_active_project.id}/cases/${caseId}`, {
      method: "DELETE"
    });
    if (!resp.ok && resp.status !== 204) throw new Error(await resp.text());
    await _loadCases(_pf_active_project.id);
  } catch (e) {
    alert(e.message);
  }
}

/** Navigate to the job in the Jobs sidebar and show PostProcess tab. */
function projectsViewJob(jobId) {
  // Select job in sidebar if possible
  if (typeof selectJob === "function") selectJob(jobId);
  showTab("postprocess", document.querySelector('[data-tab="postprocess"]'));
}

/* =========================================================================
   Utilities
   ========================================================================= */

function _showPfMsg(elId, msg, type) {
  const el = document.getElementById(elId);
  if (!el) return;
  el.className = `alert alert-${type} py-1 small mt-1`;
  el.textContent = msg;
  el.style.display = "";
  setTimeout(() => { el.style.display = "none"; }, 4000);
}

function _esc(s) {
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function _shortDate(iso) {
  if (!iso) return "—";
  try {
    return iso.replace("T", " ").slice(0, 16);
  } catch {
    return iso;
  }
}
