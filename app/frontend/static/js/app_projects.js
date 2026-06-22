/**
 * app_projects.js – TensorLBM Project/Case Management Panel
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
let _pf_cases = [];             // cases for active project
let _pf_events_bound = false;

/* =========================================================================
   Public entry points called by index.html
   ========================================================================= */

/** Refresh projects list and render. Called when Projects tab becomes active. */
async function projectsInit() {
  bindProjectsEvents();
  await projectsLoad();
}

function bindProjectsEvents() {
  if (_pf_events_bound) return;
  const panel = document.getElementById("panel-projects");
  if (!panel) return;
  _pf_events_bound = true;

  panel.addEventListener("click", (event) => {
    const actionEl = event.target && event.target.closest
      ? event.target.closest("[data-pf-action]")
      : null;
    if (!actionEl || !panel.contains(actionEl)) return;
    const action = actionEl.dataset.pfAction;
    const projectId = actionEl.dataset.pfProjectId || "";
    const caseId = actionEl.dataset.pfCaseId || "";
    const jobId = actionEl.dataset.pfJobId || "";
    switch (action) {
      case "create-project":
        projectsCreate();
        break;
      case "refresh-projects":
        projectsLoad();
        break;
      case "open-project":
        projectsOpenProject(projectId);
        break;
      case "delete-project":
        event.stopPropagation();
        projectsDeleteProject(projectId);
        break;
      case "back-to-projects":
        projectsBack();
        break;
      case "create-case":
        projectsCreateCase();
        break;
      case "refresh-cases":
        if (_pf_active_project) _loadCases(_pf_active_project.id);
        break;
      case "advance-case":
        projectsAdvanceStage(caseId);
        break;
      case "clone-case":
        projectsCloneCase(caseId);
        break;
      case "load-case":
        projectsLoadCaseToSolve(caseId);
        break;
      case "view-job":
        projectsViewJob(jobId);
        break;
      case "view-report":
        projectsViewReport(jobId);
        break;
      case "delete-case":
        projectsDeleteCase(caseId);
        break;
      case "do-clone":
        projectsDoClone();
        break;
      case "cancel-clone":
        projectsCancelClone();
        break;
      default:
        break;
    }
  });

  panel.addEventListener("keydown", (event) => {
    const card = event.target && event.target.closest
      ? event.target.closest('.pf-project-card[data-pf-action="open-project"]')
      : null;
    if (!card || !panel.contains(card)) return;
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    projectsOpenProject(card.dataset.pfProjectId || "");
  });
}

/** Load all projects from backend and render the list. */
async function projectsLoad() {
  const listEl = document.getElementById("pf-projects-list");
  if (!listEl) return;
  listEl.innerHTML = `<div class="text-muted small p-3" data-i18n="projects.loading">Loading…</div>`;
  try {
    _pf_projects = await api("GET", "/api/projects/");
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
<div class="card mb-2 pf-project-card" style="cursor:pointer" role="button" tabindex="0"
  data-pf-action="open-project" data-pf-project-id="${_esc(proj.id)}">
  <div class="card-body py-2 px-3">
    <div class="d-flex justify-content-between align-items-center">
      <div>
        <strong>${_esc(proj.name)}</strong>
        ${proj.owner ? `<span class="text-muted ms-2 small">${_esc(proj.owner)}</span>` : ""}
      </div>
      <div class="d-flex gap-1">
        <button class="btn btn-sm btn-outline-danger" data-pf-action="delete-project"
          data-pf-project-id="${_esc(proj.id)}" title="${t("projects.delete")}">
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
    await api("POST", "/api/projects/", body);
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
    await apiResponse("DELETE", `/api/projects/${projectId}`);
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
    const cases = await api("GET", `/api/projects/${projectId}/cases`);
    _pf_cases = cases;
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
  const stageColors = {
    draft: "light text-dark", setup: "info text-dark", meshed: "primary",
    solved: "success", post_processed: "dark"
  };
  const col = statusColors[c.status] || "secondary";
  const stageLabel = t(`projects.stage_${c.workflow_stage}`) || c.workflow_stage || "draft";
  const stageCls = stageColors[c.workflow_stage] || "light text-dark";
  return `
<div class="card mb-2">
  <div class="card-body py-2 px-3">
    <div class="d-flex justify-content-between align-items-center">
      <div>
        <strong>${_esc(c.name)}</strong>
        <span class="badge bg-${col} ms-1 status-badge">${_esc(c.status)}</span>
        <span class="badge bg-${stageCls} ms-1">${_esc(stageLabel)}</span>
        <span class="text-muted ms-1 small">${_esc(c.scenario)}</span>
      </div>
      <div class="d-flex gap-1">
        <button class="btn btn-sm btn-outline-secondary" data-pf-action="advance-case"
          data-pf-case-id="${_esc(c.id)}"
          title="${t('projects.advance_workflow')}">
          <i class="bi bi-arrow-right-circle"></i>
        </button>
        <button class="btn btn-sm btn-outline-info" data-pf-action="clone-case"
          data-pf-case-id="${_esc(c.id)}"
          title="${t('projects.clone_case')}">
          <i class="bi bi-copy"></i>
        </button>
        <button class="btn btn-sm btn-outline-primary" data-pf-action="load-case"
          data-pf-case-id="${_esc(c.id)}"
          title="Open in Solver">
          <i class="bi bi-sliders"></i>
        </button>
        ${c.job_id ? `
          <button class="btn btn-sm btn-outline-primary" data-pf-action="view-job"
            data-pf-job-id="${_esc(c.job_id)}" title="View Job"><i class="bi bi-eye"></i></button>
          <button class="btn btn-sm btn-outline-secondary" data-pf-action="view-report"
            data-pf-job-id="${_esc(c.job_id)}" title="Open Report"><i class="bi bi-file-earmark-text"></i></button>
        ` : ""}
        <button class="btn btn-sm btn-outline-danger" data-pf-action="delete-case"
          data-pf-case-id="${_esc(c.id)}" title="${t("projects.delete")}">
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
    await api("POST", `/api/projects/${_pf_active_project.id}/cases`, body);
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
    await apiResponse("DELETE", `/api/projects/${_pf_active_project.id}/cases/${caseId}`);
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

function projectsViewReport(jobId) {
  const input = document.getElementById("reports-job-id");
  if (input) input.value = jobId;
  const link = document.querySelector('[data-tab="reports"]');
  showTab("reports", link || null);
}

function projectsLoadCaseToSolve(caseId) {
  const projectCase = _pf_cases.find(c => c.id === caseId);
  if (!projectCase || typeof loadSolveConfiguration !== "function") return;
  const scenario = String(projectCase.scenario || "").replace(/-([a-z])/g, (_, c) => `_${c}`);
  const solverType = scenario === "ship_hull_flow" ? "ship_hull" : scenario;
  if (!loadSolveConfiguration(solverType, projectCase.config || {}, {
    message: `✓ Case loaded: "${projectCase.name}"`,
  })) {
    alert(`Unsupported solver scenario: ${projectCase.scenario}`);
    return;
  }
  const solveLink = document.querySelector('[data-tab="solve"]');
  showTab("solve", solveLink || null);
}

/* =========================================================================
   Workflow stage advancement
   ========================================================================= */

/** Advance a case to the next workflow stage. */
async function projectsAdvanceStage(caseId) {
  if (!_pf_active_project) return;
  try {
    await api("POST", `/api/projects/${_pf_active_project.id}/cases/${caseId}/advance-workflow`);
    await _loadCases(_pf_active_project.id);
  } catch (e) {
    alert(e.message);
  }
}

/* =========================================================================
   Clone Case
   ========================================================================= */

let _pf_clone_case_id = null;

function projectsCloneCase(caseId) {
  _pf_clone_case_id = caseId;
  const panel = document.getElementById('pf-clone-panel');
  if (panel) {
    panel.style.display = '';
    const nameEl = document.getElementById('pf-clone-name');
    if (nameEl) nameEl.value = '';
    const overEl = document.getElementById('pf-clone-overrides');
    if (overEl) overEl.value = '';
    const msgEl = document.getElementById('pf-clone-msg');
    if (msgEl) msgEl.style.display = 'none';
  }
}

function projectsCancelClone() {
  _pf_clone_case_id = null;
  const panel = document.getElementById('pf-clone-panel');
  if (panel) panel.style.display = 'none';
}

async function projectsDoClone() {
  if (!_pf_active_project || !_pf_clone_case_id) return;
  const name = (document.getElementById('pf-clone-name').value || '').trim() || undefined;
  const rawOverrides = (document.getElementById('pf-clone-overrides').value || '').trim();
  let config_overrides = {};
  if (rawOverrides) {
    try { config_overrides = JSON.parse(rawOverrides); }
    catch(e) {
      _showPfMsg('pf-clone-msg', 'Invalid JSON in config overrides: ' + e.message, 'danger'); return;
    }
  }
  const body = { config_overrides };
  if (name) body.name = name;
  try {
    const newCase = await api("POST", `/api/projects/${_pf_active_project.id}/cases/${_pf_clone_case_id}/clone`, body);
    _showPfMsg('pf-clone-msg', `Cloned as "${newCase.name}"`, 'success');
    projectsCancelClone();
    await _loadCases(_pf_active_project.id);
  } catch(e) {
    _showPfMsg('pf-clone-msg', e.message, 'danger');
  }
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
