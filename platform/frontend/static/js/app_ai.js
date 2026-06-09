// Benchmarks
// ============================================================
function updateBenchStatus(job) {
  const type = job.job_type;   // e.g. "benchmark_marine"
  if (!type.startsWith('benchmark_')) return;
  const bench = type.replace('benchmark_', '');
  benchJobMap[bench] = job.job_id;
  const badge = document.getElementById(`bench-status-${bench}`);
  const card = document.getElementById(`bench-card-${bench}`);
  if (!badge || !card) return;
  badge.textContent = job.status;
  const cls = {queued:'secondary',running:'warning',completed:'success',failed:'danger'}[job.status]||'secondary';
  badge.className = `badge bg-${cls} ms-auto`;
  card.className = `card bench-card ${job.status}`;

  // Show results when completed
  if (job.status === 'completed') {
    const resEl = document.getElementById(`bench-result-${bench}`);
    if (resEl && job.result) {
      resEl.innerHTML = `<pre class="bg-light border rounded p-2 small mb-0" style="max-height:120px;overflow:auto">${escHtml(JSON.stringify(job.result, null, 2))}</pre>`;
    }
  }
}

async function runBenchmark(bench) {
  let body, endpoint;
  if (bench === 'marine') {
    const fast = document.getElementById('bench-marine-fast').value === 'true';
    const device = document.getElementById('bench-marine-device').value;
    body = { fast, device, cases: ['cylinder','sloshing','pipeline','turbulent_channel','wigley','suboff','geometry_library'] };
    endpoint = '/api/benchmarks/marine';
  } else if (bench === 'multiphase') {
    const fast = document.getElementById('bench-multiphase-fast').value === 'true';
    const device = document.getElementById('bench-multiphase-device').value;
    body = { fast, device };
    endpoint = '/api/benchmarks/multiphase';
  } else if (bench === 'accuracy') {
    const fast = document.getElementById('bench-accuracy-fast').value === 'true';
    const device = document.getElementById('bench-accuracy-device').value;
    body = { fast, device, cases: ['cavity', 'bfs', 'rotating_cylinder'] };
    endpoint = '/api/benchmarks/accuracy';
  } else if (bench === 'ghia') {
    const nx = +document.getElementById('bench-ghia-nx').value;
    const re = +document.getElementById('bench-ghia-re').value;
    const n_steps = +document.getElementById('bench-ghia-steps').value;
    body = { nx, re, n_steps, device: 'cpu' };
    endpoint = '/api/benchmarks/ghia';
  } else if (bench === 'mlups') {
    const sizes = document.getElementById('bench-mlups-sizes').value.split(',').map(s => +s.trim());
    const steps = +document.getElementById('bench-mlups-steps').value;
    body = { sizes, steps, device: 'cpu' };
    endpoint = '/api/benchmarks/mlups';
  } else if (bench === 'porous') {
    const fast = document.getElementById('bench-porous-fast').value === 'true';
    const device = document.getElementById('bench-porous-device').value;
    body = { fast, device };
    endpoint = '/api/benchmarks/porous';
  }
  try {
    const r = await api('POST', endpoint, body);
    benchJobMap[bench] = r.job_id;
    showToast(`${t('bench.submitted')} (${r.job_id})`, 'success');
  } catch(e) {
    showToast(`${t('solve.error')} ${e.message}`, 'danger');
  }
}

// ============================================================
// AI Flow Transformer
// ============================================================
let aiFlowActiveJobId = null;
let aiFlowPollToken = 0;

function aiFlowSetResult(obj) {
  const el = document.getElementById('aiflow-result');
  if (!el) return;
  el.textContent = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2);
}

function aiFlowRenderHistory(history) {
  const canvas = document.getElementById('aiflow-history-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const width = Math.max(360, Math.floor(canvas.getBoundingClientRect().width || 360));
  const height = Number(canvas.getAttribute('height') || 180);
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  const rows = Array.isArray(history) ? history.filter(h => h && Number.isFinite(Number(h.train_loss))) : [];
  if (!rows.length) {
    ctx.fillStyle = '#6c757d';
    ctx.font = '12px sans-serif';
    ctx.fillText('No training history', 12, 24);
    return;
  }
  const values = rows.flatMap(row => [Number(row.train_loss), Number(row.val_loss)]).filter(Number.isFinite);
  let yMin = Math.min(...values);
  let yMax = Math.max(...values);
  if (Math.abs(yMax - yMin) < 1e-12) {
    yMin -= 1;
    yMax += 1;
  }
  const pad = { left: 38, right: 12, top: 12, bottom: 24 };
  const xScale = (idx) => {
    if (rows.length === 1) return (width - pad.left - pad.right) / 2 + pad.left;
    return pad.left + (idx / (rows.length - 1)) * (width - pad.left - pad.right);
  };
  const yScale = (value) => pad.top + ((yMax - value) / (yMax - yMin)) * (height - pad.top - pad.bottom);
  ctx.strokeStyle = '#ced4da';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.stroke();
  const drawSeries = (key, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    rows.forEach((row, idx) => {
      const x = xScale(idx);
      const y = yScale(Number(row[key]));
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  drawSeries('train_loss', '#0d6efd');
  drawSeries('val_loss', '#dc3545');
  ctx.fillStyle = '#495057';
  ctx.font = '11px sans-serif';
  ctx.fillText(`epochs: ${rows.length}`, pad.left, height - 6);
  ctx.fillStyle = '#0d6efd';
  ctx.fillText('train', width - 84, 16);
  ctx.fillStyle = '#dc3545';
  ctx.fillText('val', width - 40, 16);
}

function aiFlowHandleJob(job) {
  if (!job) return;
  const history = Array.isArray(job.diagnostics) && job.diagnostics.length
    ? job.diagnostics.filter(d => d.kind === 'ai_transformer_epoch')
    : ((job.result && job.result.history) || []);
  aiFlowRenderHistory(history);
  aiFlowSetResult({
    job_id: job.job_id,
    status: job.status,
    error: job.error,
    epochs_reported: history.length,
    result: job.result || null,
  });
}

async function aiFlowPollJob(jobId) {
  const token = ++aiFlowPollToken;
  while (token === aiFlowPollToken && aiFlowActiveJobId === jobId) {
    try {
      const job = await api('GET', `/api/ai/transformer/train/${jobId}`);
      aiFlowHandleJob(job);
      if (['completed', 'failed', 'cancelled'].includes(job.status)) {
        if (job.status === 'completed') {
          await aiFlowListModels();
          showToast(t('ai_flow.train_done'), 'success');
        } else {
          showToast(`${t('common.error')} ${job.error || job.status}`, 'danger');
        }
        break;
      }
    } catch (e) {
      aiFlowSetResult(`Error: ${e.message}`);
      showToast(`${t('common.error')} ${e.message}`, 'danger');
      break;
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}

async function aiFlowListModels() {
  const box = document.getElementById('aiflow-models-box');
  if (!box) return;
  box.innerHTML = `<span class=\"text-muted\">${t('common.loading')}</span>`;
  try {
    const r = await api('GET', '/api/ai/transformer/models');
    if (!r.models || !r.models.length) {
      box.innerHTML = `<span class=\"text-muted\">${t('ai_flow.no_models')}</span>`;
      return;
    }
    box.innerHTML = r.models.map(m => {
      const loss = m.metrics && m.metrics.final_val_loss !== undefined
        ? `val=${Number(m.metrics.final_val_loss).toExponential(3)}`
        : '';
      return `<div class=\"mb-2\"><strong>#${m.id}</strong> ${escHtml(m.name)}<br><span class=\"text-muted\">${loss} · ${escHtml(m.path)}</span></div>`;
    }).join('');
  } catch (e) {
    box.innerHTML = `<span class=\"text-danger\">${escHtml(e.message)}</span>`;
  }
}

async function aiFlowTrain() {
  const nx = Number(document.getElementById('aiflow-nx').value || 48);
  const ny = Number(document.getElementById('aiflow-ny').value || 48);
  const epochs = Number(document.getElementById('aiflow-epochs').value || 20);
  const mask_ratio = Number(document.getElementById('aiflow-mask').value || 0.15);
  aiFlowSetResult(t('common.computing'));
  aiFlowRenderHistory([]);
  try {
    const r = await api('POST', '/api/ai/transformer/train', { nx, ny, epochs, mask_ratio });
    aiFlowActiveJobId = r.job_id;
    aiFlowSetResult(r);
    showToast(t('ai_flow.train_queued') || 'Training queued', 'info');
    aiFlowPollJob(r.job_id);
  } catch (e) {
    aiFlowSetResult(`Error: ${e.message}`);
    showToast(`${t('common.error')} ${e.message}`, 'danger');
  }
}

async function aiFlowInfer() {
  const modelIdRaw = document.getElementById('aiflow-model-id').value.trim();
  const seed = Number(document.getElementById('aiflow-seed').value || 0);
  const nx = Number(document.getElementById('aiflow-nx').value || 48);
  const ny = Number(document.getElementById('aiflow-ny').value || 48);
  const body = { nx, ny, seed };
  if (modelIdRaw) body.model_id = Number(modelIdRaw);
  aiFlowSetResult(t('common.computing'));
  try {
    const r = await api('POST', '/api/ai/transformer/infer', body);
    aiFlowSetResult(r);
    showToast(t('ai_flow.infer_done'), 'success');
  } catch (e) {
    aiFlowSetResult(`Error: ${e.message}`);
    showToast(`${t('common.error')} ${e.message}`, 'danger');
  }
}

// ============================================================
// Orchestration + AI governance
// ============================================================
function orchSetResult(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

function orchParseReValues(raw) {
  return raw.split(',')
    .map(v => Number(v.trim()))
    .filter(v => Number.isFinite(v) && v > 0);
}

async function orchLoadTemplates() {
  const sel = document.getElementById('orch-template');
  if (!sel) return;
  sel.innerHTML = `<option>${t('common.loading')}</option>`;
  try {
    const r = await api('GET', '/api/orchestration/templates');
    const templates = Array.isArray(r.templates) ? r.templates : [];
    if (!templates.length) {
      sel.innerHTML = '';
      return;
    }
    sel.innerHTML = templates.map(tp => {
      const staged = tp.implemented ? '' : ` (${t('orch.staged')})`;
      return `<option value="${escHtml(tp.template_id)}">${escHtml(tp.title || tp.template_id)}${staged}</option>`;
    }).join('');
    const active = templates.find(tp => tp.implemented) || templates[0];
    if (active) sel.value = active.template_id;
  } catch (e) {
    sel.innerHTML = '';
    showToast(`${t('common.error')} ${e.message}`, 'danger');
  }
}

async function orchSubmitExperiment() {
  const template_id = document.getElementById('orch-template')?.value || '';
  const re_values = orchParseReValues(document.getElementById('orch-re-values')?.value || '');
  const base_config = {
    nx: Number(document.getElementById('orch-nx')?.value || 160),
    ny: Number(document.getElementById('orch-ny')?.value || 60),
    n_steps: Number(document.getElementById('orch-steps')?.value || 1200),
    output_interval: Number(document.getElementById('orch-output-interval')?.value || 200),
  };
  const body = { template_id, base_config };
  if (re_values.length) body.sweep = [{ name: 're', values: re_values }];
  orchSetResult('orch-submit-result', t('common.submitting'));
  try {
    const r = await api('POST', '/api/orchestration/experiments/submit', body);
    orchSetResult('orch-submit-result', r);
    showToast(`${t('orch.submitted')} (${r.submitted})`, 'success');
  } catch (e) {
    orchSetResult('orch-submit-result', `Error: ${e.message}`);
    showToast(`${t('common.error')} ${e.message}`, 'danger');
  }
}

async function orchLoadKpis() {
  orchSetResult('orch-kpi-result', t('common.loading'));
  try {
    const r = await api('GET', '/api/orchestration/kpis');
    orchSetResult('orch-kpi-result', r);
  } catch (e) {
    orchSetResult('orch-kpi-result', `Error: ${e.message}`);
  }
}

async function orchRunConfidenceGate() {
  const body = {
    prediction: Number(document.getElementById('orch-prediction')?.value || 0),
    baseline: Number(document.getElementById('orch-baseline')?.value || 1),
    uncertainty: Number(document.getElementById('orch-uncertainty')?.value || 0),
    max_relative_error: Number(document.getElementById('orch-max-rel-err')?.value || 0.15),
    max_uncertainty: Number(document.getElementById('orch-max-uncertainty')?.value || 0.2),
  };
  orchSetResult('orch-gate-result', t('common.computing'));
  try {
    const r = await api('POST', '/api/ai/governance/confidence-gate', body);
    orchSetResult('orch-gate-result', r);
    showToast(r.accepted ? t('orch.accepted') : t('orch.fallback'), r.accepted ? 'success' : 'warning');
  } catch (e) {
    orchSetResult('orch-gate-result', `Error: ${e.message}`);
    showToast(`${t('common.error')} ${e.message}`, 'danger');
  }
}

async function orchRunActiveLearning() {
  const top_k = Number(document.getElementById('orch-top-k')?.value || 3);
  const raw = document.getElementById('orch-candidates')?.value || '[]';
  orchSetResult('orch-al-result', t('common.computing'));
  try {
    const candidates = JSON.parse(raw);
    const r = await api('POST', '/api/ai/governance/active-learning/prioritize', { top_k, candidates });
    orchSetResult('orch-al-result', r);
    showToast(`${t('orch.selected')} ${r.count}`, 'success');
  } catch (e) {
    orchSetResult('orch-al-result', `Error: ${e.message}`);
    showToast(`${t('common.error')} ${e.message}`, 'danger');
  }
}

// ============================================================
// AI Agent (LLM-driven chat)
// ============================================================
let agentHistory = [];   // [{role, content, actions?}]
let agentBusy = false;

async function loadAgentInfo() {
  try {
    const info = await api('GET', '/api/agent/info');
    const el = document.getElementById('agent-info-bar');
    if (info.llm_enabled) {
      el.innerHTML = `<i class="bi bi-stars"></i> ${t('agent.llm_enabled')} <code>${escHtml(info.llm_model)}</code> · ${info.tools_count} ${t('agent.tools')}`;
    } else {
      el.innerHTML = `<i class="bi bi-cpu"></i> ${t('agent.offline')} (${escHtml(info.fallback)}) · ${info.tools_count} ${t('agent.tools')}. ${t('agent.offline_hint')}`;
    }
  } catch (e) { /* ignore */ }
}

function agentAddMessage(role, content, actions, suggestions) {
  const wrap = document.getElementById('agent-messages');
  const div = document.createElement('div');
  div.className = 'agent-msg ' + role;
  const icon = role === 'user' ? 'bi-person-fill' : 'bi-robot';
  let actionsHtml = '';
  if (actions && actions.length) {
    actionsHtml = actions.map(a => {
      const args = Object.entries(a.args || {})
        .map(([k,v]) => `${k}=${JSON.stringify(v)}`).join(', ');
      const jobId = a.result && a.result.job_id ? a.result.job_id : null;
      const jobLine = jobId
        ? ` → <a href="#" onclick="selectJob('${jobId}');return false;">view job ${jobId.substring(0,8)}…</a>`
        : '';
      return `<div class="agent-action"><i class="bi bi-tools"></i> <strong>${escHtml(a.tool)}</strong>(${escHtml(args)})${jobLine}</div>`;
    }).join('');
  }
  let sugHtml = '';
  if (suggestions && suggestions.length) {
    sugHtml = '<div class="agent-suggestions">' +
      suggestions.map(s =>
        `<button onclick="agentSend(${JSON.stringify(s).replace(/"/g,'&quot;')})">${escHtml(s)}</button>`
      ).join('') + '</div>';
  }
  div.innerHTML = `
    <div class="avatar"><i class="bi ${icon}"></i></div>
    <div>
      <div class="bubble">${escHtml(content)}</div>
      ${actionsHtml}
      ${sugHtml}
    </div>`;
  wrap.appendChild(div);
  wrap.scrollTop = wrap.scrollHeight;
}

function agentAddTyping() {
  const wrap = document.getElementById('agent-messages');
  const div = document.createElement('div');
  div.id = 'agent-typing-row';
  div.className = 'agent-msg assistant';
  div.innerHTML = `
    <div class="avatar"><i class="bi bi-robot"></i></div>
    <div><div class="bubble agent-typing"><span class="spinner-border spinner-border-sm"></span> ${t('agent.thinking')}</div></div>`;
  wrap.appendChild(div);
  wrap.scrollTop = wrap.scrollHeight;
}

function agentRemoveTyping() {
  const el = document.getElementById('agent-typing-row');
  if (el) el.remove();
}

async function agentSubmit() {
  const input = document.getElementById('agent-input');
  const message = input.value.trim();
  if (!message) return;
  input.value = '';
  await agentSend(message);
}

async function agentSend(message) {
  if (agentBusy) return;
  agentBusy = true;
  document.getElementById('agent-send-btn').disabled = true;
  agentAddMessage('user', message, null, null);
  agentHistory.push({role: 'user', content: message});
  agentAddTyping();
  try {
    const resp = await api('POST', '/api/agent/chat', {
      message, history: agentHistory.slice(0, -1),
    });
    agentRemoveTyping();
    agentAddMessage('assistant', resp.reply, resp.actions, resp.suggestions);
    agentHistory.push({
      role: 'assistant',
      content: resp.reply,
      actions: resp.actions || [],
    });
    // Refresh job sidebar if anything was submitted
    if ((resp.actions || []).some(a => a.tool && a.tool.startsWith('submit_'))) {
      loadJobs();
    }
  } catch (e) {
    agentRemoveTyping();
    agentAddMessage('assistant', '⚠️ Error: ' + e.message, null, null);
  } finally {
    agentBusy = false;
    document.getElementById('agent-send-btn').disabled = false;
    document.getElementById('agent-input').focus();
  }
}

function agentClear() {
  agentHistory = [];
  const wrap = document.getElementById('agent-messages');
  // Keep only the first welcome message
  while (wrap.children.length > 1) wrap.removeChild(wrap.lastChild);
}

// ============================================================
