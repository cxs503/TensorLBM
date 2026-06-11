function escHtml(s) {
  if (s === null || s === undefined) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function showToast(msg, type = 'info') {
  const id = 'toast-' + Date.now();
  const el = document.createElement('div');
  el.id = id;
  el.style.cssText = 'position:fixed;bottom:1.5rem;right:1.5rem;z-index:3000;max-width:340px';
  el.innerHTML = `<div class="toast show align-items-center text-bg-${type} border-0" role="alert">
    <div class="d-flex"><div class="toast-body">${escHtml(msg)}</div>
    <button type="button" class="btn-close btn-close-white me-2 m-auto" onclick="this.closest('.toast').remove()"></button>
    </div></div>`;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

