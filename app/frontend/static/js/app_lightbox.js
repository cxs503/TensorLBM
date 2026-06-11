// Lightbox
// ============================================================
function openLightbox(src) {
  const resolved = new URL(String(src), window.location.origin);
  if (resolved.origin !== window.location.origin) return;
  document.getElementById('lightbox-img').src = resolved.toString();
  document.getElementById('lightbox').classList.add('show');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('show');
}
