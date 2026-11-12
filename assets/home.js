/* The catalogue is rendered at build time. These controls progressively filter it. */
(() => {
  'use strict';
  const root = document.getElementById('editorial-refined');
  if (!root) return;
  const input = document.getElementById('collection-search');
  const goal = document.getElementById('collection-goal');
  const buttons = root.querySelectorAll('[data-kind]');
  const resources = Array.from(root.querySelectorAll('.resource'));
  const reset = document.getElementById('collection-reset');
  let kind = '';
  function update() {
    const terms = input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    let count = 0;
    resources.forEach(item => {
      const matches = (!kind || item.dataset.type === kind) &&
        (!goal.value || item.dataset.routes.split(' ').includes(goal.value)) &&
        terms.every(term => item.textContent.toLowerCase().includes(term));
      item.hidden = !matches;
      if (matches) count++;
    });
    document.getElementById('collection-count').textContent = `Showing ${count} of ${resources.length} resources`;
    document.getElementById('collection-empty').hidden = count !== 0;
    reset.hidden = !kind && !goal.value && !input.value;
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.kind === kind)));
  }
  input.addEventListener('input', update);
  goal.addEventListener('change', update);
  buttons.forEach(button => button.addEventListener('click', () => { kind = button.dataset.kind; update(); }));
  reset.addEventListener('click', () => { kind = ''; goal.value = ''; input.value = ''; update(); input.focus(); });
  document.getElementById('deep-search').addEventListener('click', () => window.MLLab.open());
  update();
})();
