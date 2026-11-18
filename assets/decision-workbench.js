/* =====================================================================
   ML_LAB — Decision Workbench Helper
   ---------------------------------------------------------------------
   1. Deep-link BFS trail reconstruction + URL hash sync for decision trees
   2. Markdown Architecture Decision Record (ADR) export & clipboard copy
   3. Light / Dark theme toggle for Strategy Workbenches (10 & 11)
   ===================================================================== */

(function () {
  'use strict';

  var THEME_KEY = 'ml_lab_wb_theme_v1';

  function buildPathToNode(tree, targetId) {
    if (!tree || !targetId || targetId === 'start' || !tree[targetId]) return [];
    var queue = [{ id: 'start', trail: [] }];
    var visited = { start: true };
    while (queue.length > 0) {
      var cur = queue.shift();
      var node = tree[cur.id];
      if (!node) continue;
      var edges = []
        .concat(Array.isArray(node.options) ? node.options : [])
        .concat(Array.isArray(node.continueOptions) ? node.continueOptions : []);
      for (var i = 0; i < edges.length; i++) {
        var opt = edges[i];
        if (!opt || !opt.next) continue;
        var nextTrail = cur.trail.concat([{ node: cur.id, choice: opt.label }]);
        if (opt.next === targetId) return nextTrail;
        if (!visited[opt.next] && tree[opt.next]) {
          visited[opt.next] = true;
          queue.push({ id: opt.next, trail: nextTrail });
        }
      }
    }
    return [];
  }

  function syncHash(nodeId) {
    if (!window.history || !window.history.replaceState) return;
    var base = window.location.pathname + window.location.search;
    if (!nodeId || nodeId === 'start') {
      window.history.replaceState(null, '', base);
    } else {
      window.history.replaceState(null, '', base + '#' + encodeURIComponent(nodeId));
    }
  }

  function buildMarkdownRecord(opts) {
    var title = opts.title || document.title.replace(/\s+—\s+ML_LAB.*$/, '');
    var nodeId = opts.nodeId || 'start';
    var data = opts.nodeData || {};
    var path = Array.isArray(opts.path) ? opts.path : [];
    var tree = opts.tree || {};
    var lines = [];

    lines.push('# Decision Record: ' + title);
    lines.push('');
    lines.push('- **Node:** `' + nodeId + '`');
    lines.push('- **Link:** ' + window.location.origin + window.location.pathname + '#' + encodeURIComponent(nodeId));
    lines.push('- **Date:** ' + new Date().toISOString().slice(0, 10));
    lines.push('');

    if (path.length > 0) {
      lines.push('## Decision Path');
      path.forEach(function (step, idx) {
        var q = (tree[step.node] && tree[step.node].question) || step.node;
        lines.push((idx + 1) + '. ' + q + ' -> **' + step.choice + '**');
      });
      lines.push('');
    }

    lines.push('## Outcome: ' + (data.question || nodeId));
    if (data.goal) lines.push('**Goal:** ' + data.goal);
    if (data.info) lines.push('> ' + data.info);
    if (data.warning) lines.push('**Warning:** ' + data.warning);
    lines.push('');

    var items = data.algorithms || data.techniques || data.metrics || data.sections || [];
    if (items.length > 0) {
      lines.push('## Recommendations');
      items.forEach(function (item) {
        var name = item.name || item.title || 'Item';
        var meta = item.priority || item.category || '';
        lines.push('### ' + name + (meta ? ' (' + meta + ')' : ''));
        if (item.description) lines.push(item.description);
        if (item.body) lines.push(item.body);
        if (item.formula) lines.push('- **Formula:** `' + item.formula + '`');
        if (item.bestFor) lines.push('- **Best for:** ' + item.bestFor);
        if (item.whenToUse) lines.push('- **When to use:** ' + item.whenToUse);
        if (item.sklearn) lines.push('- **Library / API:** `' + item.sklearn + '`');
        lines.push('');
      });
    }

    if (data.additionalInfo && Array.isArray(data.additionalInfo.points)) {
      lines.push('### ' + (data.additionalInfo.title || 'Notes'));
      data.additionalInfo.points.forEach(function (pt) {
        lines.push('- ' + pt);
      });
      lines.push('');
    }

    return lines.join('\n').trim() + '\n';
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve) {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.left = '-9999px';
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); } catch (e) {}
      document.body.removeChild(ta);
      resolve();
    });
  }

  /* ---------------------------------------------------------------- *
   * Strategy Workbench Theme Toggle (10 & 11)
   * ---------------------------------------------------------------- */
  function initStrategyTheme() {
    var path = window.location.pathname || '';
    var isStrategy =
      path.indexOf('ai_opportunity_scorer') !== -1 ||
      path.indexOf('ai_solution_navigator') !== -1;
    if (!isStrategy) return;

    document.body.classList.add('archetype-strategy');
    var saved = 'light';
    try {
      var stored = localStorage.getItem(THEME_KEY);
      if (stored === 'dark' || stored === 'light') saved = stored;
    } catch (e) {}
    document.documentElement.setAttribute('data-wb-theme', saved);

    var sunIcon = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>';
    var moonIcon = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

    function mountToggle() {
      if (document.getElementById('wb-theme-toggle-btn')) return;
      var bar = document.createElement('div');
      bar.className = 'wb-theme-bar';
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.id = 'wb-theme-toggle-btn';
      btn.className = 'wb-theme-toggle';

      function syncLabel(theme) {
        btn.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
        btn.innerHTML = (theme === 'light' ? moonIcon + '<span>Dark Studio</span>' : sunIcon + '<span>Light Workbench</span>');
      }

      syncLabel(saved);
      btn.addEventListener('click', function () {
        var cur = document.documentElement.getAttribute('data-wb-theme') === 'dark' ? 'dark' : 'light';
        var next = cur === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-wb-theme', next);
        try { localStorage.setItem(THEME_KEY, next); } catch (e) {}
        syncLabel(next);
      });

      bar.appendChild(btn);
      var root = document.getElementById('root');
      if (root && root.parentNode) {
        root.parentNode.insertBefore(bar, root);
      }
    }

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', mountToggle);
    } else {
      mountToggle();
    }
  }

  window.MLDecisionWorkbench = {
    buildPathToNode: buildPathToNode,
    syncHash: syncHash,
    buildMarkdownRecord: buildMarkdownRecord,
    copyText: copyText
  };

  initStrategyTheme();
})();
