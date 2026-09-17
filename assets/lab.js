/* =====================================================================
   ML_LAB — shared shell
   ---------------------------------------------------------------------
   Injects the header, the breadcrumb and the footer into every page, and
   runs the command palette (Cmd/Ctrl+K).

   A page opts in with one line in <head>:
     <link rel="stylesheet" href="/assets/lab.css">
     <script defer src="/assets/lab.js" data-lab-page="Algorithm Selector"></script>

   data-lab-page  — the current page's title, shown in the breadcrumb.
                    Omit it on the home page.
   data-lab-root  — path back to the site root. Defaults to "/".

   The search index is fetched lazily, the first time the palette opens,
   so no page pays for it on load.
   ===================================================================== */

(function () {
  'use strict';

  var script = document.currentScript ||
    document.querySelector('script[src*="lab.js"]');
  var PAGE = script ? script.getAttribute('data-lab-page') : null;
  var ROOT = (script && script.getAttribute('data-lab-root')) || '/';

  var NAV = [
    { label: 'Decide', href: ROOT + '#routes' },
    { label: 'Playbooks', href: ROOT + '#index' },
    { label: 'Paths', href: ROOT + '#paths' },
    { label: 'About', href: ROOT + '#about' }
  ];

  var LINKS = [
    { label: 'GitHub', href: 'https://github.com/tatwan' },
    { label: 'LinkedIn', href: 'https://www.linkedin.com/in/tarekatwan' },
    { label: 'Site', href: 'https://www.tarekatwan.com' }
  ];

  /* ---------------------------------------------------------------- *
   * Tiny DOM helper
   * ---------------------------------------------------------------- */
  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === 'html') node.innerHTML = attrs[k];
        else if (k === 'text') node.textContent = attrs[k];
        else if (attrs[k] != null) node.setAttribute(k, attrs[k]);
      });
    }
    (children || []).forEach(function (c) {
      if (c) node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return node;
  }

  var ICON_MARK = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#060912" stroke-width="2.6" stroke-linecap="round" aria-hidden="true"><path d="M5 19h14M7 19V9m5 10V5m5 14v-7"/></svg>';
  var ICON_SEARCH = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5" stroke-linecap="round"/></svg>';
  var ICON_CHEVRON = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M9 5l7 7-7 7"/></svg>';

  var isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
  var HOTKEY = isMac ? '⌘K' : 'Ctrl K';

  /* ---------------------------------------------------------------- *
   * Header / breadcrumb / footer
   * ---------------------------------------------------------------- */
  function buildHeader() {
    var brand = el('a', { class: 'lab-brand', href: ROOT, 'aria-label': 'ML_LAB home' }, [
      el('span', { class: 'lab-brand__mark', html: ICON_MARK, 'aria-hidden': 'true' }),
      el('span', { class: 'lab-brand__name', text: 'ML_LAB' }),
      el('span', { class: 'lab-brand__by', text: '/ TAREK ATWAN' })
    ]);

    var nav = el('nav', { class: 'lab-nav', 'aria-label': 'Sections' },
      NAV.map(function (n) { return el('a', { href: n.href, text: n.label }); }));

    var search = el('button', {
      class: 'lab-search',
      type: 'button',
      id: 'lab-search-trigger',
      'aria-label': 'Search all playbooks (' + HOTKEY + ')'
    }, [
      el('span', { html: ICON_SEARCH, 'aria-hidden': 'true' }),
      el('span', { class: 'lab-search__label', text: 'Search everything' }),
      el('kbd', { class: 'lab-kbd', text: HOTKEY })
    ]);

    return el('header', { class: 'lab-header lab-chrome' }, [brand, nav, search]);
  }

  function buildCrumb() {
    if (!PAGE) return null;
    return el('div', { class: 'lab-crumb lab-chrome' }, [
      el('a', { href: ROOT, text: '← All playbooks' }),
      el('span', { class: 'lab-crumb__sep', text: '/' }),
      el('span', { class: 'lab-crumb__here', text: PAGE })
    ]);
  }

  function buildFooter() {
    return el('footer', { class: 'lab-footer lab-chrome' }, [
      el('span', {
        text: '© ' + new Date().getFullYear() +
          ' ML_LAB · built by Tarek Atwan · browser-based, no sign-up'
      }),
      el('span', { class: 'lab-footer__links' },
        LINKS.map(function (l) {
          return el('a', { href: l.href, text: l.label, rel: 'noopener' });
        }))
    ]);
  }

  /* ---------------------------------------------------------------- *
   * Search
   * ---------------------------------------------------------------- */
  var index = null;
  var loading = null;

  function loadIndex() {
    if (index) return Promise.resolve(index);
    if (loading) return loading;
    loading = fetch(ROOT + 'assets/search-index.json')
      .then(function (r) { return r.ok ? r.json() : { entries: [] }; })
      .then(function (data) {
        index = (data.entries || []).map(function (e) {
          return {
            t: e.t, d: e.d || '', p: e.p || '', u: e.u, k: e.k || '',
            hay: (e.t + ' ' + (e.p || '') + ' ' + (e.d || '')).toLowerCase()
          };
        });
        return index;
      })
      .catch(function () { index = []; return index; });
    return loading;
  }

  /* Words that carry no signal in a query like "when to fine-tune". */
  var STOP = {
    a: 1, an: 1, the: 1, to: 1, of: 1, for: 1, and: 1, or: 1, in: 1, on: 1,
    is: 1, it: 1, do: 1, i: 1, my: 1, with: 1, when: 1, how: 1, what: 1,
    why: 1, should: 1, does: 1, vs: 1, me: 1, best: 1, use: 1, which: 1,
    that: 1, this: 1, these: 1, there: 1, can: 1, are: 1, be: 1, will: 1,
    need: 1, want: 1, from: 1, by: 1, about: 1, if: 1, am: 1, we: 1
  };

  /**
   * Crude suffix stripping so "classes" finds "class" and "fine-tune"
   * finds "fine-tuning". Cheap, and good enough for 500 rows.
   */
  function stem(term) {
    if (term.length < 5) return term;
    return term.replace(/(ations|ation|ings|ing|ies|ed|es|s|e)$/, '');
  }

  function terminize(query) {
    var raw = query.toLowerCase().split(/[\s,]+/).filter(Boolean);
    var kept = raw.filter(function (t) { return !STOP[t]; });
    if (!kept.length) kept = raw;
    return kept.map(stem).filter(Boolean);
  }

  /** Score a row: title matches beat path matches beat body matches. */
  function score(row, terms) {
    var total = 0;
    var title = row.t.toLowerCase();
    var path = row.p.toLowerCase();
    for (var i = 0; i < terms.length; i++) {
      var term = terms[i];
      if (row.hay.indexOf(term) === -1) return 0;
      var ti = title.indexOf(term);
      if (ti === 0) total += 12;
      else if (ti > 0) total += 8;
      else if (path.indexOf(term) > -1) total += 4;
      else total += 1;
    }
    if (title.indexOf(terms.join(' ')) > -1) total += 6;
    // A whole playbook is usually a better landing than one node inside it.
    if (row.k === 'Playbook') total += 4;
    return total;
  }

  function search(query, kind) {
    var terms = terminize(query);
    if (!terms.length) return [];
    var out = [];
    for (var i = 0; i < index.length; i++) {
      var row = index[i];
      if (kind && row.k !== kind) continue;
      var s = score(row, terms);
      if (s > 0) out.push({ row: row, s: s });
    }
    out.sort(function (a, b) { return b.s - a.s; });
    return out.slice(0, 40).map(function (o) { return o.row; });
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  function highlight(text, terms) {
    var safe = escapeHtml(text);
    terms.forEach(function (term) {
      if (!term) return;
      var re = new RegExp('(' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig');
      safe = safe.replace(re, '<mark>$1</mark>');
    });
    return safe;
  }

  /* ---------------------------------------------------------------- *
   * Palette
   * ---------------------------------------------------------------- */
  var backdrop, input, results, filterBar;
  var active = 0;
  var rows = [];
  var kindFilter = null;
  var KINDS = ['Playbook', 'Foundations', 'Modeling', 'Features', 'Evaluation', 'GenAI', 'Product', 'Certification'];

  function buildPalette() {
    input = el('input', {
      class: 'lab-palette__input',
      type: 'text',
      id: 'lab-palette-input',
      placeholder: 'Search every playbook and every decision inside them…',
      autocomplete: 'off',
      spellcheck: 'false',
      'aria-controls': 'lab-palette-results',
      'aria-autocomplete': 'list'
    });

    var searchRow = el('div', { class: 'lab-palette__search' }, [
      el('span', { html: ICON_SEARCH, 'aria-hidden': 'true', style: 'color:var(--lab-amber)' }),
      input,
      el('kbd', { class: 'lab-kbd', text: 'ESC' })
    ]);

    filterBar = el('div', { class: 'lab-palette__filters' },
      [el('button', {
        class: 'lab-chip', type: 'button', 'aria-pressed': 'true', 'data-kind': ''
      }, ['All'])].concat(KINDS.map(function (k) {
        return el('button', {
          class: 'lab-chip', type: 'button', 'aria-pressed': 'false', 'data-kind': k
        }, [k]);
      })));

    filterBar.addEventListener('click', function (e) {
      var chip = e.target.closest('.lab-chip');
      if (!chip) return;
      kindFilter = chip.getAttribute('data-kind') || null;
      Array.prototype.forEach.call(filterBar.children, function (c) {
        c.setAttribute('aria-pressed', c === chip ? 'true' : 'false');
      });
      render();
      input.focus();
    });

    results = el('div', {
      class: 'lab-palette__results',
      id: 'lab-palette-results',
      role: 'listbox',
      'aria-label': 'Search results'
    });

    var hints = el('div', { class: 'lab-palette__hints' }, [
      el('span', {}, [el('kbd', { class: 'lab-kbd', text: '↑↓' }), ' navigate']),
      el('span', {}, [el('kbd', { class: 'lab-kbd', text: '↵' }), ' open']),
      el('span', {}, [el('kbd', { class: 'lab-kbd', text: 'TAB' }), ' filter by type']),
      el('span', {}, [el('kbd', { class: 'lab-kbd', text: 'ESC' }), ' close'])
    ]);

    var panel = el('div', {
      class: 'lab-palette',
      role: 'dialog',
      'aria-modal': 'true',
      'aria-label': 'Search the lab'
    }, [searchRow, filterBar, results, hints]);

    backdrop = el('div', { class: 'lab-palette-backdrop lab-chrome', hidden: 'hidden' }, [panel]);

    backdrop.addEventListener('mousedown', function (e) {
      if (e.target === backdrop) close();
    });
    input.addEventListener('input', function () { active = 0; render(); });
    input.addEventListener('keydown', onKey);

    return backdrop;
  }

  function render() {
    var q = input.value.trim();
    var terms = q ? terminize(q) : [];
    rows = q ? search(q, kindFilter) : [];
    results.innerHTML = '';

    if (!q) {
      results.appendChild(el('div', { class: 'lab-palette__empty' }, [
        'Try ', el('strong', { text: 'imbalanced classes' }), ', ',
        el('strong', { text: 'when to fine-tune' }), ' or ',
        el('strong', { text: 'RAGAS' }), '.',
        el('br'), 'Search runs across every decision node, not just page titles.'
      ]));
      return;
    }

    if (!rows.length) {
      results.appendChild(el('div', { class: 'lab-palette__empty' }, [
        'Nothing for “' + q + '”' + (kindFilter ? ' in ' + kindFilter : '') + '.',
        el('br'), 'Try a broader term, or clear the type filter.'
      ]));
      return;
    }

    results.appendChild(el('div', {
      class: 'lab-palette__group',
      text: rows.length + ' match' + (rows.length === 1 ? '' : 'es')
    }));

    rows.forEach(function (row, i) {
      var hit = el('a', {
        class: 'lab-hit',
        href: row.u,
        role: 'option',
        'data-active': i === active ? 'true' : 'false',
        'aria-selected': i === active ? 'true' : 'false'
      }, [
        el('span', { class: 'lab-hit__kind', text: row.k }),
        el('span', { class: 'lab-hit__body' }, [
          el('span', { class: 'lab-hit__title', html: highlight(row.t, terms) }),
          el('span', { class: 'lab-hit__path', html: highlight(row.p, terms) })
        ]),
        el('span', { class: 'lab-hit__chevron', html: ICON_CHEVRON, 'aria-hidden': 'true' })
      ]);
      hit.addEventListener('mouseenter', function () { setActive(i); });
      results.appendChild(hit);
    });
  }

  function setActive(i) {
    if (!rows.length) return;
    active = (i + rows.length) % rows.length;
    var hits = results.querySelectorAll('.lab-hit');
    Array.prototype.forEach.call(hits, function (h, n) {
      var on = n === active;
      h.setAttribute('data-active', on ? 'true' : 'false');
      h.setAttribute('aria-selected', on ? 'true' : 'false');
      if (on) h.scrollIntoView({ block: 'nearest' });
    });
  }

  function cycleFilter(back) {
    var chips = Array.prototype.slice.call(filterBar.children);
    var current = chips.findIndex(function (c) { return c.getAttribute('aria-pressed') === 'true'; });
    var next = (current + (back ? -1 : 1) + chips.length) % chips.length;
    chips[next].click();
  }

  function onKey(e) {
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive(active + 1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(active - 1); }
    else if (e.key === 'Tab') { e.preventDefault(); cycleFilter(e.shiftKey); }
    else if (e.key === 'Enter') {
      if (rows[active]) { e.preventDefault(); window.location.href = rows[active].u; }
    } else if (e.key === 'Escape') { e.preventDefault(); close(); }
  }

  var lastFocus = null;

  function open() {
    lastFocus = document.activeElement;
    backdrop.hidden = false;
    document.documentElement.style.overflow = 'hidden';
    input.value = '';
    active = 0;
    render();
    input.focus();
    loadIndex().then(function () { if (!backdrop.hidden) render(); });
  }

  function close() {
    backdrop.hidden = true;
    document.documentElement.style.overflow = '';
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  /* ---------------------------------------------------------------- *
   * Boot
   * ---------------------------------------------------------------- */
  function boot() {
    var body = document.body;
    if (!body || body.hasAttribute('data-lab-ready')) return;
    body.setAttribute('data-lab-ready', '');

    var main = document.getElementById('root') ||
      document.querySelector('main') ||
      body.firstElementChild;
    if (main && !main.id) main.id = 'lab-main';

    var skip = el('a', {
      class: 'lab-skip lab-chrome',
      href: '#' + ((main && main.id) || 'lab-main'),
      text: 'Skip to content'
    });

    var crumb = buildCrumb();
    body.insertBefore(buildHeader(), body.firstChild);
    if (crumb) body.insertBefore(crumb, body.firstChild.nextSibling);
    body.insertBefore(skip, body.firstChild);
    body.appendChild(buildFooter());
    body.appendChild(buildPalette());

    document.getElementById('lab-search-trigger').addEventListener('click', open);

    document.addEventListener('keydown', function (e) {
      if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
        e.preventDefault();
        backdrop.hidden ? open() : close();
      } else if (e.key === '/' && backdrop.hidden) {
        var tag = (document.activeElement && document.activeElement.tagName) || '';
        if (tag !== 'INPUT' && tag !== 'TEXTAREA' && !document.activeElement.isContentEditable) {
          e.preventDefault();
          open();
        }
      }
    });

    // Warm the index once the page is idle.
    if ('requestIdleCallback' in window) requestIdleCallback(loadIndex, { timeout: 6000 });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

  window.MLLab = { open: open, close: close };
})();
