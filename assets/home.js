/* Workbench homepage: unified deep search, goal route cards, and archetype filtering. */
(() => {
  'use strict';

  const root = document.getElementById('editorial-refined');
  if (!root) return;

  const input = document.getElementById('collection-search');
  const searchClear = document.getElementById('search-clear');
  const deepSearchBtn = document.getElementById('deep-search');
  const deepResultsEl = document.getElementById('deep-results');
  const countEl = document.getElementById('collection-count');
  const emptyEl = document.getElementById('collection-empty');
  const resetBtn = document.getElementById('collection-reset');
  const routeAllBtn = document.getElementById('route-all');
  const kbdHint = document.getElementById('wb-kbd-hint');

  const kindButtons = Array.from(root.querySelectorAll('#filters [data-kind]'));
  const routeButtons = Array.from(root.querySelectorAll('.route-card[data-route]'));
  const archButtons = Array.from(root.querySelectorAll('[data-kind-jump]'));
  const quickButtons = Array.from(root.querySelectorAll('.wb-quick-btn[data-query]'));
  const cards = Array.from(root.querySelectorAll('.resource'));

  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
  if (kbdHint) kbdHint.textContent = isMac ? '⌘K' : 'Ctrl K';

  const prefersReduced = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const cardByHref = {};
  const cardMeta = cards.map((el) => {
    const href = el.getAttribute('data-href') || '';
    cardByHref[href] = el;
    return {
      el,
      href,
      type: el.getAttribute('data-type') || '',
      routes: (el.getAttribute('data-routes') || '').split(/\s+/).filter(Boolean),
      hay: el.textContent.toLowerCase(),
      deepBox: el.querySelector('.resource__deep-hits'),
    };
  });

  let kind = '';
  let route = '';
  let showAllDeep = false;

  /* ---------------------------------------------------------------- *
   * Deep-Search Engine (shares scoring model with assets/lab.js)
   * ---------------------------------------------------------------- */
  let searchIndex = null;
  let indexPromise = null;

  function mapUrlToCardHref(u) {
    if (!u) return '';
    const clean = u.split('#')[0].replace(/^\//, '');
    if (clean.indexOf('DataEngineering/') === 0) return 'DataEngineering/index.html';
    if (clean.indexOf('GaTech/CSE%208803/') === 0 || clean.indexOf('GaTech/CSE 8803/') === 0) {
      return 'GaTech/CSE%208803/index.html';
    }
    if (clean.indexOf('aip/') === 0) return 'aip/index.html';
    if (clean.indexOf('rag-academy/') === 0) return 'rag-academy/';
    return clean;
  }

  function loadIndex() {
    if (searchIndex) return Promise.resolve(searchIndex);
    if (indexPromise) return indexPromise;
    indexPromise = fetch('/assets/search-index.json')
      .then((r) => (r.ok ? r.json() : { entries: [] }))
      .then((data) => {
        searchIndex = (data.entries || []).map((e) => ({
          t: e.t || '',
          d: e.d || '',
          p: e.p || '',
          u: e.u || '',
          k: e.k || '',
          parentHref: mapUrlToCardHref(e.u || ''),
          hay: ((e.t || '') + ' ' + (e.p || '') + ' ' + (e.d || '')).toLowerCase(),
        }));
        if (input && input.value.trim()) update();
        return searchIndex;
      })
      .catch(() => {
        searchIndex = [];
        return searchIndex;
      });
    return indexPromise;
  }

  const STOP = {
    a: 1, an: 1, the: 1, to: 1, of: 1, for: 1, and: 1, or: 1, in: 1, on: 1,
    is: 1, it: 1, do: 1, i: 1, my: 1, with: 1, when: 1, how: 1, what: 1,
    why: 1, should: 1, does: 1, vs: 1, me: 1, best: 1, use: 1, which: 1,
    that: 1, this: 1, these: 1, there: 1, can: 1, are: 1, be: 1, will: 1,
    need: 1, want: 1, from: 1, by: 1, about: 1, if: 1, am: 1, we: 1,
  };

  function stem(term) {
    if (term.length < 5) return term;
    return term.replace(/(ations|ation|ings|ing|ies|ed|es|s|e)$/, '');
  }

  function terminize(query) {
    const raw = query.toLowerCase().split(/[\s,]+/).filter(Boolean);
    let kept = raw.filter((t) => !STOP[t]);
    if (!kept.length) kept = raw;
    return kept.map(stem).filter(Boolean);
  }

  function scoreRow(row, terms) {
    let total = 0;
    const title = row.t.toLowerCase();
    const path = row.p.toLowerCase();
    for (let i = 0; i < terms.length; i++) {
      const term = terms[i];
      if (row.hay.indexOf(term) === -1) return 0;
      const ti = title.indexOf(term);
      if (ti === 0) total += 12;
      else if (ti > 0) total += 8;
      else if (path.indexOf(term) > -1) total += 4;
      else total += 1;
    }
    if (terms.length > 1 && title.indexOf(terms.join(' ')) > -1) total += 6;
    if (row.k === 'Playbook') total += 3;
    return total;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
    }[c]));
  }

  function highlight(text, terms) {
    let safe = escapeHtml(text);
    terms.forEach((term) => {
      if (!term) return;
      const re = new RegExp('(' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig');
      safe = safe.replace(re, '<mark>$1</mark>');
    });
    return safe;
  }

  function isSpaRoute(u) {
    return u &&
      u.indexOf('/rag-academy/') === 0 &&
      u !== '/rag-academy/' &&
      u !== '/rag-academy/index.html';
  }

  function navigateSpaOrFollow(e, href) {
    if (!isSpaRoute(href)) return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;
    e.preventDefault();
    try { sessionStorage.setItem('rag-academy-redirect', href); } catch (_) {}
    window.location.href = '/rag-academy/';
  }

  function runDeepSearch(query) {
    const terms = terminize(query);
    if (!terms.length || !searchIndex) return { terms, hits: [], byCard: {} };

    const allHits = [];
    const byCard = {};

    for (let i = 0; i < searchIndex.length; i++) {
      const row = searchIndex[i];
      const parentEl = cardByHref[row.parentHref];
      if (kind || route) {
        if (!parentEl) continue;
        if (kind && parentEl.getAttribute('data-type') !== kind) continue;
        if (route) {
          const rList = (parentEl.getAttribute('data-routes') || '').split(/\s+/);
          if (rList.indexOf(route) === -1) continue;
        }
      }
      const s = scoreRow(row, terms);
      if (s > 0) {
        allHits.push({ row, s });
        if (row.parentHref) {
          if (!byCard[row.parentHref]) byCard[row.parentHref] = [];
          byCard[row.parentHref].push({ row, s });
        }
      }
    }

    allHits.sort((a, b) => b.s - a.s);
    Object.keys(byCard).forEach((k) => {
      byCard[k].sort((a, b) => b.s - a.s);
    });

    return {
      terms,
      hits: allHits.map((o) => o.row),
      byCard,
    };
  }

  function renderDeepPanel(query, terms, hits) {
    if (!deepResultsEl) return;
    if (!query) {
      deepResultsEl.hidden = true;
      deepResultsEl.innerHTML = '';
      return;
    }

    if (!searchIndex) {
      deepResultsEl.hidden = false;
      deepResultsEl.innerHTML =
        '<div class="deep-results__head">' +
          '<span class="deep-results__title">Searching internal nodes…</span>' +
          '<span class="deep-results__sub">Loading 2,202 entries</span>' +
        '</div>';
      return;
    }

    if (!hits.length) {
      deepResultsEl.hidden = true;
      deepResultsEl.innerHTML = '';
      return;
    }

    const limit = showAllDeep ? 40 : 10;
    const shown = hits.slice(0, limit);

    const itemsHtml = shown.map((row) => {
      const spaAttr = isSpaRoute(row.u) ? ' data-spa="true"' : '';
      const descHtml = row.d
        ? '<span class="deep-hit__desc">' + highlight(row.d, terms) + '</span>'
        : '';
      return '<a class="deep-hit" href="' + escapeHtml(row.u) + '"' + spaAttr + '>' +
        '<span class="deep-hit__top">' +
          '<span class="deep-hit__kind">' + escapeHtml(row.k) + '</span>' +
          '<span class="deep-hit__path">' + highlight(row.p, terms) + '</span>' +
        '</span>' +
        '<span class="deep-hit__title">' + highlight(row.t, terms) + '</span>' +
        descHtml +
      '</a>';
    }).join('');

    const toggleBtn = hits.length > 10
      ? '<button type="button" class="deep-results__more" id="deep-toggle-more">' +
          (showAllDeep
            ? 'Show top 10 results'
            : 'Show all ' + Math.min(hits.length, 40) + ' of ' + hits.length + ' matches') +
        '</button>'
      : '<span></span>';

    deepResultsEl.hidden = false;
    deepResultsEl.innerHTML =
      '<div class="deep-results__head">' +
        '<span class="deep-results__title">Deep matches inside playbooks &amp; courses</span>' +
        '<span class="deep-results__sub">' + hits.length + ' ' + (hits.length === 1 ? 'match' : 'matches') + ' across 2,202 nodes</span>' +
      '</div>' +
      '<div class="deep-results__grid">' + itemsHtml + '</div>' +
      '<div class="deep-results__foot">' +
        toggleBtn +
        '<button type="button" class="deep-results__more" id="deep-open-palette">Open in Command Palette (' + (isMac ? '⌘K' : 'Ctrl K') + ')</button>' +
      '</div>';

    const toggleEl = document.getElementById('deep-toggle-more');
    if (toggleEl) {
      toggleEl.addEventListener('click', () => {
        showAllDeep = !showAllDeep;
        update();
      });
    }
    const openPaletteEl = document.getElementById('deep-open-palette');
    if (openPaletteEl) {
      openPaletteEl.addEventListener('click', () => {
        if (window.MLLab) window.MLLab.open();
      });
    }
  }

  /* ---------------------------------------------------------------- *
   * Main Update Loop
   * ---------------------------------------------------------------- */
  function update() {
    const rawQuery = input ? input.value.trim() : '';
    const cardTerms = rawQuery.toLowerCase().split(/\s+/).filter(Boolean);
    const deep = rawQuery ? runDeepSearch(rawQuery) : { terms: [], hits: [], byCard: {} };

    let visibleCount = 0;

    cardMeta.forEach((m) => {
      const matchesKind = !kind || m.type === kind;
      const matchesRoute = !route || m.routes.indexOf(route) > -1;
      const matchesText = !cardTerms.length || cardTerms.every((t) => m.hay.indexOf(t) > -1);
      const cardDeepHits = (deep.byCard[m.href] || []).map((o) => o.row);
      const matchesQuery = !rawQuery || matchesText || cardDeepHits.length > 0;

      const show = matchesKind && matchesRoute && matchesQuery;
      m.el.hidden = !show;
      if (show) visibleCount++;

      if (m.deepBox) {
        const internalHits = cardDeepHits.filter((r) => {
          const clean = r.u.replace(/^\//, '');
          return r.k !== 'Playbook' || r.u.indexOf('#') > -1 || clean !== m.href;
        });
        if (show && rawQuery && internalHits.length > 0) {
          const top = internalHits[0];
          const spaAttr = isSpaRoute(top.u) ? ' data-spa="true"' : '';
          m.deepBox.hidden = false;
          m.deepBox.innerHTML =
            '<strong>' + internalHits.length + ' internal ' +
            (internalHits.length === 1 ? 'match' : 'matches') + ':</strong> ' +
            '<a href="' + escapeHtml(top.u) + '"' + spaAttr + '>' +
            escapeHtml(top.t) + ' →</a>';
        } else {
          m.deepBox.hidden = true;
          m.deepBox.innerHTML = '';
        }
      }
    });

    renderDeepPanel(rawQuery, deep.terms, deep.hits);

    if (searchClear) searchClear.hidden = !rawQuery;

    if (countEl) {
      let summary = 'Showing ' + visibleCount + ' of ' + cards.length + ' resources';
      if (route) {
        const activeRouteBtn = routeButtons.find((b) => b.getAttribute('data-route') === route);
        const routeHeading = activeRouteBtn ? activeRouteBtn.querySelector('h3') : null;
        if (routeHeading) summary += ' · Goal: ' + routeHeading.textContent;
      }
      if (rawQuery && searchIndex && deep.hits.length) {
        summary += ' · ' + deep.hits.length + ' deep node ' + (deep.hits.length === 1 ? 'match' : 'matches');
      }
      countEl.textContent = summary;
    }

    if (emptyEl) {
      emptyEl.hidden = !(visibleCount === 0 && (!rawQuery || !deep.hits.length));
    }

    if (resetBtn) resetBtn.hidden = !kind && !route && !rawQuery;

    kindButtons.forEach((btn) => {
      btn.setAttribute('aria-pressed', String(btn.getAttribute('data-kind') === kind));
    });

    routeButtons.forEach((btn) => {
      btn.setAttribute('aria-pressed', String(btn.getAttribute('data-route') === route));
    });

    if (routeAllBtn) {
      routeAllBtn.setAttribute('aria-pressed', String(!route));
    }
  }

  function scrollToCatalogue(onlyIfFar) {
    const target = document.getElementById('index');
    if (!target) return;
    if (onlyIfFar) {
      const rect = target.getBoundingClientRect();
      if (rect.top <= window.innerHeight + 300) return;
    }
    target.scrollIntoView({ behavior: prefersReduced ? 'auto' : 'smooth', block: 'start' });
  }

  /* ---------------------------------------------------------------- *
   * Event Listeners
   * ---------------------------------------------------------------- */
  if (input) {
    input.addEventListener('focus', loadIndex);
    input.addEventListener('input', () => {
      showAllDeep = false;
      loadIndex();
      update();
    });
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && input.value) {
        e.preventDefault();
        input.value = '';
        showAllDeep = false;
        update();
      }
    });
  }

  if (searchClear) {
    searchClear.addEventListener('click', () => {
      if (input) {
        input.value = '';
        input.focus();
      }
      showAllDeep = false;
      update();
    });
  }

  kindButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      kind = btn.getAttribute('data-kind') || '';
      update();
    });
  });

  routeButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const clicked = btn.getAttribute('data-route') || '';
      const activating = route !== clicked;
      route = activating ? clicked : '';
      update();
      if (activating) scrollToCatalogue(true);
    });
  });

  if (routeAllBtn) {
    routeAllBtn.addEventListener('click', () => {
      route = '';
      update();
    });
  }

  archButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      kind = btn.getAttribute('data-kind-jump') || '';
      update();
      scrollToCatalogue(false);
    });
  });

  quickButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const q = btn.getAttribute('data-query') || '';
      if (input) {
        input.value = q;
        input.focus();
      }
      showAllDeep = false;
      loadIndex();
      update();
    });
  });

  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      kind = '';
      route = '';
      showAllDeep = false;
      if (input) {
        input.value = '';
        input.focus();
      }
      update();
    });
  }

  if (deepSearchBtn) {
    deepSearchBtn.addEventListener('click', () => {
      if (window.MLLab) window.MLLab.open();
    });
  }

  // Handle SPA sub-routes (/rag-academy/<slug>) cleanly across static sublinks & deep-search results
  root.addEventListener('click', (e) => {
    const spaLink = e.target.closest('a[data-spa="true"]');
    if (spaLink) {
      navigateSpaOrFollow(e, spaLink.getAttribute('href'));
    }
  });

  // Warm the search index on idle so typing in the unified search bar is instantaneous
  if ('requestIdleCallback' in window) {
    requestIdleCallback(loadIndex, { timeout: 2500 });
  } else {
    setTimeout(loadIndex, 300);
  }

  update();
})();
