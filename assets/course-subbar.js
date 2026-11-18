/* =====================================================================
   ML_LAB — Unified Course Sub-Bar & Progress Sync
   ---------------------------------------------------------------------
   Loaded on DataEngineering/module1..10.html and GaTech/CSE 8803/week1..15.html
   Replaces the legacy .topnav with a sticky Architectural Course Sub-Bar:
   - <- Course Hub
   - Module / Week jump dropdown (<select>) with saved completion checks
   - Mark Complete toggle button synced with localStorage
   - Prev / Next step links + live reading progress bar
   ===================================================================== */

(function () {
  'use strict';

  var DE_ITEMS = [
    { id: 'foundations', file: 'module1.html', label: 'Module 01 · Foundations & Requirements' },
    { id: 'ingestion', file: 'module2.html', label: 'Module 02 · Sources & Ingestion' },
    { id: 'storage', file: 'module3.html', label: 'Module 03 · Storage & Formats' },
    { id: 'transform', file: 'module4.html', label: 'Module 04 · Transform & Quality' },
    { id: 'modeling', file: 'module5.html', label: 'Module 05 · Data Modeling' },
    { id: 'streaming', file: 'module6.html', label: 'Module 06 · Streaming & Real-Time' },
    { id: 'orchestration', file: 'module7.html', label: 'Module 07 · Orchestrate & Operate' },
    { id: 'governance', file: 'module8.html', label: 'Module 08 · Govern, Secure & Observe' },
    { id: 'tools', file: 'module9.html', label: 'Module 09 · Tools & Technology Landscape' },
    { id: 'architecture', file: 'module10.html', label: 'Module 10 · Scale, Cost & Architecture' }
  ];

  var CSE_ITEMS = [
    { id: 'week1', file: 'week1.html', label: 'Week 01 · Preprocessing & Text Representations' },
    { id: 'week2', file: 'week2.html', label: 'Week 02 · Naive Bayes & Classification Evaluation' },
    { id: 'week3', file: 'week3.html', label: 'Week 03 · Logistic Regression, SVM & Perceptron' },
    { id: 'week4', file: 'week4.html', label: 'Week 04 · SVD, Co-occurrence & GloVe' },
    { id: 'week5', file: 'week5.html', label: 'Week 05 · Neural Networks & Word2Vec' },
    { id: 'week6', file: 'week6.html', label: 'Week 06 · CNNs & RNNs for NLP' },
    { id: 'week7', file: 'week7.html', label: 'Week 07 · LSTM, GRU & Attention' },
    { id: 'week8', file: 'week8.html', label: 'Week 08 · Multimodal & Multilingual Models' },
    { id: 'week9', file: 'week9.html', label: 'Week 09 · Transformers, BERT & GPT' },
    { id: 'week10', file: 'week10.html', label: 'Week 10 · Sequence Labeling (POS & NER)' },
    { id: 'week11', file: 'week11.html', label: 'Week 11 · Post-Training Alignment (RLHF & DPO)' },
    { id: 'week12', file: 'week12.html', label: 'Week 12 · Topic Modeling (LSI & LDA)' },
    { id: 'week13', file: 'week13.html', label: 'Week 13 · GenAI, Prompting & RAG' },
    { id: 'week14', file: 'week14.html', label: 'Week 14 · Agentic AI & Reasoning' },
    { id: 'week15', file: 'week15.html', label: 'Week 15 · Multi-Agent Systems & The Road Ahead' }
  ];

  var script = document.currentScript || document.querySelector('script[src*="course-subbar.js"]');
  var course = script ? script.getAttribute('data-course') : '';
  var idxAttr = script ? parseInt(script.getAttribute('data-index'), 10) : 0;

  if (!course) {
    var p = window.location.pathname || '';
    if (p.indexOf('DataEngineering') !== -1) course = 'de';
    else if (p.indexOf('CSE') !== -1) course = 'cse8803';
  }

  var isDE = course === 'de';
  var items = isDE ? DE_ITEMS : CSE_ITEMS;
  var storageKey = isDE ? 'de-modules-progress-v2' : 'cse8803-weeks-progress-v1';
  var badgeText = isDE ? 'DE Foundations' : 'CSE 8803 NLP';
  var unitWord = isDE ? 'Module' : 'Week';

  if (!idxAttr || idxAttr < 1 || idxAttr > items.length) {
    var m = (window.location.pathname || '').match(/(?:module|week)(\d+)\.html/i);
    idxAttr = m ? parseInt(m[1], 10) : 1;
  }

  var curIdx = idxAttr - 1;
  var curItem = items[curIdx] || items[0];

  function getProgress() {
    try {
      var arr = JSON.parse(localStorage.getItem(storageKey) || '[]');
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function saveProgress(arr) {
    try { localStorage.setItem(storageKey, JSON.stringify(arr)); } catch (e) {}
  }

  function isDone(id) {
    return getProgress().indexOf(id) !== -1;
  }

  function toggleItem(id) {
    var arr = getProgress();
    if (arr.indexOf(id) !== -1) arr = arr.filter(function (x) { return x !== id; });
    else arr.push(id);
    saveProgress(arr);
  }

  function mount() {
    if (document.getElementById('course-subbar')) return;

    var bar = document.createElement('nav');
    bar.id = 'course-subbar';
    bar.className = 'course-subbar';
    bar.setAttribute('aria-label', badgeText + ' Navigation');

    var inner = document.createElement('div');
    inner.className = 'course-subbar__inner';

    // Left: Course Hub + Badge
    var left = document.createElement('div');
    left.className = 'course-subbar__left';

    var hub = document.createElement('a');
    hub.className = 'course-subbar__hub';
    hub.href = 'index.html';
    hub.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="15 18 9 12 15 6"/></svg><span>Course Hub</span>';

    var badge = document.createElement('span');
    badge.className = 'course-subbar__badge';
    badge.textContent = badgeText;

    left.appendChild(hub);
    left.appendChild(badge);

    // Center: Module/Week Jump Dropdown
    var center = document.createElement('div');
    center.className = 'course-subbar__center';

    var select = document.createElement('select');
    select.className = 'course-subbar__select';
    select.setAttribute('aria-label', 'Jump to ' + unitWord);

    function renderOptions() {
      var doneList = getProgress();
      select.innerHTML = '';
      items.forEach(function (it, i) {
        var opt = document.createElement('option');
        opt.value = it.file;
        var mark = doneList.indexOf(it.id) !== -1 ? '✓ ' : '';
        opt.textContent = mark + it.label;
        if (i === curIdx) opt.selected = true;
        select.appendChild(opt);
      });
    }
    renderOptions();

    select.addEventListener('change', function () {
      if (select.value) window.location.href = select.value;
    });
    center.appendChild(select);

    // Right: Complete toggle + Prev / Next
    var right = document.createElement('div');
    right.className = 'course-subbar__right';

    var compBtn = document.createElement('button');
    compBtn.type = 'button';
    compBtn.className = 'course-subbar__complete-btn';

    var bottomCseBtn = null;

    function syncCompleteUI() {
      var done = isDone(curItem.id);
      compBtn.classList.toggle('is-complete', done);
      compBtn.setAttribute('aria-pressed', done ? 'true' : 'false');
      compBtn.innerHTML = done
        ? '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg><span>' + unitWord + ' Complete</span>'
        : '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="3"/></svg><span>Mark Complete</span>';

      if (bottomCseBtn) {
        bottomCseBtn.classList.toggle('is-complete', done);
        bottomCseBtn.setAttribute('aria-pressed', done ? 'true' : 'false');
        bottomCseBtn.innerHTML = done
          ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg><span>Week Completed</span>'
          : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="3"/></svg><span>Mark Week Complete</span>';
      }

      // Sync existing DataEngineering bottom button if present
      if (isDE) {
        var deBtn = document.getElementById('markComplete');
        var deIcon = document.getElementById('completeIcon');
        var deText = document.getElementById('completeText');
        if (deBtn && deIcon && deText) {
          deBtn.classList.toggle('completed', done);
          deIcon.innerHTML = done ? '&#10003;' : '&#9744;';
          deText.textContent = done ? 'Completed' : 'Mark Complete';
        }
      }

      renderOptions();
    }

    function handleToggle() {
      toggleItem(curItem.id);
      syncCompleteUI();
    }

    compBtn.addEventListener('click', handleToggle);

    // If DataEngineering already defines window.toggleComplete, hook it so bottom button also updates top sub-bar
    if (isDE && typeof window.toggleComplete === 'function') {
      var origToggle = window.toggleComplete;
      window.toggleComplete = function () {
        origToggle();
        syncCompleteUI();
      };
    }

    // For CSE 8803 weeks, also inject Mark Week Complete into .page-nav
    if (!isDE) {
      var pageNav = document.querySelector('.page-nav');
      if (pageNav) {
        bottomCseBtn = document.createElement('button');
        bottomCseBtn.type = 'button';
        bottomCseBtn.className = 'cse-complete-btn';
        bottomCseBtn.addEventListener('click', handleToggle);
        // Insert between Prev and Next links if 2+ children exist
        if (pageNav.children.length >= 2) {
          pageNav.insertBefore(bottomCseBtn, pageNav.children[1]);
        } else {
          pageNav.appendChild(bottomCseBtn);
        }
      }
    }

    syncCompleteUI();

    var navWrap = document.createElement('div');
    navWrap.className = 'course-subbar__nav';

    var prev = curIdx > 0 ? items[curIdx - 1] : null;
    var next = curIdx < items.length - 1 ? items[curIdx + 1] : null;

    var prevEl = document.createElement('a');
    prevEl.className = 'course-subbar__step' + (prev ? '' : ' course-subbar__step--disabled');
    prevEl.href = prev ? prev.file : '#';
    if (!prev) prevEl.setAttribute('aria-disabled', 'true');
    prevEl.textContent = '← Prev';

    var nextEl = document.createElement('a');
    nextEl.className = 'course-subbar__step' + (next ? '' : ' course-subbar__step--disabled');
    nextEl.href = next ? next.file : '#';
    if (!next) nextEl.setAttribute('aria-disabled', 'true');
    nextEl.textContent = 'Next →';

    navWrap.appendChild(prevEl);
    navWrap.appendChild(nextEl);

    right.appendChild(compBtn);
    right.appendChild(navWrap);

    inner.appendChild(left);
    inner.appendChild(center);
    inner.appendChild(right);

    var prog = document.createElement('div');
    prog.className = 'course-subbar__progress';
    prog.setAttribute('aria-hidden', 'true');
    var progFill = document.createElement('div');
    progFill.className = 'course-subbar__progress-fill';
    prog.appendChild(progFill);

    bar.appendChild(inner);
    bar.appendChild(prog);

    // Replace legacy .topnav cleanly
    var legacy = document.querySelector('nav.topnav');
    if (legacy) {
      legacy.classList.add('course-subbar-replaced');
      legacy.parentNode.insertBefore(bar, legacy);
    } else {
      var crumb = document.querySelector('.lab-crumb');
      if (crumb && crumb.parentNode) {
        crumb.parentNode.insertBefore(bar, crumb.nextSibling);
      } else {
        document.body.insertBefore(bar, document.body.firstChild);
      }
    }

    function updateProgressBar() {
      var max = document.documentElement.scrollHeight - window.innerHeight;
      var pct = max > 0 ? Math.min(100, Math.max(0, (window.scrollY / max) * 100)) : 0;
      progFill.style.width = pct.toFixed(1) + '%';
    }
    window.addEventListener('scroll', updateProgressBar, { passive: true });
    window.addEventListener('resize', updateProgressBar);
    updateProgressBar();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
