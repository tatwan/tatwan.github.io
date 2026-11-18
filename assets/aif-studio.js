/* =====================================================================
   ML_LAB — AWS AIF-C01 Studio Tab Bar & Mastery Persistence
   ---------------------------------------------------------------------
   Links aws_ai_practitioner.html (Study Guide),
   aws_ai_practitioner_exam.html (189-Q Exam), and
   aws_visual_references.html (Visual Diagrams) under a unified studio bar
   with live cumulative domain mastery from localStorage.
   ===================================================================== */

(function () {
  'use strict';

  var MASTERY_KEY = 'aif_c01_mastery_v1';

  function readMastery() {
    try {
      var raw = localStorage.getItem(MASTERY_KEY);
      if (raw) return JSON.parse(raw);
    } catch (e) {}
    return { domains: {}, totalAnswered: 0, totalCorrect: 0, attempts: 0 };
  }

  function formatSummary(data) {
    if (!data || !data.totalAnswered) {
      return 'Mastery: 0 Qs practiced';
    }
    var pct = Math.round((data.totalCorrect / data.totalAnswered) * 100);
    return 'Mastery: ' + pct + '% (' + data.totalCorrect + '/' + data.totalAnswered + ' Qs)';
  }

  function currentKey() {
    var p = (window.location.pathname || '').toLowerCase();
    if (p.indexOf('aws_ai_practitioner_exam') !== -1) return 'exam';
    if (p.indexOf('aws_visual_references') !== -1) return 'visual';
    if (p.indexOf('aws_ai_practitioner') !== -1) return 'guide';
    return '';
  }

  function mountBar() {
    var key = currentKey();
    if (!key || document.getElementById('aif-studio-nav')) return;

    var tabs = [
      { id: 'guide', label: 'Study Guide (62 Services)', href: '/aws_ai_practitioner.html' },
      { id: 'exam', label: 'Practice Exam (189 Qs)', href: '/aws_ai_practitioner_exam.html' },
      { id: 'visual', label: 'Visual Diagrams (3 Blueprints)', href: '/aws_visual_references.html' }
    ];

    var nav = document.createElement('nav');
    nav.id = 'aif-studio-nav';
    nav.className = 'aif-studio-nav';
    nav.setAttribute('aria-label', 'AWS AIF-C01 Certification Studio');

    var inner = document.createElement('div');
    inner.className = 'aif-studio-nav__inner';

    var brand = document.createElement('div');
    brand.className = 'aif-studio-nav__brand';

    var badge = document.createElement('span');
    badge.className = 'aif-studio-nav__badge';
    badge.textContent = 'AWS AIF-C01 Studio';

    var mastery = document.createElement('span');
    mastery.id = 'aif-studio-mastery';
    mastery.className = 'aif-studio-nav__mastery';
    mastery.textContent = formatSummary(readMastery());

    brand.appendChild(badge);
    brand.appendChild(mastery);

    var tabWrap = document.createElement('div');
    tabWrap.className = 'aif-studio-nav__tabs';

    tabs.forEach(function (t) {
      var a = document.createElement('a');
      a.className = 'aif-studio-nav__tab';
      a.href = t.href;
      a.textContent = t.label;
      if (t.id === key) a.setAttribute('aria-current', 'page');
      tabWrap.appendChild(a);
    });

    inner.appendChild(brand);
    inner.appendChild(tabWrap);
    nav.appendChild(inner);

    // Place immediately after .lab-crumb if present, else after .lab-header, else top of body
    var crumb = document.querySelector('.lab-crumb');
    var header = document.querySelector('.lab-header');
    var anchor = crumb || header;
    if (anchor && anchor.parentNode) {
      anchor.parentNode.insertBefore(nav, anchor.nextSibling);
    } else if (document.body.firstChild) {
      document.body.insertBefore(nav, document.body.firstChild);
    } else {
      document.body.appendChild(nav);
    }

    window.addEventListener('aif-mastery-updated', function () {
      var el = document.getElementById('aif-studio-mastery');
      if (el) el.textContent = formatSummary(readMastery());
    });
  }

  // Wait until DOMContentLoaded (after lab.js injects .lab-header & .lab-crumb synchronously)
  function init() {
    if (document.querySelector('.lab-crumb, .lab-header')) {
      mountBar();
    } else {
      setTimeout(mountBar, 0);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
