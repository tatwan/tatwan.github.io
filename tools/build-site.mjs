#!/usr/bin/env node
/**
 * build-site.mjs
 *
 * One command to regenerate everything derived from assets/modules.json:
 *
 *   node tools/build-site.mjs
 *
 *   1. the hero archetype blueprint, goal route cards, resource catalogue and workflows
 *   2. the JSON-LD block inside index.html
 *   3. sitemap.xml, from the pages that actually exist on disk
 *   4. assets/search-index.json (delegates to build-search-index.mjs)
 *
 * Adding a playbook is therefore: add it to assets/modules.json, drop the
 * HTML file in, run this. Nothing else needs hand-editing.
 */

import fs from 'node:fs';
import path from 'node:path';
import url from 'node:url';
import { execFileSync } from 'node:child_process';

const HERE = path.dirname(url.fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const SITE = 'https://tatwan.github.io';

const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const write = (p, s) => fs.writeFileSync(path.join(ROOT, p), s);
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const catalogue = JSON.parse(read('assets/modules.json'));
const { modules, routes } = catalogue;

/** Replace the text between <!-- MARK:START --> and <!-- MARK:END -->. */
function fill(html, mark, body) {
  const re = new RegExp(`(<!-- ${mark}:START -->)[\\s\\S]*?(<!-- ${mark}:END -->)`);
  if (!re.test(html)) throw new Error(`marker ${mark} not found in index.html`);
  return html.replace(re, `$1\n${body}\n        $2`);
}

function fillOptional(html, mark, body) {
  const re = new RegExp(`(<!-- ${mark}:START -->)[\\s\\S]*?(<!-- ${mark}:END -->)`);
  if (!re.test(html)) return html;
  return html.replace(re, `$1\n${body}\n        $2`);
}

/* ------------------------------------------------------------------ *
 * Inline stroke SVG icon system (Design Rule 1: zero emojis)
 * ------------------------------------------------------------------ */
const ICON_ARROW = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>';

const ARCH_META = {
  'Decision tool': {
    plural: 'Decision tools',
    unit: 'tools',
    summary: '165 decision nodes · 500+ algorithms, techniques & metrics',
    icon: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="6" cy="6" r="3"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M6 9v6"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>',
  },
  Course: {
    plural: 'Courses',
    unit: 'courses',
    summary: '44 modules & weeks · 53 interactive visualizers',
    icon: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>',
  },
  'Exam prep': {
    plural: 'Exam prep',
    unit: 'studios',
    summary: '601+ practice questions · 4 cloud & data certifications',
    icon: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 11 3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>',
  },
};

const ROUTE_ICONS = {
  scope: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
  model: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
  measure: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
  llm: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/></svg>',
  exam: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="8" r="6"/><path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/></svg>',
};

const META = {
  'genai-foundations': {
    specs: ['9 decision nodes', '23 sections', '10–15 min'],
  },
  'algorithm-selector': {
    specs: ['39 decision nodes', '119 algorithms', '15 min'],
  },
  'feature-engineering': {
    specs: ['35 decision nodes', '90 techniques', '20 min'],
  },
  'metric-tree': {
    specs: ['24 decision nodes', '58 metrics', '20 min'],
  },
  'data-engineering': {
    specs: ['10 modules', '10 interactive sims', '50 quiz Qs'],
    sublinks: [
      { label: 'M1: Foundations', href: 'DataEngineering/module1.html' },
      { label: 'M4: Data Modeling', href: 'DataEngineering/module4.html' },
      { label: 'M6: Streaming & CDC', href: 'DataEngineering/module6.html' },
      { label: 'M8: Quality & Governance', href: 'DataEngineering/module8.html' },
      { label: 'M10: Capstone', href: 'DataEngineering/module10.html' },
    ],
  },
  'aws-aip-guide': {
    specs: ['62 AWS services', '42 quick-fire Qs', '30–45 min'],
  },
  'aws-aip-exam': {
    specs: ['189 timed Qs', '5 domains', '30–60 min'],
  },
  'aws-visual': {
    specs: ['3 architecture diagrams', '10 min'],
  },
  'aip-hub': {
    specs: ['13 study tools', '242 practice Qs', '65 flashcards'],
    sublinks: [
      { label: 'Domain Study Guide', href: 'aip/study-guide.html' },
      { label: '85-Q Practice Exam', href: 'aip/practice-exam.html' },
      { label: '72-Q Final Simulator', href: 'aip/final-practice.html' },
      { label: '65 Flashcards', href: 'aip/flashcards.html' },
      { label: 'Gap Cheatsheet', href: 'aip/gap-cheatsheet.html' },
    ],
  },
  dp700: {
    specs: ['15 study sections', '30 quiz Qs', '45 min'],
  },
  'opportunity-scorer': {
    specs: ['6 value dimensions', '4 verdicts', '10 min'],
  },
  'solution-navigator': {
    specs: ['7 solution classes', '6 hybrids', '10 min'],
  },
  'genai-techniques': {
    specs: ['29 decision nodes', '84 LLM techniques', '25 min'],
  },
  'fm-metrics': {
    specs: ['29 decision nodes', '126 metrics', '20 min'],
  },
  'rag-academy': {
    specs: ['19 modules', '12 RAG architectures', 'Live playground'],
    sublinks: [
      { label: 'Hybrid RAG', href: '/rag-academy/hybrid-rag', spa: true },
      { label: 'GraphRAG', href: '/rag-academy/graph-rag', spa: true },
      { label: 'Agentic RAG', href: '/rag-academy/agentic-rag', spa: true },
      { label: 'Formal Math', href: '/rag-academy/math', spa: true },
      { label: 'Live Playground', href: '/rag-academy/playground', spa: true },
    ],
  },
  cse8803: {
    specs: ['15 graduate weeks', '24 live demos', '65 quiz Qs'],
    sublinks: [
      { label: 'W1: Foundations', href: 'GaTech/CSE%208803/week1.html' },
      { label: 'W5: Transformers', href: 'GaTech/CSE%208803/week5.html' },
      { label: 'W11: RLHF & DPO', href: 'GaTech/CSE%208803/week11.html' },
      { label: 'W13: RAG & Agents', href: 'GaTech/CSE%208803/week13.html' },
      { label: 'W15: GraphRAG', href: 'GaTech/CSE%208803/week15.html' },
    ],
  },
  pl300: {
    specs: ['140 practice Qs', '6 exam formats', '11 skills guide'],
  },
};

/* ------------------------------------------------------------------ *
 * 1. Hero Workbench Blueprint, Goal Route Cards, Catalogue, Workflows
 * ------------------------------------------------------------------ */
function heroBlueprint() {
  const total = modules.length;
  const order = ['Decision tool', 'Course', 'Exam prep'];
  const counts = Object.fromEntries(order.map((k) => [k, modules.filter((m) => m.type === k).length]));

  const mixSegments = order.map((k) => {
    const pct = ((counts[k] / total) * 100).toFixed(2);
    const cls = k === 'Decision tool' ? 'tool' : k === 'Course' ? 'course' : 'exam';
    return `<i class="wb-mix__${cls}" style="width:${pct}%" title="${esc(ARCH_META[k].plural)}: ${counts[k]}"></i>`;
  }).join('');

  const rows = order.map((k) => {
    const meta = ARCH_META[k];
    const cls = k === 'Decision tool' ? 'tool' : k === 'Course' ? 'course' : 'exam';
    return `<button type="button" class="wb-arch-row wb-arch-row--${cls}" data-kind-jump="${esc(k)}">` +
      `<span class="wb-arch-row__icon">${meta.icon}</span>` +
      `<span class="wb-arch-row__body">` +
        `<span class="wb-arch-row__title">${esc(meta.plural)}<span class="wb-arch-row__count">${counts[k]} ${esc(meta.unit)}</span></span>` +
        `<span class="wb-arch-row__desc">${esc(meta.summary)}</span>` +
      `</span>` +
      `<span class="wb-arch-row__arrow">${ICON_ARROW}</span>` +
    `</button>`;
  }).join('\n');

  return `<div class="wb-blueprint__head">` +
    `<div class="eyebrow">Workbench Architecture</div>` +
    `<h2 id="hero-path-title">${total} Playbooks Across 3 Archetypes</h2>` +
    `</div>` +
    `<div class="wb-mix" aria-hidden="true">${mixSegments}</div>` +
    `<div class="wb-arch-list" role="group" aria-label="Jump to archetype in catalogue">\n${rows}\n</div>`;
}

function goalRoutes() {
  const cards = routes.map((r, i) => {
    const count = modules.filter((m) => m.routes.includes(r.id)).length;
    const num = String(i + 1).padStart(2, '0');
    const icon = ROUTE_ICONS[r.id] || '';
    const countText = `${count} ${count === 1 ? 'playbook' : 'playbooks'}`;
    return `<button type="button" class="route-card" data-route="${esc(r.id)}" aria-pressed="false">` +
      `<span class="route-card__top">` +
        `<span class="route-card__num">${num}</span>` +
        `<span class="route-card__icon">${icon}</span>` +
        `<span class="route-card__count">${esc(countText)}</span>` +
      `</span>` +
      `<h3>${esc(r.label)}</h3>` +
      `<p>${esc(r.blurb)}</p>` +
    `</button>`;
  }).join('\n');

  return `<div class="route-grid" role="group" aria-label="Filter playbooks by practitioner goal">\n${cards}\n</div>`;
}

function resourceCatalogue() {
  const cards = modules.map((m) => {
    const meta = META[m.id] || { specs: [m.time] };
    const arch = ARCH_META[m.type] || { icon: '' };
    const cls = m.type === 'Decision tool' ? 'tool' : m.type === 'Course' ? 'course' : 'exam';
    const sublinksHtml = meta.sublinks && meta.sublinks.length
      ? `<div class="resource__sublinks">` +
          `<span class="resource__sublabel">Direct jump:</span>` +
          `<div class="resource__pills">` +
            meta.sublinks.map((s) => `<a class="resource__sublink" href="${esc(s.href)}"${s.spa ? ' data-spa="true"' : ''}>${esc(s.label)}</a>`).join('') +
          `</div>` +
        `</div>`
      : '';
    const specsHtml = meta.specs.map((sp) => `<span>${esc(sp)}</span>`).join('');

    return `<article class="resource resource--${cls}" data-id="${esc(m.id)}" data-num="${esc(m.num)}" data-type="${esc(m.type)}" data-routes="${esc(m.routes.join(' '))}" data-level="${esc(m.level)}" data-href="${esc(m.href)}">` +
      `<div class="resource__main">` +
        `<div class="resource__head">` +
          `<span class="resource__num">#${esc(m.num)}</span>` +
          `<span class="resource__badge resource__badge--${cls}">${arch.icon}<span>${esc(m.type)}</span></span>` +
          `<span class="resource__topic">${esc(m.topic)}</span>` +
          `<span class="resource__level">${esc(m.level)}</span>` +
        `</div>` +
        `<div class="resource__body">` +
          `<h3><a class="resource__link" href="${esc(m.href)}">${esc(m.title)}</a></h3>` +
          `<p class="resource__desc">${esc(m.blurb)}</p>` +
        `</div>` +
        sublinksHtml +
        `<div class="resource__deep-hits" hidden></div>` +
      `</div>` +
      `<div class="resource__foot">` +
        `<div class="resource__specs">${specsHtml}</div>` +
        `<a class="resource__open" href="${esc(m.href)}" tabindex="-1" aria-hidden="true">Open${ICON_ARROW}</a>` +
      `</div>` +
    `</article>`;
  }).join('\n');

  return `<div class="resources">\n${cards}\n</div>`;
}

function learningPaths(get) {
  const workflows = [
    {
      num: 'WORKFLOW 01',
      title: 'Classical ML & Data Pipeline Lifecycle',
      blurb: 'Engineer clean data platforms and features, select the strongest model family, and align evaluation metrics to business cost.',
      steps: [
        { id: 'data-engineering', role: 'Ingest, model & govern data' },
        { id: 'feature-engineering', role: 'Transform tabular, text, TS & image signals' },
        { id: 'algorithm-selector', role: 'Pick model family & TabPFN/TSFM baselines' },
        { id: 'metric-tree', role: 'Align metric to cost & imbalance' },
      ],
    },
    {
      num: 'WORKFLOW 02',
      title: 'Production GenAI & RAG Engineering',
      blurb: 'Build core LLM fluency, choose between RAG, fine-tuning, and inference optimization, and evaluate reasoning & agentic behavior.',
      steps: [
        { id: 'genai-foundations', role: 'Master prompting, grounding & routing' },
        { id: 'genai-techniques', role: 'Decide RAG vs. fine-tuning, GRPO & serving' },
        { id: 'rag-academy', role: 'Architect Hybrid, GraphRAG & Agentic retrieval' },
        { id: 'fm-metrics', role: 'Evaluate RAGAS, reasoning & tool use' },
      ],
    },
    {
      num: 'WORKFLOW 03',
      title: 'Cloud AI & Data Certification Studio',
      blurb: 'Combine structured domain guides with timed exam simulators across AWS AI Practitioner, AWS GenAI Pro, Microsoft Fabric, and Power BI.',
      steps: [
        { id: 'aws-aip-guide', role: 'Master 62 core AWS AI/ML services (AIF-C01)' },
        { id: 'aip-hub', role: 'Drill GenAI Developer Pro scenarios & 242 Qs' },
        { id: 'dp700', role: 'Prepare Microsoft Fabric Lakehouse & KQL' },
        { id: 'pl300', role: 'Practice 140 Power BI exam questions' },
      ],
    },
  ];

  const cards = workflows.map((w) => {
    const items = w.steps.map((st, idx) => {
      const m = get(st.id);
      const stepNum = String(idx + 1).padStart(2, '0');
      return `<li>` +
        `<span class="workflow-step__num">${stepNum}</span>` +
        `<div class="workflow-step__body">` +
          `<a href="${esc(m.href)}">${esc(m.title)}</a>` +
          `<span class="workflow-step__role">${esc(st.role)}</span>` +
        `</div>` +
        `<span class="workflow-step__meta">${esc(m.time)}</span>` +
      `</li>`;
    }).join('');
    return `<article class="learning-path">` +
      `<div class="learning-path__head">` +
        `<span class="learning-path__eyebrow">${esc(w.num)}</span>` +
        `<h3>${esc(w.title)}</h3>` +
        `<p>${esc(w.blurb)}</p>` +
      `</div>` +
      `<ol class="workflow-steps">${items}</ol>` +
    `</article>`;
  }).join('\n');

  return `<div class="paths-grid">\n${cards}\n</div>`;
}

/* ------------------------------------------------------------------ *
 * 3. Structured data
 * ------------------------------------------------------------------ */
function jsonLd() {
  const blocks = [
    {
      '@context': 'https://schema.org',
      '@type': 'WebSite',
      name: 'ML_LAB',
      alternateName: 'ML Learning Lab',
      url: SITE + '/',
      description: 'Interactive decision playbooks for machine learning and GenAI practitioners.',
      author: { '@type': 'Person', name: 'Tarek Atwan', url: 'https://www.tarekatwan.com' },
    },
    {
      '@context': 'https://schema.org',
      '@type': 'Person',
      name: 'Tarek Atwan',
      url: 'https://www.tarekatwan.com',
      jobTitle: 'AI and Data Science Consultant',
      description: 'Author, educator and AI/ML consultant. Four books, four-time Pluralsight Elite instructor.',
      sameAs: [
        'https://www.tarekatwan.com',
        'https://www.ensemblemethods.com',
        'https://github.com/tatwan',
        'https://www.linkedin.com/in/tarekatwan',
        'https://twitter.com/tarekatwan',
      ],
      knowsAbout: ['Machine Learning', 'Generative AI', 'Retrieval-Augmented Generation', 'Data Engineering', 'MLOps', 'Natural Language Processing'],
    },
    {
      '@context': 'https://schema.org',
      '@type': 'ItemList',
      name: 'ML_LAB playbooks',
      numberOfItems: modules.length,
      itemListElement: modules.map((m, i) => ({
        '@type': 'ListItem',
        position: i + 1,
        name: m.title,
        description: m.short,
        url: SITE + '/' + m.href,
      })),
    },
  ];
  return blocks
    .map((b) => `  <script type="application/ld+json">\n${JSON.stringify(b, null, 2)}\n  </script>`)
    .join('\n');
}

/* ------------------------------------------------------------------ *
 * 4. Sitemap
 * ------------------------------------------------------------------ */
const EXTRA_PAGES = [
  { loc: '/', priority: '1.0', changefreq: 'weekly' },
  { loc: '/blog/', priority: '0.6', changefreq: 'weekly' },
];

function fileDate(rel) {
  const clean = decodeURIComponent(rel.replace(/^\//, ''));
  const candidates = [clean, path.join(clean, 'index.html')];
  for (const c of candidates) {
    const abs = path.join(ROOT, c);
    if (fs.existsSync(abs) && fs.statSync(abs).isFile()) {
      return fs.statSync(abs).mtime.toISOString().slice(0, 10);
    }
  }
  return null;
}

function sitemap() {
  const today = new Date().toISOString().slice(0, 10);
  const urls = [];

  for (const p of EXTRA_PAGES) {
    urls.push({ loc: p.loc, lastmod: fileDate(p.loc) || today, priority: p.priority, changefreq: p.changefreq });
  }

  for (const m of modules) {
    const loc = '/' + m.href.replace(/index\.html$/, '');
    const lastmod = fileDate(m.href);
    if (!lastmod) { console.warn(`  !  missing on disk, skipped: ${m.href}`); continue; }
    urls.push({ loc, lastmod, priority: '0.8', changefreq: 'monthly' });
  }

  // Sub-pages of the AIP hub, Data Engineering course, and Applied NLP (CSE 8803).
  for (const dir of ['aip', 'DataEngineering', 'GaTech/CSE 8803']) {
    const abs = path.join(ROOT, dir);
    if (!fs.existsSync(abs)) continue;
    const encodedDir = dir.split('/').map(encodeURIComponent).join('/');
    for (const f of fs.readdirSync(abs).sort()) {
      if (!f.endsWith('.html') || f === 'index.html' || f === '404.html') continue;
      urls.push({
        loc: `/${encodedDir}/${encodeURIComponent(f)}`,
        lastmod: fileDate(`${dir}/${f}`) || today,
        priority: '0.6',
        changefreq: 'monthly',
      });
    }
  }

  const body = urls.map((u) => `  <url>
    <loc>${SITE}${u.loc}</loc>
    <lastmod>${u.lastmod}</lastmod>
    <changefreq>${u.changefreq}</changefreq>
    <priority>${u.priority}</priority>
  </url>`).join('\n');

  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${body}\n</urlset>\n`;
}

/* ------------------------------------------------------------------ *
 * Run
 * ------------------------------------------------------------------ */
const get = (id) => {
  const module = modules.find((m) => m.id === id);
  if (!module) throw new Error(`Missing curated resource: ${id}`);
  return module;
};

let html = read('index.html');
html = fill(html, 'LAB:HERO-PATH', heroBlueprint());
html = fill(html, 'LAB:ROUTES', goalRoutes());
html = fill(html, 'LAB:INDEX', resourceCatalogue());
html = fill(html, 'LAB:PATHS', learningPaths(get));
html = fill(html, 'LAB:JSONLD', jsonLd());
html = fillOptional(html, 'LAB:FEATURED', '');
html = html.replace(
  /(<!-- LAB:DATE:START -->)[\s\S]*?(<!-- LAB:DATE:END -->)/,
  `$1${new Date().toISOString().slice(0, 7)}$2`
);
write('index.html', html);
console.log(`  index.html    ${routes.length} routes, ${modules.length} modules, 3 JSON-LD blocks`);

const xml = sitemap();
write('sitemap.xml', xml);
console.log(`  sitemap.xml   ${(xml.match(/<url>/g) || []).length} URLs`);

console.log('\n  search index:');
execFileSync(process.execPath, [path.join(HERE, 'build-search-index.mjs')], { stdio: 'inherit' });
