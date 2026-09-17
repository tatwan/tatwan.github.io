#!/usr/bin/env node
/**
 * build-site.mjs
 *
 * One command to regenerate everything derived from assets/modules.json:
 *
 *   node tools/build-site.mjs
 *
 *   1. the route tiles and the module index inside index.html
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

/* ------------------------------------------------------------------ *
 * 1. Route tiles
 * ------------------------------------------------------------------ */
const ROUTE_ICONS = {
  scope: '<path d="M12 3l8 4v5c0 5-3.4 8-8 9-4.6-1-8-4-8-9V7z"/><path d="M12 8v4m0 3h.01"/>',
  model: '<path d="M12 3v6M12 9l-6 6M12 9l6 6"/><circle cx="12" cy="3" r="1.6"/><circle cx="6" cy="17" r="2.2"/><circle cx="18" cy="17" r="2.2"/>',
  measure: '<path d="M3 20h18M6 20V11M11 20V6M16 20v-6M21 20V9"/>',
  llm: '<path d="M4 7h10M4 12h16M4 17h7"/><circle cx="18" cy="7" r="2.4"/><circle cx="15" cy="17" r="2.4"/>',
  exam: '<path d="M12 3l8 4v5c0 5-3.4 8-8 9-4.6-1-8-4-8-9V7z"/><path d="M9 12l2 2 4-4"/>',
};

function routeTiles() {
  const tiles = routes.map((r) => {
    const members = modules.filter((m) => (m.routes || []).includes(r.id));
    const first = members[0];
    const href = first ? first.href : '#index';
    const icon = ROUTE_ICONS[r.id] || ROUTE_ICONS.model;
    return `          <a class="route" href="${esc(href)}">
            <span class="route__top">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icon}</svg>
              <span class="route__count">${String(members.length).padStart(2, '0')}</span>
            </span>
            <span class="route__title">${esc(r.label)}</span>
            <span class="route__blurb">${esc(r.blurb)}</span>
          </a>`;
  }).join('\n');
  return `        <div class="routes">\n${tiles}\n        </div>`;
}

/* ------------------------------------------------------------------ *
 * 2. Module index
 * ------------------------------------------------------------------ */
function indexTable() {
  const head = `          <div class="row row--head">
            <span>ID</span><span>Playbook</span><span class="row__meta">Type</span><span class="row__meta">Topic</span><span class="row__time">Time</span>
          </div>`;

  const rows = modules.map((m) => `          <div class="row" data-type="${esc(m.type)}" data-level="${esc(m.level)}">
            <span class="row__num">${esc(m.num)}</span>
            <span>
              <a class="row__title" href="${esc(m.href)}">${esc(m.title)}</a>
              <span class="row__blurb">${esc(m.short)}</span>
            </span>
            <span class="row__meta">${esc(m.type)}</span>
            <span class="row__meta">${esc(m.topic)}</span>
            <span class="row__time">${esc(m.time)}</span>
          </div>`).join('\n');

  return `        <div class="table">\n${head}\n${rows}\n        </div>`;
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
  { loc: '/privacy.html', priority: '0.2', changefreq: 'yearly' },
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

  // Sub-pages of the AIP hub and the Data Engineering course.
  for (const dir of ['aip', 'DataEngineering']) {
    const abs = path.join(ROOT, dir);
    if (!fs.existsSync(abs)) continue;
    for (const f of fs.readdirSync(abs).sort()) {
      if (!f.endsWith('.html') || f === 'index.html' || f === '404.html') continue;
      urls.push({
        loc: `/${dir}/${f}`,
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
let html = read('index.html');
html = fill(html, 'LAB:ROUTES', routeTiles());
html = fill(html, 'LAB:INDEX', indexTable());
html = fill(html, 'LAB:JSONLD', jsonLd());
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
