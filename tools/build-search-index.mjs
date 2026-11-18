#!/usr/bin/env node
/**
 * build-search-index.mjs
 *
 * Extracts the decision-tree / question / glossary data embedded in each
 * interactive page and emits assets/search-index.json, which powers the
 * command palette (Cmd/Ctrl+K) across the site.
 *
 * The tools are self-contained React apps whose content lives in plain object
 * literals inside <script type="text/babel">. Rather than parse JSX, we locate
 * each named literal by brace-matching and evaluate it in a sandbox where
 * unknown identifiers resolve to undefined.
 *
 * Usage:  node tools/build-search-index.mjs
 */

import fs from 'node:fs';
import path from 'node:path';
import url from 'node:url';
import vm from 'node:vm';

const ROOT = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), '..');

/* ------------------------------------------------------------------ *
 * Which pages to index, and which top-level literals hold their content.
 * ------------------------------------------------------------------ */
const PAGES = [
  { id: 'genai-foundations', file: 'genai_foundations_navigator.html', title: 'GenAI Foundations Navigator', kind: 'Foundations', vars: ['tree'] },
  { id: 'algorithm-selector', file: 'ml_algorithm_selector.html', title: 'Algorithm Selector', kind: 'Modeling', vars: ['tree'] },
  { id: 'feature-engineering', file: 'feature_engineering_playbook.html', title: 'Feature Engineering Playbook', kind: 'Features', vars: ['tree'] },
  { id: 'metric-tree', file: 'model_evaluation_interactive.html', title: 'Metric Decision Tree', kind: 'Evaluation', vars: ['tree', 'quickReference'] },
  { id: 'genai-techniques', file: 'genai_techniques_selector.html', title: 'GenAI Technique Selector', kind: 'GenAI', vars: ['tree'] },
  { id: 'fm-metrics', file: 'fm_evaluation_metrics.html', title: 'FM Evaluation Metrics', kind: 'Evaluation', vars: ['tree'] },
  { id: 'opportunity-scorer', file: 'ai_opportunity_scorer.html', title: 'AI Opportunity Scorer', kind: 'Product', vars: ['DECISIONS', 'DIMENSIONS', 'DIMENSION_SCORING', 'EXAMPLE_SCENARIOS', 'GLOSSARY'] },
  { id: 'solution-navigator', file: 'ai_solution_navigator.html', title: 'AI Solution Navigator', kind: 'Foundations', vars: ['SOLUTIONS', 'GLOSSARY', 'EXAMPLE_SCENARIOS', 'HYBRID_COMBOS', 'landscape'] },
  { id: 'aws-aip-guide', file: 'aws_ai_practitioner.html', title: 'AWS AI Practitioner Guide', kind: 'Certification', vars: ['servicesData', 'quickFireQuestions'] },
  { id: 'aws-aip-exam', file: 'aws_ai_practitioner_exam.html', title: 'AIF-C01 Practice Exam', kind: 'Certification', vars: ['QUESTIONS', 'DOMAINS'] },
  { id: 'dp700', file: 'DP-700-InteractiveStudy.html', title: 'DP-700 Study Guide', kind: 'Certification', vars: ['quizData'] },
  { id: 'pl300', file: 'pl300_exam_prep.html', title: 'PL-300 Exam Prep', kind: 'Certification', vars: ['QUESTIONS', 'SKILLS', 'DOMAINS'] },
  { id: 'aip-practice', file: 'aip/practice-exam.html', title: 'AIP-C01 Practice Exam', kind: 'Certification', vars: ['QUESTIONS'] },
  { id: 'aip-quick-drill', file: 'aip/quick-drill.html', title: 'AIP-C01 Quick Drill', kind: 'Certification', vars: ['QUESTIONS_DB'] },
  { id: 'aip-final', file: 'aip/final-practice.html', title: 'AIP-C01 Final Practice', kind: 'Certification', vars: ['QUESTIONS'] },
  { id: 'aip-flashcards', file: 'aip/flashcards.html', title: 'AIP-C01 Flashcards & Drills', kind: 'Certification', vars: ['FLASHCARDS', 'QUESTIONS', 'DRILLS'] },
];

/* ------------------------------------------------------------------ *
 * Literal extraction
 * ------------------------------------------------------------------ */

/** Find `const <name> = {` / `[` and return the balanced literal source. */
function extractLiteral(src, name) {
  const re = new RegExp(`(?:const|let|var)\\s+${name}\\s*=\\s*`, 'g');
  const m = re.exec(src);
  if (!m) return null;

  let i = m.index + m[0].length;
  const open = src[i];
  if (open !== '{' && open !== '[') return null;
  const close = open === '{' ? '}' : ']';

  let depth = 0;
  let quote = null;      // ' " ` when inside a string
  let tmplDepth = 0;     // ${ } nesting inside a template literal
  let comment = null;    // 'line' | 'block'

  for (let j = i; j < src.length; j++) {
    const c = src[j];
    const next = src[j + 1];
    const prev = src[j - 1];

    if (comment === 'line') { if (c === '\n') comment = null; continue; }
    if (comment === 'block') { if (c === '*' && next === '/') { comment = null; j++; } continue; }

    if (quote) {
      if (c === '\\') { j++; continue; }
      if (quote === '`' && c === '$' && next === '{') { tmplDepth++; j++; continue; }
      if (quote === '`' && c === '}' && tmplDepth > 0) { tmplDepth--; continue; }
      if (c === quote && tmplDepth === 0) quote = null;
      continue;
    }

    if (c === '/' && next === '/') { comment = 'line'; j++; continue; }
    if (c === '/' && next === '*') { comment = 'block'; j++; continue; }
    if (c === '"' || c === "'" || c === '`') { quote = c; continue; }

    if (c === open) depth++;
    else if (c === close) {
      depth--;
      if (depth === 0) return src.slice(i, j + 1);
    }
    void prev;
  }
  return null;
}

/** Evaluate a literal with unknown identifiers resolving to undefined. */
function evalLiteral(literalSrc) {
  const sandbox = new Proxy({}, {
    has: () => true,
    get: (_t, k) => (k === Symbol.toStringTag || k === Symbol.unscopables ? undefined : undefined),
  });
  try {
    return vm.runInNewContext(`(${literalSrc})`, vm.createContext(sandbox), { timeout: 5000 });
  } catch {
    return null;
  }
}

/* ------------------------------------------------------------------ *
 * Turning data into search entries
 * ------------------------------------------------------------------ */

const TITLE_KEYS = ['question', 'title', 'name', 'term', 'heading', 'label', 'q', 'scenario', 'prompt', 'front'];
const BODY_KEYS = [
  'info', 'goal', 'desc', 'description', 'definition', 'explanation', 'exp',
  'summary', 'text', 'why', 'when', 'detail', 'details', 'answer', 'content',
  'useCase', 'tradeoff', 'note', 'rationale', 'takeaway', 'back', 'whyWrong',
];

/** Raw literal names are meaningless to a reader; label them for the breadcrumb. */
const VAR_LABELS = {
  QUESTIONS: 'Practice questions',
  QUESTIONS_DB: 'Quick drill',
  FLASHCARDS: 'Flashcards',
  DRILLS: 'Drills',
  quizData: 'Practice questions',
  quickFireQuestions: 'Quick-fire questions',
  DOMAINS: 'Exam domains',
  SKILLS: 'Study guide',
  servicesData: 'AWS services',
  GLOSSARY: 'Glossary',
  DECISIONS: 'Verdicts',
  DIMENSIONS: 'Scoring dimensions',
  DIMENSION_SCORING: 'Scoring guide',
  EXAMPLE_SCENARIOS: 'Worked examples',
  SOLUTIONS: 'Solution classes',
  HYBRID_COMBOS: 'Hybrid patterns',
  landscape: 'The AI landscape',
  quickReference: 'Quick reference',
};

const ENTITIES = {
  '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#39;': "'", '&apos;': "'",
  '&ndash;': '–', '&mdash;': '—', '&rarr;': '→', '&larr;': '←', '&harr;': '↔',
  '&rsquo;': '’', '&lsquo;': '‘', '&rdquo;': '”', '&ldquo;': '“', '&middot;': '·',
  '&times;': '×', '&nbsp;': ' ', '&sup1;': '¹', '&sup2;': '²', '&radic;': '√',
};
const decodeEntities = (s) =>
  String(s).replace(/&[a-zA-Z0-9#]+;/g, (m) => {
    if (ENTITIES[m]) return ENTITIES[m];
    if (m.startsWith('&#x')) return String.fromCodePoint(parseInt(m.slice(3, -1), 16));
    if (m.startsWith('&#')) return String.fromCodePoint(parseInt(m.slice(2, -1), 10));
    return m;
  });
const clean = (s) => decodeEntities(String(s)).replace(/\s+/g, ' ').trim();
const truncate = (s, n = 190) => (s.length > n ? s.slice(0, n - 1).trimEnd() + '…' : s);
const stripHtml = (s) => clean(String(s).replace(/<[^>]+>/g, ' '));

function titleOf(obj) {
  for (const k of TITLE_KEYS) {
    if (typeof obj[k] === 'string' && obj[k].trim()) return clean(obj[k]);
  }
  return null;
}

function bodyOf(obj) {
  const parts = [];
  for (const k of BODY_KEYS) {
    const v = obj[k];
    if (typeof v === 'string' && v.trim()) parts.push(clean(v));
    else if (Array.isArray(v)) {
      const strs = v.filter((x) => typeof x === 'string');
      if (strs.length) parts.push(clean(strs.join('. ')));
    }
  }
  for (const key of ['options', 'opts', 'continueOptions']) {
    if (!Array.isArray(obj[key])) continue;
    const labels = obj[key]
      .map((o) => {
        if (typeof o === 'string') return o;
        if (!o || typeof o !== 'object') return null;
        // Option descriptions carry a lot of searchable vocabulary
        // ("Use PEFT (LoRA, QLoRA, etc.)"), so keep them.
        return [o.label || o.text, o.desc || o.description].filter(Boolean).join(' ');
      })
      .filter(Boolean);
    if (labels.length) parts.push(clean(labels.join(' · ')));
  }
  for (const arrKey of ['algorithms', 'metrics', 'techniques', 'sections']) {
    if (!Array.isArray(obj[arrKey])) continue;
    const items = obj[arrKey]
      .map((a) => {
        if (!a || typeof a !== 'object') return null;
        return [a.name || a.title, a.description || a.body || a.when || a.formula].filter(Boolean).join(': ');
      })
      .filter(Boolean);
    if (items.length) parts.push(clean(items.join(' · ')));
  }
  return truncate(parts.join(' — '));
}

/** Breadcrumb labels for every node of a `tree`-shaped object, via BFS from start. */
function treePaths(tree) {
  const startKey = tree.start ? 'start' : Object.keys(tree)[0];
  const paths = { [startKey]: [] };
  const queue = [startKey];
  while (queue.length) {
    const key = queue.shift();
    const node = tree[key];
    if (!node) continue;
    const opts = [
      ...(Array.isArray(node.options) ? node.options : []),
      ...(Array.isArray(node.continueOptions) ? node.continueOptions : []),
    ];
    for (const opt of opts) {
      if (!opt || !opt.next || paths[opt.next]) continue;
      paths[opt.next] = [...paths[key], clean(opt.label || opt.next)];
      queue.push(opt.next);
    }
  }
  return paths;
}

function entriesFromTree(tree, page) {
  const paths = treePaths(tree);
  const out = [];
  for (const [key, node] of Object.entries(tree)) {
    if (!node || typeof node !== 'object') continue;
    const t = titleOf(node);
    if (!t) continue;
    const trail = paths[key] || [];
    out.push({
      t,
      d: bodyOf(node),
      p: [page.title, ...trail].join(' › '),
      u: `${page.href}#${key}`,
      k: page.kind,
      s: page.id,
    });

    for (const arrKey of ['algorithms', 'metrics', 'techniques', 'sections']) {
      if (!Array.isArray(node[arrKey])) continue;
      for (const item of node[arrKey]) {
        if (!item || typeof item !== 'object') continue;
        const name = clean(item.name || item.title || '');
        if (!name) continue;
        const desc = truncate(
          clean(
            [
              item.priority,
              item.description || item.body,
              item.when ? 'When: ' + item.when : '',
              item.bestFor ? 'Best for: ' + item.bestFor : '',
              item.formula ? 'Formula: ' + item.formula : '',
              item.sklearn ? 'Library: ' + item.sklearn : '',
              item.tools ? 'Tools: ' + item.tools : '',
            ]
              .filter(Boolean)
              .join(' — ')
          )
        );
        out.push({
          t: name,
          d: desc,
          p: [page.title, ...trail, t].join(' › '),
          u: `${page.href}#${key}`,
          k: page.kind,
          s: page.id,
        });
      }
    }
  }
  return out;
}

/** Generic walk for non-tree literals (glossaries, question banks, tables). */
function entriesFromGeneric(value, page, varName) {
  const out = [];
  const seen = new Set();

  const visit = (node, trail, depth) => {
    if (!node || typeof node !== 'object' || depth > 4) return;
    if (Array.isArray(node)) {
      node.forEach((child) => visit(child, trail, depth + 1));
      return;
    }
    const t = titleOf(node);
    const d = bodyOf(node);
    if (t && d) {
      const sig = t + '|' + d.slice(0, 40);
      if (!seen.has(sig)) {
        seen.add(sig);
        out.push({
          t,
          d,
          p: [page.title, ...trail].join(' › '),
          u: page.href,
          k: page.kind,
          s: page.id,
        });
      }
    }
    for (const [k, v] of Object.entries(node)) {
      if (v && typeof v === 'object') {
        const nextTrail = t ? trail : [...trail, clean(k)];
        visit(v, nextTrail.slice(0, 3), depth + 1);
      }
    }
  };

  visit(value, [VAR_LABELS[varName] || clean(varName)], 0);
  return out;
}

/* ------------------------------------------------------------------ *
 * HTML courses, study guides & SPA route indexing
 * ------------------------------------------------------------------ */

function extractNextSnippet(html, afterIndex) {
  const slice = html.slice(afterIndex, afterIndex + 1200);
  const descMatch = slice.match(/<(?:p|div)\s+class="desc"[^>]*>([\s\S]*?)<\/(?:p|div)>/i);
  if (descMatch) {
    const txt = stripHtml(descMatch[1]);
    if (txt.length >= 12) return truncate(txt);
  }
  const pMatch = slice.match(/<p[^>]*>([\s\S]*?)<\/p>/i);
  if (pMatch) {
    const txt = stripHtml(pMatch[1]);
    if (txt.length >= 12) return truncate(txt);
  }
  const tdMatches = [...slice.matchAll(/<td[^>]*>([\s\S]*?)<\/td>/gi)]
    .map((m) => stripHtml(m[1]))
    .filter((s) => s.length >= 3);
  if (tdMatches.length) return truncate(tdMatches.slice(0, 4).join(' · '));
  return '';
}

function stripAboutBand(html) {
  const at = html.indexOf('<!-- LAB:ABOUT:START -->');
  return at > -1 ? html.slice(0, at) : html;
}

function extractHeadings(html, re, pageTitle, url, kind, id, trailPrefix) {
  const cleanHtml = stripAboutBand(html);
  const out = [];
  for (const m of cleanHtml.matchAll(re)) {
    const t = stripHtml(m[1]);
    if (!t || t.length < 4 || /^\d+(\.\d+)?$/.test(t)) continue;
    const after = m.index + m[0].length;
    const d = extractNextSnippet(cleanHtml, after) || pageTitle;
    out.push({
      t,
      d,
      p: trailPrefix ? `${trailPrefix} › ${pageTitle}` : pageTitle,
      u: url,
      k: kind,
      s: id,
    });
  }
  return out;
}

function entriesFromHtmlCourses(report) {
  const out = [];

  // 1. Data Engineering glossary + 10 modules
  let deCount = 0;
  const deIdxPath = path.join(ROOT, 'DataEngineering/index.html');
  if (fs.existsSync(deIdxPath)) {
    const src = fs.readFileSync(deIdxPath, 'utf8');
    const lit = extractLiteral(src, 'terms');
    const terms = lit ? evalLiteral(lit) : null;
    if (Array.isArray(terms)) {
      for (const row of terms) {
        if (!Array.isArray(row) || row.length < 3) continue;
        out.push({
          t: clean(row[0]),
          d: truncate(clean(`${row[1]} (${row[3] || 'Data Engineering'}) — ${row[2]}`)),
          p: 'Data Engineering Foundations › Glossary',
          u: '/DataEngineering/index.html#glossary',
          k: 'Course',
          s: 'data-engineering',
        });
        deCount++;
      }
    }
  }
  for (let i = 1; i <= 10; i++) {
    const rel = `DataEngineering/module${i}.html`;
    const abs = path.join(ROOT, rel);
    if (!fs.existsSync(abs)) continue;
    const src = fs.readFileSync(abs, 'utf8');
    const url = '/' + rel;
    const rawTitle = (src.match(/<title>([^<]+)<\/title>/i) || [])[1] || `Module ${i}`;
    const modTitle = stripHtml(rawTitle.split('|')[0]);
    const metaDesc = stripHtml((src.match(/<meta\s+name="description"\s+content="([^"]+)"/i) || [])[1] || '');
    out.push({
      t: modTitle,
      d: truncate(metaDesc || extractNextSnippet(src, src.indexOf('<header'))),
      p: 'Data Engineering Foundations',
      u: url,
      k: 'Course',
      s: 'data-engineering',
    });
    deCount++;
    const h2s = extractHeadings(src, /<h2[^>]*>([\s\S]*?)<\/h2>/gi, modTitle, url, 'Course', 'data-engineering', 'Data Engineering Foundations');
    const h3s = extractHeadings(src, /<div class="concept-header"[^>]*>\s*<h3[^>]*>([\s\S]*?)<\/h3>/gi, modTitle, url, 'Course', 'data-engineering', 'Data Engineering Foundations');
    const interactives = extractHeadings(src, /<div class="interactive-box"[^>]*>\s*<h4[^>]*>([\s\S]*?)<\/h4>/gi, modTitle, url, 'Course', 'data-engineering', 'Data Engineering Foundations');
    out.push(...h2s, ...h3s, ...interactives);
    deCount += h2s.length + h3s.length + interactives.length;
  }
  report.push(`  ${String(deCount).padStart(4)}  DataEngineering/ (glossary + 10 modules)`);

  // 2. GaTech CSE 8803 Applied NLP — 15 weeks
  let cseCount = 0;
  for (let i = 1; i <= 15; i++) {
    const rel = `GaTech/CSE 8803/week${i}.html`;
    const abs = path.join(ROOT, rel);
    if (!fs.existsSync(abs)) continue;
    const src = fs.readFileSync(abs, 'utf8');
    const url = `/GaTech/CSE%208803/week${i}.html`;
    const rawTitle = (src.match(/<title>([^<]+)<\/title>/i) || [])[1] || `Week ${i}`;
    const weekTitle = stripHtml(rawTitle.split('|')[0]);
    const headerIdx = src.indexOf('class="page-header"');
    const leadDesc = headerIdx > -1 ? extractNextSnippet(src, headerIdx) : '';
    out.push({
      t: weekTitle,
      d: truncate(leadDesc || `Week ${i} study notes and interactive demos for Applied NLP (CSE 8803).`),
      p: 'Applied NLP — CSE 8803',
      u: url,
      k: 'Course',
      s: 'cse8803',
    });
    cseCount++;
    const h2s = extractHeadings(src, /<h2[^>]*>([\s\S]*?)<\/h2>/gi, weekTitle, url, 'Course', 'cse8803', 'Applied NLP — CSE 8803');
    const h3s = extractHeadings(src, /<div class="concept-header"[^>]*>\s*<h3[^>]*>([\s\S]*?)<\/h3>/gi, weekTitle, url, 'Course', 'cse8803', 'Applied NLP — CSE 8803');
    const bts = extractHeadings(src, /<div class="bts-card"[^>]*>\s*<h3[^>]*>([\s\S]*?)<\/h3>/gi, `${weekTitle} › Beyond the Slides`, url, 'Course', 'cse8803', 'Applied NLP — CSE 8803');
    const interactives = extractHeadings(src, /<div class="interactive-box"[^>]*>\s*<h4[^>]*>([\s\S]*?)<\/h4>/gi, weekTitle, url, 'Course', 'cse8803', 'Applied NLP — CSE 8803');
    out.push(...h2s, ...h3s, ...bts, ...interactives);
    cseCount += h2s.length + h3s.length + bts.length + interactives.length;
  }
  report.push(`  ${String(cseCount).padStart(4)}  GaTech/CSE 8803/ (15 weeks)`);

  // 3. AIP-C01 HTML study guides & domain deep dives
  const aipGuides = [
    'study-guide.html',
    'aip-c01-studyguide.html',
    'study-d1.html',
    'study-d2.html',
    'study-d3.html',
    'study-d45.html',
    'advanced-impl.html',
    'troubleshooting-eval.html',
    'gap-cheatsheet.html',
  ];
  let aipGuideCount = 0;
  for (const file of aipGuides) {
    const rel = `aip/${file}`;
    const abs = path.join(ROOT, rel);
    if (!fs.existsSync(abs)) continue;
    const src = fs.readFileSync(abs, 'utf8');
    const url = '/' + rel;
    const h1 = stripHtml((src.match(/<h1[^>]*>([\s\S]*?)<\/h1>/i) || [])[1] || (src.match(/<title>([^<]+)<\/title>/i) || [])[1] || file);
    out.push({
      t: h1,
      d: truncate(extractNextSnippet(src, Math.max(0, src.indexOf('<h1'))) || `AIP-C01 study guide: ${h1}`),
      p: 'AIP-C01 Prep Hub',
      u: url,
      k: 'Certification',
      s: 'aip-hub',
    });
    aipGuideCount++;
    const h2s = extractHeadings(src, /<h2[^>]*>([\s\S]*?)<\/h2>/gi, h1, url, 'Certification', 'aip-hub', 'AIP-C01 Prep Hub');
    const h3s = extractHeadings(src, /<h3[^>]*>([\s\S]*?)<\/h3>/gi, h1, url, 'Certification', 'aip-hub', 'AIP-C01 Prep Hub');
    out.push(...h2s, ...h3s);
    aipGuideCount += h2s.length + h3s.length;
  }
  report.push(`  ${String(aipGuideCount).padStart(4)}  aip/ (9 study guides)`);

  // 4. Visual References (3 AWS AIF-C01 diagrams)
  const visPath = path.join(ROOT, 'aws_visual_references.html');
  let visCount = 0;
  if (fs.existsSync(visPath)) {
    const src = fs.readFileSync(visPath, 'utf8');
    const h2s = extractHeadings(src, /<h2[^>]*>([\s\S]*?)<\/h2>/gi, 'Diagrams', '/aws_visual_references.html', 'Certification', 'aws-visual', 'Visual References');
    out.push(...h2s);
    visCount = h2s.length;
  }
  report.push(`  ${String(visCount).padStart(4)}  aws_visual_references.html`);

  // 5. RAG Academy (19 SPA routes)
  let ragCount = 0;
  const ragBundlePath = path.join(ROOT, 'rag-academy/assets/index-BZdBW4KA.js');
  if (fs.existsSync(ragBundlePath)) {
    const js = fs.readFileSync(ragBundlePath, 'utf8');
    for (const m of js.matchAll(/\{path:"(\/[a-z-]+)",title:"([^"]+)",description:"([^"]+)"\}/g)) {
      out.push({
        t: clean(m[2]),
        d: clean(m[3]),
        p: 'RAG Academy › Interactive Module',
        u: `/rag-academy${m[1]}`,
        k: 'Course',
        s: 'rag-academy',
      });
      ragCount++;
    }
  }
  report.push(`  ${String(ragCount).padStart(4)}  rag-academy/ (SPA routes)`);

  return out;
}

/* ------------------------------------------------------------------ *
 * Main
 * ------------------------------------------------------------------ */

/** The playbooks themselves, so a page-level query always lands. */
function moduleEntries() {
  const file = path.join(ROOT, 'assets/modules.json');
  if (!fs.existsSync(file)) return [];
  const { modules } = JSON.parse(fs.readFileSync(file, 'utf8'));
  return modules.map((m) => ({
    t: m.title,
    d: truncate(clean(`${m.short} ${m.blurb}`), 200),
    p: `${m.type} · ${m.topic} · ${m.time}`,
    u: '/' + m.href,
    k: 'Playbook',
    s: m.id,
  }));
}

function main() {
  const entries = moduleEntries();
  const report = [`  ${String(entries.length).padStart(4)}  (playbooks, from modules.json)`];

  for (const page of PAGES) {
    const abs = path.join(ROOT, page.file);
    if (!fs.existsSync(abs)) { report.push(`  MISSING  ${page.file}`); continue; }
    const src = fs.readFileSync(abs, 'utf8');
    page.href = '/' + page.file;

    let pageCount = 0;
    for (const varName of page.vars) {
      const literal = extractLiteral(src, varName);
      if (!literal) { report.push(`  no literal  ${page.file} :: ${varName}`); continue; }
      const value = evalLiteral(literal);
      if (!value) { report.push(`  eval failed ${page.file} :: ${varName}`); continue; }

      const got = varName === 'tree' && value.start
        ? entriesFromTree(value, page)
        : entriesFromGeneric(value, page, varName);

      entries.push(...got);
      pageCount += got.length;
    }
    report.push(`  ${String(pageCount).padStart(4)}  ${page.file}`);
  }

  entries.push(...entriesFromHtmlCourses(report));

  // De-duplicate on title + page
  const uniq = [];
  const seen = new Set();
  for (const e of entries) {
    const sig = e.s + '|' + e.t;
    if (seen.has(sig)) continue;
    seen.add(sig);
    uniq.push(e);
  }

  const outDir = path.join(ROOT, 'assets');
  fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, 'search-index.json');
  fs.writeFileSync(outFile, JSON.stringify({
    generated: new Date().toISOString().slice(0, 10),
    count: uniq.length,
    entries: uniq,
  }));

  const kb = (fs.statSync(outFile).size / 1024).toFixed(1);
  console.log(report.join('\n'));
  console.log(`\n  ${uniq.length} entries -> assets/search-index.json (${kb} KB)`);
}

main();
