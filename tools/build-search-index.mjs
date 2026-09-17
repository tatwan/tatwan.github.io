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

const TITLE_KEYS = ['question', 'title', 'name', 'term', 'heading', 'label', 'q'];
const BODY_KEYS = [
  'info', 'goal', 'desc', 'description', 'definition', 'explanation', 'exp',
  'summary', 'text', 'why', 'when', 'detail', 'details', 'answer', 'content',
  'useCase', 'tradeoff', 'note', 'rationale', 'takeaway',
];

/** Raw literal names are meaningless to a reader; label them for the breadcrumb. */
const VAR_LABELS = {
  QUESTIONS: 'Practice questions',
  quizData: 'Practice questions',
  quickFireQuestions: 'Quick-fire questions',
  DOMAINS: 'Exam domains',
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

const clean = (s) => String(s).replace(/\s+/g, ' ').trim();
const truncate = (s, n = 190) => (s.length > n ? s.slice(0, n - 1).trimEnd() + '…' : s);

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
  for (const key of ['options', 'opts']) {
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
  if (Array.isArray(obj.algorithms)) {
    const names = obj.algorithms.map((a) => (a && a.name) || null).filter(Boolean);
    if (names.length) parts.push(clean(names.join(' · ')));
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
    if (!node || !Array.isArray(node.options)) continue;
    for (const opt of node.options) {
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
 * Main
 * ------------------------------------------------------------------ */

/** The 16 playbooks themselves, so a page-level query always lands. */
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
