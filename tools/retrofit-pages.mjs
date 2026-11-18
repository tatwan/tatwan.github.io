#!/usr/bin/env node
/**
 * retrofit-pages.mjs
 *
 * Gives every page in the ML_LAB site (all 17 top-level playbooks, all 38
 * course/exam sub-pages, and 404.html) the shared shell, archetype styles,
 * hierarchical breadcrumbs, and SEO metadata:
 *
 *   - the ML_LAB header, hierarchical breadcrumb, footer and Cmd+K palette (lab.css + lab.js)
 *   - unified Google Fonts (Bricolage Grotesque, IBM Plex Sans, IBM Plex Mono, JetBrains Mono)
 *   - archetype-specific CSS/JS (archetype-decision, archetype-aip, archetype-course)
 *   - description, canonical, Open Graph, Twitter and JSON-LD (top-level playbooks)
 *   - an "About this playbook" band of real prose (top-level playbooks)
 *   - strips redundant inline <- ML_LAB backlinks and redundant page-level footers
 *
 * Every injected block is fenced by LAB:* markers, so re-running this
 * updates in place rather than stacking duplicates.
 *
 * Usage:  node tools/retrofit-pages.mjs [--dry]
 */

import fs from 'node:fs';
import path from 'node:path';
import url from 'node:url';

const HERE = path.dirname(url.fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const SITE = 'https://tatwan.github.io';
const DRY = process.argv.includes('--dry');

const catalogue = JSON.parse(fs.readFileSync(path.join(ROOT, 'assets/modules.json'), 'utf8'));
const { modules, routes } = catalogue;

const esc = (s) => String(s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const FONTS_LINK =
  '  <link rel="preconnect" href="https://fonts.googleapis.com">\n' +
  '  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n' +
  '  <link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700;12..96,800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">';

const DECISION_IDS = new Set([
  'genai-foundations',
  'algorithm-selector',
  'feature-engineering',
  'metric-tree',
  'opportunity-scorer',
  'solution-navigator',
  'genai-techniques',
  'fm-metrics',
]);

const AIF_IDS = new Set([
  'aws-aip-guide',
  'aws-aip-exam',
  'aws-visual',
]);

/** Extra context per playbook: what it decides, and who should use it. */
const CONTEXT = {
  'genai-foundations': {
    covers: 'grounding and hallucination, the parts of a RAG pipeline, prompt structure, temperature and top-p, and the threshold where fine-tuning starts to pay for itself',
    who: 'anyone who keeps nodding along in GenAI conversations and would rather actually follow them',
  },
  'algorithm-selector': {
    covers: 'supervised and unsupervised learning, classification, regression, ranking, forecasting, anomaly detection and the semi-supervised middle ground',
    who: 'data scientists choosing a first model, and engineers sanity-checking one someone else chose',
  },
  'feature-engineering': {
    covers: 'categorical encoding, scaling, outlier treatment, missing values, feature creation and selection, branched separately for tabular, text, time-series and image data',
    who: 'practitioners whose model is underperforming for reasons that are probably not the model',
  },
  'metric-tree': {
    covers: 'accuracy, precision, recall, F1, ROC-AUC, PR-AUC, MAE, RMSE, MAPE and the ranking metrics, with explicit handling of class imbalance',
    who: 'anyone about to report a number to a stakeholder who will act on it',
  },
  'data-engineering': {
    covers: 'source systems, ingestion patterns, storage layers, modelling, orchestration, quality and the governance that makes analytics trustworthy',
    who: 'analysts moving into engineering, and engineers who inherited a pipeline nobody documented',
  },
  'aws-aip-guide': {
    covers: 'all five AIF-C01 domains, the AWS AI and ML service catalogue, and the terminology the exam expects you to distinguish precisely',
    who: 'candidates preparing for the AWS Certified AI Practitioner exam',
  },
  'aws-aip-exam': {
    covers: '189 questions distributed across the five exam domains in the same proportions as the real paper, each with a written explanation',
    who: 'candidates who have studied and now need to find out what they actually retained',
  },
  'aws-visual': {
    covers: 'confusion matrices, retrieval-augmented generation pipelines, and the full foundation-model lifecycle from pre-training to deployment',
    who: 'visual learners, and anyone who needs a diagram to put in a slide',
  },
  'aip-hub': {
    covers: 'deep and visual study guides, 65 flashcards, a gap cheatsheet, 4 domain deep-dives, implementation and troubleshooting guides, and 242 practice questions across three exam simulators for the AWS Certified Generative AI Developer – Professional (AIP-C01) exam',
    who: 'candidates who want a complete preparation track rather than a single resource',
  },
  'dp700': {
    covers: 'Fabric architecture, ingestion, Delta Lake, KQL, PySpark, security boundaries, CI/CD and the specific limits that appear as exam questions',
    who: 'candidates for the Microsoft Fabric Data Engineering Associate certification',
  },
  'pl300': {
    covers: '140 original questions across the four PL-300 domains in single-answer, multiple-answer, drop-down, sequence, yes/no series and case study formats, with study notes on Power Query, data modeling, DAX, report design, workspaces and row-level security',
    who: 'candidates for the Microsoft Power BI Data Analyst Associate certification who want to practice, find their weak areas and do a last review',
  },
  'opportunity-scorer': {
    covers: 'value, feasibility, data readiness, risk, differentiation and time-to-value, scored to a build, buy, experiment or kill verdict',
    who: 'product managers and leads deciding where an AI budget should go',
  },
  'solution-navigator': {
    covers: 'rule-based systems, classical machine learning, deep learning, generative models, agents and operations research, plus the hybrids between them',
    who: 'architects and PMs framing a problem before a stack gets chosen for them',
  },
  'genai-techniques': {
    covers: 'prompt engineering, retrieval-augmented generation, full fine-tuning, LoRA and QLoRA, quantization, distillation and model routing',
    who: 'engineers whose LLM is too slow, too expensive, or too wrong, and who need to know which of those to fix first',
  },
  'fm-metrics': {
    covers: 'perplexity, ROUGE, BLEU, BERTScore, RAGAS, LLM-as-judge and human evaluation, including what each one quietly fails to capture',
    who: 'teams who need to prove an LLM feature improved, not just that it shipped',
  },
  'rag-academy': {
    covers: '12 RAG architectures (Classic, Hybrid, HyDE, Self-RAG, Corrective, Adaptive, GraphRAG, KAG, Agentic, Modular, Multimodal, Speculative), chunking strategies, query understanding, vector indexing, mathematical foundations and an interactive playground',
    who: 'engineers building or debugging a retrieval-augmented system',
  },
  'cse8803': {
    covers: 'all 15 weeks of graduate NLP: text preprocessing, classical classifiers, SVD/GloVe/Word2Vec embeddings, CNNs/RNNs/LSTMs, Transformers (BERT, GPT, T5, RoPE, GQA, MoE), sequence labeling (HMM/CRF), post-training alignment (SFT, RLHF, DPO, KTO), topic modeling, RAG, Agentic AI and multi-agent systems',
    who: 'students on the Georgia Tech OMSA programme, and anyone rebuilding NLP fundamentals from the ground up',
  },
};

/* ------------------------------------------------------------------ *
 * Block builders
 * ------------------------------------------------------------------ */

function headBlock(m) {
  const canonical = SITE + '/' + m.href.replace(/index\.html$/, '');
  const ld = {
    '@context': 'https://schema.org',
    '@type': 'LearningResource',
    name: m.title,
    description: m.blurb,
    url: canonical,
    learningResourceType: m.type,
    educationalLevel: m.level,
    about: m.topic,
    timeRequired: m.time,
    inLanguage: 'en',
    isAccessibleForFree: true,
    author: { '@type': 'Person', name: 'Tarek Atwan', url: 'https://www.tarekatwan.com' },
    isPartOf: { '@type': 'WebSite', name: 'ML_LAB', url: SITE + '/' },
  };

  const extraAssets = [];
  if (DECISION_IDS.has(m.id)) {
    extraAssets.push('  <link rel="stylesheet" href="/assets/archetype-decision.css">');
    extraAssets.push('  <script defer src="/assets/decision-workbench.js"></script>');
  }
  if (AIF_IDS.has(m.id)) {
    extraAssets.push('  <link rel="stylesheet" href="/assets/archetype-aip.css">');
    extraAssets.push('  <script defer src="/assets/aif-studio.js"></script>');
  }
  if (m.id === 'aip-hub') {
    extraAssets.push('  <link rel="stylesheet" href="/assets/archetype-aip.css">');
  }

  return `  <!-- LAB:HEAD:START -->
  <meta name="description" content="${esc(m.blurb)}">
  <meta name="author" content="Tarek Atwan">
  <meta name="robots" content="index, follow, max-image-preview:large">
  <link rel="canonical" href="${canonical}">

  <meta property="og:type" content="article">
  <meta property="og:url" content="${canonical}">
  <meta property="og:site_name" content="ML_LAB">
  <meta property="og:title" content="${esc(m.title)} — ML_LAB">
  <meta property="og:description" content="${esc(m.short)}">

  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:creator" content="@tarekatwan">
  <meta name="twitter:title" content="${esc(m.title)} — ML_LAB">
  <meta name="twitter:description" content="${esc(m.short)}">

${FONTS_LINK}
  <link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="/assets/lab.css">
${extraAssets.length ? extraAssets.join('\n') + '\n' : ''}  <script defer src="/assets/lab.js" data-lab-page="${esc(m.title)}"></script>

  <script type="application/ld+json">
${JSON.stringify(ld, null, 2)}
  </script>
  <!-- LAB:HEAD:END -->`;
}

function subHeadBlock({ pageTitle, parentTitle, parentHref, extraTags = [] }) {
  const parentAttrs = parentTitle && parentHref
    ? ` data-lab-parent="${esc(parentTitle)}" data-lab-parent-href="${esc(parentHref)}"`
    : '';
  return `  <!-- LAB:HEAD:START -->
${FONTS_LINK}
  <link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="/assets/lab.css">
${extraTags.length ? extraTags.join('\n') + '\n' : ''}  <script defer src="/assets/lab.js" data-lab-page="${esc(pageTitle)}"${parentAttrs}></script>
  <!-- LAB:HEAD:END -->`;
}

function aboutBlock(m) {
  const ctx = CONTEXT[m.id] || { covers: m.topic, who: 'machine learning practitioners' };

  const siblings = modules
    .filter((o) => o.id !== m.id && (o.routes || []).some((r) => (m.routes || []).includes(r)))
    .slice(0, 4);

  const routeLabels = (m.routes || [])
    .map((id) => (routes.find((r) => r.id === id) || {}).label)
    .filter(Boolean);

  const related = siblings.length
    ? `        <h3>Related playbooks</h3>
        <ul class="lab-about__list">
${siblings.map((s) => `          <li><a href="/${s.href}">${esc(s.title)}</a><span>${esc(s.topic)} &middot; ${esc(s.time)}</span></li>`).join('\n')}
        </ul>`
    : '';

  return `  <!-- LAB:ABOUT:START -->
  <section class="lab-about lab-chrome" id="lab-about" aria-labelledby="lab-about-h">
    <div class="lab-about__inner">
      <div>
        <div class="lab-about__label">// About this playbook</div>
        <h2 id="lab-about-h">${esc(m.title)}</h2>
        <p>${esc(m.blurb)}</p>
        <p>It covers ${esc(ctx.covers)}. It is written for ${esc(ctx.who)}.</p>
        <p>Every branch states the trade-off that decided it, so the recommendation you end on comes with the reasoning
          attached &mdash; something you can paste into a design note or defend in a review.</p>
        <div class="lab-about__facts">
          <span class="lab-about__fact">${esc(m.type)}</span>
          <span class="lab-about__fact">${esc(m.topic)}</span>
          <span class="lab-about__fact">${esc(m.time)}</span>
          <span class="lab-about__fact">${esc(m.level)}</span>
${routeLabels.map((l) => `          <span class="lab-about__fact">${esc(l)}</span>`).join('\n')}
        </div>
      </div>
      <div class="lab-about__side">
${related}
        <h3>Who made this</h3>
        <p class="lab-about__by">
          Built by <a href="https://www.tarekatwan.com" rel="noopener">Tarek Atwan</a> &mdash; twenty years in data and
          AI, four books, four-time Pluralsight Elite instructor, Fortune 500 engagements across eight countries.
          Consulting through <a href="https://www.ensemblemethods.com" rel="noopener">Ensemble Methods</a>.
          Source <a href="https://github.com/tatwan/tatwan.github.io" rel="noopener">on GitHub</a>.
        </p>
      </div>
    </div>
  </section>
  <!-- LAB:ABOUT:END -->`;
}

/* ------------------------------------------------------------------ *
 * Injection & Cleanup helpers
 * ------------------------------------------------------------------ */

function upsert(html, mark, block, anchor) {
  const re = new RegExp(`[ \\t]*<!-- ${mark}:START -->[\\s\\S]*?<!-- ${mark}:END -->\\n?`);
  if (re.test(html)) return html.replace(re, block + '\n');
  const at = html.lastIndexOf(anchor);
  if (at === -1) return null;
  return html.slice(0, at) + block + '\n' + html.slice(at);
}

function setTitle(html, m) {
  const want = `${m.title} — ML_LAB | Tarek Atwan`;
  if (/<title>[\s\S]*?<\/title>/.test(html)) {
    return html.replace(/<title>[\s\S]*?<\/title>/, `<title>${esc(want)}</title>`);
  }
  return html.replace('</head>', `  <title>${esc(want)}</title>\n</head>`);
}

function stripDuplicateChrome(rel, html) {
  let out = html;

  if (rel === 'rag-academy/index.html') {
    out = out.replace(
      /[ \t]*<a\s+href="\.\.\/index\.html"\s+style="position:fixed;[^>]*>&larr;\s*ML_LAB<\/a>\s*\n/,
      ''
    );
  }

  // Strip duplicate inline <- ML_LAB backlinks on React decision tools & AIF-C01 pages
  out = out.replace(
    /[ \t]*<a\s+href="index\.html"\s+className="[^"]*">&larr;\s*ML_LAB<\/a>\s*\n?/g,
    ''
  );

  // Strip duplicate Created by Tarek Atwan @ ML_LAB paragraphs inside page canvases
  out = out.replace(
    /[ \t]*<p\s+className="(?:mt-2 text-xs|text-slate-600 text-xs mt-2|mt-1)">[\s\S]*?Created by Tarek Atwan[\s\S]*?ML_LAB[\s\S]*?<\/p>\s*\n?/g,
    ''
  );

  if (rel === 'aws_visual_references.html') {
    out = out.replace(
      /[ \t]*<a\s+href="index\.html"[\s\S]*?&larr;\s*ML_LAB[\s\S]*?<\/a>\s*\n?/,
      ''
    );
    out = out.replace(
      /[ \t]*<!-- Footer -->\s*\n?[ \t]*<footer class="border-t[\s\S]*?<\/footer>\s*\n?/,
      ''
    );
  }

  if (rel === 'DP-700-InteractiveStudy.html') {
    out = out.replace(
      /[ \t]*<a\s+href="index\.html"\s+class="hbtn"[^>]*>←\s*ML_LAB<\/a>\s*\n?/,
      ''
    );
  }

  if (rel === 'DataEngineering/index.html') {
    out = out.replace(/[ \t]*<a href="\.\.\/index\.html" class="btn-back">← ML_LAB<\/a>\s*/g, '');
    out = out.replace(/<a href="\.\.\/index\.html"[^>]*>← ML_LAB<\/a>/g, '');
    out = out.replace(/[ \t]*<footer class="site-footer">[\s\S]*?<\/footer>\s*\n?/, '');
  }

  if (rel === 'GaTech/CSE 8803/index.html') {
    out = out.replace(
      /[ \t]*<div style="margin-bottom:\s*12px;"><a href="\.\.\/\.\.\/index\.html"[^>]*>&larr;\s*ML_LAB<\/a><\/div>\s*\n?/,
      ''
    );
    out = out.replace(
      /[ \t]*<footer>CSE 8803 Applied NLP[\s\S]*?<\/footer>\s*\n?/,
      ''
    );
  }

  return out;
}

const decodeHtmlText = (s) => String(s)
  .replace(/&amp;/g, '&')
  .replace(/&lt;/g, '<')
  .replace(/&gt;/g, '>')
  .replace(/&quot;/g, '"')
  .replace(/&#39;/g, "'")
  .replace(/&ndash;/g, '–')
  .replace(/&mdash;/g, '—')
  .replace(/\s+/g, ' ')
  .trim();

/* ------------------------------------------------------------------ *
 * Run Top-Level Playbooks (17)
 * ------------------------------------------------------------------ */

let changed = 0;
const notes = [];

for (const m of modules) {
  const rel = decodeURIComponent(m.href.endsWith('/') ? m.href + 'index.html' : m.href);
  const abs = path.join(ROOT, rel);

  if (!fs.existsSync(abs)) { notes.push(`  MISSING   ${rel}`); continue; }

  let html = fs.readFileSync(abs, 'utf8');
  const before = html;
  const steps = [];

  html = setTitle(html, m);
  if (html !== before) steps.push('title');

  const withHead = upsert(html, 'LAB:HEAD', headBlock(m), '</head>');
  if (withHead) { if (withHead !== html) steps.push('head'); html = withHead; }
  else notes.push(`  no </head>  ${rel}`);

  const withAbout = upsert(html, 'LAB:ABOUT', aboutBlock(m), '</body>');
  if (withAbout) { if (withAbout !== html) steps.push('about'); html = withAbout; }
  else notes.push(`  no </body>  ${rel}`);

  const cleaned = stripDuplicateChrome(rel, html);
  if (cleaned !== html) { steps.push('strip-dup-chrome'); html = cleaned; }

  if (html === before) { console.log(`  =  ${rel}`); continue; }
  if (!DRY) fs.writeFileSync(abs, html);
  changed++;
  console.log(`  ${DRY ? '~' : '+'}  ${rel}  [${steps.join(', ')}]`);
}

/* ------------------------------------------------------------------ *
 * Run Sub-Pages (38) + 404.html (1)
 * ------------------------------------------------------------------ */

const subTargets = [];

// 10 Data Engineering modules
for (let i = 1; i <= 10; i++) {
  const rel = `DataEngineering/module${i}.html`;
  const abs = path.join(ROOT, rel);
  if (!fs.existsSync(abs)) continue;
  const raw = fs.readFileSync(abs, 'utf8');
  const h1 = (raw.match(/<h1>([\s\S]*?)<\/h1>/) || [])[1] || `Module ${i}`;
  subTargets.push({
    rel,
    pageTitle: `Module ${i}: ${decodeHtmlText(h1.replace(/<[^>]+>/g, ''))}`,
    parentTitle: 'Data Engineering Foundations',
    parentHref: '/DataEngineering/index.html',
    extraTags: [
      '  <link rel="stylesheet" href="/assets/archetype-course.css">',
      `  <script defer src="/assets/course-subbar.js" data-course="de" data-index="${i}"></script>`,
    ],
  });
}

// 15 GaTech CSE 8803 weeks
for (let i = 1; i <= 15; i++) {
  const rel = `GaTech/CSE 8803/week${i}.html`;
  const abs = path.join(ROOT, rel);
  if (!fs.existsSync(abs)) continue;
  const raw = fs.readFileSync(abs, 'utf8');
  const h1 = (raw.match(/<h1>([\s\S]*?)<\/h1>/) || [])[1] || `Week ${i}`;
  subTargets.push({
    rel,
    pageTitle: `Week ${i}: ${decodeHtmlText(h1.replace(/<[^>]+>/g, ''))}`,
    parentTitle: 'Applied NLP — CSE 8803',
    parentHref: '/GaTech/CSE%208803/index.html',
    extraTags: [
      '  <link rel="stylesheet" href="/assets/archetype-course.css">',
      `  <script defer src="/assets/course-subbar.js" data-course="cse8803" data-index="${i}"></script>`,
    ],
  });
}

// 13 AIP-C01 sub-pages
const AIP_TITLES = {
  'study-guide.html': 'Deep Study Guide',
  'aip-c01-studyguide.html': 'Visual Study Guide',
  'flashcards.html': 'Flashcards & Drills',
  'practice-exam.html': 'Practice Exam (85 Qs)',
  'quick-drill.html': 'Quick Drill Simulator',
  'final-practice.html': 'Final Practice (72 Qs)',
  'gap-cheatsheet.html': 'Gap Cheatsheet',
  'advanced-impl.html': 'Advanced Implementation',
  'troubleshooting-eval.html': 'Troubleshooting & Evaluation',
  'study-d1.html': 'Domain 1: Foundation Models, Data & RAG',
  'study-d2.html': 'Domain 2: Implementation & Integration',
  'study-d3.html': 'Domain 3: Safety, Security & Governance',
  'study-d45.html': 'Domains 4 & 5: Ops, Eval & Troubleshooting',
};

for (const [file, pageTitle] of Object.entries(AIP_TITLES)) {
  const rel = `aip/${file}`;
  if (!fs.existsSync(path.join(ROOT, rel))) continue;
  subTargets.push({
    rel,
    pageTitle,
    parentTitle: 'AIP-C01 Prep Hub',
    parentHref: '/aip/index.html',
    extraTags: [
      '  <link rel="stylesheet" href="/assets/archetype-aip.css">',
    ],
  });
}

// 404.html
subTargets.push({
  rel: '404.html',
  pageTitle: '404 — Page Not Found',
  parentTitle: null,
  parentHref: null,
  extraTags: [],
});

for (const t of subTargets) {
  const abs = path.join(ROOT, t.rel);
  let html = fs.readFileSync(abs, 'utf8');
  const before = html;
  const withHead = upsert(html, 'LAB:HEAD', subHeadBlock(t), '</head>');
  if (withHead) html = withHead;
  else notes.push(`  no </head>  ${t.rel}`);

  if (html === before) { console.log(`  =  ${t.rel}`); continue; }
  if (!DRY) fs.writeFileSync(abs, html);
  changed++;
  console.log(`  ${DRY ? '~' : '+'}  ${t.rel}  [sub-head]`);
}

if (notes.length) console.log('\n' + notes.join('\n'));
console.log(`\n  ${changed} page${changed === 1 ? '' : 's'} ${DRY ? 'would change' : 'updated'}`);
