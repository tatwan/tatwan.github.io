#!/usr/bin/env node
/**
 * retrofit-pages.mjs
 *
 * Gives every playbook page the shared shell and the SEO it was missing:
 *
 *   - the ML_LAB header, breadcrumb, footer and Cmd+K palette (lab.css + lab.js)
 *   - a consistent <title>
 *   - description, canonical, Open Graph, Twitter and JSON-LD
 *   - an "About this playbook" band of real prose, because the interactive
 *     apps render everything client-side and ship zero indexable text
 *   - deep-link support: /page.html#node_id opens that decision node
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

/** Tools whose decision tree we can deep-link into. */
const TREE_TOOLS = new Set([
  'ml_algorithm_selector.html',
  'model_evaluation_interactive.html',
  'feature_engineering_playbook.html',
  'genai_techniques_selector.html',
  'fm_evaluation_metrics.html',
  'genai_foundations_navigator.html',
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
    covers: '114 questions distributed across the five exam domains in the same proportions as the real paper, each with a written explanation',
    who: 'candidates who have studied and now need to find out what they actually retained',
  },
  'aws-visual': {
    covers: 'confusion matrices, retrieval-augmented generation pipelines, and the full foundation-model lifecycle from pre-training to deployment',
    who: 'visual learners, and anyone who needs a diagram to put in a slide',
  },
  'aip-hub': {
    covers: 'a study guide, flashcards, a gap cheatsheet, domain drills, a full practice exam and a timed simulator for the generative AI practitioner exam',
    who: 'candidates who want a complete preparation track rather than a single resource',
  },
  'dp700': {
    covers: 'Fabric architecture, ingestion, Delta Lake, KQL, PySpark, security boundaries, CI/CD and the specific limits that appear as exam questions',
    who: 'candidates for the Microsoft Fabric Data Engineering Associate certification',
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
    covers: 'chunking, embeddings, vector stores, retrieval strategies, reranking, evaluation and the failure modes that only appear in production',
    who: 'engineers building or debugging a retrieval-augmented system',
  },
  'cse8803': {
    covers: 'text preprocessing, classical classifiers, word embeddings, neural networks, RNNs, LSTMs and attention mechanisms',
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

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="/assets/lab.css">
  <script defer src="/assets/lab.js" data-lab-page="${esc(m.title)}"></script>

  <script type="application/ld+json">
${JSON.stringify(ld, null, 2)}
  </script>
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
 * Injection helpers
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

/** Jump straight to a decision node when the URL carries #node_id. */
const DEEPLINK = `
      // LAB: deep link — /page.html#node_id opens that decision node.
      React.useEffect(() => {
        const jump = () => {
          const h = decodeURIComponent((window.location.hash || '').slice(1));
          if (h && tree[h]) setCurrentNode(h);
        };
        jump();
        window.addEventListener('hashchange', jump);
        return () => window.removeEventListener('hashchange', jump);
      }, []);`;

function addDeepLink(html) {
  if (html.includes('LAB: deep link')) return html;
  const needle = "const [currentNode, setCurrentNode] = useState('start');";
  if (!html.includes(needle)) return null;
  return html.replace(needle, needle + DEEPLINK);
}

/* ------------------------------------------------------------------ *
 * Run
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

  if (TREE_TOOLS.has(rel)) {
    const linked = addDeepLink(html);
    if (linked) { if (linked !== html) steps.push('deeplink'); html = linked; }
    else notes.push(`  no useState anchor  ${rel}`);
  }

  if (html === before) { console.log(`  =  ${rel}`); continue; }
  if (!DRY) fs.writeFileSync(abs, html);
  changed++;
  console.log(`  ${DRY ? '~' : '+'}  ${rel}  [${steps.join(', ')}]`);
}

if (notes.length) console.log('\n' + notes.join('\n'));
console.log(`\n  ${changed} page${changed === 1 ? '' : 's'} ${DRY ? 'would change' : 'updated'}`);
