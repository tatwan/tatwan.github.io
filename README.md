# ML_LAB

**Decision playbooks for machine learning and GenAI practitioners.**

🔗 **Live site**: [tatwan.github.io](https://tatwan.github.io)

---

## About

Sixteen short, self-contained playbooks. Each one answers a question you are actually stuck on —
which algorithm, which metric, whether to fine-tune or retrieve — and leaves you with a decision you
can defend in a design review. No sign-up, no install, everything runs in the browser.

Press <kbd>⌘K</kbd> (or <kbd>/</kbd>) anywhere on the site to search across every playbook **and every
decision node inside them** — around 530 indexed entries. Results deep-link straight to the node.

---

## The playbooks

### Scope and strategy

| | Playbook | What it decides |
| --- | --- | --- |
| 10 | [AI Opportunity Scorer](https://tatwan.github.io/ai_opportunity_scorer.html) | Build, buy, experiment or kill, scored across six value dimensions |
| 11 | [AI Solution Navigator](https://tatwan.github.io/ai_solution_navigator.html) | Rules, classical ML, deep learning, GenAI, agents or operations research |

### Model and method

| | Playbook | What it decides |
| --- | --- | --- |
| 01 | [Algorithm Selector](https://tatwan.github.io/ml_algorithm_selector.html) | The right algorithm for your problem type and data |
| 02 | [Feature Engineering Playbook](https://tatwan.github.io/feature_engineering_playbook.html) | Encoding, scaling, outliers and selection, per data type |
| 04 | [Data Engineering Foundations](https://tatwan.github.io/DataEngineering/) | A ten-module, vendor-neutral course |
| 15 | [Applied NLP — CSE 8803](https://tatwan.github.io/GaTech/CSE%208803/) | Weekly study notes from the Georgia Tech OMSA course |

### Measurement

| | Playbook | What it decides |
| --- | --- | --- |
| 03 | [Metric Decision Tree](https://tatwan.github.io/model_evaluation_interactive.html) | Classification, regression and ranking metrics, imbalance included |
| 13 | [FM Evaluation Metrics](https://tatwan.github.io/fm_evaluation_metrics.html) | Perplexity, ROUGE, BERTScore, RAGAS, LLM-as-judge |

### Building with LLMs

| | Playbook | What it decides |
| --- | --- | --- |
| 00 | [GenAI Foundations Navigator](https://tatwan.github.io/genai_foundations_navigator.html) | The vocabulary everything else assumes |
| 12 | [GenAI Technique Selector](https://tatwan.github.io/genai_techniques_selector.html) | RAG, fine-tuning, LoRA, quantization, distillation |
| 14 | [RAG Academy](https://tatwan.github.io/rag-academy/) | Retrieval architectures that survive production |

### Certification

| | Playbook | Exam |
| --- | --- | --- |
| 05 | [AWS AI Practitioner Guide](https://tatwan.github.io/aws_ai_practitioner.html) | AWS AIF-C01 |
| 06 | [AIF-C01 Practice Exam](https://tatwan.github.io/aws_ai_practitioner_exam.html) | AWS AIF-C01 — 114 timed questions |
| 07 | [Visual References](https://tatwan.github.io/aws_visual_references.html) | AWS AIF-C01 — diagrams |
| 08 | [AIP-C01 Prep Hub](https://tatwan.github.io/aip/) | AWS AI Practitioner (Generative AI) — 6 tools |
| 09 | [DP-700 Study Guide](https://tatwan.github.io/DP-700-InteractiveStudy.html) | Microsoft Fabric Data Engineering Associate |

---

## Local development

```bash
git clone https://github.com/tatwan/tatwan.github.io.git
cd tatwan.github.io
python3 -m http.server 8000     # http://localhost:8000
```

No bundler, no package manager, nothing to install.

## Build

Node 18+ regenerates everything that is derived rather than authored:

```bash
node tools/build-site.mjs        # home page sections, sitemap.xml, search index
node tools/retrofit-pages.mjs    # shared shell + SEO on every playbook page
```

Both are idempotent. `assets/modules.json` is the single source of truth for the catalogue — edit it,
re-run, and the home page, the sitemap and the search index all follow.

## Architecture

| Path | Role |
| --- | --- |
| `assets/modules.json` | The catalogue: 16 playbooks, 5 routes |
| `assets/lab.css` / `assets/lab.js` | Shared shell — header, breadcrumb, footer, ⌘K palette |
| `assets/search-index.json` | Generated; lazy-loaded on first search |
| `tools/*.mjs` | The build scripts |

Each playbook is a self-contained page that keeps its own layout and theme. The shell is namespaced
(`lab-*`) dark chrome that sits above any canvas, light or dark, without touching it.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a playbook.

---

## Author

**Tarek Atwan** — author, educator and AI/ML consultant. Twenty years in data and AI, four books,
four-time Pluralsight Elite instructor, Fortune 500 engagements across eight countries.

- Website: [tarekatwan.com](https://www.tarekatwan.com)
- Consulting: [Ensemble Methods](https://www.ensemblemethods.com)
- LinkedIn: [tarekatwan](https://www.linkedin.com/in/tarekatwan) · GitHub: [@tatwan](https://github.com/tatwan)

## License

MIT — see [LICENSE](LICENSE).

> **Note:** this repository must remain **public** for GitHub Pages to serve
> `https://tatwan.github.io/`.
