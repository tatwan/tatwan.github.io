# Contributing Guidelines

## File Naming Convention

### HTML Pages
Use lowercase with underscores to separate words:

**Format**: `{category}_{description}.html`

**Examples**:
- `ml_algorithm_selector.html` - Machine Learning tools
- `model_evaluation_interactive.html` - Model evaluation tools
- `aws_ai_practitioner.html` - Certification prep
- `ds_statistical_tests.html` - Data Science tools (future)
- `dl_architecture_selector.html` - Deep Learning tools (future)

### Category Prefixes
- `ml_` - Machine Learning
- `dl_` - Deep Learning
- `ds_` - Data Science
- `cert_` or `aws_`, `azure_` - Certifications
- No prefix - General/standalone tools

---

## Current Structure

```
tatwan.github.io/
├── index.html                          # Main landing page
├── ml_algorithm_selector.html          # ML tool
├── model_evaluation_interactive.html   # ML tool
├── aws_ai_practitioner.html            # Certification
├── 404.html                            # Error page
├── sitemap.xml                         # SEO
├── robots.txt                          # SEO
├── archive/                            # Legacy blog posts & images
├── learn/                              # Learning resources
├── README.md
├── LICENSE
└── CONTRIBUTING.md                     # This file
```

---

## Adding New Pages

### 1. Create HTML File
Follow the naming convention above and place in the root directory.

### 2. Update the catalogue
Add an entry to `assets/modules.json`, including its type, topic, and routes. This is the source for the full homepage collection and goal filter.

### 3. Regenerate the site
```bash
node tools/build-site.mjs
node tools/retrofit-pages.mjs
```

The first command updates the catalogue, featured resources, learning paths, structured data, sitemap, and search index. The second adds the shared shell and page metadata. Do not edit content inside `LAB:*:START/END` markers by hand. Curated resource IDs and learning paths are selected in `tools/build-site.mjs`.

### 4. Test Locally
```bash
python3 -m http.server 8000
# Visit http://localhost:8000
```

---

## When to Reorganize

Consider moving to a folder-based structure when:
- You have 10+ HTML pages
- Multiple pages share common assets (CSS/JS)
- Categories become more defined

Suggested future structure:
```
tatwan.github.io/
├── index.html
├── ml/
├── certifications/
├── data-science/
└── assets/
```

---

## Design Guidelines

- **Consistency**: Use the Editorial Lab shell (navy, cream, and restrained gold accents); keep each interactive tool’s own canvas theme
- **Typography**: Use readable sans-serif body text and serif editorial headings; reserve monospace for code
- **Interactivity**: Use subtle hover underlines and visible keyboard focus; avoid repeated decorative arrows
- **Mobile-first**: Ensure responsive design
- **SEO**: Include proper meta tags, title, and description

---

## Questions?

Contact: [Tarek Atwan](https://www.ensemblemethods.com)
