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

### 2. Update index.html
Add a new card in the resources grid section:

```html
<!-- Card N: Your New Tool -->
<a href="your_new_tool.html" class="resource-card rounded-lg p-6 block fade-up delay-N">
  <!-- Card content -->
</a>
```

### 3. Update sitemap.xml
Add the new page:

```xml
<url>
  <loc>https://tatwan.github.io/your_new_tool.html</loc>
  <lastmod>YYYY-MM-DD</lastmod>
  <changefreq>monthly</changefreq>
  <priority>0.8</priority>
</url>
```

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

- **Consistency**: Match the dark theme aesthetic (slate background, amber accents)
- **Typography**: Use JetBrains Mono for code/technical content
- **Interactivity**: Include hover effects and micro-animations
- **Mobile-first**: Ensure responsive design
- **SEO**: Include proper meta tags, title, and description

---

## Questions?

Contact: [Tarek Atwan](https://www.ensemblemethods.com)
