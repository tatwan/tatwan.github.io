# ML Learning Lab

**Interactive guides and decision frameworks for machine learning practitioners.**

🔗 **Live Site**: [https://tatwan.github.io](https://tatwan.github.io)

---

## About

ML Learning Lab is a collection of interactive educational resources designed to help data scientists and ML engineers make informed decisions about algorithms, evaluation metrics, and certifications. Each tool provides hands-on, practical guidance for real-world machine learning challenges.

---

## Available Resources

### 🤖 [ML Algorithm Selector](https://tatwan.github.io/ml_algorithm_selector.html)
Interactive decision tree to find the optimal machine learning algorithm for your specific problem type and data characteristics.

### 📊 [Model Evaluation Metrics Guide](https://tatwan.github.io/model_evaluation_interactive.html)
Navigate the landscape of model evaluation metrics. Choose the right metric for classification, regression, and beyond.

### 🎓 [AWS AI Practitioner Study Guide](https://tatwan.github.io/aws_ai_practitioner.html)
Comprehensive study guide with interactive flashcards and quizzes for the AWS Certified AI Practitioner exam.

---

## Technology Stack

- **Pure HTML/CSS/JavaScript** - No build process required
- **Tailwind CSS** - Modern, responsive design
- **Static hosting** - Fast, reliable GitHub Pages deployment
- **SEO optimized** - Structured data, sitemap, and meta tags

---

## Local Development

Serve the site locally for testing:

```bash
# Clone the repository
git clone https://github.com/tatwan/tatwan.github.io.git
cd tatwan.github.io

# Serve locally (Python 3)
python3 -m http.server 8000

# Visit http://localhost:8000
```

---

## Contributing

Interested in adding new tools or improving existing ones? Check out [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- File naming conventions
- Adding new pages
- Design consistency
- SEO best practices

---

## Repository Structure

```
tatwan.github.io/
├── index.html                          # Main landing page
├── ml_algorithm_selector.html          # ML algorithm decision tree
├── model_evaluation_interactive.html   # Metrics selection guide
├── aws_ai_practitioner.html            # AWS certification prep
├── 404.html                            # Custom error page
├── sitemap.xml                         # SEO sitemap
├── robots.txt                          # Search engine directives
├── archive/                            # Legacy blog content
├── learn/                              # Additional learning resources
├── CONTRIBUTING.md                     # Contribution guidelines
├── LICENSE                             # MIT License
└── README.md                           # This file
```

---

## Archive

This repository previously hosted a Jekyll-based blog. Legacy blog posts and images have been moved to the `archive/` folder. The current blog lives at [tatwan.github.io/blog](https://tatwan.github.io/blog/) (separate repository).

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Author

**Tarek Atwan**
- Website: [ensemblemethods.com](https://www.ensemblemethods.com)
- Twitter: [@tarekatwan](https://twitter.com/tarekatwan)
- LinkedIn: [tarekatwan](https://www.linkedin.com/in/tarekatwan)
- GitHub: [@tatwan](https://github.com/tatwan)

---

## Privacy & GitHub Pages

**Note**: This repository must remain **public** for GitHub Pages to work. Making it private will break the site at `https://tatwan.github.io/`.

If you need to keep certain files private (e.g., drafts, notes), add them to `.gitignore` or store them in a separate private repository.
