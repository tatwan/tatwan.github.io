# Publish this lesson with GitHub Pages

The lesson is ready to publish directly from this `docs` folder. It does not need Node.js, a package install, or a build step.

## Option A — add it to the `tatwan.github.io` repository

1. Copy `index.html`, `.nojekyll`, and `og.png` into a subfolder in that repository, for example `data-engineering-system/`.
2. Push the changes.
3. Open `https://tatwan.github.io/data-engineering-system/`.

If `tatwan.github.io` is already the repository's Pages site, no additional Pages setting should be necessary for a new subfolder.

## Option B — publish this repository's `docs` folder

1. Push this project to GitHub.
2. In the repository, open **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Choose the branch (usually `main`) and the `/docs` folder, then save.

GitHub's current documentation for publishing from a branch or `/docs` folder is available at <https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site>.

## Files

- `index.html` — the complete lesson, including its CSS and JavaScript.
- `og.png` — optional social-sharing preview; the lesson still works without it.
- `.nojekyll` — tells GitHub Pages to serve the folder as plain static files.
