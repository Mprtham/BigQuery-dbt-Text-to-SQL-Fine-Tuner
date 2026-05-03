# SQL Forge — Portfolio Landing Page

## View locally

```bash
cd "C:\BigQuery-dbt Text-to-SQL Fine-Tuner\docs"
python -m http.server 8000
```

Then open: http://localhost:8000

> **Note:** The page also works when opened directly as a file (`file://`), but serving via HTTP enables Prism.js CDN syntax highlighting.

## Structure

```
docs/
├── index.html        # Single-page landing site
├── assets/
│   ├── style.css     # All styles (dark theme, responsive)
│   └── app.js        # TypeWriter, scroll animations, mock demo
└── README.md         # This file
```

## Sections

1. **Hero** — Animated typing demo cycling through 3 SQL examples
2. **Architecture** — Interactive flow diagram with hover tooltips
3. **Why SQL Forge** — Side-by-side comparison + 4 tabbed code examples
4. **Training Pipeline** — Step timeline + hyperparameter config cards + benchmark table
5. **Interactive Demo** — Mock SQL generation (BigQuery + dbt dialect)
6. **Tech Stack** — 9-card grid of all tools used
7. **Footer** — HuggingFace links, GitHub, license

## Deploy to GitHub Pages

Once pushed, enable GitHub Pages from `Settings → Pages → Source: docs/` on the `main` branch.
