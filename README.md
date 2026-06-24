# GradLab — interactive ML playground

[![CI](https://github.com/nrwre/gradlab/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/gradlab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

An interactive, mobile-friendly web app that lets you **watch optimisers descend and classifiers learn**, powered by a **from-scratch deep-learning + classical-ML engine** I wrote in NumPy. It ships with an **AI tutor** that explains each run, runs on **AWS Lambda**, and deploys through **GitHub Actions CI/CD**.

> Built on top of my AIML Lab coursework (IIT Kharagpur, EE22202): gradient methods, regression, SVMs, decision trees & random forests, CNNs.

**Live demo:** _add your GitHub Pages URL here_ · works on laptop **and** phone, no install.

---

## What it does

- **Optimiser playground** — pick a loss surface (a quadratic with an adjustable condition number κ, or Rosenbrock), an optimiser (GD / Momentum / Nesterov / Adam) and a learning rate, then watch the descent trajectory animate over a contour plot.
- **Classifier playground** — choose a dataset (moons / circles / blobs) and a model (RBF or linear SVM, decision tree, random forest, or a small neural net) and watch the decision boundary form, with live accuracy (and OOB score for the forest).
- **AI tutor** — explains, in plain language grounded in the actual numbers, *why* a run behaved the way it did.

The site runs **fully in-browser by default** (offline demo). Point it at the deployed backend to use the real Python engine for everything.

## Architecture

```
frontend/ (static, GitHub Pages)  ──HTTP──▶  backend/ (AWS Lambda)
        │                                          │
        │  in-browser engine.js (offline mode)     ├─ engine/  nanograd (NumPy)
        └─ Plotly visualisations                   └─ ai_tutor (Anthropic / OpenAI / stub)
CI/CD:  .github/workflows  →  lint + test the engine, deploy frontend & Lambda
```

| Folder | What's inside |
|--------|----------------|
| `engine/` | `nanograd` — autograd, optimisers, NN layers, SVM, decision tree, random forest, plus pytest suite |
| `backend/` | Lambda handler (`/optimize`, `/classify`, `/explain`), pluggable AI tutor, SAM template |
| `frontend/` | Responsive single-page app (HTML + Plotly), in-browser fallback engine |
| `.github/workflows/` | `ci.yml` (lint + test) and `deploy.yml` (Pages + Lambda) |
| `scripts/` | `run_local.py` — run the backend locally with no AWS |

## Quick start

```bash
# 1. Engine + tests
cd engine
pip install -e ".[dev]"
pytest -q                 # 9 passing tests
ruff check nanograd

# 2. Run the backend locally (no AWS needed)
cd ..
python scripts/run_local.py        # http://localhost:8000

# 3. Open the frontend
#    - offline demo: just open frontend/index.html in a browser
#    - with backend: set API_BASE = "http://localhost:8000" in frontend/config.js
```

## Deploy

- **Frontend → GitHub Pages**: enable Pages (Settings → Pages → "GitHub Actions"). The `deploy.yml` workflow publishes `frontend/` on every push to `main`.
- **Backend → AWS Lambda**: add repo **secrets** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (and optionally `ANTHROPIC_API_KEY`), set repo **variable** `DEPLOY_BACKEND=true`, then push. Or deploy manually:
  ```bash
  cd backend && bash build.sh && sam build && sam deploy --guided
  ```
  Copy the printed Function URL into `frontend/config.js`.

## AI tutor providers

Set environment variables on the Lambda (or locally):

| `AI_PROVIDER` | needs | behaviour |
|---------------|-------|-----------|
| `stub` (default) | nothing | deterministic, rule-based explanation — runs with zero keys |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude explanation |
| `openai` | `OPENAI_API_KEY` | GPT explanation |

## Roadmap

- [ ] Animate the trajectory step-by-step (play/pause)
- [ ] Add per-feature importance view for the random forest
- [ ] Cache backend responses; add request throttling
- [ ] Optional WebAssembly build of the engine for offline parity

## License

MIT © Dhruv Narware
