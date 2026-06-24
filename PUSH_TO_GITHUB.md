# Putting GradLab on GitHub

Do these on **your own machine** (Windows), where git and your GitHub login live.

## 0. One-time cleanup
A partial `.git` folder was left inside `gradlab` and could not be removed automatically.
**Delete the `gradlab\.git` folder first** (it may be hidden — enable "Hidden items" in
File Explorer's View tab), or run in PowerShell from inside `gradlab`:

```powershell
Remove-Item -Recurse -Force .git
```

## 1. Install prerequisites (if needed)
- Git: https://git-scm.com/download/win
- (optional) GitHub CLI `gh`: https://cli.github.com  — makes step 3 one command

## 2. Create the repo on GitHub
Go to https://github.com/new → name it `gradlab` → **Public** → **do not** add a README
or .gitignore (we already have them) → Create repository.

## 3. Initialise and push
Open a terminal in the `gradlab` folder and run:

```bash
git init
git add .
git commit -m "GradLab: from-scratch ML engine + interactive playground + serverless backend + CI/CD"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/gradlab.git
git push -u origin main
```

(Or, with the GitHub CLI, replace steps 2–3 with: `gh repo create gradlab --public --source=. --push`.)

## 4. Replace the placeholder
Find-and-replace `YOUR_GITHUB_USERNAME` with your real username in:
- `README.md` (CI badge + clone links)
- `frontend/config.js` only later, once you have the Lambda URL

## 5. Turn on the free hosting
- **GitHub Pages** (the live site): repo → Settings → Pages → Source = "GitHub Actions".
  After the next push, your demo is live at `https://YOUR_GITHUB_USERNAME.github.io/gradlab/`.
  > The offline demo works immediately — no backend required.
- **CI**: already runs automatically on every push (lint + tests). Watch it under the Actions tab.

## 6. (Later) Deploy the Python backend to AWS Lambda
Only when you want the "real engine" + live AI tutor:
1. Create an AWS account (free tier).
2. Add repo **secrets**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (and `ANTHROPIC_API_KEY` for the AI tutor).
3. Add repo **variable** `DEPLOY_BACKEND` = `true` (and optionally `AI_PROVIDER` = `anthropic`).
4. Push. The `deploy.yml` workflow builds and deploys it; copy the printed Function URL into `frontend/config.js`.

That's it. Order of impressiveness for your CV: a green CI badge + a live Pages link first;
the Lambda + AI tutor is the "wow" upgrade you can add over a weekend.
