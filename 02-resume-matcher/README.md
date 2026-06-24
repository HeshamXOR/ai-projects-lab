# 🎯 Resume ↔ Job Matcher

Paste a resume and a job description → get a **semantic match score**, the skills you both share, and a **gap analysis** of what the job wants that your resume is missing.

![preview](preview.gif)
<!-- Record a short clip on Lightning and save it as preview.gif here. -->

## What I implemented from scratch

- **TF-IDF vectorizer** (n-grams, L2-normalized) — `core/tfidf.py`
- **Logistic regression** via gradient descent — `core/logreg.py`
- **Skill co-occurrence graph** for transferable-skill suggestions — `core/skillgraph.py`

The embedding model is just one of three blended signals. See [EXPLAINER.md](EXPLAINER.md). Verify with `pytest`.

## What it does

- **Semantic similarity** — embeds the whole resume and job description and measures cosine similarity, so it understands meaning, not just keywords.
- **Skill extraction** — detects skills from a curated vocabulary (with aliases like `js → javascript`, `k8s → kubernetes`).
- **Gap analysis** — shows matched skills, missing skills (in the job but not your resume), and extras you bring.
- **Blended score** — 60% semantic + 40% explicit skill coverage, with a plain-English verdict.

## Why it's real

Every job seeker and recruiter does this manually. It's the core of applicant-tracking and job-matching tools — a clear, relatable problem with an obviously useful output.

## Run it

```bash
pip install -r requirements.txt
python app.py            # http://localhost:7860  (+ public gradio.live link)
```

No GPU and no API key needed — it runs on a free CPU Studio.

## How it works (files)

- `matcher.py` — `Matcher.match()` returns score, matched/missing/extra skills, verdict. The `SKILLS` dict is easy to extend.
- `app.py` — the Gradio UI, pre-filled with a sample resume + JD so the preview works on first click.

## Extend it

- Add domain skills to the `SKILLS` dict in `matcher.py`.
- Parse PDF/DOCX resumes (reuse the PDF reader from project 1).
- Weight skills by how often they appear in the JD.
