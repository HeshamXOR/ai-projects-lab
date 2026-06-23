# ai-projects-lab

A collection of **real, runnable AI applications** — each one solves an actual problem, ships with a working **Gradio** preview, and runs on **[Lightning AI](https://lightning.ai)** (or any machine, CPU or GPU).

Every project is self-contained: its own `app.py`, `requirements.txt`, and README. Open a project folder, install, run `python app.py`, and you get a shareable web UI.

---

## Projects

| # | Project | What it does | Tech | Preview |
|---|---------|--------------|------|---------|
| 1 | [**RAG Document Q&A**](01-rag-doc-qa/) | Upload a PDF, ask questions, get answers **with citations** to the source pages | Sentence-Transformers + FAISS + an LLM (local or API) | Gradio |
| 2 | [**Resume ↔ Job Matcher**](02-resume-matcher/) | Paste a resume + a job description → match score, matched skills, and **gap analysis** | Semantic embeddings + keyword extraction | Gradio |
| 3 | [**Image Q&A (Multimodal)**](03-image-qa/) | Upload an image → auto caption + ask **visual questions** about it | BLIP vision-language model | Gradio |
| 4 | [**Smart Summarizer**](04-smart-summarizer/) | Paste a long article/transcript → structured summary + **action items** | Transformer summarization (chunked) | Gradio |

Each table row links to a folder with a full README, a screenshot slot, and run instructions.

---

## Quickstart (any project)

```bash
cd 01-rag-doc-qa            # or any project folder
pip install -r requirements.txt
python app.py               # opens a Gradio app at http://localhost:7860
```

On **Lightning AI**: open a Studio, clone this repo, `cd` into a project, install, and run — Gradio prints a public `*.gradio.live` share link you can use as the preview. See [LIGHTNING.md](LIGHTNING.md) for the exact steps and GPU advice.

---

## Design principles (why these are "real" projects)

- **Runs anywhere.** Every app detects GPU vs CPU and picks sensible model sizes, so the preview works even on a free CPU Studio — just slower.
- **No mandatory API keys.** Projects default to open local models. Where an LLM helps (Project 1), you can optionally plug in an API key, but there's always a working local/extractive fallback.
- **Graceful failure.** Missing model, no GPU, bad upload — the UI shows a clear message instead of crashing.
- **Honest previews.** Screenshots in each README are from real runs, not mockups.

---

## Repo layout

```
ai-projects-lab/
├── README.md                ← you are here
├── LIGHTNING.md             ← how to run + preview on Lightning AI
├── LICENSE
├── 01-rag-doc-qa/
├── 02-resume-matcher/
├── 03-image-qa/
└── 04-smart-summarizer/
```

## License

MIT — see [LICENSE](LICENSE).
