# ai-projects-lab

A collection of **real AI applications, with the hard parts built from scratch.**

Most "AI project" portfolios are thin wrappers: `pipeline(...)` + a UI. This repo is the opposite — every project has a **core algorithm I implemented by hand** (a vector index, an autodiff engine, a transformer, a diffusion sampler, classical CV, …), wrapped in a runnable **Gradio** app that previews on **[Lightning AI](https://lightning.ai)**. Pretrained models appear as *one component*, never the whole project.

Each project has a `core/` (the from-scratch engine), `tests/` (proof it's correct — run `pytest`), an `EXPLAINER.md` ("what I implemented from scratch" + the math), and an `app.py` (the demo).

---

## Projects

### Foundations (built from scratch, reused across the lab)
| # | Project | What I implemented from scratch |
|---|---------|-------------------------------|
| 10 | [**nanograd**](10-nanograd/) | A reverse-mode **autodiff engine** + NN library — the core of PyTorch. Gradient-checked. |
| 09 | [**microgpt**](09-microgpt/) | A **GPT transformer + BPE tokenizer** in pure PyTorch (no Hugging Face). Trains live. |

### Generative AI / LLMs / RAG
| # | Project | From scratch |
|---|---------|-------------|
| 01 | [**RAG Document Q&A**](01-rag-doc-qa/) | **HNSW** vector index + **BM25** + reciprocal-rank fusion |
| 12 | [**agentic-rag**](12-agentic-rag/) | Query **planner**, citation **verifier**, multi-step reasoning **loop** |
| 08 | [**AI Image Generator**](08-image-generator/) | **DDPM/DDIM** diffusion math + a toy diffusion model |

### NLP
| # | Project | From scratch |
|---|---------|-------------|
| 02 | [**Resume ↔ Job Matcher**](02-resume-matcher/) | **TF-IDF** + **logistic regression** (SGD) + skill graph |
| 04 | [**Smart Summarizer**](04-smart-summarizer/) | **TextRank** (PageRank) + **ROUGE** scoring |
| 05 | [**Sentiment & Emotion**](05-sentiment-emotion/) | **MLP classifier** with hand-derived backprop + calibration metrics |
| 06 | [**Semantic Code Search**](06-code-search/) | **Inverted index** + **BM25F** + AST **call graph** |

### Multimodal / Vision / Audio
| # | Project | From scratch |
|---|---------|-------------|
| 03 | [**Image Q&A**](03-image-qa/) | **Attention rollout** + **Grad-CAM** explainability |
| 07 | [**Speech-to-Text**](07-speech-to-text/) | **Mel-spectrogram** DSP front-end + **VAD** |
| 11 | [**seg-studio**](11-seg-studio/) | **K-means** + **region growing** + connected components (classical CV) |

> See also **[openforge](https://github.com/HeshamXOR)** — an open-source terminal coding agent that makes local models tool-call reliably via JSON-Schema→GBNF constrained decoding.

---

## Quickstart (any project)

```bash
cd 10-nanograd            # or any project folder
pip install -r requirements.txt
pytest -q                 # see the from-scratch core proven correct
python app.py             # launch the Gradio demo (prints a public link)
```

On **Lightning AI**: open a Studio, clone the repo, `cd` into a project, install, and run — Gradio prints a public `*.gradio.live` link for previews. See [LIGHTNING.md](LIGHTNING.md) for GPU advice.

---

## Design principles

- **Build the hard part.** Every project answers "what did you actually write?" with an algorithm, not an import.
- **Prove it.** Each core ships with tests (gradient checks, recall-vs-brute-force, round-trip, known-answer cases).
- **Be honest.** EXPLAINERs state limitations plainly — including where a from-scratch version is slower or weaker than a library (and why it exists anyway: to demonstrate understanding).
- **Runs anywhere.** GPU-aware, CPU-friendly where feasible; no mandatory API keys.

## License

MIT — see [LICENSE](LICENSE).
