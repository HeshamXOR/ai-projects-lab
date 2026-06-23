# Running & previewing on Lightning AI

Every project here is a Gradio app, which makes previews trivial on Lightning AI.

## One-time setup

1. Create a **Studio** at [lightning.ai](https://lightning.ai).
2. Open the terminal in the Studio and clone this repo:
   ```bash
   git clone <your-repo-url> ai-projects-lab
   cd ai-projects-lab
   ```

## Run any project

```bash
cd 01-rag-doc-qa
pip install -r requirements.txt
python app.py
```

Gradio launches with `share=True`, so it prints two URLs:
- a local one (`http://localhost:7860`)
- a **public** one (`https://xxxx.gradio.live`) — open it in your browser, use it for screenshots, or share it as the live preview.

On Lightning you can also use the Studio's built-in port forwarding (port `7860`) instead of the share link.

## Which GPU to pick

| Project | Minimum | Recommended | Notes |
|---------|---------|-------------|-------|
| 1 · RAG Doc Q&A | CPU | **L4** | Embeddings run fine on CPU; a GPU speeds up the optional local LLM. |
| 2 · Resume Matcher | CPU | CPU | Embedding-only; no GPU needed. |
| 3 · Image Q&A | CPU (slow) | **L4** | BLIP is much snappier on a GPU; CPU works for a demo. |
| 4 · Smart Summarizer | CPU | **L4** | Summarization model benefits from GPU on long inputs. |
| 5 · Sentiment & Emotion | CPU | CPU | Small classifiers; runs fine on CPU. |
| 6 · Code Search | CPU | CPU | Embedding-only; no GPU needed. |
| 7 · Speech-to-Text | CPU (slow) | **L4** | Whisper is much faster on GPU; CPU works for short clips. Needs `ffmpeg`. |
| 8 · Image Generator | CPU (very slow) | **L4** | Stable Diffusion really wants a GPU — record this preview on L4. |

**Recommendation:** start every project on a **free CPU Studio** to verify it runs and to grab a screenshot. Switch to an **L4** GPU only for projects 1/3/4 when you want fast, smooth previews. An L4 is plenty — none of these need an A100.

> Tip: Lightning bills GPU by the second. Develop/screenshot on CPU, switch to L4 for the polished demo recording, then stop the GPU.

## Recording a preview GIF

1. Run the app, open the share link.
2. Use any screen recorder (or Lightning's built-in tools) to capture a short interaction.
3. Save it into the project folder as `preview.gif` and reference it in that project's README (each README has a slot for it).
