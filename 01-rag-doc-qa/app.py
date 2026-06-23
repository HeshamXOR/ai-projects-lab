"""RAG Document Q&A — Gradio preview.

Upload a PDF, ask questions, get answers grounded in the document with page
citations. Runs on CPU; no API key required (extractive answers by default).
"""

from __future__ import annotations

import gradio as gr

from rag import RagEngine

# Load the embedding model once at startup.
engine = RagEngine()
STATE = {"indexed": False, "name": ""}


def on_upload(file):
    if file is None:
        return "Upload a PDF to begin.", gr.update(interactive=False)
    try:
        n = engine.index_pdf(file.name)
    except Exception as e:
        return f"Could not read that PDF: {e}", gr.update(interactive=False)
    if n == 0:
        return "No extractable text found (is it a scanned image PDF?).", gr.update(interactive=False)
    STATE["indexed"] = True
    STATE["name"] = file.name.split("/")[-1].split("\\")[-1]
    return f"Indexed **{STATE['name']}** — {n} chunks. Ask away!", gr.update(interactive=True)


def on_ask(question, history):
    history = history or []
    if not STATE["indexed"]:
        history.append((question, "Please upload a PDF first."))
        return history, ""
    if not question.strip():
        return history, ""
    ans = engine.answer(question)
    src = ""
    if ans.sources:
        pages = sorted({c.page for c in ans.sources})
        src = "\n\n**Sources:** " + ", ".join(f"page {p}" for p in pages)
    history.append((question, ans.text + src))
    return history, ""


with gr.Blocks(title="RAG Document Q&A", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 📄 RAG Document Q&A\n"
        "Upload a PDF and ask questions about it. Answers are grounded in the "
        "document and cite the pages they came from.\n\n"
        "*Default mode is extractive (no API key needed). Set `OPENAI_API_KEY` "
        "for fully generated answers.*"
    )
    with gr.Row():
        with gr.Column(scale=1):
            pdf = gr.File(label="PDF", file_types=[".pdf"])
            status = gr.Markdown("Upload a PDF to begin.")
        with gr.Column(scale=2):
            chat = gr.Chatbot(label="Q&A", height=420)
            q = gr.Textbox(placeholder="Ask a question about the document…", label="Question", interactive=False)
            ask_btn = gr.Button("Ask", variant="primary")

    pdf.change(on_upload, inputs=pdf, outputs=[status, q])
    ask_btn.click(on_ask, inputs=[q, chat], outputs=[chat, q])
    q.submit(on_ask, inputs=[q, chat], outputs=[chat, q])

    gr.Examples(
        examples=["What is the main conclusion?", "Summarize the key points.", "What methods were used?"],
        inputs=q,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
